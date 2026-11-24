"""task1/runner.py — Complete serverless orchestrator for Task 1 with auto-resume."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task1.allocation import solve_p3_sparsity_allocation, solve_q3_bit_allocation
from task1.calibration import collect_calibration_statistics, load_c4_calibration_tokens
from task1.checkpoint import Task1StateManager
from task1.config import Task1Config
from task1.evaluate import (
    benchmark_model_latency_and_vram,
    compute_paired_bootstrap_confidence_interval,
    compute_paired_pooled_ppl_bootstrap,
    evaluate_hellaswag,
    evaluate_token_nll,
    evaluate_perplexity,
    pretokenize_hellaswag,
)
from task1.packed_storage import compute_model_packed_storage
from task1.pruners import (
    get_eligible_linear_modules,
    prune_magnitude_p1,
    prune_nonuniform_p3,
    prune_structured_2_4,
    prune_wanda_p2,
)
from task1.quantizers import (
    quantize_awq_q2,
    quantize_mixed_precision_q3,
    quantize_uniform_q1,
    verify_awq_rescaling_preserves_outputs,
)
from task1.sensitivity import (
    compute_activation_outlier_metrics,
    profile_layer_pruning_sensitivity,
    profile_layer_awq_quantization_sensitivity,
    profile_layer_quantization_sensitivity,
)


def run_task1_pipeline(
    config: Optional[Task1Config] = None,
    modal_volume: Optional[Any] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute the complete Task 1 suite with zero-loss auto-resume."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    if config is None:
        config = Task1Config()

    state_mgr = Task1StateManager(config.checkpoint_dir, modal_volume=modal_volume)
    data_dir = config.base_data_dir

    # Resolve locked data files
    calib_path = data_dir / "public_locked/c4_calibration_locked.jsonl.gz"
    dev_path = data_dir / "public_locked/c4_development_locked.jsonl.gz"
    wiki_path = data_dir / "public_locked/wikitext2_test_windows_locked.jsonl.gz"
    c4_val_path = data_dir / "public_locked/c4_validation_windows_locked.jsonl.gz"
    hellaswag_path = data_dir / "public_locked/hellaswag_val_locked.jsonl.gz"

    for p in [calib_path, wiki_path, c4_val_path, hellaswag_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {p}. Ensure 'setup_data.py --public' has been executed."
            )

    print("=" * 80)
    print("ATML PA4 — Task 1 Execution Pipeline")
    print(f"Model:          {config.model_name_or_path}")
    print(f"Device:         {config.device}")
    print(f"Checkpoint Dir: {config.checkpoint_dir}")
    print(f"Smoke Test:     {smoke_test}")
    print("=" * 80)

    # 1. Load Model & Tokenizer onto GPU
    print(f"\n[Model Setup] Loading {config.model_name_or_path} into VRAM in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.float16,
        device_map=config.device,
    )
    model.eval()

    eligible_modules = get_eligible_linear_modules(model, config.eligible_projections)
    print(f"[Model Setup] Loaded {len(eligible_modules)} eligible linear projections across all blocks.")

    # Cache pristine FP16 weights in pinned HOST memory. Keeping a second GPU copy
    # (~3 GB) inflated every peak-allocated-memory measurement after the baseline.
    pristine_weights = {
        name: module.weight.data.detach().to("cpu").pin_memory() for name, module in eligible_modules.items()
    }

    def restore_pristine_weights():
        with torch.no_grad():
            for name, module in eligible_modules.items():
                module.weight.data.copy_(pristine_weights[name], non_blocking=True)
        torch.cuda.synchronize()

    def layer_output_errors(activation_stats) -> Dict[str, float]:
        """Relative layer-output error ||XW^T - XW_c^T||^2 / ||XW^T||^2 on calibration inputs."""
        errs: Dict[str, float] = {}
        with torch.no_grad():
            for name, module in eligible_modules.items():
                x = activation_stats.sample_inputs.get(name)
                if x is None:
                    continue
                x = x.to(module.weight.device).float()
                w0 = pristine_weights[name].to(module.weight.device, non_blocking=True).float()
                y0 = x @ w0.t()
                y1 = x @ module.weight.data.float().t()
                errs[name] = float((((y0 - y1) ** 2).sum() / (y0 ** 2).sum().clamp_min(1e-12)).item())
        return errs

    def summarize_errors(errs: Dict[str, float]) -> Dict[str, Any]:
        vals = list(errs.values())
        by_proj: Dict[str, List[float]] = {}
        for n, v in errs.items():
            by_proj.setdefault(n.split(".")[-1], []).append(v)
        return {
            "mean_relative_layer_output_error": sum(vals) / max(1, len(vals)),
            "by_projection_mean": {k: sum(v) / len(v) for k, v in by_proj.items()},
            "per_layer": errs,
        }

    # Pre-tokenize HellaSwag evaluation dataset once to eliminate CPU tokenizer bottleneck
    print(f"\n[HellaSwag Setup] Pre-tokenizing HellaSwag evaluation dataset once...")
    hs_pretokenized = pretokenize_hellaswag(
        tokenizer,
        hellaswag_path,
        limit_samples=10 if smoke_test else config.hellaswag_total_samples,
    )

    # Helper function to evaluate a compressed condition
    def evaluate_condition(
        cond_name: str,
        is_full_hellaswag: bool = False,
        sparsity_allocations: Optional[Dict[str, float]] = None,
        uniform_sparsity: Optional[float] = None,
        bit_allocations: Optional[Dict[str, int]] = None,
        uniform_bit_width: Optional[int] = None,
        awq_channel_scales: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        h_samples = (
            config.hellaswag_total_samples
            if (is_full_hellaswag and not smoke_test)
            else (10 if smoke_test else config.hellaswag_diagnostic_samples)
        )
        wiki_target = 1000 if smoke_test else config.wikitext_target_tokens
        c4_target = 1000 if smoke_test else config.c4_target_tokens

        # Perplexity evaluations
        wiki_res = evaluate_perplexity(model, wiki_path, device=config.device, target_scored_tokens=wiki_target)
        c4_res = evaluate_perplexity(model, c4_val_path, device=config.device, target_scored_tokens=c4_target)

        # HellaSwag evaluation (fast batched GPU evaluation using pre-tokenized cache)
        hs_res = evaluate_hellaswag(
            model,
            pretokenized_examples=hs_pretokenized[:h_samples],
            device=config.device,
        )

        # Latency & VRAM benchmark
        perf = benchmark_model_latency_and_vram(model, device=config.device)

        # Packed storage accounting
        storage = compute_model_packed_storage(
            model,
            eligible_projections=config.eligible_projections,
            sparsity_allocations=sparsity_allocations,
            uniform_sparsity=uniform_sparsity,
            bit_allocations=bit_allocations,
            uniform_bit_width=uniform_bit_width,
            group_size=config.quant_group_size,
            awq_channel_scales=awq_channel_scales,
        )

        metrics = {
            "condition": cond_name,
            "wikitext_ppl": wiki_res["perplexity"],
            "wikitext_nll": wiki_res["mean_nll"],
            "c4_ppl": c4_res["perplexity"],
            "c4_nll": c4_res["mean_nll"],
            "hellaswag_acc": hs_res["acc"],
            "hellaswag_acc_norm": hs_res["acc_norm"],
            "hellaswag_samples": hs_res["total_samples"],
            "packed_bytes": storage["total_bytes"],
            "packed_mb": storage["total_megabytes"],
            "avg_bits_per_param": storage["avg_bits_per_param"],
            "effective_sparsity": storage["effective_sparsity"],
            "prefill_512_s": perf["prefill_512"],
            "prefill_2048_s": perf["prefill_2048"],
            "decode_64_s": perf["decode_64_tokens"],
            "peak_vram_mb": perf["peak_vram_mb"],
            "timing_repetitions": perf["timing_repetitions"],
            "total_nonzeros": storage["total_nonzeros"],
            "eligible_metadata_bytes": storage["eligible_metadata_bytes"],
            "memory_note": "Pruned/fake-quantized weights execute as dense FP16; packed_bytes is a "
                           "serialized-storage estimate, peak_vram_mb is measured dense execution.",
            "_per_window_wiki": wiki_res["per_window_records"],
            "_per_window_c4": c4_res["per_window_records"],
            "_per_example_hs": hs_res["per_example_correct"],
        }
        if extra:
            metrics.update(extra)
        return metrics

    # ── STEP 1: Uncompressed Baseline ─────────────────────────────────────────
    cond = "baseline_fp16"
    min_baseline_samples = 10 if smoke_test else config.hellaswag_total_samples
    if not state_mgr.is_completed(cond, min_samples=min_baseline_samples):
        print("\n" + "=" * 80)
        print("STEP 1: Evaluating Uncompressed Baseline (FP16)...")
        print("=" * 80)
        state_mgr.mark_in_progress(cond)
        metrics = evaluate_condition(cond, is_full_hellaswag=True, uniform_bit_width=16)
        state_mgr.record_completed(cond, metrics)
    else:
        print(f"\n[Resume] Step 1 ({cond}) already completed. Skipping.")

    # ── STEP 2: Collect Calibration Statistics ─────────────────────────────────
    calib_stats_filename = "calibration_stats_smoke.pt" if smoke_test else "calibration_stats_full.pt"
    calib_stats_path = config.checkpoint_dir / calib_stats_filename
    if calib_stats_path.exists():
        print(f"\n[Resume] Loading cached calibration stats from {calib_stats_path}...")
        activation_stats = torch.load(calib_stats_path, map_location=config.device, weights_only=False)
    else:
        print("\n" + "=" * 80)
        print("STEP 2: Collecting GPU-Resident Calibration Statistics...")
        print("=" * 80)
        calib_tokens = load_c4_calibration_tokens(
            calib_path,
            max_sequences=10 if smoke_test else config.calibration_seq_count,
            max_len=config.calibration_seq_len,
        )
        t_calib = time.perf_counter()
        activation_stats = collect_calibration_statistics(
            model=model,
            calib_tokens=calib_tokens,
            eligible_projections=config.eligible_projections,
            batch_size=8,
            device=config.device,
        )
        calib_seconds = time.perf_counter() - t_calib
        (config.checkpoint_dir / "calibration_time.json").write_text(json.dumps({"calibration_seconds": calib_seconds}))
        torch.save(activation_stats, calib_stats_path)
        state_mgr.commit_volume()
        print(f"[Calibration] Saved stats to {calib_stats_path}")

    # ── STEP 3: P1 Magnitude Pruning ──────────────────────────────────────────
    sparsities = (0.25,) if smoke_test else config.p1_sparsities
    for s in sparsities:
        cond = f"P1_magnitude_s{int(s*100)}"
        is_full_hs = (s == 0.50)
        min_hs = 10 if smoke_test else (config.hellaswag_total_samples if is_full_hs else config.hellaswag_diagnostic_samples)
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 3: Running P1 Magnitude Pruning at {s:.0%} sparsity...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            restore_pristine_weights()
            t_c = time.perf_counter()
            prune_magnitude_p1(model, sparsity=s, eligible_projections=config.eligible_projections)
            comp_s = time.perf_counter() - t_c
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_sparsity=s, extra={
                "compression_seconds": comp_s,
                "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
            })
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 4: P2 Wanda Pruning ──────────────────────────────────────────────
    sparsities = (0.25,) if smoke_test else config.p2_sparsities
    for s in sparsities:
        cond = f"P2_wanda_s{int(s*100)}"
        is_full_hs = (s == 0.50)
        min_hs = 10 if smoke_test else (config.hellaswag_total_samples if is_full_hs else config.hellaswag_diagnostic_samples)
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 4: Running P2 Wanda Pruning at {s:.0%} sparsity...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            restore_pristine_weights()
            t_c = time.perf_counter()
            prune_wanda_p2(
                model, sparsity=s, activation_stats=activation_stats, eligible_projections=config.eligible_projections
            )
            comp_s = time.perf_counter() - t_c
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_sparsity=s, extra={
                "compression_seconds": comp_s,
                "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
            })
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 5: Structured 2:4 Sparsity Diagnostic ─────────────────────────────
    cond = "P_structured_2_4"
    min_hs = 10 if smoke_test else config.hellaswag_total_samples
    if not state_mgr.is_completed(cond, min_samples=min_hs):
        print("\n" + "=" * 80)
        print("STEP 5: Running Structured 2:4 Sparsity Diagnostic (50% sparsity)...")
        print("=" * 80)
        state_mgr.mark_in_progress(cond)
        restore_pristine_weights()
        prune_structured_2_4(
            model, activation_stats=activation_stats, eligible_projections=config.eligible_projections
        )
        metrics = evaluate_condition(cond, is_full_hellaswag=True, uniform_sparsity=0.50, extra={
            "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
        })
        state_mgr.record_completed(cond, metrics)
    else:
        print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 6: P3 Non-Uniform Pruning ────────────────────────────────────────
    # Sensitivity: isolated per-projection Wanda pruning at 50% on calibration inputs,
    # measured two ways (relative and absolute output error). The design (measure, alpha)
    # is selected ONLY on the separate 32-sequence C4 development split; alpha=0 is the
    # uniform-Wanda fallback. All candidates and their dev NLLs are saved.
    pruning_sens_filename = "pruning_sensitivity_smoke.json" if smoke_test else "pruning_sensitivity_full.json"
    pruning_sens_path = config.checkpoint_dir / pruning_sens_filename
    if pruning_sens_path.exists():
        with open(pruning_sens_path, "r", encoding="utf-8") as f:
            sens_blob = json.load(f)
    else:
        sens_blob = {}
    if "relative" not in sens_blob or "absolute" not in sens_blob:
        restore_pristine_weights()
        t_sens = time.perf_counter()
        rel_curves, abs_curves = profile_layer_pruning_sensitivity(
            model, activation_stats, eligible_projections=config.eligible_projections, return_absolute=True
        )
        sens_blob = {
            "relative": {n: v[0.50] for n, v in rel_curves.items()},
            "absolute": {n: v[0.50] for n, v in abs_curves.items()},
            "relative_curves": {n: {str(k): x for k, x in v.items()} for n, v in rel_curves.items()},
            "absolute_curves": {n: {str(k): x for k, x in v.items()} for n, v in abs_curves.items()},
            "profiling_seconds": time.perf_counter() - t_sens,
        }
        with open(pruning_sens_path, "w", encoding="utf-8") as f:
            json.dump(sens_blob, f, indent=2)

    dev_tokens = load_c4_calibration_tokens(
        dev_path,
        max_sequences=4 if smoke_test else config.development_seq_count,
        max_len=config.development_seq_len,
    )
    p3_alpha_grid = (0.0, 0.05, 0.10, 0.15, 0.25)
    p3_selection_path = config.checkpoint_dir / ("p3_dev_selection_smoke.json" if smoke_test else "p3_dev_selection.json")
    p3_selection = json.loads(p3_selection_path.read_text()) if p3_selection_path.exists() else {}

    p3_sparsities = (0.50,) if smoke_test else config.p3_global_sparsities
    for s_target in p3_sparsities:
        key = f"s{int(s_target*100)}"
        if key not in p3_selection:
            t_sel = time.perf_counter()
            candidates = []
            for measure in ("relative", "absolute"):
                for alpha in p3_alpha_grid:
                    if alpha == 0.0 and measure == "absolute":
                        continue  # identical to uniform; evaluated once
                    alloc = solve_p3_sparsity_allocation(
                        model=model,
                        eligible_projections=config.eligible_projections,
                        sensitivity_scores=sens_blob[measure],
                        target_global_sparsity=s_target,
                        min_sparsity=config.p3_min_sparsity,
                        max_sparsity=config.p3_max_sparsity,
                        alpha=alpha,
                    )
                    restore_pristine_weights()
                    prune_nonuniform_p3(model, alloc, activation_stats, config.eligible_projections)
                    dev_nll = evaluate_token_nll(model, dev_tokens, device=config.device)
                    candidates.append({"measure": measure, "alpha": alpha, "dev_nll": dev_nll})
                    print(f"[P3 dev] {key} measure={measure} alpha={alpha:.2f} dev NLL={dev_nll:.4f}")
            best = min(candidates, key=lambda c: c["dev_nll"])
            p3_selection[key] = {
                "selected": best,
                "candidates": candidates,
                "selection_rule": "argmin mean token NLL on the 32x512 C4 development split",
                "selection_seconds": time.perf_counter() - t_sel,
            }
            p3_selection_path.write_text(json.dumps(p3_selection, indent=2))
            state_mgr.commit_volume()

        sel = p3_selection[key]["selected"]
        cond = f"P3_nonuniform_s{int(s_target*100)}"
        min_hs = 10 if smoke_test else config.hellaswag_total_samples
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 6: Running P3 Non-Uniform Pruning at target {s_target:.0%} "
                  f"(dev-selected measure={sel['measure']}, alpha={sel['alpha']})...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            alloc_map = solve_p3_sparsity_allocation(
                model=model,
                eligible_projections=config.eligible_projections,
                sensitivity_scores=sens_blob[sel["measure"]],
                target_global_sparsity=s_target,
                min_sparsity=config.p3_min_sparsity,
                max_sparsity=config.p3_max_sparsity,
                alpha=sel["alpha"],
            )
            restore_pristine_weights()
            t_c = time.perf_counter()
            achieved = prune_nonuniform_p3(
                model,
                allocation_map=alloc_map,
                activation_stats=activation_stats,
                eligible_projections=config.eligible_projections,
            )
            comp_s = time.perf_counter() - t_c
            metrics = evaluate_condition(
                cond, is_full_hellaswag=True, sparsity_allocations=alloc_map, extra={
                    "compression_seconds": comp_s,
                    "allocation_map": alloc_map,
                    "achieved_layer_sparsity": achieved,
                    "selected_design": sel,
                    "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
                }
            )
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 7: Q1 Uniform PTQ ────────────────────────────────────────────────
    q1_bits = (8,) if smoke_test else config.q1_bit_widths
    for b in q1_bits:
        cond = f"Q1_uniform_int{b}"
        is_full_hs = (b in (4, 3))
        min_hs = 10 if smoke_test else (config.hellaswag_total_samples if is_full_hs else config.hellaswag_diagnostic_samples)
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 7: Running Q1 Uniform PTQ at {b}-bit...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            restore_pristine_weights()
            t_c = time.perf_counter()
            quantize_uniform_q1(
                model, bit_width=b, group_size=config.quant_group_size, eligible_projections=config.eligible_projections
            )
            comp_s = time.perf_counter() - t_c
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_bit_width=b, extra={
                "compression_seconds": comp_s,
                "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
                "quantizer_convention": "asymmetric uniform, group 128 along input columns, round-to-nearest, "
                                        "FP16 scale + FP16 unclamped integer zero-point per group, "
                                        "b-bit codes bit-packed densely (no padding: in_features % 128 == 0)",
            })
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 8: Q2 AWQ ────────────────────────────────────────────────────────
    q2_bits = (8,) if smoke_test else config.q2_bit_widths
    for b in q2_bits:
        cond = f"Q2_awq_int{b}"
        is_full_hs = (b in (4, 3))
        min_hs = 10 if smoke_test else (config.hellaswag_total_samples if is_full_hs else config.hellaswag_diagnostic_samples)
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 8: Running Q2 AWQ at {b}-bit...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            restore_pristine_weights()
            awq_preserve = verify_awq_rescaling_preserves_outputs(
                model, activation_stats, alpha=0.5, eligible_projections=config.eligible_projections
            )
            t_c = time.perf_counter()
            quantize_awq_q2(
                model,
                bit_width=b,
                activation_stats=activation_stats,
                group_size=config.quant_group_size,
                candidate_alphas=config.awq_grid_alphas,
                eligible_projections=config.eligible_projections,
            )
            comp_s = time.perf_counter() - t_c
            metrics = evaluate_condition(
                cond, is_full_hellaswag=is_full_hs, uniform_bit_width=b, awq_channel_scales=True, extra={
                    "compression_seconds": comp_s,
                    "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
                    "awq_fp_rescaling_max_rel_output_error": max(awq_preserve.values()) if awq_preserve else None,
                    "awq_scale_storage": "inverse scales NOT folded (q/k/v and gate/up have per-projection "
                                         "scales); one FP16 scale per input channel serialized and counted",
                })
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 9: Q3 Mixed-Precision AWQ ────────────────────────────────────────
    # Sensitivity: isolated per-projection calibration output error of the SAME AWQ quantizer
    # used by Q2/Q3, at every allowed bit width, in absolute and relative form. Allocation:
    # greedy marginal error reduction per added packed byte under the Q2 byte budget. The
    # error measure is chosen per budget on the C4 development split (all candidates saved).
    q_sens_filename = "quant_sensitivity_smoke.json" if smoke_test else "quant_sensitivity_full.json"
    q_sens_path = config.checkpoint_dir / q_sens_filename
    q_sens_blob = json.loads(q_sens_path.read_text()) if q_sens_path.exists() else {}
    if "absolute" not in q_sens_blob:
        restore_pristine_weights()
        t_sens = time.perf_counter()
        q_sens_blob = profile_layer_awq_quantization_sensitivity(
            model,
            activation_stats,
            eligible_projections=config.eligible_projections,
            candidate_bits=config.q3_allowed_bits,
            group_size=config.quant_group_size,
            candidate_alphas=config.awq_grid_alphas,
        )
        q_sens_blob["profiling_seconds"] = time.perf_counter() - t_sens
        q_sens_path.write_text(json.dumps(q_sens_blob, indent=2))
    # JSON stores bit widths as strings; the allocator needs integer keys.
    quant_sens_by_measure = {
        m: {n: {int(b_): v for b_, v in d.items()} for n, d in q_sens_blob[m].items()}
        for m in ("absolute", "relative")
    }

    # Reference budgets from Q2 INT4 and INT3
    ref_storage_int4 = compute_model_packed_storage(
        model,
        eligible_projections=config.eligible_projections,
        uniform_bit_width=4,
        group_size=config.quant_group_size,
        awq_channel_scales=True,
    )
    ref_storage_int3 = compute_model_packed_storage(
        model,
        eligible_projections=config.eligible_projections,
        uniform_bit_width=3,
        group_size=config.quant_group_size,
        awq_channel_scales=True,
    )

    targets = (
        [("Q3_mixed_match_int4", ref_storage_int4["total_bytes"])]
        if smoke_test
        else [
            ("Q3_mixed_match_int4", ref_storage_int4["total_bytes"]),
            ("Q3_mixed_match_int3", ref_storage_int3["total_bytes"]),
        ]
    )

    q3_selection_path = config.checkpoint_dir / ("q3_dev_selection_smoke.json" if smoke_test else "q3_dev_selection.json")
    q3_selection = json.loads(q3_selection_path.read_text()) if q3_selection_path.exists() else {}

    def q3_alloc(measure: str, target_bytes: int) -> Dict[str, int]:
        return solve_q3_bit_allocation(
            model=model,
            eligible_projections=config.eligible_projections,
            quant_sensitivity=quant_sens_by_measure[measure],
            target_packed_bytes=target_bytes,
            allowed_bits=config.q3_allowed_bits,
            group_size=config.quant_group_size,
            awq_channel_scales=True,
        )

    for cond, target_bytes in targets:
        if cond not in q3_selection:
            t_sel = time.perf_counter()
            candidates = []
            for measure in ("absolute", "relative"):
                alloc = q3_alloc(measure, target_bytes)
                restore_pristine_weights()
                quantize_mixed_precision_q3(
                    model, allocation_map=alloc, activation_stats=activation_stats,
                    group_size=config.quant_group_size, candidate_alphas=config.awq_grid_alphas,
                    eligible_projections=config.eligible_projections,
                )
                dev_nll = evaluate_token_nll(model, dev_tokens, device=config.device)
                candidates.append({"measure": measure, "dev_nll": dev_nll,
                                   "bit_histogram": {str(b_): sum(1 for v in alloc.values() if v == b_)
                                                     for b_ in config.q3_allowed_bits}})
                print(f"[Q3 dev] {cond} measure={measure} dev NLL={dev_nll:.4f}")
            q3_selection[cond] = {
                "selected": min(candidates, key=lambda c: c["dev_nll"]),
                "candidates": candidates,
                "selection_rule": "argmin mean token NLL on the 32x512 C4 development split",
                "selection_seconds": time.perf_counter() - t_sel,
            }
            q3_selection_path.write_text(json.dumps(q3_selection, indent=2))
            state_mgr.commit_volume()

        sel = q3_selection[cond]["selected"]
        min_hs = 10 if smoke_test else config.hellaswag_total_samples
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 9: Running Q3 Mixed Precision matching budget {target_bytes:,} bytes "
                  f"(dev-selected measure={sel['measure']})...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            alloc_map = q3_alloc(sel["measure"], target_bytes)
            restore_pristine_weights()
            quantize_mixed_precision_q3(
                model,
                allocation_map=alloc_map,
                activation_stats=activation_stats,
                group_size=config.quant_group_size,
                candidate_alphas=config.awq_grid_alphas,
                eligible_projections=config.eligible_projections,
            )
            metrics = evaluate_condition(
                cond, is_full_hellaswag=True, bit_allocations=alloc_map, awq_channel_scales=True, extra={
                    "bit_allocation_map": alloc_map,
                    "selected_design": sel,
                    "target_budget_bytes": target_bytes,
                    "bit_histogram": {str(b_): sum(1 for v in alloc_map.values() if v == b_) for b_ in config.q3_allowed_bits},
                    "calibration_layer_output_error": summarize_errors(layer_output_errors(activation_stats)),
                }
            )
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 10: Statistical Paired Bootstrap Confidence Intervals ─────────────
    print("\n" + "=" * 80)
    print("STEP 10: Computing Paired 95% Bootstrap Confidence Intervals...")
    print("=" * 80)
    bootstrap_results = {}
    all_res = state_mgr.get_all_results()

    comparisons = [
        ("P3_vs_P2_50", "P2_wanda_s50", "P3_nonuniform_s50"),
        ("P3_vs_P2_75", "P2_wanda_s75", "P3_nonuniform_s75"),
        ("Q3_vs_Q2_int4", "Q2_awq_int4", "Q3_mixed_match_int4"),
        ("Q3_vs_Q2_int3", "Q2_awq_int3", "Q3_mixed_match_int3"),
    ]

    for comp_name, base_cond, adapt_cond in comparisons:
        if base_cond in all_res and adapt_cond in all_res:
            res_a = all_res[base_cond]
            res_b = all_res[adapt_cond]

            for hs_key in ("norm", "raw"):
                hs_a = [x[hs_key] for x in res_a["_per_example_hs"]]
                hs_b = [x[hs_key] for x in res_b["_per_example_hs"]]
                min_len = min(len(hs_a), len(hs_b))
                if min_len > 0:
                    bootstrap_results[f"{comp_name}_hellaswag_acc_{hs_key}"] = compute_paired_bootstrap_confidence_interval(
                        hs_a[:min_len], hs_b[:min_len], num_resamples=config.bootstrap_resamples
                    )

            # Perplexity: resample whole windows, recompute pooled token NLL, exponentiate.
            for suite, key in (("wikitext", "_per_window_wiki"), ("c4", "_per_window_c4")):
                if res_a.get(key) and res_b.get(key):
                    bootstrap_results[f"{comp_name}_{suite}_ppl"] = compute_paired_pooled_ppl_bootstrap(
                        res_a[key], res_b[key], num_resamples=config.bootstrap_resamples
                    )

    # Paired changes from the unchanged model on the SAME HellaSwag IDs each condition used.
    base = all_res.get("baseline_fp16")
    paired_vs_baseline = {}
    if base:
        for c_name, m in all_res.items():
            n = len(m.get("_per_example_hs", []))
            if n == 0 or c_name == "baseline_fp16":
                continue
            b_hs = base["_per_example_hs"][:n]
            paired_vs_baseline[c_name] = {
                "hellaswag_ids": n,
                "baseline_acc_norm_same_ids": sum(x["norm"] for x in b_hs) / n,
                "delta_acc_norm_pp": 100 * (m["hellaswag_acc_norm"] - sum(x["norm"] for x in b_hs) / n),
                "delta_acc_pp": 100 * (m["hellaswag_acc"] - sum(x["raw"] for x in b_hs) / n),
                "delta_wikitext_ppl": m["wikitext_ppl"] - base["wikitext_ppl"],
                "delta_c4_ppl": m["c4_ppl"] - base["c4_ppl"],
            }
    bootstrap_results["paired_changes_vs_baseline"] = paired_vs_baseline

    ci_file = config.checkpoint_dir / "bootstrap_ci.json"
    with open(ci_file, "w", encoding="utf-8") as f:
        json.dump(bootstrap_results, f, indent=2)
    state_mgr.commit_volume()
    print(f"[Bootstrap] Computed {len(bootstrap_results)} confidence intervals. Saved to {ci_file}")

    # Generate Markdown Summary Table
    print("\n" + "=" * 105)
    print(f"{'Condition':<26} | {'Wiki PPL':>8} | {'C4 PPL':>8} | {'HS Raw':>8} | {'HS Norm':>8} | {'MB':>7} | {'Avg Bits':>8}")
    print("=" * 105)
    for c_name, m in all_res.items():
        print(
            f"{c_name:<26} | {m['wikitext_ppl']:>8.2f} | {m['c4_ppl']:>8.2f} | "
            f"{m['hellaswag_acc']:>7.2%} | {m['hellaswag_acc_norm']:>7.2%} | "
            f"{m['packed_mb']:>7.1f} | {m['avg_bits_per_param']:>8.2f}"
        )
    print("=" * 105)

    print("\n[Pipeline Complete] All Task 1 experiments finished and checkpointed to volume.")
    return all_res
