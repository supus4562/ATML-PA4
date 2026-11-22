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
    evaluate_hellaswag,
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
)
from task1.sensitivity import (
    compute_activation_outlier_metrics,
    profile_layer_pruning_sensitivity,
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

    # Cache pristine FP16 weights directly in GPU VRAM for instant restoration (<0.1s)
    pristine_weights = {name: module.weight.data.clone() for name, module in eligible_modules.items()}

    def restore_pristine_weights():
        with torch.no_grad():
            for name, module in eligible_modules.items():
                module.weight.data.copy_(pristine_weights[name])

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
            "_per_window_wiki": wiki_res["per_window_records"],
            "_per_window_c4": c4_res["per_window_records"],
            "_per_example_hs": hs_res["per_example_correct"],
        }
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
        activation_stats = collect_calibration_statistics(
            model=model,
            calib_tokens=calib_tokens,
            eligible_projections=config.eligible_projections,
            batch_size=8,
            device=config.device,
        )
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
            prune_magnitude_p1(model, sparsity=s, eligible_projections=config.eligible_projections)
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_sparsity=s)
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
            prune_wanda_p2(
                model, sparsity=s, activation_stats=activation_stats, eligible_projections=config.eligible_projections
            )
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_sparsity=s)
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
        metrics = evaluate_condition(cond, is_full_hellaswag=True, uniform_sparsity=0.50)
        state_mgr.record_completed(cond, metrics)
    else:
        print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 6: P3 Non-Uniform Pruning ────────────────────────────────────────
    pruning_sens_filename = "pruning_sensitivity_smoke.json" if smoke_test else "pruning_sensitivity_full.json"
    pruning_sens_path = config.checkpoint_dir / pruning_sens_filename
    if pruning_sens_path.exists():
        with open(pruning_sens_path, "r", encoding="utf-8") as f:
            p_sens_scores = json.load(f)
    else:
        restore_pristine_weights()
        p_curves = profile_layer_pruning_sensitivity(
            model, activation_stats, eligible_projections=config.eligible_projections
        )
        p_sens_scores = {name: vals.get(0.50, 1.0) for name, vals in p_curves.items()}
        with open(pruning_sens_path, "w", encoding="utf-8") as f:
            json.dump(p_sens_scores, f, indent=2)

    p3_sparsities = (0.50,) if smoke_test else config.p3_global_sparsities
    for s_target in p3_sparsities:
        cond = f"P3_nonuniform_s{int(s_target*100)}"
        min_hs = 10 if smoke_test else config.hellaswag_total_samples
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 6: Running P3 Non-Uniform Pruning at target {s_target:.0%}...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            alloc_map = solve_p3_sparsity_allocation(
                model=model,
                eligible_projections=config.eligible_projections,
                sensitivity_scores=p_sens_scores,
                target_global_sparsity=s_target,
                min_sparsity=config.p3_min_sparsity,
                max_sparsity=config.p3_max_sparsity,
            )
            restore_pristine_weights()
            prune_nonuniform_p3(
                model,
                allocation_map=alloc_map,
                activation_stats=activation_stats,
                eligible_projections=config.eligible_projections,
            )
            metrics = evaluate_condition(
                cond, is_full_hellaswag=True, sparsity_allocations=alloc_map
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
            quantize_uniform_q1(
                model, bit_width=b, group_size=config.quant_group_size, eligible_projections=config.eligible_projections
            )
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_bit_width=b)
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
            quantize_awq_q2(
                model,
                bit_width=b,
                activation_stats=activation_stats,
                group_size=config.quant_group_size,
                candidate_alphas=config.awq_grid_alphas,
                eligible_projections=config.eligible_projections,
            )
            metrics = evaluate_condition(cond, is_full_hellaswag=is_full_hs, uniform_bit_width=b)
            state_mgr.record_completed(cond, metrics)
        else:
            print(f"[Resume] {cond} already completed. Skipping.")

    # ── STEP 9: Q3 Mixed-Precision AWQ ────────────────────────────────────────
    q_sens_filename = "quant_sensitivity_smoke.json" if smoke_test else "quant_sensitivity_full.json"
    q_sens_path = config.checkpoint_dir / q_sens_filename
    if q_sens_path.exists():
        with open(q_sens_path, "r", encoding="utf-8") as f:
            quant_sensitivity = json.load(f)
    else:
        restore_pristine_weights()
        quant_sensitivity = profile_layer_quantization_sensitivity(
            model,
            activation_stats,
            eligible_projections=config.eligible_projections,
            candidate_bits=config.q3_allowed_bits,
            group_size=config.quant_group_size,
        )
        with open(q_sens_path, "w", encoding="utf-8") as f:
            json.dump(quant_sensitivity, f, indent=2)

    # Reference budgets from Q2 INT4 and INT3
    ref_storage_int4 = compute_model_packed_storage(
        model,
        eligible_projections=config.eligible_projections,
        uniform_bit_width=4,
        group_size=config.quant_group_size,
    )
    ref_storage_int3 = compute_model_packed_storage(
        model,
        eligible_projections=config.eligible_projections,
        uniform_bit_width=3,
        group_size=config.quant_group_size,
    )

    targets = (
        [("Q3_mixed_match_int4", ref_storage_int4["total_bytes"])]
        if smoke_test
        else [
            ("Q3_mixed_match_int4", ref_storage_int4["total_bytes"]),
            ("Q3_mixed_match_int3", ref_storage_int3["total_bytes"]),
        ]
    )

    for cond, target_bytes in targets:
        min_hs = 10 if smoke_test else config.hellaswag_total_samples
        if not state_mgr.is_completed(cond, min_samples=min_hs):
            print("\n" + "=" * 80)
            print(f"STEP 9: Running Q3 Mixed Precision matching budget {target_bytes:,} bytes...")
            print("=" * 80)
            state_mgr.mark_in_progress(cond)
            alloc_map = solve_q3_bit_allocation(
                model=model,
                eligible_projections=config.eligible_projections,
                quant_sensitivity=quant_sensitivity,
                target_packed_bytes=target_bytes,
                allowed_bits=config.q3_allowed_bits,
                group_size=config.quant_group_size,
            )
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
                cond, is_full_hellaswag=True, bit_allocations=alloc_map
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

            # HellaSwag bootstrap
            hs_a = [x["norm"] for x in res_a["_per_example_hs"]]
            hs_b = [x["norm"] for x in res_b["_per_example_hs"]]
            min_len = min(len(hs_a), len(hs_b))
            if min_len > 0:
                ci_hs = compute_paired_bootstrap_confidence_interval(
                    hs_a[:min_len], hs_b[:min_len], num_resamples=config.bootstrap_resamples
                )
                bootstrap_results[f"{comp_name}_hellaswag_acc_norm"] = ci_hs

            # WikiText-2 bootstrap (resample windows)
            w_a = [x["nll"] for x in res_a["_per_window_wiki"]]
            w_b = [x["nll"] for x in res_b["_per_window_wiki"]]
            min_len = min(len(w_a), len(w_b))
            if min_len > 0:
                ci_w = compute_paired_bootstrap_confidence_interval(
                    w_a[:min_len], w_b[:min_len], num_resamples=config.bootstrap_resamples
                )
                bootstrap_results[f"{comp_name}_wikitext_nll"] = ci_w

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
