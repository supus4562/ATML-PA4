"""task2/runner.py — Master orchestrator for Task 2 (Efficient Attention & Context Memory).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task2.config import (
    LOCAL_RESULTS_DIR,
    CLOUD_CHECKPOINT_DIR,
    TRAIN_SEEDS,
    KV_MODEL_NAME,
)
from task2.checkpoint import Task2StateManager
from task2.operators_bench import (
    verify_causality,
    run_operator_grid,
    run_extra_operator_ablations,
)
from task2.mqar_train import train_mqar_model
from task2.mqar_eval import (
    evaluate_mqar_model,
    run_sequence_length_sweep,
)
from task2.mqar_dataset import load_frozen_mqar_test
from task2.kv_eval import run_full_kv_evaluation


def run_part_a(state_mgr: Task2StateManager, device: str = "cuda", smoke_test: bool = False) -> Dict[str, Any]:
    """Execute Part A: Attention Operators, Causality Verification, and Tiling Ablations."""
    if state_mgr.is_completed("part_a_full") and not smoke_test:
        print("\n[Resume] Part A already completed. Skipping.")
        return state_mgr.get_part_a()

    state_mgr.mark_in_progress("part_a")
    print("\n" + "=" * 80)
    print("STEP 1: RUNNING PART A (EXACT ATTENTION OPERATORS & ONLINE TILING)")
    print("=" * 80)

    # 1. Causality check
    causality_res = verify_causality(device=device)

    # 2. Operator grid benchmark
    lengths = [256, 512] if smoke_test else None
    grid_res = run_operator_grid(lengths=lengths, device=device)

    # 3. Extra ablations (block size sweep & dynamic range stress test)
    ablations_res = run_extra_operator_ablations(device=device)

    part_a_results = {
        "causality": causality_res,
        "grid": grid_res,
        "extra_ablations": ablations_res,
    }

    if not smoke_test:
        state_mgr.record_part_a(part_a_results)

    return part_a_results


def run_part_b(state_mgr: Task2StateManager, device: str = "cuda", smoke_test: bool = False) -> Dict[str, Any]:
    """Execute Part B: Multi-Query Associative Recall (MQAR) training & evaluation."""
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING PART B (MULTI-QUERY ASSOCIATIVE RECALL BENCHMARK)")
    print("=" * 80)

    test_examples = load_frozen_mqar_test()
    updates = 50 if smoke_test else 4500
    seeds = [6304] if smoke_test else TRAIN_SEEDS

    # Required model training combinations:
    # 1. Dense: Mixed (K=4, 8, 16) + Near-only (K=16)
    # 2. Window W=64: Near-only (K=4, 8, 16) + Mixed (K=16)
    # 3. Linear Attention: Mixed (K=4, 8, 16)
    # 4. Mixed Dilated: Mixed (K=4, 16)
    # 5. DGLA (Intervention): Mixed (K=16)
    training_grid = [
        ("dense", 4, "mixed"),
        ("dense", 8, "mixed"),
        ("dense", 16, "mixed"),
        ("dense", 16, "near"),
        ("window", 4, "near"),
        ("window", 8, "near"),
        ("window", 16, "near"),
        ("window", 16, "mixed"),
        ("linear", 4, "mixed"),
        ("linear", 8, "mixed"),
        ("linear", 16, "mixed"),
        ("dilated", 4, "mixed"),
        ("dilated", 16, "mixed"),
        ("dgla", 16, "mixed"),  # Controlled Intervention
    ]

    if smoke_test:
        training_grid = [("dense", 16, "mixed")]

    part_b_results = {}

    for attn_type, k, regime in training_grid:
        for seed in seeds:
            run_key = f"{attn_type}_k{k}_{regime}_seed{seed}"
            step_id = f"part_b_{run_key}"

            if state_mgr.is_completed(step_id) and not smoke_test:
                print(f"[Resume] Model '{run_key}' already completed. Skipping.")
                continue

            state_mgr.mark_in_progress(step_id)

            # Train model
            model, train_summary = train_mqar_model(
                attn_type=attn_type,
                k=k,
                regime=regime,
                seed=seed,
                device=device,
                total_updates=updates,
            )

            # Evaluate on near, far, mixed test sets
            eval_metrics = {}
            for eval_regime in ["near", "far", "mixed"]:
                eval_res = evaluate_mqar_model(
                    model=model,
                    test_examples=test_examples,
                    k=k,
                    regime=eval_regime,
                    device=device,
                )
                eval_metrics[eval_regime] = eval_res
                print(f"  [{run_key}] Test ({eval_regime:<5}): Acc = {eval_res['overall_accuracy']*100:5.2f}%")

            # Test-time mask control (for learned dense models)
            if attn_type == "dense" and k == 16 and regime == "mixed":
                row_idx = torch.arange(256, device=device).unsqueeze(1)
                col_idx = torch.arange(256, device=device).unsqueeze(0)
                dist = row_idx - col_idx
                w64_mask = ((dist >= 0) & (dist < 64)).unsqueeze(0).unsqueeze(0)
                control_res = evaluate_mqar_model(
                    model=model,
                    test_examples=test_examples,
                    k=k,
                    regime="mixed",
                    device=device,
                    mask_override=w64_mask,
                )
                eval_metrics["test_time_w64_mask_control"] = control_res
                print(f"  [{run_key}] Test-time W=64 Mask Control Acc: {control_res['overall_accuracy']*100:5.2f}%")

            model_record = {
                "train_summary": train_summary,
                "eval_metrics": eval_metrics,
            }
            part_b_results[run_key] = model_record

            if not smoke_test:
                state_mgr.record_part_b_model(run_key, model_record, state_dict=model.state_dict())

    # Sequence length forward sweeps
    if not smoke_test:
        sweep_records = {}
        for a_type in ["dense", "window", "linear", "dilated", "dgla"]:
            sweep_records[a_type] = run_sequence_length_sweep(a_type, device=device)
        state_mgr.state["part_b"]["sequence_length_sweeps"] = sweep_records
        state_mgr._save_state()

    return part_b_results


def run_part_c(
    state_mgr: Task2StateManager,
    device: str = "cuda",
    smoke_test: bool = False,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Execute Part C: Pretrained KV Memory Compression across the 5 matched-budget policies."""
    if state_mgr.is_completed("part_c_full") and not force_rerun and not smoke_test:
        print("\n[Resume] Part C already completed. Skipping.")
        return state_mgr.get_part_c()

    state_mgr.mark_in_progress("part_c")
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING PART C (PRETRAINED KV MEMORY COMPRESSION)")
    print("=" * 80)

    print(f"Loading '{KV_MODEL_NAME}' in Bfloat16 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(KV_MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        KV_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map=device,
    )
    model.eval()

    max_cases = 2 if smoke_test else None
    kv_results = run_full_kv_evaluation(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_cases=max_cases,
    )

    if not smoke_test:
        state_mgr.record_part_c(kv_results)

        # Extension: Run Complete Mini-RULER Context Benchmarks
        try:
            from task2.eval_context_benchmark import run_context_benchmarks
            print("\n" + "=" * 80)
            print("STEP 4: RUNNING MINI-RULER CONTEXT BENCHMARKS")
            print("=" * 80)
            ctx_results = run_context_benchmarks(model=model, tokenizer=tokenizer, device=device)
            state_mgr.state["context_benchmarks"] = ctx_results
            state_mgr._save_state()
        except Exception as e:
            print(f"[Warning] Context benchmarks encountered error: {e}")

    return kv_results


def main():
    parser = argparse.ArgumentParser(description="ATML PA4 Task 2 Orchestrator")
    parser.add_argument("--part-a", action="store_true", help="Run Part A attention operator benchmarks")
    parser.add_argument("--part-b", action="store_true", help="Run Part B MQAR training and evaluation")
    parser.add_argument("--part-c", action="store_true", help="Run Part C KV memory compression evaluation")
    parser.add_argument("--run-all", action="store_true", help="Run Part A, B, and C sequentially")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick micro-verification test")
    parser.add_argument("--checkpoint-dir", type=str, default=str(CLOUD_CHECKPOINT_DIR))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Task 2 Orchestrator] Initializing on {device.upper()}...")

    # Determine checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.parent.exists():
        ckpt_dir = LOCAL_RESULTS_DIR
    state_mgr = Task2StateManager(checkpoint_dir=ckpt_dir)

    # Default to run_all if no specific part selected
    if not (args.part_a or args.part_b or args.part_c or args.smoke_test):
        args.run_all = True

    t0 = time.perf_counter()

    if args.smoke_test:
        print("\n[Mode] Running Fast Smoke Test...")
        run_part_a(state_mgr, device=device, smoke_test=True)
        run_part_b(state_mgr, device=device, smoke_test=True)
        run_part_c(state_mgr, device=device, smoke_test=True)
        print(f"\n✓ Smoke Test completed successfully in {time.perf_counter() - t0:.1f}s.")
        return

    if args.part_a or args.run_all:
        run_part_a(state_mgr, device=device)

    if args.part_b or args.run_all:
        run_part_b(state_mgr, device=device)

    if args.part_c or args.run_all:
        run_part_c(state_mgr, device=device)

    total_time = time.perf_counter() - t0
    print(f"\n{'='*80}")
    print(f"✓ ALL REQUESTED TASK 2 STEPS COMPLETED IN {total_time/60.0:.2f} MINUTES")
    print(f"Results saved to: {ckpt_dir / 'all_results.json'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
