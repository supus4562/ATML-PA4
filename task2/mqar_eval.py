"""task2/mqar_eval.py — Evaluation suite for MQAR: distance binning, chance baselines, and sequence length sweeps.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from task2.config import (
    MQAR_SEQ_LEN,
    MQAR_VALUE_RANGE,
    MQAR_DISTANCE_BINS,
    MQAR_INFERENCE_SWEEP_LENGTHS,
)
from task2.mqar_models import MQARTransformer
from task2.mqar_dataset import MQARDataset, load_frozen_mqar_test


def evaluate_mqar_model(
    model: MQARTransformer,
    test_examples: List[Dict],
    k: int,
    regime: str,
    device: str = "cuda",
    batch_size: int = 64,
    mask_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Evaluate accuracy on test examples, broke down by distance bins and compared to chance."""
    # Filter test examples matching (k, regime)
    filtered = [ex for ex in test_examples if ex["K"] == k and ex["regime"] == regime]
    if not filtered:
        raise ValueError(f"No test examples found for K={k}, regime={regime}")

    model.eval()
    dataset = MQARDataset(filtered)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_queries = 0
    correct_queries = 0

    # Distance bin trackers: bin_key -> [correct, total]
    bin_stats = {label: [0, 0] for _, _, label in MQAR_DISTANCE_BINS}

    # Dilated modulo-4 trackers: [correct, total]
    mod4_stats = {"mod4_zero": [0, 0], "mod4_nonzero": [0, 0]}

    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)          # (B, T) with class idx 0..63 or -100
            query_mask = batch["query_mask"].to(device)  # (B, T) bool
            distances = batch["distances"].numpy()       # (B, K)
            query_pos = batch["query_positions"].numpy() # (B, K)

            logits = model(input_ids, mask_override=mask_override)  # (B, T, 64)
            preds = torch.argmax(logits, dim=-1)                   # (B, T)

            # Evaluate accuracy strictly at marked query positions
            b_size, t_size = query_mask.shape
            for b in range(b_size):
                for q_idx in range(len(query_pos[b])):
                    pos = query_pos[b][q_idx]
                    dist = distances[b][q_idx]
                    gold = target[b, pos].item()
                    pred = preds[b, pos].item()

                    is_correct = (pred == gold)
                    total_queries += 1
                    if is_correct:
                        correct_queries += 1

                    # Bin attribution
                    for low, high, label in MQAR_DISTANCE_BINS:
                        if low <= dist < high:
                            bin_stats[label][1] += 1
                            if is_correct:
                                bin_stats[label][0] += 1
                            break

                    # Modulo-4 attribution (for dilated ablation)
                    if dist % 4 == 0:
                        mod4_stats["mod4_zero"][1] += 1
                        if is_correct:
                            mod4_stats["mod4_zero"][0] += 1
                    else:
                        mod4_stats["mod4_nonzero"][1] += 1
                        if is_correct:
                            mod4_stats["mod4_nonzero"][0] += 1

    overall_acc = (correct_queries / total_queries) if total_queries > 0 else 0.0
    label_chance = 1.0 / 64.0  # 0.015625
    in_context_chance = 1.0 / float(k)

    bin_accuracies = {}
    for label, (corr, tot) in bin_stats.items():
        bin_accuracies[label] = {
            "accuracy": (corr / tot) if tot > 0 else 0.0,
            "correct": corr,
            "total": tot,
        }

    return {
        "k": k,
        "regime": regime,
        "overall_accuracy": overall_acc,
        "total_queries": total_queries,
        "correct_queries": correct_queries,
        "label_chance": label_chance,
        "in_context_chance": in_context_chance,
        "distance_bins": bin_accuracies,
        "mod4_breakdown": {
            "dist_mod_4_zero_acc": (mod4_stats["mod4_zero"][0] / max(1, mod4_stats["mod4_zero"][1])),
            "dist_mod_4_nonzero_acc": (mod4_stats["mod4_nonzero"][0] / max(1, mod4_stats["mod4_nonzero"][1])),
        },
    }


def run_sequence_length_sweep(
    attn_type: str,
    device: str = "cuda",
    batch_size: int = 2,
    hidden_dim: int = 128,
) -> List[Dict[str, Any]]:
    """Forward-only inference latency & VRAM sweep at T in {256, 1024, 4096, 8192}."""
    model = MQARTransformer(attn_type=attn_type, d_model=hidden_dim).to(device)
    model.eval()

    results = []
    print(f"\n[Sequence Length Sweep] Benchmarking {attn_type} up to T=8192...")

    with torch.inference_mode():
        for T in MQAR_INFERENCE_SWEEP_LENGTHS:
            # Memory safety: check if T is feasible for dense
            if attn_type == "dense" and T > 4096 and not torch.cuda.is_available():
                continue

            dummy_input = torch.randint(2, 64, (batch_size, T), device=device)

            # Warmup
            for _ in range(3):
                _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            reps = 10
            for _ in range(reps):
                _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = ((time.perf_counter() - t0) / reps) * 1000.0

            peak_vram_mb = 0.0
            if torch.cuda.is_available():
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            record = {
                "length": T,
                "latency_ms": latency_ms,
                "peak_vram_mb": peak_vram_mb,
            }
            results.append(record)
            print(f"  T={T:<5} | Latency: {latency_ms:6.2f}ms | Peak VRAM: {peak_vram_mb:6.1f}MB")

    return results
