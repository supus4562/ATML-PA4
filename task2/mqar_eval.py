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


IMPLEMENTATION_NOTES = {
    "dense": "explicit QK^T scores with a dense T x T boolean causal mask (materialized)",
    "window": "dense T x T scores + materialized boolean window mask (correctness implementation, "
              "NOT an efficient sparse kernel: same O(T^2) work and memory as dense)",
    "dilated": "dense T x T scores + materialized per-head boolean local/dilated masks (correctness "
               "implementation, NOT an efficient sparse kernel)",
    "linear": "parallel causal form materializing per-position prefix states (B, H, T, d, d) via cumsum; "
              "the recurrent O(d^2) state form is not what is timed",
    "dgla": "parallel form materializing a (B, H, T, T) decay matrix; the recurrent O(d^2) state form "
            "is not what is timed",
}


def theoretical_attention_cost(attn_type: str, T: int, n_heads: int = 4, d_head: int = 32,
                               n_layers: int = 2, window: int = 64, stride: int = 4,
                               dilated_span: int = 252) -> Dict[str, Any]:
    """Analytical per-sequence costs (independent of the implementation that is timed).

    attended_pairs: number of (query, key) pairs with nonzero attention, summed over layers/heads.
    candidates_per_query_max: largest number of key/value positions one query can read in one head.
    decode_state_bytes_fp32: persistent per-sequence inference state after T tokens
        (attention: cached K and V rows that can still be attended; linear/DGLA: S (d x d) + z (d)).
    """
    def pairs_window(w):
        return sum(min(i + 1, w) for i in range(T))

    def pairs_dilated():
        return sum(len(range(0, min(i, dilated_span) + 1, stride)) for i in range(T))

    if attn_type == "dense":
        per_head, cand, cached = [T * (T + 1) // 2] * n_heads, T, T
        state = n_layers * n_heads * cached * 2 * d_head * 4
    elif attn_type == "window":
        per_head, cand, cached = [pairs_window(window)] * n_heads, window, min(T, window)
        state = n_layers * n_heads * cached * 2 * d_head * 4
    elif attn_type == "dilated":
        per_head = [pairs_window(window)] * 2 + [pairs_dilated()] * 2
        cand = max(window, len(range(0, dilated_span + 1, stride)))
        cached = min(T, dilated_span + 1)
        state = n_layers * (2 * min(T, window) + 2 * cached) * 2 * d_head * 4
    elif attn_type in ("linear", "dgla"):
        per_head, cand = [T] * n_heads, None  # one state read per query (no pairwise scores)
        state = n_layers * n_heads * (d_head * d_head + d_head) * 4
    else:
        raise ValueError(attn_type)
    return {
        "T": T,
        "attended_pairs_total": int(sum(per_head) * n_layers),
        "candidates_per_query_max": cand,
        "decode_state_bytes_fp32": int(state),
        "two_layer_receptive_field": (2 * (window - 1) if attn_type == "window" else None),
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

            # Fixed random token inputs (seeded) so every mechanism sees identical shapes/values.
            gen = torch.Generator(device=device).manual_seed(6304 + T)
            dummy_input = torch.randint(2, 64, (batch_size, T), device=device, generator=gen)

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
                "repetitions": reps,
                "warmup": 3,
                "batch_size": batch_size,
                "implementation": IMPLEMENTATION_NOTES[attn_type],
                "theoretical": theoretical_attention_cost(attn_type, T),
            }
            results.append(record)
            print(f"  T={T:<5} | Latency: {latency_ms:6.2f}ms | Peak VRAM: {peak_vram_mb:6.1f}MB")

    return results
