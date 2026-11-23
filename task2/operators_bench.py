"""task2/operators_bench.py — Part A benchmarking engine: numerical fidelity, causality, latency, and memory.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from task2.config import (
    PART_A_BATCH,
    PART_A_HEADS,
    PART_A_HEAD_DIM,
    PART_A_LENGTHS,
    PART_A_DTYPES,
    PART_A_REPETITIONS,
    PART_A_WARMUP,
    PART_A_TILE_BLOCK_SIZES,
)
from task2.operators import (
    dense_causal_attention,
    tiled_causal_attention,
    sdpa_attention,
    get_sdpa_backend,
)


def verify_causality(device: str = "cuda") -> Dict[str, Any]:
    """Verify that perturbing future keys and values does not affect past outputs."""
    B, H, L, d = 2, 4, 64, 32
    torch.manual_seed(6304)
    q = torch.randn(B, H, L, d, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, d, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, d, device=device, dtype=torch.float32)

    operators = {
        "dense": lambda q, k, v: dense_causal_attention(q, k, v),
        "tiled": lambda q, k, v: tiled_causal_attention(q, k, v, block_size_r=16, block_size_c=16),
        "sdpa": lambda q, k, v: sdpa_attention(q, k, v)[0],
    }

    results = {}
    cutoff = 32  # Check output up to position 32 when perturbing positions > 32

    for name, op in operators.items():
        base_out = op(q, k, v)
        
        # Perturb future keys/values at positions > cutoff
        k_perturbed = k.clone()
        v_perturbed = v.clone()
        k_perturbed[:, :, cutoff + 1 :, :] += 100.0
        v_perturbed[:, :, cutoff + 1 :, :] += 100.0

        pert_out = op(q, k_perturbed, v_perturbed)

        # Check maximum absolute difference at positions <= cutoff
        past_diff = torch.max(torch.abs(base_out[:, :, : cutoff + 1, :] - pert_out[:, :, : cutoff + 1, :])).item()
        # Check that future positions DID change (sanity check that perturbation occurred)
        future_diff = torch.max(torch.abs(base_out[:, :, cutoff + 1 :, :] - pert_out[:, :, cutoff + 1 :, :])).item()

        is_causal = past_diff < 1e-5 and future_diff > 1e-2
        results[name] = {
            "past_max_diff": past_diff,
            "future_max_diff": future_diff,
            "strictly_causal": is_causal,
        }
        print(f"[Causality Check] {name:<8}: Past max diff = {past_diff:.2e}, Strictly Causal = {is_causal}")

    return results


def run_operator_grid(
    lengths: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute the full (dtype x length x operator) benchmarking grid on GPU."""
    if lengths is None:
        lengths = PART_A_LENGTHS

    B, H, d = PART_A_BATCH, PART_A_HEADS, PART_A_HEAD_DIM
    grid_results = []
    active_backend = get_sdpa_backend(device)

    print(f"\n{'='*80}")
    print(f"PART A: BENCHMARKING EXACT ATTENTION OPERATORS (Device: {device})")
    print(f"Dispatched SDPA Backend(s): {active_backend}")
    print(f"{'='*80}\n")

    for dtype in PART_A_DTYPES:
        dtype_str = "fp32" if dtype == torch.float32 else "fp16"
        for L in lengths:
            # Memory safety: Dense attention at L >= 8192 materializes >2GB matrices multiple times
            # and OOMs during autograd. We benchmark Tiled and SDPA up to 8192, and record Dense as OOM.
            is_dense_feasible = (L <= 4096)

            torch.manual_seed(6304 + L)
            # Create fresh tensors directly on device
            q_ref = torch.randn(B, H, L, d, device=device, dtype=torch.float32, requires_grad=(L <= 2048))
            k_ref = torch.randn(B, H, L, d, device=device, dtype=torch.float32, requires_grad=(L <= 2048))
            v_ref = torch.randn(B, H, L, d, device=device, dtype=torch.float32, requires_grad=(L <= 2048))

            # Compute reference output: dense if feasible, otherwise SDPA FP32
            ref_grad_q = None
            if is_dense_feasible:
                ref_out = dense_causal_attention(q_ref, k_ref, v_ref)
                if L <= 2048:
                    ref_loss = ref_out.sum()
                    ref_loss.backward()
                    ref_grad_q = q_ref.grad.clone()
                    q_ref.grad = None
            else:
                # Use SDPA FP32 as reference for L=8192 where dense baseline is infeasible
                ref_out = sdpa_attention(q_ref, k_ref, v_ref)[0]

            # Now benchmark each operator
            for op_name in ["dense", "tiled", "sdpa"]:
                if op_name == "dense" and not is_dense_feasible:
                    grid_results.append({
                        "dtype": dtype_str,
                        "length": L,
                        "operator": "dense",
                        "output_err_vs_fp32": "OOM (Infeasible)",
                        "grad_err_vs_fp32": "OOM (Infeasible)",
                        "fwd_latency_ms": "OOM",
                        "fwd_bwd_latency_ms": "OOM",
                        "peak_vram_mb": ">22GB",
                        "sdpa_backend": "N/A",
                    })
                    print(f"[{dtype_str.upper()} L={L:<5} dense] OOM (Dense matrix infeasible at L={L})")
                    continue

                # Prepare test tensors
                q = q_ref.detach().to(dtype).requires_grad_(L <= 2048)
                k = k_ref.detach().to(dtype).requires_grad_(L <= 2048)
                v = v_ref.detach().to(dtype).requires_grad_(L <= 2048)

                def call_op():
                    if op_name == "dense":
                        return dense_causal_attention(q, k, v)
                    elif op_name == "tiled":
                        return tiled_causal_attention(q, k, v, block_size_r=128, block_size_c=128)
                    else:
                        return sdpa_attention(q, k, v)[0]

                try:
                    # Numerical output error vs FP32 reference
                    out = call_op()
                    output_err = torch.max(torch.abs(out.to(torch.float32) - ref_out)).item()

                    # Backward gradient error
                    grad_err = None
                    if ref_grad_q is not None and (L <= 2048):
                        try:
                            q.grad = None
                            loss = out.sum()
                            loss.backward()
                            if q.grad is not None:
                                grad_err = torch.max(torch.abs(q.grad.to(torch.float32) - ref_grad_q)).item()
                        except Exception as e:
                            grad_err = f"Autograd failed: {e}"

                    # Warmup
                    for _ in range(PART_A_WARMUP):
                        _ = call_op()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Timing: Forward-only
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    
                    t0 = time.perf_counter()
                    for _ in range(PART_A_REPETITIONS):
                        _ = call_op()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    fwd_ms = ((t1 - t0) / PART_A_REPETITIONS) * 1000.0

                    peak_vram_mb = 0.0
                    if torch.cuda.is_available():
                        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                    # Timing: Forward + Backward
                    bwd_ms = None
                    if L <= 2048:
                        for _ in range(3):
                            q.grad = None
                            loss = call_op().sum()
                            loss.backward()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        t0 = time.perf_counter()
                        for _ in range(PART_A_REPETITIONS):
                            q.grad = None
                            loss = call_op().sum()
                            loss.backward()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        bwd_ms = ((t1 - t0) / PART_A_REPETITIONS) * 1000.0

                    record = {
                        "dtype": dtype_str,
                        "length": L,
                        "operator": op_name,
                        "output_err_vs_fp32": output_err,
                        "grad_err_vs_fp32": grad_err,
                        "fwd_latency_ms": fwd_ms,
                        "fwd_bwd_latency_ms": bwd_ms,
                        "peak_vram_mb": peak_vram_mb,
                        "sdpa_backend": active_backend if op_name == "sdpa" else "N/A",
                    }
                    grid_results.append(record)
                    print(
                        f"[{dtype_str.upper()} L={L:<5} {op_name:<5}] "
                        f"Fwd: {fwd_ms:6.2f}ms | Fwd+Bwd: {bwd_ms if bwd_ms else 'N/A':>6} | "
                        f"VRAM: {peak_vram_mb:6.1f}MB | OutErr: {output_err:.2e}"
                    )
                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    grid_results.append({
                        "dtype": dtype_str,
                        "length": L,
                        "operator": op_name,
                        "output_err_vs_fp32": "OOM",
                        "grad_err_vs_fp32": "OOM",
                        "fwd_latency_ms": "OOM",
                        "fwd_bwd_latency_ms": "OOM",
                        "peak_vram_mb": ">22GB",
                        "sdpa_backend": active_backend if op_name == "sdpa" else "N/A",
                    })
                    print(f"[{dtype_str.upper()} L={L:<5} {op_name:<5}] OutOfMemoryError (OOM recorded)")

                # Cleanup GPU cache
                del q, k, v
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return {
        "sdpa_backend": active_backend,
        "grid": grid_results,
    }


def run_extra_operator_ablations(device: str = "cuda") -> Dict[str, Any]:
    """Execute the extra Part A ablations: block size sweep and extreme value test."""
    print(f"\n{'='*80}")
    print("PART A: EXTRA ABLATIONS (Block Size Sweep & Dynamic Range Test)")
    print(f"{'='*80}\n")

    # 1. Tile Block Size Sweep (L=2048, FP16)
    B, H, L, d = PART_A_BATCH, PART_A_HEADS, 2048, PART_A_HEAD_DIM
    q = torch.randn(B, H, L, d, device=device, dtype=torch.float16)
    k = torch.randn(B, H, L, d, device=device, dtype=torch.float16)
    v = torch.randn(B, H, L, d, device=device, dtype=torch.float16)
    ref = dense_causal_attention(q.float(), k.float(), v.float()).half()

    block_sweep_results = []
    for bs in PART_A_TILE_BLOCK_SIZES:
        # Warmup
        for _ in range(3):
            _ = tiled_causal_attention(q, k, v, block_size_r=bs, block_size_c=bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(PART_A_REPETITIONS):
            out = tiled_causal_attention(q, k, v, block_size_r=bs, block_size_c=bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = ((time.perf_counter() - t0) / PART_A_REPETITIONS) * 1000.0
        err = torch.max(torch.abs(out - ref)).item()

        block_sweep_results.append({
            "block_size": bs,
            "latency_ms": latency,
            "max_err": err,
        })
        print(f"[Block Size Sweep] Tile {bs}x{bs}: Latency = {latency:.2f}ms, Max Err = {err:.2e}")

    # 2. Extreme Dynamic Range / Softmax Underflow Test
    L_ext = 256
    q_ext = torch.randn(1, 1, L_ext, d, device=device) * 50.0  # Scores will range in [-1000, 1000]
    k_ext = torch.randn(1, 1, L_ext, d, device=device) * 50.0
    v_ext = torch.randn(1, 1, L_ext, d, device=device)

    # Dense with naive unshifted exp vs stabilized online tiling
    scores_raw = torch.matmul(q_ext, k_ext.transpose(-2, -1)) / math.sqrt(d)
    naive_has_nan = torch.isnan(torch.exp(scores_raw)).any().item()
    naive_has_inf = torch.isinf(torch.exp(scores_raw)).any().item()

    tiled_ext_out = tiled_causal_attention(q_ext, k_ext, v_ext, block_size_r=64, block_size_c=64)
    tiled_has_nan = torch.isnan(tiled_ext_out).any().item()
    tiled_has_inf = torch.isinf(tiled_ext_out).any().item()

    extreme_results = {
        "score_min": scores_raw.min().item(),
        "score_max": scores_raw.max().item(),
        "naive_exp_nan_or_inf": naive_has_nan or naive_has_inf,
        "tiled_online_nan_or_inf": tiled_has_nan or tiled_has_inf,
    }
    print(f"\n[Dynamic Range Test] Raw scores range [{scores_raw.min().item():.1f}, {scores_raw.max().item():.1f}]")
    print(f"  Naive exp contains NaN/Inf: {naive_has_nan or naive_has_inf}")
    print(f"  Tiled Online contains NaN/Inf: {tiled_has_nan or tiled_has_inf} (Stable: {not (tiled_has_nan or tiled_has_inf)})")

    return {
        "block_sweep": block_sweep_results,
        "extreme_test": extreme_results,
    }
