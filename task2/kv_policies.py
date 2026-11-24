"""task2/kv_policies.py — The 5 matched-budget KV cache compression and eviction policies.

Every policy returns a `PolicyResult` with:
  k / v              per-layer tensors placed into the cache (dequantized where quantized)
  keep_indices       original positions physically present in the cache (sorted)
  fp16_selected      original positions held in FP16 by a selection rule (eviction / hybrid);
                     used for the retained-evidence metric
  packed_bytes       serialized bytes of the pre-question context cache, including payload,
                     scale/offset metadata, FP16 residual and any position mask
  quantized_upto     number of leading cache slots already stored at low precision (complete
                     64-token groups); later tokens form the FP16 tail that is re-grouped as the
                     cache grows (see kv_core.requantize_tail)
  recon_error_k/v    relative Frobenius reconstruction error of K and V (quantized policies)

Retention is SHARED across all layers and heads (one global set of positions), so position
metadata is stored once: a 1-bit-per-position mask of ceil(L/8) bytes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from task2.config import KV_SINK_TOKENS, KV_RECENT_TOKENS, KV_GROUP_SIZE
from task2.kv_cache_models import (
    compute_matched_int4_bytes,
    compute_fp16_retained_token_count,
    compute_hybrid_fp16_token_count,
    position_mask_bytes,
    bytes_per_fp16_token,
)
from task2.kv_quantizers import quantize_affine_group


@dataclass
class PolicyResult:
    k: List[torch.Tensor]
    v: List[torch.Tensor]
    keep_indices: torch.Tensor
    fp16_selected: Optional[torch.Tensor]
    packed_bytes: int
    budget_bytes: int
    quantized_upto: int = 0
    quant_layout: Optional[str] = None  # "token_wise" | "channel_k_token_v" | None
    quant_bits: Optional[int] = None
    recon_error_k: Optional[float] = None
    recon_error_v: Optional[float] = None
    extra: Dict = field(default_factory=dict)


def pool_importance(scores: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """1D max-pooling of per-position importance (SnapKV-style clustering)."""
    s = scores.float().view(1, 1, -1)
    return F.max_pool1d(s, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).view(-1)


def _select_middle(importance: torch.Tensor, L: int, n_total: int) -> torch.Tensor:
    """8 sink + 64 recent safeguards, remaining budget filled with top-importance positions."""
    device = importance.device
    if n_total < min(L, KV_SINK_TOKENS + KV_RECENT_TOKENS):
        raise ValueError(f"Budget allows {n_total} positions, fewer than the 8 sink + 64 recent safeguards (L={L})")
    sink = min(KV_SINK_TOKENS, L)
    recent = min(KV_RECENT_TOKENS, L - sink)
    sink_idx = torch.arange(0, sink, device=device)
    recent_idx = torch.arange(L - recent, L, device=device)
    n_mid = max(0, n_total - sink - recent)
    mid_lo, mid_hi = sink, L - recent
    if n_mid > 0 and mid_hi > mid_lo:
        pooled = pool_importance(importance[:L])
        top = torch.topk(pooled[mid_lo:mid_hi], k=min(n_mid, mid_hi - mid_lo)).indices + mid_lo
    else:
        top = torch.empty(0, dtype=torch.long, device=device)
    return torch.sort(torch.cat([sink_idx, top, recent_idx]).unique()).values


def _rel_err(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    num = sum(float(((x.float() - y.float()) ** 2).sum()) for x, y in zip(a, b))
    den = sum(float((x.float() ** 2).sum()) for x in a)
    return math.sqrt(num / max(den, 1e-30))


def quantize_kv_block(k: torch.Tensor, v: torch.Tensor, layout: str, bits: int):
    """Quantize a block of complete 64-token groups (shapes (B, H, T, d), T % 64 == 0).

    channel_k_token_v: K grouped along tokens per channel, V grouped along channels per token.
    token_wise:        K and V both grouped along channels per token.
    Returns dequantized K, V and serialized bytes.
    """
    if layout == "channel_k_token_v":
        k_deq, _, _, kb = quantize_affine_group(k, bits=bits, group_size=KV_GROUP_SIZE, dim=2)
    elif layout == "token_wise":
        k_deq, _, _, kb = quantize_affine_group(k, bits=bits, group_size=KV_GROUP_SIZE, dim=-1)
    else:
        raise ValueError(layout)
    v_deq, _, _, vb = quantize_affine_group(v, bits=bits, group_size=KV_GROUP_SIZE, dim=-1)
    return k_deq, v_deq, kb + vb


# ──────────────────────────────────────────────────────────────────────────────
# Policy 1: Sink + Recent FP16 Eviction
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_sink_recent(k_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> PolicyResult:
    """8 sink + 64 recent, remainder filled with the most recent positions.

    The retained set is two contiguous ranges, so no per-position mask is stored.
    """
    L = k_list[0].shape[2]
    budget = compute_matched_int4_bytes(L, b=4)
    n_retain = min(L, compute_fp16_retained_token_count(L, b=4, metadata_bytes=0))
    sink = min(KV_SINK_TOKENS, L)
    tail_start = max(sink, L - (n_retain - sink))
    device = k_list[0].device
    keep = torch.cat([torch.arange(0, sink, device=device), torch.arange(tail_start, L, device=device)]).unique()
    packed = len(keep) * bytes_per_fp16_token()
    assert packed <= budget
    return PolicyResult(
        k=[k[:, :, keep, :] for k in k_list], v=[v[:, :, keep, :] for v in v_list],
        keep_indices=keep, fp16_selected=keep, packed_bytes=packed, budget_bytes=budget,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Policy 2: Question-Aware FP16 Eviction (simplified SnapKV-inspired, shared across heads/layers)
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_question_aware(
    k_list: List[torch.Tensor], v_list: List[torch.Tensor], importance: torch.Tensor
) -> PolicyResult:
    L = k_list[0].shape[2]
    budget = compute_matched_int4_bytes(L, b=4)
    n_retain = min(L, compute_fp16_retained_token_count(L, b=4, metadata_bytes=position_mask_bytes(L)))
    keep = _select_middle(importance, L, n_retain)
    packed = len(keep) * bytes_per_fp16_token() + position_mask_bytes(L)
    assert packed <= budget
    return PolicyResult(
        k=[k[:, :, keep, :] for k in k_list], v=[v[:, :, keep, :] for v in v_list],
        keep_indices=keep, fp16_selected=keep, packed_bytes=packed, budget_bytes=budget,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Policies 3 & 4: uniform INT4 quantization, two layouts, identical trailing-FP16 rule
# ──────────────────────────────────────────────────────────────────────────────

def _apply_uniform_quant(k_list, v_list, layout: str, bits: int = 4) -> PolicyResult:
    L = k_list[0].shape[2]
    full = (L // KV_GROUP_SIZE) * KV_GROUP_SIZE
    k_out, v_out, total = [], [], 0
    for k, v in zip(k_list, v_list):
        k_new, v_new = k.clone(), v.clone()
        if full > 0:
            kq, vq, b = quantize_kv_block(k[:, :, :full], v[:, :, :full], layout, bits)
            k_new[:, :, :full], v_new[:, :, :full] = kq, vq
            total += b
        k_out.append(k_new)
        v_out.append(v_new)
    # FP16 trailing residual r = L mod 64 for both K and V
    total += (L - full) * bytes_per_fp16_token()
    budget = compute_matched_int4_bytes(L, b=4)
    return PolicyResult(
        k=k_out, v=v_out, keep_indices=torch.arange(L, device=k_list[0].device), fp16_selected=None,
        packed_bytes=total, budget_bytes=budget, quantized_upto=full, quant_layout=layout, quant_bits=bits,
        recon_error_k=_rel_err(k_list, k_out), recon_error_v=_rel_err(v_list, v_out),
    )


def apply_policy_token_wise_int4(k_list, v_list) -> PolicyResult:
    return _apply_uniform_quant(k_list, v_list, "token_wise", 4)


def apply_policy_channel_k_token_v_int4(k_list, v_list) -> PolicyResult:
    """The reference: its packed bytes define the matched budget M_4(L) exactly."""
    res = _apply_uniform_quant(k_list, v_list, "channel_k_token_v", 4)
    assert res.packed_bytes == res.budget_bytes, (res.packed_bytes, res.budget_bytes)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Policy 5: Hybrid — FP16 selection + channel-K / token-V INT2 for the rest
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_hybrid(
    k_list: List[torch.Tensor], v_list: List[torch.Tensor], importance: torch.Tensor
) -> PolicyResult:
    """8 sink + 64 recent + top question-aware positions in FP16; remaining positions in
    channel-wise-K / token-wise-V INT2. INT2 positions always form complete 64-token groups
    (the low-precision set is gathered in chronological order and grouped along that order).
    """
    L = k_list[0].shape[2]
    budget = compute_matched_int4_bytes(L, b=4)
    n_fp16, n_int2 = compute_hybrid_fp16_token_count(L)
    fp16_idx = _select_middle(importance, L, n_fp16)
    mask = torch.zeros(L, dtype=torch.bool, device=fp16_idx.device)
    mask[fp16_idx] = True
    int2_idx = torch.arange(L, device=fp16_idx.device)[~mask]
    assert len(int2_idx) % KV_GROUP_SIZE == 0, len(int2_idx)

    k_out, v_out, total = [], [], 0
    for k, v in zip(k_list, v_list):
        k_new, v_new = k.clone(), v.clone()
        if len(int2_idx) > 0:
            kq, vq, b = quantize_kv_block(k[:, :, int2_idx], v[:, :, int2_idx], "channel_k_token_v", 2)
            k_new[:, :, int2_idx], v_new[:, :, int2_idx] = kq, vq
            total += b
        k_out.append(k_new)
        v_out.append(v_new)
    total += len(fp16_idx) * bytes_per_fp16_token() + position_mask_bytes(L)
    assert total <= budget, (total, budget)
    return PolicyResult(
        k=k_out, v=v_out, keep_indices=torch.arange(L, device=fp16_idx.device), fp16_selected=fp16_idx,
        packed_bytes=total, budget_bytes=budget, quantized_upto=0,  # positions are interleaved: no tail regrouping
        quant_layout=None, quant_bits=2,
        recon_error_k=_rel_err(k_list, k_out), recon_error_v=_rel_err(v_list, v_out),
        extra={"n_fp16": int(len(fp16_idx)), "n_int2": int(len(int2_idx))},
    )


POLICIES = [
    "fp16_reference",
    "sink_recent_fp16",
    "question_aware_fp16",
    "token_wise_int4",
    "channel_k_token_v_int4",
    "hybrid_fp16_int2",
]


def apply_policy(name: str, k_list, v_list, importance: Optional[torch.Tensor]) -> PolicyResult:
    L = k_list[0].shape[2]
    if name == "fp16_reference":
        return PolicyResult(
            k=k_list, v=v_list, keep_indices=torch.arange(L, device=k_list[0].device), fp16_selected=None,
            packed_bytes=L * bytes_per_fp16_token(), budget_bytes=compute_matched_int4_bytes(L, b=4),
        )
    if name == "sink_recent_fp16":
        return apply_policy_sink_recent(k_list, v_list)
    if name == "question_aware_fp16":
        return apply_policy_question_aware(k_list, v_list, importance)
    if name == "token_wise_int4":
        return apply_policy_token_wise_int4(k_list, v_list)
    if name == "channel_k_token_v_int4":
        return apply_policy_channel_k_token_v_int4(k_list, v_list)
    if name == "hybrid_fp16_int2":
        return apply_policy_hybrid(k_list, v_list, importance)
    raise ValueError(f"Unknown policy: {name}")
