"""task2/kv_policies.py — The 5 matched-budget KV cache compression and eviction policies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from task2.config import (
    KV_SINK_TOKENS,
    KV_RECENT_TOKENS,
    KV_GROUP_SIZE,
)
from task2.kv_cache_models import (
    compute_matched_int4_bytes,
    compute_fp16_retained_token_count,
    compute_hybrid_fp16_token_count,
)
from task2.kv_quantizers import (
    quantize_token_wise_kv,
    quantize_channel_k_token_v,
    quantize_affine_group,
)


def compute_question_importance_scores(
    k_cache: List[torch.Tensor],
    q_question: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token importance scores using question-suffix attention weights.
    
    Args:
        k_cache: List of key tensors per layer, each (B, H_kv, L, d_h)
        q_question: Query tensor of the question suffix (B, H_q, L_q, d_h) or precomputed 1D scores (L,)
        
    Returns:
        importance: (L,) importance score per prompt position with 1D max pooling
    """
    device = k_cache[0].device
    L = k_cache[0].shape[2]

    # If already a 1D importance score vector (e.g. from true layer attention)
    if q_question.ndim == 1:
        scores = q_question.to(device).float()
        if len(scores) != L:
            scores = F.interpolate(scores.view(1, 1, -1), size=L, mode="nearest").squeeze()
        # 1D max pooling with kernel size 5 to capture neighboring tokens (entity/code clusters)
        pooled = F.max_pool1d(scores.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
        return pooled

    total_scores = torch.zeros(L, device=device, dtype=torch.float32)

    # Sample top/middle/bottom layers or all layers to accumulate attention
    step = max(1, len(k_cache) // 4)
    sampled_layers = list(range(0, len(k_cache), step))

    with torch.no_grad():
        for layer_idx in sampled_layers:
            k = k_cache[layer_idx].float()  # (B, H_kv, L, d_h)
            B, H_kv, _, d_h = k.shape
            # Repeat KV heads to match query heads if GQA
            H_q = q_question.shape[1]
            rep = H_q // H_kv
            if rep > 1:
                k = k.repeat_interleave(rep, dim=1)  # (B, H_q, L, d_h)

            # Dot-product between question queries and prompt keys: (B, H_q, L_q, L)
            scores = torch.matmul(q_question.float(), k.transpose(-2, -1)) / math.sqrt(d_h)
            attn_probs = torch.softmax(scores, dim=-1)  # (B, H_q, L_q, L)

            # Sum across batch, heads, and question length
            layer_imp = attn_probs.sum(dim=(0, 1, 2))  # (L,)
            total_scores += layer_imp

    # Apply 1D max pooling of kernel size 5 to cluster tokens
    pooled_total = F.max_pool1d(total_scores.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
    return pooled_total


# ──────────────────────────────────────────────────────────────────────────────
# Policy 1: Sink + Recent FP16 Eviction
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_sink_recent(
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
    """Policy 1: Retains largest legal number of FP16 positions: 8 sink + 64 recent + fill budget with recent."""
    L = k_list[0].shape[2]
    budget_bytes = compute_matched_int4_bytes(L, b=4)
    n_retain = compute_fp16_retained_token_count(L, b=4)

    sink_count = min(KV_SINK_TOKENS, L)
    recent_count = min(KV_RECENT_TOKENS, L - sink_count)
    rem_budget = max(0, n_retain - (sink_count + recent_count))

    # Selected indices: sink tokens (0..sink_count-1) and most recent (L - recent_count - rem_budget .. L-1)
    sink_idx = torch.arange(0, sink_count, device=k_list[0].device)
    tail_start = max(sink_count, L - (recent_count + rem_budget))
    tail_idx = torch.arange(tail_start, L, device=k_list[0].device)
    keep_indices = torch.cat([sink_idx, tail_idx]).unique()

    k_out = [k[:, :, keep_indices, :] for k in k_list]
    v_out = [v[:, :, keep_indices, :] for v in v_list]
    actual_bytes = len(keep_indices) * (4 * len(k_list) * k_list[0].shape[1] * k_list[0].shape[3])

    return k_out, v_out, keep_indices, actual_bytes


# ──────────────────────────────────────────────────────────────────────────────
# Policy 2: Question-Aware FP16 Eviction (SnapKV)
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_question_aware(
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    q_question: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
    """Policy 2: 8 sink + 64 recent + top question-aware importance positions up to matched budget."""
    L = k_list[0].shape[2]
    n_retain = compute_fp16_retained_token_count(L, b=4)

    sink_count = min(KV_SINK_TOKENS, L)
    recent_count = min(KV_RECENT_TOKENS, L - sink_count)
    num_middle_select = max(0, n_retain - (sink_count + recent_count))

    # Mandatory safeguard indices
    sink_idx = torch.arange(0, sink_count, device=k_list[0].device)
    recent_idx = torch.arange(L - recent_count, L, device=k_list[0].device)

    # Rank middle candidate positions
    middle_start = sink_count
    middle_end = max(sink_count, L - recent_count)
    
    if num_middle_select > 0 and middle_end > middle_start:
        importance = compute_question_importance_scores(k_list, q_question)
        middle_scores = importance[middle_start:middle_end]
        k_top = min(num_middle_select, len(middle_scores))
        _, top_rel_idx = torch.topk(middle_scores, k=k_top)
        middle_idx = middle_start + top_rel_idx
    else:
        middle_idx = torch.empty(0, dtype=torch.long, device=k_list[0].device)

    keep_indices = torch.cat([sink_idx, middle_idx, recent_idx]).unique()
    keep_indices, _ = torch.sort(keep_indices)  # Maintain chronological order

    k_out = [k[:, :, keep_indices, :] for k in k_list]
    v_out = [v[:, :, keep_indices, :] for v in v_list]
    actual_bytes = len(keep_indices) * (4 * len(k_list) * k_list[0].shape[1] * k_list[0].shape[3])

    return k_out, v_out, keep_indices, actual_bytes


# ──────────────────────────────────────────────────────────────────────────────
# Policy 3: Token-wise INT4 Quantization
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_token_wise_int4(
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
    """Policy 3: Group-64 INT4 quantization along hidden dimension for both K and V."""
    L = k_list[0].shape[2]
    k_out = []
    v_out = []
    total_bytes = 0

    for k, v in zip(k_list, v_list):
        k_deq, v_deq, layer_bytes = quantize_token_wise_kv(k, v, bits=4, group_size=KV_GROUP_SIZE)
        k_out.append(k_deq)
        v_out.append(v_deq)
        total_bytes += layer_bytes

    keep_indices = torch.arange(L, device=k_list[0].device)
    return k_out, v_out, keep_indices, total_bytes


# ──────────────────────────────────────────────────────────────────────────────
# Policy 4: Channel-wise-K / Token-wise-V INT4 Quantization (KIVI)
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_channel_k_token_v_int4(
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
    """Policy 4: Group-64 INT4 along tokens for K, and along hidden channels for V,
    with 8 sink tokens and 64 recent streaming buffer tokens preserved in FP16 (KIVI specification).
    """
    L = k_list[0].shape[2]
    device = k_list[0].device
    k_out = []
    v_out = []
    total_bytes = 0

    sink_count = min(KV_SINK_TOKENS, L)
    recent_count = min(KV_RECENT_TOKENS, max(0, L - sink_count))
    mid_start = sink_count
    mid_end = max(sink_count, L - recent_count)

    for k, v in zip(k_list, v_list):
        k_layer = k.clone()
        v_layer = v.clone()
        layer_bytes = 0

        # Account for FP16 sink + recent streaming buffer
        num_fp16 = sink_count + recent_count
        layer_bytes += num_fp16 * (k.shape[1] * k.shape[3] * 4)

        if mid_end > mid_start:
            k_mid = k[:, :, mid_start:mid_end, :]
            v_mid = v[:, :, mid_start:mid_end, :]
            k_mid_deq, v_mid_deq, mid_bytes = quantize_channel_k_token_v(k_mid, v_mid, bits=4, group_size=KV_GROUP_SIZE)
            k_layer[:, :, mid_start:mid_end, :] = k_mid_deq
            v_layer[:, :, mid_start:mid_end, :] = v_mid_deq
            layer_bytes += mid_bytes

        k_out.append(k_layer)
        v_out.append(v_layer)
        total_bytes += layer_bytes

    keep_indices = torch.arange(L, device=device)
    return k_out, v_out, keep_indices, total_bytes


# ──────────────────────────────────────────────────────────────────────────────
# Policy 5: Hybrid Precision & Token Selection (FP16 + INT2)
# ──────────────────────────────────────────────────────────────────────────────

def apply_policy_hybrid(
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    q_question: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
    """Policy 5: Retains 8 sink + 64 recent + top question-aware in FP16; compresses remainder to INT2."""
    L = k_list[0].shape[2]
    n_fp16, n_int2 = compute_hybrid_fp16_token_count(L, group_size=KV_GROUP_SIZE)

    sink_count = min(KV_SINK_TOKENS, L)
    recent_count = min(KV_RECENT_TOKENS, L - sink_count)
    num_middle_select = max(0, n_fp16 - (sink_count + recent_count))

    sink_idx = torch.arange(0, sink_count, device=k_list[0].device)
    recent_idx = torch.arange(L - recent_count, L, device=k_list[0].device)
    middle_start = sink_count
    middle_end = max(sink_count, L - recent_count)

    if num_middle_select > 0 and middle_end > middle_start:
        importance = compute_question_importance_scores(k_list, q_question)
        middle_scores = importance[middle_start:middle_end]
        k_top = min(num_middle_select, len(middle_scores))
        _, top_rel_idx = torch.topk(middle_scores, k=k_top)
        middle_fp16_idx = middle_start + top_rel_idx
    else:
        middle_fp16_idx = torch.empty(0, dtype=torch.long, device=k_list[0].device)

    fp16_indices = torch.cat([sink_idx, middle_fp16_idx, recent_idx]).unique()
    fp16_mask = torch.zeros(L, dtype=torch.bool, device=k_list[0].device)
    fp16_mask[fp16_indices] = True
    int2_indices = torch.arange(L, device=k_list[0].device)[~fp16_mask]

    # Quantize INT2 tokens (Channel-K / Token-V)
    k_out = []
    v_out = []
    total_bytes = 0
    bytes_per_fp16_token = 4 * len(k_list) * k_list[0].shape[1] * k_list[0].shape[3]
    total_bytes += len(fp16_indices) * bytes_per_fp16_token

    for k, v in zip(k_list, v_list):
        k_new = k.clone()
        v_new = v.clone()

        if len(int2_indices) > 0:
            k_int2 = k[:, :, int2_indices, :]
            v_int2 = v[:, :, int2_indices, :]
            k_deq, v_deq, layer_int2_bytes = quantize_channel_k_token_v(k_int2, v_int2, bits=2, group_size=KV_GROUP_SIZE)
            k_new[:, :, int2_indices, :] = k_deq
            v_new[:, :, int2_indices, :] = v_deq
            total_bytes += layer_int2_bytes

        k_out.append(k_new)
        v_out.append(v_new)

    keep_indices = torch.arange(L, device=k_list[0].device)
    return k_out, v_out, keep_indices, total_bytes
