"""task2/kv_cache_models.py — KV cache byte budget math, RoPE position preservation, and cache editing abstractions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from task2.config import (
    KV_NUM_LAYERS,
    KV_NUM_KV_HEADS,
    KV_HEAD_DIM,
    KV_GROUP_SIZE,
    KV_SINK_TOKENS,
    KV_RECENT_TOKENS,
)


def compute_matched_int4_bytes(L: int, b: int = 4, group_size: int = KV_GROUP_SIZE) -> int:
    """Calculate exact serialized bytes for uniform INT4 channel-wise-K / token-wise-V reference cache.
    
    Formula from PA4 specification:
        M_b(L) = 2 * n_layers * H_kv * d_h * [ floor(L / 64) * (64 * b / 8 + 4) + 2 * (L mod 64) ]
    """
    total_channels = 2 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM  # 14,336 channels
    num_full_groups = L // group_size
    rem_tokens = L % group_size
    bytes_per_group = (group_size * b) // 8 + 4  # (64 * 4) / 8 + 4 = 36 bytes for INT4
    
    bytes_per_channel = num_full_groups * bytes_per_group + 2 * rem_tokens
    return total_channels * bytes_per_channel


def compute_uncompressed_fp16_bytes(L: int) -> int:
    """Exact uncompressed FP16 cache bytes: M_16(L) = 4 * n_layers * H_kv * d_h * L."""
    return 4 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM * L


def bytes_per_fp16_token() -> int:
    """K and V for one position across all layers and KV heads in FP16: 4 * n_l * H_kv * d_h."""
    return 4 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM


def position_mask_bytes(L: int) -> int:
    """One shared 1-bit-per-position retention mask (selection shared across layers/heads)."""
    return math.ceil(L / 8)


def compute_fp16_retained_token_count(L: int, b: int = 4, metadata_bytes: int = 0) -> int:
    """Largest number of FP16 positions whose bytes (+ position metadata) fit the INT4 budget."""
    budget_bytes = compute_matched_int4_bytes(L, b=b)
    return max(0, (budget_bytes - metadata_bytes) // bytes_per_fp16_token())


def compute_hybrid_fp16_token_count(L: int, group_size: int = KV_GROUP_SIZE) -> Tuple[int, int]:
    """Largest N_fp16 >= 72 such that N_int2 = L - N_fp16 is a multiple of 64 and
    FP16 payload + INT2 channel-K/token-V groups (20 bytes per 64-element group incl. FP16
    scale+offset) + shared position mask fit within the matched INT4 budget M_4(L).
    """
    total_budget = compute_matched_int4_bytes(L, b=4)
    total_channels = 2 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM
    min_fp16 = min(L, KV_SINK_TOKENS + KV_RECENT_TOKENS)
    best = None
    for n_fp16 in range(min_fp16, L + 1):
        n_int2 = L - n_fp16
        if n_int2 % group_size != 0:
            continue
        cost = (
            n_fp16 * bytes_per_fp16_token()
            + total_channels * (n_int2 // group_size) * ((group_size * 2) // 8 + 4)
            + position_mask_bytes(L)
        )
        if cost <= total_budget:
            best = n_fp16
    if best is None:
        raise ValueError(f"No legal hybrid allocation for L={L}")
    return best, L - best
