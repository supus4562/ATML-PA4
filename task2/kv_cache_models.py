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


def compute_fp16_retained_token_count(L: int, b: int = 4) -> int:
    """Compute the maximum legal number of FP16 tokens that fit strictly within the matched INT4 budget.
    
    Since each FP16 token consumes 4 * n_layers * H_kv * d_h bytes (28,672 bytes):
        N_retain = floor(M_b(L) / (4 * n_layers * H_kv * d_h))
    """
    budget_bytes = compute_matched_int4_bytes(L, b=b)
    bytes_per_fp16_token = 4 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM
    return budget_bytes // bytes_per_fp16_token


def compute_hybrid_fp16_token_count(L: int, group_size: int = KV_GROUP_SIZE) -> Tuple[int, int]:
    """Calculate the budget allocation for Policy 5 (Hybrid FP16 + INT2):
    
    Under the matched INT4 budget M_4(L):
    - Let N_fp16 be the number of positions retained in full FP16.
    - Let N_int2 = L - N_fp16 be stored in 2-bit (INT2 channel-K / token-V) in complete groups of 64.
    - Bytes for INT2 group: (64 * 2) / 8 + 4 = 16 + 4 = 20 bytes per channel per group.
    
    Returns:
        (N_fp16, N_int2)
    """
    total_budget = compute_matched_int4_bytes(L, b=4)
    total_channels = 2 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM
    bytes_per_fp16_token = 4 * KV_NUM_LAYERS * KV_NUM_KV_HEADS * KV_HEAD_DIM

    # Mandatory safeguards: 8 sink + 64 recent = 72 tokens
    min_fp16 = KV_SINK_TOKENS + KV_RECENT_TOKENS

    # Search for largest N_fp16 >= min_fp16 such that N_int2 forms complete 64-token groups
    # and total serialized bytes <= total_budget
    best_n_fp16 = min_fp16
    for n_fp16 in range(min_fp16, L):
        n_int2 = L - n_fp16
        if n_int2 % group_size != 0:
            continue
        # Bytes consumed by N_fp16 tokens:
        fp16_cost = n_fp16 * bytes_per_fp16_token
        # Bytes consumed by N_int2 tokens:
        int2_groups = n_int2 // group_size
        int2_cost = total_channels * (int2_groups * 20)  # 20 bytes per group
        total_cost = fp16_cost + int2_cost
        if total_cost <= total_budget:
            best_n_fp16 = n_fp16

    return best_n_fp16, L - best_n_fp16
