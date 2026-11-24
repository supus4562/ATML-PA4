"""task1/packed_storage.py — Exact mathematical and serialized byte storage accounting."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from task1.pruners import get_eligible_linear_modules


def compute_model_packed_storage(
    model: nn.Module,
    eligible_projections: tuple[str, ...],
    bit_allocations: Optional[Dict[str, int]] = None,
    uniform_bit_width: Optional[int] = None,
    sparsity_allocations: Optional[Dict[str, float]] = None,
    uniform_sparsity: Optional[float] = None,
    group_size: int = 128,
    awq_channel_scales: bool = False,
) -> Dict[str, Any]:
    """Compute exact serialized/packed storage in bytes, nonzeros, and average bits per parameter.

    Accounts for:
    - Quantized integer payloads
    - Quantization metadata per group (FP16 scale [2 bytes] + FP16 zero-point [2 bytes] = 4 bytes)
    - Uncompressed parameters (embeddings, norms, head) at 16 bits (2 bytes)
    - Tied parameters counted exactly once
    - Sparse format (1 mask bit per eligible weight + 16 bits per nonzero + byte alignment).
      Nonzeros are COUNTED from the current (pruned) weights, not taken from the nominal target.
    - AWQ: inverse channel scales are not folded into an adjacent layer, so when
      `awq_channel_scales=True` one FP16 scale per input channel per eligible projection
      (in_features * 2 bytes) is serialized and counted as metadata.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    eligible_param_count = sum(m.weight.numel() for m in eligible_modules.values())

    # Map tensor data pointers to avoid double-counting tied weights (e.g. embed_tokens and lm_head)
    seen_ptrs = set()
    uncompressed_bytes = 0
    uncompressed_param_count = 0

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)

        # Check if this parameter belongs to an eligible linear weight
        is_eligible = any(name == f"{m_name}.weight" for m_name in eligible_modules.keys())
        if not is_eligible:
            # 2 bytes per FP16 parameter
            bytes_param = param.numel() * 2
            uncompressed_bytes += bytes_param
            uncompressed_param_count += param.numel()

    # Calculate eligible compressed bytes
    eligible_payload_bytes = 0
    eligible_metadata_bytes = 0
    total_nonzeros = 0

    # Pruning branch
    if sparsity_allocations is not None or uniform_sparsity is not None:
        for name, module in eligible_modules.items():
            numel = module.weight.numel()
            nonzeros = int(torch.count_nonzero(module.weight.data).item())
            total_nonzeros += nonzeros

            # 1 mask bit per weight + 16 bits (2 bytes) per nonzero + byte alignment
            mask_bytes = math.ceil(numel / 8)
            nonzero_bytes = nonzeros * 2
            eligible_payload_bytes += (mask_bytes + nonzero_bytes)

        total_bytes = uncompressed_bytes + eligible_payload_bytes
        avg_bits_per_param = (total_bytes * 8) / (uncompressed_param_count + eligible_param_count)

        return {
            "mode": "sparse",
            "total_bytes": total_bytes,
            "total_megabytes": total_bytes / (1024 ** 2),
            "uncompressed_bytes": uncompressed_bytes,
            "eligible_payload_bytes": eligible_payload_bytes,
            "eligible_metadata_bytes": 0,
            "eligible_param_count": eligible_param_count,
            "uncompressed_param_count": uncompressed_param_count,
            "total_nonzeros": total_nonzeros,
            "effective_sparsity": 1.0 - (total_nonzeros / eligible_param_count),
            "avg_bits_per_param": avg_bits_per_param,
        }

    # Quantization branch
    for name, module in eligible_modules.items():
        numel = module.weight.numel()
        b = bit_allocations.get(name, uniform_bit_width) if bit_allocations else uniform_bit_width
        assert b is not None, "Must provide bit_allocations or uniform_bit_width"

        # Group count
        num_groups = math.ceil(numel / group_size)
        # Payload bytes: b bits per weight
        payload_b = math.ceil((numel * b) / 8)
        # Metadata bytes: 2 bytes scale + 2 bytes zero = 4 bytes per group
        metadata_b = num_groups * 4
        if awq_channel_scales:
            metadata_b += module.weight.shape[1] * 2

        eligible_payload_bytes += payload_b
        eligible_metadata_bytes += metadata_b
        total_nonzeros += numel

    total_bytes = uncompressed_bytes + eligible_payload_bytes + eligible_metadata_bytes
    avg_bits_per_param = (total_bytes * 8) / (uncompressed_param_count + eligible_param_count)

    return {
        "mode": "quantized",
        "total_bytes": total_bytes,
        "total_megabytes": total_bytes / (1024 ** 2),
        "uncompressed_bytes": uncompressed_bytes,
        "eligible_payload_bytes": eligible_payload_bytes,
        "eligible_metadata_bytes": eligible_metadata_bytes,
        "eligible_param_count": eligible_param_count,
        "uncompressed_param_count": uncompressed_param_count,
        "total_nonzeros": total_nonzeros,
        "effective_sparsity": 0.0,
        "avg_bits_per_param": avg_bits_per_param,
        "awq_channel_scale_bytes_included": awq_channel_scales,
    }
