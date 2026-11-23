"""task2/kv_quantizers.py — Vectorized GPU quantizers for KV cache: Token-wise and Channel-wise-K / Token-wise-V.
"""

from __future__ import annotations

from typing import Tuple
import torch

from task2.config import KV_GROUP_SIZE


def quantize_affine_group(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = KV_GROUP_SIZE,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Asymmetric affine quantization along a specified dimension in groups of group_size.
    
    Args:
        tensor: Floating-point tensor on GPU
        bits: Target integer bit-width (4 or 2)
        group_size: Quantization group size (64)
        dim: Dimension to group along
        
    Returns:
        dequant: Dequantized tensor in original dtype
        scales: FP16 scale tensor
        zeros: FP16 zero-point tensor
        total_bytes: Exact serialized payload and metadata bytes
    """
    orig_shape = tensor.shape
    orig_dtype = tensor.dtype

    # Permute target dimension to the last axis
    if dim != -1 and dim != tensor.ndim - 1:
        perm = list(range(tensor.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        t = tensor.permute(perm).contiguous()
    else:
        perm = None
        t = tensor.contiguous()

    L = t.shape[-1]
    num_full_groups = L // group_size
    rem = L % group_size

    # Split into full groups and trailing residual
    full_len = num_full_groups * group_size
    t_full = t[..., :full_len]
    t_rem = t[..., full_len:]

    q_max = (1 << bits) - 1

    # Reshape full groups to (..., num_groups, group_size)
    reshaped = t_full.view(*t_full.shape[:-1], num_full_groups, group_size)

    # Compute min and max along group axis
    w_min = torch.amin(reshaped, dim=-1, keepdim=True)
    w_max = torch.amax(reshaped, dim=-1, keepdim=True)

    # Compute scale and zero point
    scale = ((w_max - w_min) / float(q_max)).clamp_min(1e-8)
    zero = torch.round(-w_min / scale).clamp(0, q_max)

    # Quantize and dequantize
    q_int = torch.clamp(torch.round(reshaped / scale) + zero, 0, q_max)
    dequant_full = (q_int - zero) * scale
    dequant_full = dequant_full.view_as(t_full).to(orig_dtype)

    # Trailing tokens remain full precision FP16
    if rem > 0:
        dequant = torch.cat([dequant_full, t_rem], dim=-1)
    else:
        dequant = dequant_full

    # Restore original dimension order
    if perm is not None:
        dequant = dequant.permute(perm).contiguous()

    # Exact serialized byte accounting:
    # Payload: num_groups * (group_size * bits / 8)
    # Metadata: num_groups * 4 bytes (FP16 scale + FP16 zero)
    # Trailing: rem * 2 bytes (FP16)
    total_elements = tensor.numel()
    num_total_groups = (total_elements // L) * num_full_groups
    num_total_rem = (total_elements // L) * rem

    payload_bytes = num_total_groups * ((group_size * bits) // 8)
    meta_bytes = num_total_groups * 4
    rem_bytes = num_total_rem * 2
    total_bytes = payload_bytes + meta_bytes + rem_bytes

    return dequant, scale.to(torch.float16), zero.to(torch.float16), total_bytes


def quantize_token_wise_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: int = 4,
    group_size: int = KV_GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Policy 3: Token-wise quantization of both Keys and Values along the hidden dimension (group 64)."""
    # Shapes: (B, H, L, d_h). Grouping along dim -1 (d_h)
    k_deq, _, _, k_bytes = quantize_affine_group(k, bits=bits, group_size=group_size, dim=-1)
    v_deq, _, _, v_bytes = quantize_affine_group(v, bits=bits, group_size=group_size, dim=-1)
    return k_deq, v_deq, k_bytes + v_bytes


def quantize_channel_k_token_v(
    k: torch.Tensor,
    v: torch.Tensor,
    bits: int = 4,
    group_size: int = KV_GROUP_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Policy 4: Channel-wise-K (group 64 along tokens) and Token-wise-V (group 64 along hidden channels)."""
    # K shape: (B, H, L, d_h). Grouping along dim 2 (token sequence length L)
    k_deq, _, _, k_bytes = quantize_affine_group(k, bits=bits, group_size=group_size, dim=2)
    # V shape: (B, H, L, d_h). Grouping along dim 3 (hidden dim d_h)
    v_deq, _, _, v_bytes = quantize_affine_group(v, bits=bits, group_size=group_size, dim=-1)
    return k_deq, v_deq, k_bytes + v_bytes
