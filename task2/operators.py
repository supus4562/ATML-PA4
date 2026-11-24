"""task2/operators.py — Exact attention implementations: Dense Causal, Online-Softmax Tiling, and PyTorch SDPA.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dense Causal Attention (Numerical Reference)
# ──────────────────────────────────────────────────────────────────────────────

def dense_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Explicit dense causal attention with full L x L score matrix.
    
    Args:
        q: (B, H, L, d) query tensor
        k: (B, H, L, d) key tensor
        v: (B, H, L, d) value tensor
        scale: 1 / sqrt(d) scaling factor (default: 1.0 / sqrt(head_dim))
        
    Returns:
        out: (B, H, L, d) attention output tensor
    """
    B, H, L, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Compute raw dot-product scores (B, H, L, L)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Create causal mask (upper triangular strictly > 0 is masked)
    causal_mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)

    # Mask future positions with numerical minimum
    mask_val = torch.finfo(scores.dtype).min if scores.dtype != torch.float16 else -65504.0
    scores = scores.masked_fill(causal_mask, mask_val)

    # Numerically stable softmax: subtract row max
    max_scores = torch.amax(scores, dim=-1, keepdim=True)
    # Clamp to avoid NaN if entire row is masked (only position 0 in non-causal edge cases)
    max_scores = torch.nan_to_num(max_scores, nan=0.0)
    exp_scores = torch.exp(scores - max_scores)
    
    # Re-zero masked entries to ensure strict causality and prevent underflow bias
    exp_scores = exp_scores.masked_fill(causal_mask, 0.0)
    sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True).clamp_min(1e-12)
    probs = exp_scores / sum_exp

    # Multiply by values
    out = torch.matmul(probs, v)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tiled Online-Softmax Causal Attention
# ──────────────────────────────────────────────────────────────────────────────

def tiled_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size_r: int = 128,
    block_size_c: int = 128,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Exact blockwise causal attention using online-softmax updates without materializing L x L matrix.
    
    Implements the online-softmax recurrence:
        m' = max(m, max(s))
        l' = exp(m - m') * l + sum(exp(s - m'))
        o' = exp(m - m') * o + exp(s - m') * V
        A  = o' / l'
        
    Args:
        q: (B, H, L, d) query tensor
        k: (B, H, L, d) key tensor
        v: (B, H, L, d) value tensor
        block_size_r: Tile size for query rows
        block_size_c: Tile size for key/value columns
        scale: Scaling factor (default: 1.0 / sqrt(head_dim))
        
    Returns:
        out: (B, H, L, d) attention output tensor
    """
    B, H, L, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    out = torch.empty_like(q)
    neg_inf = -65504.0 if q.dtype == torch.float16 else -1e9

    num_blocks_r = (L + block_size_r - 1) // block_size_r
    num_blocks_c = (L + block_size_c - 1) // block_size_c

    for i in range(num_blocks_r):
        r_start = i * block_size_r
        r_end = min(r_start + block_size_r, L)
        q_tile = q[:, :, r_start:r_end, :]  # (B, H, Br, d)
        cur_br = r_end - r_start

        # Running online statistics for this row block
        # m: (B, H, Br, 1), l: (B, H, Br, 1), o: (B, H, Br, d)
        m = torch.full((B, H, cur_br, 1), neg_inf, device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, cur_br, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, cur_br, d), device=q.device, dtype=torch.float32)

        for j in range(num_blocks_c):
            c_start = j * block_size_c
            c_end = min(c_start + block_size_c, L)

            # Causal skipping: if entire K/V tile is in the future, skip!
            if c_start >= r_end:
                break

            k_tile = k[:, :, c_start:c_end, :]  # (B, H, Bc, d)
            v_tile = v[:, :, c_start:c_end, :]  # (B, H, Bc, d)

            # Compute tile scores (B, H, Br, Bc)
            s_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale

            # If tile overlaps the causal diagonal, mask future positions
            if c_end > r_start:
                # Row indices in sequence: r_start .. r_end - 1
                # Col indices in sequence: c_start .. c_end - 1
                r_idx = torch.arange(r_start, r_end, device=q.device).unsqueeze(1)
                c_idx = torch.arange(c_start, c_end, device=q.device).unsqueeze(0)
                mask = r_idx < c_idx  # True for future positions
                s_tile = s_tile.masked_fill(mask.unsqueeze(0).unsqueeze(0), neg_inf)

            # Online-softmax updates in FP32 accumulator precision for numerical stability
            s_tile_f32 = s_tile.to(torch.float32)
            m_tile = torch.amax(s_tile_f32, dim=-1, keepdim=True)
            m_new = torch.maximum(m, m_tile)

            # Factors for rescaling previous accumulator and current tile
            alpha = torch.exp(m - m_new)
            p_tile = torch.exp(s_tile_f32 - m_new)

            # Re-zero strictly masked entries where m_new == neg_inf
            p_tile = torch.nan_to_num(p_tile, nan=0.0)

            l = alpha * l + torch.sum(p_tile, dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p_tile, v_tile.to(torch.float32))
            m = m_new

        # Normalize output tile: A_i = o / l
        out_tile = o / l.clamp_min(1e-12)
        out[:, :, r_start:r_end, :] = out_tile.to(q.dtype)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. PyTorch SDPA Wrapper with Backend Detection
# ──────────────────────────────────────────────────────────────────────────────

def get_sdpa_backend(device: str = "cuda") -> str:
    """Detect available and dispatched PyTorch SDPA backend."""
    if not torch.cuda.is_available() or device == "cpu":
        return "CPU Math"
    backends = []
    if torch.backends.cuda.flash_sdp_enabled():
        backends.append("FlashAttention-2")
    if torch.backends.cuda.mem_efficient_sdp_enabled():
        backends.append("MemoryEfficient")
    if torch.backends.cuda.math_sdp_enabled():
        backends.append("Math")
    return ", ".join(backends) if backends else "Unknown"


def probe_sdpa_backend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
    """Determine which SDPA backends accept these exact tensors and which one default
    dispatch selects. Each backend is forced alone via torch.nn.attention.sdpa_kernel; the
    default dispatcher picks the first supported backend in PyTorch's priority order
    (cuDNN, Flash, Efficient, Math in PyTorch >= 2.5; Flash, Efficient, Math earlier).
    """
    if not q.is_cuda:
        return {"supported": ["MATH"], "dispatched": "MATH"}
    from torch.nn.attention import SDPBackend, sdpa_kernel
    order = []
    for name in ("CUDNN_ATTENTION", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        if hasattr(SDPBackend, name):
            order.append((name, getattr(SDPBackend, name)))
    supported = []
    with torch.no_grad():
        for name, be in order:
            try:
                with sdpa_kernel([be]):
                    F.scaled_dot_product_attention(q[:1, :1, :128], k[:1, :1, :128], v[:1, :1, :128], is_causal=True)
                supported.append(name)
            except RuntimeError:
                pass
    # Default dispatch priority (matches torch's sdp_utils ordering).
    priority = [n for n, _ in order]
    dispatched = next((n for n in priority if n in supported), "UNKNOWN")
    if "CUDNN_ATTENTION" in supported and torch.__version__ < "2.5":
        dispatched = next((n for n in priority[1:] if n in supported), dispatched)
    return {"supported": supported, "dispatched": dispatched, "torch": torch.__version__}


def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, str]:
    """PyTorch native scaled_dot_product_attention with causal flag and backend tracking."""
    backend_info = get_sdpa_backend(str(q.device))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out, backend_info
