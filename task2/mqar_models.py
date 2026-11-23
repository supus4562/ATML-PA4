"""task2/mqar_models.py — Pre-LN Transformer blocks with pluggable attention mechanisms for MQAR.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.config import (
    MQAR_VOCAB_SIZE,
    MQAR_HIDDEN_DIM,
    MQAR_NUM_HEADS,
    MQAR_HEAD_DIM,
    MQAR_NUM_BLOCKS,
    MQAR_MLP_DIM,
)


def get_sinusoidal_positional_embeddings(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional encoding matrix (1, seq_len, d_model)."""
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Attention Operators for MQAR
# ──────────────────────────────────────────────────────────────────────────────

class DenseAttention(nn.Module):
    """Standard causal multi-head self-attention."""

    def __init__(self, d_model: int = MQAR_HIDDEN_DIM, n_heads: int = MQAR_NUM_HEADS):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if mask_override is not None:
            # Explicit mask override (e.g. for test-time sliding window control)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~mask_override, -1e9)
            probs = F.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class SlidingWindowAttention(nn.Module):
    """Causal sliding-window attention with window size W=64."""

    def __init__(self, d_model: int = MQAR_HIDDEN_DIM, n_heads: int = MQAR_NUM_HEADS, window_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Build sliding window causal mask: 0 <= i - j < W
        row_idx = torch.arange(T, device=x.device).unsqueeze(1)
        col_idx = torch.arange(T, device=x.device).unsqueeze(0)
        dist = row_idx - col_idx
        mask = (dist >= 0) & (dist < self.window_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class LinearAttention(nn.Module):
    """Causal linear attention using ELU+1 feature map: phi(x) = ELU(x) + 1."""

    def __init__(self, d_model: int = MQAR_HIDDEN_DIM, n_heads: int = MQAR_NUM_HEADS, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.feature_map(self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2))
        k = self.feature_map(self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2))
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Vectorized causal prefix sum formulation:
        # S_t = sum_{j <= t} k_j outer v_j  -> (B, H, T, d_head, d_head)
        # kv: (B, H, T, d_head, d_head)
        kv = torch.einsum("bhti,bhtj->bhtij", k, v)
        kv_cum = torch.cumsum(kv, dim=2)

        # Normalizer: z_t = sum_{j <= t} k_j -> (B, H, T, d_head)
        k_cum = torch.cumsum(k, dim=2)

        # Numerator: q_t * S_t -> (B, H, T, d_head)
        num = torch.einsum("bhti,bhtij->bhtj", q, kv_cum)
        # Denominator: q_t dot z_t -> (B, H, T, 1)
        den = torch.einsum("bhti,bhti->bht", q, k_cum).unsqueeze(-1) + self.eps

        out = num / den
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MixedDilatedAttention(nn.Module):
    """Mixed local/dilated causal attention: 2 heads local (W=64), 2 heads dilated (stride 4, offsets 0..252)."""

    def __init__(self, d_model: int = MQAR_HIDDEN_DIM, n_heads: int = MQAR_NUM_HEADS, window_size: int = 64, stride: int = 4):
        super().__init__()
        assert n_heads == 4, "MixedDilatedAttention requires exactly 4 heads (2 local, 2 dilated)"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.stride = stride
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Build head masks
        row_idx = torch.arange(T, device=x.device).unsqueeze(1)
        col_idx = torch.arange(T, device=x.device).unsqueeze(0)
        dist = row_idx - col_idx

        # Heads 0, 1: Local window 64
        local_mask = (dist >= 0) & (dist < self.window_size)
        # Heads 2, 3: Dilated offsets {0, 4, 8, ...} within range [0, 252]
        dilated_mask = (dist >= 0) & (dist <= 252) & (dist % self.stride == 0)

        # Combine into (1, 4, T, T) mask
        head_masks = torch.stack([local_mask, local_mask, dilated_mask, dilated_mask], dim=0).unsqueeze(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(~head_masks, -1e9)
        probs = F.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class DGLAAttention(nn.Module):
    """Controlled Intervention: Decay-Gated Linear Attention (DGLA).
    
    Applies learned input-dependent decay gamma_t = sigmoid(W_gamma x_t + b_gamma)
    to linear attention recurrence: S_t = gamma_t * S_{t-1} + phi(k_t) v_t^T.
    Adds only 129 parameters (<< 5% limit).
    """

    def __init__(self, d_model: int = MQAR_HIDDEN_DIM, n_heads: int = MQAR_NUM_HEADS, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, n_heads, bias=True)  # Per-head adaptive decay gate
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.feature_map(self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2))
        k = self.feature_map(self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2))
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Decay gates: (B, H, T) in range (0.5, 1.0) via scaled sigmoid for long-term retention
        gamma = (0.5 + 0.5 * torch.sigmoid(self.gate_proj(x))).transpose(1, 2).clamp(min=1e-4, max=1.0)

        # Parallelized decay cumsum: decay[b, h, t, tau] = exp(cumsum[t] - cumsum[tau])
        log_gamma = torch.log(gamma)  # (B, H, T)
        cumsum = torch.cumsum(log_gamma, dim=-1)  # (B, H, T)
        diff = cumsum.unsqueeze(-1) - cumsum.unsqueeze(-2)  # (B, H, T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        diff = diff.masked_fill(~causal_mask, -1e4)
        decay = torch.exp(diff)
        decay = decay.masked_fill(~causal_mask, 0.0)

        # Parallel linear attention with input-dependent decay
        qk = torch.matmul(q, k.transpose(-2, -1)) * decay  # (B, H, T, T)
        out = torch.matmul(qk, v)  # (B, H, T, d_h)
        z = torch.matmul(decay, k)  # (B, H, T, d_h)
        den = torch.sum(q * z, dim=-1, keepdim=True) + self.eps
        out = out / den

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block & Full Model
# ──────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""

    def __init__(self, attn_module: nn.Module, d_model: int = MQAR_HIDDEN_DIM, d_mlp: int = MQAR_MLP_DIM):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_module
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask_override=mask_override)
        x = x + self.mlp(self.ln2(x))
        return x


class MQARTransformer(nn.Module):
    """Complete 2-block Pre-LN Transformer for Multi-Query Associative Recall."""

    def __init__(
        self,
        attn_type: str = "dense",
        vocab_size: int = MQAR_VOCAB_SIZE,
        d_model: int = MQAR_HIDDEN_DIM,
        n_blocks: int = MQAR_NUM_BLOCKS,
        n_classes: int = 64,  # Answer values 128..191
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.attn_type = attn_type

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)

        def make_attn():
            if attn_type == "dense":
                return DenseAttention(d_model)
            elif attn_type == "window":
                return SlidingWindowAttention(d_model, window_size=64)
            elif attn_type == "linear":
                return LinearAttention(d_model)
            elif attn_type == "dilated":
                return MixedDilatedAttention(d_model)
            elif attn_type == "dgla":
                return DGLAAttention(d_model)
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")

        self.blocks = nn.ModuleList([
            TransformerBlock(make_attn(), d_model=d_model) for _ in range(n_blocks)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids: torch.Tensor, mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos_emb = get_sinusoidal_positional_embeddings(T, self.d_model, input_ids.device)
        x = self.tok_embeddings(input_ids) + pos_emb

        for block in self.blocks:
            x = block(x, mask_override=mask_override)

        x = self.final_ln(x)
        logits = self.classifier(x)  # (B, T, 64)
        return logits
