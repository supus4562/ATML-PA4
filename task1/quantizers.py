"""task1/quantizers.py — Vectorized GPU quantizers for Q1 (Uniform PTQ), Q2 (AWQ), and Q3 (Mixed Precision)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from task1.calibration import ActivationStats
from task1.pruners import get_eligible_linear_modules


def quantize_weight_asymmetric(
    w: torch.Tensor,
    bit_width: int,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize and dequantize a 2D weight tensor using asymmetric uniform group-wise PTQ.

    Args:
        w: Weight tensor of shape [out_features, in_features]
        bit_width: Number of bits (e.g. 8, 4, 3, 2)
        group_size: Number of elements per quantization group along dim=1

    Returns:
        w_hat: Dequantized weight tensor of shape [out_features, in_features] in same dtype as w
        scales: Scale tensor of shape [out_features, num_groups]
        zeros: Zero-point tensor of shape [out_features, num_groups]
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, f"in_features {in_features} must be divisible by group_size {group_size}"

    num_groups = in_features // group_size
    w_grouped = w.view(out_features, num_groups, group_size).float()

    w_min = torch.min(w_grouped, dim=-1, keepdim=True).values
    w_max = torch.max(w_grouped, dim=-1, keepdim=True).values

    max_int = (1 << bit_width) - 1
    scales = (w_max - w_min) / max_int
    scales = torch.clamp(scales, min=1e-8)

    zeros = torch.round(-w_min / scales)
    zeros = torch.clamp(zeros, 0, max_int)

    q = torch.clamp(torch.round(w_grouped / scales) + zeros, 0, max_int)
    w_hat_grouped = (q - zeros) * scales

    w_hat = w_hat_grouped.view(out_features, in_features).to(w.dtype)
    return w_hat, scales.squeeze(-1).to(torch.float16), zeros.squeeze(-1).to(torch.int32)


def quantize_uniform_q1(
    model: nn.Module,
    bit_width: int,
    group_size: int = 128,
    eligible_projections: tuple[str, ...] = (),
) -> Dict[str, float]:
    """Q1: Uniform asymmetric post-training weight quantization.

    Applies group-wise asymmetric quantization to all eligible linear layers.
    Returns relative reconstruction error per layer: ||W - W_hat||_F^2 / ||W||_F^2.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    errors: Dict[str, float] = {}

    print(f"\n[Q1 Uniform PTQ] Quantizing to {bit_width}-bit (group size {group_size}) across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            w_hat, _, _ = quantize_weight_asymmetric(w, bit_width=bit_width, group_size=group_size)

            # Compute relative Frobenius norm error
            err = torch.sum((w.float() - w_hat.float()) ** 2) / torch.clamp(torch.sum(w.float() ** 2), min=1e-8)
            errors[name] = float(err.item())

            # In-place replace weight with dequantized weight for downstream evaluation
            w.copy_(w_hat)

    return errors


def quantize_awq_q2(
    model: nn.Module,
    bit_width: int,
    activation_stats: ActivationStats,
    group_size: int = 128,
    candidate_alphas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    eligible_projections: tuple[str, ...] = (),
) -> Dict[str, float]:
    """Q2: Activation-Aware Weight Quantization (AWQ).

    Searches for optimal channel scaling factor s_X = s_a^alpha derived from calibration activations,
    scales weights prior to low-bit PTQ, and applies inverse compensation to preserve the linear function.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    errors: Dict[str, float] = {}

    print(f"\n[Q2 AWQ] Running activation-aware scaling + {bit_width}-bit PTQ across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            out_features, in_features = w.shape

            # Sampled input matrix for reconstruction loss check: [N, in_features]
            sample_x = activation_stats.sample_inputs.get(name)
            if sample_x is not None:
                sample_x = sample_x.to(w.device).float()
                # Target output before quantization
                y_target = sample_x @ w.t().float()
            else:
                y_target = None

            # Channel average absolute activation
            act_mean = activation_stats.get_mean_abs(name).to(w.device).float()
            act_mean = torch.clamp(act_mean, min=1e-6)

            best_err = float("inf")
            best_w_hat = None

            for alpha in candidate_alphas:
                if alpha == 0.0:
                    s_x = torch.ones_like(act_mean)
                else:
                    s_x = act_mean ** alpha
                    s_x = s_x / torch.mean(s_x)  # normalize around 1.0

                # Scale weight channels: W' = W * diag(s_x)
                w_scaled = w.float() * s_x.unsqueeze(0)

                # Quantize scaled weight
                w_hat_scaled, _, _ = quantize_weight_asymmetric(
                    w_scaled, bit_width=bit_width, group_size=group_size
                )

                # Inverse compensate: W_hat = W_hat' * diag(s_x)^(-1)
                w_hat = w_hat_scaled / torch.clamp(s_x.unsqueeze(0), min=1e-8)

                if sample_x is not None and y_target is not None:
                    # Evaluate layer output reconstruction error
                    y_pred = sample_x @ w_hat.t().float()
                    err = float(torch.mean((y_target - y_pred) ** 2).item())
                else:
                    # Fallback to weight reconstruction error
                    err = float(torch.sum((w.float() - w_hat.float()) ** 2).item())

                if err < best_err or best_w_hat is None:
                    best_err = err
                    best_w_hat = w_hat

            errors[name] = best_err
            module.weight.data.copy_(best_w_hat.to(w.dtype))

    return errors


def quantize_mixed_precision_q3(
    model: nn.Module,
    allocation_map: Dict[str, int],
    activation_stats: ActivationStats,
    group_size: int = 128,
    candidate_alphas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    eligible_projections: tuple[str, ...] = (),
) -> Dict[str, float]:
    """Q3: Mixed-precision AWQ quantization.

    `allocation_map` assigns a specific bit width b_ell in {2, 3, 4, 8} to each layer.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    errors: Dict[str, float] = {}

    print(f"\n[Q3 Mixed Precision] Applying adaptive bit widths across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            b_ell = allocation_map.get(name, 4)
            w = module.weight.data

            sample_x = activation_stats.sample_inputs.get(name)
            if sample_x is not None:
                sample_x = sample_x.to(w.device).float()
                y_target = sample_x @ w.t().float()
            else:
                y_target = None

            act_mean = activation_stats.get_mean_abs(name).to(w.device).float()
            act_mean = torch.clamp(act_mean, min=1e-6)

            best_err = float("inf")
            best_w_hat = None

            for alpha in candidate_alphas:
                s_x = torch.ones_like(act_mean) if alpha == 0.0 else (act_mean ** alpha) / torch.mean(act_mean ** alpha)
                w_scaled = w.float() * s_x.unsqueeze(0)
                w_hat_scaled, _, _ = quantize_weight_asymmetric(w_scaled, bit_width=b_ell, group_size=group_size)
                w_hat = w_hat_scaled / torch.clamp(s_x.unsqueeze(0), min=1e-8)

                if sample_x is not None and y_target is not None:
                    y_pred = sample_x @ w_hat.t().float()
                    err = float(torch.mean((y_target - y_pred) ** 2).item())
                else:
                    err = float(torch.sum((w.float() - w_hat.float()) ** 2).item())

                if err < best_err or best_w_hat is None:
                    best_err = err
                    best_w_hat = w_hat

            errors[name] = best_err
            module.weight.data.copy_(best_w_hat.to(w.dtype))

    return errors
