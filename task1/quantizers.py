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

    # Integer zero-point z = round(-w_min / s). It is NOT clamped to [0, 2^b - 1]:
    # for single-signed groups (w_min > 0 or w_max < 0) clamping z destroys the affine
    # range and saturates most of the group. z is serialized as FP16 metadata.
    zeros = torch.round(-w_min / scales)

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
    """Q3: the Q2 AWQ quantizer with per-projection bit widths b_ell in {2, 3, 4, 8}."""
    errors: Dict[str, float] = {}
    with torch.no_grad():
        for name, module in get_eligible_linear_modules(model, eligible_projections).items():
            w_hat, err, _ = awq_quantize_layer(
                module.weight.data, activation_stats.sample_inputs.get(name), activation_stats.get_mean_abs(name),
                allocation_map[name], group_size, candidate_alphas,
            )
            errors[name] = err
            module.weight.data.copy_(w_hat)
    return errors


def verify_awq_rescaling_preserves_outputs(
    model: nn.Module,
    activation_stats: ActivationStats,
    alpha: float = 0.5,
    eligible_projections: tuple[str, ...] = (),
) -> Dict[str, float]:
    """Check that full-precision AWQ rescaling is function-preserving before rounding.

    Computes max |X W^T - (X diag(s)^-1)(W diag(s))^T| / max|X W^T| per projection in FP32
    on cached calibration inputs. Values should be at FP32 round-off level (~1e-6).
    """
    out: Dict[str, float] = {}
    with torch.no_grad():
        for name, module in get_eligible_linear_modules(model, eligible_projections).items():
            x = activation_stats.sample_inputs.get(name)
            if x is None:
                continue
            w = module.weight.data.float()
            x = x.to(w.device).float()
            s_x = torch.clamp(activation_stats.get_mean_abs(name).to(w.device).float(), min=1e-6) ** alpha
            s_x = s_x / s_x.mean()
            y = x @ w.t()
            y_scaled = (x / s_x.unsqueeze(0)) @ (w * s_x.unsqueeze(0)).t()
            out[name] = float(((y - y_scaled).abs().max() / y.abs().max().clamp_min(1e-12)).item())
    return out


def awq_quantize_layer(
    w: torch.Tensor,
    sample_x: Optional[torch.Tensor],
    act_mean: torch.Tensor,
    bit_width: int,
    group_size: int = 128,
    candidate_alphas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> Tuple[torch.Tensor, float, float]:
    """The Q2/Q3 AWQ-style quantizer for one projection.

    Grid-searches s = mean|X|^alpha (normalized to mean 1), quantizes W diag(s) with the Q1
    quantizer, compensates with diag(s)^-1 and keeps the alpha with the lowest calibration
    output MSE. Returns (W_hat, absolute error = mean over calibration tokens of the squared
    output-vector error ||y - y_hat||^2, relative output error ||Y - Y_hat||^2 / ||Y||^2).
    """
    act_mean = torch.clamp(act_mean.to(w.device).float(), min=1e-6)
    x = sample_x.to(w.device).float() if sample_x is not None else None
    y = x @ w.float().t() if x is not None else None
    best = (None, float("inf"), float("inf"))
    for alpha in candidate_alphas:
        s_x = torch.ones_like(act_mean) if alpha == 0.0 else act_mean ** alpha / torch.mean(act_mean ** alpha)
        w_hat_s, _, _ = quantize_weight_asymmetric(w.float() * s_x.unsqueeze(0), bit_width=bit_width, group_size=group_size)
        w_hat = w_hat_s / s_x.unsqueeze(0)
        if y is not None:
            diff = (y - x @ w_hat.t()) ** 2
            err = float(diff.sum() / x.shape[0])
            rel = float(diff.sum() / (y ** 2).sum().clamp_min(1e-12))
        else:
            err = float(((w.float() - w_hat) ** 2).sum())
            rel = err / float((w.float() ** 2).sum().clamp_min(1e-12))
        if err < best[1]:
            best = (w_hat, err, rel)
    return best[0].to(w.dtype), best[1], best[2]
