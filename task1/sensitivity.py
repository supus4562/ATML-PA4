"""task1/sensitivity.py — Layer and projection sensitivity estimators for P3 and Q3."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from task1.calibration import ActivationStats
from task1.pruners import get_eligible_linear_modules
from task1.quantizers import awq_quantize_layer, quantize_weight_asymmetric


def compute_activation_outlier_metrics(
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
) -> Dict[str, Dict[str, float]]:
    """Compute channel outlier statistics (max/mean ratio and kurtosis proxy) per projection."""
    metrics: Dict[str, Dict[str, float]] = {}
    for name in activation_stats.sq_sum.keys():
        mean_abs = activation_stats.get_mean_abs(name)
        max_abs = activation_stats.get_max_abs(name)
        l2_norm = activation_stats.get_l2_norm(name)

        outlier_ratio = float((max_abs / torch.clamp(mean_abs, min=1e-6)).max().item())
        mean_ratio = float((max_abs / torch.clamp(mean_abs, min=1e-6)).mean().item())

        metrics[name] = {
            "max_to_mean_ratio": outlier_ratio,
            "avg_outlier_ratio": mean_ratio,
            "mean_l2_norm": float(l2_norm.mean().item()),
        }
    return metrics


def profile_layer_pruning_sensitivity(
    model: nn.Module,
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
    candidate_sparsities: tuple[float, ...] = (0.25, 0.50, 0.75),
    return_absolute: bool = False,
):
    """Measure layer-wise output reconstruction error ||Y - Y_pruned||_F^2 / ||Y||_F^2 under Wanda pruning.

    With `return_absolute=True` also returns the un-normalized mean squared output error
    ||Y - Y_pruned||_F^2 / N. The relative measure is dominated by massive-activation
    channels (e.g. early down_proj layers) and can rank those layers as "insensitive";
    the absolute measure is kept as an alternative design selected on the dev split.

    Evaluated directly on GPU using cached calibration sample activations.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    sensitivity_curve: Dict[str, Dict[float, float]] = {}
    absolute_curve: Dict[str, Dict[float, float]] = {}

    print(f"\n[Sensitivity] Profiling pruning sensitivity across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            out_features, in_features = w.shape
            sample_x = activation_stats.sample_inputs.get(name)

            if sample_x is None:
                # If no sampled activations, fallback to weight Frobenius sensitivity
                sample_x = torch.randn(256, in_features, device=w.device, dtype=w.dtype)

            sample_x = sample_x.to(w.device).float()
            y_target = sample_x @ w.t().float()
            y_norm = torch.clamp(torch.sum(y_target ** 2), min=1e-8)

            l2_norm = activation_stats.get_l2_norm(name).to(w.device).unsqueeze(0)
            scores = torch.abs(w) * l2_norm

            sensitivity_curve[name] = {}
            absolute_curve[name] = {}
            for s in candidate_sparsities:
                k_prune = int(round(s * in_features))
                if k_prune <= 0:
                    sensitivity_curve[name][s] = 0.0
                    absolute_curve[name][s] = 0.0
                    continue

                thresh = torch.topk(scores, k=k_prune, largest=False, dim=-1).values[:, -1:]
                mask = scores > thresh
                w_pruned = w.float() * mask.float()

                y_pruned = sample_x @ w_pruned.t()
                sq_err = torch.sum((y_target - y_pruned) ** 2)
                sensitivity_curve[name][s] = float((sq_err / y_norm).item())
                absolute_curve[name][s] = float((sq_err / sample_x.shape[0]).item())

    if return_absolute:
        return sensitivity_curve, absolute_curve
    return sensitivity_curve


def profile_layer_quantization_sensitivity(
    model: nn.Module,
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
    candidate_bits: tuple[int, ...] = (2, 3, 4, 8),
    group_size: int = 128,
) -> Dict[str, Dict[int, float]]:
    """Measure layer-wise output reconstruction error under PTQ at candidate bit widths."""
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    sensitivity_curve: Dict[str, Dict[int, float]] = {}

    print(f"\n[Sensitivity] Profiling quantization sensitivity across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            in_features = w.shape[1]
            sample_x = activation_stats.sample_inputs.get(name)

            if sample_x is None:
                sample_x = torch.randn(256, in_features, device=w.device, dtype=w.dtype)

            sample_x = sample_x.to(w.device).float()
            y_target = sample_x @ w.t().float()
            y_norm = torch.clamp(torch.sum(y_target ** 2), min=1e-8)

            sensitivity_curve[name] = {}
            for b in candidate_bits:
                w_hat, _, _ = quantize_weight_asymmetric(w, bit_width=b, group_size=group_size)
                y_quant = sample_x @ w_hat.t().float()
                rel_err = float((torch.sum((y_target - y_quant) ** 2) / y_norm).item())
                sensitivity_curve[name][b] = rel_err

    return sensitivity_curve


def profile_layer_awq_quantization_sensitivity(
    model: nn.Module,
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
    candidate_bits: tuple[int, ...] = (2, 3, 4, 8),
    group_size: int = 128,
    candidate_alphas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Isolated per-projection calibration output error of the SAME AWQ quantizer Q3 applies.

    Returns {"absolute": {name: {b: MSE}}, "relative": {name: {b: rel_err}}}. The absolute
    measure weights projections by how much output energy they actually perturb; the relative
    measure ignores output scale. Which one drives the allocation is chosen on the dev split.
    """
    absolute: Dict[str, Dict[int, float]] = {}
    relative: Dict[str, Dict[int, float]] = {}
    print(f"\n[Sensitivity] Profiling AWQ quantization sensitivity at bits {candidate_bits}...")
    with torch.no_grad():
        for name, module in get_eligible_linear_modules(model, eligible_projections).items():
            absolute[name], relative[name] = {}, {}
            for b in candidate_bits:
                _, err, rel = awq_quantize_layer(
                    module.weight.data, activation_stats.sample_inputs.get(name),
                    activation_stats.get_mean_abs(name), b, group_size, candidate_alphas,
                )
                absolute[name][b], relative[name][b] = err, rel
    return {"absolute": absolute, "relative": relative}
