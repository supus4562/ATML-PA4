"""task1/pruners.py — Vectorized PyTorch CUDA pruners for P1, P2, P3, and Structured 2:4."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from task1.calibration import ActivationStats


def get_eligible_linear_modules(
    model: nn.Module, eligible_projections: tuple[str, ...]
) -> Dict[str, nn.Linear]:
    """Retrieve all eligible linear layers from the model."""
    eligible_modules: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            proj_name = name.split(".")[-1]
            if proj_name in eligible_projections:
                eligible_modules[name] = module
    return eligible_modules


def prune_magnitude_p1(
    model: nn.Module,
    sparsity: float,
    eligible_projections: tuple[str, ...],
) -> Dict[str, float]:
    """P1: Unstructured per-output-row magnitude pruning.

    For each output row of W, sets the lowest `sparsity` fraction of weights to zero.
    Fully vectorized on GPU.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    achieved_sparsities: Dict[str, float] = {}

    print(f"\n[P1 Magnitude Pruning] Applying target sparsity {sparsity:.2%} across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            # w shape: [out_features, in_features]
            n_in = w.shape[1]
            k_prune = int(round(sparsity * n_in))

            if k_prune <= 0:
                achieved_sparsities[name] = 0.0
                continue
            if k_prune >= n_in:
                w.zero_()
                achieved_sparsities[name] = 1.0
                continue

            # Per-row magnitude scores
            scores = torch.abs(w)

            # Find threshold per row using topk along input dim
            # Values smaller than the k_prune-th smallest value are zeroed
            thresh = torch.topk(scores, k=k_prune, largest=False, dim=-1).values[:, -1:]
            mask = scores > thresh

            # Apply mask in place
            w.mul_(mask.to(w.dtype))
            achieved = float((w == 0).sum().item()) / w.numel()
            achieved_sparsities[name] = achieved

    return achieved_sparsities


def prune_wanda_p2(
    model: nn.Module,
    sparsity: float,
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
) -> Dict[str, float]:
    """P2: Activation-aware pruning (Wanda).

    Score S_ij = |W_ij| * ||X_j,:||_2 per output row.
    Zeroes out the lowest `sparsity` fraction per output row.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    achieved_sparsities: Dict[str, float] = {}

    print(f"\n[P2 Wanda Pruning] Applying target sparsity {sparsity:.2%} across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            n_in = w.shape[1]
            k_prune = int(round(sparsity * n_in))

            if k_prune <= 0:
                achieved_sparsities[name] = 0.0
                continue

            # Input channel L2 norm
            l2_norm = activation_stats.get_l2_norm(name).to(w.device).unsqueeze(0)
            scores = torch.abs(w) * l2_norm

            thresh = torch.topk(scores, k=k_prune, largest=False, dim=-1).values[:, -1:]
            mask = scores > thresh

            w.mul_(mask.to(w.dtype))
            achieved = float((w == 0).sum().item()) / w.numel()
            achieved_sparsities[name] = achieved

    return achieved_sparsities


def prune_structured_2_4(
    model: nn.Module,
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
) -> Dict[str, float]:
    """Structured 2:4 sparsity diagnostic.

    In every contiguous group of 4 input weights, retains exactly the 2 highest-ranked
    weights according to Wanda score.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    achieved_sparsities: Dict[str, float] = {}

    print(f"\n[Structured 2:4 Pruning] Enforcing 2:4 mask across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            w = module.weight.data
            out_features, in_features = w.shape

            assert in_features % 4 == 0, f"in_features {in_features} must be divisible by 4 for 2:4 sparsity"

            l2_norm = activation_stats.get_l2_norm(name).to(w.device).unsqueeze(0)
            scores = torch.abs(w) * l2_norm

            # Reshape into groups of 4: [out_features, in_features // 4, 4]
            scores_grouped = scores.view(out_features, in_features // 4, 4)

            # Find indices of top-2 in each group
            top2_indices = torch.topk(scores_grouped, k=2, largest=True, dim=-1).indices

            # Construct boolean mask
            mask_grouped = torch.zeros_like(scores_grouped, dtype=torch.bool)
            mask_grouped.scatter_(-1, top2_indices, True)

            mask = mask_grouped.view(out_features, in_features)
            w.mul_(mask.to(w.dtype))

            achieved = float((w == 0).sum().item()) / w.numel()
            achieved_sparsities[name] = achieved

    return achieved_sparsities


def prune_nonuniform_p3(
    model: nn.Module,
    allocation_map: Dict[str, float],
    activation_stats: ActivationStats,
    eligible_projections: tuple[str, ...],
) -> Dict[str, float]:
    """P3: Non-uniform pruning using layer-specific Wanda sparsity allocations.

    `allocation_map` specifies the assigned sparsity s_ell in [0.10, 0.85] for each layer.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    achieved_sparsities: Dict[str, float] = {}

    print(f"\n[P3 Non-Uniform Pruning] Applying adaptive allocations across {len(eligible_modules)} layers...")

    with torch.no_grad():
        for name, module in eligible_modules.items():
            s_ell = allocation_map.get(name, 0.50)
            w = module.weight.data
            n_in = w.shape[1]
            k_prune = int(round(s_ell * n_in))

            if k_prune <= 0:
                achieved_sparsities[name] = 0.0
                continue

            l2_norm = activation_stats.get_l2_norm(name).to(w.device).unsqueeze(0)
            scores = torch.abs(w) * l2_norm

            thresh = torch.topk(scores, k=k_prune, largest=False, dim=-1).values[:, -1:]
            mask = scores > thresh

            w.mul_(mask.to(w.dtype))
            achieved = float((w == 0).sum().item()) / w.numel()
            achieved_sparsities[name] = achieved

    return achieved_sparsities
