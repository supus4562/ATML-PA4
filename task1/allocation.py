"""task1/allocation.py — Constrained budget solvers for P3 (Non-uniform pruning) and Q3 (Mixed precision)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from task1.packed_storage import compute_model_packed_storage
from task1.pruners import get_eligible_linear_modules


def solve_p3_sparsity_allocation(
    model: nn.Module,
    eligible_projections: tuple[str, ...],
    sensitivity_scores: Dict[str, float],
    target_global_sparsity: float,
    min_sparsity: float = 0.10,
    max_sparsity: float = 0.85,
) -> Dict[str, float]:
    """Solve for layer-wise sparsities s_ell in [min_sparsity, max_sparsity] matching exact total zeros.

    Uses water-filling / bisection on Lagrange multiplier:
    More sensitive layers (high error) get lower sparsity; robust layers get higher sparsity.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    layer_names = list(eligible_modules.keys())
    layer_param_counts = {name: eligible_modules[name].weight.numel() for name in layer_names}

    total_eligible_params = sum(layer_param_counts.values())
    target_total_zeros = int(round(target_global_sparsity * total_eligible_params))

    # Normalize log sensitivities (z-score)
    sens_vals = [max(1e-6, sensitivity_scores.get(name, 1.0)) for name in layer_names]
    log_sens = [math.log(v) for v in sens_vals]
    mean_log_sens = sum(log_sens) / len(log_sens)
    var_log_sens = sum((x - mean_log_sens) ** 2 for x in log_sens) / len(log_sens)
    std_log_sens = math.sqrt(var_log_sens) if var_log_sens > 1e-6 else 1.0
    norm_sens = [(x - mean_log_sens) / std_log_sens for x in log_sens]

    # Responsiveness scaling alpha: determines how much sensitive layers diverge from insensitive layers
    alpha = 0.15

    # Bisection search over offset lambda to match exact global zero count
    low, high = -1.0, 1.0
    best_allocations: Dict[str, float] = {}

    for _ in range(100):
        mid = (low + high) / 2.0
        current_zeros = 0
        allocations = {}

        for name, ns in zip(layer_names, norm_sens):
            # More sensitive layers (higher ns) get lower sparsity; less sensitive layers get higher sparsity
            raw_s = target_global_sparsity - alpha * ns + mid
            clamped_s = min(max_sparsity, max(min_sparsity, raw_s))
            allocations[name] = clamped_s

            numel = layer_param_counts[name]
            current_zeros += int(round(clamped_s * numel))

        best_allocations = allocations
        if abs(current_zeros - target_total_zeros) <= 1:
            break
        elif current_zeros < target_total_zeros:
            low = mid
        else:
            high = mid

    # Final fine-tuning to guarantee EXACT integer zero count
    current_zeros = sum(int(round(best_allocations[name] * layer_param_counts[name])) for name in layer_names)
    diff = target_total_zeros - current_zeros

    if diff != 0:
        # Fine-tune by adjusting layers whose column step (n_out) reduces |diff| without oscillating
        sorted_layers = sorted(layer_names, key=lambda n: sensitivity_scores.get(n, 1.0))
        for target_name in sorted_layers:
            n_out, n_in = eligible_modules[target_name].weight.shape
            step_zeros = n_out
            if diff >= step_zeros and best_allocations[target_name] + (1.0 / n_in) <= max_sparsity:
                best_allocations[target_name] += 1.0 / n_in
                diff -= step_zeros
            elif diff <= -step_zeros and best_allocations[target_name] - (1.0 / n_in) >= min_sparsity:
                best_allocations[target_name] -= 1.0 / n_in
                diff += step_zeros
            if diff == 0:
                break

    achieved_global_sparsity = sum(
        best_allocations[n] * layer_param_counts[n] for n in layer_names
    ) / total_eligible_params

    print(f"[P3 Allocation] Target: {target_global_sparsity:.2%}, Achieved: {achieved_global_sparsity:.4%}")
    return best_allocations


def solve_q3_bit_allocation(
    model: nn.Module,
    eligible_projections: tuple[str, ...],
    quant_sensitivity: Dict[str, Dict[int, float]],
    target_packed_bytes: int,
    allowed_bits: tuple[int, ...] = (2, 3, 4, 8),
    group_size: int = 128,
) -> Dict[str, int]:
    """Solve for layer-wise bit widths b_ell in {2, 3, 4, 8} strictly within target_packed_bytes.

    Uses greedy marginal-utility knapsack:
    Starts all layers at the lowest bit width (2-bit), then iteratively upgrades
    the layer that yields the highest error reduction per additional byte.
    """
    eligible_modules = get_eligible_linear_modules(model, eligible_projections)
    layer_names = list(eligible_modules.keys())

    # Initialize all eligible layers at lowest allowed bit width
    min_bit = min(allowed_bits)
    allocations: Dict[str, int] = {name: min_bit for name in layer_names}

    sorted_bits = sorted(allowed_bits)

    # Compute baseline storage
    current_storage = compute_model_packed_storage(
        model,
        eligible_projections=eligible_projections,
        bit_allocations=allocations,
        group_size=group_size,
    )
    current_bytes = current_storage["total_bytes"]

    if current_bytes > target_packed_bytes:
        raise ValueError(
            f"Even all layers at {min_bit}-bit exceeds target budget: {current_bytes} > {target_packed_bytes}"
        )

    # Greedily upgrade layers while budget permits
    while True:
        best_layer = None
        best_next_bit = None
        best_marginal_utility = -1.0
        best_delta_bytes = 0

        for name in layer_names:
            curr_b = allocations[name]
            curr_idx = sorted_bits.index(curr_b)
            if curr_idx + 1 >= len(sorted_bits):
                continue  # already at maximum bit width (8-bit)

            next_b = sorted_bits[curr_idx + 1]

            # Calculate byte increase for this specific layer
            numel = eligible_modules[name].weight.numel()
            delta_payload_bytes = math.ceil((numel * (next_b - curr_b)) / 8)
            cand_bytes = current_bytes + delta_payload_bytes

            if cand_bytes <= target_packed_bytes:
                # Error reduction: Error(curr_b) - Error(next_b)
                err_curr = quant_sensitivity[name].get(curr_b, 1.0)
                err_next = quant_sensitivity[name].get(next_b, 0.0)
                delta_error = max(0.0, err_curr - err_next)

                marginal_utility = delta_error / max(1, delta_payload_bytes)

                if marginal_utility > best_marginal_utility:
                    best_marginal_utility = marginal_utility
                    best_layer = name
                    best_next_bit = next_b
                    best_delta_bytes = delta_payload_bytes

        if best_layer is None or best_marginal_utility <= 0:
            break  # No further beneficial upgrade within budget

        allocations[best_layer] = best_next_bit
        current_bytes += best_delta_bytes

    final_storage = compute_model_packed_storage(
        model,
        eligible_projections=eligible_projections,
        bit_allocations=allocations,
        group_size=group_size,
    )
    print(
        f"[Q3 Allocation] Target Budget: {target_packed_bytes:,} bytes, "
        f"Achieved: {final_storage['total_bytes']:,} bytes ({final_storage['avg_bits_per_param']:.2f} avg bits)"
    )
    assert final_storage["total_bytes"] <= target_packed_bytes, "Safety check: allocation exceeded budget!"
    return allocations
