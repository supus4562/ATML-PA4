"""task1/calibration.py — Fast GPU-resident activation collector via forward hooks."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


def load_c4_calibration_tokens(
    data_path: Path | str,
    max_sequences: int = 128,
    max_len: int = 512,
) -> torch.Tensor:
    """Load pre-tokenized calibration sequences from locked JSONL.gz manifest."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Calibration data not found at: {data_path}")

    sequences = []
    opener = gzip.open if str(data_path).endswith(".gz") else open
    with opener(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("token_ids", [])
            if len(tokens) >= max_len:
                sequences.append(tokens[:max_len])
            if len(sequences) >= max_sequences:
                break

    if len(sequences) < max_sequences:
        print(f"[Calibration] Warning: Only found {len(sequences)} sequences in {data_path}")

    tensor = torch.tensor(sequences, dtype=torch.long)
    return tensor


class ActivationStats:
    """Container for layerwise input activation norms and sampled representations."""

    def __init__(self):
        # Maps module_name -> tensor of shape [in_features] (sum of squares in FP32)
        self.sq_sum: Dict[str, torch.Tensor] = {}
        # Maps module_name -> tensor of shape [in_features] (sum of absolute values in FP32)
        self.abs_sum: Dict[str, torch.Tensor] = {}
        # Maps module_name -> tensor of shape [in_features] (maximum absolute value)
        self.abs_max: Dict[str, torch.Tensor] = {}
        # Sampled input matrix for fast layer-level reconstruction loss [N_samples, in_features]
        self.sample_inputs: Dict[str, torch.Tensor] = {}
        self.token_count: Dict[str, int] = {}

    def get_l2_norm(self, name: str) -> torch.Tensor:
        """Return L2 norm per input channel: sqrt(sum(X_j^2))."""
        return torch.sqrt(self.sq_sum[name].float())

    def get_mean_abs(self, name: str) -> torch.Tensor:
        """Return mean absolute magnitude per input channel: E[|X_j|]."""
        return (self.abs_sum[name].float() / max(1, self.token_count[name]))

    def get_max_abs(self, name: str) -> torch.Tensor:
        return self.abs_max[name].float()


def collect_calibration_statistics(
    model: nn.Module,
    calib_tokens: torch.Tensor,
    eligible_projections: tuple[str, ...],
    batch_size: int = 8,
    sample_tokens_for_reconstruction: int = 2048,
    device: str = "cuda",
) -> ActivationStats:
    """Pass calibration sequences through the model with forward hooks to collect activation statistics.

    Executes entirely on GPU with zero host-device overhead.
    """
    stats = ActivationStats()
    hooks = []

    # Map name -> module for all eligible linear layers
    target_modules: Dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if module ends with any eligible projection name
            proj_name = name.split(".")[-1]
            if proj_name in eligible_projections:
                target_modules[name] = module

    print(f"[Calibration] Registering forward hooks on {len(target_modules)} eligible projections...")

    def make_hook(layer_name: str):
        def hook_fn(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
            # Input shape: [batch_size, seq_len, in_features]
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x_f32 = x.float()

            num_tokens = x.shape[0]
            with torch.no_grad():
                # Accumulate online sum of squares and absolute values
                batch_sq = torch.sum(x_f32 ** 2, dim=0)
                batch_abs = torch.sum(torch.abs(x_f32), dim=0)
                batch_max = torch.max(torch.abs(x_f32), dim=0).values

                if layer_name not in stats.sq_sum:
                    stats.sq_sum[layer_name] = batch_sq
                    stats.abs_sum[layer_name] = batch_abs
                    stats.abs_max[layer_name] = batch_max
                    stats.token_count[layer_name] = num_tokens
                else:
                    stats.sq_sum[layer_name] += batch_sq
                    stats.abs_sum[layer_name] += batch_abs
                    stats.abs_max[layer_name] = torch.maximum(stats.abs_max[layer_name], batch_max)
                    stats.token_count[layer_name] += num_tokens

                # Retain a fixed sample slice for fast layer reconstruction testing
                if layer_name not in stats.sample_inputs:
                    take = min(num_tokens, sample_tokens_for_reconstruction)
                    stats.sample_inputs[layer_name] = x[:take].to(torch.float16)
                elif stats.sample_inputs[layer_name].shape[0] < sample_tokens_for_reconstruction:
                    needed = sample_tokens_for_reconstruction - stats.sample_inputs[layer_name].shape[0]
                    take = min(num_tokens, needed)
                    stats.sample_inputs[layer_name] = torch.cat(
                        [stats.sample_inputs[layer_name], x[:take].to(torch.float16)], dim=0
                    )

        return hook_fn

    for name, module in target_modules.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Run batched forward passes
    model.eval()
    num_seqs = calib_tokens.shape[0]
    print(f"[Calibration] Forwarding {num_seqs} sequences through model on {device}...")

    with torch.inference_mode():
        for start_idx in range(0, num_seqs, batch_size):
            end_idx = min(start_idx + batch_size, num_seqs)
            batch = calib_tokens[start_idx:end_idx].to(device)
            # Forward through model
            model(input_ids=batch)

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"[Calibration] Finished collecting statistics for {len(stats.sq_sum)} projections.")
    return stats
