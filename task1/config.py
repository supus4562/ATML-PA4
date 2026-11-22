"""task1/config.py — Central configuration and constants for Task 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task1Config:
    # Model specification
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    device: str = "cuda"
    torch_dtype: str = "float16"

    # Modules subject to compression (7 linear projections per Transformer block)
    eligible_projections: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Excluded from compression (remain pure FP16)
    excluded_modules: tuple[str, ...] = (
        "embed_tokens",
        "lm_head",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    )

    # Calibration & Development split specifications
    calibration_seq_count: int = 128
    calibration_seq_len: int = 512
    development_seq_count: int = 32
    development_seq_len: int = 512

    # Pruning specifications
    p1_sparsities: tuple[float, ...] = (0.25, 0.50, 0.75)
    p2_sparsities: tuple[float, ...] = (0.25, 0.50, 0.75)
    p3_global_sparsities: tuple[float, ...] = (0.50, 0.75)
    p3_min_sparsity: float = 0.10
    p3_max_sparsity: float = 0.85

    # Quantization specifications
    quant_group_size: int = 128
    q1_bit_widths: tuple[int, ...] = (8, 4, 3)
    q2_bit_widths: tuple[int, ...] = (8, 4, 3)
    q3_allowed_bits: tuple[int, ...] = (2, 3, 4, 8)
    awq_grid_alphas: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # Evaluation specifications
    eval_window_len: int = 512
    wikitext_target_tokens: int = 50_000
    c4_target_tokens: int = 50_000
    hellaswag_total_samples: int = 2_000
    hellaswag_diagnostic_samples: int = 512
    bootstrap_resamples: int = 2_000

    # Paths (defaults support both cloud container /root/volume and local relative paths)
    base_data_dir: Path = Path("/root/ATML-PA4/data")
    volume_dir: Path = Path("/root/volume")
    output_dir: Path = Path("/root/volume/results/task1")
    checkpoint_dir: Path = Path("/root/volume/checkpoints/task1")

    # Master random seed
    seed: int = 6304
