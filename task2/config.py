"""task2/config.py — Central configuration, hyperparameter definitions, and paths for Task 2.
"""

from __future__ import annotations

import os
from pathlib import Path
import torch

# Base directories
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RESULTS_DIR = REPO_ROOT / "runs" / "task2_results"
LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cloud volume directories
CLOUD_VOLUME_ROOT = Path(os.environ.get("MODAL_VOLUME_DIR", "/root/volume"))
CLOUD_CHECKPOINT_DIR = CLOUD_VOLUME_ROOT / "checkpoints" / "task2"
CLOUD_DATA_DIR = CLOUD_VOLUME_ROOT / "data"

# Master Seeds
MASTER_SEED = 6304
TRAIN_SEEDS = [6304, 6305, 6306]

# ──────────────────────────────────────────────────────────────────────────────
# Part A: Attention Operators Configuration
# ──────────────────────────────────────────────────────────────────────────────
PART_A_BATCH = 2
PART_A_HEADS = 4
PART_A_HEAD_DIM = 64
PART_A_LENGTHS = [256, 512, 1024, 2048, 4096, 8192]
PART_A_DTYPES = [torch.float32, torch.float16]
PART_A_REPETITIONS = 10
PART_A_WARMUP = 5
PART_A_TILE_BLOCK_SIZES = [32, 64, 128, 256]

# ──────────────────────────────────────────────────────────────────────────────
# Part B: Multi-Query Associative Recall (MQAR) Configuration
# ──────────────────────────────────────────────────────────────────────────────
MQAR_SEQ_LEN = 256
MQAR_VOCAB_SIZE = 256
MQAR_HIDDEN_DIM = 128
MQAR_NUM_HEADS = 4
MQAR_HEAD_DIM = 32  # 128 / 4
MQAR_NUM_BLOCKS = 2
MQAR_MLP_DIM = 512  # 4 * 128

# Key/Value Token IDs
MQAR_KEY_RANGE = (64, 128)      # [64, 127]
MQAR_VALUE_RANGE = (128, 192)   # [128, 191] -> 64 candidate answer classes
MQAR_FILLER_RANGE = (2, 64)     # [2, 63]

# Training Hyperparameters
MQAR_BATCH_SIZE = 32
MQAR_TOTAL_UPDATES = 4500
MQAR_WARMUP_UPDATES = 200
MQAR_PEAK_LR = 8e-4
MQAR_MIN_LR_RATIO = 0.15        # Linear decay to 15% of peak LR (1.2e-4)
MQAR_WEIGHT_DECAY = 0.01
MQAR_GRAD_CLIP = 1.0
MQAR_CHECKPOINT_UPDATES = [500, 1000, 1500, 2000, 3000, 4500]

# Loads & Regimes
MQAR_K_VALUES = [4, 8, 16]
MQAR_REGIMES = ["near", "far", "mixed"]
MQAR_DISTANCE_BINS = [
    (0, 32, "[0, 32)"),
    (32, 64, "[32, 64)"),
    (64, 128, "[64, 128)"),
    (128, 256, "[128, 256)"),
]

# Required Model Training Grid:
# 1. Dense: Mixed (K=4, 8, 16) + Near-only (K=16)
# 2. Sliding Window (W=64): Near-only (K=4, 8, 16) + Mixed (K=16)
# 3. Linear Attention (ELU+1): Mixed (K=4, 8, 16)
# 4. Mixed Local/Dilated Sparse: Mixed (K=4, 16)
# 5. Controlled Intervention (DGLA): Mixed (K=16) across 3 seeds
MQAR_INFERENCE_SWEEP_LENGTHS = [256, 1024, 4096, 8192]

# ──────────────────────────────────────────────────────────────────────────────
# Part C: Pretrained KV Memory Compression Configuration
# ──────────────────────────────────────────────────────────────────────────────
KV_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
KV_NUM_LAYERS = 28
KV_NUM_KV_HEADS = 2
KV_NUM_Q_HEADS = 14
KV_HEAD_DIM = 128
KV_GROUP_SIZE = 64

# Prompt factorial factors
KV_CONTEXT_LENGTHS = [2048, 4096]
KV_EVIDENCE_POSITIONS = ["start", "middle", "end"]
KV_DISTRACTOR_COUNTS = [0, 2, 8]
KV_CASES_PER_CELL = 10
KV_TOTAL_PROMPTS = 180

# Retention safeguards
KV_SINK_TOKENS = 8
KV_RECENT_TOKENS = 64

# Bootstrap & Benchmark Repetitions
KV_BOOTSTRAP_RESAMPLES = 2000
KV_BENCHMARK_REPETITIONS = 10
KV_BENCHMARK_BATCHES = [1, 8]

# ──────────────────────────────────────────────────────────────────────────────
# Part B3: Controlled intervention (stated BEFORE the rerun's evaluation)
# ──────────────────────────────────────────────────────────────────────────────
INTERVENTION_SPEC = {
    "name": "Decay-gated linear attention (DGLA)",
    "comparator": "causal ELU+1 linear attention, K=16, mixed regime",
    "change": "input-dependent per-head forgetting gate gamma_t in (0.5, 1): "
              "S_t = gamma_t S_{t-1} + phi(k_t) v_t^T, z_t = gamma_t z_{t-1} + phi(k_t)",
    "motivation": "linear attention accumulates every key into one fixed-size state, so filler tokens "
                  "and earlier pairs interfere with recall; a learned gate can down-weight filler writes",
    "prediction": "DGLA will match or exceed linear attention on near queries and will not exceed it on "
                  "far queries (distance >= 128), because any decay gamma < 1 attenuates old pairs "
                  "geometrically (0.5^128 at the gate's lower bound).",
    "extra_trainable_params": "gate_proj: 128*4 weights + 4 biases = 516 per layer, 1,032 total "
                              "(well under 5% of the baseline)",
    "persistent_state_per_layer": "identical to the comparator: per head S (32x32) + z (32), FP32 = "
                                  "4 * (1024 + 32) * 4 heads = 16,896 bytes; the gate adds no stored state",
    "attended_positions": "all preceding positions via the recurrent state (no pairwise selection)",
    "development_selection_rule": "no hyperparameter search; gate range fixed a priori; final 4,500-update "
                                  "checkpoint reported; development-stream accuracy logged at the "
                                  "prescribed checkpoints only for learning curves",
    "seeds": [6304, 6305, 6306],
}
