"""task2/mqar_train.py — GPU-accelerated training loop for MQAR models.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from task2.config import (
    MQAR_BATCH_SIZE,
    MQAR_TOTAL_UPDATES,
    MQAR_WARMUP_UPDATES,
    MQAR_PEAK_LR,
    MQAR_MIN_LR_RATIO,
    MQAR_WEIGHT_DECAY,
    MQAR_GRAD_CLIP,
    MQAR_CHECKPOINT_UPDATES,
)
from task2.mqar_models import MQARTransformer
from task2.mqar_dataset import pregenerate_gpu_training_pool


def get_lr_scheduler(optimizer: torch.optim.Optimizer, step: int) -> float:
    """Linear warmup for 200 steps followed by linear decay to 15% of peak LR at 4500 steps."""
    if step < MQAR_WARMUP_UPDATES:
        lr = MQAR_PEAK_LR * float(step + 1) / float(max(1, MQAR_WARMUP_UPDATES))
    else:
        progress = float(step - MQAR_WARMUP_UPDATES) / float(max(1, MQAR_TOTAL_UPDATES - MQAR_WARMUP_UPDATES))
        decay_factor = (1.0 - progress) * (1.0 - MQAR_MIN_LR_RATIO) + MQAR_MIN_LR_RATIO
        lr = MQAR_PEAK_LR * max(MQAR_MIN_LR_RATIO, decay_factor)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_mqar_model(
    attn_type: str,
    k: int,
    regime: str,
    seed: int,
    device: str = "cuda",
    total_updates: int = MQAR_TOTAL_UPDATES,
    batch_size: int = MQAR_BATCH_SIZE,
    verbose: bool = True,
) -> Tuple[MQARTransformer, Dict[str, Any]]:
    """Train a single MQAR model from scratch using the exact PA4 protocol."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MQARTransformer(attn_type=attn_type).to(device)
    optimizer = AdamW(model.parameters(), lr=MQAR_PEAK_LR, weight_decay=MQAR_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Pre-generate a GPU training pool for zero data loader bottleneck
    pool_size = min(14400, total_updates * batch_size)
    if verbose:
        print(f"\n[MQAR Train] Pre-generating {pool_size} {regime} K={k} sequences in GPU VRAM (seed={seed})...")
    input_ids_pool, targets_pool = pregenerate_gpu_training_pool(k=k, regime=regime, pool_size=pool_size, device=device)

    loss_history: List[Tuple[int, float, float]] = []  # (step, loss, lr)
    ckpt_losses: Dict[int, float] = {}

    model.train()
    t0 = time.perf_counter()
    pool_idx = 0

    for step in range(1, total_updates + 1):
        # Update learning rate
        cur_lr = get_lr_scheduler(optimizer, step)

        # Slice mini-batch directly from GPU memory
        batch_inputs = input_ids_pool[pool_idx : pool_idx + batch_size]
        batch_targets = targets_pool[pool_idx : pool_idx + batch_size]
        pool_idx += batch_size
        if pool_idx + batch_size > pool_size:
            pool_idx = 0  # Wrap around

        optimizer.zero_grad()
        logits = model(batch_inputs)  # (B, T, 64)
        loss = criterion(logits.view(-1, 64), batch_targets.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MQAR_GRAD_CLIP)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append((step, loss_val, cur_lr))

        if step in MQAR_CHECKPOINT_UPDATES or step % 500 == 0:
            ckpt_losses[step] = loss_val
            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"  Step {step:4d}/{total_updates} | Loss: {loss_val:.4f} | LR: {cur_lr:.2e} | Elapsed: {elapsed:.1f}s", flush=True)

    train_time = time.perf_counter() - t0
    if verbose:
        print(f"[MQAR Train] Completed {attn_type} (K={k}, {regime}, seed={seed}) in {train_time:.1f}s. Final loss: {loss_val:.4f}", flush=True)

    train_summary = {
        "attn_type": attn_type,
        "k": k,
        "regime": regime,
        "seed": seed,
        "train_time_sec": train_time,
        "final_loss": loss_val,
        "checkpoint_losses": ckpt_losses,
        "loss_curve": loss_history[::20],  # Subsampled for compact logging
    }

    return model, train_summary
