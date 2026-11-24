"""task2/mqar_dataset.py — High-throughput GPU dataset and batch generator for MQAR.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from task2.config import (
    REPO_ROOT,
    MQAR_SEQ_LEN,
    MQAR_KEY_RANGE,
    MQAR_VALUE_RANGE,
    MQAR_FILLER_RANGE,
    MQAR_DISTANCE_BINS,
)
from scripts.mqar_generator import make_example, validate


class MQARDataset(Dataset):
    """In-memory or on-the-fly MQAR dataset converting raw examples to torch tensors."""

    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        tokens = torch.tensor(ex["tokens"], dtype=torch.long)
        
        # Supervised target tensor: -100 for unmasked positions, class index (0..63) for queries
        target = torch.full((MQAR_SEQ_LEN,), -100, dtype=torch.long)
        query_mask = torch.zeros(MQAR_SEQ_LEN, dtype=torch.bool)

        for pos, ans in zip(ex["query_positions"], ex["answer_ids"]):
            # Answer IDs are 128..191 -> map to class 0..63
            class_idx = ans - MQAR_VALUE_RANGE[0]
            target[pos] = class_idx
            query_mask[pos] = True

        return {
            "input_ids": tokens,
            "target": target,
            "query_mask": query_mask,
            "distances": torch.tensor(ex["distances"], dtype=torch.long),
            "answer_ids": torch.tensor(ex["answer_ids"], dtype=torch.long),
            "query_positions": torch.tensor(ex["query_positions"], dtype=torch.long),
            "source_positions": torch.tensor(ex["source_positions"], dtype=torch.long),
            "k": ex["K"],
            "regime": ex["regime"],
        }


def load_frozen_mqar_test() -> List[Dict]:
    """Load the official frozen test set from data/mqar/mqar_test.jsonl.gz."""
    test_path = REPO_ROOT / "data" / "mqar" / "mqar_test.jsonl.gz"
    if not test_path.exists():
        # Fallback to generating test examples deterministically
        print(f"[MQAR Dataset] Generating test examples with seed {6304}...")
        examples = []
        for regime in ("near", "far", "mixed"):
            for k in (4, 8, 16):
                for i in range(1024):
                    examples.append(make_example("test", k, regime, i))
        return examples

    records = []
    with gzip.open(test_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_mqar_development() -> List[Dict]:
    """Load the supplied development stream (disjoint from training and final test)."""
    dev_path = REPO_ROOT / "data" / "mqar" / "mqar_development.jsonl.gz"
    with gzip.open(dev_path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pregenerate_gpu_training_pool(
    k: int,
    regime: str,
    pool_size: int = 14400,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-generate a large pool of valid MQAR training sequences directly into GPU VRAM.
    
    Returns:
        input_ids: (pool_size, 256) on GPU
        targets:   (pool_size, 256) on GPU with class labels (0..63) at queries, -100 elsewhere
    """
    input_ids_list = []
    targets_list = []

    for i in range(pool_size):
        ex = make_example("train", k, regime, i)
        tokens = ex["tokens"]
        target = [-100] * MQAR_SEQ_LEN
        for pos, ans in zip(ex["query_positions"], ex["answer_ids"]):
            target[pos] = ans - MQAR_VALUE_RANGE[0]
        input_ids_list.append(tokens)
        targets_list.append(target)

    input_ids_t = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets_list, dtype=torch.long, device=device)
    return input_ids_t, targets_t
