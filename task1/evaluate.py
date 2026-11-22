"""task1/evaluate.py — Fast GPU evaluation for WikiText-2, C4, HellaSwag, and Paired Bootstrap CIs."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def evaluate_perplexity(
    model: nn.Module,
    window_data_path: Path | str,
    device: str = "cuda",
    batch_size: int = 16,
    target_scored_tokens: int = 50_000,
) -> Dict[str, Any]:
    """Evaluate language model perplexity over locked non-overlapping 512-token evaluation windows.

    Strict compliance with PA4 instructions:
    - Excludes the first token of each independently evaluated sequence from the scored-token denominator.
    - Correctly averages NLL before exponentiating: PPL = exp(mean_nll).
    - Returns per-window NLLs for paired bootstrap confidence intervals.
    """
    window_data_path = Path(window_data_path)
    if not window_data_path.exists():
        raise FileNotFoundError(f"Evaluation windows file not found: {window_data_path}")

    windows = []
    opener = gzip.open if str(window_data_path).endswith(".gz") else open
    with opener(window_data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item.get("token_ids", [])
            if len(tokens) >= 2:
                windows.append(tokens)

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    per_window_records: List[Dict[str, float]] = []

    print(f"[Eval PPL] Evaluating {len(windows)} windows from {window_data_path.name}...")

    with torch.inference_mode():
        for start_idx in range(0, len(windows), batch_size):
            if total_tokens >= target_scored_tokens:
                break

            end_idx = min(start_idx + batch_size, len(windows))
            batch_list = windows[start_idx:end_idx]

            max_len = max(len(s) for s in batch_list)
            input_tensor = torch.full((len(batch_list), max_len), fill_value=0, dtype=torch.long, device=device)
            mask_tensor = torch.zeros((len(batch_list), max_len), dtype=torch.bool, device=device)

            for i, s in enumerate(batch_list):
                input_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                mask_tensor[i, :len(s)] = True

            # Forward pass
            outputs = model(input_ids=input_tensor)
            logits = outputs.logits  # [B, L, V]

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_tensor[:, 1:].contiguous()
            valid_mask = mask_tensor[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.shape)

            for b in range(len(batch_list)):
                seq_valid = valid_mask[b]
                num_valid = int(seq_valid.sum().item())
                if num_valid <= 0:
                    continue

                take = min(num_valid, target_scored_tokens - total_tokens)
                seq_loss = loss[b, :take].sum().item()

                total_nll += seq_loss
                total_tokens += take

                per_window_records.append({
                    "nll": seq_loss / take,
                    "tokens": take,
                })

                if total_tokens >= target_scored_tokens:
                    break

    mean_nll = total_nll / max(1, total_tokens)
    ppl = float(np.exp(mean_nll))

    return {
        "dataset": window_data_path.name,
        "scored_tokens": total_tokens,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "per_window_records": per_window_records,
    }


def pretokenize_hellaswag(
    tokenizer: Any,
    hellaswag_path: Path | str,
    limit_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Pre-tokenize HellaSwag examples once to eliminate CPU tokenizer overhead during sweeps."""
    hellaswag_path = Path(hellaswag_path)
    if not hellaswag_path.exists():
        raise FileNotFoundError(f"HellaSwag data not found at: {hellaswag_path}")

    raw_examples = []
    opener = gzip.open if str(hellaswag_path).endswith(".gz") else open
    with opener(hellaswag_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            raw_examples.append(item.get("example", item))
            if limit_samples and len(raw_examples) >= limit_samples:
                break

    tokenized = []
    print(f"[Pretokenize HellaSwag] Tokenizing {len(raw_examples)} examples once...")
    for ex in raw_examples:
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])

        enc_ctx = tokenizer(ctx, add_special_tokens=False)["input_ids"]
        enc_ctx_len = len(enc_ctx)

        candidate_input_ids = []
        candidate_char_lens = []
        for end in endings:
            prompt = ctx + " " + end
            enc_prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            candidate_input_ids.append(enc_prompt)
            candidate_char_lens.append(max(1, len(end)))

        tokenized.append({
            "enc_ctx_len": enc_ctx_len,
            "candidate_input_ids": candidate_input_ids,
            "candidate_char_lens": candidate_char_lens,
            "label": label,
        })
    print(f"[Pretokenize HellaSwag] ✓ Pre-tokenized {len(tokenized)} examples.")
    return tokenized


def evaluate_hellaswag(
    model: nn.Module,
    tokenizer: Optional[Any] = None,
    hellaswag_path: Optional[Path | str] = None,
    device: str = "cuda",
    limit_samples: Optional[int] = None,
    pretokenized_examples: Optional[List[Dict[str, Any]]] = None,
    batch_examples: int = 4,
) -> Dict[str, Any]:
    """Evaluate zero-shot HellaSwag using conditional log likelihood scoring across 4 candidates.

    Reports:
    - Raw Accuracy: argmax of total log-likelihood
    - acc_norm: argmax of character-length-normalized log-likelihood (as used by lm-evaluation-harness)
    """
    if pretokenized_examples is not None:
        examples = pretokenized_examples[:limit_samples] if limit_samples else pretokenized_examples
    else:
        assert tokenizer is not None and hellaswag_path is not None, "Provide either pretokenized_examples or (tokenizer, hellaswag_path)"
        examples = pretokenize_hellaswag(tokenizer, hellaswag_path, limit_samples=limit_samples)

    model.eval()
    correct_raw = 0
    correct_norm = 0
    total_evaluated = 0
    per_example_correct: List[Dict[str, int]] = []

    print(f"[Eval HellaSwag] Evaluating {len(examples)} examples (batched GPU inference)...")

    loss_fct = nn.CrossEntropyLoss(reduction="none")

    with torch.inference_mode():
        for b_start in range(0, len(examples), batch_examples):
            b_end = min(b_start + batch_examples, len(examples))
            batch = examples[b_start:b_end]

            # Flatten all candidates in batch: each example has 4 candidates
            seqs_in_batch = []
            for ex in batch:
                for cand_ids in ex["candidate_input_ids"]:
                    seqs_in_batch.append(cand_ids)

            max_len = max(len(s) for s in seqs_in_batch)
            total_seqs = len(seqs_in_batch)

            input_tensor = torch.zeros((total_seqs, max_len), dtype=torch.long, device=device)
            for i, s in enumerate(seqs_in_batch):
                input_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

            outputs = model(input_ids=input_tensor)
            logits = outputs.logits  # [total_seqs, max_len, V]

            # Score each example
            for ex_idx, ex in enumerate(batch):
                label = ex["label"]
                enc_ctx_len = ex["enc_ctx_len"]
                cand_log_probs = []
                cand_norm_log_probs = []

                for cand_idx in range(4):
                    seq_idx = ex_idx * 4 + cand_idx
                    cand_len = len(ex["candidate_input_ids"][cand_idx])
                    char_len = ex["candidate_char_lens"][cand_idx]

                    if enc_ctx_len >= cand_len:
                        cand_log_probs.append(-99999.0)
                        cand_norm_log_probs.append(-99999.0)
                        continue

                    target_ids = input_tensor[seq_idx, enc_ctx_len:cand_len]
                    pred_logits = logits[seq_idx, enc_ctx_len - 1 : cand_len - 1]

                    losses = loss_fct(pred_logits, target_ids)
                    sum_log_prob = -float(losses.sum().item())
                    cand_log_probs.append(sum_log_prob)
                    cand_norm_log_probs.append(sum_log_prob / char_len)

                pred_raw = int(np.argmax(cand_log_probs))
                pred_norm = int(np.argmax(cand_norm_log_probs))

                is_raw = 1 if pred_raw == label else 0
                is_norm = 1 if pred_norm == label else 0

                correct_raw += is_raw
                correct_norm += is_norm
                total_evaluated += 1

                per_example_correct.append({
                    "raw": is_raw,
                    "norm": is_norm,
                })

    raw_acc = correct_raw / max(1, total_evaluated)
    norm_acc = correct_norm / max(1, total_evaluated)

    return {
        "total_samples": total_evaluated,
        "acc": raw_acc,
        "acc_norm": norm_acc,
        "per_example_correct": per_example_correct,
    }


def compute_paired_bootstrap_confidence_interval(
    scores_a: List[float | int],
    scores_b: List[float | int],
    num_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 6304,
) -> Dict[str, float]:
    """Compute paired bootstrap difference (b - a) and 95% confidence interval."""
    assert len(scores_a) == len(scores_b), f"Length mismatch: {len(scores_a)} vs {len(scores_b)}"
    n = len(scores_a)
    diffs = np.array(scores_b) - np.array(scores_a)

    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(num_resamples):
        sample_indices = rng.integers(0, n, size=n)
        boot_means.append(float(np.mean(diffs[sample_indices])))

    alpha = (1.0 - confidence_level) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff": mean_diff,
        "ci_low": low,
        "ci_high": high,
        "confidence_level": confidence_level,
        "samples": n,
    }


def benchmark_model_latency_and_vram(
    model: nn.Module,
    device: str = "cuda",
    prefill_lengths: tuple[int, ...] = (512, 2048),
    decode_tokens: int = 64,
    repetitions: int = 5,
) -> Dict[str, Any]:
    """Measure prefill latency, decode latency, and peak allocated GPU memory."""
    model.eval()
    results: Dict[str, Any] = {}

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # 1. Prefill latency
    prefill_latencies = {}
    with torch.inference_mode():
        for seq_len in prefill_lengths:
            dummy_input = torch.randint(100, 1000, (1, seq_len), device=device)

            # Warmup
            for _ in range(2):
                _ = model(dummy_input)
            torch.cuda.synchronize(device)

            times = []
            for _ in range(repetitions):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - t0)

            prefill_latencies[f"prefill_{seq_len}"] = float(np.mean(times))

    # 2. Decode latency (64 tokens)
    with torch.inference_mode():
        dummy_input = torch.randint(100, 1000, (1, 1), device=device)
        for _ in range(2):
            _ = model(dummy_input)
        torch.cuda.synchronize(device)

        times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            for _ in range(decode_tokens):
                _ = model(dummy_input)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

        decode_latency = float(np.mean(times))

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    results.update(prefill_latencies)
    results["decode_64_tokens"] = decode_latency
    results["peak_vram_mb"] = peak_vram_mb
    return results
