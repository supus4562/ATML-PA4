"""task2/eval_context_benchmark.py — Evaluates context memory and aggregation via:
Option A: Official OOLONG (synth-with-labels) with baseline gating.
Option B: Multi-Needle in a Haystack (M-NIAH) across 2k & 4k tokens with K in {2, 4, 8}.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from task2.config import (
    REPO_ROOT,
    LOCAL_RESULTS_DIR,
    KV_MODEL_NAME,
)
from task2.kv_policies import (
    apply_policy_sink_recent,
    apply_policy_question_aware,
    apply_policy_token_wise_int4,
    apply_policy_channel_k_token_v_int4,
    apply_policy_hybrid,
)
from task2.eval_oolong import synth_process_response


# ──────────────────────────────────────────────────────────────────────────────
# Option B: Multi-Needle in a Haystack (M-NIAH) Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_mniah_prompt(
    context_tokens: int = 2048,
    num_needles: int = 4,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Construct a Multi-Needle prompt with K distinct keys and codes distributed across context."""
    rng = random.Random(seed)
    # Generate K unique codes and labels
    needles = []
    for i in range(num_needles):
        code = f"{rng.randint(10000, 99999)}"
        label = f"RECORD-{chr(65 + i)}"
        needles.append((label, code))

    # Filler text bank
    filler_templates = [
        "System telemetry report: node {n} operating at nominal frequency of {f} MHz with {p}% packet routing efficiency.\n",
        "Warehouse shipment manifest: batch {n} received {p} cartons transferred from depot {f} on schedule.\n",
        "Laboratory sensor reading: probe {n} recorded ambient humidity at {p}% with temperature delta {f} K.\n",
        "Network switch log entry: interface gigabit-{n} handled {p} incoming frames without cyclic redundancy errors.\n",
    ]

    # Target words (approx context_tokens // 1.3 words)
    target_words = int(context_tokens * 0.75)
    filler_lines = []
    word_count = 0
    line_idx = 0
    while word_count < target_words:
        tmpl = filler_templates[line_idx % len(filler_templates)]
        line = tmpl.format(n=line_idx, f=rng.randint(100, 900), p=rng.randint(10, 99))
        filler_lines.append(line)
        word_count += len(line.split())
        line_idx += 1

    # Insert needles evenly across the filler lines
    step = len(filler_lines) // (num_needles + 1)
    for i, (label, code) in enumerate(needles):
        pos = (i + 1) * step
        needle_line = f"\n*** CRITICAL ACCESS DATA: The secret code for {label} is {code}. ***\n\n"
        filler_lines.insert(pos + i, needle_line)

    context_body = "".join(filler_lines)
    # Formulate explicit multi-item query
    query_labels = ", ".join([lbl for lbl, _ in needles])
    question_text = (
        f"\nQUESTION_BEGIN\nWhat are the secret codes for {query_labels}? "
        f"List each record and its five-digit code in order, e.g. {needles[0][0]}: {needles[0][1]}."
    )

    full_prompt = (
        f"<|im_start|>system\nYou are a precise database query assistant.<|im_end|>\n"
        f"<|im_start|>user\n{context_body}{question_text}<|im_end|>\n<|im_start|>assistant\n"
    )

    return {
        "context_tokens": context_tokens,
        "num_needles": num_needles,
        "needles": dict(needles),
        "full_prompt": full_prompt,
        "context_body": context_body,
        "question_text": question_text,
    }


def evaluate_mniah_case(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    case: Dict,
    policy_name: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate multi-needle retrieval under a specific KV policy."""
    context_text = f"<|im_start|>system\nYou are a precise database query assistant.<|im_end|>\n<|im_start|>user\n{case['context_body']}"
    question_text = f"{case['question_text']}<|im_end|>\n<|im_start|>assistant\n"

    enc_context = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).to(device)
    enc_question = tokenizer(question_text, return_tensors="pt", add_special_tokens=False).to(device)

    L_ctx = enc_context.input_ids.shape[1]
    L_q = enc_question.input_ids.shape[1]

    with torch.no_grad():
        out_ctx = model(**enc_context, use_cache=True)
        pkv = out_ctx.past_key_values

    k_list = [pkv[i][0].clone() for i in range(len(pkv))]
    v_list = [pkv[i][1].clone() for i in range(len(pkv))]

    with torch.no_grad():
        emb = model.model.embed_tokens(enc_question.input_ids)
        H_q = model.config.num_attention_heads
        d_h = model.config.hidden_size // H_q
        q_question = model.model.layers[0].self_attn.q_proj(emb).view(1, L_q, H_q, d_h).transpose(1, 2)

    # Apply policy
    if policy_name == "fp16_reference":
        k_comp, v_comp, keep_indices = k_list, v_list, torch.arange(L_ctx, device=device)
    elif policy_name == "sink_recent_fp16":
        k_comp, v_comp, keep_indices, _ = apply_policy_sink_recent(k_list, v_list)
    elif policy_name == "question_aware_fp16":
        k_comp, v_comp, keep_indices, _ = apply_policy_question_aware(k_list, v_list, q_question)
    elif policy_name == "token_wise_int4":
        k_comp, v_comp, keep_indices, _ = apply_policy_token_wise_int4(k_list, v_list)
    elif policy_name == "channel_k_token_v_int4":
        k_comp, v_comp, keep_indices, _ = apply_policy_channel_k_token_v_int4(k_list, v_list)
    elif policy_name == "hybrid_fp16_int2":
        k_comp, v_comp, keep_indices, _ = apply_policy_hybrid(k_list, v_list, q_question)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    cache = DynamicCache()
    for layer_idx in range(len(k_comp)):
        cache.update(k_comp[layer_idx], v_comp[layer_idx], layer_idx)

    question_pos_ids = torch.arange(L_ctx, L_ctx + L_q, device=device).unsqueeze(0)

    with torch.no_grad():
        out_q = model(
            input_ids=enc_question.input_ids,
            position_ids=question_pos_ids,
            past_key_values=cache,
            use_cache=True,
        )

    # Decode up to 64 tokens
    curr_input = torch.argmax(out_q.logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens = [curr_input.item()]
    curr_cache = out_q.past_key_values
    curr_pos = L_ctx + L_q

    with torch.no_grad():
        for _ in range(63):
            out_gen = model(
                input_ids=curr_input,
                position_ids=torch.tensor([[curr_pos]], device=device),
                past_key_values=curr_cache,
                use_cache=True,
            )
            curr_input = torch.argmax(out_gen.logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = curr_input.item()
            generated_tokens.append(tok_id)
            if tok_id in [tokenizer.eos_token_id, 151643, 151645]:
                break
            curr_cache = out_gen.past_key_values
            curr_pos += 1

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Score how many of the K needles were correctly retrieved
    correct_needles = 0
    needles = case["needles"]
    for label, gold_code in needles.items():
        if gold_code in gen_text:
            correct_needles += 1

    recall = correct_needles / float(len(needles))
    return {
        "policy": policy_name,
        "context_tokens": case["context_tokens"],
        "num_needles": case["num_needles"],
        "retrieved_needles": correct_needles,
        "total_needles": len(needles),
        "needle_recall": recall,
        "gen_text": gen_text.strip(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Master Context Benchmark Runner (Option A + Option B)
# ──────────────────────────────────────────────────────────────────────────────

def run_context_benchmarks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute Option A (OOLONG with baseline check) and Option B (M-NIAH)."""
    results = {}

    print("\n" + "=" * 80)
    print("OPTION A: OOLONG (synth-with-labels) BASELINE CHECK")
    print("=" * 80)

    from datasets import load_dataset
    try:
        print("[Option A] Fetching oolongbench/oolong-synth validation split...")
        ds = load_dataset("oolongbench/oolong-synth", split="validation", streaming=True)
        sample_cases = []
        for row in ds:
            ctx = row.get("context_window_text_with_labels") or row.get("context_window_text") or ""
            q = row.get("question", "")
            prompt = f"{ctx}\n\nQuestion: {q}"
            if len(prompt.split()) <= 3500:
                sample_cases.append({
                    "prompt": prompt,
                    "answer": row.get("answer", ""),
                    "answer_type": row.get("answer_type", "list"),
                    "task": row.get("task", ""),
                })
            if len(sample_cases) >= 10:
                break
    except Exception as e:
        print(f"[Option A] Could not load oolongbench/oolong-synth ({e}). Skipping Option A.")
        sample_cases = []

    option_a_results = {"status": "skipped", "reason": "dataset unavailable"}

    if sample_cases:
        from task2.eval_oolong import evaluate_oolong_case
        print(f"[Option A] Evaluating uncompressed FP16 baseline on {len(sample_cases)} sample cases...")
        base_scores = []
        for case in sample_cases:
            res = evaluate_oolong_case(model, tokenizer, case, "fp16_reference", device=device)
            base_scores.append(res["score"])

        mean_base_score = float(np.mean(base_scores))
        print(f"[Option A] Base FP16 Score: {mean_base_score*100:.1f}%")

        # Baseline gate check: if baseline is < 20%, abort Option A due to floor effect
        if mean_base_score < 0.20:
            print(f"[Option A] ⚠️ Baseline FP16 score ({mean_base_score*100:.1f}%) is too low for 1.5B scale.")
            print("[Option A] Floor effect detected: not worth evaluating compressed policies on noise.")
            option_a_results = {
                "status": "aborted_floor_effect",
                "base_fp16_score": mean_base_score,
                "conclusion": "Model capacity floor effect: 1.5B model fails atomic aggregation even in FP16.",
            }
        else:
            print("[Option A] ✓ Baseline score viable! Running compressed KV cache policies...")
            policies = ["sink_recent_fp16", "question_aware_fp16", "token_wise_int4", "channel_k_token_v_int4", "hybrid_fp16_int2"]
            pol_scores = {"fp16_reference": mean_base_score}
            for p in policies:
                scores = []
                for case in sample_cases:
                    res = evaluate_oolong_case(model, tokenizer, case, p, device=device)
                    scores.append(res["score"])
                pol_scores[p] = float(np.mean(scores))
                print(f"  [{p:<25}] Score: {pol_scores[p]*100:.1f}%")
            option_a_results = {"status": "completed", "policy_scores": pol_scores}

    results["option_a_oolong"] = option_a_results

    # ──────────────────────────────────────────────────────────────────────────
    # Option B: Multi-Needle in a Haystack (M-NIAH)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("OPTION B: MULTI-NEEDLE IN A HAYSTACK (M-NIAH) BENCHMARK")
    print("=" * 80)

    mniah_cases = []
    for L in [2048, 4096]:
        for k in [2, 4, 8]:
            for rep in range(3):
                case = generate_mniah_prompt(context_tokens=L, num_needles=k, seed=6304 + L + k + rep)
                mniah_cases.append(case)

    print(f"[Option B] Generated {len(mniah_cases)} M-NIAH cases across lengths [2048, 4096] and loads K in [2, 4, 8].")

    policies = [
        "fp16_reference",
        "sink_recent_fp16",
        "question_aware_fp16",
        "token_wise_int4",
        "channel_k_token_v_int4",
        "hybrid_fp16_int2",
    ]

    mniah_policy_recalls = {p: [] for p in policies}

    for idx, case in enumerate(mniah_cases):
        if idx % 3 == 0:
            print(f"Evaluating M-NIAH Case {idx+1}/{len(mniah_cases)} [Len: {case['context_tokens']}, K: {case['num_needles']}]")
        for p in policies:
            res = evaluate_mniah_case(model, tokenizer, case, p, device=device)
            mniah_policy_recalls[p].append(res["needle_recall"])

    mniah_summary = {}
    for p in policies:
        mean_rec = float(np.mean(mniah_policy_recalls[p]))
        mniah_summary[p] = {"mean_needle_recall": mean_rec}
        print(f"  [{p:<25}] Multi-Needle Recall: {mean_rec*100:5.1f}%")

    results["option_b_mniah"] = {
        "status": "completed",
        "summary": mniah_summary,
    }

    # Save to disk
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LOCAL_RESULTS_DIR / "context_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved Context Benchmarking results to {out_file}")

    cloud_out = Path("/root/volume/checkpoints/task2/context_benchmarks.json")
    if cloud_out.parent.exists():
        with open(cloud_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved Context Benchmarking results to cloud volume at {cloud_out}")

    return results
