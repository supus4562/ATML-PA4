"""task2/eval_context_benchmark.py — Mini-RULER Context Memory & Reasoning Suite.
Includes:
1. Single-Needle in a Haystack (NIAH, K=1) across depths (10%, 50%, 90%) at 2k & 4k tokens.
2. Multi-Needle in a Haystack (M-NIAH, K in {2, 4, 8}) across 2k & 4k tokens.
3. Variable Tracking (VT, Multi-hop variable assignment chains) from RULER benchmark.
4. Test-Time RoPE Base Frequency Scaling Study (theta in {10^4, 10^5, 10^6, 10^7}).
5. Option A: OOLONG (synth-with-labels) Baseline Gating Check.
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
from task2.kv_core import build_cache, greedy_decode, policy_tail_state, prefill_question, question_importance
from task2.kv_policies import apply_policy
from task2.kv_policies import (
    apply_policy_sink_recent,
    apply_policy_question_aware,
    apply_policy_token_wise_int4,
    apply_policy_channel_k_token_v_int4,
    apply_policy_hybrid,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Single-Needle in a Haystack (NIAH K=1) Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_single_needle_prompt(
    context_tokens: int = 2048,
    depth_ratio: float = 0.5,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Construct a Single-Needle prompt with one key-code needle placed at specified depth."""
    rng = random.Random(seed)
    code = f"{rng.randint(10000, 99999)}"
    target_system = f"SYSTEM-{chr(65 + rng.randint(0, 25))}"

    filler_templates = [
        "System telemetry report: node {n} operating at nominal frequency of {f} MHz with {p}% packet routing efficiency.\n",
        "Warehouse shipment manifest: batch {n} received {p} cartons transferred from depot {f} on schedule.\n",
        "Laboratory sensor reading: probe {n} recorded ambient humidity at {p}% with temperature delta {f} K.\n",
        "Network switch log entry: interface gigabit-{n} handled {p} incoming frames without cyclic redundancy errors.\n",
    ]

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

    # Insert needle at specified depth
    insert_idx = max(0, min(len(filler_lines), int(len(filler_lines) * depth_ratio)))
    needle_line = f"\n*** CRITICAL SECURITY RECORD: The master authorization passcode for {target_system} is {code}. ***\n\n"
    filler_lines.insert(insert_idx, needle_line)

    context_body = "".join(filler_lines)
    question_text = (
        f"\nQUESTION_BEGIN\nWhat is the master authorization passcode for {target_system}? "
        f"Answer with only the five-digit code."
    )

    return {
        "task": "single_needle_niah",
        "context_tokens": context_tokens,
        "depth_ratio": depth_ratio,
        "target_system": target_system,
        "gold_code": code,
        "context_body": context_body,
        "question_text": question_text,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-Needle in a Haystack (M-NIAH K in {2, 4, 8}) Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_mniah_prompt(
    context_tokens: int = 2048,
    num_needles: int = 4,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Construct a Multi-Needle prompt with K distinct keys and codes distributed across context."""
    rng = random.Random(seed)
    needles = []
    for i in range(num_needles):
        code = f"{rng.randint(10000, 99999)}"
        label = f"RECORD-{chr(65 + i)}"
        needles.append((label, code))

    filler_templates = [
        "System telemetry report: node {n} operating at nominal frequency of {f} MHz with {p}% packet routing efficiency.\n",
        "Warehouse shipment manifest: batch {n} received {p} cartons transferred from depot {f} on schedule.\n",
        "Laboratory sensor reading: probe {n} recorded ambient humidity at {p}% with temperature delta {f} K.\n",
        "Network switch log entry: interface gigabit-{n} handled {p} incoming frames without cyclic redundancy errors.\n",
    ]

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

    # Insert needles evenly across filler lines
    step = len(filler_lines) // (num_needles + 1)
    for i, (label, code) in enumerate(needles):
        pos = (i + 1) * step
        needle_line = f"\n*** CRITICAL ACCESS DATA: The secret code for {label} is {code}. ***\n\n"
        filler_lines.insert(pos + i, needle_line)

    context_body = "".join(filler_lines)
    query_labels = ", ".join([lbl for lbl, _ in needles])
    question_text = (
        f"\nQUESTION_BEGIN\nWhat are the secret codes for {query_labels}? "
        f"List each record and its five-digit code in order, e.g. {needles[0][0]}: {needles[0][1]}."
    )

    return {
        "task": "multi_needle_mniah",
        "context_tokens": context_tokens,
        "num_needles": num_needles,
        "needles": dict(needles),
        "context_body": context_body,
        "question_text": question_text,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Variable Tracking (VT from RULER) Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_variable_tracking_prompt(
    context_tokens: int = 2048,
    num_hops: int = 3,
    seed: int = 6304,
) -> Dict[str, Any]:
    """Construct a RULER Variable Tracking task: chained variable assignments with distractors."""
    rng = random.Random(seed)
    gold_val = f"{rng.randint(10000, 99999)}"

    # Target variable chain: V0 = gold_val, V1 = V0, V2 = V1, ...
    chain_vars = [f"var_{chr(65 + i)}" for i in range(num_hops + 1)]
    target_chain = [f"{chain_vars[0]} = {gold_val}"]
    for h in range(num_hops):
        target_chain.append(f"{chain_vars[h+1]} = {chain_vars[h]}")

    # Distractor chains
    distractor_chains = []
    for d in range(4):
        d_val = f"{rng.randint(10000, 99999)}"
        d_vars = [f"aux_{d}_{chr(80 + j)}" for j in range(num_hops + 1)]
        d_chain = [f"{d_vars[0]} = {d_val}"]
        for h in range(num_hops):
            d_chain.append(f"{d_vars[h+1]} = {d_vars[h]}")
        distractor_chains.append(d_chain)

    # Causal topological ordering: Stage 0 (initializations), then Stage 1 (hop 1), etc.
    all_assigns = []
    num_stages = num_hops + 1
    for stage in range(num_stages):
        stage_assigns = [(True, target_chain[stage])]
        for d_chain in distractor_chains:
            stage_assigns.append((False, d_chain[stage]))
        rng.shuffle(stage_assigns)
        all_assigns.extend(stage_assigns)

    # Filler text bank
    filler_templates = [
        "System telemetry report: node {n} operating at nominal frequency of {f} MHz with {p}% packet routing efficiency.\n",
        "Warehouse shipment manifest: batch {n} received {p} cartons transferred from depot {f} on schedule.\n",
        "Laboratory sensor reading: probe {n} recorded ambient humidity at {p}% with temperature delta {f} K.\n",
    ]

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

    # Interleave assignments into filler lines
    step = max(1, len(filler_lines) // (len(all_assigns) + 1))
    for i, (_, assign_str) in enumerate(all_assigns):
        pos = min(len(filler_lines), (i + 1) * step)
        line = f"\n>>> SCRIPT DECLARATION: {assign_str} <<<\n"
        filler_lines.insert(pos, line)

    context_body = "".join(filler_lines)
    query_var = chain_vars[-1]
    question_text = (
        f"\nQUESTION_BEGIN\nFollowing the declarations in the text, what is the final numeric value of {query_var}? "
        f"Answer with only the five-digit integer."
    )

    return {
        "task": "variable_tracking",
        "context_tokens": context_tokens,
        "num_hops": num_hops,
        "query_var": query_var,
        "gold_val": gold_val,
        "context_body": context_body,
        "question_text": question_text,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generic Context Case Evaluator with True SnapKV Attention
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_context_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    case: Dict[str, Any],
    policy_name: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute prefill, KV compression, question prefill, and greedy generation."""
    context_text = f"<|im_start|>system\nYou are an exact data extraction assistant.<|im_end|>\n<|im_start|>user\n{case['context_body']}"
    question_text = f"{case['question_text']}<|im_end|>\n<|im_start|>assistant\n"

    enc_context = tokenizer(context_text, return_tensors="pt", add_special_tokens=False).to(device)
    enc_question = tokenizer(question_text, return_tensors="pt", add_special_tokens=False).to(device)

    L_ctx = enc_context.input_ids.shape[1]
    L_q = enc_question.input_ids.shape[1]

    # Prefill context (FP16 SDPA model), then apply the shared Part C policy code.
    with torch.no_grad():
        out_ctx = model(**enc_context, use_cache=True)
    n_layers = len(model.model.layers)
    k_list = [out_ctx.past_key_values.layers[i].keys.clone() for i in range(n_layers)]
    v_list = [out_ctx.past_key_values.layers[i].values.clone() for i in range(n_layers)]
    del out_ctx

    q_importance = None
    if policy_name in ["question_aware_fp16", "hybrid_fp16_int2"]:
        q_importance, _ = question_importance(model, k_list, v_list, enc_question.input_ids, L_ctx)

    res = apply_policy(policy_name, k_list, v_list, q_importance)
    state = policy_tail_state(res)
    cache = build_cache(res.k, res.v)
    first = prefill_question(model, cache, enc_question.input_ids, L_ctx, state)
    generated_tokens, _ = greedy_decode(model, cache, first, L_ctx + L_q, state, max_new_tokens=48)

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Determine correctness based on task type
    task_type = case.get("task", "mniah")
    if task_type == "single_needle_niah":
        gold = case["gold_code"]
        correct = (gold in gen_text)
        return {
            "policy": policy_name,
            "task": task_type,
            "context_tokens": case["context_tokens"],
            "depth_ratio": case["depth_ratio"],
            "gold": gold,
            "gen_text": gen_text,
            "score": 1.0 if correct else 0.0,
        }
    elif task_type == "variable_tracking":
        gold = case["gold_val"]
        matches = re.findall(r"\b\d{5}\b", gen_text)
        correct = (gold in gen_text) or (bool(matches) and gold in matches)
        return {
            "policy": policy_name,
            "task": task_type,
            "context_tokens": case["context_tokens"],
            "num_hops": case["num_hops"],
            "gold": gold,
            "gen_text": gen_text,
            "score": 1.0 if correct else 0.0,
        }
    else:  # multi_needle_mniah
        needles = case["needles"]
        retrieved = sum(1 for _, code in needles.items() if code in gen_text)
        recall = retrieved / float(len(needles))
        return {
            "policy": policy_name,
            "task": task_type,
            "context_tokens": case["context_tokens"],
            "num_needles": case["num_needles"],
            "retrieved_needles": retrieved,
            "total_needles": len(needles),
            "score": recall,
            "gen_text": gen_text,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Test-Time RoPE Base Scaling Study Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_rope_frequency_study(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate zero-shot long-context retrieval at L=4096 across RoPE theta_base in {10^4, 10^5, 10^6, 10^7}."""
    print("\n" + "=" * 80)
    print("ROPE CONTEXT STUDY: TEST-TIME BASE FREQUENCY SCALING (ZERO-SHOT)")
    print("=" * 80)

    # Check rotary embedding module
    rotary_module = None
    if hasattr(model.model, "rotary_emb"):
        rotary_module = model.model.rotary_emb
    elif hasattr(model.model.layers[0].self_attn, "rotary_emb"):
        rotary_module = model.model.layers[0].self_attn.rotary_emb

    if rotary_module is None or not hasattr(rotary_module, "base"):
        print("[RoPE Study] Rotary embedding module does not expose configurable base frequency. Skipping.")
        return {"status": "unsupported_architecture"}

    orig_base = rotary_module.base
    orig_inv_freq = rotary_module.inv_freq.clone() if hasattr(rotary_module, "inv_freq") else None

    # Test cases: single-needle cases at 4096 tokens across depths
    test_cases = [
        generate_single_needle_prompt(context_tokens=4096, depth_ratio=d, seed=6304 + int(d * 100))
        for d in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]

    theta_values = [1e4, 1e5, 1e6, 1e7]
    rope_results = {}

    try:
        dim = rotary_module.dim
        for theta in theta_values:
            # Dynamically update base frequency and inverse frequencies
            rotary_module.base = float(theta)
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            rotary_module.inv_freq = inv_freq

            scores = []
            for c in test_cases:
                res = evaluate_context_prompt(model, tokenizer, c, "fp16_reference", device=device)
                scores.append(res["score"])

            mean_acc = float(np.mean(scores))
            rope_results[f"theta_{int(theta):.0e}"] = {
                "theta_base": float(theta),
                "accuracy": mean_acc,
                "n_cases": len(scores),
            }
            print(f"  [RoPE theta={theta:<8.0e}] L=4096 Retrieval Acc: {mean_acc*100:5.1f}%")

    finally:
        # Restore original base and frequencies
        rotary_module.base = orig_base
        if orig_inv_freq is not None:
            rotary_module.inv_freq = orig_inv_freq

    return rope_results


# ──────────────────────────────────────────────────────────────────────────────
# Master Context Benchmark Runner (Mini-RULER Suite)
# ──────────────────────────────────────────────────────────────────────────────

def run_context_benchmarks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Execute complete Mini-RULER Suite: NIAH K=1, M-NIAH, Variable Tracking, and RoPE Study."""
    results = {}
    policies = [
        "fp16_reference",
        "sink_recent_fp16",
        "question_aware_fp16",
        "token_wise_int4",
        "channel_k_token_v_int4",
        "hybrid_fp16_int2",
    ]

    # 1. Single-Needle NIAH (K=1)
    print("\n" + "=" * 80)
    print("MINI-RULER TASK 1: SINGLE-NEEDLE IN A HAYSTACK (NIAH K=1)")
    print("=" * 80)

    niah_cases = []
    for L in [2048, 4096]:
        for depth in [0.1, 0.5, 0.9]:
            niah_cases.append(generate_single_needle_prompt(context_tokens=L, depth_ratio=depth, seed=6304 + L + int(depth*10)))

    niah_policy_scores = {p: [] for p in policies}
    for idx, case in enumerate(niah_cases):
        print(f"Evaluating Single-Needle Case {idx+1}/{len(niah_cases)} [Len: {case['context_tokens']}, Depth: {case['depth_ratio']*100:.0f}%]")
        for p in policies:
            res = evaluate_context_prompt(model, tokenizer, case, p, device=device)
            niah_policy_scores[p].append(res["score"])

    niah_summary = {}
    for p in policies:
        mean_s = float(np.mean(niah_policy_scores[p]))
        niah_summary[p] = {"mean_accuracy": mean_s}
        print(f"  [{p:<25}] Single-Needle Acc: {mean_s*100:5.1f}%")

    results["single_needle_niah"] = niah_summary

    # 2. Multi-Needle M-NIAH (K in {2, 4, 8})
    print("\n" + "=" * 80)
    print("MINI-RULER TASK 2: MULTI-NEEDLE IN A HAYSTACK (M-NIAH K in {2, 4, 8})")
    print("=" * 80)

    mniah_cases = []
    for L in [2048, 4096]:
        for k in [2, 4, 8]:
            mniah_cases.append(generate_mniah_prompt(context_tokens=L, num_needles=k, seed=6304 + L + k))

    mniah_policy_scores = {p: [] for p in policies}
    for idx, case in enumerate(mniah_cases):
        print(f"Evaluating Multi-Needle Case {idx+1}/{len(mniah_cases)} [Len: {case['context_tokens']}, K: {case['num_needles']}]")
        for p in policies:
            res = evaluate_context_prompt(model, tokenizer, case, p, device=device)
            mniah_policy_scores[p].append(res["score"])

    mniah_summary = {}
    for p in policies:
        mean_s = float(np.mean(mniah_policy_scores[p]))
        mniah_summary[p] = {"mean_needle_recall": mean_s}
        print(f"  [{p:<25}] Multi-Needle Recall: {mean_s*100:5.1f}%")

    results["multi_needle_mniah"] = mniah_summary

    # 3. Variable Tracking (VT from RULER)
    print("\n" + "=" * 80)
    print("MINI-RULER TASK 3: VARIABLE TRACKING (MULTI-HOP REASONING)")
    print("=" * 80)

    vt_cases = []
    for L in [2048, 4096]:
        for hops in [2, 3]:
            vt_cases.append(generate_variable_tracking_prompt(context_tokens=L, num_hops=hops, seed=6304 + L + hops))

    vt_policy_scores = {p: [] for p in policies}
    for idx, case in enumerate(vt_cases):
        print(f"Evaluating Variable Tracking Case {idx+1}/{len(vt_cases)} [Len: {case['context_tokens']}, Hops: {case['num_hops']}]")
        for p in policies:
            res = evaluate_context_prompt(model, tokenizer, case, p, device=device)
            vt_policy_scores[p].append(res["score"])

    vt_summary = {}
    for p in policies:
        mean_s = float(np.mean(vt_policy_scores[p]))
        vt_summary[p] = {"mean_accuracy": mean_s}
        print(f"  [{p:<25}] Variable Tracking Acc: {mean_s*100:5.1f}%")

    results["variable_tracking"] = vt_summary

    # 4. RoPE Frequency Study
    rope_summary = run_rope_frequency_study(model=model, tokenizer=tokenizer, device=device)
    results["rope_frequency_study"] = rope_summary

    # Save to disk
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LOCAL_RESULTS_DIR / "context_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved Complete Mini-RULER Context Benchmarking results to {out_file}")

    cloud_out = Path("/root/volume/checkpoints/task2/context_benchmarks.json")
    if cloud_out.parent.exists():
        with open(cloud_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to cloud volume at {cloud_out}")

    return results
