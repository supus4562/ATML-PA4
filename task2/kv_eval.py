"""task2/kv_eval.py — Complete evaluation suite for Task 2 Part C (Pretrained KV Memory Compression).
"""

from __future__ import annotations

import gc
import gzip
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from task2.config import (
    REPO_ROOT,
    KV_MODEL_NAME,
    KV_BOOTSTRAP_RESAMPLES,
    KV_BENCHMARK_REPETITIONS,
    KV_BENCHMARK_BATCHES,
)
from task2.kv_cache_models import (
    compute_matched_int4_bytes,
    compute_uncompressed_fp16_bytes,
)
from task2.kv_policies import (
    apply_policy_sink_recent,
    apply_policy_question_aware,
    apply_policy_token_wise_int4,
    apply_policy_channel_k_token_v_int4,
    apply_policy_hybrid,
)


def load_retrieval_prompts() -> List[Dict]:
    """Load the 180 factorial retrieval prompt cases from data/kv_retrieval/kv_retrieval_cases.jsonl.gz."""
    prompts_path = REPO_ROOT / "data" / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Required prompt file not found: {prompts_path}")

    records = []
    with gzip.open(prompts_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_five_digit_code(text: str) -> Optional[str]:
    """Deterministic extractor for the 5-digit passcode answer."""
    matches = re.findall(r"\b\d{5}\b", text)
    return matches[0] if matches else None


def compute_paired_bootstrap_ci(
    diffs: List[float],
    b_resamples: int = KV_BOOTSTRAP_RESAMPLES,
    seed: int = 6304,
) -> Dict[str, float]:
    """Compute 95% paired bootstrap confidence interval."""
    if not diffs:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.RandomState(seed)
    n = len(diffs)
    arr = np.array(diffs, dtype=np.float64)
    boot_means = np.empty(b_resamples, dtype=np.float64)
    for i in range(b_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
    }


def evaluate_kv_prompt_case(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_record: Dict,
    policy_name: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate a single long-context prompt under a specific cache policy with RoPE-preserved greedy decode."""
    user_text = prompt_record.get("user_text", prompt_record.get("prompt", ""))
    gold_code = str(prompt_record.get("code", prompt_record.get("target_code", "")))
    evidence_text = prompt_record.get("evidence_text", "")

    # Format user text through the official chat template
    full_chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Split into context body and question suffix at QUESTION_BEGIN
    if "QUESTION_BEGIN" in full_chat:
        ctx_part, q_part = full_chat.split("QUESTION_BEGIN", 1)
        q_part = "QUESTION_BEGIN" + q_part
    else:
        split_idx = max(0, len(full_chat) - 200)
        ctx_part = full_chat[:split_idx]
        q_part = full_chat[split_idx:]

    enc_context = tokenizer(ctx_part, return_tensors="pt", add_special_tokens=False).to(device)
    enc_question = tokenizer(q_part, return_tensors="pt", add_special_tokens=False).to(device)

    L_ctx = enc_context.input_ids.shape[1]
    L_q = enc_question.input_ids.shape[1]

    # 1. Prefill context through model to extract initial past_key_values
    with torch.no_grad():
        out_ctx = model(**enc_context, use_cache=True)
        pkv = out_ctx.past_key_values

    # In modern transformers, DynamicCache is used: unpack layers into k_list and v_list
    k_list = []
    v_list = []
    for layer_idx in range(len(pkv)):
        # DynamicCache can be indexed by layer_idx
        k_list.append(pkv[layer_idx][0].clone())
        v_list.append(pkv[layer_idx][1].clone())

    # Get query projection of question for question-aware policies
    # Pass question through first layer or embedding to get query states
    with torch.no_grad():
        emb = model.model.embed_tokens(enc_question.input_ids)
        # Approximate question queries using first self-attention q_proj
        q_question = model.model.layers[0].self_attn.q_proj(emb)  # (1, L_q, H_q * d_h)
        H_q = model.config.num_attention_heads
        d_h = model.config.hidden_size // H_q
        q_question = q_question.view(1, L_q, H_q, d_h).transpose(1, 2)  # (1, H_q, L_q, d_h)

    # 2. Apply chosen compression/eviction policy to context cache
    if policy_name == "fp16_reference":
        k_comp = k_list
        v_comp = v_list
        keep_indices = torch.arange(L_ctx, device=device)
        actual_bytes = compute_uncompressed_fp16_bytes(L_ctx)
    elif policy_name == "sink_recent_fp16":
        k_comp, v_comp, keep_indices, actual_bytes = apply_policy_sink_recent(k_list, v_list)
    elif policy_name == "question_aware_fp16":
        k_comp, v_comp, keep_indices, actual_bytes = apply_policy_question_aware(k_list, v_list, q_question)
    elif policy_name == "token_wise_int4":
        k_comp, v_comp, keep_indices, actual_bytes = apply_policy_token_wise_int4(k_list, v_list)
    elif policy_name == "channel_k_token_v_int4":
        k_comp, v_comp, keep_indices, actual_bytes = apply_policy_channel_k_token_v_int4(k_list, v_list)
    elif policy_name == "hybrid_fp16_int2":
        k_comp, v_comp, keep_indices, actual_bytes = apply_policy_hybrid(k_list, v_list, q_question)
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Calculate evidence retention fraction
    # Calculate evidence retention fraction
    if evidence_text:
        ev_ids = tokenizer(evidence_text, add_special_tokens=False)["input_ids"]
        ctx_ids = enc_context.input_ids[0].tolist()
        ev_len = len(ev_ids)
        ev_start = -1
        for i in range(len(ctx_ids) - ev_len + 1):
            if ctx_ids[i : i + ev_len] == ev_ids:
                ev_start = i
                break
        if ev_start != -1:
            gold_positions = set(range(ev_start, ev_start + ev_len))
            retained_positions = set(keep_indices.cpu().numpy().tolist())
            retained_fraction = len(gold_positions.intersection(retained_positions)) / float(len(gold_positions))
        else:
            retained_fraction = 1.0
    else:
        retained_fraction = 1.0

    # 3. Prefill question suffix on top of compressed cache
    from transformers.cache_utils import DynamicCache
    compressed_cache = DynamicCache()
    for layer_idx in range(len(k_comp)):
        compressed_cache.update(k_comp[layer_idx], v_comp[layer_idx], layer_idx)

    # Question position IDs: continue from context sequence length to maintain absolute RoPE alignment
    past_len = keep_indices.shape[0]
    question_pos_ids = torch.arange(L_ctx, L_ctx + L_q, device=device).unsqueeze(0)

    with torch.no_grad():
        out_q = model(
            input_ids=enc_question.input_ids,
            position_ids=question_pos_ids,
            past_key_values=compressed_cache,
            use_cache=True,
        )
        first_token_logits = out_q.logits[:, -1, :].clone()

    # 4. Autoregressive greedy decode for 16 tokens to extract 5-digit code
    curr_input = torch.argmax(first_token_logits, dim=-1, keepdim=True)
    generated_tokens = [curr_input.item()]
    curr_cache = out_q.past_key_values
    curr_pos = L_ctx + L_q

    with torch.no_grad():
        for _ in range(15):
            out_gen = model(
                input_ids=curr_input,
                position_ids=torch.tensor([[curr_pos]], device=device),
                past_key_values=curr_cache,
                use_cache=True,
            )
            curr_input = torch.argmax(out_gen.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(curr_input.item())
            curr_cache = out_gen.past_key_values
            curr_pos += 1

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    pred_code = extract_five_digit_code(gen_text)
    is_correct = (pred_code == gold_code)

    # 5. Answer NLL: compute log probability of the true gold answer tokens
    gold_ids = tokenizer(gold_code, add_special_tokens=False)["input_ids"]
    answer_nll = 0.0
    # Evaluate NLL using first token logits
    if gold_ids:
        log_probs = F.log_softmax(first_token_logits, dim=-1)
        answer_nll = -log_probs[0, gold_ids[0]].item()

    return {
        "prompt_id": prompt_record.get("id", "case"),
        "policy": policy_name,
        "is_correct": is_correct,
        "predicted_code": pred_code,
        "gold_code": gold_code,
        "answer_nll": answer_nll,
        "retained_evidence_fraction": retained_fraction,
        "actual_packed_bytes": actual_bytes,
        "first_logits": first_token_logits[0].cpu().half(),
        "context_len": prompt_record.get("target_tokens", L_ctx),
        "evidence_position": prompt_record.get("position", "middle"),
        "distractors": prompt_record.get("distractors", 0),
    }


def run_full_kv_evaluation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Run evaluation across all 180 prompts and 5 policies + FP16 baseline."""
    prompts = load_retrieval_prompts()
    if max_cases is not None:
        prompts = prompts[:max_cases]

    policies = [
        "fp16_reference",
        "sink_recent_fp16",
        "question_aware_fp16",
        "token_wise_int4",
        "channel_k_token_v_int4",
        "hybrid_fp16_int2",
    ]

    print(f"\n{'='*80}")
    print(f"PART C: EVALUATING PRETRAINED KV CACHE COMPRESSION ({len(prompts)} cases)")
    print(f"{'='*80}\n")

    policy_results = {p: [] for p in policies}

    for idx, prompt in enumerate(prompts):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Evaluating prompt {idx + 1:3d}/{len(prompts)} [Len: {prompt.get('target_tokens', 2048)} | Pos: {prompt.get('position', 'middle')} | Distr: {prompt.get('distractors', 0)}]")

        # Evaluate reference first to collect FP16 logits for KL divergence
        ref_res = evaluate_kv_prompt_case(model, tokenizer, prompt, "fp16_reference", device=device)
        policy_results["fp16_reference"].append(ref_res)
        ref_logits = ref_res["first_logits"].float()
        ref_probs = F.softmax(ref_logits, dim=-1)

        # Evaluate compressed policies
        for p in policies[1:]:
            res = evaluate_kv_prompt_case(model, tokenizer, prompt, p, device=device)
            # Compute logit KL divergence from FP16 reference
            comp_logits = res["first_logits"].float()
            comp_log_probs = F.log_softmax(comp_logits, dim=-1)
            kl = F.kl_div(comp_log_probs, ref_probs, reduction="sum").item()
            res["logit_kl"] = kl
            policy_results[p].append(res)

    # Aggregate metrics per policy
    summary = {}
    ref_correct = [r["is_correct"] for r in policy_results["fp16_reference"]]

    for p in policies:
        cases = policy_results[p]
        acc = np.mean([r["is_correct"] for r in cases])
        nll = np.mean([r["answer_nll"] for r in cases])
        retained = np.mean([r["retained_evidence_fraction"] for r in cases])
        bytes_avg = np.mean([r["actual_packed_bytes"] for r in cases])
        kl_avg = np.mean([r.get("logit_kl", 0.0) for r in cases])

        # Conditional accuracy on FP16 reference correctness
        corr_indices = [i for i, corr in enumerate(ref_correct) if corr]
        cond_acc = np.mean([cases[i]["is_correct"] for i in corr_indices]) if corr_indices else 0.0

        summary[p] = {
            "exact_match_accuracy": float(acc),
            "conditional_accuracy": float(cond_acc),
            "mean_answer_nll": float(nll),
            "mean_logit_kl": float(kl_avg),
            "mean_retained_evidence": float(retained),
            "mean_packed_mb": float(bytes_avg / (1024 * 1024)),
        }
        print(
            f"[{p:<25}] Acc: {acc*100:5.1f}% (Cond: {cond_acc*100:5.1f}%) | "
            f"NLL: {nll:5.2f} | KL: {kl_avg:5.2f} | Evidence: {retained*100:5.1f}% | MB: {bytes_avg/(1024*1024):5.1f}"
        )

    # Compute paired bootstrap confidence intervals against Question-Aware and KIVI
    bootstrap_results = {}
    for comp_p in ["sink_recent_fp16", "token_wise_int4", "channel_k_token_v_int4", "hybrid_fp16_int2"]:
        diffs_acc = [
            float(policy_results[comp_p][i]["is_correct"]) - float(policy_results["question_aware_fp16"][i]["is_correct"])
            for i in range(len(prompts))
        ]
        diffs_nll = [
            policy_results[comp_p][i]["answer_nll"] - policy_results["question_aware_fp16"][i]["answer_nll"]
            for i in range(len(prompts))
        ]
        bootstrap_results[f"{comp_p}_vs_question_aware_acc"] = compute_paired_bootstrap_ci(diffs_acc)
        bootstrap_results[f"{comp_p}_vs_question_aware_nll"] = compute_paired_bootstrap_ci(diffs_nll)

    return {
        "summary": summary,
        "bootstrap": bootstrap_results,
        "raw_cases": {p: [{k: v for k, v in r.items() if k != "first_logits"} for r in policy_results[p]] for p in policies},
    }
