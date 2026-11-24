"""task2/kv_eval.py — Task 2 Part C evaluation (C1 validation, C2–C4 policies, C5 metrics).

Model: Qwen/Qwen2.5-1.5B-Instruct, FP16, SDPA attention (the working generation path).
Reference answer: the unchanged FP16 model's own greedy answer tokens for that prompt.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from task2.config import REPO_ROOT, KV_BOOTSTRAP_RESAMPLES
from task2.kv_cache_models import compute_matched_int4_bytes, compute_uncompressed_fp16_bytes
from task2.kv_core import (
    EOS_IDS,
    build_cache,
    greedy_decode,
    policy_tail_state,
    prefill_question,
    question_importance,
    teacher_forced_logprobs,
)
from task2.kv_policies import POLICIES, apply_policy

MAX_NEW_TOKENS = 48


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")


def load_retrieval_prompts() -> List[Dict]:
    prompts_path = REPO_ROOT / "data" / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz"
    with gzip.open(prompts_path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_five_digit_code(text: str) -> Optional[str]:
    """Deterministic extractor: first standalone 5-digit number in the answer."""
    m = re.findall(r"\b\d{5}\b", text)
    return m[0] if m else None


def paired_bootstrap(diffs: List[float], b: int = KV_BOOTSTRAP_RESAMPLES, seed: int = 6304) -> Dict[str, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(b, arr.size))].mean(axis=1)
    return {"mean": float(arr.mean()), "ci_lower": float(np.percentile(boots, 2.5)),
            "ci_upper": float(np.percentile(boots, 97.5)), "n": int(arr.size), "resamples": b}


def split_prompt(tokenizer, record: Dict):
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": record["user_text"]}], tokenize=False, add_generation_prompt=True
    )
    if "QUESTION_BEGIN" in chat:
        ctx, q = chat.split("QUESTION_BEGIN", 1)
        q = "QUESTION_BEGIN" + q
    else:
        raise ValueError(f"Prompt {record.get('id')} has no QUESTION_BEGIN marker")
    return ctx, q


def evidence_token_positions(tokenizer, ctx_text: str, evidence_text: str) -> List[int]:
    start = ctx_text.find(evidence_text)
    if start < 0:
        return []
    end = start + len(evidence_text)
    offsets = tokenizer(ctx_text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return [i for i, (s, e) in enumerate(offsets) if max(s, start) < min(e, end)]


@torch.no_grad()
def validate_fp16_reference(model, tokenizer, records: List[Dict], device: str = "cuda") -> Dict[str, Any]:
    """C1: split-prefill equivalence, 100%-retention equivalence, finite logits."""
    rows = []
    for rec in records:
        ctx, q = split_prompt(tokenizer, rec)
        c_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        q_ids = tokenizer(q, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        L = c_ids.shape[1]
        full_ids = torch.cat([c_ids, q_ids], dim=1)

        full_logits = model(input_ids=full_ids).logits[0, -1].float()
        out_ctx = model(input_ids=c_ids, use_cache=True)
        k_list = [out_ctx.past_key_values.layers[i].keys for i in range(len(model.model.layers))]
        v_list = [out_ctx.past_key_values.layers[i].values for i in range(len(model.model.layers))]
        split_logits = prefill_question(model, build_cache(k_list, v_list), q_ids, L, {"layout": None})[0]

        # Ordinary HF greedy generation on the full prompt vs our edited-cache path at 100% retention.
        gen = model.generate(full_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             eos_token_id=list(EOS_IDS), pad_token_id=EOS_IDS[0])
        hf_tokens = gen[0, full_ids.shape[1]:].tolist()
        cache = build_cache(k_list, v_list)
        first = prefill_question(model, cache, q_ids, L, {"layout": None})
        ours, _ = greedy_decode(model, cache, first, L + q_ids.shape[1], {"layout": None}, MAX_NEW_TOKENS)
        rows.append({
            "id": rec["id"],
            "max_abs_logit_diff_full_vs_split": float((full_logits - split_logits).abs().max()),
            "argmax_equal": bool(full_logits.argmax() == split_logits.argmax()),
            "all_logits_finite": bool(torch.isfinite(full_logits).all() and torch.isfinite(split_logits).all()),
            "full_retention_generation_matches_hf_generate": hf_tokens[: len(ours)] == ours,
        })
    return {
        "cases": rows,
        "max_abs_logit_diff": max(r["max_abs_logit_diff_full_vs_split"] for r in rows),
        "all_argmax_equal": all(r["argmax_equal"] for r in rows),
        "all_finite": all(r["all_logits_finite"] for r in rows),
        "generation_match_rate": float(np.mean([r["full_retention_generation_matches_hf_generate"] for r in rows])),
        "note": "FP16 SDPA; small logit differences between one-shot and split prefill are expected "
                "from different kernel reduction orders.",
    }


@torch.no_grad()
def evaluate_prompt(model, tokenizer, rec: Dict, device: str = "cuda") -> Dict[str, Dict[str, Any]]:
    """Run all six cache conditions on one prompt with a shared FP16 context prefill."""
    ctx, q = split_prompt(tokenizer, rec)
    c_ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    q_ids = tokenizer(q, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    L, L_q = c_ids.shape[1], q_ids.shape[1]
    gold = str(rec["code"])
    gold_pos = set(evidence_token_positions(tokenizer, ctx, rec.get("evidence_text", "")))

    _reset_peak()
    _sync(); t0 = time.perf_counter()
    out_ctx = model(input_ids=c_ids, use_cache=True)
    _sync(); prefill_s = time.perf_counter() - t0
    prefill_peak = _peak_mb()
    n_layers = len(model.model.layers)
    k_list = [out_ctx.past_key_values.layers[i].keys.clone() for i in range(n_layers)]
    v_list = [out_ctx.past_key_values.layers[i].values.clone() for i in range(n_layers)]
    del out_ctx

    importance, scoring_s = question_importance(model, k_list, v_list, q_ids, L)

    results: Dict[str, Dict[str, Any]] = {}
    ref_tokens: List[int] = []
    ref_lp: Optional[torch.Tensor] = None
    for name in POLICIES:
        _reset_peak()
        _sync(); t0 = time.perf_counter()
        res = apply_policy(name, k_list, v_list, importance)
        _sync(); policy_s = time.perf_counter() - t0

        # Free-running greedy answer.
        state = policy_tail_state(res)
        cache = build_cache(res.k, res.v)
        _sync(); t0 = time.perf_counter()
        first = prefill_question(model, cache, q_ids, L, state)
        _sync(); qprefill_s = time.perf_counter() - t0
        tokens, decode_s = greedy_decode(model, cache, first, L + L_q, state, MAX_NEW_TOKENS)
        decode_peak = _peak_mb()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        pred = extract_five_digit_code(text)

        if name == "fp16_reference":
            ref_tokens = tokens
        # Teacher-forced reference-answer NLL and per-position KL from FP16.
        tf_cache = build_cache(res.k, res.v)
        tf_first = prefill_question(model, tf_cache, q_ids, L, policy_tail_state(res))
        lp = teacher_forced_logprobs(model, tf_cache, tf_first, ref_tokens, L + L_q)
        tgt = torch.tensor(ref_tokens, device=lp.device)
        ref_nll = float(-lp.gather(1, tgt[:, None]).mean())
        if name == "fp16_reference":
            ref_lp = lp
        tf_kl = float((ref_lp.exp() * (ref_lp - lp)).sum(-1).mean())
        first_kl = float((ref_lp[0].exp() * (ref_lp[0] - lp[0])).sum())

        sel = None if res.fp16_selected is None else set(res.fp16_selected.tolist())
        evid = None
        if sel is not None and gold_pos:
            evid = len(gold_pos & sel) / len(gold_pos)

        # Live cache memory: compressed pre-question context + shared FP16 question/generated part.
        shared_tokens = L_q + len(tokens)
        shared_bytes = shared_tokens * (compute_uncompressed_fp16_bytes(1))
        results[name] = {
            "prompt_id": rec["id"], "policy": name, "context_len_tokens": L, "question_len_tokens": L_q,
            "target_tokens": rec["target_tokens"], "evidence_position": rec["position"],
            "distractors": rec["distractors"], "gold_code": gold, "predicted_code": pred,
            "generated_text": text, "is_correct": pred == gold,
            "reference_answer_nll": ref_nll, "teacher_forced_mean_kl": tf_kl, "post_prefill_kl": first_kl,
            "retained_evidence_fraction": evid,
            "fp16_selected_positions": None if sel is None else sorted(sel),
            "evidence_token_positions": sorted(gold_pos),
            "packed_context_bytes": res.packed_bytes, "budget_bytes": res.budget_bytes,
            "fraction_of_fp16_bytes": res.packed_bytes / compute_uncompressed_fp16_bytes(L),
            "shared_decode_cache_bytes_fp16": shared_bytes,
            "total_live_cache_bytes": res.packed_bytes + shared_bytes,
            "k_reconstruction_rel_error": res.recon_error_k, "v_reconstruction_rel_error": res.recon_error_v,
            "context_prefill_s": prefill_s, "importance_scoring_s": scoring_s if name in (
                "question_aware_fp16", "hybrid_fp16_int2") else 0.0,
            "policy_apply_s": policy_s, "question_prefill_s": qprefill_s, "decode_s": decode_s,
            "decode_tokens": len(tokens), "prefill_peak_alloc_mb": prefill_peak,
            "decode_peak_alloc_mb": decode_peak, "extra": res.extra,
        }
        del cache, tf_cache
    return results


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(per_policy: Dict[str, List[Dict]]) -> Dict[str, Any]:
    ref = per_policy["fp16_reference"]
    ref_ok = [r["is_correct"] for r in ref]
    summary, breakdown = {}, {}
    for p, rows in per_policy.items():
        summary[p] = {
            "exact_match_accuracy": _mean([r["is_correct"] for r in rows]),
            "accuracy_conditional_on_fp16_correct": _mean([r["is_correct"] for r, ok in zip(rows, ref_ok) if ok]),
            "mean_reference_answer_nll": _mean([r["reference_answer_nll"] for r in rows]),
            "mean_teacher_forced_kl": _mean([r["teacher_forced_mean_kl"] for r in rows]),
            "mean_post_prefill_kl": _mean([r["post_prefill_kl"] for r in rows]),
            "mean_retained_evidence": _mean([r["retained_evidence_fraction"] for r in rows]),
            "mean_packed_mb": _mean([r["packed_context_bytes"] / 2**20 for r in rows]),
            "max_bytes_over_budget": max(r["packed_context_bytes"] - r["budget_bytes"] for r in rows) if p != "fp16_reference" else None,
            "mean_fraction_of_fp16_bytes": _mean([r["fraction_of_fp16_bytes"] for r in rows]),
            "mean_k_recon_error": _mean([r["k_reconstruction_rel_error"] for r in rows]),
            "mean_v_recon_error": _mean([r["v_reconstruction_rel_error"] for r in rows]),
            "mean_scoring_s": _mean([r["importance_scoring_s"] for r in rows]),
            "mean_policy_apply_s": _mean([r["policy_apply_s"] for r in rows]),
            "mean_context_prefill_s": _mean([r["context_prefill_s"] for r in rows]),
            "mean_decode_s": _mean([r["decode_s"] for r in rows]),
            "mean_decode_peak_alloc_mb": _mean([r["decode_peak_alloc_mb"] for r in rows]),
            "n": len(rows),
        }
        cells = defaultdict(list)
        for r in rows:
            for key in (f"len={r['target_tokens']}", f"pos={r['evidence_position']}", f"distr={r['distractors']}",
                        f"cell={r['target_tokens']}/{r['evidence_position']}/{r['distractors']}"):
                cells[key].append(r["is_correct"])
        breakdown[p] = {k: {"accuracy": float(np.mean(v)), "n": len(v)} for k, v in sorted(cells.items())}

    boot = {}
    for p in POLICIES[1:]:
        for base in ("fp16_reference", "question_aware_fp16", "channel_k_token_v_int4"):
            if base == p:
                continue
            d = [float(a["is_correct"]) - float(b["is_correct"]) for a, b in zip(per_policy[p], per_policy[base])]
            boot[f"{p}_minus_{base}_accuracy"] = paired_bootstrap(d)
            d = [a["reference_answer_nll"] - b["reference_answer_nll"] for a, b in zip(per_policy[p], per_policy[base])]
            boot[f"{p}_minus_{base}_ref_nll"] = paired_bootstrap(d)
    return {"summary": summary, "breakdown": breakdown, "bootstrap": boot}


def run_full_kv_evaluation(model, tokenizer, device: str = "cuda", max_cases: Optional[int] = None) -> Dict[str, Any]:
    prompts = load_retrieval_prompts()
    if max_cases is not None:
        prompts = prompts[:max_cases]

    # C1 on one prompt per factorial cell before interpreting compressed variants.
    seen, c1_records = set(), []
    for r in prompts:
        cell = (r["target_tokens"], r["position"], r["distractors"])
        if cell not in seen:
            seen.add(cell)
            c1_records.append(r)
    c1 = validate_fp16_reference(model, tokenizer, c1_records, device=device)
    print(f"[C1] max |logit diff| full vs split = {c1['max_abs_logit_diff']:.4g}; "
          f"argmax equal: {c1['all_argmax_equal']}; generation match rate: {c1['generation_match_rate']:.2f}")

    per_policy: Dict[str, List[Dict]] = {p: [] for p in POLICIES}
    for i, rec in enumerate(prompts):
        if i % 10 == 0:
            print(f"[Part C] prompt {i + 1}/{len(prompts)} ({rec['id']})", flush=True)
        for p, row in evaluate_prompt(model, tokenizer, rec, device=device).items():
            per_policy[p].append(row)

    out = summarize(per_policy)
    for p, s in out["summary"].items():
        print(f"[{p:<24}] acc {s['exact_match_accuracy']:.3f} | ref-NLL {s['mean_reference_answer_nll']:.3f} "
              f"| TF-KL {s['mean_teacher_forced_kl']:.3f} | MB {s['mean_packed_mb']:.2f}")
    out["c1_validation"] = c1
    out["raw_cases"] = per_policy
    out["config"] = {
        "model_dtype": "float16", "attn_implementation": "sdpa", "max_new_tokens": MAX_NEW_TOKENS,
        "answer_extractor": r"first \b\d{5}\b match", "retention_sharing": "shared across all layers and heads",
        "importance": "question-suffix q_proj queries (hooked, RoPE-rotated) vs cached keys, FP32 softmax over "
                      "context keys, mean over heads/query positions/upper-half layers, max-pool k=5",
        "position_metadata": "1-bit-per-position mask, ceil(L/8) bytes, for question-aware and hybrid",
        "budget": "M_4(L) of uniform INT4 channel-K/token-V, group 64, FP16 scale+offset, trailing L mod 64 FP16",
    }
    return out
