"""task2/eval_oolong.py — Context aggregation benchmarking using the official OOLONG evaluation harness.

Reference:
    "Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities"
    Bertsch et al., 2025 (arXiv:2511.02817)
    Official Repo: https://github.com/abertsch72/oolong
"""

from __future__ import annotations

import ast
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from task2.config import (
    REPO_ROOT,
    KV_MODEL_NAME,
)
from task2.kv_core import build_cache, policy_tail_state, prefill_question, question_importance
from task2.kv_policies import apply_policy
from task2.kv_policies import (
    apply_policy_sink_recent,
    apply_policy_question_aware,
    apply_policy_token_wise_int4,
    apply_policy_channel_k_token_v_int4,
    apply_policy_hybrid,
)


# ──────────────────────────────────────────────────────────────────────────────
# Official OOLONG Parsing and Scoring Logic (abertsch72/oolong/src/eval/eval_helpers.py)
# ──────────────────────────────────────────────────────────────────────────────

def synth_attempt_answer_parse(answer: str) -> Tuple[str, str]:
    """Official OOLONG answer extraction function."""
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer.strip(), parse_confidence
        else:
            return answer.split()[-1].strip(), parse_confidence

    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")  # Strip bolding
    candidate_answer = candidate_answer.replace("[", "").replace("]", "")  # Strip brackets
    parse_confidence = "med"

    if any(k in answer for k in ["User:", "Answer:", "Date:", "Label"]):
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint: Dict, output: str, model_name: str) -> Dict[str, Any]:
    """Official OOLONG response processing and graded scoring function."""
    score = 0.0
    ans_raw = datapoint.get("answer", "")

    # Parse gold target
    try:
        if "datetime" not in ans_raw:
            gold = ast.literal_eval(ans_raw)[0] if ans_raw.startswith("[") else ans_raw
        else:
            gold = datetime.strptime(ans_raw, "[datetime.date(%Y, %m, %d)]")
    except Exception:
        gold = ans_raw.strip("[]'\" ")

    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)

    if str(trimmed_output).lower() == str(gold).lower():
        score = 1.0
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        if str(trimmed_output) in str(gold):
            score = 1.0
    elif datapoint.get("answer_type") == "ANSWER_TYPE.NUMERIC":
        # Graded partial credit for numeric counts: score = 0.75 ** |gold - pred|
        try:
            pred_num = int(trimmed_output)
            gold_num = int(gold)
            score = 0.75 ** (abs(gold_num - pred_num))
        except Exception:
            parse_confidence = "low"
    elif datapoint.get("answer_type") == "ANSWER_TYPE.DATE":
        try:
            import dateutil.parser
            pred_date = dateutil.parser.parse(trimmed_output)
            score = 1.0 if pred_date == gold else 0.0
        except Exception:
            parse_confidence = "low"

    return {
        "id": datapoint.get("id", "case"),
        "context_window_id": datapoint.get("context_window_id", 0),
        "dataset": datapoint.get("dataset", "synth"),
        "model": model_name,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": float(score),
        "answer": str(gold),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Loop across KV Cache Policies
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_oolong_case(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    datapoint: Dict,
    policy_name: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run forward prefill, policy compression, and greedy generation on an OOLONG prompt."""
    prompt_text = datapoint["prompt"]

    # In OOLONG, questions appear at the end: split context body from final question prompt
    if "\n\nQuestion:" in prompt_text:
        context_part, question_part = prompt_text.rsplit("\n\nQuestion:", 1)
        question_part = "\n\nQuestion:" + question_part
    else:
        split_idx = max(0, len(prompt_text) - 300)
        context_part = prompt_text[:split_idx]
        question_part = prompt_text[split_idx:]

    enc_context = tokenizer(context_part, return_tensors="pt", add_special_tokens=False).to(device)
    enc_question = tokenizer(question_part, return_tensors="pt", add_special_tokens=False).to(device)

    L_ctx = enc_context.input_ids.shape[1]
    L_q = enc_question.input_ids.shape[1]

    # Prefill context
    with torch.no_grad():
        out_ctx = model(**enc_context, use_cache=True)
        pkv = out_ctx.past_key_values

    n_layers = len(model.model.layers)
    k_list = [pkv.layers[i].keys.clone() for i in range(n_layers)]
    v_list = [pkv.layers[i].values.clone() for i in range(n_layers)]

    q_importance = None
    if policy_name in ("question_aware_fp16", "hybrid_fp16_int2"):
        q_importance, _ = question_importance(model, k_list, v_list, enc_question.input_ids, L_ctx)
    res = apply_policy(policy_name, k_list, v_list, q_importance)
    cache = build_cache(res.k, res.v)
    first = prefill_question(model, cache, enc_question.input_ids, L_ctx, policy_tail_state(res))

    # Greedy decode for up to 32 tokens
    curr_input = torch.argmax(first, dim=-1, keepdim=True)
    generated_tokens = [curr_input.item()]
    curr_cache = cache
    curr_pos = L_ctx + L_q

    with torch.no_grad():
        for _ in range(31):
            out_gen = model(
                input_ids=curr_input,
                position_ids=torch.tensor([[curr_pos]], device=device),
                past_key_values=curr_cache,
                use_cache=True,
            )
            curr_input = torch.argmax(out_gen.logits[:, -1, :], dim=-1, keepdim=True)
            tok_id = curr_input.item()
            generated_tokens.append(tok_id)
            if tok_id in [tokenizer.eos_token_id, 151643, 151645]:  # Stop on EOS or chat stop tokens
                break
            curr_cache = out_gen.past_key_values
            curr_pos += 1

    gen_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    processed = synth_process_response(datapoint, gen_text, f"{model.config.name_or_path}_{policy_name}")
    processed["policy"] = policy_name
    return processed
