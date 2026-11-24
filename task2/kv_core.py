"""task2/kv_core.py — Shared cache-editing utilities for Part C (SDPA model, FP16).

* Question-aware importance is computed WITHOUT switching to eager attention: forward hooks
  capture each layer's q_proj output while the ordinary SDPA model processes the question
  suffix; the queries are rotated with the model's own rotary embedding at their absolute
  positions and a masked FP32 softmax is computed against the existing rotary-positioned
  cached context keys. This is a simplified fixed policy inspired by SnapKV (scores averaged
  over heads, question positions and the upper half of the layers, then shared by every
  layer/head), not the published head-specific algorithm.
* Retained tokens keep their ORIGINAL absolute RoPE positions: cached keys are already
  rotated, and new tokens are given explicit position_ids that continue from the original
  context length, while the physical cache length is whatever was retained.
* For quantized layouts, newly appended tokens stay FP16 until they complete a 64-token
  group, at which point that group is quantized with the same layout (KIVI-style tail).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from task2.config import KV_GROUP_SIZE
from task2.kv_policies import PolicyResult, quantize_kv_block

EOS_IDS = (151643, 151645)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_cache(k_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> DynamicCache:
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(k_list, v_list)):
        cache.update(k.clone(), v.clone(), i)
    return cache


def _layer_kv(cache: DynamicCache, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def _set_layer_kv(cache: DynamicCache, i: int, k: torch.Tensor, v: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    else:
        cache.key_cache[i], cache.value_cache[i] = k, v


def cache_physical_len(cache: DynamicCache) -> int:
    return _layer_kv(cache, 0)[0].shape[2]


def requantize_tail(cache: DynamicCache, state: Dict) -> None:
    """Quantize every newly completed 64-token FP16 tail group with the policy's layout."""
    layout, bits = state.get("layout"), state.get("bits")
    if layout is None:
        return
    n_layers = len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)
    while cache_physical_len(cache) - state["upto"] >= KV_GROUP_SIZE:
        lo, hi = state["upto"], state["upto"] + KV_GROUP_SIZE
        for i in range(n_layers):
            k, v = _layer_kv(cache, i)
            kq, vq, _ = quantize_kv_block(k[:, :, lo:hi], v[:, :, lo:hi], layout, bits)
            k = k.clone(); v = v.clone()
            k[:, :, lo:hi], v[:, :, lo:hi] = kq, vq
            _set_layer_kv(cache, i, k, v)
        state["upto"] = hi
        state["groups_quantized_after_prefill"] = state.get("groups_quantized_after_prefill", 0) + 1


@torch.no_grad()
def question_importance(
    model,
    k_list: List[torch.Tensor],
    v_list: List[torch.Tensor],
    question_ids: torch.Tensor,
    ctx_len: int,
    layer_fraction: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """Per-context-position importance from question-suffix queries (FP32, masked to context)."""
    device = question_ids.device
    layers = model.model.layers
    n_layers = len(layers)
    start = int(n_layers * (1 - layer_fraction))
    captured: Dict[int, torch.Tensor] = {}
    hooks = []
    for li in range(start, n_layers):
        hooks.append(layers[li].self_attn.q_proj.register_forward_hook(
            lambda m, inp, out, li=li: captured.__setitem__(li, out.detach())
        ))
    _sync()
    t0 = time.perf_counter()
    L_q = question_ids.shape[1]
    pos = torch.arange(ctx_len, ctx_len + L_q, device=device).unsqueeze(0)
    try:
        model(input_ids=question_ids, position_ids=pos, past_key_values=build_cache(k_list, v_list), use_cache=True)
    finally:
        for h in hooks:
            h.remove()

    cfg = model.config
    H_q, d_h = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    cos, sin = model.model.rotary_emb(captured[start].to(torch.float32), pos)
    L = k_list[0].shape[2]
    imp = torch.zeros(L, device=device, dtype=torch.float32)
    for li in range(start, n_layers):
        q = captured[li].view(1, L_q, H_q, d_h).transpose(1, 2).float()
        q, _ = apply_rotary_pos_emb(q, q, cos.float(), sin.float())
        k = k_list[li].float()
        k = k.repeat_interleave(H_q // k.shape[1], dim=1)
        probs = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_h), dim=-1)  # (1, H, Lq, L)
        imp += probs.mean(dim=(0, 1, 2))
    _sync()
    return imp / (n_layers - start), time.perf_counter() - t0


@torch.no_grad()
def prefill_question(model, cache: DynamicCache, question_ids: torch.Tensor, ctx_len: int, state: Dict):
    L_q = question_ids.shape[1]
    pos = torch.arange(ctx_len, ctx_len + L_q, device=question_ids.device).unsqueeze(0)
    out = model(input_ids=question_ids, position_ids=pos, past_key_values=cache, use_cache=True)
    requantize_tail(cache, state)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def greedy_decode(model, cache, first_logits, next_pos: int, state: Dict, max_new_tokens: int = 48):
    tokens: List[int] = []
    cur = torch.argmax(first_logits, dim=-1, keepdim=True)
    device = first_logits.device
    _sync()
    t0 = time.perf_counter()
    for step in range(max_new_tokens):
        tok = int(cur.item())
        tokens.append(tok)
        if tok in EOS_IDS or step == max_new_tokens - 1:
            break
        out = model(
            input_ids=cur, position_ids=torch.tensor([[next_pos]], device=device),
            past_key_values=cache, use_cache=True,
        )
        requantize_tail(cache, state)
        next_pos += 1
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    _sync()
    return tokens, time.perf_counter() - t0


@torch.no_grad()
def teacher_forced_logprobs(model, cache, first_logits, ref_tokens: List[int], next_pos: int) -> torch.Tensor:
    """Log-probs at every reference-answer position (first from the question prefill)."""
    lp = [F.log_softmax(first_logits.float(), dim=-1)]
    if len(ref_tokens) > 1:
        ids = torch.tensor([ref_tokens[:-1]], device=first_logits.device)
        pos = torch.arange(next_pos, next_pos + ids.shape[1], device=ids.device).unsqueeze(0)
        out = model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True)
        lp.append(F.log_softmax(out.logits[0].float(), dim=-1))
    return torch.cat(lp, dim=0)  # (len(ref_tokens), V)


def policy_tail_state(res: PolicyResult) -> Dict:
    return {"layout": res.quant_layout, "bits": res.quant_bits, "upto": res.quantized_upto}
