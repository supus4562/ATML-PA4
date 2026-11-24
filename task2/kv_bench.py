"""task2/kv_bench.py — C5: real packed-cache encode/decode benchmark.

Packing is physical: integer codes are bit-packed into uint8 (two INT4 or four INT2 codes per
byte) and FP16 scale/offset tensors are stored alongside, so `serialized_bytes` is the sum of
real tensor sizes. Shapes follow Qwen2.5-1.5B (28 layers, 2 KV heads, d_h = 128).

Measured per (layout, batch in {1, 8}, context in {2048, 4096}):
  * pack_ms      quantize + bit-pack the whole pre-question cache
  * unpack_ms    unpack + dequantize back to FP16
  * decode_step_fp16_ms           one model decode step on an FP16 cache of the same shape
  * decode_step_with_unpack_ms    unpack/dequantize + the same decode step (ordinary SDPA on the
                                  dequantized FP16 tensors — no fused low-bit kernel)
  * allocated MB of the packed cache vs FP16 cache (GPU allocation, not just byte formulas)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch
from transformers.cache_utils import DynamicCache

from task2.config import KV_GROUP_SIZE, KV_BENCHMARK_BATCHES, KV_BENCHMARK_REPETITIONS
from task2.kv_cache_models import compute_matched_int4_bytes


def _sync():
    torch.cuda.synchronize()


def _quant_codes(x: torch.Tensor, bits: int):
    """Asymmetric group quantization along the last axis (groups of 64). Returns codes + meta."""
    q_max = (1 << bits) - 1
    g = x.reshape(*x.shape[:-1], x.shape[-1] // KV_GROUP_SIZE, KV_GROUP_SIZE).float()
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((mx - mn) / q_max).clamp_min(1e-8)
    zero = torch.round(-mn / scale)
    codes = torch.clamp(torch.round(g / scale) + zero, 0, q_max).to(torch.uint8)
    return codes, scale.half(), zero.half()


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    per = 8 // bits
    c = codes.reshape(-1, per)
    out = torch.zeros(c.shape[0], dtype=torch.uint8, device=c.device)
    for i in range(per):
        out |= c[:, i] << (i * bits)
    return out


def _unpack(packed: torch.Tensor, bits: int, shape) -> torch.Tensor:
    per, mask = 8 // bits, (1 << bits) - 1
    parts = [(packed >> (i * bits)) & mask for i in range(per)]
    return torch.stack(parts, dim=1).reshape(shape)


def encode_layer(k: torch.Tensor, v: torch.Tensor, layout: str, bits: int) -> Dict[str, Any]:
    """k, v: (B, H, L, d) FP16 with L % 64 == 0 (full groups only; trailing tokens stay FP16)."""
    kk = k.transpose(-1, -2) if layout == "channel_k_token_v" else k  # group along tokens for K
    kc, ks, kz = _quant_codes(kk.contiguous(), bits)
    vc, vs, vz = _quant_codes(v, bits)
    return {"k": _pack(kc, bits), "ks": ks, "kz": kz, "v": _pack(vc, bits), "vs": vs, "vz": vz,
            "kshape": kc.shape, "vshape": vc.shape, "layout": layout, "bits": bits, "korig": kk.shape}


def decode_layer(e: Dict[str, Any]):
    bits = e["bits"]
    k = ((_unpack(e["k"], bits, e["kshape"]).float() - e["kz"].float()) * e["ks"].float()).half()
    v = ((_unpack(e["v"], bits, e["vshape"]).float() - e["vz"].float()) * e["vs"].float()).half()
    k = k.reshape(e["korig"])
    if e["layout"] == "channel_k_token_v":
        k = k.transpose(-1, -2)
    v = v.reshape(*v.shape[:-2], -1)
    return k.contiguous(), v.contiguous()


def serialized_bytes(encoded: List[Dict[str, Any]]) -> int:
    return sum(t.numel() * t.element_size() for e in encoded for key in ("k", "ks", "kz", "v", "vs", "vz")
               for t in [e[key]])


def _time(fn, reps: int) -> List[float]:
    for _ in range(3):
        fn()
    _sync()
    out = []
    for _ in range(reps):
        _sync(); t0 = time.perf_counter(); fn(); _sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


@torch.no_grad()
def run_kv_pack_benchmark(model, device: str = "cuda", lengths=(2048, 4096),
                          batches=tuple(KV_BENCHMARK_BATCHES), reps: int = KV_BENCHMARK_REPETITIONS):
    cfg = model.config
    n_l, H, d = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    gen = torch.Generator(device=device).manual_seed(6304)
    rows = []
    for L in lengths:
        for B in batches:
            ks = [torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=gen) for _ in range(n_l)]
            vs = [torch.randn(B, H, L, d, device=device, dtype=torch.float16, generator=gen) for _ in range(n_l)]
            tok = torch.randint(100, 1000, (B, 1), device=device)
            pos = torch.full((B, 1), L, device=device)

            def step(k_list, v_list):
                c = DynamicCache()
                for i in range(n_l):
                    c.update(k_list[i], v_list[i], i)
                model(input_ids=tok, position_ids=pos, past_key_values=c, use_cache=True)

            fp16_step = _time(lambda: step(ks, vs), reps)
            fp16_alloc = sum(t.numel() * 2 for t in ks + vs) / 2**20
            for layout in ("token_wise", "channel_k_token_v"):
                enc_holder = {}
                pack_ms = _time(lambda: enc_holder.__setitem__(
                    "e", [encode_layer(k, v, layout, 4) for k, v in zip(ks, vs)]), reps)
                enc = enc_holder["e"]
                unpack_ms = _time(lambda: [decode_layer(e) for e in enc], reps)

                def step_unpack():
                    kv = [decode_layer(e) for e in enc]
                    step([a for a, _ in kv], [b for _, b in kv])

                dec_step = _time(step_unpack, reps)
                sb = serialized_bytes(enc)
                rows.append({
                    "layout": layout, "bits": 4, "batch": B, "context_len": L,
                    "serialized_bytes": sb, "formula_bytes": B * compute_matched_int4_bytes(L, b=4),
                    "packed_alloc_mb": sb / 2**20, "fp16_cache_alloc_mb": fp16_alloc,
                    "pack_ms_mean": float(sum(pack_ms) / reps), "unpack_ms_mean": float(sum(unpack_ms) / reps),
                    "decode_step_fp16_ms_mean": float(sum(fp16_step) / reps),
                    "decode_step_with_unpack_ms_mean": float(sum(dec_step) / reps),
                    "pack_ms": pack_ms, "unpack_ms": unpack_ms,
                    "decode_step_fp16_ms": fp16_step, "decode_step_with_unpack_ms": dec_step,
                    "repetitions": reps, "warmup": 3, "unit": "milliseconds, CUDA-synchronized",
                })
                print(f"[C5] {layout:18} B={B} L={L}: pack {rows[-1]['pack_ms_mean']:.2f} ms | unpack "
                      f"{rows[-1]['unpack_ms_mean']:.2f} ms | step fp16 {rows[-1]['decode_step_fp16_ms_mean']:.2f} "
                      f"vs +unpack {rows[-1]['decode_step_with_unpack_ms_mean']:.2f} ms | {sb/2**20:.1f} MB")
            del ks, vs
            torch.cuda.empty_cache()
    return {"rows": rows, "note": "Random FP16 caches at Qwen2.5-1.5B shapes; L is a multiple of 64 so "
                                  "serialized bytes equal M_4(L) per sequence. Dequantized values feed "
                                  "ordinary SDPA, so GPU allocation during decode is FP16-sized."}
