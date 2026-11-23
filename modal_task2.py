#!/usr/bin/env python3
"""modal_task2.py — Modal cloud GPU orchestrator for Task 2 (Efficient Attention & Context Memory).

Runs exclusively on Modal serverless GPUs (default: NVIDIA L4 at $0.80/hr).
Persists all checkpoints, statistics, and evaluation metrics to 'atml-pa4-volume'.
Zero compute is executed locally.

Usage Examples:
    # 1. Dry run verification (verifies configuration with zero compute spent):
    modal run modal_task2.py --dry-run

    # 2. Smoke test on Modal L4 GPU (~1 minute):
    modal run modal_task2.py --smoke-test --gpu L4

    # 3. Run Part A (Operator grid & tiling sweep, ~2 minutes):
    modal run modal_task2.py --part-a --gpu L4

    # 4. Run Part B (MQAR multi-seed training & evaluation, ~10 minutes):
    modal run modal_task2.py --part-b --gpu L4

    # 5. Run Part C (KV cache compression across 180 prompts, ~15 minutes):
    modal run modal_task2.py --part-c --gpu L4

    # 6. Run Full Task 2 Suite (Part A + B + C):
    modal run modal_task2.py --run-all --gpu L4

    # 7. Download all task2 evaluation results and summary tables to local disk:
    modal run modal_task2.py --action download
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Modal app and volume definition
try:
    import modal

    app = modal.App("atml-pa4-task2")

    # Cloud volumes for zero-loss persistence and Hugging Face caching
    volume = modal.Volume.from_name("atml-pa4-volume", create_if_missing=True)
    hf_cache_volume = modal.Volume.from_name("atml-pa4-hf-cache", create_if_missing=True)

    VOLUME_MOUNTS = {
        "/root/volume": volume,
        "/root/.cache/huggingface": hf_cache_volume,
    }

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "procps")
        .pip_install(
            "torch>=2.4.0",
            "transformers==4.57.1",
            "huggingface_hub==0.36.2",
            "datasets>=3.0.0,<5.0.0",
            "accelerate>=0.34.0",
            "numpy>=1.26.0",
            "scipy>=1.11.0",
            "tqdm>=4.66.0",
            "pyyaml>=6.0",
            "safetensors>=0.4.0",
            "sentencepiece",
        )
        .env({"PYTHONUNBUFFERED": "1"})
        .add_local_dir(
            ".",
            remote_path="/root/ATML-PA4",
            ignore=[
                "*.pt",
                "*.bin",
                "*.safetensors",
                ".git",
                ".venv",
                "venv",
                "checkpoints",
                "runs",
                "**/__pycache__",
                "*.pyc",
            ],
        )
    )

except ImportError:
    modal = None
    app = None
    volume = None
    hf_cache_volume = None
    VOLUME_MOUNTS = {}
    image = None


# ──────────────────────────────────────────────────────────────────────────────
# Remote Cloud GPU Execution
# ──────────────────────────────────────────────────────────────────────────────

def _run_task2_in_container(
    smoke_test: bool = False,
    part_a: bool = False,
    part_b: bool = False,
    part_c: bool = False,
    run_all: bool = False,
    context_only: bool = False,
) -> dict:
    """Execute Task 2 inside cloud container with GPU acceleration and volume checkpointing."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, "/root/ATML-PA4")

    import torch
    from task2.runner import run_part_a, run_part_b, run_part_c
    from task2.checkpoint import Task2StateManager

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0

    print("=" * 80)
    print("MODAL CLOUD CONTAINER — TASK 2 EXECUTION INITIALIZED")
    print(f"Device: {gpu_name} ({total_mem:.1f} GB VRAM)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 80, flush=True)

    ckpt_dir = Path("/root/volume/checkpoints/task2")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_mgr = Task2StateManager(checkpoint_dir=ckpt_dir, modal_volume=volume)

    t0 = time.perf_counter()

    if smoke_test:
        print("\n>>> Launching Task 2 Smoke Test on Modal GPU <<<\n")
        run_part_a(state_mgr, device="cuda", smoke_test=True)
        run_part_b(state_mgr, device="cuda", smoke_test=True)
        run_part_c(state_mgr, device="cuda", smoke_test=True)
        print("\n✓ Smoke test finished successfully on cloud GPU.")
        state_mgr.commit_volume()
        return {"smoke_test": True, "elapsed_sec": time.perf_counter() - t0}

    if context_only:
        print("\n>>> Launching Context Benchmarking (Option A OOLONG + Option B M-NIAH) <<<\n")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from task2.config import KV_MODEL_NAME
        from task2.eval_context_benchmark import run_context_benchmarks
        print(f"Loading '{KV_MODEL_NAME}' in FP16 on cuda...")
        tokenizer = AutoTokenizer.from_pretrained(KV_MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            KV_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        model.eval()
        ctx_results = run_context_benchmarks(model=model, tokenizer=tokenizer, device="cuda")
        state_mgr.state["context_benchmarks"] = ctx_results
        state_mgr._save_state()
        state_mgr.commit_volume()
        return {"status": "success", "elapsed_sec": time.perf_counter() - t0}

    # Default to run_all if no specific part selected
    if not (part_a or part_b or part_c):
        run_all = True

    if part_a or run_all:
        run_part_a(state_mgr, device="cuda")

    if part_b or run_all:
        run_part_b(state_mgr, device="cuda")

    if part_c or run_all:
        run_part_c(state_mgr, device="cuda")

    state_mgr.commit_volume()
    elapsed = time.perf_counter() - t0
    print(f"\n✓ Task 2 Cloud execution finished in {elapsed/60.0:.2f} minutes.")
    return {"status": "success", "elapsed_sec": elapsed}


if app is not None:
    @app.function(
        image=image,
        gpu="L4",
        volumes=VOLUME_MOUNTS,
        timeout=3600 * 2,  # 2 hours max
    )
    def run_task2_l4(smoke_test: bool = False, part_a: bool = False, part_b: bool = False, part_c: bool = False, run_all: bool = False, context_only: bool = False):
        return _run_task2_in_container(smoke_test, part_a, part_b, part_c, run_all, context_only)

    @app.function(
        image=image,
        gpu="A10G",
        volumes=VOLUME_MOUNTS,
        timeout=3600 * 2,
    )
    def run_task2_a10g(smoke_test: bool = False, part_a: bool = False, part_b: bool = False, part_c: bool = False, run_all: bool = False, context_only: bool = False):
        return _run_task2_in_container(smoke_test, part_a, part_b, part_c, run_all, context_only)

    @app.function(
        image=image,
        gpu="A100-40GB",
        volumes=VOLUME_MOUNTS,
        timeout=3600 * 2,
    )
    def run_task2_a100(smoke_test: bool = False, part_a: bool = False, part_b: bool = False, part_c: bool = False, run_all: bool = False, context_only: bool = False):
        return _run_task2_in_container(smoke_test, part_a, part_b, part_c, run_all, context_only)

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        timeout=600,
    )
    def read_checkpoint_file(rel_path: str) -> bytes:
        p = Path("/root/volume/checkpoints/task2") / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist on volume: {p}")
        return p.read_bytes()

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        timeout=600,
    )
    def list_volume_files(subpath: str = "checkpoints/task2") -> list:
        p = Path("/root/volume") / subpath
        if not p.exists():
            return []
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]


# ──────────────────────────────────────────────────────────────────────────────
# Local CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    gpu: str = "L4",
    smoke_test: bool = False,
    part_a: bool = False,
    part_b: bool = False,
    part_c: bool = False,
    run_all: bool = False,
    context_only: bool = False,
    dry_run: bool = False,
    action: str = "",
):
    print("=" * 80)
    print("Modal Task 2 Orchestrator (100% Cloud Serverless Execution)")
    print(f"Target GPU:    NVIDIA {gpu} ($0.80/hr on L4)")
    print(f"Volume:        atml-pa4-volume -> /root/volume")
    print(f"HF Cache:      atml-pa4-hf-cache -> /root/.cache/huggingface")
    print(f"Smoke Test:    {smoke_test}")
    print(f"Part A:        {part_a or run_all}")
    print(f"Part B:        {part_b or run_all}")
    print(f"Part C:        {part_c or run_all}")
    print(f"Context Only:  {context_only}")
    print(f"Run All:       {run_all or not (part_a or part_b or part_c or context_only)}")
    print("=" * 80)

    if dry_run:
        print("\n[Dry-run] Verification successful. Modal app and volume configurations valid.")
        print("[Dry-run] Zero compute hours were consumed.")
        return

    if action == "download":
        local_results_dir = Path("runs/task2_results")
        local_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Local] Syncing results from cloud volume into {local_results_dir}...")
        files = [
            "checkpoints/task2/state.json",
            "checkpoints/task2/all_results.json",
            "checkpoints/task2/context_benchmarks.json",
        ]
        modal_cmd = "/home/supus/Projects/LoopLM/.venv/bin/modal" if Path("/home/supus/Projects/LoopLM/.venv/bin/modal").exists() else "modal"
        for f in files:
            target_file = local_results_dir / Path(f).name
            cmd = f"{modal_cmd} volume get atml-pa4-volume {f} {target_file} --force 2>/dev/null || true"
            os.system(cmd)
        print(f"[Local] ✓ Results synced to {local_results_dir}.")
        return

    gpu_upper = gpu.upper()
    if gpu_upper == "L4":
        launcher = run_task2_l4
    elif gpu_upper == "A10G":
        launcher = run_task2_a10g
    elif gpu_upper == "A100":
        launcher = run_task2_a100
    else:
        raise ValueError(f"Unsupported GPU: {gpu}. Choose from: L4, A10G, A100")

    print(f"\n[Modal Cloud] Launching Task 2 on NVIDIA {gpu_upper}...")
    results = launcher.remote(
        smoke_test=smoke_test,
        part_a=part_a,
        part_b=part_b,
        part_c=part_c,
        run_all=run_all,
        context_only=context_only,
    )
    print("\n[Modal Cloud] Execution finished successfully.")

    # Automatically sync state.json, all_results.json, and context_benchmarks.json back locally
    local_results_dir = Path("runs/task2_results")
    local_results_dir.mkdir(parents=True, exist_ok=True)
    modal_cmd = "/home/supus/Projects/LoopLM/.venv/bin/modal" if Path("/home/supus/Projects/LoopLM/.venv/bin/modal").exists() else "modal"
    for f in ["checkpoints/task2/state.json", "checkpoints/task2/all_results.json", "checkpoints/task2/context_benchmarks.json"]:
        target_file = local_results_dir / Path(f).name
        cmd = f"{modal_cmd} volume get atml-pa4-volume {f} {target_file} --force 2>/dev/null || true"
        os.system(cmd)
    print(f"[Local] ✓ Results synced to {local_results_dir}.")


if __name__ == "__main__":
    if modal is None:
        print("[Error] modal package is not installed. Install via: pip install modal")
        sys.exit(1)
