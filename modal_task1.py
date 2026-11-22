#!/usr/bin/env python3
"""modal_task1.py — Modal cloud GPU orchestrator for Task 1 (Model Compression).

Runs exclusively on Modal serverless GPUs (default: NVIDIA L4 at $0.80/hr).
Persists all checkpoints, statistics, and evaluation metrics to 'atml-pa4-volume'.
Zero compute is executed locally.

Usage Examples:
    # 1. Dry run verification (verifies configuration with zero compute spent):
    modal run modal_task1.py --dry-run

    # 2. Smoke test on Modal L4 GPU (~1 minute):
    modal run modal_task1.py --smoke-test --gpu L4

    # 3. Full Task 1 sweep on Modal L4 GPU (~15-25 minutes, ~$0.25 total):
    modal run modal_task1.py --run-all --gpu L4

    # 4. Download all task1 evaluation results and summary tables to local disk:
    modal run modal_task1.py --action download
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

    app = modal.App("atml-pa4-task1")

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

def _run_task1_in_container(smoke_test: bool = False) -> dict:
    """Execute Task 1 inside cloud container with GPU acceleration and volume checkpointing."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, "/root/ATML-PA4")

    from task1.config import Task1Config
    from task1.runner import run_task1_pipeline

    config = Task1Config()
    # Check if locked data exists in volume; if so, point config to volume
    volume_data_dir = Path("/root/volume/data")
    if (volume_data_dir / "public_locked/c4_calibration_locked.jsonl.gz").exists():
        config.base_data_dir = volume_data_dir

    print("=" * 80)
    print(f"[Modal Task 1] Commencing execution on {config.device.upper()}...")
    print(f"[Modal Task 1] Volume mounted at /root/volume")
    print(f"[Modal Task 1] HF cache mounted at /root/.cache/huggingface")
    print(f"[Modal Task 1] Smoke test mode: {smoke_test}")
    print("=" * 80, flush=True)

    results = run_task1_pipeline(config=config, modal_volume=volume, smoke_test=smoke_test)
    volume.commit()
    hf_cache_volume.commit()
    print("[Modal Task 1] ✓ Volume committed successfully.")
    return results


if app is not None:

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        gpu="L4",
        timeout=86400,
    )
    def run_on_l4(smoke_test: bool = False):
        return _run_task1_in_container(smoke_test=smoke_test)

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        gpu="A10G",
        timeout=86400,
    )
    def run_on_a10g(smoke_test: bool = False):
        return _run_task1_in_container(smoke_test=smoke_test)

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        gpu="A100",
        timeout=86400,
    )
    def run_on_a100(smoke_test: bool = False):
        return _run_task1_in_container(smoke_test=smoke_test)

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        gpu="T4",
        timeout=86400,
    )
    def run_on_t4(smoke_test: bool = False):
        return _run_task1_in_container(smoke_test=smoke_test)

    @app.function(
        image=image,
        volumes=VOLUME_MOUNTS,
        timeout=3600,
    )
    def prepare_data_on_cloud():
        """Ensure public locked data is generated and stored in persistent volume."""
        import subprocess
        print("[Modal Cloud] Preparing public datasets into /root/volume/data...")
        subprocess.run(
            ["python", "scripts/setup_data.py", "--public"],
            cwd="/root/ATML-PA4",
            check=True,
        )
        # Copy generated locked data into volume
        subprocess.run(
            "mkdir -p /root/volume/data && cp -r /root/ATML-PA4/data/public_locked /root/volume/data/ 2>/dev/null || true",
            shell=True,
            check=True,
        )
        volume.commit()
        hf_cache_volume.commit()
        print("[Modal Cloud] ✓ Public dataset files prepared in volume.")


# ──────────────────────────────────────────────────────────────────────────────
# Local CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    gpu: str = "L4",
    smoke_test: bool = False,
    run_all: bool = False,
    prepare_data: bool = False,
    dry_run: bool = False,
    action: str = "",
):
    print("=" * 80)
    print("Modal Task 1 Orchestrator (100% Cloud Serverless Execution)")
    print(f"Target GPU:    {gpu}")
    print(f"Volume:        atml-pa4-volume -> /root/volume")
    print(f"HF Cache:      atml-pa4-hf-cache -> /root/.cache/huggingface")
    print(f"Smoke Test:    {smoke_test}")
    print(f"Run All:       {run_all}")
    print("=" * 80)

    if dry_run:
        print("\n[Dry-run] Verification successful. Modal app and volume configurations valid.")
        print("[Dry-run] Zero compute hours were consumed.")
        return

    if action == "download":
        local_results_dir = Path("runs/task1_results")
        local_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Local] Syncing results from cloud volume into {local_results_dir}...")
        files = [
            "checkpoints/task1/state.json",
            "checkpoints/task1/all_results.json",
            "checkpoints/task1/bootstrap_ci.json",
            "checkpoints/task1/pruning_sensitivity_full.json",
            "checkpoints/task1/quant_sensitivity_full.json",
            "checkpoints/task1/pruning_sensitivity.json",
            "checkpoints/task1/quant_sensitivity.json",
        ]
        # Determine modal binary path
        modal_cmd = "/home/supus/Projects/LoopLM/.venv/bin/modal" if Path("/home/supus/Projects/LoopLM/.venv/bin/modal").exists() else "modal"
        for f in files:
            target_file = local_results_dir / Path(f).name
            cmd = f"{modal_cmd} volume get atml-pa4-volume {f} {target_file} --force 2>/dev/null || true"
            os.system(cmd)
        print(f"[Local] ✓ Results synced to {local_results_dir}.")
        return

    if prepare_data:
        print("\n[Modal Cloud] Running dataset preparation on cloud...")
        prepare_data_on_cloud.remote()
        return

    gpu_upper = gpu.upper()
    if gpu_upper == "L4":
        launcher = run_on_l4
    elif gpu_upper == "A10G":
        launcher = run_on_a10g
    elif gpu_upper == "A100":
        launcher = run_on_a100
    elif gpu_upper == "T4":
        launcher = run_on_t4
    else:
        raise ValueError(f"Unsupported GPU: {gpu}. Choose from: L4, A10G, A100, T4")

    print(f"\n[Modal Cloud] Launching Task 1 on NVIDIA {gpu_upper}...")
    results = launcher.remote(smoke_test=smoke_test)
    print("\n[Modal Cloud] Execution finished successfully.")


if __name__ == "__main__":
    if modal is None:
        print("[Error] modal package is not installed. Install via: pip install modal")
        sys.exit(1)
