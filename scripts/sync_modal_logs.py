#!/usr/bin/env python3
"""scripts/sync_modal_logs.py — Sync and parse live logs from running Modal apps.

Adapted from LoopLM for ATML-PA4.
Fetches logs from active or stopped Modal apps, saves them to runs/modal_logs/,
and prints recent output.

Usage:
    # 1. Sync logs for a specific app ID:
    python scripts/sync_modal_logs.py --app-id <app_id>

    # 2. List recently active apps and sync the latest:
    python scripts/sync_modal_logs.py --latest
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MODAL_BIN = (
    "/home/supus/Projects/LoopLM/.venv/bin/modal"
    if Path("/home/supus/Projects/LoopLM/.venv/bin/modal").exists()
    else (shutil.which("modal") or "modal")
)


def get_latest_app_id() -> str | None:
    try:
        res = subprocess.run(
            [MODAL_BIN, "app", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        apps = json.loads(res.stdout)
        if not apps:
            return None
        # Return the most recent app ID
        return apps[0]["App ID"]
    except Exception:
        # Fallback to plain text list
        res = subprocess.run(
            [MODAL_BIN, "app", "list"],
            capture_output=True,
            text=True,
        )
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("ap-"):
                parts = line.split()
                return parts[0]
        return None


def sync_app_logs(app_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{app_id}.log"
    print(f"Fetching logs for Modal App: {app_id}...")
    res = subprocess.run(
        [MODAL_BIN, "app", "logs", app_id],
        capture_output=True,
        text=True,
    )
    log_file.write_text(res.stdout)
    print(f"Saved {len(res.stdout.splitlines())} lines to {log_file}")
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Sync live logs from Modal apps.")
    parser.add_argument("--app-id", type=str, default=None, help="Modal App ID (ap-...)")
    parser.add_argument("--latest", action="store_true", help="Sync logs from the most recent app")
    parser.add_argument("--tail", type=int, default=30, help="Number of trailing lines to print")
    args = parser.parse_args()

    app_id = args.app_id
    if not app_id and args.latest:
        app_id = get_latest_app_id()
        if not app_id:
            print("No recent Modal apps found.")
            sys.exit(1)
        print(f"Detected latest app: {app_id}")

    if not app_id:
        print("Please provide --app-id <app-id> or --latest.")
        sys.exit(1)

    output_dir = Path("runs/modal_logs")
    log_file = sync_app_logs(app_id, output_dir)

    print(f"\n--- Last {args.tail} lines of {log_file} ---")
    lines = log_file.read_text().splitlines()
    for line in lines[-args.tail:]:
        print(line)


if __name__ == "__main__":
    main()
