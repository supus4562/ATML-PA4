"""task2/checkpoint.py — State manager for persistent volume checkpointing & auto-resume in Task 2.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch


class Task2StateManager:
    """Tracks completed Task 2 steps (Part A, Part B, Part C) and enables zero-overhead auto-resume."""

    def __init__(self, checkpoint_dir: Path | str, modal_volume: Optional[Any] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.checkpoint_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / "state.json"
        self.results_file = self.checkpoint_dir / "all_results.json"
        self.modal_volume = modal_volume
        self.state: Dict[str, Any] = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Task2StateManager] Warning: could not parse {self.state_file}: {e}")
        return {
            "completed_steps": [],
            "in_progress": None,
            "part_a": {},
            "part_b": {},
            "part_c": {},
        }

    def _save_state(self) -> None:
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        temp_file.replace(self.state_file)

        # Also write consolidated results file
        consolidated = {
            "part_a": self.state.get("part_a", {}),
            "part_b": self.state.get("part_b", {}),
            "part_c": self.state.get("part_c", {}),
        }
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)

        self.commit_volume()

    def commit_volume(self) -> None:
        """Commit modal volume changes to durable cloud storage."""
        if self.modal_volume is not None:
            try:
                self.modal_volume.commit()
            except Exception as e:
                print(f"[Task2StateManager] Volume commit warning: {e}")

    def is_completed(self, step_name: str) -> bool:
        return step_name in self.state.get("completed_steps", [])

    def mark_in_progress(self, step_name: str) -> None:
        self.state["in_progress"] = step_name
        self._save_state()

    def record_part_a(self, results: Dict[str, Any]) -> None:
        self.state["part_a"] = results
        if "part_a_full" not in self.state["completed_steps"]:
            self.state["completed_steps"].append("part_a_full")
        self.state["in_progress"] = None
        self._save_state()
        print("\n[Task2StateManager] ✓ Checkpointed Part A operator benchmarks successfully.")

    def record_part_b_model(self, run_key: str, results: Dict[str, Any], state_dict: Optional[Dict[str, Any]] = None) -> None:
        self.state["part_b"][run_key] = results
        step_id = f"part_b_{run_key}"
        if step_id not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_id)
        if state_dict is not None:
            model_path = self.models_dir / f"{run_key}.pt"
            torch.save(state_dict, model_path)
        self.state["in_progress"] = None
        self._save_state()
        print(f"\n[Task2StateManager] ✓ Checkpointed Part B model '{run_key}' successfully.")

    def record_part_c(self, results: Dict[str, Any]) -> None:
        self.state["part_c"] = results
        if "part_c_full" not in self.state["completed_steps"]:
            self.state["completed_steps"].append("part_c_full")
        self.state["in_progress"] = None
        self._save_state()
        print("\n[Task2StateManager] ✓ Checkpointed Part C KV cache evaluation successfully.")

    def get_part_a(self) -> Dict[str, Any]:
        return self.state.get("part_a", {})

    def get_part_b(self) -> Dict[str, Any]:
        return self.state.get("part_b", {})

    def get_part_c(self) -> Dict[str, Any]:
        return self.state.get("part_c", {})
