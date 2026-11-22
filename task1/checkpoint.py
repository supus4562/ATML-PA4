"""task1/checkpoint.py — State manager for persistent volume checkpointing & auto-resume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class Task1StateManager:
    """Tracks completed conditions, stores intermediate results, and enables zero-loss auto-resume."""

    def __init__(self, checkpoint_dir: Path | str, modal_volume: Optional[Any] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
                print(f"[StateManager] Warning: could not parse {self.state_file}: {e}")
        return {
            "completed_conditions": [],
            "in_progress": None,
            "metrics": {},
        }

    def _save_state(self) -> None:
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        temp_file.replace(self.state_file)

        # Also write clean results summary
        with open(self.results_file, "w", encoding="utf-8") as f:
            json.dump(self.state["metrics"], f, indent=2)

        self.commit_volume()

    def commit_volume(self) -> None:
        """Commit modal volume changes to durable storage if running under Modal."""
        if self.modal_volume is not None:
            try:
                self.modal_volume.commit()
            except Exception as e:
                print(f"[StateManager] Volume commit warning: {e}")

    def is_completed(self, condition_name: str, min_samples: int = 0) -> bool:
        """Check if condition has already been successfully evaluated with sufficient samples."""
        if condition_name not in self.state.get("completed_conditions", []):
            return False
        if min_samples > 0:
            m = self.state.get("metrics", {}).get(condition_name, {})
            if m.get("hellaswag_samples", 0) < min_samples:
                return False
        return True

    def mark_in_progress(self, condition_name: str) -> None:
        self.state["in_progress"] = condition_name
        self._save_state()

    def record_completed(self, condition_name: str, metrics: Dict[str, Any]) -> None:
        """Record condition completion and store evaluated metrics."""
        if condition_name not in self.state["completed_conditions"]:
            self.state["completed_conditions"].append(condition_name)
        self.state["in_progress"] = None
        self.state["metrics"][condition_name] = metrics
        self._save_state()
        print(f"\n[StateManager] ✓ Checkpointed condition '{condition_name}' successfully.")

    def get_metrics(self, condition_name: str) -> Optional[Dict[str, Any]]:
        return self.state["metrics"].get(condition_name)

    def get_all_results(self) -> Dict[str, Any]:
        return self.state["metrics"]

    def save_tensor_checkpoint(self, name: str, state_dict: Dict[str, Any]) -> Path:
        """Save tensor weights/masks/scales to persistent volume."""
        import torch
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(state_dict, ckpt_path)
        self.commit_volume()
        return ckpt_path

    def load_tensor_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Load tensor checkpoint if exists."""
        import torch
        ckpt_path = self.checkpoint_dir / f"{name}.pt"
        if ckpt_path.exists():
            return torch.load(ckpt_path, map_location="cpu")
        return None
