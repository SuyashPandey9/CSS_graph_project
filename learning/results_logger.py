"""Results logger for CSS learning loop trials.

Logs each optimization trial's parameters, scores, and timestamps
to a JSON file for analysis and comparison.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResultsLogger:
    """Logs optimization trial results to JSON files."""
    
    def __init__(self, results_dir: str = "learning/results"):
        """Initialize logger.
        
        Args:
            results_dir: Directory to store result files
        """
        self._dir = Path(results_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trials: List[Dict[str, Any]] = []
        self._start_time = time.time()
    
    @property
    def run_id(self) -> str:
        return self._run_id
    
    def log_trial(
        self,
        trial_id: int,
        phase: str,
        params: Dict[str, float],
        per_query_scores: List[Dict[str, Any]],
        mean_reward: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single optimization trial.
        
        Args:
            trial_id: Sequential trial number
            phase: "feature_discovery", "css_weights", or "edge_weights"
            params: The parameter values tested in this trial
            per_query_scores: List of per-query score dicts
            mean_reward: Mean reward across all queries
            extra: Optional additional metadata
        """
        entry = {
            "trial_id": trial_id,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self._start_time,
            "params": params,
            "mean_reward": mean_reward,
            "per_query_scores": per_query_scores,
        }
        if extra:
            entry["extra"] = extra
        
        self._trials.append(entry)
        
        # Auto-save after each trial
        self._save()
    
    def log_best(self, phase: str, params: Dict[str, float], reward: float) -> None:
        """Log the best result for a phase."""
        best_entry = {
            "phase": phase,
            "best_params": params,
            "best_reward": reward,
            "timestamp": datetime.now().isoformat(),
        }
        
        best_path = self._dir / f"{self._run_id}_best_{phase}.json"
        best_path.write_text(json.dumps(best_entry, indent=2))
        print(f"[Logger] Saved best {phase} result: reward={reward:.4f}")
    
    def get_best_trial(self, phase: str = None) -> Optional[Dict]:
        """Get the best trial so far (optionally filtered by phase)."""
        trials = self._trials
        if phase:
            trials = [t for t in trials if t["phase"] == phase]
        
        if not trials:
            return None
        
        return max(trials, key=lambda t: t["mean_reward"])
    
    def get_all_trials(self, phase: str = None) -> List[Dict]:
        """Get all trials (optionally filtered by phase)."""
        if phase:
            return [t for t in self._trials if t["phase"] == phase]
        return list(self._trials)
    
    def summary(self, phase: str = None) -> str:
        """Print a summary of trials."""
        trials = self.get_all_trials(phase)
        if not trials:
            return "No trials logged."
        
        rewards = [t["mean_reward"] for t in trials]
        best = self.get_best_trial(phase)
        
        lines = [
            f"Phase: {phase or 'all'}",
            f"Trials: {len(trials)}",
            f"Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]",
            f"Best trial #{best['trial_id']}: reward={best['mean_reward']:.4f}",
        ]
        return "\n".join(lines)
    
    def _save(self) -> None:
        """Save all trials to JSON file."""
        path = self._dir / f"{self._run_id}_optimization_log.json"
        path.write_text(json.dumps(self._trials, indent=2, default=str))
