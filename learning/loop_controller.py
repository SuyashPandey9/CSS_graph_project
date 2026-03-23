"""Master loop controller for CSS Learning Loop.

Orchestrates the 3-phase optimization:
1. Feature Discovery → select best feature set
2. CSS Weight Optimization → tune feature weights
3. Edge Weight Optimization → tune graph construction

Can be run as a script:
    python -m learning.loop_controller --trials 50 --queries 10
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from learning.results_logger import ResultsLogger


class LearningLoopController:
    """Orchestrates the 3-phase CSS learning loop."""
    
    def __init__(self, results_dir: str = "learning/results"):
        self._logger = ResultsLogger(results_dir)
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_loop(
        self,
        n_trials: int = 50,
        n_queries: int = 10,
        skip_feature_discovery: bool = False,
        selected_features: List[str] = None,
    ) -> Dict:
        """Run the full 3-phase learning loop.
        
        Args:
            n_trials: Number of optimization trials per phase
            n_queries: Number of eval queries per trial
            skip_feature_discovery: Skip Phase 1 (use all existing features)
            selected_features: Override feature set (skips Phase 1)
        
        Returns:
            Dict with final optimized config
        """
        from core.frozen_state import get_shared_state
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"  CSS LEARNING LOOP - Full Optimization Pipeline")
        print(f"  Trials per phase: {n_trials}  |  Queries: {n_queries}")
        print(f"{'='*70}\n")
        
        # Build shared state once
        print("Loading corpus and building FAISS index...")
        state = get_shared_state(refresh=True)
        print(f"Corpus loaded: {len(state.corpus.documents)} documents\n")
        
        # ============================================================
        # Phase 1: Feature Discovery
        # ============================================================
        if selected_features:
            print("Phase 1: SKIPPED (features provided)")
            features = selected_features
        elif skip_feature_discovery:
            from css.calculator import WEIGHTS
            features = list(WEIGHTS.keys())
            features = [f.replace("_penalty", "") for f in features]
            print(f"Phase 1: SKIPPED (using default features: {features})")
        else:
            from learning.feature_discovery import greedy_forward_selection
            features = greedy_forward_selection(
                n_queries=n_queries, state=state, logger=self._logger
            )
        
        print(f"\nSelected features: {features}\n")
        
        # ============================================================
        # Phase 2: CSS Weight Optimization
        # ============================================================
        from learning.weight_optimizer import optimize_weights_optuna
        css_weights = optimize_weights_optuna(
            n_trials=n_trials,
            n_queries=n_queries,
            selected_features=features,
            state=state,
            logger=self._logger,
        )
        
        print(f"\nOptimized CSS weights: {css_weights}\n")
        
        # ============================================================
        # Phase 3: Edge Weight Optimization
        # ============================================================
        from learning.edge_optimizer import optimize_edge_weights
        edge_weights = optimize_edge_weights(
            css_weights=css_weights,
            n_trials=n_trials,
            n_queries=n_queries,
            selected_features=features,
            state=state,
            logger=self._logger,
        )
        
        print(f"\nOptimized edge weights: {edge_weights}\n")
        
        # ============================================================
        # Save final config
        # ============================================================
        total_time = time.time() - start_time
        
        final_config = {
            "selected_features": features,
            "css_weights": css_weights,
            "edge_weights": edge_weights,
            "metadata": {
                "n_trials": n_trials,
                "n_queries": n_queries,
                "total_time_seconds": total_time,
                "run_id": self._logger.run_id,
            }
        }
        
        config_path = self._results_dir / f"{self._logger.run_id}_final_config.json"
        config_path.write_text(json.dumps(final_config, indent=2))
        
        print(f"\n{'='*70}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Config saved: {config_path}")
        print(f"  Logs saved: {self._results_dir}")
        print(f"{'='*70}\n")
        
        # Print summary
        print(self._logger.summary("feature_discovery"))
        print()
        print(self._logger.summary("css_weights"))
        print()
        print(self._logger.summary("edge_weights"))
        
        return final_config
    
    def run_single_trial(self) -> Dict:
        """Quick smoke test: run 1 trial of weight optimization.
        
        Useful for verifying the pipeline works end-to-end.
        """
        return self.run_full_loop(
            n_trials=2,
            n_queries=3,
            skip_feature_discovery=True,
        )
    
    def load_config(self, config_path: str) -> Dict:
        """Load a previously saved optimization config."""
        return json.loads(Path(config_path).read_text())
    
    def apply_config(self, config: Dict) -> None:
        """Apply optimized config to the global CSS weights and edge params.
        
        WARNING: This modifies global state. Use with caution.
        
        Args:
            config: Dict with 'css_weights' and 'edge_weights' keys
        """
        import css.calculator as css_mod
        import graph.edge_builder as edge_mod
        
        if "css_weights" in config:
            css_mod.WEIGHTS.update(config["css_weights"])
            print(f"[Config] Applied CSS weights: {css_mod.WEIGHTS}")
        
        if "edge_weights" in config:
            edge_mod.DEFAULT_EDGE_PARAMS.update(config["edge_weights"])
            print(f"[Config] Applied edge params: {edge_mod.DEFAULT_EDGE_PARAMS}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CSS Learning Loop")
    parser.add_argument("--trials", type=int, default=50, help="Trials per phase")
    parser.add_argument("--queries", type=int, default=10, help="Eval queries per trial")
    parser.add_argument("--skip-discovery", action="store_true", help="Skip feature discovery")
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-trial smoke test")
    args = parser.parse_args()
    
    controller = LearningLoopController()
    
    if args.smoke_test:
        controller.run_single_trial()
    else:
        controller.run_full_loop(
            n_trials=args.trials,
            n_queries=args.queries,
            skip_feature_discovery=args.skip_discovery,
        )


if __name__ == "__main__":
    main()
