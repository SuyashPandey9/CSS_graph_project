"""Cost tracking utilities for RAG comparison.

Tracks initial cost (indexing/setup) and run cost (per query) for all systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Approximate token costs for different LLM providers
# Cost per 1000 tokens (input/output combined average)
TOKEN_COSTS_USD = {
    "gpt-4": 0.03,
    "gpt-4-turbo": 0.01,
    "gpt-3.5-turbo": 0.0015,
    "gemini-1.5-flash": 0.000075,  # Very cheap
    "gemini-1.5-pro": 0.00125,
    "gemini-2.0-flash": 0.0001,
    "local": 0.0,  # Free for local models
}


@dataclass
class CostTracker:
    """Track token costs for a RAG system.
    
    Attributes:
        system_name: Name of the system (e.g., "V3", "Traditional RAG")
        initial_tokens: One-time indexing/setup tokens
        run_tokens: Tokens used per query (cumulative)
        num_queries: Number of queries run
        model: LLM model used (for $ estimation)
    """
    
    system_name: str
    model: str = "gemini-2.0-flash"
    initial_tokens: int = 0
    run_tokens: int = 0
    num_queries: int = 0
    details: Dict[str, int] = field(default_factory=dict)
    
    def add_initial(self, tokens: int, description: str = "indexing") -> None:
        """Add initial/indexing tokens (one-time cost)."""
        self.initial_tokens += tokens
        self.details[f"initial_{description}"] = self.details.get(f"initial_{description}", 0) + tokens
    
    def add_run(self, tokens: int, description: str = "query") -> None:
        """Add per-query run tokens."""
        self.run_tokens += tokens
        self.details[f"run_{description}"] = self.details.get(f"run_{description}", 0) + tokens
    
    def record_query(self) -> None:
        """Record that a query was performed."""
        self.num_queries += 1
    
    def total_tokens(self) -> int:
        """Total tokens used (initial + run)."""
        return self.initial_tokens + self.run_tokens
    
    def projected_cost_for_n_queries(self, n: int) -> int:
        """Project total tokens for N queries.
        
        Formula: initial + (avg_run_per_query × N)
        """
        if self.num_queries == 0:
            return self.initial_tokens
        avg_run_per_query = self.run_tokens / self.num_queries
        return int(self.initial_tokens + (avg_run_per_query * n))
    
    def estimate_usd(self, tokens: Optional[int] = None) -> float:
        """Estimate cost in USD."""
        if tokens is None:
            tokens = self.total_tokens()
        cost_per_1k = TOKEN_COSTS_USD.get(self.model, 0.001)
        return (tokens / 1000) * cost_per_1k
    
    def report(self) -> str:
        """Generate a cost report string."""
        total = self.total_tokens()
        usd = self.estimate_usd()
        avg_per_query = self.run_tokens / max(1, self.num_queries)
        
        lines = [
            f"=== {self.system_name} Cost Report ===",
            f"Model: {self.model}",
            f"Initial Cost: {self.initial_tokens:,} tokens",
            f"Run Cost: {self.run_tokens:,} tokens ({self.num_queries} queries)",
            f"Avg per Query: {avg_per_query:,.0f} tokens",
            f"Total: {total:,} tokens (~${usd:.4f})",
        ]
        
        if self.details:
            lines.append("\nBreakdown:")
            for key, value in self.details.items():
                lines.append(f"  - {key}: {value:,} tokens")
        
        return "\n".join(lines)


@dataclass
class ComparisonCostTracker:
    """Track and compare costs across multiple systems."""
    
    systems: Dict[str, CostTracker] = field(default_factory=dict)
    
    def add_system(self, name: str, model: str = "gemini-2.0-flash") -> CostTracker:
        """Add a new system to track."""
        tracker = CostTracker(system_name=name, model=model)
        self.systems[name] = tracker
        return tracker
    
    def get(self, name: str) -> Optional[CostTracker]:
        """Get a tracker by name."""
        return self.systems.get(name)
    
    def comparison_table(self, n_queries: int = 10) -> str:
        """Generate a comparison table for N queries."""
        lines = [
            f"\n{'=' * 70}",
            f"COST COMPARISON (projected for {n_queries} queries)",
            f"{'=' * 70}",
            f"{'System':<20} {'Initial':>10} {'Run/Query':>12} {'Total':>12} {'Est. $':>10}",
            f"{'-' * 70}",
        ]
        
        for name, tracker in self.systems.items():
            avg_per_query = tracker.run_tokens / max(1, tracker.num_queries)
            projected = tracker.projected_cost_for_n_queries(n_queries)
            usd = tracker.estimate_usd(projected)
            
            lines.append(
                f"{name:<20} {tracker.initial_tokens:>10,} {avg_per_query:>12,.0f} "
                f"{projected:>12,} ${usd:>9.4f}"
            )
        
        lines.append(f"{'=' * 70}\n")
        return "\n".join(lines)


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3).
    
    This is a rough estimate. For exact counts, use tiktoken.
    """
    import re
    words = len(re.findall(r'\b\w+\b', text))
    return int(words * 1.3)
