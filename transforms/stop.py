"""Stop operation for V3 graph transformation engine."""

from __future__ import annotations

from core.frozen_state import FrozenState
from core.types import Graph


def stop(graph: Graph, state: FrozenState) -> Graph:
    """Return graph unchanged to indicate termination."""

    return graph
