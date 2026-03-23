"""Core data structures for V3.

Defines Node, Edge, Graph, and Corpus using Pydantic models.
Updated to match V3 spec with node attributes: confidence p(v), embedding z(v).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Node(BaseModel):
    """A graph node representing a concept or document chunk.
    
    V3 Spec Attributes:
        ϕ(v) = (p(v), t_v, z(v))
        - p(v) ∈ [0,1]: confidence score
        - z(v) ∈ ℝ^d: embedding vector (computed lazily if not provided)
    """

    id: str = Field(..., description="Unique node identifier.")
    text: str = Field(..., description="Node text content.")
    metadata: dict = Field(default_factory=dict, description="Arbitrary node metadata.")
    
    # V3 Spec attributes
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Node confidence p(v) ∈ [0,1]")
    embedding: Optional[List[float]] = Field(default=None, description="Pre-computed embedding z(v)")


class Edge(BaseModel):
    """A directed edge connecting two nodes.
    
    V3 Spec: ψ(e) ∈ [0,1] is the edge confidence/weight.
    """

    source: str = Field(..., description="Source node id.")
    target: str = Field(..., description="Target node id.")
    relation: str = Field(..., description="Relationship label.")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Edge confidence ψ(e) ∈ [0,1]")


class Graph(BaseModel):
    """A simple directed graph with helper methods."""

    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node if it is not already present."""

        if any(existing.id == node.id for existing in self.nodes):
            return
        self.nodes.append(node)

    def get_nodes(self) -> List[Node]:
        """Return all nodes in the graph."""

        return list(self.nodes)
    
    def token_count(self) -> int:
        """Count total tokens in the graph (for Cost feature)."""
        import re
        word_re = re.compile(r"[A-Za-z]+")
        total = 0
        for node in self.nodes:
            total += len(word_re.findall(node.text))
        return total


class CorpusDocument(BaseModel):
    """A single document in the immutable corpus."""

    id: str
    title: str
    text: str
    source: Optional[str] = None


class Corpus(BaseModel):
    """Immutable corpus container."""

    documents: List[CorpusDocument] = Field(default_factory=list)
