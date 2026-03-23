"""Query graph builder for V3.

V3 Spec: Build initial graph G₀ from parsed query entities and relationships.
"""

from __future__ import annotations

from typing import List, Tuple

from core.types import Edge, Graph, Node
from tools.parser_stub import ParserStub


def build_query_graph(query: str, parser: ParserStub | None = None) -> Graph:
    """Build initial graph G₀ from parsed query.
    
    V3 Spec:
        (E_x, R_x, Q(x)) = f_parse(x)
        G₀ = graph where nodes = E_x, edges = R_x
    
    Args:
        query: The user's query string
        parser: Parser to use (creates default if None)
    
    Returns:
        Initial graph with nodes from entities and edges from relationships
    """
    if parser is None:
        parser = ParserStub()
    
    parsed = parser.extract(query)
    entities: List[str] = parsed.get("entities", [])
    relations: List[Tuple[str, str, str]] = parsed.get("relations", [])
    
    # Create nodes from entities
    nodes: List[Node] = []
    for entity in entities:
        node = Node(
            id=f"query_{entity}",
            text=entity,
            metadata={"type": "query_entity", "source": "query_parse"},
            confidence=1.0,  # Query entities have full confidence
        )
        nodes.append(node)
    
    # Create edges from relationships
    edges: List[Edge] = []
    for e1, rel, e2 in relations:
        edge = Edge(
            source=f"query_{e1}",
            target=f"query_{e2}",
            relation=rel,
            weight=1.0,  # Query relations have full confidence
        )
        edges.append(edge)
    
    return Graph(nodes=nodes, edges=edges)


def get_query_entities(query: str, parser: ParserStub | None = None) -> List[str]:
    """Extract just the entities from a query.
    
    Useful for expand operations that need to find similar documents.
    """
    if parser is None:
        parser = ParserStub()
    
    parsed = parser.extract(query)
    return parsed.get("entities", [])


def get_subqueries(query: str, parser: ParserStub | None = None) -> List[str]:
    """Extract subqueries from a query.
    
    V3 Spec: Q(x) = decomposed subquestions
    """
    if parser is None:
        parser = ParserStub()
    
    parsed = parser.extract(query)
    return parsed.get("subqueries", [query])
