"""Simple GraphRAG baseline for V3 comparison.

Builds a lightweight knowledge graph from the fixed corpus using
rule-based entity extraction, then answers via Gemini.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx

from core.types import Corpus
from llm.llm_client import generate_text
from storage.corpus_store import corpus_to_documents, load_active_corpus
from tools.parser_stub import ParserStub


class SimpleGraphRAG:
    """Minimal GraphRAG pipeline: corpus -> entities -> graph -> answer."""

    def __init__(self, corpus: Corpus | None = None) -> None:
        """Initialize parser and build graph."""

        self._parser = ParserStub()
        corpus = corpus or load_active_corpus()
        self._documents = corpus_to_documents(corpus)
        self._graph = self._build_graph(self._documents)

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities using parser stub nouns."""

        parsed = self._parser.extract(text)
        return set(parsed["nouns"])

    def _build_graph(self, documents: List[dict]) -> nx.Graph:
        """Build an entity co-occurrence graph from documents."""

        graph = nx.Graph()
        entity_docs: Dict[str, Set[str]] = defaultdict(set)

        for doc in documents:
            entities = self._extract_entities(doc["text"])
            for entity in entities:
                entity_docs[entity].add(doc["id"])

            # Add co-occurrence edges for entities within the same doc
            entity_list = list(entities)
            for i, ent_a in enumerate(entity_list):
                for ent_b in entity_list[i + 1 :]:
                    graph.add_edge(ent_a, ent_b, weight=graph.get_edge_data(ent_a, ent_b, {"weight": 0})["weight"] + 1)

        # Attach document IDs as node attributes
        for entity, doc_ids in entity_docs.items():
            graph.nodes[entity]["documents"] = list(doc_ids)

        return graph

    def _retrieve_subgraph(self, query: str, top_k: int = 5) -> nx.Graph:
        """Retrieve a subgraph based on query entities."""

        query_entities = self._extract_entities(query)
        if not query_entities:
            return self._graph.subgraph(list(self._graph.nodes)[:top_k])

        # Collect neighbors for query entities
        nodes_to_include: Set[str] = set(query_entities)
        for entity in query_entities:
            if entity in self._graph:
                neighbors = list(self._graph.neighbors(entity))
                nodes_to_include.update(neighbors[:top_k])

        return self._graph.subgraph(nodes_to_include)

    def answer(self, query: str, top_k: int = 5) -> dict:
        """Run GraphRAG: retrieve subgraph, aggregate context, generate answer."""

        start_time = time.time()

        subgraph = self._retrieve_subgraph(query, top_k=top_k)
        nodes = list(subgraph.nodes)

        # Collect relevant document texts from node-linked docs
        doc_ids = set()
        for node in nodes:
            doc_ids.update(subgraph.nodes[node].get("documents", []))

        context_chunks = []
        for doc in self._documents:
            if doc["id"] in doc_ids:
                context_chunks.append(f"{doc['title']}: {doc['text']}")

        context = "\n".join(context_chunks) if context_chunks else "No relevant context found."

        prompt = (
            "Use the following graph-based context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        answer_text = generate_text(prompt).text.strip()
        latency = time.time() - start_time

        return {
            "answer": answer_text,
            "nodes_used": nodes,
            "documents_used": list(doc_ids),
            "latency_seconds": latency,
            "context_size": len(context_chunks),
            "context": context,
        }
