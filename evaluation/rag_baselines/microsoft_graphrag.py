"""Microsoft GraphRAG wrapper for V3 comparison.

Integrates the official Microsoft graphrag package with our comparison framework.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.types import Corpus
from evaluation.cost_tracker import CostTracker, count_tokens_approx
from storage.corpus_store import load_active_corpus


class MicrosoftGraphRAG:
    """Wrapper for Microsoft's official GraphRAG package.
    
    GraphRAG requires:
    1. Indexing phase (expensive, one-time): Extract entities, build graph, create summaries
    2. Query phase (cheap, per-query): Search the indexed graph
    """
    
    def __init__(
        self,
        corpus: Corpus | None = None,
        workspace_dir: str = "./graphrag_workspace",
        cost_tracker: CostTracker | None = None,
        sample_size: int | None = 100,  # Limit docs to reduce indexing cost
    ) -> None:
        """Initialize GraphRAG wrapper.
        
        Args:
            corpus: Corpus to index (loads active if None)
            workspace_dir: Directory for GraphRAG workspace
            cost_tracker: Optional cost tracker
            sample_size: Number of documents to sample (None = all)
        """
        self._corpus = corpus or load_active_corpus()
        self._workspace = Path(workspace_dir)
        self._cost_tracker = cost_tracker
        self._sample_size = sample_size
        self._is_indexed = False
        
        # Check if already indexed
        if (self._workspace / "output").exists():
            self._is_indexed = True
    
    def _prepare_workspace(self) -> None:
        """Prepare the GraphRAG workspace with corpus documents."""
        input_dir = self._workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample documents if needed
        docs = self._corpus.documents
        if self._sample_size and len(docs) > self._sample_size:
            import random
            docs = random.sample(list(docs), self._sample_size)
        
        # Write documents as text files
        for i, doc in enumerate(docs):
            doc_path = input_dir / f"doc_{i:04d}.txt"
            content = f"{doc.title}\n\n{doc.text}"
            doc_path.write_text(content, encoding="utf-8")
            
            # Track initial cost (document preparation)
            if self._cost_tracker:
                self._cost_tracker.add_initial(
                    count_tokens_approx(content),
                    "corpus_preparation"
                )
    
    def _create_settings(self) -> None:
        """Create GraphRAG settings file."""
        settings_content = """
encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4o-mini
  model_supports_json: true

parallelization:
  stagger: 0.3
  num_threads: 10

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small

chunks:
  size: 1200
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: input

cache:
  type: file
  base_dir: cache

reporting:
  type: file
  base_dir: reports

storage:
  type: file
  base_dir: output
  
entity_extraction:
  max_gleanings: 1

community_reports:
  max_length: 2000
  max_input_length: 8000

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  top_k_mapped_entities: 10
  top_k_relationships: 10
  max_tokens: 12000

global_search:
  max_tokens: 12000
  data_max_tokens: 12000
  map_max_tokens: 1000
  reduce_max_tokens: 2000
  concurrency: 32
"""
        settings_path = self._workspace / "settings.yaml"
        settings_path.write_text(settings_content.strip())
        
        # Create .env file for API key
        env_content = f"GRAPHRAG_API_KEY={os.getenv('OPENAI_API_KEY', '')}\n"
        env_path = self._workspace / ".env"
        env_path.write_text(env_content)
    
    def index(self, force: bool = False) -> Dict[str, Any]:
        """Run GraphRAG indexing.
        
        This is the expensive one-time operation that:
        1. Extracts entities from all documents
        2. Builds the knowledge graph
        3. Detects communities
        4. Generates community summaries
        
        Args:
            force: If True, re-index even if already indexed
        
        Returns:
            Indexing statistics
        """
        if self._is_indexed and not force:
            return {"status": "already_indexed", "skipped": True}
        
        start_time = time.time()
        
        # Prepare workspace
        if force and self._workspace.exists():
            shutil.rmtree(self._workspace)
        
        self._prepare_workspace()
        self._create_settings()
        
        # Run GraphRAG indexing via CLI
        import subprocess
        
        result = subprocess.run(
            ["python", "-m", "graphrag.index", "--root", str(self._workspace)],
            capture_output=True,
            text=True,
            cwd=str(self._workspace),
            env={**os.environ, "GRAPHRAG_API_KEY": os.getenv("OPENAI_API_KEY", "")},
        )
        
        latency = time.time() - start_time
        
        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr,
                "latency_seconds": latency,
            }
        
        self._is_indexed = True
        
        # Estimate indexing cost from report files if available
        indexing_tokens = self._estimate_indexing_cost()
        if self._cost_tracker:
            self._cost_tracker.add_initial(indexing_tokens, "graphrag_indexing")
        
        return {
            "status": "success",
            "latency_seconds": latency,
            "estimated_tokens": indexing_tokens,
        }
    
    def _estimate_indexing_cost(self) -> int:
        """Estimate total tokens used during indexing."""
        # Read from GraphRAG's report files if available
        report_dir = self._workspace / "reports"
        total_tokens = 0
        
        if report_dir.exists():
            for report_file in report_dir.glob("*.json"):
                try:
                    import json
                    data = json.loads(report_file.read_text())
                    # GraphRAG reports include token counts
                    total_tokens += data.get("total_tokens", 0)
                except Exception:
                    pass
        
        # Fallback estimate: ~500 tokens per document for entity extraction
        if total_tokens == 0:
            num_docs = len(list((self._workspace / "input").glob("*.txt")))
            total_tokens = num_docs * 500  # Conservative estimate
        
        return total_tokens
    
    def query(self, query: str, method: str = "local") -> Dict[str, Any]:
        """Query the indexed GraphRAG.
        
        Args:
            query: The question to answer
            method: "local" (entity-focused) or "global" (community summaries)
        
        Returns:
            Answer and metadata
        """
        if not self._is_indexed:
            return {"error": "Not indexed. Call index() first."}
        
        start_time = time.time()
        
        if self._cost_tracker:
            self._cost_tracker.record_query()
        
        # Run GraphRAG query via CLI
        import subprocess
        
        result = subprocess.run(
            [
                "python", "-m", "graphrag.query",
                "--root", str(self._workspace),
                "--method", method,
                "--query", query,
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GRAPHRAG_API_KEY": os.getenv("OPENAI_API_KEY", "")},
        )
        
        latency = time.time() - start_time
        
        if result.returncode != 0:
            return {
                "error": result.stderr,
                "latency_seconds": latency,
            }
        
        answer = result.stdout.strip()
        
        # Track query cost
        query_tokens = count_tokens_approx(query)
        answer_tokens = count_tokens_approx(answer)
        total_run_tokens = query_tokens + answer_tokens + 500  # + context tokens
        
        if self._cost_tracker:
            self._cost_tracker.add_run(total_run_tokens, f"{method}_query")
        
        return {
            "answer": answer,
            "latency_seconds": latency,
            "method": method,
            "tokens_used": total_run_tokens,
        }
    
    def answer(self, query: str, method: str = "local") -> Dict[str, Any]:
        """Alias for query() to match TraditionalRAG interface."""
        result = self.query(query, method)
        
        # Standardize output format
        return {
            "answer": result.get("answer", result.get("error", "No answer")),
            "latency_seconds": result.get("latency_seconds", 0),
            "context": f"GraphRAG {method} search",
            "retrieved_doc_ids": [],  # GraphRAG doesn't expose this directly
        }


def check_graphrag_available() -> bool:
    """Check if graphrag package is properly installed with all dependencies."""
    try:
        import graphrag
        # Check for actual submodules that indicate full installation
        from graphrag import index  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception:
        return False


def estimate_indexing_cost(corpus: Corpus, sample_size: int = 100) -> Dict[str, Any]:
    """Estimate the cost of indexing a corpus without actually running it.
    
    Returns:
        Estimated tokens and USD cost
    """
    docs = corpus.documents
    if sample_size and len(docs) > sample_size:
        num_docs = sample_size
    else:
        num_docs = len(docs)
    
    # Estimates based on GraphRAG paper and benchmarks:
    # - Entity extraction: ~500 tokens/doc
    # - Community detection: minimal
    # - Community summarization: ~200 tokens/community (est. 1 community per 5 docs)
    
    entity_tokens = num_docs * 500
    summary_tokens = (num_docs // 5) * 200
    total_tokens = entity_tokens + summary_tokens
    
    # Using GPT-4o-mini pricing (~$0.15/1M input, $0.60/1M output)
    estimated_usd = (total_tokens / 1_000_000) * 0.375  # Average of input/output
    
    return {
        "num_documents": num_docs,
        "estimated_tokens": total_tokens,
        "estimated_usd": estimated_usd,
        "breakdown": {
            "entity_extraction": entity_tokens,
            "community_summaries": summary_tokens,
        }
    }
