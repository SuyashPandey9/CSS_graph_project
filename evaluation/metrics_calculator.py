"""Metrics calculator for V3 comparisons.

Standard RAG evaluation metrics including semantic similarity,
faithfulness, and context relevance.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from core.types import Corpus


_WORD_RE = re.compile(r"[A-Za-z]+")

# Cached embedder for semantic metrics
_EMBEDDER = None


def _get_embedder():
    """Get cached embedder for semantic similarity calculations."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from tools.neural_embedder import NeuralEmbedder
        _EMBEDDER = NeuralEmbedder()
    return _EMBEDDER


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    return sum(a * b for a, b in zip(vec1, vec2))


def token_count(text: str) -> int:
    """Count word-like tokens in a text."""
    return len(_WORD_RE.findall(text))


# =============================================================================
# ANSWER QUALITY METRICS
# =============================================================================

def relevance_score(query: str, answer: str) -> float:
    """Semantic relevance using embedding cosine similarity.
    
    Standard metric: Measures how well the answer addresses the query.
    
    Args:
        query: The user's question
        answer: The generated answer
    
    Returns:
        Cosine similarity [0, 1] between query and answer embeddings
    """
    if not query or not answer:
        return 0.0
    
    embedder = _get_embedder()
    query_vec = embedder.embed(query)
    answer_vec = embedder.embed(answer)
    
    sim = _cosine_similarity(query_vec, answer_vec)
    return max(0.0, min(1.0, sim))


def faithfulness_score(answer: str, context: str) -> float:
    """Measure if answer is grounded in the provided context.
    
    Standard RAGAS metric: High scores mean answer claims are supported by context.
    Uses semantic similarity as a proxy for NLI-based faithfulness.
    
    Args:
        answer: The generated answer
        context: The retrieved context used to generate the answer
    
    Returns:
        Faithfulness score [0, 1], higher = more grounded in context
    """
    if not answer or not context:
        return 0.0
    
    embedder = _get_embedder()
    
    # Split answer into sentences for fine-grained checking
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        return 0.0
    
    context_vec = embedder.embed(context)
    
    # Check each sentence's similarity to context
    total_score = 0.0
    for sentence in sentences:
        sent_vec = embedder.embed(sentence)
        sim = _cosine_similarity(sent_vec, context_vec)
        total_score += max(0.0, sim)
    
    return min(1.0, total_score / len(sentences))


def context_relevance_score(query: str, context: str) -> float:
    """Measure if retrieved context is relevant to the query.
    
    Standard RAGAS metric: High scores mean context is useful for answering.
    
    Args:
        query: The user's question  
        context: The retrieved context
    
    Returns:
        Context relevance [0, 1], higher = more relevant context
    """
    if not query or not context:
        return 0.0
    
    embedder = _get_embedder()
    query_vec = embedder.embed(query)
    context_vec = embedder.embed(context)
    
    sim = _cosine_similarity(query_vec, context_vec)
    return max(0.0, min(1.0, sim))


def answer_completeness(query: str, answer: str) -> float:
    """Estimate if answer fully addresses the query.
    
    Uses answer length relative to query complexity as a proxy.
    Very short answers for complex queries score lower.
    
    Args:
        query: The user's question
        answer: The generated answer
    
    Returns:
        Completeness score [0, 1]
    """
    if not answer:
        return 0.0
    
    query_tokens = token_count(query)
    answer_tokens = token_count(answer)
    
    # Expect at least 3x query length for a complete answer
    expected_min = max(query_tokens * 3, 20)
    
    if answer_tokens >= expected_min:
        return 1.0
    
    return min(1.0, answer_tokens / expected_min)


def llm_judge_score(query: str, answer: str, context: str = "") -> dict:
    """LLM-as-Judge evaluation - STANDARD metric used in RAG research.
    
    Uses GPT to evaluate answer quality on multiple dimensions.
    This is the gold standard for RAG evaluation, used in papers like RAGAS.
    
    Reference: https://arxiv.org/abs/2309.15217
    
    Args:
        query: The user's question
        answer: The generated answer
        context: Optional context used to generate answer
    
    Returns:
        Dict with scores for: correctness, completeness, conciseness, overall
    """
    import os
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "No OPENAI_API_KEY set", "overall": 0.0}
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are an expert evaluator for question-answering systems.

Given a question and an answer, rate the answer on these dimensions (1-5 scale):

1. **Correctness**: Is the answer factually accurate based on general knowledge?
2. **Completeness**: Does the answer fully address all parts of the question?
3. **Conciseness**: Is the answer appropriately detailed without unnecessary content?
4. **Overall Quality**: Your overall assessment of the answer quality.

Question: {query}

Answer: {answer}

Respond ONLY with a JSON object like:
{{"correctness": 4, "completeness": 5, "conciseness": 3, "overall": 4}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper, fast enough for judging
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        
        import json
        result_text = response.choices[0].message.content.strip()
        # Extract JSON from response
        if "{" in result_text and "}" in result_text:
            json_str = result_text[result_text.find("{"):result_text.rfind("}")+1]
            scores = json.loads(json_str)
            # Normalize to 0-1 scale
            return {
                "correctness": scores.get("correctness", 3) / 5.0,
                "completeness": scores.get("completeness", 3) / 5.0,
                "conciseness": scores.get("conciseness", 3) / 5.0,
                "overall": scores.get("overall", 3) / 5.0,
            }
        else:
            return {"error": "Invalid response format", "overall": 0.6}
    except Exception as e:
        return {"error": str(e), "overall": 0.0}


# =============================================================================
# LENGTH METRICS
# =============================================================================

def answer_length(answer: str) -> int:
    """Count tokens in the answer."""
    return token_count(answer)


def context_length(context: str) -> int:
    """Count tokens in the context."""
    return token_count(context)


# =============================================================================
# LEGACY / UTILITY METRICS
# =============================================================================

def lexical_relevance_score(query: str, answer: str) -> float:
    """Legacy lexical relevance via token overlap ratio."""
    query_tokens = set(t.lower() for t in _WORD_RE.findall(query))
    answer_tokens = set(t.lower() for t in _WORD_RE.findall(answer))
    if not query_tokens or not answer_tokens:
        return 0.0
    overlap = query_tokens.intersection(answer_tokens)
    return len(overlap) / len(query_tokens)


def precision_recall(
    retrieved_ids: Iterable[str], relevant_ids: Iterable[str]
) -> Tuple[float, float]:
    """Compute retrieval precision and recall.
    
    Note: Requires ground-truth relevant_ids to be meaningful.
    """
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    if not retrieved_set:
        return 0.0, 0.0
    true_pos = len(retrieved_set.intersection(relevant_set))
    precision = true_pos / len(retrieved_set)
    recall = true_pos / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def token_efficiency(prompt: str, answer: str) -> float:
    """Compute a simple token efficiency ratio."""
    prompt_tokens = token_count(prompt)
    answer_tokens = token_count(answer)
    if prompt_tokens == 0:
        return 0.0
    return answer_tokens / prompt_tokens


def corpus_token_count(corpus: Corpus) -> int:
    """Count tokens across the corpus for initial cost estimation."""
    total = 0
    for doc in corpus.documents:
        total += token_count(f"{doc.title} {doc.text}")
    return total


def run_cost_tokens(prompt: str, answer: str) -> int:
    """Estimate run cost as prompt + answer token count."""
    return token_count(prompt) + token_count(answer)


# =============================================================================
# OFFICIAL RAGAS METRICS (Academic Standard)
# Reference: https://arxiv.org/abs/2309.15217 (RAGAS Paper)
#
# Fix P2a: Corrected metric implementations to match the RAGAS paper.
#   - ragas_faithfulness: answer claims supported by context (was mislabeled as context_recall)
#   - ragas_context_recall: ground truth claims attributable to context (NOW REQUIRES GROUND TRUTH)
#   - ragas_context_precision: fraction of context sentences relevant to query (NEW)
#   - ragas_answer_relevancy: reverse-question cosine similarity (unchanged)
#   - ragas_aspect_coverage: query aspect coverage (supplementary, unchanged)
# =============================================================================


def _get_openai_client():
    """Get OpenAI client (shared helper for RAGAS metrics)."""
    import os
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def ragas_answer_relevancy(query: str, answer: str, n_questions: int = 3) -> float:
    """RAGAS Answer Relevancy — Official Calculation.
    
    Method (from RAGAS paper):
    1. Generate N questions that the answer could be responding to
    2. Compute cosine similarity between original query and each generated question
    3. Return mean similarity score
    
    Higher score = answer is more relevant to the query.
    Does NOT require ground truth.
    
    Reference: https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html
    """
    client = _get_openai_client()
    if client is None or not answer.strip():
        return 0.0
    
    embedder = _get_embedder()
    
    prompt = f"""Given this answer, generate {n_questions} different questions that this answer could be responding to.

Answer: {answer}

Output ONLY the questions, one per line, no numbering or prefixes."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        
        generated_text = response.choices[0].message.content.strip()
        generated_questions = [q.strip() for q in generated_text.split("\n") if q.strip()]
        
        if not generated_questions:
            return 0.0
        
        query_vec = embedder.embed(query)
        
        similarities = []
        for gen_q in generated_questions[:n_questions]:
            gen_vec = embedder.embed(gen_q)
            sim = _cosine_similarity(query_vec, gen_vec)
            similarities.append(max(0.0, sim))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"[RAGAS Answer Relevancy] Error: {e}")
        return 0.0


def ragas_faithfulness(answer: str, context: str) -> float:
    """RAGAS Faithfulness — Official Calculation.
    
    Method (from RAGAS paper):
    1. Extract factual claims from the generated answer
    2. For each claim, check if it can be inferred from the context
    3. Return proportion of claims supported by context
    
    Higher score = answer is more grounded in context (less hallucination).
    Does NOT require ground truth.
    
    Reference: https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html
    
    NOTE: This was previously mislabeled as ragas_context_recall.
    """
    client = _get_openai_client()
    if client is None or not answer.strip() or not context.strip():
        return 0.0
    
    # Step 1: Extract claims from answer
    claim_prompt = f"""Extract the key factual claims from this answer. 
Output each claim on a separate line. Be specific and granular.

Answer: {answer}

Output ONLY the claims, one per line:"""

    try:
        claim_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": claim_prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        
        claims_text = claim_response.choices[0].message.content.strip()
        claims = [c.strip() for c in claims_text.split("\n") if c.strip()]
        
        if not claims:
            return 1.0  # No claims = nothing to verify
        
        # Step 2: Check each claim against context
        supported_count = 0
        claims_to_check = claims[:5]  # Limit to control API cost
        
        for claim in claims_to_check:
            verify_prompt = f"""Given the context below, determine if the following claim can be inferred from the context.

Context:
{context[:3000]}

Claim: {claim}

Answer with ONLY "Yes" if the claim is supported by the context, or "No" if not."""

            verify_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            
            verdict = verify_response.choices[0].message.content.strip().lower()
            if "yes" in verdict:
                supported_count += 1
        
        return supported_count / len(claims_to_check)
        
    except Exception as e:
        print(f"[RAGAS Faithfulness] Error: {e}")
        return 0.0


def ragas_context_recall(ground_truth: str, context: str) -> float:
    """RAGAS Context Recall — Official Calculation (REQUIRES GROUND TRUTH).
    
    Method (from RAGAS paper):
    1. Extract factual claims from the GROUND TRUTH answer
    2. For each claim, check if it can be attributed to the retrieved context
    3. Return proportion of ground truth claims covered by context
    
    Higher score = the retrieval system found the right information.
    
    THIS IS THE KEY METRIC FOR EVALUATING RETRIEVAL QUALITY.
    It measures: "Did the retrieval system pull the right chunks?"
    
    Reference: https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html
    
    Args:
        ground_truth: The expert-annotated correct answer (NOT the generated answer)
        context: The retrieved context
    
    Returns:
        Score [0, 1] - proportion of ground truth claims found in context.
        Returns -1.0 if ground truth is missing (sentinel for "not computed").
    """
    if not ground_truth or not ground_truth.strip():
        return -1.0  # Sentinel: ground truth required but missing
    
    client = _get_openai_client()
    if client is None or not context.strip():
        return 0.0
    
    # Step 1: Extract claims from ground truth
    claim_prompt = f"""Extract the key factual claims from this reference answer.
Output each claim on a separate line. Be specific and granular.

Reference Answer: {ground_truth}

Output ONLY the claims, one per line:"""

    try:
        claim_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": claim_prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        
        claims_text = claim_response.choices[0].message.content.strip()
        claims = [c.strip() for c in claims_text.split("\n") if c.strip()]
        
        if not claims:
            return 1.0  # No claims in ground truth = trivially satisfied
        
        # Step 2: Check each ground truth claim against context
        attributable_count = 0
        claims_to_check = claims[:8]  # Ground truth claims are precious, check more
        
        for claim in claims_to_check:
            verify_prompt = f"""Given the retrieved context below, determine if the following ground truth claim can be attributed to (found in) the context.

Retrieved Context:
{context[:3000]}

Ground Truth Claim: {claim}

Answer with ONLY "Yes" if the claim is attributable to the context, or "No" if not."""

            verify_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            
            verdict = verify_response.choices[0].message.content.strip().lower()
            if "yes" in verdict:
                attributable_count += 1
        
        return attributable_count / len(claims_to_check)
        
    except Exception as e:
        print(f"[RAGAS Context Recall] Error: {e}")
        return 0.0


def ragas_context_precision(query: str, context: str) -> float:
    """RAGAS Context Precision — Official Calculation (NEW).
    
    Method (from RAGAS paper):
    1. Split context into individual chunks/sentences
    2. For each chunk, check if it's relevant to the query
    3. Return proportion of context chunks that are relevant
    
    Higher score = less noise in the retrieved context.
    Does NOT require ground truth.
    
    This measures: "Is the retrieval system returning useful chunks, or noise?"
    V3 should score higher because the CSS optimizer prunes irrelevant nodes.
    
    Reference: https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html
    """
    client = _get_openai_client()
    if client is None or not query.strip() or not context.strip():
        return 0.0
    
    # Split context into chunks (by section separators or paragraph breaks)
    chunks = re.split(r'\n\s*[-=]{3,}\s*\n|\n\n+', context)
    chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 20]
    
    if not chunks:
        return 0.0
    
    # Limit to first 10 chunks to control cost
    chunks_to_check = chunks[:10]
    
    try:
        # Batch check: ask LLM to classify each chunk's relevance
        chunks_numbered = "\n".join(
            f"[Chunk {i+1}]: {chunk[:300]}" for i, chunk in enumerate(chunks_to_check)
        )
        
        prompt = f"""Given the user's question, classify each context chunk as "Relevant" or "Irrelevant".
A chunk is Relevant if it contains information useful for answering the question.

Question: {query}

{chunks_numbered}

For each chunk, respond with ONLY the chunk number and verdict, one per line, like:
1: Relevant
2: Irrelevant
3: Relevant"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        relevant_count = 0
        for line in result_text.split("\n"):
            line_lower = line.strip().lower()
            if "relevant" in line_lower and "irrelevant" not in line_lower:
                relevant_count += 1
        
        return relevant_count / len(chunks_to_check)
        
    except Exception as e:
        print(f"[RAGAS Context Precision] Error: {e}")
        return 0.0


def ragas_aspect_coverage(query: str, answer: str) -> float:
    """RAGAS-style Aspect Coverage (supplementary metric).
    
    Method:
    1. Extract key aspects/topics from the query
    2. Check if each aspect is addressed in the answer
    3. Return proportion of aspects covered
    
    This measures how completely the answer addresses all parts of the query.
    Does NOT require ground truth.
    """
    client = _get_openai_client()
    if client is None or not query.strip() or not answer.strip():
        return 0.0
    
    prompt = f"""Analyze how well this answer covers the question's key aspects.

Question: {query}

Answer: {answer}

For each key aspect of the question:
1. Identify 2-4 key aspects the question is asking about
2. For each aspect, determine if the answer addresses it (Yes/No)

Respond with JSON like:
{{"aspects": [{{"aspect": "...", "covered": true}}, ...]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        
        result_text = response.choices[0].message.content.strip()
        
        import json
        if "{" in result_text:
            json_str = result_text[result_text.find("{"):result_text.rfind("}")+1]
            result = json.loads(json_str)
            aspects = result.get("aspects", [])
            
            if not aspects:
                return 1.0
            
            covered = sum(1 for a in aspects if a.get("covered", False))
            return covered / len(aspects)
        else:
            return 0.5  # Fallback
            
    except Exception as e:
        print(f"[RAGAS Aspect Coverage] Error: {e}")
        return 0.0

