"""
RAG retriever — combines embedding + vector store + optional reranking.

This is the single entry point for retrieval.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .config import RetrievalConfig
from .document_loader import Document
from .embeddings import EmbeddingEngine
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    End-to-end retrieval: embed query → search store → (optionally) rerank.

    Usage:
        retriever = Retriever(embedding_engine, vector_store)
        results = retriever.retrieve("How effective is GiveDirectly?", top_k=5)
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        config: Optional[RetrievalConfig] = None,
    ) -> None:
        self._embedder = embedding_engine
        self._store = vector_store
        self._config = config or RetrievalConfig()
        self._reranker = None

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return self._store.size

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Retrieve documents relevant to a query.

        Args:
            query: Natural language query string.
            top_k: Override default top_k.
            score_threshold: Override default score threshold.

        Returns:
            List of SearchResult sorted by relevance.
        """
        if not query.strip():
            return []

        k = top_k or self._config.top_k
        threshold = score_threshold if score_threshold is not None else self._config.score_threshold

        # Embed the query
        query_vec = self._embedder.embed_single(query)

        # Search the vector store
        results = self._store.search(query_vec, top_k=k, score_threshold=threshold)

        # Optional reranking
        if self._config.rerank and len(results) > 1 and self._reranker is not None:
            results = self._rerank(query, results)

        logger.debug(
            "Retrieved %d results for query: '%s' (top_k=%d, threshold=%.2f)",
            len(results), query[:50], k, threshold,
        )
        return results

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[SearchResult]]:
        """
        Retrieve for multiple queries efficiently (batch embedding).

        Args:
            queries: List of query strings.
            top_k: Override default top_k.

        Returns:
            List of result lists, one per query.
        """
        if not queries:
            return []

        k = top_k or self._config.top_k
        query_vecs = self._embedder.embed(queries)

        all_results: List[List[SearchResult]] = []
        for i, query_vec in enumerate(query_vecs):
            results = self._store.search(
                query_vec, top_k=k, score_threshold=self._config.score_threshold
            )
            all_results.append(results)

        return all_results

    def format_context(self, results: List[SearchResult], max_chars: int = 4000) -> str:
        """
        Format search results into a context string for LLM prompting.

        Args:
            results: Search results to format.
            max_chars: Maximum total characters in the context.

        Returns:
            Formatted context string with source citations.
        """
        if not results:
            return ""

        parts: List[str] = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            source = result.document.source
            content = result.document.content
            score = result.score

            entry = f"[Source {i}] ({source}, relevance: {score:.2f})\n{content}"
            if total_chars + len(entry) > max_chars:
                # Truncate the last entry to fit
                remaining = max_chars - total_chars - 10
                if remaining > 100:
                    entry = entry[:remaining] + "\n..."
                    parts.append(entry)
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n\n---\n\n".join(parts)

    def _rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank results using a cross-encoder model.

        Falls back to original ordering if reranker fails.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("sentence-transformers not installed; skipping reranking")
            return results

        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, r.document.content) for r in results]
        scores = self._reranker.predict(pairs)

        # Re-sort by cross-encoder score
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(SearchResult(
                document=result.document,
                score=float(score),
                vector_id=result.vector_id,
            ))
        return sorted(reranked, key=lambda r: r.score, reverse=True)
