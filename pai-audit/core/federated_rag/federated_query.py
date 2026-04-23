"""
Federated query router — routes queries to multiple institutions
and aggregates results without exposing raw documents.

Privacy guarantee: only document IDs and similarity scores leave
each institution. The actual document content stays local.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import FederatedConfig
from .document_loader import Document
from .retriever import Retriever
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class FederatedNode:
    """
    Represents a remote federated institution node.

    In production, this would connect via gRPC (organoid-fl).
    For demo/simulation, it wraps a local Retriever.
    """
    node_id: str
    institution_name: str
    retriever: Optional[Retriever] = None
    endpoint: Optional[str] = None  # gRPC URL for production
    is_local: bool = True  # True = simulated, False = remote gRPC


@dataclass
class FederatedSearchResult:
    """Aggregated result from federated search."""
    query: str
    results: List[SearchResult]
    node_results: Dict[str, List[SearchResult]] = field(default_factory=dict)
    total_nodes_queried: int = 0
    successful_nodes: int = 0
    latency_seconds: float = 0.0


class FederatedQueryRouter:
    """
    Routes queries to multiple federated nodes and aggregates results.

    Privacy model:
        1. Query is sent to each node (the question itself is not sensitive)
        2. Each node returns ONLY (document_id, similarity_score, metadata)
        3. Document content is NEVER transmitted
        4. The requesting node can then fetch full content from selected
           documents via a separate authorized request

    Usage:
        router = FederatedQueryRouter()
        router.register_node("give_well", retriever_gw)
        router.register_node("hospital_a", retriever_ha)
        result = router.search("rare disease treatment costs")
    """

    def __init__(self, config: Optional[FederatedConfig] = None) -> None:
        self._config = config or FederatedConfig()
        self._nodes: Dict[str, FederatedNode] = {}

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    def get_nodes(self) -> Dict[str, FederatedNode]:
        """Return a copy of registered nodes."""
        return dict(self._nodes)

    def register_node(
        self,
        node_id: str,
        retriever: Optional[Retriever] = None,
        institution_name: str = "",
        endpoint: Optional[str] = None,
    ) -> None:
        """
        Register a federated node.

        Args:
            node_id: Unique node identifier.
            retriever: Local Retriever instance (for simulation/demo).
            institution_name: Human-readable name.
            endpoint: gRPC URL (for production use).
        """
        is_local = endpoint is None
        self._nodes[node_id] = FederatedNode(
            node_id=node_id,
            institution_name=institution_name or node_id,
            retriever=retriever,
            endpoint=endpoint,
            is_local=is_local,
        )
        logger.info(
            "Registered node '%s' (%s, %s)",
            node_id, institution_name, "local" if is_local else endpoint,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        include_content: bool = False,
        score_threshold: float = 0.0,
    ) -> FederatedSearchResult:
        """
        Search across all federated nodes.

        Args:
            query: Search query string.
            top_k: Results to retrieve per node.
            include_content: If True, include document content (local demo only).
                In production, this should be False to preserve privacy.
            score_threshold: Minimum similarity score (0-1). Default 0 to
                return all results; callers can filter downstream.

        Returns:
            FederatedSearchResult with aggregated results.
        """
        start_time = time.time()
        all_results: List[SearchResult] = []
        node_results: Dict[str, List[SearchResult]] = {}
        successful = 0

        for node_id, node in self._nodes.items():
            try:
                results = self._query_node(
                    node, query, top_k, include_content, score_threshold
                )
                node_results[node_id] = results
                all_results.extend(results)
                successful += 1
            except Exception as e:
                logger.error("Node '%s' query failed: %s", node_id, e)
                node_results[node_id] = []

        # Sort by score across all nodes
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:top_k * 2]  # cap total results

        latency = time.time() - start_time

        return FederatedSearchResult(
            query=query,
            results=all_results,
            node_results=node_results,
            total_nodes_queried=len(self._nodes),
            successful_nodes=successful,
            latency_seconds=round(latency, 3),
        )

    def search_with_content(
        self,
        query: str,
        top_k: int = 5,
    ) -> FederatedSearchResult:
        """
        Search and include document content.

        For demo/testing only. In production, use search() + separate
        authorized content fetch.
        """
        return self.search(query, top_k, include_content=True)

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5,
        generator=None,
    ) -> dict:
        """
        Federated search + LLM answer generation.

        Retrieves context from all nodes, then generates an answer
        using the provided RAGGenerator (or demo mode if None).

        Args:
            query: User's question.
            top_k: Results per node.
            generator: Optional RAGGenerator instance.

        Returns:
            Dict with 'answer', 'sources', 'federated_result'.
        """
        # Always fetch content for generation (local nodes only)
        fed_result = self.search(query, top_k=top_k, include_content=True)

        if not fed_result.results:
            return {
                "answer": "No relevant information found across any institution.",
                "sources": [],
                "federated_result": fed_result,
            }

        # Build context from top results
        context_parts = []
        sources = []
        for r in fed_result.results[:top_k]:
            if "[content hidden" in r.document.content:
                continue
            context_parts.append(
                f"[Source: {r.document.source}, relevance: {r.score:.2f}]\n"
                f"{r.document.content}"
            )
            sources.append({
                "source": r.document.source,
                "score": round(r.score, 4),
                "preview": r.document.content[:150].replace("\n", " "),
            })

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        if generator is not None:
            answer = generator.generate(query, context)
        else:
            # Demo fallback
            answer = (
                f"Based on federated search across {fed_result.successful_nodes} "
                f"institutions:\n\n"
            )
            for i, src in enumerate(sources[:5], 1):
                answer += f"{i}. **{src['source']}** (score: {src['score']})\n"
                answer += f"   {src['preview']}...\n\n"
            answer += "---\n*Demo mode: connect an LLM API for AI-generated answers.*"

        return {
            "answer": answer,
            "sources": sources,
            "federated_result": fed_result,
        }

    # ── Private methods ─────────────────────────────────

    def _query_node(
        self,
        node: FederatedNode,
        query: str,
        top_k: int,
        include_content: bool,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Query a single node.

        For local nodes: call the retriever directly.
        For remote nodes: would use gRPC (placeholder).
        """
        if node.is_local:
            if node.retriever is None:
                raise RuntimeError(
                    f"Local node '{node.node_id}' has no retriever"
                )
            results = node.retriever.retrieve(
                query, top_k=top_k, score_threshold=score_threshold
            )

            if not include_content:
                # Strip content for privacy — return only metadata
                results = [
                    SearchResult(
                        document=Document(
                            content="[content hidden — federated privacy]",
                            source=r.document.source,
                            chunk_index=r.document.chunk_index,
                            metadata=r.document.metadata,
                        ),
                        score=r.score,
                        vector_id=r.vector_id,
                    )
                    for r in results
                ]

            return results
        else:
            # Placeholder for gRPC call to organoid-fl
            return self._query_grpc(node, query, top_k)

    @staticmethod
    def _query_grpc(
        node: FederatedNode, query: str, top_k: int
    ) -> List[SearchResult]:
        """
        Query a remote node via gRPC.

        This would integrate with organoid-fl's gRPC service.
        For now, returns empty results as a placeholder.
        """
        logger.warning(
            "gRPC query to '%s' not yet implemented. "
            "Use local nodes for demo.",
            node.endpoint,
        )
        return []
