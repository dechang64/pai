"""
FAISS-based vector store.

Stores document embeddings and supports similarity search.
Designed to be serializable (save/load) for persistence.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import VectorStoreConfig
from .document_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    document: Document
    score: float  # cosine similarity (0-1 for normalized vectors)
    vector_id: int  # internal FAISS id


class VectorStore:
    """
    FAISS-backed vector store for document retrieval.

    Usage:
        store = VectorStore(config)
        store.add(documents, embeddings)
        results = store.search(query_embedding, top_k=5)
        store.save("path/to/index")
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None) -> None:
        self._config = config or VectorStoreConfig()
        self._index = None
        self._documents: List[Document] = []
        self._id_to_doc: dict[int, int] = {}  # faiss_id -> documents list index

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self._documents)

    def add(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of Document objects.
            embeddings: np.ndarray of shape (len(documents), dimension).

        Raises:
            ValueError: If lengths don't match or store is not empty.
        """
        if len(documents) == 0:
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Document count ({len(documents)}) != embedding count "
                f"({len(embeddings)})"
            )

        if self.size > 0:
            raise ValueError(
                "VectorStore already has data. Use a fresh instance or "
                "clear() first."
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D, got shape {embeddings.shape}"
            )

        if embeddings.shape[1] != self._config.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) != "
                f"configured dimension ({self._config.dimension})"
            )

        self._documents = list(documents)
        self._build_index(embeddings)
        logger.info("Indexed %d documents (dim=%d)", len(documents), embeddings.shape[1])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector, shape (dimension,) or (1, dimension).
            top_k: Number of results to return.
            score_threshold: Minimum similarity score (0-1).

        Returns:
            List of SearchResult, sorted by score descending.
        """
        if self._index is None or self.size == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        actual_k = min(top_k, self.size)
        scores, indices = self._index.search(query, actual_k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            if score < score_threshold:
                continue
            doc_idx = self._id_to_doc.get(int(idx))
            if doc_idx is None:
                continue
            results.append(SearchResult(
                document=self._documents[doc_idx],
                score=float(score),
                vector_id=int(idx),
            ))

        return sorted(results, key=lambda r: r.score, reverse=True)

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._index = None
        self._documents = []
        self._id_to_doc = {}

    def save(self, path: str | Path) -> None:
        """
        Persist the vector store to disk.

        Saves FAISS index + documents as separate files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            logger.warning("Saving empty vector store to %s", path)
            return

        import faiss

        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        logger.info("Saved %d documents to %s", self.size, path)

    def load(self, path: str | Path) -> None:
        """
        Load a previously saved vector store from disk.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found: {path}")

        import faiss

        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "documents.pkl", "rb") as f:
            self._documents = pickle.load(f)

        # Rebuild id mapping
        self._id_to_doc = {i: i for i in range(len(self._documents))}
        logger.info("Loaded %d documents from %s", self.size, path)

    # ── Private helpers ──────────────────────────────────

    def _build_index(self, embeddings: np.ndarray) -> None:
        """Build the FAISS index from embeddings."""
        import faiss

        dimension = embeddings.shape[1]

        if self._config.index_type == "Flat":
            # Exact search — best for < 100k vectors
            self._index = faiss.IndexFlatIP(dimension)  # inner product = cosine for normalized vectors
        elif self._config.index_type == "IVFFlat":
            # Approximate — good for 100k-1M vectors
            nlist = min(self._config.nlist, len(embeddings))
            quantizer = faiss.IndexFlatIP(dimension)
            self._index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
        elif self._config.index_type == "HNSW":
            # Fast approximate — good for 1M+ vectors
            self._index = faiss.IndexHNSWFlat(dimension, self._config.ef_construction, faiss.METRIC_INNER_PRODUCT)
            self._index.hnsw.efSearch = self._config.ef_search
        else:
            raise ValueError(f"Unknown index type: {self._config.index_type}")

        if self._config.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)

        self._index.add(embeddings)
        self._id_to_doc = {i: i for i in range(len(self._documents))}
