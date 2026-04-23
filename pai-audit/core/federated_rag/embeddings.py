"""
Embedding model wrapper.

Supports sentence-transformers (local, free) and OpenAI API.
All embeddings are L2-normalized by default for cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .config import EmbeddingConfig, EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Unified interface for text embedding."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self._config = config or EmbeddingConfig()
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Output dimension of the embedding model."""
        if self._dimension is None:
            self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        self._ensure_model()

        if self._config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return self._embed_local(texts)
        elif self._config.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(texts)
        else:
            raise ValueError(f"Unknown provider: {self._config.provider}")

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns shape (dimension,)."""
        result = self.embed([text])
        return result[0]

    # ── Provider implementations ────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        if self._config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            self._load_sentence_transformer()
        elif self._config.provider == EmbeddingProvider.OPENAI:
            self._dimension = 1536  # text-embedding-3-small
            logger.info("Using OpenAI embeddings (dimension=%d)", self._dimension)

    def _load_sentence_transformer(self) -> None:
        """Load a sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(
            "Loading sentence-transformer model '%s' on %s...",
            self._config.model_name,
            self._config.device,
        )
        self._model = SentenceTransformer(
            self._config.model_name,
            device=self._config.device,
        )
        # Override max length if configured
        if self._config.max_seq_length:
            self._model.max_seq_length = self._config.max_seq_length

        dim_fn = getattr(self._model, "get_embedding_dimension", None)
        if dim_fn is None:
            dim_fn = self._model.get_sentence_embedding_dimension
        self._dimension = dim_fn()
        logger.info("Model loaded: dimension=%d", self._dimension)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Embed using sentence-transformers."""
        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed using OpenAI API."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Cannot use OpenAI embeddings."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        client = OpenAI(api_key=api_key)

        # OpenAI has a batch limit of 2048 texts
        all_embeddings: List[np.ndarray] = []
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small",
            )
            batch_emb = np.array(
                [item.embedding for item in response.data], dtype=np.float32
            )
            all_embeddings.append(batch_emb)

        result = np.vstack(all_embeddings)
        if self._config.normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # avoid division by zero
            result = result / norms
        return result
