"""
Local RAG pipeline — the single entry point for local retrieval + generation.

Combines DocumentLoader, EmbeddingEngine, VectorStore, Retriever, and
RAGGenerator into one easy-to-use interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .config import FederatedRAGConfig
from .document_loader import Document, DocumentLoader
from .embeddings import EmbeddingEngine
from .generator import RAGGenerator
from .retriever import Retriever
from .vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class LocalRAG:
    """
    Complete local RAG pipeline.

    Usage:
        rag = LocalRAG()
        rag.index_directory("data/knowledge_base")
        answer = rag.query("How effective is GiveDirectly?")
    """

    def __init__(self, config: Optional[FederatedRAGConfig] = None) -> None:
        self._config = config or FederatedRAGConfig()
        self._loader = DocumentLoader(self._config.document)
        self._embedder = EmbeddingEngine(self._config.embedding)
        self._store = VectorStore(self._config.vector_store)
        self._retriever = Retriever(
            self._embedder, self._store, self._config.retrieval
        )
        self._generator = RAGGenerator(self._config.generation)

    @property
    def document_count(self) -> int:
        """Number of indexed documents."""
        return self._store.size

    def index_documents(self, documents: List[Document]) -> None:
        """
        Index a list of pre-loaded documents.

        Args:
            documents: List of Document objects to index.
        """
        if not documents:
            logger.warning("No documents to index")
            return

        logger.info("Embedding %d documents...", len(documents))
        texts = [doc.content for doc in documents]
        embeddings = self._embedder.embed(texts)

        self._store.clear()
        self._store.add(documents, embeddings)
        logger.info("Indexing complete: %d documents", self._store.size)

    def index_directory(self, directory: str | Path) -> int:
        """
        Load and index all supported files in a directory.

        Args:
            directory: Path to directory containing documents.

        Returns:
            Number of documents indexed.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        all_documents = self._loader.load_directory(str(directory))
        if not all_documents:
            logger.warning("No documents found in %s", directory)
            return 0

        self.index_documents(all_documents)
        return self._store.size

    def index_texts(
        self,
        texts: List[str],
        source: str = "inline",
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Index raw text strings directly (no file loading).

        Args:
            texts: List of text strings.
            source: Identifier for the source.
            metadata: Optional metadata to attach to all chunks.

        Returns:
            Number of chunks indexed.
        """
        documents = [
            Document(content=text, source=source, chunk_index=i, metadata=metadata or ())
            for i, text in enumerate(texts)
        ]
        self.index_documents(documents)
        return self._store.size

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = False,
    ) -> str:
        """
        Query the knowledge base and generate an answer.

        Args:
            question: Natural language question.
            top_k: Number of documents to retrieve.
            return_sources: If True, append source citations.

        Returns:
            Generated answer string.
        """
        results = self._retriever.retrieve(question, top_k=top_k)
        context = self._retriever.format_context(results)

        answer = self._generator.generate(question, context)

        if return_sources and results:
            sources = self._format_sources(results)
            answer = f"{answer}\n\n---\n### Sources\n{sources}"

        return answer

    def query_with_sources(
        self, question: str, top_k: Optional[int] = None
    ) -> dict:
        """
        Query and return both answer and structured source information.

        Returns:
            Dict with 'answer', 'sources', and 'retrieval_scores'.
        """
        results = self._retriever.retrieve(question, top_k=top_k)
        context = self._retriever.format_context(results)
        answer = self._generator.generate(question, context)

        return {
            "answer": answer,
            "sources": [
                {
                    "source": r.document.source,
                    "chunk_index": r.document.chunk_index,
                    "score": round(r.score, 4),
                    "content_preview": r.document.content[:200],
                }
                for r in results
            ],
            "context_used": bool(context),
        }

    def search_only(
        self, question: str, top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search without generating an answer (retrieval only).

        Useful for debugging or custom generation logic.
        """
        return self._retriever.retrieve(question, top_k=top_k)

    def save_index(self, path: str | Path) -> None:
        """Persist the vector index to disk."""
        self._store.save(path)

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved vector index."""
        self._store.load(path)

    @staticmethod
    def _format_sources(results: List[SearchResult]) -> str:
        """Format search results into a readable source list."""
        lines = []
        for i, r in enumerate(results, 1):
            preview = r.document.content[:100].replace("\n", " ")
            lines.append(
                f"{i}. **{r.document.source}** (score: {r.score:.3f}) — {preview}..."
            )
        return "\n".join(lines)
