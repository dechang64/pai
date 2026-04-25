"""
Federated RAG for PAI — Philanthropic Asset Intelligence
=========================================================

Privacy-preserving Retrieval-Augmented Generation across institutions.

Architecture:
    1. Local RAG: each institution indexes its own documents locally
    2. Federated Embedding: FedAvg fine-tunes a shared embedding model
    3. Federated Query: queries are routed to multiple nodes; only
       document IDs + similarity scores leave the institution.

Key design decisions:
    - sentence-transformers for embeddings (runs locally, no API cost)
    - FAISS for vector search (fast, well-maintained, pip-installable)
    - PyTorch for federated fine-tuning (compatible with organoid-fl FedAvg)
    - No raw documents ever leave the local node
"""

from .config import (
    DocumentConfig,
    EmbeddingConfig,
    EmbeddingProvider,
    FederatedConfig,
    FederatedRAGConfig,
    GenerationConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from .document_loader import Document, DocumentLoader
from .embeddings import EmbeddingEngine
from .generator import RAGGenerator
from .local_rag import LocalRAG
from .retriever import Retriever
from .vector_store import SearchResult, VectorStore
from .federated_query import FederatedNode, FederatedQueryRouter, FederatedSearchResult
from .federated_trainer import FederatedEmbeddingTrainer, ClientState, RoundMetrics

__all__ = [
    # Config
    "DocumentConfig",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "FederatedConfig",
    "FederatedRAGConfig",
    "GenerationConfig",
    "RetrievalConfig",
    "VectorStoreConfig",
    # Core
    "Document",
    "DocumentLoader",
    "EmbeddingEngine",
    "RAGGenerator",
    "LocalRAG",
    "Retriever",
    "SearchResult",
    "VectorStore",
    # Federated
    "FederatedNode",
    "FederatedQueryRouter",
    "FederatedSearchResult",
    "FederatedEmbeddingTrainer",
    "ClientState",
    "RoundMetrics",
]

__version__ = "0.4.0"
