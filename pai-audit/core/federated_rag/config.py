"""
Configuration for Federated RAG module.

All magic numbers and defaults live here — nowhere else.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class EmbeddingProvider(Enum):
    """Supported embedding backends."""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the embedding model."""
    provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
    batch_size: int = 32
    max_seq_length: int = 384
    normalize: bool = True  # L2-normalize for cosine similarity
    device: str = "cpu"  # "cpu", "cuda", "mps"


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for the FAISS vector store."""
    index_type: str = "Flat"  # "Flat" (exact), "IVFFlat", "HNSW"
    dimension: int = 384  # must match embedding model output
    nlist: int = 100  # IVF: number of clusters
    nprobe: int = 10  # IVF: clusters to search
    ef_construction: int = 200  # HNSW: build parameter
    ef_search: int = 50  # HNSW: search parameter
    use_gpu: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for retrieval."""
    top_k: int = 10
    score_threshold: float = 0.3  # minimum cosine similarity
    rerank: bool = False  # cross-encoder reranking (slower but better)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for LLM generation."""
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.3  # low for factual answers
    system_prompt: str = (
        "You are PAI (Philanthropic Asset Intelligence), an expert advisor "
        "on charitable giving, nonprofit effectiveness, and philanthropic "
        "strategy. Answer based on the provided context. If the context is "
        "insufficient, say so clearly rather than guessing. Cite sources "
        "when possible."
    )


@dataclass(frozen=True)
class FederatedConfig:
    """Configuration for federated training and querying."""
    rounds: int = 5
    local_epochs: int = 2
    learning_rate: float = 2e-5
    fraction_clients: float = 1.0  # fraction of clients per round
    min_clients: int = 1
    aggregation: str = "fedavg"  # "fedavg" or "fedprox"
    fedprox_mu: float = 0.01  # FedProx regularization
    query_timeout_seconds: float = 10.0
    max_retries: int = 3
    # Training hyperparameters
    train_batch_size: int = 8
    train_max_seq_length: int = 128
    contrastive_temperature: float = 0.05


@dataclass(frozen=True)
class DocumentConfig:
    """Configuration for document loading and chunking."""
    chunk_size: int = 512  # characters
    chunk_overlap: int = 64  # characters
    supported_extensions: tuple = (
        ".pdf", ".txt", ".md", ".csv", ".json",
    )


@dataclass(frozen=True)
class FederatedRAGConfig:
    """Top-level configuration — compose all sub-configs."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    institution_id: str = "local"
    knowledge_base_path: str = "data/knowledge_base"
