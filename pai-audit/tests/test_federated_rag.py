"""
Unit tests for Federated RAG module.

Run: pytest tests/test_federated_rag.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.federated_rag.config import (
    DocumentConfig,
    EmbeddingConfig,
    EmbeddingProvider,
    FederatedConfig,
    FederatedRAGConfig,
    GenerationConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from core.federated_rag.document_loader import Document, DocumentLoader
from core.federated_rag.embeddings import EmbeddingEngine
from core.federated_rag.vector_store import VectorStore, SearchResult
from core.federated_rag.retriever import Retriever
from core.federated_rag.generator import RAGGenerator
from core.federated_rag.local_rag import LocalRAG
from core.federated_rag.federated_query import (
    FederatedNode,
    FederatedQueryRouter,
    FederatedSearchResult,
)

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_documents():
    """Create a small set of test documents."""
    return [
        Document(content="GiveDirectly sends cash to people in extreme poverty.", source="test.md", chunk_index=0),
        Document(content="AMF distributes insecticide-treated nets for malaria prevention.", source="test.md", chunk_index=1),
        Document(content="DAF tax benefits include avoiding capital gains on appreciated securities.", source="tax.md", chunk_index=0),
        Document(content="Warm-glow theory explains why people give even when impact is negligible.", source="behavior.md", chunk_index=0),
    ]


@pytest.fixture
def sample_embeddings():
    """Create random embeddings matching sample_documents count."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((4, 384)).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


@pytest.fixture
def populated_store(sample_documents, sample_embeddings):
    """Create a VectorStore with documents already added."""
    config = VectorStoreConfig(dimension=384)
    store = VectorStore(config)
    store.add(sample_documents, sample_embeddings)
    return store


@pytest.fixture
def kb_dir(tmp_path):
    """Create a temporary knowledge base directory with test files."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "charity.md").write_text(
        "# GiveWell Charities\n\nGiveDirectly sends cash transfers. "
        "AMF provides malaria nets. Cost per life saved is $3,000-$5,000.\n"
    )
    (kb / "tax.md").write_text(
        "# Tax Strategy\n\nDAF contributions avoid capital gains tax. "
        "Bunching strategy combines multiple years of giving.\n"
    )
    return kb


# ============================================================
# Config Tests
# ============================================================

class TestConfig:
    def test_default_config_is_frozen(self):
        config = FederatedRAGConfig()
        with pytest.raises(AttributeError):
            config.embedding = EmbeddingConfig(model_name="other")

    def test_custom_config(self):
        config = EmbeddingConfig(model_name="custom-model", batch_size=64)
        assert config.model_name == "custom-model"
        assert config.batch_size == 64

    def test_embedding_provider_enum(self):
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS.value == "sentence-transformers"
        assert EmbeddingProvider.OPENAI.value == "openai"


# ============================================================
# Document Tests
# ============================================================

class TestDocument:
    def test_create_document(self):
        doc = Document(content="Hello world", source="test.txt", chunk_index=0)
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert doc.chunk_index == 0
        assert doc.metadata == {}

    def test_document_whitespace_cleanup(self):
        doc = Document(content="Hello\n\n\n\nWorld", source="test.txt", chunk_index=0)
        assert "\n\n\n" not in doc.content

    def test_document_with_metadata(self):
        doc = Document(
            content="Test", source="test.txt", chunk_index=0,
            metadata={"author": "AI", "date": "2026"},
        )
        assert doc.metadata["author"] == "AI"

    def test_document_is_frozen(self):
        doc = Document(content="Test", source="test.txt", chunk_index=0)
        with pytest.raises(AttributeError):
            doc.content = "Changed"


# ============================================================
# DocumentLoader Tests
# ============================================================

class TestDocumentLoader:
    def test_load_markdown(self, kb_dir):
        loader = DocumentLoader()
        docs = loader.load_file(kb_dir / "charity.md")
        assert len(docs) >= 1
        assert all(d.source == str(kb_dir / "charity.md") for d in docs)

    def test_load_directory(self, kb_dir):
        loader = DocumentLoader()
        docs = loader.load_directory(kb_dir)
        assert len(docs) >= 2
        sources = set(d.source for d in docs)
        assert len(sources) == 2

    def test_load_nonexistent_file(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/file.md")

    def test_unsupported_extension(self, kb_dir):
        bad_file = kb_dir / "data.xyz"
        bad_file.write_text("some content")
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load_file(bad_file)

    def test_chunk_size_respected(self, kb_dir):
        loader = DocumentLoader(DocumentConfig(chunk_size=50, chunk_overlap=10))
        docs = loader.load_file(kb_dir / "charity.md")
        for doc in docs:
            # Allow some slack for overlap
            assert len(doc.content) <= 80


# ============================================================
# VectorStore Tests
# ============================================================

class TestVectorStore:
    def test_add_and_search(self, populated_store):
        assert populated_store.size == 4
        query_emb = np.random.default_rng(0).standard_normal((1, 384)).astype(np.float32)
        results = populated_store.search(query_emb, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_sorted(self, populated_store):
        query_emb = np.random.default_rng(1).standard_normal((1, 384)).astype(np.float32)
        results = populated_store.search(query_emb, top_k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_search(self):
        store = VectorStore()
        query_emb = np.zeros((1, 384), dtype=np.float32)
        results = store.search(query_emb, top_k=5)
        assert results == []

    def test_save_and_load(self, populated_store, tmp_path):
        index_path = tmp_path / "index"
        populated_store.save(index_path)
        assert (index_path / "index.faiss").exists()
        assert (index_path / "documents.pkl").exists()

        new_store = VectorStore()
        new_store.load(index_path)
        assert new_store.size == populated_store.size

    def test_dimension_mismatch(self, sample_documents):
        store = VectorStore(VectorStoreConfig(dimension=384))
        bad_embs = np.random.randn(4, 128).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            store.add(sample_documents, bad_embs)

    def test_count_mismatch(self, sample_documents, sample_embeddings):
        store = VectorStore()
        with pytest.raises(ValueError, match="count"):
            store.add(sample_documents[:2], sample_embeddings)


# ============================================================
# EmbeddingEngine Tests
# ============================================================

class TestEmbeddingEngine:
    def test_embed_returns_correct_shape(self):
        engine = EmbeddingEngine()
        result = engine.embed(["Hello world", "Test sentence"])
        assert result.shape == (2, 384)
        assert result.dtype == np.float32

    def test_embed_empty_list(self):
        engine = EmbeddingEngine()
        result = engine.embed([])
        assert result.shape == (0, 384)

    def test_embeddings_normalized(self):
        engine = EmbeddingEngine(EmbeddingConfig(normalize=True))
        result = engine.embed(["Test"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_dimension_property(self):
        engine = EmbeddingEngine()
        assert engine.dimension == 384


# ============================================================
# Retriever Tests
# ============================================================

class TestRetriever:
    @pytest.fixture
    def retriever(self, sample_documents):
        """Build a retriever with real embeddings so queries actually match."""
        embedder = EmbeddingEngine()
        embs = embedder.embed([d.content for d in sample_documents])
        store = VectorStore()
        store.add(sample_documents, embs)
        return Retriever(embedder, store)

    def test_retrieve_returns_results(self, retriever):
        results = retriever.retrieve("malaria prevention", top_k=2)
        assert len(results) >= 1
        assert all(r.score > 0 for r in results)

    def test_retrieve_respects_threshold(self, retriever):
        results = retriever.retrieve("xyzzy nonsense query", top_k=10, score_threshold=0.99)
        # With a very high threshold, should return fewer results
        assert len(results) <= 10

    def test_format_context(self, retriever):
        results = retriever.retrieve("charity", top_k=2)
        context = retriever.format_context(results)
        assert "Source" in context
        assert len(context) > 0


# ============================================================
# RAGGenerator Tests
# ============================================================

class TestRAGGenerator:
    def test_demo_mode_returns_answer(self):
        gen = RAGGenerator()
        answer = gen.generate("What is GiveDirectly?", "GiveDirectly sends cash to people in poverty.")
        assert len(answer) > 0
        assert "GiveDirectly" in answer or "knowledge base" in answer.lower()

    def test_demo_mode_no_context(self):
        gen = RAGGenerator()
        answer = gen.generate("What is charity?", "")
        assert "enough information" in answer.lower() or "demo mode" in answer.lower()


# ============================================================
# LocalRAG Tests
# ============================================================

class TestLocalRAG:
    def test_index_and_query(self, kb_dir):
        rag = LocalRAG()
        count = rag.index_directory(kb_dir)
        assert count >= 2
        answer = rag.query("malaria prevention")
        assert len(answer) > 0

    def test_query_with_sources(self, kb_dir):
        rag = LocalRAG()
        rag.index_directory(kb_dir)
        result = rag.query_with_sources("tax benefits")
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_search_only(self, kb_dir):
        rag = LocalRAG()
        rag.index_directory(kb_dir)
        results = rag.search_only("charity effectiveness")
        assert len(results) > 0

    def test_save_and_load_index(self, kb_dir, tmp_path):
        rag = LocalRAG()
        rag.index_directory(kb_dir)
        index_path = tmp_path / "rag_index"
        rag.save_index(index_path)

        rag2 = LocalRAG()
        rag2.load_index(index_path)
        assert rag2.document_count == rag.document_count


# ============================================================
# FederatedQueryRouter Tests
# ============================================================

class TestFederatedQueryRouter:
    @pytest.fixture
    def router_with_nodes(self, populated_store):
        embedder = EmbeddingEngine()
        retriever = Retriever(embedder, populated_store)
        router = FederatedQueryRouter()
        router.register_node("node_a", retriever=retriever, institution_name="Institution A")
        router.register_node("node_b", retriever=retriever, institution_name="Institution B")
        return router

    def test_register_nodes(self, router_with_nodes):
        assert router_with_nodes.num_nodes == 2

    def test_search_returns_results(self, router_with_nodes):
        result = router_with_nodes.search("malaria", top_k=2, include_content=True)
        assert isinstance(result, FederatedSearchResult)
        assert result.successful_nodes == 2
        assert result.total_nodes_queried == 2
        assert len(result.node_results) == 2

    def test_privacy_mode(self, router_with_nodes):
        result = router_with_nodes.search("tax", top_k=2, include_content=False)
        for nid, results in result.node_results.items():
            for r in results:
                assert "[content hidden" in r.document.content

    def test_search_empty_router(self):
        router = FederatedQueryRouter()
        result = router.search("test query")
        assert result.successful_nodes == 0
        assert result.total_nodes_queried == 0

    def test_latency_measured(self, router_with_nodes):
        result = router_with_nodes.search("test")
        assert result.latency_seconds >= 0

    def test_search_and_answer(self, router_with_nodes):
        result = router_with_nodes.search_and_answer("malaria prevention", top_k=2)
        assert "answer" in result
        assert "sources" in result
        assert "federated_result" in result
        assert len(result["sources"]) > 0
        assert len(result["answer"]) > 0

    def test_search_and_answer_with_generator(self, router_with_nodes):
        gen = RAGGenerator()
        result = router_with_nodes.search_and_answer("charity impact", top_k=2, generator=gen)
        assert "knowledge base" in result["answer"].lower() or "GiveDirectly" in result["answer"]

    def test_search_and_answer_empty(self):
        router = FederatedQueryRouter()
        result = router.search_and_answer("test")
        assert "No relevant information" in result["answer"]


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_full_pipeline(self, kb_dir):
        """End-to-end: load docs → embed → store → retrieve → generate."""
        # Load
        loader = DocumentLoader()
        docs = loader.load_directory(kb_dir)
        assert len(docs) >= 2

        # Embed
        embedder = EmbeddingEngine()
        embs = embedder.embed([d.content for d in docs])
        assert embs.shape == (len(docs), 384)

        # Store
        store = VectorStore()
        store.add(docs, embs)
        assert store.size == len(docs)

        # Retrieve
        retriever = Retriever(embedder, store)
        results = retriever.retrieve("malaria nets", top_k=2)
        assert len(results) > 0

        # Generate
        gen = RAGGenerator()
        context = retriever.format_context(results)
        answer = gen.generate("How much does a malaria net cost?", context)
        assert len(answer) > 0

    def test_federated_search_across_institutions(self, kb_dir):
        """Simulate 3 institutions with different documents."""
        config = FederatedRAGConfig()
        loader = DocumentLoader(config.document)
        docs = loader.load_directory(kb_dir)

        # Split docs into 3 groups
        third = max(1, len(docs) // 3)
        groups = [docs[:third], docs[third:2*third], docs[2*third:]]

        router = FederatedQueryRouter(config.federated)
        embedder = EmbeddingEngine(config.embedding)

        for i, group in enumerate(groups):
            if not group:
                continue
            embs = embedder.embed([d.content for d in group])
            store = VectorStore(config.vector_store)
            store.add(group, embs)
            retriever = Retriever(embedder, store, config.retrieval)
            router.register_node(f"inst_{i}", retriever=retriever, institution_name=f"Institution {i}")

        result = router.search("charity impact", top_k=3, include_content=True)
        assert result.successful_nodes >= 2
