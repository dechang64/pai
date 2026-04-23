"""
Federated RAG Demo — Simulates multi-institution knowledge retrieval.

Run standalone: python -m core.federated_rag.demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure parent directory is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.federated_rag.config import (
    DocumentConfig,
    EmbeddingConfig,
    FederatedConfig,
    FederatedRAGConfig,
    GenerationConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from core.federated_rag.document_loader import Document, DocumentLoader
from core.federated_rag.embeddings import EmbeddingEngine
from core.federated_rag.federated_query import FederatedNode, FederatedQueryRouter
from core.federated_rag.federated_trainer import FederatedEmbeddingTrainer
from core.federated_rag.local_rag import LocalRAG
from core.federated_rag.retriever import Retriever
from core.federated_rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve knowledge base path relative to this file
_KB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base"


def _build_demo_config() -> FederatedRAGConfig:
    """Build config optimized for demo (fast, lightweight)."""
    return FederatedRAGConfig(
        embedding=EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            batch_size=16,
            device="cpu",
        ),
        vector_store=VectorStoreConfig(
            index_type="Flat",
            dimension=384,
        ),
        retrieval=RetrievalConfig(
            top_k=5,
            score_threshold=0.2,
            rerank=False,
        ),
        generation=GenerationConfig(
            max_tokens=500,
            temperature=0.3,
        ),
        federated=FederatedConfig(
            rounds=2,
            local_epochs=1,
            learning_rate=2e-5,
            fraction_clients=1.0,
        ),
        document=DocumentConfig(
            chunk_size=400,
            chunk_overlap=50,
        ),
        knowledge_base_path=str(_KB_DIR),
    )


def _create_institution_nodes(
    config: FederatedRAGConfig,
    embedder: Optional[EmbeddingEngine] = None,
) -> dict[str, FederatedNode]:
    """
    Create simulated federated nodes for different institutions.

    Each node has its own subset of the knowledge base.
    A shared EmbeddingEngine avoids loading the model multiple times.
    """
    kb_path = Path(config.knowledge_base_path)
    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {kb_path}. "
            "Run from the pai-audit directory."
        )

    all_files = sorted(kb_path.glob("*.md"))
    if not all_files:
        raise FileNotFoundError(f"No .md files found in {kb_path}")

    # Share one embedding engine across all nodes
    if embedder is None:
        embedder = EmbeddingEngine(config.embedding)

    # Split files across institutions
    institutions = {
        "givewell": {
            "name": "GiveWell Research Institute",
            "files": [f for f in all_files if "givewell" in f.name or "impact" in f.name],
        },
        "tax_advisor": {
            "name": "National Philanthropic Tax Center",
            "files": [f for f in all_files if "tax" in f.name or "daf" in f.name],
        },
        "behavioral_lab": {
            "name": "Center for Charitable Behavior Research",
            "files": [f for f in all_files if "behavioral" in f.name],
        },
    }

    # Assign any unassigned files to the first institution
    assigned_files = set()
    for inst in institutions.values():
        assigned_files.update(inst["files"])
    unassigned = [f for f in all_files if f not in assigned_files]
    if unassigned:
        institutions["givewell"]["files"].extend(unassigned)

    nodes = {}
    for node_id, info in institutions.items():
        if not info["files"]:
            logger.warning("No files for institution %s, skipping", node_id)
            continue

        loader = DocumentLoader(config.document)
        documents = []
        for f in info["files"]:
            documents.extend(loader.load_file(f))

        embeddings = embedder.embed([d.content for d in documents])

        store = VectorStore(config.vector_store)
        store.add(documents, embeddings)

        retriever = Retriever(embedder, store, config.retrieval)

        nodes[node_id] = FederatedNode(
            node_id=node_id,
            institution_name=info["name"],
            retriever=retriever,
            is_local=True,
        )

        logger.info(
            "Node '%s' (%s): %d documents indexed",
            node_id, info["name"], len(documents),
        )

    return nodes


def run_federated_demo() -> None:
    """Run the full federated RAG demo."""
    print("=" * 60)
    print("PAI Federated RAG Demo")
    print("=" * 60)

    config = _build_demo_config()

    # Step 1: Build federated nodes (shared embedder — loads model once)
    print("\n📦 Step 1: Building federated nodes...")
    shared_embedder = EmbeddingEngine(config.embedding)
    nodes = _create_institution_nodes(config, embedder=shared_embedder)
    print(f"   {len(nodes)} institutions connected\n")

    for nid, node in nodes.items():
        doc_count = node.retriever.document_count if node.retriever else 0
        print(f"   • {node.institution_name} ({nid}): {doc_count} docs")

    # Step 2: Create query router and register nodes
    print("\n🔍 Step 2: Federated query router ready")
    router = FederatedQueryRouter(config.federated)
    for nid, node in nodes.items():
        router.register_node(
            node_id=nid,
            retriever=node.retriever,
            institution_name=node.institution_name,
        )

    # Step 3: Run sample queries
    queries = [
        "How cost-effective is GiveDirectly compared to AMF?",
        "What are the tax benefits of donating appreciated stock to a DAF?",
        "How do matching grants affect charitable giving behavior?",
        "What is a DALY and how is it used to measure charitable impact?",
    ]

    # Use router's node registry (single source of truth)
    router_nodes = router.get_nodes()

    print("\n" + "=" * 60)
    print("Running Federated Queries")
    print("=" * 60)

    for query in queries:
        print(f"\n❓ Query: {query}")
        result = router.search(query, top_k=3, include_content=True)

        print(f"   ⏱  {result.latency_seconds:.2f}s | "
              f"{result.successful_nodes}/{result.total_nodes_queried} nodes responded")

        for nid, results in result.node_results.items():
            if results:
                best = results[0]
                node_name = router_nodes.get(nid, FederatedNode(nid, nid)).institution_name
                print(f"   📌 {node_name}: "
                      f"score={best.score:.3f} | {best.document.source}")
                print(f"      → {best.document.content[:120].replace(chr(10), ' ')}...")

    # Step 4: Privacy demonstration
    print("\n" + "=" * 60)
    print("Privacy Mode: Federated Search (content hidden)")
    print("=" * 60)

    private_result = router.search(
        "What is the warm-glow theory of giving?",
        top_k=2,
        include_content=False,  # Privacy mode!
    )

    for nid, results in private_result.node_results.items():
        if results:
            node_name = router_nodes.get(nid, FederatedNode(nid, nid)).institution_name
            print(f"\n   📌 {node_name}:")
            for r in results:
                print(f"      Score: {r.score:.3f} | Source: {r.document.source}")
                print(f"      Content: {r.document.content}")  # Should show "[content hidden]"

    # Step 5: AI Answer mode (federated search + LLM generation)
    print("\n" + "=" * 60)
    print("AI Answer Mode: Federated Search + Generation")
    print("=" * 60)

    ai_queries = [
        "Which charity has the lowest cost per life saved?",
        "How can I maximize tax benefits from charitable giving?",
    ]

    for query in ai_queries:
        print(f"\n❓ Query: {query}")
        result = router.search_and_answer(query, top_k=3)
        print(f"   ⏱  {result['federated_result'].latency_seconds:.2f}s | "
              f"{len(result['sources'])} sources used")
        print(f"   💡 Answer preview: {result['answer'][:200].replace(chr(10), ' ')}...")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    run_federated_demo()
