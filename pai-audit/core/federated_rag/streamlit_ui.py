"""
Streamlit UI for Federated RAG — integrates with PAI dashboard.

Add to pai-audit/app.py via:
    from core.federated_rag.streamlit_ui import render_federated_rag
"""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def render_federated_rag() -> None:
    """Render the Federated RAG page in Streamlit."""

    st.markdown("## 🔗 Federated RAG — Knowledge Base")

    st.markdown("""
    **Privacy-Preserving Cross-Institutional Knowledge Retrieval**

    Query multiple institutions' knowledge bases simultaneously.
    Raw documents never leave each institution — only similarity scores are shared.

    *Powered by sentence-transformers + FAISS + FedAvg*
    """)

    # ── Sidebar: Configuration ──
    with st.sidebar:
        st.markdown("### ⚙️ RAG Configuration")

        top_k = st.slider("Top-K Results", 1, 20, 5)
        score_threshold = st.slider("Min Similarity Score", 0.0, 1.0, 0.2, 0.05)
        privacy_mode = st.checkbox("🔒 Privacy Mode (hide content)", value=False)
        ai_mode = st.checkbox("🤖 AI Answer Mode", value=True,
                              help="Use LLM to generate answers from retrieved context")

        st.markdown("---")
        st.markdown("### 📊 Institution Status")

    # ── Initialize RAG system (cached in session state) ──
    if "fed_rag_router" not in st.session_state:
        with st.spinner("Loading Federated RAG system (first time may take 30s to download embedding model)..."):
            try:
                router = _init_federated_rag()
                st.session_state["fed_rag_router"] = router
                st.session_state["fed_rag_ready"] = True
                st.session_state["fed_rag_error"] = None
            except Exception as e:
                logger.error("Failed to init Federated RAG: %s", e)
                st.session_state["fed_rag_ready"] = False
                st.session_state["fed_rag_error"] = str(e)

    if not st.session_state.get("fed_rag_ready", False):
        error_msg = st.session_state.get("fed_rag_error", "Unknown error")
        st.error(f"Failed to initialize Federated RAG: {error_msg}")
        st.info("Make sure the `data/knowledge_base/` directory contains `.md` files.")
        return

    router = st.session_state["fed_rag_router"]

    # ── Show institution status ──
    with st.sidebar:
        for nid, node in router.get_nodes().items():
            doc_count = node.retriever.document_count if node.retriever else 0
            st.metric(
                label=node.institution_name,
                value=f"{doc_count} docs",
            )

    # ── Query Input ──
    query = st.text_input(
        "Ask a question about philanthropy, tax strategy, or charitable impact:",
        placeholder="e.g., How cost-effective is GiveDirectly?",
    )

    if not query:
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        examples = [
            "How cost-effective is GiveDirectly compared to AMF?",
            "What are the tax benefits of donating appreciated stock to a DAF?",
            "How do matching grants affect charitable giving?",
            "What is a DALY and how is it used to measure impact?",
            "What is the warm-glow theory of charitable giving?",
            "How does federated learning protect patient privacy?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            with cols[i % 2]:
                if st.button(ex, key=f"example_{i}", use_container_width=True):
                    st.session_state["rag_query"] = ex
                    st.rerun()
        return

    # ── Run Query ──
    if st.button("🔍 Search", type="primary", use_container_width=True):
        with st.spinner("Querying federated nodes..."):
            if ai_mode:
                result = router.search_and_answer(query, top_k=top_k)
                fed_result = result["federated_result"]

                # Show AI answer
                st.markdown("### 🤖 AI Answer")
                st.markdown(result["answer"])

                # Show sources
                if result["sources"]:
                    st.markdown("---")
                    st.markdown(f"### 📚 Sources ({len(result['sources'])} from {fed_result.successful_nodes} institutions)")
                    for i, src in enumerate(result["sources"], 1):
                        with st.expander(f"{i}. {src['source']} (score: {src['score']:.3f})"):
                            st.markdown(src["preview"] + "...")
            else:
                result = router.search(
                    query,
                    top_k=top_k,
                    include_content=not privacy_mode,
                    score_threshold=score_threshold,
                )
                fed_result = result

                for nid, results in fed_result.node_results.items():
                    if not results:
                        continue

                    node = router.get_nodes()[nid]
                    with st.expander(f"📌 {node.institution_name} ({len(results)} results)", expanded=True):
                        for i, r in enumerate(results, 1):
                            st.markdown(f"**Result {i}** — Score: `{r.score:.3f}` | Source: `{r.document.source}`")
                            if privacy_mode:
                                st.code(r.document.content, language=None)
                            else:
                                st.markdown(r.document.content)
                            st.markdown("---")

        # ── Summary ──
        st.markdown("---")
        col_time, col_nodes = st.columns(2)
        with col_time:
            st.metric("Query Latency", f"{fed_result.latency_seconds:.2f}s")
        with col_nodes:
            st.metric("Nodes Responded", f"{fed_result.successful_nodes}/{fed_result.total_nodes_queried}")

        if privacy_mode:
            st.info("🔒 **Privacy Mode**: Document content is hidden. "
                    "Only similarity scores and source identifiers are shared across institutions.")
        else:
            st.success("✅ Federated search complete. "
                       "All results were retrieved locally — no raw documents were transmitted.")


def _init_federated_rag():
    """Initialize the federated RAG system. Returns router."""
    from .config import FederatedRAGConfig
    from .demo import _build_demo_config, _create_institution_nodes
    from .embeddings import EmbeddingEngine
    from .federated_query import FederatedQueryRouter

    config = _build_demo_config()
    shared_embedder = EmbeddingEngine(config.embedding)
    nodes = _create_institution_nodes(config, embedder=shared_embedder)
    router = FederatedQueryRouter(config.federated)
    for nid, node in nodes.items():
        router.register_node(
            node_id=nid,
            retriever=node.retriever,
            institution_name=node.institution_name,
        )
    return router
