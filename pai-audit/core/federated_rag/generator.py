"""
RAG generator — LLM-powered answer generation with retrieved context.

Wraps the existing LLMDonationAdvisor and adds RAG context injection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .config import GenerationConfig
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class RAGGenerator:
    """
    Generate answers using an LLM with RAG context.

    Supports OpenAI, Anthropic, and a local demo mode.
    """

    def __init__(self, config: Optional[GenerationConfig] = None) -> None:
        self._config = config or GenerationConfig()
        self._openai_client = None
        self._anthropic_client = None

    def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate an answer given a query and retrieved context.

        Args:
            query: User's question.
            context: Formatted context from Retriever.format_context().
            conversation_history: Optional list of {"role": ..., "content": ...} dicts.

        Returns:
            Generated answer string.
        """
        provider = self._detect_provider()

        if provider == "openai":
            return self._generate_openai(query, context, conversation_history)
        elif provider == "anthropic":
            return self._generate_anthropic(query, context, conversation_history)
        else:
            return self._generate_demo(query, context)

    def generate_stream(
        self,
        query: str,
        context: str,
    ):
        """
        Stream tokens for Streamlit's st.write_stream.

        Yields content chunks.
        """
        provider = self._detect_provider()

        if provider == "openai":
            yield from self._stream_openai(query, context)
        else:
            # Fallback: yield the full response at once
            yield self.generate(query, context)

    # ── Provider detection ──────────────────────────────

    @staticmethod
    def _detect_provider() -> str:
        import os
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "demo"

    # ── OpenAI ──────────────────────────────────────────

    def _get_openai_client(self):
        """Lazy-init and cache OpenAI client."""
        if self._openai_client is None:
            import os
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client

    def _generate_openai(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        client = self._get_openai_client()

        messages = self._build_messages(query, context, history)
        response = client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _stream_openai(self, query: str, context: str):
        client = self._get_openai_client()
        messages = self._build_messages(query, context)

        stream = client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ── Anthropic ───────────────────────────────────────

    def _get_anthropic_client(self):
        """Lazy-init and cache Anthropic client."""
        if self._anthropic_client is None:
            import os
            from anthropic import Anthropic
            self._anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._anthropic_client

    def _generate_anthropic(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        client = self._get_anthropic_client()

        messages = self._build_messages(query, context, history)
        # Anthropic puts system prompt separately
        system_msg = self._config.system_prompt
        user_messages = [m for m in messages if m["role"] != "system"]

        response = client.messages.create(
            model=self._config.model,
            system=system_msg,
            messages=user_messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        return response.content[0].text

    # ── Demo mode (no API key) ─────────────────────────

    def _generate_demo(self, query: str, context: str) -> str:
        """Rule-based fallback when no LLM API is available."""
        if not context:
            return (
                "I don't have enough information to answer that question. "
                "Please try connecting an LLM API key (OpenAI or Anthropic) "
                "for full AI-powered responses."
            )

        # Simple extractive approach: return the most relevant context
        lines = context.split("\n")
        relevant = [l for l in lines if l.strip() and not l.startswith("[Source")]

        answer = f"Based on the knowledge base:\n\n"
        for line in relevant[:8]:
            answer += f"• {line.strip()}\n"
        answer += (
            "\n---\n*Demo mode: connect an API key for AI-generated answers.*"
        )
        return answer

    # ── Message building ────────────────────────────────

    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build the message list for LLM chat completion."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._config.system_prompt},
        ]

        # Add conversation history
        if history:
            messages.extend(history)

        # Add context + query
        user_content = f"## Relevant Knowledge Base Context\n\n{context}\n\n## User Question\n\n{query}"
        messages.append({"role": "user", "content": user_content})

        return messages
