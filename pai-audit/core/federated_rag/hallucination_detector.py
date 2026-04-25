"""
Hallucination Detection Module for PAI GiveSmart.

Detects and scores factual accuracy of LLM-generated donation advice
by cross-referencing claims against verified knowledge base sources.

Architecture:
    1. Claim Extraction: Parse LLM output into discrete factual claims
    2. Claim Verification: Match each claim against KB sources via RAG
    3. Confidence Scoring: Aggregate verification scores per claim
    4. Hallucination Flag: Flag unverifiable or contradicted claims

This is the core differentiator described in the Gates Foundation application:
"confidence-scored citations from partner knowledge bases and verified
sourcing for every claim."
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .document_loader import Document
from .embeddings import EmbeddingEngine
from .retriever import Retriever
from .vector_store import SearchResult

logger = logging.getLogger(__name__)


class ClaimStatus(Enum):
    """Verification status of a single claim."""
    VERIFIED = "verified"           # Found in KB with high similarity
    PARTIAL = "partial"             # Partially supported
    UNVERIFIABLE = "unverifiable"   # No matching source found
    CONTRADICTED = "contradicted"   # KB contradicts the claim


@dataclass
class Claim:
    """A single factual claim extracted from LLM output."""
    text: str
    claim_type: str = "factual"     # factual, numerical, recommendation
    status: ClaimStatus = ClaimStatus.UNVERIFIABLE
    confidence: float = 0.0         # 0.0 - 1.0
    sources: List[str] = field(default_factory=list)
    source_scores: List[float] = field(default_factory=list)
    explanation: str = ""


@dataclass
class HallucinationReport:
    """Full hallucination analysis for an LLM response."""
    query: str
    response: str
    claims: List[Claim]
    overall_confidence: float = 0.0
    hallucination_rate: float = 0.0
    verified_count: int = 0
    unverifiable_count: int = 0
    contradicted_count: int = 0
    partial_count: int = 0
    latency_seconds: float = 0.0

    @property
    def is_safe(self) -> bool:
        """Response is safe if no contradicted claims and hallucination rate < 20%."""
        return self.contradicted_count == 0 and self.hallucination_rate < 0.20

    def summary(self) -> str:
        """Human-readable summary."""
        status = "✅ SAFE" if self.is_safe else "⚠️ FLAGS FOUND"
        lines = [
            f"Hallucination Report {status}",
            f"  Confidence: {self.overall_confidence:.1%}",
            f"  Hallucination Rate: {self.hallucination_rate:.1%}",
            f"  Claims: {len(self.claims)} total "
            f"({self.verified_count} verified, {self.partial_count} partial, "
            f"{self.unverifiable_count} unverifiable, {self.contradicted_count} contradicted)",
        ]
        for i, claim in enumerate(self.claims, 1):
            icon = {
                ClaimStatus.VERIFIED: "✅",
                ClaimStatus.PARTIAL: "🔶",
                ClaimStatus.UNVERIFIABLE: "❓",
                ClaimStatus.CONTRADICTED: "❌",
            }[claim.status]
            lines.append(f"  {icon} [{claim.status.value}] {claim.text[:80]}... "
                         f"(confidence: {claim.confidence:.2f})")
        return "\n".join(lines)


class ClaimExtractor:
    """
    Extract factual claims from LLM-generated text.

    Uses pattern-based extraction to identify:
    - Numerical claims (percentages, dollar amounts, statistics)
    - Factual statements (charity names, program descriptions)
    - Recommendations (giving strategies, tax advice)
    """

    # Patterns for numerical claims
    NUMERICAL_PATTERNS = [
        r'\$[\d,]+(?:\.\d{1,2})?(?:\s*(?:million|billion|thousand))?',
        r'\d+(?:\.\d{1,2})?%',
        r'\d+(?:\.\d{1,2})?\s*(?:percent|per cent)',
        r'(?:costs?|saves?|provides?|reaches?|serves?)\s+(?:approximately\s+)?'
        r'[\$]?\s*[\d,]+',
        r'(?:every|per|each)\s+\$?[\d,]+',
    ]

    # Patterns for factual claims (charity/program references)
    FACTUAL_PATTERNS = [
        r'(?:GiveWell|GiveDirectly|AMF|Against Malaria|Schistosomiasis Control|'
        r'Helen Keller International|New Incentives|Evidence Action|'
        r'Direct Relief|PIH|Partners In Health)[\w\s,]*',
        r'(?:DAF|donor-advised fund)[\w\s,]*',
        r'(?:tax deduction|tax benefit|charitable deduction)[\w\s,]*',
        r'(?:cost-effectiveness|cost per life|cost per DALY)[\w\s,]*',
        r'(?:federated learning|Federated RAG|privacy-preserving)[\w\s,]*',
    ]

    # Sentence splitter
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')

    def extract(self, text: str) -> List[Claim]:
        """
        Extract claims from LLM response text.

        Args:
            text: LLM-generated response.

        Returns:
            List of Claim objects.
        """
        claims: List[Claim] = []

        # Split into sentences
        sentences = self.SENTENCE_PATTERN.split(text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        for sentence in sentences:
            claim_type = self._classify_sentence(sentence)
            claims.append(Claim(text=sentence, claim_type=claim_type))

        return claims

    def _classify_sentence(self, sentence: str) -> str:
        """Classify a sentence as numerical, factual, or recommendation."""
        for pattern in self.NUMERICAL_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return "numerical"
        for pattern in self.FACTUAL_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return "factual"
        # Heuristic: sentences with "should", "recommend", "consider" = recommendation
        if re.search(r'\b(should|recommend|consider|suggest|advise|may want to)\b',
                     sentence, re.IGNORECASE):
            return "recommendation"
        return "factual"


class HallucinationDetector:
    """
    Detect hallucinations in LLM-generated donation advice.

    Uses RAG retrieval to verify each extracted claim against the
    knowledge base. Produces a HallucinationReport with per-claim
    confidence scores.

    Usage:
        detector = HallucinationDetector(retriever, embedder)
        report = detector.detect(query, llm_response)
        if report.is_safe:
            # Safe to show to donor
        else:
            # Flag for review or regenerate
    """

    def __init__(
        self,
        retriever: Retriever,
        embedder: EmbeddingEngine,
        verified_threshold: float = 0.65,
        partial_threshold: float = 0.40,
        contradiction_threshold: float = 0.30,
    ) -> None:
        """
        Args:
            retriever: RAG retriever with indexed knowledge base.
            embedder: Embedding engine for claim embedding.
            verified_threshold: Min similarity for VERIFIED status.
            partial_threshold: Min similarity for PARTIAL status.
            contradiction_threshold: Max similarity where we check for contradiction.
        """
        self._retriever = retriever
        self._embedder = embedder
        self._extractor = ClaimExtractor()
        self._verified_threshold = verified_threshold
        self._partial_threshold = partial_threshold
        self._contradiction_threshold = contradiction_threshold

    def detect(
        self,
        query: str,
        response: str,
        top_k: int = 5,
    ) -> HallucinationReport:
        """
        Analyze an LLM response for hallucinations.

        Args:
            query: Original user query.
            response: LLM-generated response to analyze.
            top_k: Number of KB sources to check per claim.

        Returns:
            HallucinationReport with per-claim analysis.
        """
        import time
        start = time.time()

        # Step 1: Extract claims
        claims = self._extractor.extract(response)

        # Step 2: Verify each claim
        verified_claims = []
        for claim in claims:
            verified = self._verify_claim(claim, top_k)
            verified_claims.append(verified)

        # Step 3: Aggregate report
        elapsed = time.time() - start
        report = self._build_report(query, response, verified_claims, elapsed)

        logger.info(
            "Hallucination check: confidence=%.2f, rate=%.2f, safe=%s",
            report.overall_confidence,
            report.hallucination_rate,
            report.is_safe,
        )

        return report

    def _verify_claim(self, claim: Claim, top_k: int) -> Claim:
        """Verify a single claim against the knowledge base."""
        # Search for similar content
        results = self._retriever.retrieve(claim.text, top_k=top_k)

        if not results:
            claim.status = ClaimStatus.UNVERIFIABLE
            claim.confidence = 0.0
            claim.explanation = "No matching sources found in knowledge base."
            return claim

        # Score based on top result
        best_score = results[0].score
        claim.source_scores = [r.score for r in results]
        claim.sources = [r.document.source for r in results]

        if best_score >= self._verified_threshold:
            claim.status = ClaimStatus.VERIFIED
            claim.confidence = best_score
            claim.explanation = f"Verified against {results[0].document.source} (score: {best_score:.3f})"
        elif best_score >= self._partial_threshold:
            claim.status = ClaimStatus.PARTIAL
            claim.confidence = best_score
            claim.explanation = f"Partially supported by {results[0].document.source} (score: {best_score:.3f})"
        else:
            claim.status = ClaimStatus.UNVERIFIABLE
            claim.confidence = best_score
            claim.explanation = f"Low similarity to all sources (best: {best_score:.3f})"

        # Check for numerical claim mismatches
        if claim.claim_type == "numerical":
            self._check_numerical_consistency(claim, results)

        return claim

    def _check_numerical_consistency(
        self, claim: Claim, results: List[SearchResult]
    ) -> None:
        """
        For numerical claims, extract numbers from claim and sources
        and check for contradictions.
        """
        claim_numbers = self._extract_numbers(claim.text)
        if not claim_numbers:
            return

        for result in results[:3]:
            source_numbers = self._extract_numbers(result.document.content)
            # Check if any source number is close to a claim number
            for cn in claim_numbers:
                for sn in source_numbers:
                    if self._numbers_contradict(cn, sn):
                        claim.status = ClaimStatus.CONTRADICTED
                        claim.confidence = result.score * 0.3  # Penalize
                        claim.explanation = (
                            f"Numerical contradiction: claim has {cn}, "
                            f"source has {sn} in {result.document.source}"
                        )
                        return

    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Extract numerical values from text."""
        # Match dollar amounts, percentages, and plain numbers
        patterns = [
            r'\$([\d,]+(?:\.\d{1,2})?)',
            r'([\d,]+(?:\.\d{1,2})?)\s*%',
            r'([\d,]+(?:\.\d{1,2})?)\s*(?:percent|per cent)',
        ]
        numbers = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    val = float(match.group(1).replace(',', ''))
                    numbers.append(val)
                except ValueError:
                    continue
        return numbers

    @staticmethod
    def _numbers_contradict(a: float, b: float) -> bool:
        """
        Check if two numbers contradict each other.
        Contradiction = same order of magnitude but >30% difference.
        """
        if a == 0 and b == 0:
            return False
        if a == 0 or b == 0:
            return True
        ratio = max(a, b) / min(a, b)
        # Same order of magnitude (within 10x) but >30% different
        return 0.1 < ratio < 10.0 and ratio > 1.3

    def _build_report(
        self,
        query: str,
        response: str,
        claims: List[Claim],
        elapsed: float,
    ) -> HallucinationReport:
        """Build the aggregated HallucinationReport."""
        verified = sum(1 for c in claims if c.status == ClaimStatus.VERIFIED)
        partial = sum(1 for c in claims if c.status == ClaimStatus.PARTIAL)
        unverifiable = sum(1 for c in claims if c.status == ClaimStatus.UNVERIFIABLE)
        contradicted = sum(1 for c in claims if c.status == ClaimStatus.CONTRADICTED)

        total = len(claims)
        if total == 0:
            return HallucinationReport(
                query=query, response=response, claims=claims,
                overall_confidence=0.0, hallucination_rate=1.0,
                latency_seconds=elapsed,
            )

        # Overall confidence: weighted average (verified=1.0, partial=0.6, unverifiable=0.2, contradicted=0.0)
        weights = {
            ClaimStatus.VERIFIED: 1.0,
            ClaimStatus.PARTIAL: 0.6,
            ClaimStatus.UNVERIFIABLE: 0.2,
            ClaimStatus.CONTRADICTED: 0.0,
        }
        overall = sum(weights[c.status] for c in claims) / total

        # Hallucination rate: fraction of unverifiable + contradicted
        hallucination = (unverifiable + contradicted) / total

        return HallucinationReport(
            query=query,
            response=response,
            claims=claims,
            overall_confidence=overall,
            hallucination_rate=hallucination,
            verified_count=verified,
            unverifiable_count=unverifiable,
            contradicted_count=contradicted,
            partial_count=partial,
            latency_seconds=elapsed,
        )
