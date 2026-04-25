"""
Impact Feedback Loop — Closed-loop impact measurement for PAI.

Connects measured charitable outcomes back to investment strategies
and grant recommendations. This is PAI's most transformative component
(Phase 3 in the Gates Foundation application).

Architecture:
    1. Impact Signal Ingestion: Collect outcome data from grantees
    2. Impact Scoring: Score outcomes against predictions
    3. Feedback Propagation: Update recommendations based on actual impact
    4. Saturation Detection: Detect diminishing returns per program area
    5. Reallocation Engine: Suggest portfolio rebalancing

Key innovation: "connecting what happened to what should we do next"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ImpactCategory(Enum):
    """Categories of charitable impact."""
    HEALTH = "health"
    EDUCATION = "education"
    ENVIRONMENT = "environment"
    POVERTY = "poverty"
    HUMAN_RIGHTS = "human_rights"
    RARE_DISEASE = "rare_disease"
    DISASTER_RELIEF = "disaster_relief"


class SignalDirection(Enum):
    """Direction of impact signal relative to prediction."""
    ABOVE = "above"       # Better than predicted
    ON_TRACK = "on_track" # Within expected range
    BELOW = "below"       # Worse than predicted
    CRITICAL = "critical" # Significantly worse, requires action


@dataclass
class ImpactSignal:
    """
    A single measured outcome from a grantee.

    Represents real-world evidence of charitable impact.
    """
    grantee_id: str
    category: ImpactCategory
    metric_name: str           # e.g., "lives_saved", "students_graduated"
    metric_value: float        # Actual measured value
    predicted_value: float     # What was predicted at grant time
    unit: str = "count"        # count, percentage, dollars, DALY
    measurement_period: str = ""  # e.g., "2026-Q1"
    confidence: float = 0.8    # Measurement reliability (0-1)
    source: str = ""           # Data source (grant report, third-party eval)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def direction(self) -> SignalDirection:
        """Determine if impact is above/below prediction."""
        if self.predicted_value == 0:
            return SignalDirection.ON_TRACK if self.metric_value > 0 else SignalDirection.BELOW

        ratio = self.metric_value / self.predicted_value
        if ratio >= 1.15:
            return SignalDirection.ABOVE
        elif ratio >= 0.85:
            return SignalDirection.ON_TRACK
        elif ratio >= 0.50:
            return SignalDirection.BELOW
        else:
            return SignalDirection.CRITICAL

    @property
    def deviation_pct(self) -> float:
        """Percentage deviation from prediction."""
        if self.predicted_value == 0:
            return 0.0
        return (self.metric_value - self.predicted_value) / self.predicted_value * 100


@dataclass
class GrantAllocation:
    """Current allocation to a grantee."""
    grantee_id: str
    category: ImpactCategory
    amount: float
    predicted_impact: float
    allocation_date: str = ""
    status: str = "active"  # active, completed, flagged


@dataclass
class ReallocationRecommendation:
    """A recommendation to rebalance allocations."""
    grantee_id: str
    category: ImpactCategory
    current_amount: float
    recommended_amount: float
    reason: str
    signal_direction: SignalDirection
    priority: float = 0.5  # 0-1, higher = more urgent


# ── Impact Scoring ─────────────────────────────────────────

class ImpactScorer:
    """
    Score charitable outcomes against predictions.

    Produces a composite impact score that accounts for:
    - Outcome vs. prediction (accuracy of initial assessment)
    - Measurement confidence (data quality)
    - Category weight (aligned with donor preferences)
    - Time decay (recent outcomes matter more)
    """

    # Default category weights (can be customized per donor)
    DEFAULT_CATEGORY_WEIGHTS = {
        ImpactCategory.HEALTH: 1.0,
        ImpactCategory.EDUCATION: 0.9,
        ImpactCategory.POVERTY: 0.95,
        ImpactCategory.RARE_DISEASE: 1.1,
        ImpactCategory.ENVIRONMENT: 0.8,
        ImpactCategory.HUMAN_RIGHTS: 0.85,
        ImpactCategory.DISASTER_RELIEF: 0.75,
    }

    def __init__(
        self,
        category_weights: Optional[Dict[ImpactCategory, float]] = None,
        time_decay_rate: float = 0.1,  # Exponential decay per quarter
    ) -> None:
        self._weights = category_weights or self.DEFAULT_CATEGORY_WEIGHTS
        self._decay_rate = time_decay_rate

    def score_signal(self, signal: ImpactSignal) -> float:
        """
        Score a single impact signal.

        Returns a score from 0 (worst) to 1 (best).
        """
        # Direction score
        direction_scores = {
            SignalDirection.ABOVE: 1.0,
            SignalDirection.ON_TRACK: 0.8,
            SignalDirection.BELOW: 0.4,
            SignalDirection.CRITICAL: 0.1,
        }
        dir_score = direction_scores[signal.direction]

        # Confidence adjustment
        conf_adj = 0.5 + 0.5 * signal.confidence

        # Category weight
        cat_weight = self._weights.get(signal.category, 0.8)

        # Composite
        composite = dir_score * conf_adj * cat_weight
        return min(composite, 1.0)

    def score_grantee(
        self,
        signals: List[ImpactSignal],
    ) -> Dict:
        """
        Aggregate scores for a grantee across all signals.

        Returns dict with composite score, per-category scores, and trend.
        """
        if not signals:
            return {
                "composite_score": 0.0,
                "category_scores": {},
                "signal_count": 0,
                "trend": "no_data",
                "health": "unknown",
            }

        # Per-signal scores
        scores = [self.score_signal(s) for s in signals]

        # Per-category aggregation
        cat_scores: Dict[str, List[float]] = {}
        for signal, score in zip(signals, scores):
            cat = signal.category.value
            cat_scores.setdefault(cat, []).append(score)

        cat_avg = {cat: np.mean(vals) for cat, vals in cat_scores.items()}

        # Composite (weighted average)
        composite = float(np.mean(scores))

        # Trend: compare recent vs older signals
        trend = self._compute_trend(signals)

        # Health assessment
        health = self._assess_health(composite, signals)

        return {
            "composite_score": composite,
            "category_scores": cat_avg,
            "signal_count": len(signals),
            "trend": trend,
            "health": health,
        }

    def _compute_trend(self, signals: List[ImpactSignal]) -> str:
        """Determine if impact is improving, stable, or declining."""
        if len(signals) < 2:
            return "insufficient_data"

        # Sort by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        mid = len(sorted_signals) // 2

        recent_scores = [self.score_signal(s) for s in sorted_signals[mid:]]
        older_scores = [self.score_signal(s) for s in sorted_signals[:mid]]

        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)

        diff = recent_avg - older_avg
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"

    def _assess_health(self, composite: float, signals: List[ImpactSignal]) -> str:
        """Assess overall grantee health."""
        critical_count = sum(1 for s in signals if s.direction == SignalDirection.CRITICAL)
        below_count = sum(1 for s in signals if s.direction == SignalDirection.BELOW)

        if critical_count > 0:
            return "critical"
        elif composite < 0.4:
            return "poor"
        elif below_count > len(signals) * 0.5:
            return "at_risk"
        elif composite > 0.8:
            return "excellent"
        else:
            return "good"


# ── Saturation Detection ───────────────────────────────────

class SaturationDetector:
    """
    Detect diminishing returns in program areas.

    When a program area receives too much funding relative to its
    absorption capacity, marginal impact declines. This module
    detects saturation and flags areas for rebalancing.

    Model: I(N) = I_max * N / (N + N_half)
    Where N_half is the funding level at which impact reaches 50% of maximum.
    """

    def __init__(self, default_n_half: float = 1_000_000) -> None:
        """
        Args:
            default_n_half: Default half-saturation funding ($1M).
        """
        self._n_half = default_n_half

    def impact_at_funding(
        self,
        total_funding: float,
        n_half: Optional[float] = None,
    ) -> float:
        """
        Calculate expected impact at a given funding level.

        Uses Michaelis-Menten saturation curve:
        I(N) = I_max * N / (N + N_half)

        Returns impact as fraction of maximum (0-1).
        """
        nh = n_half or self._n_half
        if total_funding <= 0:
            return 0.0
        return total_funding / (total_funding + nh)

    def marginal_impact(
        self,
        additional_funding: float,
        current_funding: float,
        n_half: Optional[float] = None,
    ) -> float:
        """
        Calculate marginal impact of additional funding.

        Returns the incremental impact (0-1) from adding more funding.
        """
        current = self.impact_at_funding(current_funding, n_half)
        new = self.impact_at_funding(current_funding + additional_funding, n_half)
        return new - current

    def detect_saturation(
        self,
        category: ImpactCategory,
        total_funding: float,
        n_half: Optional[float] = None,
    ) -> Dict:
        """
        Detect if a category is saturated.

        Returns dict with saturation level and recommendation.
        """
        saturation = self.impact_at_funding(total_funding, n_half)

        if saturation >= 0.90:
            level = "highly_saturated"
            recommendation = (
                f"{category.value} is highly saturated ({saturation:.1%} of max impact). "
                f"Additional funding will have minimal marginal impact. "
                f"Consider redirecting to underfunded areas."
            )
        elif saturation >= 0.70:
            level = "approaching_saturation"
            recommendation = (
                f"{category.value} is approaching saturation ({saturation:.1%}). "
                f"Marginal impact per dollar is declining. "
                f"Monitor closely and consider diversification."
            )
        elif saturation >= 0.40:
            level = "optimal"
            recommendation = (
                f"{category.value} is in the optimal funding zone ({saturation:.1%}). "
                f"Additional funding still produces strong marginal impact."
            )
        else:
            level = "underfunded"
            recommendation = (
                f"{category.value} is underfunded ({saturation:.1%}). "
                f"Additional funding would produce high marginal impact. "
                f"Consider increasing allocation."
            )

        return {
            "category": category.value,
            "total_funding": total_funding,
            "saturation_level": saturation,
            "saturation_label": level,
            "recommendation": recommendation,
        }


# ── Reallocation Engine ────────────────────────────────────

class ReallocationEngine:
    """
    Suggest portfolio rebalancing based on impact signals.

    Combines impact scores, saturation detection, and donor
    preferences to generate actionable reallocation recommendations.
    """

    def __init__(
        self,
        scorer: Optional[ImpactScorer] = None,
        saturation: Optional[SaturationDetector] = None,
    ) -> None:
        self._scorer = scorer or ImpactScorer()
        self._saturation = saturation or SaturationDetector()

    def recommend(
        self,
        allocations: List[GrantAllocation],
        signals: Dict[str, List[ImpactSignal]],  # grantee_id -> signals
        total_budget: float,
        min_reallocation_pct: float = 0.10,  # Don't suggest changes < 10%
    ) -> List[ReallocationRecommendation]:
        """
        Generate reallocation recommendations.

        Args:
            allocations: Current grant allocations.
            signals: Impact signals per grantee.
            total_budget: Total portfolio budget.
            min_reallocation_pct: Minimum change to recommend (as fraction).

        Returns:
            List of ReallocationRecommendation, sorted by priority.
        """
        recommendations = []

        for alloc in allocations:
            grantee_signals = signals.get(alloc.grantee_id, [])
            if not grantee_signals:
                continue

            # Score this grantee
            score_result = self._scorer.score_grantee(grantee_signals)
            composite = score_result["composite_score"]

            # Check saturation
            sat_result = self._saturation.detect_saturation(
                alloc.category, alloc.amount
            )

            # Calculate recommended amount
            if composite < 0.4 or sat_result["saturation_label"] in (
                "highly_saturated", "approaching_saturation"
            ):
                # Reduce allocation
                reduction_factor = max(0.3, composite)
                recommended = alloc.amount * reduction_factor
                reason = self._build_reduction_reason(
                    score_result, sat_result, alloc
                )
                direction = SignalDirection.BELOW
            elif composite > 0.8 and sat_result["saturation_label"] == "underfunded":
                # Increase allocation
                increase_factor = min(1.5, 1.0 + (composite - 0.8))
                recommended = alloc.amount * increase_factor
                reason = self._build_increase_reason(
                    score_result, sat_result, alloc
                )
                direction = SignalDirection.ABOVE
            else:
                # Maintain
                recommended = alloc.amount
                reason = "Performance on track. Maintain current allocation."
                direction = SignalDirection.ON_TRACK

            # Only recommend if change exceeds threshold
            change_pct = abs(recommended - alloc.amount) / alloc.amount
            if change_pct >= min_reallocation_pct:
                recommendations.append(ReallocationRecommendation(
                    grantee_id=alloc.grantee_id,
                    category=alloc.category,
                    current_amount=alloc.amount,
                    recommended_amount=recommended,
                    reason=reason,
                    signal_direction=direction,
                    priority=self._calculate_priority(composite, change_pct),
                ))

        # Sort by priority (most urgent first)
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations

    def _build_reduction_reason(
        self,
        score_result: Dict,
        sat_result: Dict,
        alloc: GrantAllocation,
    ) -> str:
        parts = []
        if score_result["health"] in ("critical", "poor"):
            parts.append(
                f"Grantee health: {score_result['health']} "
                f"(score: {score_result['composite_score']:.2f})"
            )
        if score_result["trend"] == "declining":
            parts.append("Impact trend is declining")
        if sat_result["saturation_label"] in ("highly_saturated", "approaching_saturation"):
            parts.append(
                f"Category saturation: {sat_result['saturation_level']:.1%}"
            )
        return ". ".join(parts) + ". Recommend reducing allocation."

    def _build_increase_reason(
        self,
        score_result: Dict,
        sat_result: Dict,
        alloc: GrantAllocation,
    ) -> str:
        parts = []
        if score_result["health"] == "excellent":
            parts.append(
                f"Grantee health: excellent "
                f"(score: {score_result['composite_score']:.2f})"
            )
        if score_result["trend"] == "improving":
            parts.append("Impact trend is improving")
        if sat_result["saturation_label"] == "underfunded":
            parts.append(
                f"Category is underfunded ({sat_result['saturation_level']:.1%})"
            )
        return ". ".join(parts) + ". Recommend increasing allocation."

    @staticmethod
    def _calculate_priority(composite_score: float, change_pct: float) -> float:
        """Calculate recommendation priority (0-1)."""
        # Higher priority for lower scores and larger changes
        urgency = 1.0 - composite_score
        magnitude = min(change_pct, 1.0)
        return 0.5 * urgency + 0.5 * magnitude


# ── Impact Feedback Loop (Main Interface) ──────────────────

class ImpactFeedbackLoop:
    """
    Main interface for the Impact Feedback Loop.

    Orchestrates signal ingestion, scoring, saturation detection,
    and reallocation recommendations.

    Usage:
        loop = ImpactFeedbackLoop()
        loop.ingest_signals(signals)
        report = loop.generate_report()
        recommendations = loop.get_reallocations(budget=1_000_000)
    """

    def __init__(
        self,
        scorer: Optional[ImpactScorer] = None,
        saturation: Optional[SaturationDetector] = None,
        reallocation: Optional[ReallocationEngine] = None,
    ) -> None:
        self._scorer = scorer or ImpactScorer()
        self._saturation = saturation or SaturationDetector()
        self._reallocation = reallocation or ReallocationEngine(
            self._scorer, self._saturation
        )
        self._signals: Dict[str, List[ImpactSignal]] = {}
        self._allocations: List[GrantAllocation] = []

    def ingest_signals(self, signals: List[ImpactSignal]) -> None:
        """Ingest new impact signals."""
        for signal in signals:
            self._signals.setdefault(signal.grantee_id, []).append(signal)
        logger.info(
            "Ingested %d signals for %d grantees",
            len(signals), len(set(s.grantee_id for s in signals)),
        )

    def set_allocations(self, allocations: List[GrantAllocation]) -> None:
        """Set current grant allocations."""
        self._allocations = allocations

    def get_grantee_report(self, grantee_id: str) -> Dict:
        """Get impact report for a specific grantee."""
        signals = self._signals.get(grantee_id, [])
        return self._scorer.score_grantee(signals)

    def get_reallocations(
        self,
        total_budget: float,
        min_change_pct: float = 0.10,
    ) -> List[ReallocationRecommendation]:
        """Get reallocation recommendations."""
        return self._reallocation.recommend(
            self._allocations, self._signals, total_budget, min_change_pct,
        )

    def generate_dashboard_data(self) -> Dict:
        """
        Generate data for the Streamlit dashboard.

        Returns dict with:
            - grantee_scores: per-grantee composite scores
            - category_saturation: per-category saturation levels
            - signal_summary: total signals, by direction
            - top_reallocations: top 5 reallocation recommendations
        """
        # Grantee scores
        grantee_scores = {}
        for grantee_id, signals in self._signals.items():
            grantee_scores[grantee_id] = self._scorer.score_grantee(signals)

        # Category saturation
        category_funding: Dict[ImpactCategory, float] = {}
        for alloc in self._allocations:
            category_funding[alloc.category] = (
                category_funding.get(alloc.category, 0) + alloc.amount
            )
        category_saturation = {
            cat.value: self._saturation.detect_saturation(cat, funding)
            for cat, funding in category_funding.items()
        }

        # Signal summary
        all_signals = [s for signals in self._signals.values() for s in signals]
        signal_summary = {
            "total": len(all_signals),
            "by_direction": {
                d.value: sum(1 for s in all_signals if s.direction == d)
                for d in SignalDirection
            },
            "by_category": {
                c.value: sum(1 for s in all_signals if s.category == c)
                for c in ImpactCategory
            },
        }

        # Top reallocations
        total_budget = sum(a.amount for a in self._allocations)
        top_reallocations = self.get_reallocations(total_budget)[:5]

        return {
            "grantee_scores": grantee_scores,
            "category_saturation": category_saturation,
            "signal_summary": signal_summary,
            "top_reallocations": [
                {
                    "grantee_id": r.grantee_id,
                    "category": r.category.value,
                    "current": r.current_amount,
                    "recommended": r.recommended_amount,
                    "change_pct": (r.recommended_amount - r.current_amount) / r.current_amount * 100,
                    "reason": r.reason,
                    "priority": r.priority,
                }
                for r in top_reallocations
            ],
        }
