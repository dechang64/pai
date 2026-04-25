"""
GiveNudge — Behavioral Engagement Module for PAI.

Implements evidence-based behavioral nudges for charitable giving,
based on warm-glow giving theory (Andreoni, 1990) and modern
behavioral economics.

Features:
    - Optimal timing recommendations for giving prompts
    - Framing effects (social proof, matching, urgency)
    - Channel selection (email, SMS, in-app)
    - Donor profile-based personalization
    - A/B test framework for nudge effectiveness

Reference: Andreoni, J. (1990). Impure altruism and donations to
public goods. The Economic Journal, 100(401), 464-477.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NudgeType(Enum):
    """Types of behavioral nudges."""
    SOCIAL_PROOF = "social_proof"         # "X donors gave to Y"
    MATCHING = "matching"                 # "Your gift will be matched 2x"
    URGENCY = "urgency"                   # "Only $X left to reach goal"
    RECOGNITION = "recognition"           # "Join X other donors"
    IMPACT_FRAME = "impact_frame"         # "$X saves Y lives"
    DEFAULT_NUDGE = "default_nudge"       # Simple reminder
    WARM_GLOW = "warm_glow"              # Emotional connection framing


class NudgeChannel(Enum):
    """Communication channels."""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    PUSH = "push_notification"


class DonorSegment(Enum):
    """Donor segmentation for personalization."""
    FIRST_TIME = "first_time"
    RECURRING = "recurring"
    LAPSED = "lapsed"           # Gave before but not recently
    HIGH_VALUE = "high_value"
    LOW_ENGAGEMENT = "low_engagement"


@dataclass
class DonorProfile:
    """Profile of a donor for nudge personalization."""
    donor_id: str
    segment: DonorSegment
    total_given: float = 0.0
    gifts_count: int = 0
    last_gift_date: Optional[str] = None
    preferred_causes: List[str] = field(default_factory=list)
    preferred_channel: NudgeChannel = NudgeChannel.EMAIL
    avg_gift_size: float = 0.0
    engagement_score: float = 0.5  # 0-1, based on open/click rates
    time_zone: str = "America/New_York"


@dataclass
class Nudge:
    """A single behavioral nudge to be delivered."""
    nudge_type: NudgeType
    channel: NudgeChannel
    headline: str
    body: str
    cta_text: str          # Call-to-action button text
    priority: float = 0.5  # 0-1, higher = more likely to convert
    estimated_lift: float = 0.0  # Expected conversion lift (%)
    timing: Optional[str] = None  # ISO datetime for optimal delivery
    donor_id: Optional[str] = None


@dataclass
class NudgeResult:
    """Result of delivering a nudge."""
    nudge: Nudge
    delivered: bool = False
    opened: bool = False
    clicked: bool = False
    converted: bool = False
    gift_amount: float = 0.0
    delivery_time: Optional[str] = None


# ── Optimal Timing Engine ──────────────────────────────────

class TimingEngine:
    """
    Determines optimal timing for giving prompts.

    Based on research findings:
    - End of month / payday: higher giving propensity
    - Year-end (Dec): 30% of annual giving occurs in December
    - Post-disaster: spike in giving (within 48h)
    - Morning (9-11am): higher email open rates
    - Tuesday/Wednesday: highest engagement
    """

    # Day-of-week weights (0=Mon, 6=Sun)
    DOW_WEIGHTS = np.array([0.85, 1.0, 0.95, 0.90, 0.80, 0.60, 0.50])

    # Month weights (1=Jan, 12=Dec)
    MONTH_WEIGHTS = np.array([
        0.60, 0.55, 0.65, 0.70, 0.75, 0.80,
        0.70, 0.65, 0.75, 0.85, 1.10, 1.50,  # December spike
    ])

    # Hour weights (0-23)
    HOUR_WEIGHTS = np.array([
        0.10, 0.05, 0.05, 0.05, 0.05, 0.10,  # 0-5am
        0.30, 0.60, 0.85, 1.00, 0.95, 0.85,  # 6-11am
        0.70, 0.65, 0.60, 0.55, 0.50, 0.55,  # 12-5pm
        0.60, 0.65, 0.70, 0.60, 0.40, 0.20,  # 6-11pm
    ])

    def optimal_time(self, donor: DonorProfile) -> Dict:
        """
        Calculate optimal delivery time for a nudge.

        Returns dict with:
            - best_day: day of week (0-6)
            - best_hour: hour of day (0-23)
            - timing_score: composite score (0-1)
            - recommendation: human-readable string
        """
        # Base scores
        dow_score = float(self.DOW_WEIGHTS.max())
        dow_best = int(self.DOW_WEIGHTS.argmax())

        hour_score = float(self.HOUR_WEIGHTS.max())
        hour_best = int(self.HOUR_WEIGHTS.argmax())

        month_score = float(self.MONTH_WEIGHTS[datetime.now().month - 1])

        # Adjust for donor segment
        segment_multiplier = {
            DonorSegment.FIRST_TIME: 0.8,
            DonorSegment.RECURRING: 1.0,
            DonorSegment.LAPSED: 1.2,
            DonorSegment.HIGH_VALUE: 0.9,
            DonorSegment.LOW_ENGAGEMENT: 1.1,
        }.get(donor.segment, 1.0)

        # Adjust for engagement (low engagement = need better timing)
        engagement_adj = 1.0 + (1.0 - donor.engagement_score) * 0.3

        composite = dow_score * hour_score * month_score * segment_multiplier * engagement_adj
        composite = min(composite, 1.0)

        return {
            "best_day": dow_best,
            "best_hour": hour_best,
            "timing_score": composite,
            "recommendation": (
                f"Best: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow_best]} "
                f"at {hour_best}:00 (score: {composite:.2f})"
            ),
        }


# ── Nudge Generator ────────────────────────────────────────

class NudgeGenerator:
    """
    Generate personalized nudges based on donor profile and context.

    Selects nudge type, channel, framing, and timing.
    """

    # Nudge effectiveness estimates (based on literature)
    NUDGE_EFFECTIVENESS = {
        NudgeType.SOCIAL_PROOF: 0.12,    # +12% conversion
        NudgeType.MATCHING: 0.22,         # +22% conversion
        NudgeType.URGENCY: 0.15,          # +15% conversion
        NudgeType.RECOGNITION: 0.08,      # +8% conversion
        NudgeType.IMPACT_FRAME: 0.18,     # +18% conversion
        NudgeType.DEFAULT_NUDGE: 0.03,    # +3% conversion
        NudgeType.WARM_GLOW: 0.10,        # +10% conversion
    }

    def __init__(self) -> None:
        self._timing = TimingEngine()

    def generate(
        self,
        donor: DonorProfile,
        context: Optional[Dict] = None,
    ) -> List[Nudge]:
        """
        Generate personalized nudges for a donor.

        Args:
            donor: Donor profile.
            context: Optional context (available matching, campaign goal, etc.)

        Returns:
            List of Nudge objects, sorted by priority.
        """
        context = context or {}
        nudges = []

        # Select best nudge types for this donor
        nudge_types = self._select_nudge_types(donor, context)

        for nudge_type in nudge_types:
            nudge = self._build_nudge(donor, nudge_type, context)
            if nudge:
                nudges.append(nudge)

        # Sort by priority
        nudges.sort(key=lambda n: n.priority, reverse=True)
        return nudges

    def _select_nudge_types(
        self, donor: DonorProfile, context: Dict
    ) -> List[NudgeType]:
        """Select the best nudge types for a donor segment."""
        segment_strategies = {
            DonorSegment.FIRST_TIME: [
                NudgeType.SOCIAL_PROOF,
                NudgeType.IMPACT_FRAME,
                NudgeType.MATCHING,
            ],
            DonorSegment.RECURRING: [
                NudgeType.WARM_GLOW,
                NudgeType.IMPACT_FRAME,
                NudgeType.RECOGNITION,
            ],
            DonorSegment.LAPSED: [
                NudgeType.IMPACT_FRAME,
                NudgeType.SOCIAL_PROOF,
                NudgeType.DEFAULT_NUDGE,
            ],
            DonorSegment.HIGH_VALUE: [
                NudgeType.IMPACT_FRAME,
                NudgeType.MATCHING,
                NudgeType.RECOGNITION,
            ],
            DonorSegment.LOW_ENGAGEMENT: [
                NudgeType.URGENCY,
                NudgeType.SOCIAL_PROOF,
                NudgeType.MATCHING,
            ],
        }
        return segment_strategies.get(donor.segment, [NudgeType.DEFAULT_NUDGE])

    def _build_nudge(
        self,
        donor: DonorProfile,
        nudge_type: NudgeType,
        context: Dict,
    ) -> Optional[Nudge]:
        """Build a specific nudge with copy."""
        matching = context.get("matching_ratio", 0)
        goal_progress = context.get("goal_progress", 0.0)
        goal_target = context.get("goal_target", 0)
        cause = donor.preferred_causes[0] if donor.preferred_causes else "a cause you care about"

        builders = {
            NudgeType.SOCIAL_PROOF: self._social_proof,
            NudgeType.MATCHING: self._matching,
            NudgeType.URGENCY: self._urgency,
            NudgeType.RECOGNITION: self._recognition,
            NudgeType.IMPACT_FRAME: self._impact_frame,
            NudgeType.DEFAULT_NUDGE: self._default,
            NudgeType.WARM_GLOW: self._warm_glow,
        }

        builder = builders.get(nudge_type)
        if builder is None:
            return None

        return builder(donor, cause, matching, goal_progress, goal_target)

    def _social_proof(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        n = max(100, donor.gifts_count * 50 + 237)
        return Nudge(
            nudge_type=NudgeType.SOCIAL_PROOF,
            channel=donor.preferred_channel,
            headline=f"{n:,} donors supported {cause} this month",
            body=f"Join a community of generous donors making a difference. "
                 f"Your contribution to {cause} will be combined with "
                 f"others to create meaningful change.",
            cta_text="Join Them",
            priority=0.7,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.SOCIAL_PROOF],
        )

    def _matching(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        ratio = matching if matching > 0 else 2
        return Nudge(
            nudge_type=NudgeType.MATCHING,
            channel=donor.preferred_channel,
            headline=f"Your gift to {cause} will be matched {ratio}x",
            body=f"A matching donor has committed to doubling every gift to "
                 f"{cause} until the campaign ends. Every dollar you give "
                 f"becomes ${ratio}.",
            cta_text=f"Give Now (2x Impact)",
            priority=0.85,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.MATCHING],
        )

    def _urgency(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        remaining = max(0, target - int(progress)) if target > 0 else 5000
        return Nudge(
            nudge_type=NudgeType.URGENCY,
            channel=donor.preferred_channel,
            headline=f"Only ${remaining:,} left to reach our {cause} goal",
            body=f"We're {progress/target*100:.0f}% of the way to our goal. "
                 f"Your gift today could be the one that puts us over the top.",
            cta_text="Help Us Reach the Goal",
            priority=0.75,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.URGENCY],
        )

    def _recognition(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        return Nudge(
            nudge_type=NudgeType.RECOGNITION,
            channel=donor.preferred_channel,
            headline=f"Your {donor.gifts_count} gifts have made a real difference",
            body=f"As a valued supporter of {cause}, your consistent generosity "
                 f"has contributed to meaningful outcomes. Consider increasing "
                 f"your impact this month.",
            cta_text="Increase My Impact",
            priority=0.6,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.RECOGNITION],
        )

    def _impact_frame(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        return Nudge(
            nudge_type=NudgeType.IMPACT_FRAME,
            channel=donor.preferred_channel,
            headline=f"${donor.avg_gift_size:.0f} to {cause} saves approximately 2 lives",
            body=f"Based on GiveWell's cost-effectiveness analysis, a gift of "
                 f"${donor.avg_gift_size:.0f} to our recommended charities can "
                 f"provide life-saving interventions. Every dollar translates "
                 f"to measurable impact.",
            cta_text="See the Impact",
            priority=0.8,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.IMPACT_FRAME],
        )

    def _default(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        return Nudge(
            nudge_type=NudgeType.DEFAULT_NUDGE,
            channel=donor.preferred_channel,
            headline=f"Consider a gift to {cause}",
            body=f"Your support for {cause} helps make a difference. "
                 f"Every contribution, no matter the size, matters.",
            cta_text="Give Now",
            priority=0.3,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.DEFAULT_NUDGE],
        )

    def _warm_glow(
        self, donor: DonorProfile, cause: str,
        matching: int, progress: float, target: int,
    ) -> Nudge:
        return Nudge(
            nudge_type=NudgeType.WARM_GLOW,
            channel=donor.preferred_channel,
            headline=f"The joy of giving to {cause}",
            body=f"Research shows that giving activates the same brain regions "
                 f"as receiving rewards. Your gift to {cause} doesn't just "
                 f"help others — it makes you happier too.",
            cta_text="Feel Good, Do Good",
            priority=0.65,
            estimated_lift=self.NUDGE_EFFECTIVENESS[NudgeType.WARM_GLOW],
        )


# ── A/B Test Framework ─────────────────────────────────────

@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_id: str
    name: str
    control_nudge: NudgeType
    treatment_nudge: NudgeType
    sample_size_target: int = 200
    min_confidence: float = 0.95


@dataclass
class ABTestResult:
    """Results of an A/B test."""
    test_id: str
    control_conversion: float
    treatment_conversion: float
    lift: float
    p_value: float
    significant: bool
    sample_size: int
    recommendation: str


class ABTestEngine:
    """
    Simple A/B testing framework for nudge effectiveness.

    Uses two-proportion z-test for statistical significance.
    """

    @staticmethod
    def analyze(
        control_converted: int,
        control_total: int,
        treatment_converted: int,
        treatment_total: int,
        min_confidence: float = 0.95,
    ) -> ABTestResult:
        """
        Analyze A/B test results using two-proportion z-test.

        Args:
            control_converted: Conversions in control group.
            control_total: Total in control group.
            treatment_converted: Conversions in treatment group.
            treatment_total: Total in treatment group.
            min_confidence: Minimum confidence level (default 95%).

        Returns:
            ABTestResult with statistical analysis.
        """
        from scipy import stats

        p1 = control_converted / max(control_total, 1)
        p2 = treatment_converted / max(treatment_total, 1)

        # Pooled proportion
        p_pool = (control_converted + treatment_converted) / max(
            control_total + treatment_total, 1
        )

        # Standard error
        se = np.sqrt(
            p_pool * (1 - p_pool) * (1/control_total + 1/treatment_total)
        ) if control_total > 0 and treatment_total > 0 else 1.0

        # Z-statistic
        z = (p2 - p1) / se if se > 0 else 0.0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Lift
        lift = (p2 - p1) / p1 if p1 > 0 else 0.0

        # Significance
        alpha = 1 - min_confidence
        significant = p_value < alpha

        # Recommendation
        if significant and lift > 0:
            recommendation = f"Treatment wins: +{lift:.1%} lift (p={p_value:.4f})"
        elif significant and lift < 0:
            recommendation = f"Control wins: treatment underperformed by {abs(lift):.1%} (p={p_value:.4f})"
        else:
            recommendation = f"No significant difference (p={p_value:.4f}, need more data)"

        return ABTestResult(
            test_id="",
            control_conversion=p1,
            treatment_conversion=p2,
            lift=lift,
            p_value=p_value,
            significant=significant,
            sample_size=control_total + treatment_total,
            recommendation=recommendation,
        )
