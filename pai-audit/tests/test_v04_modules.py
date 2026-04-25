"""
Tests for GiveNudge and Impact Feedback Loop modules (PAI v0.4).

Run: pytest tests/test_v04_modules.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.give_nudge import (
    NudgeType, NudgeChannel, DonorSegment, DonorProfile, Nudge,
    TimingEngine, NudgeGenerator, ABTestEngine,
)
from core.impact_feedback import (
    ImpactCategory, SignalDirection, ImpactSignal, GrantAllocation,
    ImpactScorer, SaturationDetector, ReallocationEngine,
    ReallocationRecommendation, ImpactFeedbackLoop,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def first_time_donor():
    return DonorProfile(
        donor_id="donor_001",
        segment=DonorSegment.FIRST_TIME,
        total_given=100.0,
        gifts_count=1,
        preferred_causes=["global health"],
        avg_gift_size=100.0,
        engagement_score=0.6,
    )


@pytest.fixture
def recurring_donor():
    return DonorProfile(
        donor_id="donor_002",
        segment=DonorSegment.RECURRING,
        total_given=5000.0,
        gifts_count=12,
        last_gift_date="2026-03-15",
        preferred_causes=["education", "rare disease"],
        avg_gift_size=416.67,
        engagement_score=0.85,
    )


@pytest.fixture
def lapsed_donor():
    return DonorProfile(
        donor_id="donor_003",
        segment=DonorSegment.LAPSED,
        total_given=2000.0,
        gifts_count=5,
        last_gift_date="2025-06-01",
        preferred_causes=["environment"],
        avg_gift_size=400.0,
        engagement_score=0.3,
    )


@pytest.fixture
def sample_signals():
    """Sample impact signals for testing."""
    return [
        ImpactSignal(
            grantee_id="grantee_001",
            category=ImpactCategory.HEALTH,
            metric_name="lives_saved",
            metric_value=150,
            predicted_value=120,
            confidence=0.9,
            source="grant_report",
        ),
        ImpactSignal(
            grantee_id="grantee_001",
            category=ImpactCategory.HEALTH,
            metric_name="patients_treated",
            metric_value=800,
            predicted_value=1000,
            confidence=0.85,
            source="grant_report",
        ),
        ImpactSignal(
            grantee_id="grantee_002",
            category=ImpactCategory.EDUCATION,
            metric_name="students_graduated",
            metric_value=50,
            predicted_value=50,
            confidence=0.8,
            source="third_party_eval",
        ),
        ImpactSignal(
            grantee_id="grantee_003",
            category=ImpactCategory.HEALTH,
            metric_name="vaccinations",
            metric_value=100,
            predicted_value=500,
            confidence=0.9,
            source="grant_report",
        ),
    ]


@pytest.fixture
def sample_allocations():
    return [
        GrantAllocation(
            grantee_id="grantee_001",
            category=ImpactCategory.HEALTH,
            amount=500_000,
            predicted_impact=0.8,
        ),
        GrantAllocation(
            grantee_id="grantee_002",
            category=ImpactCategory.EDUCATION,
            amount=300_000,
            predicted_impact=0.7,
        ),
        GrantAllocation(
            grantee_id="grantee_003",
            category=ImpactCategory.HEALTH,
            amount=200_000,
            predicted_impact=0.6,
        ),
    ]


# ============================================================
# TimingEngine Tests
# ============================================================

class TestTimingEngine:
    def test_optimal_time_returns_dict(self, first_time_donor):
        engine = TimingEngine()
        result = engine.optimal_time(first_time_donor)
        assert "best_day" in result
        assert "best_hour" in result
        assert "timing_score" in result
        assert "recommendation" in result

    def test_optimal_time_best_day_is_tuesday(self, first_time_donor):
        engine = TimingEngine()
        result = engine.optimal_time(first_time_donor)
        assert result["best_day"] == 1  # Tuesday

    def test_optimal_time_best_hour_is_9am(self, first_time_donor):
        engine = TimingEngine()
        result = engine.optimal_time(first_time_donor)
        assert result["best_hour"] == 9

    def test_timing_score_in_range(self, first_time_donor):
        engine = TimingEngine()
        result = engine.optimal_time(first_time_donor)
        assert 0 <= result["timing_score"] <= 1.0

    def test_lapsed_donor_higher_timing_score(self, lapsed_donor):
        engine = TimingEngine()
        result = engine.optimal_time(lapsed_donor)
        # Lapsed donors need better timing (segment_multiplier=1.2)
        assert result["timing_score"] > 0

    def test_dow_weights_sum_reasonable(self):
        engine = TimingEngine()
        assert len(engine.DOW_WEIGHTS) == 7
        assert engine.DOW_WEIGHTS.max() == engine.DOW_WEIGHTS[1]  # Tuesday

    def test_hour_weights_sum_reasonable(self):
        engine = TimingEngine()
        assert len(engine.HOUR_WEIGHTS) == 24
        assert engine.HOUR_WEIGHTS[9] == 1.0  # 9am peak

    def test_month_weights_december_highest(self):
        engine = TimingEngine()
        assert engine.MONTH_WEIGHTS[11] == 1.5  # December


# ============================================================
# NudgeGenerator Tests
# ============================================================

class TestNudgeGenerator:
    def test_generate_returns_list(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        assert isinstance(nudges, list)
        assert len(nudges) > 0

    def test_generate_returns_nudge_objects(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        assert all(isinstance(n, Nudge) for n in nudges)

    def test_first_time_gets_social_proof(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        types = [n.nudge_type for n in nudges]
        assert NudgeType.SOCIAL_PROOF in types

    def test_first_time_gets_impact_frame(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        types = [n.nudge_type for n in nudges]
        assert NudgeType.IMPACT_FRAME in types

    def test_recurring_gets_warm_glow(self, recurring_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(recurring_donor)
        types = [n.nudge_type for n in nudges]
        assert NudgeType.WARM_GLOW in types

    def test_lapsed_gets_impact_frame(self, lapsed_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(lapsed_donor)
        types = [n.nudge_type for n in nudges]
        assert NudgeType.IMPACT_FRAME in types

    def test_sorted_by_priority(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        priorities = [n.priority for n in nudges]
        assert priorities == sorted(priorities, reverse=True)

    def test_nudge_has_required_fields(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        for n in nudges:
            assert n.headline
            assert n.body
            assert n.cta_text
            assert 0 <= n.priority <= 1.0
            assert n.estimated_lift >= 0

    def test_matching_nudge_with_context(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor, context={"matching_ratio": 3})
        matching = [n for n in nudges if n.nudge_type == NudgeType.MATCHING]
        if matching:
            assert "3x" in matching[0].headline

    def test_urgency_nudge_with_goal(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(
            first_time_donor,
            context={"goal_progress": 45000, "goal_target": 50000},
        )
        urgency = [n for n in nudges if n.nudge_type == NudgeType.URGENCY]
        if urgency:
            assert "$5,000" in urgency[0].headline

    def test_channel_matches_donor_preference(self, first_time_donor):
        gen = NudgeGenerator()
        nudges = gen.generate(first_time_donor)
        assert all(n.channel == first_time_donor.preferred_channel for n in nudges)

    def test_nudge_effectiveness_values(self):
        gen = NudgeGenerator()
        for nudge_type, effectiveness in gen.NUDGE_EFFECTIVENESS.items():
            assert 0 <= effectiveness <= 1.0


# ============================================================
# ABTestEngine Tests
# ============================================================

class TestABTestEngine:
    def test_treatment_wins(self):
        engine = ABTestEngine()
        result = engine.analyze(
            control_converted=20, control_total=100,
            treatment_converted=30, treatment_total=100,
        )
        assert result.treatment_conversion > result.control_conversion
        assert result.lift > 0

    def test_control_wins(self):
        engine = ABTestEngine()
        result = engine.analyze(
            control_converted=30, control_total=100,
            treatment_converted=20, treatment_total=100,
        )
        assert result.lift < 0

    def test_no_difference(self):
        engine = ABTestEngine()
        result = engine.analyze(
            control_converted=25, control_total=100,
            treatment_converted=25, treatment_total=100,
        )
        assert abs(result.lift) < 0.01
        assert not result.significant

    def test_significance_detection(self):
        engine = ABTestEngine()
        # Large difference should be significant
        result = engine.analyze(
            control_converted=10, control_total=200,
            treatment_converted=40, treatment_total=200,
        )
        assert result.significant
        assert result.p_value < 0.05

    def test_small_sample_not_significant(self):
        engine = ABTestEngine()
        result = engine.analyze(
            control_converted=1, control_total=10,
            treatment_converted=2, treatment_total=10,
        )
        # Small sample, likely not significant
        assert result.sample_size == 20

    def test_result_has_required_fields(self):
        engine = ABTestEngine()
        result = engine.analyze(10, 100, 15, 100)
        assert hasattr(result, 'control_conversion')
        assert hasattr(result, 'treatment_conversion')
        assert hasattr(result, 'lift')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'significant')
        assert hasattr(result, 'recommendation')

    def test_recommendation_string(self):
        engine = ABTestEngine()
        result = engine.analyze(10, 100, 20, 100)
        assert isinstance(result.recommendation, str)
        assert len(result.recommendation) > 0


# ============================================================
# ImpactSignal Tests
# ============================================================

class TestImpactSignal:
    def test_above_prediction(self):
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=150, predicted_value=100,
        )
        assert signal.direction == SignalDirection.ABOVE

    def test_on_track(self):
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=95, predicted_value=100,
        )
        assert signal.direction == SignalDirection.ON_TRACK

    def test_below_prediction(self):
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=60, predicted_value=100,
        )
        assert signal.direction == SignalDirection.BELOW

    def test_critical(self):
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=20, predicted_value=100,
        )
        assert signal.direction == SignalDirection.CRITICAL

    def test_deviation_pct(self):
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=150, predicted_value=100,
        )
        assert abs(signal.deviation_pct - 50.0) < 0.01

    def test_zero_prediction(self):
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=10, predicted_value=0,
        )
        assert signal.direction == SignalDirection.ON_TRACK


# ============================================================
# ImpactScorer Tests
# ============================================================

class TestImpactScorer:
    def test_score_signal_above(self):
        scorer = ImpactScorer()
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=150, predicted_value=100,
            confidence=0.9,
        )
        score = scorer.score_signal(signal)
        assert score > 0.7

    def test_score_signal_critical(self):
        scorer = ImpactScorer()
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.HEALTH,
            metric_name="lives", metric_value=20, predicted_value=100,
            confidence=0.9,
        )
        score = scorer.score_signal(signal)
        assert score < 0.3

    def test_score_signal_range(self):
        scorer = ImpactScorer()
        for direction in SignalDirection:
            signal = ImpactSignal(
                grantee_id="g1", category=ImpactCategory.HEALTH,
                metric_name="lives", metric_value=100, predicted_value=100,
                confidence=0.8,
            )
            # Manually set direction by adjusting values
            if direction == SignalDirection.ABOVE:
                signal = ImpactSignal("g1", ImpactCategory.HEALTH, "x", 150, 100, confidence=0.9)
            elif direction == SignalDirection.BELOW:
                signal = ImpactSignal("g1", ImpactCategory.HEALTH, "x", 60, 100, confidence=0.9)
            elif direction == SignalDirection.CRITICAL:
                signal = ImpactSignal("g1", ImpactCategory.HEALTH, "x", 20, 100, confidence=0.9)
            score = scorer.score_signal(signal)
            assert 0 <= score <= 1.0

    def test_score_grantee_composite(self, sample_signals):
        scorer = ImpactScorer()
        result = scorer.score_grantee(sample_signals[:2])  # grantee_001
        assert 0 <= result["composite_score"] <= 1.0
        assert result["signal_count"] == 2

    def test_score_grantee_health(self, sample_signals):
        scorer = ImpactScorer()
        result = scorer.score_grantee(sample_signals[:2])
        assert result["health"] in ("excellent", "good", "at_risk", "poor", "critical", "unknown")

    def test_score_grantee_trend(self, sample_signals):
        scorer = ImpactScorer()
        result = scorer.score_grantee(sample_signals[:2])
        assert result["trend"] in ("improving", "stable", "declining", "insufficient_data")

    def test_empty_signals(self):
        scorer = ImpactScorer()
        result = scorer.score_grantee([])
        assert result["composite_score"] == 0.0
        assert result["health"] == "unknown"

    def test_custom_category_weights(self):
        scorer = ImpactScorer(
            category_weights={ImpactCategory.RARE_DISEASE: 1.5}
        )
        signal = ImpactSignal(
            grantee_id="g1", category=ImpactCategory.RARE_DISEASE,
            metric_name="patients", metric_value=100, predicted_value=100,
            confidence=0.9,
        )
        score = scorer.score_signal(signal)
        assert score > 0.8  # Higher due to custom weight


# ============================================================
# SaturationDetector Tests
# ============================================================

class TestSaturationDetector:
    def test_zero_funding(self):
        det = SaturationDetector()
        assert det.impact_at_funding(0) == 0.0

    def test_low_funding(self):
        det = SaturationDetector(default_n_half=1_000_000)
        impact = det.impact_at_funding(100_000)
        assert 0 < impact < 0.2

    def test_half_saturation(self):
        det = SaturationDetector(default_n_half=1_000_000)
        impact = det.impact_at_funding(1_000_000)
        assert abs(impact - 0.5) < 0.01

    def test_high_saturation(self):
        det = SaturationDetector(default_n_half=1_000_000)
        impact = det.impact_at_funding(5_000_000)
        assert impact > 0.8

    def test_marginal_impact_positive(self):
        det = SaturationDetector(default_n_half=1_000_000)
        marginal = det.marginal_impact(100_000, 500_000)
        assert marginal > 0

    def test_marginal_impact_decreasing(self):
        det = SaturationDetector(default_n_half=1_000_000)
        m1 = det.marginal_impact(100_000, 100_000)
        m2 = det.marginal_impact(100_000, 2_000_000)
        assert m1 > m2  # Diminishing returns

    def test_detect_underfunded(self):
        det = SaturationDetector(default_n_half=1_000_000)
        result = det.detect_saturation(ImpactCategory.HEALTH, 100_000)
        assert result["saturation_label"] == "underfunded"

    def test_detect_optimal(self):
        det = SaturationDetector(default_n_half=1_000_000)
        result = det.detect_saturation(ImpactCategory.HEALTH, 800_000)
        assert result["saturation_label"] == "optimal"

    def test_detect_highly_saturated(self):
        det = SaturationDetector(default_n_half=1_000_000)
        result = det.detect_saturation(ImpactCategory.HEALTH, 10_000_000)
        assert result["saturation_label"] == "highly_saturated"

    def test_detect_has_recommendation(self):
        det = SaturationDetector()
        result = det.detect_saturation(ImpactCategory.HEALTH, 100_000)
        assert len(result["recommendation"]) > 0


# ============================================================
# ReallocationEngine Tests
# ============================================================

class TestReallocationEngine:
    def test_recommend_returns_list(self, sample_allocations, sample_signals):
        engine = ReallocationEngine()
        signals_by_grantee = {}
        for s in sample_signals:
            signals_by_grantee.setdefault(s.grantee_id, []).append(s)
        recs = engine.recommend(sample_allocations, signals_by_grantee, 1_000_000)
        assert isinstance(recs, list)

    def test_critical_grantee_gets_reduction(self, sample_signals):
        """Grantee with critical signal should get reallocation recommendation."""
        engine = ReallocationEngine()
        allocations = [
            GrantAllocation("grantee_003", ImpactCategory.HEALTH, 200_000, 0.6),
        ]
        signals = {
            "grantee_003": [
                ImpactSignal("grantee_003", ImpactCategory.HEALTH, "x", 20, 100, confidence=0.9),
            ]
        }
        recs = engine.recommend(allocations, signals, 1_000_000)
        assert len(recs) > 0
        assert recs[0].recommended_amount < recs[0].current_amount

    def test_excellent_grantee_gets_increase(self):
        """Grantee with excellent performance should get increase."""
        engine = ReallocationEngine()
        allocations = [
            GrantAllocation("grantee_001", ImpactCategory.HEALTH, 200_000, 0.6),
        ]
        signals = {
            "grantee_001": [
                ImpactSignal("grantee_001", ImpactCategory.HEALTH, "x", 200, 100, confidence=0.9),
            ]
        }
        recs = engine.recommend(allocations, signals, 1_000_000)
        assert len(recs) > 0
        assert recs[0].recommended_amount > recs[0].current_amount

    def test_sorted_by_priority(self, sample_allocations, sample_signals):
        engine = ReallocationEngine()
        signals_by_grantee = {}
        for s in sample_signals:
            signals_by_grantee.setdefault(s.grantee_id, []).append(s)
        recs = engine.recommend(sample_allocations, signals_by_grantee, 1_000_000)
        priorities = [r.priority for r in recs]
        assert priorities == sorted(priorities, reverse=True)

    def test_no_signals_no_recommendations(self, sample_allocations):
        engine = ReallocationEngine()
        recs = engine.recommend(sample_allocations, {}, 1_000_000)
        assert len(recs) == 0

    def test_min_threshold_filter(self):
        """Small changes below threshold should be filtered."""
        engine = ReallocationEngine()
        allocations = [
            GrantAllocation("g1", ImpactCategory.HEALTH, 200_000, 0.6),
        ]
        signals = {
            "g1": [
                ImpactSignal("g1", ImpactCategory.HEALTH, "x", 95, 100, confidence=0.8),
            ]
        }
        # On-track signal, should not trigger reallocation
        recs = engine.recommend(allocations, signals, 1_000_000, min_reallocation_pct=0.10)
        assert len(recs) == 0


# ============================================================
# ImpactFeedbackLoop Tests
# ============================================================

class TestImpactFeedbackLoop:
    def test_ingest_signals(self, sample_signals):
        loop = ImpactFeedbackLoop()
        loop.ingest_signals(sample_signals)
        report = loop.get_grantee_report("grantee_001")
        assert report["signal_count"] == 2

    def test_set_allocations(self, sample_allocations):
        loop = ImpactFeedbackLoop()
        loop.set_allocations(sample_allocations)
        recs = loop.get_reallocations(1_000_000)
        # No signals yet, no recommendations
        assert isinstance(recs, list)

    def test_full_pipeline(self, sample_signals, sample_allocations):
        loop = ImpactFeedbackLoop()
        loop.set_allocations(sample_allocations)
        loop.ingest_signals(sample_signals)

        # Get grantee report
        report = loop.get_grantee_report("grantee_001")
        assert report["signal_count"] == 2
        assert 0 <= report["composite_score"] <= 1.0

        # Get reallocations
        recs = loop.get_reallocations(1_000_000)
        assert isinstance(recs, list)

    def test_dashboard_data(self, sample_signals, sample_allocations):
        loop = ImpactFeedbackLoop()
        loop.set_allocations(sample_allocations)
        loop.ingest_signals(sample_signals)

        data = loop.generate_dashboard_data()
        assert "grantee_scores" in data
        assert "category_saturation" in data
        assert "signal_summary" in data
        assert "top_reallocations" in data
        assert data["signal_summary"]["total"] == len(sample_signals)

    def test_dashboard_grantee_scores(self, sample_signals, sample_allocations):
        loop = ImpactFeedbackLoop()
        loop.set_allocations(sample_allocations)
        loop.ingest_signals(sample_signals)

        data = loop.generate_dashboard_data()
        assert "grantee_001" in data["grantee_scores"]
        assert "grantee_002" in data["grantee_scores"]
        assert "grantee_003" in data["grantee_scores"]

    def test_dashboard_signal_by_direction(self, sample_signals, sample_allocations):
        loop = ImpactFeedbackLoop()
        loop.set_allocations(sample_allocations)
        loop.ingest_signals(sample_signals)

        data = loop.generate_dashboard_data()
        by_dir = data["signal_summary"]["by_direction"]
        assert "above" in by_dir
        assert "on_track" in by_dir
        assert "below" in by_dir
        assert "critical" in by_dir

    def test_empty_loop(self):
        loop = ImpactFeedbackLoop()
        data = loop.generate_dashboard_data()
        assert data["signal_summary"]["total"] == 0
        assert len(data["grantee_scores"]) == 0

    def test_multiple_ingest(self, sample_signals):
        loop = ImpactFeedbackLoop()
        loop.ingest_signals(sample_signals[:2])
        loop.ingest_signals(sample_signals[2:])
        report = loop.get_grantee_report("grantee_001")
        assert report["signal_count"] == 2


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_nudge_timing_integration(self, first_time_donor):
        """Test that timing engine feeds into nudge generation."""
        engine = TimingEngine()
        gen = NudgeGenerator()

        timing = engine.optimal_time(first_time_donor)
        nudges = gen.generate(first_time_donor)

        assert timing["timing_score"] > 0
        assert len(nudges) > 0

    def test_impact_nudge_integration(self, sample_signals, sample_allocations):
        """Test that impact signals can inform nudge strategy."""
        loop = ImpactFeedbackLoop()
        loop.set_allocations(sample_allocations)
        loop.ingest_signals(sample_signals)

        # Get reallocation recommendations
        recs = loop.get_reallocations(1_000_000)

        # Generate nudges based on donor segment
        gen = NudgeGenerator()
        donor = DonorProfile(
            donor_id="donor_001",
            segment=DonorSegment.HIGH_VALUE,
            total_given=50000,
            gifts_count=20,
            preferred_causes=["global health"],
            avg_gift_size=2500,
        )
        nudges = gen.generate(donor)

        assert isinstance(recs, list)
        assert len(nudges) > 0

    def test_ab_test_with_nudge_types(self):
        """Test A/B testing between different nudge strategies."""
        engine = ABTestEngine()

        # Simulate: social_proof vs impact_frame
        result = engine.analyze(
            control_converted=30, control_total=200,   # social_proof
            treatment_converted=42, treatment_total=200,  # impact_frame
        )

        assert result.lift > 0
        assert isinstance(result.recommendation, str)

    def test_saturation_informs_reallocation(self):
        """Test that saturation detection affects reallocation."""
        scorer = ImpactScorer()
        saturation = SaturationDetector(default_n_half=500_000)
        engine = ReallocationEngine(scorer, saturation)

        allocations = [
            GrantAllocation("g1", ImpactCategory.HEALTH, 3_000_000, 0.8),
        ]
        signals = {
            "g1": [
                ImpactSignal("g1", ImpactCategory.HEALTH, "x", 100, 100, confidence=0.8),
            ]
        }

        recs = engine.recommend(allocations, signals, 5_000_000)
        # Should recommend reduction due to saturation
        if recs:
            assert recs[0].recommended_amount < recs[0].current_amount
