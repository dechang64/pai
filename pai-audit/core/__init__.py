"""
PAI Core Modules

Portfolio Optimization, LLM Integration, Federated Learning,
Federated RAG, Hallucination Detection, Behavioral Nudges,
and Impact Feedback Loop.
"""

from .portfolio_optimizer import PortfolioOptimizer, optimize_daf_portfolio
from .llm_client import LLMDonationAdvisor, get_llm_advisor, check_llm_status
from .federated_learning import (
    FederatedLearningCoordinator,
    FederatedConfig,
    create_fl_system,
    demonstrate_fl_usage
)
from .give_nudge import (
    NudgeGenerator, NudgeType, NudgeChannel,
    DonorProfile, DonorSegment, Nudge, NudgeResult,
    TimingEngine, ABTestEngine, ABTestResult,
)
from .impact_feedback import (
    ImpactFeedbackLoop, ImpactScorer, SaturationDetector,
    ReallocationEngine, ImpactSignal, ImpactCategory,
    SignalDirection, GrantAllocation, ReallocationRecommendation,
)

__all__ = [
    # Portfolio
    'PortfolioOptimizer',
    'optimize_daf_portfolio',
    # LLM
    'LLMDonationAdvisor',
    'get_llm_advisor',
    'check_llm_status',
    # Federated Learning
    'FederatedLearningCoordinator',
    'FederatedConfig',
    'create_fl_system',
    'demonstrate_fl_usage',
    # GiveNudge (v0.4)
    'NudgeGenerator',
    'NudgeType',
    'NudgeChannel',
    'DonorProfile',
    'DonorSegment',
    'Nudge',
    'NudgeResult',
    'TimingEngine',
    'ABTestEngine',
    'ABTestResult',
    # Impact Feedback Loop (v0.4)
    'ImpactFeedbackLoop',
    'ImpactScorer',
    'SaturationDetector',
    'ReallocationEngine',
    'ImpactSignal',
    'ImpactCategory',
    'SignalDirection',
    'GrantAllocation',
    'ReallocationRecommendation',
]
