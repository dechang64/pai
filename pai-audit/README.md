# 💎 PAI — Philanthropic Asset Intelligence

**AI-Powered Charitable Investment Optimization, Giving Strategy & Impact Measurement**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-110%20passed-brightgreen.svg)]()
[![LOC](https://img.shields.io/badge/LOC-7%2C713-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Prototype v0.4 · [Gates Foundation Grand Challenges 2026](https://gcgh.grandchallenges.org/challenge/artificial-intelligence-ai-accelerate-charitable-giving) · Project 3: AI to Accelerate Charitable Giving*

---

## What is PAI?

PAI uses **Federated Retrieval-Augmented Generation (Federated RAG)** to ground every AI recommendation in verified institutional knowledge while ensuring sensitive data never leaves local servers. It uniquely unifies investment optimization, grant recommendations, and impact measurement into a **closed-loop system** where real-world outcomes continuously improve future decisions.

**Key stat:** A pre-registered experiment (White et al., 2026, N=1,949) showed LLM-based dialogue increases effective charitable donations by **45.9%** — the strongest causal evidence to date that AI can accelerate giving.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py

# Run tests
pytest tests/ -v
# 110 tests passed
```

---

## Modules

| Module | Description | Since |
|--------|-------------|-------|
| 📊 **InvestOpt** | Impact-aware portfolio optimization (Markowitz MVO) | v0.1 |
| 🤖 **GiveSmart** | LLM donation advisor with hallucination detection | v0.1 |
| 🎯 **ImpactLens** | Charity effectiveness evaluation (QALY/DALY) | v0.1 |
| 💡 **GiveNudge** | Behavioral engagement engine (warm-glow theory) | v0.4 |
| 🔄 **Impact Loop** | Closed-loop outcome measurement & reallocation | v0.4 |
| 🛡️ **FedShield** | Federated learning for privacy-preserving collaboration | v0.2 |
| 🔍 **Federated RAG** | Cross-institutional knowledge retrieval | v0.3 |

---

## Dashboard (7 Tabs)

| Tab | Module | Description |
|-----|--------|-------------|
| 1 | InvestOpt | Portfolio optimization with 63 mutual funds |
| 2 | GiveSmart | AI donation advisor with hallucination detection |
| 3 | ImpactLens | Charity effectiveness evaluation |
| 4 | GiveNudge | Behavioral engagement engine |
| 5 | Impact Loop | Closed-loop outcome measurement |
| 6 | Rare Disease | Rare disease foundation blueprint + FedShield |
| 7 | Federated RAG | Cross-institutional knowledge retrieval |

---

## Project Structure

```
pai-audit/
├── app.py                      # Streamlit dashboard (7 tabs)
├── pyproject.toml              # Package config (v0.4.0)
├── requirements.txt            # Dependencies
├── core/
│   ├── portfolio_optimizer.py  # Markowitz MVO + Impact-Aware
│   ├── llm_client.py           # GiveSmart LLM advisor
│   ├── give_nudge.py           # GiveNudge behavioral engine
│   ├── impact_feedback.py      # Impact Feedback Loop
│   ├── federated_learning.py   # FedShield reference
│   └── federated_rag/          # Federated RAG + Hallucination Detector
├── tests/                      # 110 tests
│   ├── test_portfolio.py
│   ├── test_llm_client.py
│   ├── test_federated_rag.py
│   └── test_v04_modules.py
└── data/knowledge_base/        # Verified reference data
```

---

## Key References

1. White, J.P. et al. (2026). [LLMs and charitable giving](https://osf.io/preprints/psyarxiv/6cyn4_v2). — **+45.9% effective donations**
2. Andreoni, J. (1990). [Impure Altruism](https://doi.org/10.2307/2234295). — **Warm-glow theory**
3. Lo, A. et al. (2025). [Nonprofit Endowment Funds](https://www.nber.org/digest/202511/investment-returns-nonprofit-endowments). — **Endowments underperform**
4. GiveWell. [Cost-Effectiveness](https://www.givewell.org). — **~$3,000-5,000 per life saved**
5. McMahan, B. et al. (2017). [FedAvg](https://arxiv.org/abs/1602.05629). — **Federated averaging**

---

## License

[MIT License](https://github.com/dechang64/PAI/blob/main/LICENSE)
