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

## Architecture

```
PAI/
├── app.py                          # Streamlit Cloud single-file version (v0.4, 1,226 lines)
├── requirements.txt                # Cloud dependencies
├── LICENSE                         # MIT License
├── README.md                       # This file
│
└── pai-audit/                      # Full modular version (v0.4) ← main development
    ├── app.py                      # Streamlit dashboard (7 tabs)
    ├── pyproject.toml              # Package config (v0.4.0)
    ├── requirements.txt            # Full dependencies
    ├── core/
    │   ├── __init__.py             # Module exports
    │   ├── portfolio_optimizer.py  # Markowitz MVO + Impact-Aware optimization
    │   ├── llm_client.py           # GiveSmart LLM advisor
    │   ├── give_nudge.py           # GiveNudge behavioral engine
    │   ├── impact_feedback.py      # Impact Feedback Loop
    │   ├── federated_learning.py   # FedShield reference implementation
    │   └── federated_rag/          # Federated RAG module
    │       ├── __init__.py
    │       ├── config.py           # Dataclass configs
    │       ├── document_loader.py  # PDF/CSV/JSON/MD loader
    │       ├── embeddings.py       # sentence-transformers wrapper
    │       ├── vector_store.py     # FAISS vector store
    │       ├── federated_client.py # Federated aggregation client
    │       ├── reranker.py         # Cross-encoder reranker
    │       ├── hallucination_detector.py  # Claim verification engine
    │       └── streamlit_ui.py     # Streamlit integration
    ├── tests/
    │   ├── test_portfolio.py       # Portfolio optimizer tests
    │   ├── test_llm_client.py      # LLM advisor tests
    │   ├── test_federated_rag.py   # Federated RAG tests
    │   └── test_v04_modules.py     # GiveNudge + Impact Loop + Hallucination tests
    └── data/
        └── knowledge_base/         # Verified reference data
            ├── givewell_charities.md
            ├── daf_investment.md
            ├── daf_tax_strategies.md
            ├── impact_measurement.md
            ├── behavioral_economics.md
            ├── rare_disease.md
            └── federated_learning.md
```

---

## Modules

### 📊 InvestOpt — Impact-Aware Portfolio Optimization
Extends Markowitz Mean-Variance Optimization to jointly maximize financial returns and charitable impact. Recognizes that the highest-returning portfolio may produce less good if gains flow to saturated programs.

- Maximum Sharpe, Minimum Variance, Risk Parity strategies
- **Impact-Aware optimization** (v0.4): maximizes `U = α·Sharpe + β·Impact`
- Efficient frontier visualization
- DAF-specific optimization with charitable impact metrics
- Financial vs. Impact portfolio comparison

### 🤖 GiveSmart — LLM Donation Advisor
Personalized donation strategy with hallucination detection. Every claim is verified against institutional knowledge bases.

- Real OpenAI/Anthropic API integration (demo fallback)
- Tax optimization (DAFs, bunching, appreciated securities)
- Charity matching using GiveWell/ACE evidence ratings
- **Hallucination Detection** (v0.4): claim extraction → KB verification → confidence scoring
- **+45.9%** effective giving increase (White et al. 2026)

### 🎯 ImpactLens — Charity Effectiveness Evaluation
Multi-dimensional charity scoring aligned with GiveWell's cost-effectiveness methodology.

- QALY/DALY-based scoring
- Cost per life saved analysis
- Evidence strength ratings
- Radar chart visualization

### 💡 GiveNudge — Behavioral Engagement Engine *(v0.4)*
Data-driven donation optimization based on behavioral economics (Andreoni 1990 warm-glow theory).

- 6 nudge types: default reminder, social proof, matching, impact framing, urgency, warm-glow
- 4 delivery channels: email, in-app, SMS, push
- 5 donor segments with tailored strategies
- Built-in A/B testing framework
- Optimal timing engine (year-end 1.45× conversion boost)

### 🔄 Impact Feedback Loop *(v0.4)*
PAI's most transformative component — connects measured outcomes to future allocation decisions.

- Effect signal collection (health, education, environmental metrics)
- **Saturation detection**: diminishing returns curves per cause area
- Automatic reallocation recommendations
- "What happened" → "What should we do next" closed loop

### 🛡️ FedShield — Federated Learning
Reference implementation for privacy-preserving cross-institutional collaboration.

- FedAvg algorithm (McMahan et al. 2017)
- Differential privacy support
- Audit trail with blockchain-style logging
- Simulated multi-institution training

### 🔍 Federated RAG — Cross-Institutional Knowledge Retrieval
First architecture combining LLM conversational ability, RAG factual grounding, and federated privacy for charitable giving.

- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector search: FAISS
- Cross-encoder reranking (ms-marco-MiniLM)
- Federated embedding training
- Knowledge sharing **without** data sharing

---

## Quick Start

### Streamlit Cloud (single file)
```bash
pip install streamlit pandas numpy plotly scipy
streamlit run app.py
```

### Full modular version
```bash
cd pai-audit
pip install -r requirements.txt
streamlit run app.py
```

### Run tests
```bash
cd pai-audit
pytest tests/ -v
# 110 tests passed
```

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

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Charts | Plotly |
| Optimization | SciPy |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| Federated Training | PyTorch + FedAvg |
| LLM | OpenAI / Anthropic / Demo fallback |
| Data | Pandas + NumPy |

---

## Key References

1. White, J.P. et al. (2026). [Increasing the effectiveness of charitable giving with AI-generated persuasion](https://osf.io/preprints/psyarxiv/6cyn4_v2). *PsyArXiv*. — **LLM dialogue increases effective donations by 45.9%**
2. Andreoni, J. (1990). [Impure Altruism and Donations to Public Goods](https://doi.org/10.2307/2234295). *The Economic Journal*, 100(401), 464-477. — **Warm-glow giving theory**
3. Lo, A., Matveyev, A., & Zeume, S. (2025). [The Risk, Reward, and Asset Allocation of Nonprofit Endowment Funds](https://www.nber.org/digest/202511/investment-returns-nonprofit-endowments). *NBER Working Paper*. — **Charity endowments systematically underperform**
4. Gates Foundation (2026). [AI to Accelerate Charitable Giving](https://gcgh.grandchallenges.org/challenge/artificial-intelligence-ai-accelerate-charitable-giving). *Grand Challenges RFP*.
5. GiveWell. [Cost-Effectiveness Analysis](https://www.givewell.org/how-we-work/our-criteria/cost-effectiveness). — **~$3,000-5,000 per life saved (top charities)**
6. McMahan, B. et al. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). *AISTATS*. — **FedAvg algorithm**

---

## Related Projects

| Project | Description |
|---------|-------------|
| [FundFL](https://github.com/dechang64/FundFL) | Mutual fund analysis (Rust + Python) |
| [organoid-fl](https://github.com/dechang64/organoid-fl) | Federated learning for medical images |
| [defect-fl](https://github.com/dechang64/defect-fl) | PCB defect detection with FL |

---

## License

[MIT License](LICENSE)

---

*Built for the Gates Foundation Grand Challenges 2026*
