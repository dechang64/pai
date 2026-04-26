# 💎 PAI — Philanthropic Asset Intelligence

**AI-Powered Charitable Investment Optimization, Giving Strategy & Impact Measurement**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Streamlit Cloud Ready** — Single-file version, zero configuration.

---

## Features

### 📊 InvestOpt — Impact-Aware Portfolio Optimization
- Markowitz Mean-Variance Optimization (MVO) with SciPy
- **Impact-Aware strategy** (v0.4): jointly maximizes financial return × charitable impact
- Maximum Sharpe, Minimum Variance, Risk Parity strategies
- Efficient frontier visualization
- DAF-specific optimization with charitable impact metrics

### 🤖 GiveSmart — AI Donation Advisor
- Personalized donation strategy based on donor profile
- **Hallucination Detection** (v0.4): confidence-scored claims with verified sourcing
- Tax optimization (DAFs, bunching, appreciated securities)
- Charity matching using GiveWell/ACE evidence ratings
- **+45.9%** effective giving increase (White et al. 2026)

### 🎯 ImpactLens — Charity Effectiveness Evaluation
- QALY/DALY-based charity scoring
- Cost per life saved analysis
- Evidence strength ratings
- Radar chart visualization

### 💡 GiveNudge — Behavioral Engagement Engine (v0.4)
- 6 nudge strategies: social proof, matching, impact framing, urgency, warm-glow, default
- Optimal timing analysis (12月 = 1.45× conversion)
- 5 donor segments with tailored strategies
- Built-in A/B testing framework

### 🔄 Impact Feedback Loop (v0.4)
- Closed-loop: allocation → execution → measurement → feedback
- Impact saturation detection (diminishing returns curves)
- Automatic reallocation recommendations
- Cross-institutional benchmarking

### 🔬 Rare Disease Blueprint
- 3亿 patients, 7,000+ diseases, 95% no approved treatment
- Venture Philanthropy economic model
- FedShield integration for cross-institutional patient data

### 🔍 Federated RAG
- Knowledge sharing without data sharing
- FAISS vector search + sentence-transformers
- Privacy-preserving cross-institutional retrieval

---

## Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New app → select this repo → `pai-cloud/app.py`
4. Requirements: `pai-cloud/requirements.txt`
5. Deploy!

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Charts | Plotly |
| Optimization | SciPy |
| Data | Pandas + NumPy |

---

## License

[MIT License](https://github.com/dechang64/PAI/blob/main/LICENSE)

---

*Built for Gates Foundation Grand Challenges 2026*
