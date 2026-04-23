# 💎 PAI — Philanthropic Asset Intelligence

**AI-Powered Charitable Investment Optimization, Giving Strategy & Impact Measurement**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Prototype v0.2 · [Gates Foundation Grand Challenges 2026](https://gcgh.grandchallenges.org/challenge/artificial-intelligence-ai-accelerate-charitable-giving) · Project 3: AI to Accelerate Charitable Giving*

---

## What's New in v0.2

### ✅ Completed Fixes

1. **Real Portfolio Optimization (Markowitz MVO)**
   - Implemented Mean-Variance Optimization using SciPy
   - Supports: Maximum Sharpe, Minimum Variance, Risk Parity strategies
   - Efficient frontier visualization
   - DAF-specific optimization with charitable impact metrics

2. **LLM Integration**
   - Real OpenAI/Anthropic API integration
   - Fallback demo mode when API keys not available
   - Personalized donation advice based on donor profile
   - Tax optimization recommendations

3. **Federated Learning Module (FedShield)**
   - Reference implementation for privacy-preserving collaboration
   - FedAvg aggregation algorithm
   - Blockchain-like audit trail
   - Integration point for organoid-fl

4. **Code Modularization**
   - Separated into `core/` module package
   - `portfolio_optimizer.py` - Markowitz MVO
   - `llm_client.py` - LLM integration
   - `federated_learning.py` - Privacy-preserving FL

---

## The Problem

Charitable ecosystems suffer from **three interconnected efficiency losses** that are typically treated in isolation:

| Waste | Evidence | Scale |
|-------|----------|-------|
| **Investment Waste** — Charity endowments systematically underperform market benchmarks | NBER 2025: shorting charity portfolios vs. 60/40 yields positive returns | $326B trapped in US DAFs alone |
| **Information Waste** — Donors can't find effective causes; 4 of 5 first-time donors never return | Global giving participation dropped to 33% (CAF 2024); first-time retention: 19.4% | Only 5% of HIC donations reach international causes |
| **Impact Waste** — Cost-effectiveness evaluation is fragmented and unscalable | GiveWell covers only a handful of organizations; EA represents ~1% of global giving | 300M+ rare disease patients, 95% without approved treatments |

**PAI connects these three disconnected nodes into a closed-loop system.**

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       PAI System                                │
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  💰 InvestOpt │    │  🤖 GiveSmart │    │  🎯 ImpactLens│     │
│   │              │    │              │    │              │     │
│   │  Markowitz   │    │   LLM API    │    │ QALY/DALY   │     │
│   │  MVO Opt.    │    │  Dialogue    │    │  Scoring     │     │
│   │ Portfolio    │    │ Donor Match  │    │ Star Rating  │     │
│   │ optimization │    │ Tax Optimize │    │ Tracking     │     │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│          │                   │                   │              │
│          └───────────────────┼───────────────────┘              │
│                              │                                  │
│                              ▼                                  │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │   🔒 FedShield — Federated Learning Privacy Layer        │ │
│   │   Privacy-Preserving | Cross-Institutional | Audit Trail  │ │
│   └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
          ▲                ▲                ▲
          │                │                │
┌─────────┴───────┐┌──────┴─────┐┌─────────┴───────┐
│ FundFL          ││ organoid-fl ││ defect-fl      │
│ (Finance)       ││ (Medical)   ││ (PCB)          │
└─────────────────┘└─────────────┘└─────────────────┘
```

---

## Modules

### 💰 InvestOpt — Charitable Investment Optimization

Optimizes how charitable assets are invested *before* they're granted.

- **63 mutual funds** with full risk profiling: Sharpe, Sortino, Treynor, Jensen's Alpha, VaR, CVaR, M², Calmar Ratio
- **Mean-Variance Optimization (Markowitz MVO)** using SciPy:
  - Maximum Sharpe Ratio portfolio
  - Minimum Variance portfolio
  - Risk Parity portfolio
  - DAF-specific optimization
- **Efficient frontier visualization** for portfolio construction
- **DAF portfolio optimization** — addressing the $326B in underperforming donor-advised funds
- Built on [**FundFL**](https://github.com/dechang64/FundFL) — open-source fund analysis platform

### 🤖 GiveSmart — AI Giving Advisor

Converts donor *intent* into *action* using LLM-powered personalization.

- **Real LLM integration** (OpenAI GPT-4o / Anthropic Claude)
  - Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable
  - Falls back to demo mode when not configured
- Interactive donation strategy consultation via LLM dialogue
- **Tax optimization engine**: appreciated securities donation, bunching strategy, DAF vs. CRT comparison
- **Donor profiling**: risk preferences, value alignment, behavioral nudges
- Grounded in **Warm-Glow Giving** theory (Andreoni 1990) and **Nudge Theory** (Thaler & Sunstein)
- Evidence: LLM-personalized dialogues increase effective donations by **45.9%** (White et al. 2026)

### 🎯 ImpactLens — Impact Measurement & Evaluation

Measures *where the money goes* and *what it achieves*.

- QALY/DALY-based charity effectiveness scoring
- Morningstar-style star rating system for charitable organizations
- Donation impact tracking dashboard
- Benchmarked against **GiveWell** cost-effectiveness data

### 🔒 FedShield — Federated Learning Layer

Privacy-preserving cross-institutional data collaboration — the technical differentiator.

- **Reference implementation** of federated learning for rare disease research
- Enables multi-site model training **without sharing raw data**
- Directly reusable from [**organoid-fl**](https://github.com/dechang64/organoid-fl) (99.17% accuracy, medical image segmentation)
- FedAvg aggregation algorithm (McMahan et al. 2017)
- Blockchain audit trail for training provenance
- Critical for rare disease patient registries, hospital networks, and international NGO data partnerships

---

## Installation

```bash
# Clone the repository
git clone https://github.com/dechang64/pai.git
cd pai

# Install dependencies
pip install -r requirements.txt

# Optional: Set LLM API keys for full AI features
export OPENAI_API_KEY=sk-your-key-here
# or
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Quick Start

```bash
# Run the prototype
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly |
| Data Analysis | Pandas + NumPy |
| Portfolio Optimization | SciPy (Markowitz MVO) |
| AI / NLP | OpenAI GPT-4o / Anthropic Claude |
| Privacy Computing | Federated Learning (organoid-fl) |
| Audit Trail | Blockchain (organoid-fl) |
| Fund Analysis | [FundFL](https://github.com/dechang64/FundFL) |

---

## Project Structure

```
pai/
├── app.py                      # Main Streamlit application
├── requirements.txt             # Dependencies
├── core/                        # Core modules package
│   ├── __init__.py
│   ├── portfolio_optimizer.py   # Markowitz MVO implementation
│   ├── llm_client.py            # LLM integration
│   └── federated_learning.py    # FedShield reference impl
└── README.md
```

---

## Related Projects

| Project | Description | Stars |
|---------|-------------|-------|
| [**FundFL**](https://github.com/dechang64/FundFL) | Open-source mutual fund analysis & risk profiling (Rust + Python) | — |
| [**organoid-fl**](https://github.com/dechang64/organoid-fl) | Federated learning for medical image segmentation (99.17% accuracy) | ⭐1 |
| [**defect-fl**](https://github.com/dechang64/defect-fl) | PCB defect detection with federated continual learning | — |

---

## Key References

1.  White, J.P. et al. (2026). [Increasing the effectiveness of charitable giving with AI-generated persuasion](https://osf.io/preprints/psyarxiv/6cyn4_v2). *PsyArXiv*. — **LLM dialogue increases effective donations by 45.9%**
2.  Andreoni, J. (1990). [Impure Altruism and Donations to Public Goods](https://doi.org/10.2307/2234295). *The Economic Journal*, 100(401), 464-477. — **Warm-glow giving theory**
3.  Lo, A., Matveyev, A., & Zeume, S. (2025). [The Risk, Reward, and Asset Allocation of Nonprofit Endowment Funds](https://www.nber.org/digest/202511/investment-returns-nonprofit-endowments). *NBER Working Paper*. — **Charity endowments systematically underperform**
4.  Gates Foundation (2026). [AI to Accelerate Charitable Giving](https://gcgh.grandchallenges.org/challenge/artificial-intelligence-ai-accelerate-charitable-giving). *Grand Challenges RFP*.
5.  GiveWell. [Cost-Effectiveness Analysis](https://www.givewell.org/how-we-work/our-criteria/cost-effectiveness). — **~$3,000-5,000 per life saved (top charities)**
6.  McMahan, B. et al. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). *AISTATS*. — **FedAvg algorithm**

---

## License

MIT License

---

*Built for the Gates Foundation Grand Challenges 2026*
