# 💎 PAI - Philanthropic Asset Intelligence

**AI-Powered Charitable Investment Optimization for Streamlit Cloud**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Single File Version** — Ready for Streamlit Cloud deployment!

---

## Features

### 📊 InvestOpt — Portfolio Optimization
- **Markowitz Mean-Variance Optimization (MVO)** using SciPy
- 63 mutual funds with risk metrics (Sharpe, Sortino, Calmar, etc.)
- Maximum Sharpe Ratio & Minimum Variance portfolios
- Efficient frontier visualization
- DAF-specific optimization with charitable impact metrics

### 🤖 GiveSmart — AI Donation Advisor
- Personalized donation strategy based on donor profile
- Tax optimization (DAFs, bunching, appreciated securities)
- Charity matching using GiveWell/ACE evidence ratings
- Warm-glow theory (Andreoni 1990) integration
- **+45.9%** effective giving increase (White et al. 2026)

### 🎯 ImpactLens — Impact Evaluation
- QALY/DALY-based charity scoring
- Cost per life saved analysis
- Evidence strength ratings
- Radar chart visualization

---

## Deploy to Streamlit Cloud

### One-Click Deploy

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_github.svg)](https://share.streamlit.io/deploy)

Or manually:

1. Fork this repository to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "Deploy an app"
4. Select your forked repository
5. Set main file path: `app.py`

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

---

## Environment Variables (Optional)

For enhanced AI features, set:

```bash
# OpenAI (recommended)
export OPENAI_API_KEY=sk-...

# or Anthropic Claude
export ANTHROPIC_API_KEY=sk-ant-...
```

The app works in demo mode without these keys.

---

## Project Structure

```
pai-cloud/
├── app.py              # Single-file Streamlit app (no external dependencies)
├── requirements.txt     # Minimal dependencies (scipy included)
└── README.md          # This file
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│         PAI Streamlit Cloud App          │
├─────────────────────────────────────────┤
│  💰 InvestOpt    │  🤖 GiveSmart        │
│  Markowitz MVO   │  LLM Advisor        │
│  Portfolio Opt   │  Tax Strategy       │
├─────────────────────────────────────────┤
│  🎯 ImpactLens   │  📊 Data Viz        │
│  QALY/DALY      │  Plotly Charts      │
└─────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Charts | Plotly |
| Optimization | SciPy |
| Data | Pandas + NumPy |

---

## Related Projects

| Project | Description |
|---------|-------------|
| [FundFL](https://github.com/dechang64/FundFL) | Mutual fund analysis (Rust + Python) |
| [organoid-fl](https://github.com/dechang64/organoid-fl) | Federated learning for medical images |
| [defect-fl](https://github.com/dechang64/defect-fl) | PCB defect detection with FL |

---

## License

MIT License

---

*Built for Gates Foundation Grand Challenges 2026*
*Project 3: AI to Accelerate Charitable Giving*
