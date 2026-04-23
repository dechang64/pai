# 💎 PAI — Philanthropic Asset Intelligence

**AI-Powered Charitable Investment Optimization, Giving Strategy & Impact Measurement**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/Tests-41%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Prototype v0.3 · [Gates Foundation Grand Challenges 2026](https://gcgh.grandchallenges.org/challenge/artificial-intelligence-ai-accelerate-charitable-giving) · Project 3: AI to Accelerate Charitable Giving*

---

## Architecture

```
PAI/
├── app.py                    # Streamlit Cloud single-file version (v0.2)
├── pai-audit/                # Full modular version (v0.3) ← main development
│   ├── app.py                # Streamlit dashboard (5 tabs)
│   ├── core/
│   │   ├── portfolio_optimizer.py   # Markowitz MVO
│   │   ├── llm_client.py            # GiveSmart LLM advisor
│   │   ├── federated_learning.py    # FedShield reference
│   │   └── federated_rag/           # Federated RAG module (v0.3)
│   │       ├── config.py            # Dataclass configs
│   │       ├── document_loader.py   # PDF/CSV/JSON/MD loader
│   │       ├── embeddings.py        # sentence-transformers
│   │       ├── vector_store.py      # FAISS index
│   │       ├── retriever.py         # Cosine similarity search
│   │       ├── generator.py         # LLM answer generation
│   │       ├── local_rag.py         # Single-node RAG pipeline
│   │       ├── federated_query.py   # Multi-node federated router
│   │       ├── federated_trainer.py # FedAvg embedding fine-tuning
│   │       ├── streamlit_ui.py      # Streamlit integration
│   │       └── demo.py              # Standalone demo
│   ├── data/knowledge_base/         # 7 domain documents
│   └── tests/test_federated_rag.py  # 41 unit tests
├── pai-deploy/               # Streamlit Cloud deployment
├── pai-cloud/                # Cloud variant
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Modules

### 📊 InvestOpt — Portfolio Optimization
- **Markowitz Mean-Variance Optimization (MVO)** using SciPy
- 63 mutual funds with risk metrics (Sharpe, Sortino, Calmar, Jensen α)
- Maximum Sharpe Ratio & Minimum Variance portfolios
- Efficient frontier visualization
- DAF-specific optimization with charitable impact metrics

### 🤖 GiveSmart — AI Donation Advisor
- Personalized donation strategy based on donor profile
- Tax optimization (DAFs, bunching, appreciated securities)
- Charity matching using GiveWell/ACE evidence ratings
- Warm-glow theory (Andreoni 1990) integration
- **+45.9%** effective giving increase (White et al. 2026)
- **Federated RAG enrichment** — advice grounded in cross-institutional knowledge

### 🎯 ImpactLens — Impact Evaluation
- QALY/DALY-based charity scoring
- Cost per life saved analysis
- Evidence strength ratings
- Radar chart visualization

### 🔗 Federated RAG — Cross-Institutional Knowledge Retrieval (v0.3)
- **Privacy-preserving**: raw documents never leave local nodes
- **3 simulated institutions**: GiveWell, Tax Center, Behavioral Lab
- **sentence-transformers** (all-MiniLM-L6-v2) + **FAISS** vector search
- **FedAvg** fine-tuning for shared embedding model
- **LLM generation**: retrieved context → AI-powered answers
- **41 unit tests**, all passing

### 🛡️ FedShield — Federated Learning (Reference)
- FedAvg implementation for privacy-preserving collaboration
- Cross-institutional model training without data sharing

---

## Quick Start

### 1. Streamlit Cloud (Single File)
```bash
pip install streamlit pandas numpy plotly scipy
streamlit run app.py
```

### 2. Full Modular Version
```bash
cd pai-audit
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu  # Federated RAG
streamlit run app.py
```

### 3. Federated RAG Demo
```bash
cd pai-audit
python -m core.federated_rag
```

### 4. Run Tests
```bash
cd pai-audit
pip install pytest
pytest tests/test_federated_rag.py -v
```

---

## Federated RAG Deep Dive

### How It Works
```
User Query
    ↓
┌─────────────────────────────────────────────┐
│           Federated Query Router            │
├──────────┬──────────┬──────────┬────────────┤
│ GiveWell │ Tax Ctr  │ Behav Lab│  Remote... │
│ (local)  │ (local)  │ (local)  │  (gRPC)    │
│          │          │          │            │
│ FAISS    │ FAISS    │ FAISS    │            │
│ index    │ index    │ index    │            │
└────┬─────┴────┬─────┴────┬─────┴────────────┘
     │          │          │
     ↓          ↓          ↓
  scores     scores     scores    ← Only scores shared!
     │          │          │
     └──────────┴──────────┘
                ↓
         Aggregated Results
                ↓
         LLM Answer Generation
```

### Privacy Guarantees
- Raw documents **never** leave the local node
- Only document IDs + similarity scores are transmitted
- Federated fine-tuning uses **FedAvg** (no raw gradients shared)
- Optional privacy mode hides content in UI

### Knowledge Base (7 documents)
| Document | Domain | Lines |
|----------|--------|-------|
| `givewell_charities.md` | Charity evaluation | 19 |
| `impact_measurement.md` | Health economics (QALY/DALY) | 37 |
| `daf_tax_strategies.md` | DAF tax optimization | 26 |
| `daf_investment.md` | DAF investment (Markowitz) | 61 |
| `behavioral_economics.md` | Giving behavior science | 45 |
| `federated_learning.md` | FL theory & privacy | 62 |
| `rare_disease.md` | Rare disease economics | 62 |

---

## Key References

1. White, J.P. et al. (2026). [Increasing the effectiveness of charitable giving with AI-generated persuasion](https://osf.io/preprints/psyarxiv/6cyn4_v2). *PsyArXiv*. — **LLM dialogue increases effective donations by 45.9%**
2. Andreoni, J. (1990). [Impure Altruism and Donations to Public Goods](https://doi.org/10.2307/2234295). *The Economic Journal*, 100(401), 464-477. — **Warm-glow giving theory**
3. Lo, A., Matveyev, A., & Zeume, S. (2025). [The Risk, Reward, and Asset Allocation of Nonprofit Endowment Funds](https://www.nber.org/digest/202511/investment-returns-nonprofit-endowments). *NBER Working Paper*. — **Charity endowments systematically underperform**
4. Gates Foundation (2026). [AI to Accelerate Charitable Giving](https://gcgh.grandchallenges.org/challenge/artificial-intelligence-ai-accelerate-charitable-giving). *Grand Challenges RFP*.
5. GiveWell. [Cost-Effectiveness Analysis](https://www.givewell.org/how-we-work/our-criteria/cost-effectiveness). — **~$3,000-5,000 per life saved (top charities)**
6. McMahan, B. et al. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). *AISTATS*. — **FedAvg algorithm**

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
