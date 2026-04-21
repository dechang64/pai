# PAI — Philanthropic Asset Intelligence

> AI驱动的慈善投资优化、捐赠策略与效果评估系统
> 
> Prototype v0.1 — Gates Foundation Grand Challenges 2026 · Project 3

## 快速启动

```bash
pip install streamlit plotly pandas numpy
streamlit run app.py --server.port 8501
```

浏览器打开 `http://localhost:8501`

## 四个模块

### 1. 💰 InvestOpt — 慈善投资优化
- 63只基金的风险画像（Sharpe/Sortino/Treynor/Jensen Alpha/VaR/CVaR）
- 投资组合构建与有效前沿可视化
- DAF投资组合优化建议
- **核心数据来源：** FundFL项目（63只基金，60个月度数据）

### 2. 🤖 GiveSmart — AI捐赠顾问
- LLM驱动的捐赠策略对话
- 税务优化建议（增值证券捐赠、Bunching、DAF vs CRT）
- 捐赠者画像与个性化推荐
- **核心理论：** Warm-Glow Giving (Andreoni 1990), Nudge Theory

### 3. 🎯 ImpactLens — 公益效果评估
- 基于QALY/DALY的公益项目效果评分
- 类Morningstar的星级系统
- 捐赠效果追踪仪表盘
- **核心数据：** GiveWell cost-effectiveness数据

### 4. 🔗 Rare Disease Blueprint — 罕见病基金会蓝图
- Venture Philanthropy模式
- 联邦学习患者注册表架构
- "投资-捐赠-研发"三角模型
- 三阶段建设路径

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit + Plotly |
| 数据分析 | Pandas + NumPy |
| AI对话 | z-ai-web-dev-sdk (LLM) |
| 隐私计算 | Federated Learning (organoid-fl) |
| 审计 | 区块链审计链 |

## 文件结构

```
pai_prototype/
├── app.py                          # 主应用（864行）
├── README.md                       # 本文件
└── Gates_GC2026_Project3_Proposal_Framework.md  # 提案框架文档
```

## 关键文献

1. White et al. (2026) "Increasing the effectiveness of charitable giving with AI-generated persuasion" — LLM对话+45.9%
2. Andreoni (1990) "Impure Altruism and Donations to Public Goods" — warm-glow理论
3. Lo, Matveyev, Zeume (2025) NBER — 慈善基金投资系统性跑输市场
4. WEF (2026) "Making Rare Diseases Count" — 罕见病万亿机会
5. GiveWell — cost per life saved数据

## 许可

MIT License — 用于Gates Foundation Grand Challenges 2026提案展示
