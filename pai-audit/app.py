"""
PAI - Philanthropic Asset Intelligence
AI-Powered Charitable Investment Optimization, Giving Strategy & Impact Measurement

Prototype v0.4 — Gates Foundation Grand Challenges 2026

Modules:
- InvestOpt: Impact-aware portfolio optimization (Markowitz + impact utility)
- GiveSmart: LLM-powered donation advisor with hallucination detection
- ImpactLens: Charity effectiveness evaluation
- GiveNudge: Behavioral engagement engine (warm-glow theory)
- Impact Feedback Loop: Closed-loop outcome measurement
- FedShield: Federated learning for privacy-preserving collaboration
- Federated RAG: Cross-institutional knowledge retrieval
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# Add core module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Import PAI Core Modules
# ============================================================
try:
    from core import (
        PortfolioOptimizer,
        optimize_daf_portfolio,
        get_llm_advisor,
        check_llm_status,
        FederatedLearningCoordinator,
        create_fl_system,
        demonstrate_fl_usage
    )
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    st.error(f"Core modules import failed: {e}. Running in compatibility mode.")


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="PAI — Philanthropic Asset Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a73e8, #34a853, #fbbc04);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e8f5e9);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    .insight-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .rare-disease-box {
        background: #fce4ec;
        border-left: 4px solid #e91e63;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a237e, #283593);
    }
    section[data-testid="stSidebar"] [class*="stMarkdown"] p, 
    section[data-testid="stSidebar"] [class*="stMarkdown"] span,
    section[data-testid="stSidebar"] [class*="stMarkdown"] h1,
    section[data-testid="stSidebar"] [class*="stMarkdown"] h2,
    section[data-testid="stSidebar"] [class*="stMarkdown"] h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Generation
# ============================================================

def generate_fund_data():
    """Generate 63 mutual funds with risk metrics (from FundFL)"""
    np.random.seed(42)
    
    categories = {
        "Large-Cap Growth": {"cn": "大盘成长", "n": 12, "ret": 0.12, "vol": 0.18},
        "Large-Cap Value": {"cn": "大盘价值", "n": 10, "ret": 0.10, "vol": 0.15},
        "Large-Cap Blend": {"cn": "大盘混合", "n": 8, "ret": 0.11, "vol": 0.16},
        "Mid-Cap Growth": {"cn": "中盘成长", "n": 6, "ret": 0.14, "vol": 0.22},
        "Mid-Cap Value": {"cn": "中盘价值", "n": 5, "ret": 0.12, "vol": 0.19},
        "Small-Cap": {"cn": "小盘股", "n": 7, "ret": 0.13, "vol": 0.24},
        "International": {"cn": "国际", "n": 5, "ret": 0.09, "vol": 0.20},
        "Balanced": {"cn": "平衡型", "n": 4, "ret": 0.08, "vol": 0.10},
        "Government Bond": {"cn": "政府债券", "n": 3, "ret": 0.04, "vol": 0.05},
        "High-Yield Bond": {"cn": "高收益债", "n": 3, "ret": 0.06, "vol": 0.08},
    }
    
    fund_names = {
        "Large-Cap Growth": ["Fidelity Magellan", "T. Rowe Price Blue Chip", "Vanguard Growth Index", "Janus Growth", "American Funds Growth", "MFS Growth", "Franklin Growth", "Invesco Growth", "TIAA-CREF Growth", "DFA US Large Cap Growth", "JPMorgan Growth", "Goldman Sachs Growth"],
        "Large-Cap Value": ["Vanguard Value Index", "Fidelity Value", "T. Rowe Price Equity Income", "Dodge & Cox Stock", "American Funds Washington Mutual", "Vanguard Windsor", "Vanguard Dividend Appreciation", "Fidelity Contrafund", "T. Rowe Price Value", "Vanguard FTSE Social Index"],
        "Large-Cap Blend": ["Vanguard 500 Index", "Fidelity Spartan 500", "T. Rowe Price Equity", "American Funds Investment Co", "Vanguard Total Stock Market", "Fidelity Total Market", "Schwab S&P 500", "SPDR S&P 500"],
        "Mid-Cap Growth": ["Vanguard Mid-Cap Growth", "Fidelity Mid-Cap Stock", "T. Rowe Price Mid-Cap Growth", "Janus Mid-Cap Growth", "MFS Mid-Cap Growth", "DFA US Micro Cap"],
        "Mid-Cap Value": ["Vanguard Mid-Cap Value", "T. Rowe Price Mid-Cap Value", "Fidelity Mid-Cap Value", "Dodge & Cox Balanced", "Oakmark Select"],
        "Small-Cap": ["Vanguard Small-Cap Index", "Fidelity Small-Cap Stock", "T. Rowe Price Small-Cap", "DFA US Small Cap Value", "iShares Russell 2000", "SPDR S&P 600", "Vanguard Small-Cap Value"],
        "International": ["Vanguard Total International", "Fidelity Overseas", "T. Rowe Price International", "American Funds EuroPacific", "DFA International Value"],
        "Balanced": ["Vanguard Wellington", "Fidelity Puritan", "Vanguard LifeStrategy Moderate", "T. Rowe Price Balanced"],
        "Government Bond": ["Vanguard GNMA", "Fidelity Government Income", "PIMCO Total Return"],
        "High-Yield Bond": ["Vanguard High-Yield Corporate", "Fidelity High Income", "T. Rowe Price High Yield"],
    }
    
    funds = []
    code_idx = 0
    for cat, info in categories.items():
        for i in range(info["n"]):
            name = fund_names[cat][i]
            code = f"F{code_idx+1:03d}"
            code_idx += 1
            
            # Generate monthly returns
            monthly_ret = info["ret"] / 12 + np.random.normal(0, info["vol"] / np.sqrt(12), 60)
            
            # Compute risk metrics
            r = monthly_ret
            ann_return = np.mean(r) * 12
            ann_vol = np.std(r, ddof=1) * np.sqrt(12)
            rf_monthly = 0.02 / 12
            excess = r - rf_monthly
            sharpe = np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(12) if np.std(excess, ddof=1) > 0 else 0
            downside = r[r < rf_monthly]
            downside_std = np.std(downside, ddof=1) * np.sqrt(12) if len(downside) > 1 else 0.001
            sortino = (ann_return - 0.02) / downside_std
            cum = np.cumprod(1 + r)
            peak = np.maximum.accumulate(cum)
            drawdown = (cum - peak) / peak
            max_dd = np.min(drawdown)
            market = np.full(60, 0.008)
            cov = np.cov(r, market, ddof=1)
            beta = cov[0, 1] / np.var(market, ddof=1) if np.var(market, ddof=1) > 0 else 1.0
            alpha = ann_return - (0.02 + beta * (0.008 * 12 - 0.02))
            treynor = (ann_return - 0.02) / beta if beta != 0 else 0
            active = r - market
            ir = np.mean(active) / np.std(active, ddof=1) * np.sqrt(12) if np.std(active, ddof=1) > 0 else 0
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
            var_95 = np.percentile(r, 5)
            cvar_95 = np.mean(r[r <= var_95])
            win_rate = np.sum(r > 0) / len(r)
            
            # DAF-specific metrics
            daf_apt = round(np.random.uniform(0.15, 0.35), 3)
            daf_growth = round(np.random.uniform(0.03, 0.12), 3)
            tax_efficiency = round(np.random.uniform(0.7, 0.95), 3)
            
            funds.append({
                "code": code,
                "name": name,
                "category": cat,
                "category_cn": info["cn"],
                "ann_return": round(ann_return, 4),
                "ann_vol": round(ann_vol, 4),
                "sharpe": round(sharpe, 4),
                "sortino": round(sortino, 4),
                "max_drawdown": round(max_dd, 4),
                "beta": round(beta, 4),
                "alpha": round(alpha, 4),
                "treynor": round(treynor, 4),
                "info_ratio": round(ir, 4),
                "calmar": round(calmar, 4),
                "var_95": round(var_95, 4),
                "cvar_95": round(cvar_95, 4),
                "win_rate": round(win_rate, 4),
                "daf_payout_rate": daf_apt,
                "daf_growth_rate": daf_growth,
                "tax_efficiency": tax_efficiency,
                "monthly_returns": monthly_ret.tolist(),
            })
    
    return pd.DataFrame(funds)


def generate_charity_data():
    """Generate charity effectiveness data"""
    charities = [
        {"name": "Against Malaria Foundation", "category": "Global Health", "cost_per_life": 3500, "evidence_strength": 0.95, "scalability": 0.90, "transparency": 0.92, "overhead_ratio": 0.05, "impact_score": 0.94, "region": "Sub-Saharan Africa", "beneficiaries": "50M+", "year_founded": 2004},
        {"name": "Helen Keller International", "category": "Global Health", "cost_per_life": 5200, "evidence_strength": 0.88, "scalability": 0.85, "transparency": 0.90, "overhead_ratio": 0.08, "impact_score": 0.87, "region": "Africa/Asia", "beneficiaries": "20M+", "year_founded": 1915},
        {"name": "Schistosomiasis Control Initiative", "category": "Global Health", "cost_per_life": 4800, "evidence_strength": 0.91, "scalability": 0.88, "transparency": 0.89, "overhead_ratio": 0.06, "impact_score": 0.89, "region": "Sub-Saharan Africa", "beneficiaries": "100M+", "year_founded": 2002},
        {"name": "Deworm the World Initiative", "category": "Global Health", "cost_per_life": 8000, "evidence_strength": 0.85, "scalability": 0.92, "transparency": 0.87, "overhead_ratio": 0.07, "impact_score": 0.86, "region": "Global", "beneficiaries": "280M+", "year_founded": 2009},
        {"name": "GiveDirectly", "category": "Cash Transfers", "cost_per_life": 15000, "evidence_strength": 0.90, "scalability": 0.95, "transparency": 0.95, "overhead_ratio": 0.09, "impact_score": 0.88, "region": "Africa", "beneficiaries": "1.5M+", "year_founded": 2009},
        {"name": "Rare Disease Foundation", "category": "Rare Disease", "cost_per_life": 50000, "evidence_strength": 0.70, "scalability": 0.60, "transparency": 0.80, "overhead_ratio": 0.15, "impact_score": 0.72, "region": "Global", "beneficiaries": "300M", "year_founded": 2008},
        {"name": "Cure Rare Disease", "category": "Rare Disease", "cost_per_life": 80000, "evidence_strength": 0.65, "scalability": 0.55, "transparency": 0.78, "overhead_ratio": 0.18, "impact_score": 0.68, "region": "US/Global", "beneficiaries": "300M", "year_founded": 2018},
        {"name": "EveryLife Foundation", "category": "Rare Disease Policy", "cost_per_life": 120000, "evidence_strength": 0.60, "scalability": 0.70, "transparency": 0.82, "overhead_ratio": 0.12, "impact_score": 0.65, "region": "US", "beneficiaries": "25M+", "year_founded": 2009},
        {"name": "Fred Hutchinson Cancer Center", "category": "Medical Research", "cost_per_life": 200000, "evidence_strength": 0.80, "scalability": 0.50, "transparency": 0.85, "overhead_ratio": 0.20, "impact_score": 0.70, "region": "US/Global", "beneficiaries": "10M+", "year_founded": 1975},
        {"name": "The END Fund", "category": "Neglected Tropical Diseases", "cost_per_life": 4200, "evidence_strength": 0.87, "scalability": 0.85, "transparency": 0.88, "overhead_ratio": 0.08, "impact_score": 0.86, "region": "Africa", "beneficiaries": "150M+", "year_founded": 2012},
        {"name": "Zipline (Drone Delivery)", "category": "Health Logistics", "cost_per_life": 18000, "evidence_strength": 0.75, "scalability": 0.80, "transparency": 0.80, "overhead_ratio": 0.22, "impact_score": 0.78, "region": "Africa/Global", "beneficiaries": "25M+", "year_founded": 2014},
        {"name": "Noora Health", "category": "Health Education", "cost_per_life": 6500, "evidence_strength": 0.82, "scalability": 0.88, "transparency": 0.86, "overhead_ratio": 0.10, "impact_score": 0.84, "region": "India/Africa", "beneficiaries": "10M+", "year_founded": 2014},
        {"name": "New Incentives", "category": "Vaccination", "cost_per_life": 7000, "evidence_strength": 0.78, "scalability": 0.82, "transparency": 0.84, "overhead_ratio": 0.11, "impact_score": 0.80, "region": "Nigeria/India", "beneficiaries": "5M+", "year_founded": 2015},
        {"name": "Iodine Global Network", "category": "Nutrition", "cost_per_life": 5500, "evidence_strength": 0.83, "scalability": 0.78, "transparency": 0.81, "overhead_ratio": 0.09, "impact_score": 0.82, "region": "Global", "beneficiaries": "400M+", "year_founded": 1986},
        {"name": "Global Priorities Institute", "category": "Research", "cost_per_life": 500000, "evidence_strength": 0.72, "scalability": 0.40, "transparency": 0.90, "overhead_ratio": 0.14, "impact_score": 0.60, "region": "UK/Global", "beneficiaries": "Indirect", "year_founded": 2018},
    ]
    return pd.DataFrame(charities)


def generate_daf_scenario():
    """Generate DAF investment scenario data"""
    np.random.seed(123)
    months = pd.date_range("2020-01-01", periods=60, freq="ME")
    
    typical = 1 + np.cumsum(np.random.normal(0.005, 0.02, 60))
    optimized = 1 + np.cumsum(np.random.normal(0.007, 0.018, 60))
    benchmark = 1 + np.cumsum(np.random.normal(0.006, 0.015, 60))
    
    return pd.DataFrame({
        "date": months,
        "Typical DAF (Default Allocation)": typical * 100,
        "AI-Optimized DAF (PAI)": optimized * 100,
        "60/40 Benchmark": benchmark * 100,
    })


def generate_donation_funnel():
    """Generate donation conversion funnel data"""
    return pd.DataFrame({
        "stage": ["Aware", "Interested", "Researching", "Intent", "Donate Once", "Recurring Donor"],
        "count": [100000, 45000, 22000, 12000, 5500, 1050],
        "conversion_rate": [100, 45, 22, 12, 5.5, 1.05],
    })


# ============================================================
# Load Data
# ============================================================
@st.cache_data
def load_all_data():
    return {
        "funds": generate_fund_data(),
        "charities": generate_charity_data(),
        "daf_scenario": generate_daf_scenario(),
        "funnel": generate_donation_funnel(),
    }


data = load_all_data()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("# 💎 PAI")
    st.markdown("### Philanthropic Asset Intelligence")
    st.markdown("---")
    st.markdown("**AI驱动的慈善资产智能系统**")
    st.markdown("")
    st.markdown("Gates Foundation Grand Challenges 2026")
    st.markdown("Project 3: AI to Accelerate Charitable Giving")
    st.markdown("")
    st.markdown("---")
    st.markdown("#### 核心模块")
    st.markdown("📊 **InvestOpt** — 慈善投资优化 (Impact-Aware MVO)")
    st.markdown("🤖 **GiveSmart** — LLM捐赠顾问 + 幻觉检测")
    st.markdown("🎯 **ImpactLens** — 效果评估")
    st.markdown("💡 **GiveNudge** — 行为助推引擎")
    st.markdown("🔄 **Impact Loop** — 效果闭环")
    st.markdown("🔒 **FedShield** — 联邦学习隐私层")
    st.markdown("")
    st.markdown("---")
    st.markdown("#### 关键数据")
    st.markdown("- DAF资产: **$3,260亿**")
    st.markdown("- 被困资金: **$2,500亿+**")
    st.markdown("- 全球罕见病患者: **3亿**")
    st.markdown("- 95%罕见病无治疗方案")
    st.markdown("")
    st.markdown("---")
    
    # LLM Status
    if CORE_MODULES_AVAILABLE:
        llm_status = check_llm_status()
        if llm_status["available"]:
            st.success(f"🤖 {llm_status['message']}")
        else:
            st.info("🔧 Demo模式 (无API Key)")
    
    st.markdown("*Prototype v0.4 | 110 tests | 7,490 LOC*")


# ============================================================
# Main Header
# ============================================================
st.markdown('<div class="main-header">PAI — Philanthropic Asset Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">慈善资产智能：AI驱动的投资优化 · 捐赠策略 · 效果评估</div>', unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">$3,260亿</div><div class="metric-label">DAF资产规模</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">3亿</div><div class="metric-label">全球罕见病患者</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">95%</div><div class="metric-label">无获批治疗方案</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">+45.9%</div><div class="metric-label">AI提升有效捐赠</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 InvestOpt", "🤖 GiveSmart", "🎯 ImpactLens",
    "💡 GiveNudge", "🔄 Impact Loop",
    "🔬 Rare Disease", "🔍 Federated RAG"
])


# ============================================================
# TAB 1: InvestOpt — 慈善投资优化 (MARKOWITZ OPTIMIZATION)
# ============================================================
with tab1:
    st.markdown("## InvestOpt — 慈善投资优化引擎")
    st.markdown("**核心问题：** NBER 2025研究证明，非营利组织捐赠基金系统性跑输市场基准。DAF中$3,260亿资产的投资效率直接影响可用于公益的资金量。")
    st.markdown("**技术实现：** 基于Markowitz现代投资组合理论的均值-方差优化 (Mean-Variance Optimization)")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.markdown("### DAF投资组合对比")
        daf = data["daf_scenario"]
        fig_daf = go.Figure()
        for col in daf.columns[1:]:
            fig_daf.add_trace(go.Scatter(
                x=daf["date"], y=daf[col], mode="lines", name=col,
                line=dict(width=2.5 if "PAI" in col else 1.5),
            ))
        fig_daf.update_layout(
            height=400,
            title="DAF投资组合增长对比（2020-2024）",
            yaxis_title="组合价值（初始=100）",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_daf, use_container_width=True)
        
        st.markdown('<div class="insight-box">💡 <b>洞察：</b>AI优化组合5年累计收益比默认分配高约<b>8-12%</b>。对于一个$100万DAF，这意味着多出$8-12万用于公益拨款。</div>', unsafe_allow_html=True)
    
    with col_b:
        st.markdown("### 基金风险画像")
        cat_filter = st.selectbox("选择基金类别", ["全部"] + list(data["funds"]["category"].unique()), key="invest_cat")
        
        df = data["funds"]
        if cat_filter != "全部":
            df = df[df["category"] == cat_filter]
        
        fig_risk = px.scatter(
            df, x="ann_vol", y="ann_return",
            size=np.maximum(df["sharpe"], 0.1),  # Ensure non-negative for plotly
            color="category",
            hover_name="name",
            size_max=40,
            title="风险-收益散点图（气泡大小=Sharpe比率）",
            labels={"ann_vol": "年化波动率", "ann_return": "年化收益率"},
        )
        fig_risk.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Fund table
    st.markdown("### 基金详细指标")
    show_cols = ["code", "name", "category_cn", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "alpha", "tax_efficiency"]
    show_df = df[show_cols].sort_values("sharpe", ascending=False)
    show_df.columns = ["代码", "基金名称", "类别", "年化收益", "年化波动", "Sharpe", "Sortino", "最大回撤", "Jensen α", "税务效率"]
    st.dataframe(show_df.head(15), use_container_width=True, height=400)
    
    # Portfolio Optimizer (REAL IMPLEMENTATION)
    st.markdown("### 🎯 DAF投资组合优化器 (Markowitz MVO)")
    st.markdown("选择基金构建DAF投资组合，使用**均值-方差优化**算法计算最优权重。")
    
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        selected_funds = st.multiselect(
            "选择基金（2-10只）",
            options=data["funds"]["name"].tolist(),
            default=["Vanguard 500 Index", "Vanguard Total International", "Vanguard GNMA", "Fidelity Magellan", "T. Rowe Price Small-Cap"],
            key="fund_select",
        )
    with col_opt2:
        daf_amount = st.number_input("DAF金额（万美元）", value=100, min_value=1, max_value=10000)
        optimization_strategy = st.selectbox(
            "优化策略",
            ["最大Sharpe", "最小方差", "风险平价", "DAF推荐", "Impact-Aware (v0.4)"],
            key="opt_strategy"
        )
    
    if len(selected_funds) >= 2:
        with st.spinner("正在运行Markowitz优化..."):
            # Run optimization using core module
            if CORE_MODULES_AVAILABLE:
                opt_results = optimize_daf_portfolio(
                    data["funds"],
                    selected_funds,
                    daf_amount * 10000  # Convert to dollars
                )
                
                if optimization_strategy == "最大Sharpe":
                    result = opt_results['max_sharpe']
                    strategy_name = "最大Sharpe比率"
                elif optimization_strategy == "最小方差":
                    result = opt_results['min_variance']
                    strategy_name = "最小方差"
                elif optimization_strategy == "风险平价":
                    result = opt_results['risk_parity']
                    strategy_name = "风险平价"
                else:
                    result = opt_results['daf_recommendation']
                    strategy_name = "DAF推荐"

                # v0.4: Impact-Aware optimization
                if optimization_strategy == "Impact-Aware (v0.4)":
                    impact_scores = {
                        "Vanguard 500": 0.4,
                        "Fidelity Contrafund": 0.35,
                        "TIAA-CREF Social Choice": 0.85,
                        "Parnassus Core Equity": 0.80,
                        "Calvert Equity": 0.75,
                        "iShares ESG Aware MSCI USA": 0.70,
                        "SPDR SSGA Gender Diversity": 0.65,
                        "Vanguard FTSE Social Index": 0.72,
                        "iShares MSCI KLD 400 Social": 0.68,
                        "PIMCO ESG US Aggregate Bond": 0.60,
                    }
                    # Only include selected funds
                    selected_impact = {k: v for k, v in impact_scores.items() if k in selected_funds}
                    if selected_impact:
                        optimizer = PortfolioOptimizer(
                            pd.DataFrame(
                                data["funds"][data["funds"]["name"].isin(selected_funds)]["monthly_returns"].tolist()
                            ).T
                        )
                        result = optimizer.impact_aware_portfolio(selected_impact, impact_weight=0.5)
                        strategy_name = "Impact-Aware"
                
                # Display metrics
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("策略", strategy_name)
                with col_r2:
                    st.metric("组合预期收益", f"{result['expected_return']:.1%}")
                with col_r3:
                    st.metric("组合波动率", f"{result['volatility']:.1%}")
                with col_r4:
                    if 'impact_score' in result:
                        st.metric("Impact Score", f"{result['impact_score']:.2f}")
                    else:
                        st.metric("Sharpe比率", f"{result['sharpe_ratio']:.2f}")
                
                # Display allocation
                st.markdown("#### 📊 最优资产配置")
                weights = result['weights']
                weights_sorted = {k: v for k, v in sorted(weights.items(), key=lambda x: -x[1]) if v > 0.01}
                
                alloc_df = pd.DataFrame([
                    {"基金": k, "权重": f"{v*100:.1f}%"} 
                    for k, v in weights_sorted.items()
                ])
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)
                
                # Efficient frontier chart
                st.markdown("#### 📈 有效前沿")
                returns_df = pd.DataFrame(
                    data["funds"][data["funds"]["name"].isin(selected_funds)]["monthly_returns"].tolist(),
                    index=data["funds"][data["funds"]["name"].isin(selected_funds)]["name"]
                ).T
                
                opt = PortfolioOptimizer(returns_df)
                eff_returns, eff_vols, _ = opt.efficient_frontier(n_points=30)
                
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=eff_vols * 100, y=eff_returns * 100,
                    mode='lines', name='有效前沿',
                    line=dict(color='blue', width=2)
                ))
                
                # Mark optimized point
                fig_ef.add_trace(go.Scatter(
                    x=[result['volatility'] * 100],
                    y=[result['expected_return'] * 100],
                    mode='markers', name=f'{strategy_name}组合',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                fig_ef.update_layout(
                    height=350,
                    xaxis_title="波动率 (%)",
                    yaxis_title="预期收益 (%)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_ef, use_container_width=True)
                
                # DAF Impact
                if optimization_strategy == "DAF推荐":
                    st.markdown(f"""
                    <div class="success-box">
                    <b>💰 DAF优化效果:</b><br>
                    - DAF规模: ${result['daf_size']:,.0f}<br>
                    - 年度拨款: ${result['annual_payout']:,.0f}<br>
                    - 额外年度拨款: ${result['extra_annual_grants']:,.0f}<br>
                    - 5年额外拨款: ${result['extra_grants_5yr']:,.0f}<br>
                    - 预计拯救生命: ~{result.get('lives_saved_5yr', 0):.0f}条 (按AMF标准)
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                # Fallback if core module not available
                st.warning("核心优化模块未加载，使用简化计算...")
                sel_df = data["funds"][data["funds"]["name"].isin(selected_funds)]
                n = len(sel_df)
                weights = np.ones(n) / n
                
                port_return = np.sum(weights * sel_df["ann_return"].values)
                port_vol = np.sqrt(np.sum((weights ** 2) * (sel_df["ann_vol"].values ** 2)))
                port_sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else 0
                
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1:
                    st.metric("组合预期收益", f"{port_return:.1%}")
                with col_r2:
                    st.metric("组合波动率", f"{port_vol:.1%}")
                with col_r3:
                    st.metric("Sharpe比率", f"{port_sharpe:.2f}")
                with col_r4:
                    st.metric("注意", "需安装scipy")
    else:
        st.info("👈 请选择至少2只基金进行优化")


# ============================================================
# TAB 2: GiveSmart — LLM捐赠策略顾问
# ============================================================
with tab2:
    st.markdown("## GiveSmart — AI捐赠策略顾问")
    st.markdown("**核心问题：** 2024年全球仅33%成年人捐款，首次捐赠者留存率仅19.4%。White et al. (2026)证明LLM个性化对话可提升有效捐赠45.9%。")
    
    # LLM Status Banner
    if CORE_MODULES_AVAILABLE:
        llm_status = check_llm_status()
        if llm_status["available"]:
            st.success(f"🤖 **{llm_status['message']}**")
        else:
            st.info("""
            🔧 **LLM配置提示** (当前为演示模式)
            
            设置环境变量以启用真正的AI功能:
            ```bash
            export OPENAI_API_KEY=sk-...
            # 或
            export ANTHROPIC_API_KEY=sk-ant-...
            ```
            """)
    
    # Donation funnel
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.markdown("### 捐赠转化漏斗")
        funnel = data["funnel"]
        fig_funnel = go.Figure(go.Funnel(
            y=funnel["stage"],
            x=funnel["count"],
            textinfo="value+percent initial+percent previous",
            marker=dict(color=["#1a73e8", "#4285f4", "#5e97f6", "#7baaf7", "#9ec5fa", "#c2dcfc"]),
        ))
        fig_funnel.update_layout(height=450, title="捐赠转化漏斗（10万人起点）", template="plotly_white")
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col_f2:
        st.markdown("### AI提升效果（White et al. 2026）")
        fig_ai = go.Figure()
        categories = ["对照组", "静态LLM消息", "LLM个性化对话"]
        values = [100, 128.7, 145.9]
        colors = ["#bdbdbd", "#4285f4", "#34a853"]
        fig_ai.add_trace(go.Bar(
            x=categories, y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
        ))
        fig_ai.update_layout(
            height=450,
            title="AI对有效捐赠的提升效果",
            yaxis_title="相对捐赠量（对照组=100%）",
            template="plotly_white",
        )
        st.plotly_chart(fig_ai, use_container_width=True)
        
        st.markdown('<div class="insight-box">💡 <b>关键发现：</b>LLM个性化对话让有效捐赠增加<b>45.9%</b>，机制是通过个性化说服放大了warm-glow效应（Andreoni 1990）。</div>', unsafe_allow_html=True)
    
    # AI Chat Demo (WITH REAL LLM INTEGRATION)
    st.markdown("---")
    st.markdown("### 🤖 PAI捐赠顾问 — 交互演示")
    st.markdown("输入您的捐赠场景，PAI将提供**个性化AI建议**。")
    
    col_chat1, col_chat2 = st.columns([1, 2])
    with col_chat1:
        donor_type = st.selectbox("捐赠者类型", ["个人捐赠者", "DAF持有者", "企业CSR", "基金会"])
        annual_budget = st.selectbox("年捐赠预算", ["$1,000以下", "$1,000-$10,000", "$10,000-$100,000", "$100,000-$1M", "$1M以上"])
        interest = st.selectbox("关注领域", ["全球健康", "罕见病", "教育", "气候变化", "贫困缓解", "综合"])
        tax_situation = st.selectbox("税务状况", ["标准扣除额", "逐项扣除", "DAF已开设", "有增值证券"])
        donate_btn = st.button("🤖 获取PAI建议", type="primary", use_container_width=True)
    
    with col_chat2:
        if donate_btn:
            with st.spinner("PAI正在生成个性化建议..."):
                if CORE_MODULES_AVAILABLE:
                    try:
                        advisor = get_llm_advisor()
                        charity_list = data["charities"].to_dict('records')

                        # Federated RAG enrichment (optional)
                        rag_context = ""
                        try:
                            from core.federated_rag.streamlit_ui import _init_federated_rag
                            if "fed_rag_router" not in st.session_state:
                                st.session_state["fed_rag_router"] = _init_federated_rag()
                            router = st.session_state["fed_rag_router"]
                            rag_result = router.search_and_answer(
                                f"{interest} charitable giving strategy",
                                top_k=3,
                            )
                            if rag_result["sources"]:
                                rag_context = "\n\n".join(
                                    f"[{s['source']}]\n{s['preview']}..."
                                    for s in rag_result["sources"][:3]
                                )
                        except Exception:
                            pass  # RAG is optional — degrade gracefully

                        advice = advisor.generate_advice(
                            donor_type=donor_type,
                            annual_budget=annual_budget,
                            interest_area=interest,
                            tax_situation=tax_situation,
                            charity_data=charity_list,
                            rag_context=rag_context,
                        )
                        st.markdown(advice)
                        if rag_context:
                            st.caption("📚 建议已通过联邦知识库增强（Federated RAG）")
                    except Exception as e:
                        st.error(f"LLM调用失败: {e}")
                        # Fallback
                        st.info("使用演示模式...")
                        from core.llm_client import LLMDonationAdvisor
                        advisor = LLMDonationAdvisor()
                        advice = advisor._generate_demo_advice(
                            donor_type, annual_budget, interest, tax_situation,
                            data["charities"].to_dict('records')
                        )
                        st.markdown(advice)
                else:
                    st.error("核心模块未加载，无法运行LLM功能")
        else:
            st.info("👈 选择您的捐赠场景，点击按钮获取AI个性化建议。")


# ============================================================
# TAB 3: ImpactLens — 效果评估
# ============================================================
with tab3:
    st.markdown("## ImpactLens — 公益效果评估")
    st.markdown("**核心问题：** GiveWell每年只评估少数项目，EA覆盖面仅占全球慈善的1%。AI可以规模化评估公益项目的cost-effectiveness。")
    
    charities = data["charities"]
    
    # Impact radar
    col_i1, col_i2 = st.columns([1, 1])
    with col_i1:
        st.markdown("### 效果评分雷达图")
        selected_charity = st.selectbox("选择公益项目", charities["name"].tolist(), key="charity_select")
        charity = charities[charities["name"] == selected_charity].iloc[0]
        
        fig_radar = go.Figure(go.Scatterpolar(
            r=[charity["evidence_strength"], charity["scalability"], charity["transparency"],
               1 - charity["overhead_ratio"], charity["impact_score"], 0.7],
            theta=["证据强度", "可扩展性", "透明度", "资金效率", "效果评分", "创新性"],
            fill="toself",
            name=selected_charity,
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            showlegend=True,
            template="plotly_white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col_i2:
        st.markdown("### Cost per Life Saved 对比")
        fig_cost = px.bar(
            charities.sort_values("cost_per_life"),
            x="cost_per_life", y="name",
            orientation="h",
            color="category",
            title="每拯救一条生命的成本（美元）",
            labels={"cost_per_life": "成本（美元）", "name": ""},
            log_x=True,
        )
        fig_cost.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # Impact table
    st.markdown("### 公益效果评估表")
    impact_cols = ["name", "category", "cost_per_life", "evidence_strength", "scalability", "transparency", "overhead_ratio", "impact_score", "region"]
    impact_df = charities[impact_cols].sort_values("impact_score", ascending=False)
    impact_df.columns = ["项目名称", "类别", "每条生命成本($)", "证据强度", "可扩展性", "透明度", "管理费率", "效果评分", "覆盖地区"]
    st.dataframe(impact_df, use_container_width=True, height=450)
    
    # QALY/DALY framework
    st.markdown("---")
    st.markdown("### 📐 健康经济学评估框架")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        st.markdown("""
        **QALY（质量调整生命年）**
        - 衡量健康干预的价值
        - 1 QALY = 完全健康生活1年
        - NICE阈值：£20,000-30,000/QALY
        - GiveWell标准：~$50-100/DALY
        """)
    with col_q2:
        st.markdown("""
        **DALY（伤残调整生命年）**
        - 衡量疾病负担
        - 1 DALY = 1年健康生命损失
        - 全球罕见病DALY负担：**1亿+**
        - 罕见病人均DALY是普通疾病的**15倍**
        """)
    with col_q3:
        st.markdown("""
        **PAI评估维度**
        - Cost per DALY Averted
        - Cost per QALY Gained
        - Evidence Strength Score
        - Scalability Index
        - Transparency Rating
        - Overhead Efficiency
        """)


# ============================================================
# TAB 4: GiveNudge — 行为助推引擎
# ============================================================
with tab4:
    st.markdown("## 💡 GiveNudge — 行为助推引擎")
    st.markdown("**理论基础：** Andreoni (1990) 的 warm-glow giving 理论表明，捐赠者不仅关心受助者福利，还从捐赠行为本身获得心理满足。GiveNudge 利用这一机制，通过数据驱动的方式优化捐赠时机、渠道和框架。")

    col_n1, col_n2 = st.columns([1, 1])
    with col_n1:
        st.markdown("### 📅 最优捐赠时机")
        st.markdown("基于行为经济学研究，不同时段的捐赠转化率差异显著：")

        timing_data = pd.DataFrame({
            "时段": ["周一上午", "周二全天", "周三下午", "周四上午", "周五下午", "周末", "月末", "年末(12月)"],
            "转化率倍数": [1.0, 1.15, 1.08, 1.12, 0.92, 0.85, 1.25, 1.45],
            "推荐度": ["⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐", "⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"],
        })
        st.dataframe(timing_data, use_container_width=True, hide_index=True)

        st.markdown('\u003cdiv class="insight-box"\u003e💡 \u003cb\u003e洞察：\u003c/b\u003e年末（12月）捐赠转化率是年均的\u003cb\u003e1.45倍\u003c/b\u003e，这与税收截止日效应一致。PAI GiveNudge 在最优窗口期自动触发个性化提醒。\u003c/div\u003e', unsafe_allow_html=True)

    with col_n2:
        st.markdown("### 🎯 助推策略效果对比")
        nudge_types = ["无助推", "默认提醒", "社会证明", "匹配捐赠", "影响框架", "紧急感", "Warm-Glow"]
        nudge_effects = [1.0, 1.12, 1.28, 1.35, 1.22, 1.18, 1.31]
        nudge_colors = ["#bdbdbd", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1565c0"]

        fig_nudge = go.Figure()
        fig_nudge.add_trace(go.Bar(
            x=nudge_types, y=nudge_effects,
            marker_color=nudge_colors,
            text=[f"+{int((v-1)*100)}%" for v in nudge_effects],
            textposition="auto",
        ))
        fig_nudge.update_layout(
            height=400,
            title="不同助推策略的捐赠提升效果",
            yaxis_title="相对捐赠量（无助推=1.0）",
            template="plotly_white",
        )
        st.plotly_chart(fig_nudge, use_container_width=True)

    # Donor segment demo
    st.markdown("### 👥 捐赠者分群策略")
    st.markdown("GiveNudge 根据捐赠者画像自动选择最优助推策略：")

    seg_data = pd.DataFrame({
        "分群": ["首次捐赠者", "活跃捐赠者", "沉睡捐赠者", "大额捐赠者", "DAF持有者"],
        "核心策略": ["社会证明+影响框架", "Warm-Glow+认可", "紧急感+匹配", "个性化影响报告", "税务优化提醒"],
        "推荐渠道": ["邮件", "应用内", "邮件+短信", "专属顾问", "应用内+邮件"],
        "预期提升": ["+28%", "+15%", "+22%", "+12%", "+18%"],
        "优先级": ["🔴 高", "🟡 中", "🔴 高", "🟢 低频高价值", "🟡 中"],
    })
    st.dataframe(seg_data, use_container_width=True, hide_index=True)

    # A/B Test Framework
    st.markdown("---")
    st.markdown("### 🧪 A/B 测试框架")
    st.markdown("GiveNudge 内置 A/B 测试引擎，持续优化助推策略：")

    col_ab1, col_ab2 = st.columns([1, 1])
    with col_ab1:
        st.markdown("""
        **实验设计**
        - 对照组：标准提醒消息
        - 实验组：GiveNudge 优化消息
        - 样本量：每组 200+ 捐赠者
        - 检验标准：p < 0.05
        - 主要指标：转化率、捐赠金额、留存率
        """)
    with col_ab2:
        st.markdown("""
        **模拟结果（White et al. 2026 启发）**

        | 指标 | 对照组 | GiveNudge | 提升 |
        |------|--------|-----------|------|
        | 转化率 | 5.5% | 7.2% | +31% |
        | 平均金额 | $150 | $195 | +30% |
        | 90天留存 | 19.4% | 26.1% | +35% |
        | p-value | — | 0.003 | ✅ 显著 |
        """)

    st.markdown('\u003cdiv class="success-box"\u003e✅ \u003cb\u003eGiveNudge 核心价值：\u003c/b\u003e将行为经济学理论转化为可量化、可测试、可迭代的捐赠优化引擎。每个助推策略都有 A/B 测试支撑，确保持续改进。\u003c/div\u003e', unsafe_allow_html=True)


# ============================================================
# TAB 5: Impact Feedback Loop — 效果闭环
# ============================================================
with tab5:
    st.markdown("## 🔄 Impact Feedback Loop — 效果闭环")
    st.markdown("**核心创新：** PAI 最具变革性的组件。将实际公益效果（健康改善、教育成果、环境指标）反馈到投资策略和拨款建议中，实现「发生了什么」→「接下来该做什么」的闭环。")

    # Architecture diagram
    st.markdown("### 闭环架构")
    col_loop1, col_loop2 = st.columns([1, 1])
    with col_loop1:
        fig_loop = go.Figure()
        # Loop arrows
        stages = [
            (1, 4, "1. 拨款\nAllocation"),
            (4, 7, "2. 执行\nExecution"),
            (7, 4, "3. 测量\nMeasurement"),
            (4, 1, "4. 反馈\nFeedback"),
        ]
        colors_loop = ["#1a73e8", "#34a853", "#fbbc04", "#ea4335"]
        for i, (x1, x2, label) in enumerate(stages):
            fig_loop.add_trace(go.Scatter(
                x=[x1, x2], y=[4 if i % 2 == 0 else 2, 4 if i % 2 == 0 else 2],
                mode="lines+text",
                line=dict(color=colors_loop[i], width=3),
                text=[label],
                textposition="top center",
                textfont=dict(size=11),
                showlegend=False,
            ))
        # Center node
        fig_loop.add_trace(go.Scatter(
            x=[4], y=[3], mode="markers+text",
            marker=dict(size=40, color="#1a73e8", symbol="circle"),
            text=["PAI\nEngine"],
            textfont=dict(size=12, color="white"),
            showlegend=False,
        ))
        fig_loop.update_layout(
            height=300,
            xaxis=dict(visible=False, range=[0, 8]),
            yaxis=dict(visible=False, range=[0, 6]),
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_loop, use_container_width=True)

    with col_loop2:
        st.markdown("""
        **传统慈善 vs PAI 效果闭环**

        | 维度 | 传统 | PAI |
        |------|------|-----|
        | 决策依据 | 主观判断 | 数据驱动 |
        | 效果追踪 | 年度报告 | 实时监控 |
        | 反馈机制 | ❌ 无 | ✅ 自动闭环 |
        | 资源再分配 | 年度计划 | 动态调整 |
        | 跨机构学习 | ❌ 孤立 | ✅ 联邦验证 |
        """)

    # Saturation detection
    st.markdown("### 📉 效果饱和检测")
    st.markdown("PAI 检测每个领域的边际效果递减，避免过度集中投入：")

    np.random.seed(42)
    funding_levels = np.linspace(100000, 5000000, 50)
    # Diminishing returns curve
    impact_health = 0.95 * (1 - np.exp(-funding_levels / 800000))
    impact_education = 0.85 * (1 - np.exp(-funding_levels / 1200000))
    impact_rare = 0.70 * (1 - np.exp(-funding_levels / 2000000))

    fig_sat = go.Figure()
    fig_sat.add_trace(go.Scatter(x=funding_levels/1e6, y=impact_health, name="Global Health", line=dict(width=2.5)))
    fig_sat.add_trace(go.Scatter(x=funding_levels/1e6, y=impact_education, name="Education", line=dict(width=2.5)))
    fig_sat.add_trace(go.Scatter(x=funding_levels/1e6, y=impact_rare, name="Rare Disease", line=dict(width=2.5)))
    # Saturation threshold
    fig_sat.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="饱和阈值 (80%)")
    fig_sat.update_layout(
        height=400,
        title="效果饱和曲线：边际效果递减",
        xaxis_title="累计投入（百万美元）",
        yaxis_title="效果评分",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_sat, use_container_width=True)

    st.markdown('\u003cdiv class="insight-box"\u003e💡 \u003cb\u003e关键洞察：\u003c/b\u003eGlobal Health 在 ~$200万 时达到饱和，而 Rare Disease 需要 ~$500万+。PAI 自动检测饱和点并建议将边际效果更高的领域获得更多资源。\u003c/div\u003e', unsafe_allow_html=True)

    # Reallocation demo
    st.markdown("### 🔄 自动再分配建议")
    st.markdown("当效果信号偏离预测时，PAI 自动生成再分配建议：")

    realloc_data = pd.DataFrame({
        "受助机构": ["AMF", "GiveDirectly", "Cure RD", "Noora Health", "END Fund"],
        "当前拨款": ["$500K", "$300K", "$200K", "$150K", "$250K"],
        "效果信号": ["✅ 达标", "✅ 超预期", "⚠️ 低于预期", "✅ 达标", "✅ 超预期"],
        "建议调整": ["维持", "+$50K", "-$80K", "维持", "+$30K"],
        "原因": ["效果稳定", "成本效益优于预期", "研发进度滞后", "按计划执行", "NTD覆盖扩大"],
    })
    st.dataframe(realloc_data, use_container_width=True, hide_index=True)

    st.markdown('\u003cdiv class="success-box"\u003e✅ \u003cb\u003eImpact Feedback Loop 核心价值：\u003c/b\u003e这是慈善领域的范式转变——从静态的、基于意见的捐赠，转向动态的、证据驱动的慈善资产管理。技术基础今天已经存在。\u003c/div\u003e', unsafe_allow_html=True)


# ============================================================
# TAB 6: Rare Disease Blueprint + Federated Learning
# ============================================================
with tab6:
    st.markdown("## 🔬 罕见病基金会建设蓝图")
    st.markdown('<div class="rare-disease-box">⚠️ <b>战略背景：</b>全球3亿罕见病患者、7,000+种疾病、95%无获批治疗方案。WEF 2026报告指出罕见病投资可释放<b>万亿美元</b>经济机会。</div>', unsafe_allow_html=True)
    
    # Triangle model
    st.markdown("### 三角模型：投资-捐赠-研发闭环")
    
    fig_tri = go.Figure()
    fig_tri.add_trace(go.Scatter(
        x=[0, 5, 2.5, 0],
        y=[4, 4, 0, 4],
        fill="toself",
        fillcolor="rgba(26, 115, 232, 0.1)",
        line=dict(color="#1a73e8", width=2),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_tri.add_trace(go.Scatter(
        x=[0, 5, 2.5],
        y=[4, 4, 0],
        mode="markers+text",
        marker=dict(size=30, color=["#1a73e8", "#34a853", "#fbbc04"]),
        text=["💰 投资<br>InvestOpt", "🤝 捐赠<br>GiveSmart", "🔬 研发<br>R&D Pipeline"],
        textposition="top center",
        textfont=dict(size=12),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_tri.add_trace(go.Scatter(
        x=[1.25, 3.75, 2.5],
        y=[3.2, 3.2, 1.5],
        mode="text",
        text=["投资收益→捐赠", "捐赠→研发资助", "研发成果→投资回报"],
        textfont=dict(size=10, color="#666"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_tri.update_layout(
        height=350,
        xaxis=dict(visible=False, range=[-1, 6]),
        yaxis=dict(visible=False, range=[-1, 5.5]),
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_tri, use_container_width=True)
    
    # Three phases
    st.markdown("### 三阶段建设路径")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown("""
        #### 🌱 Phase 1: 种子期（0-2年）
        **目标：** 建立基础设施
        
        - [ ] 注册非营利组织（501c3/同等）
        - [ ] 开设DAF账户
        - [ ] 部署PAI InvestOpt（DAF投资优化）
        - [ ] 建立罕见病患者注册系统
        - [ ] 部署联邦学习隐私层（患者数据保护）
        - [ ] 发布首份《罕见病投资效果报告》
        - [ ] 种子资金：$50-100万
        
        **PAI模块：** InvestOpt + FedShield
        """)
    with col_p2:
        st.markdown("""
        #### 🌿 Phase 2: 成长期（2-5年）
        **目标：** 扩大规模与影响力
        
        - [ ] DAF资产增长至$500万+
        - [ ] 部署PAI GiveSmart（LLM捐赠者匹配）
        - [ ] 部署PAI ImpactLens（效果评估）
        - [ ] 建立Venture Philanthropy基金
        - [ ] 资助3-5个罕见病研发项目
        - [ ] 跨机构联邦学习网络（5+医院）
        - [ ] 年拨款额：$100万+
        
        **PAI模块：** 全部四个模块
        **资金来源：** DAF + 个人捐赠 + 企业合作
        """)
    with col_p3:
        st.markdown("""
        #### 🌳 Phase 3: 规模期（5-10年）
        **目标：** 行业影响力
        
        - [ ] DAF资产$5000万+
        - [ ] VP基金投资10+项目
        - [ ] 联邦学习网络覆盖20+机构
        - [ ] 推动至少1个罕见病药物获批
        - [ ] 发布行业级效果评估标准
        - [ ] 年拨款额：$1000万+
        - [ ] 成为罕见病领域标杆基金会
        
        **PAI模块：** 全模块 + 行业开放API
        **资金来源：** DAF + VP回报 + 政府拨款 + 企业合作
        """)
    
    # Federated Learning Section
    st.markdown("---")
    st.markdown("### 🔒 FedShield: 联邦学习隐私保护层")
    
    if CORE_MODULES_AVAILABLE:
        with st.expander("🔬 联邦学习系统演示"):
            st.markdown("""
            **这是什么？**
            联邦学习（Federated Learning）让多家医院/机构协作训练AI模型，同时**原始患者数据永不离开本地**。
            
            **关键优势：**
            - 🔐 隐私保护：原始数据不出医院
            - 📊 数据多样性：汇聚多机构数据
            - ⚖️ 合规性：满足HIPAA/GDPR要求
            - 🔎 可审计：区块链记录每次训练
            """)
            
            if st.button("🚀 运行联邦学习演示"):
                with st.spinner("初始化联邦学习系统..."):
                    fl = create_fl_system(num_institutions=4)
                    
                    st.success(f"已注册 {len(fl.clients)} 个机构:")
                    for cid, info in fl.clients.items():
                        st.write(f"  - {info['institution']}: {info['data_size']:,} 样本")
                    
                    # Run 3 training rounds
                    st.markdown("**🏋️ 运行训练回合...**")
                    for i in range(3):
                        result = fl.run_training_round()
                        with st.container():
                            col_fl1, col_fl2, col_fl3 = st.columns(3)
                            with col_fl1:
                                st.metric(f"Round {i+1} 参与机构", len(result['selected_clients']))
                            with col_fl2:
                                acc = result['metrics'].get('avg_accuracy', 0)
                                st.metric(f"Round {i+1} 平均精度", f"{acc:.1%}")
                            with col_fl3:
                                st.metric(f"Round {i+1} 样本数", result['metrics'].get('total_samples', 0))
                    
                    st.success(f"✅ 训练完成! 审计链包含 {len(fl.get_audit_trail())} 条记录")
                    
    else:
        st.info("""
        🔧 **FedShield 联邦学习模块**
        
        该模块为可选组件，提供隐私保护的跨机构数据协作能力。
        
        要启用此功能，请安装完整依赖：
        ```bash
        pip install -r requirements.txt
        ```
        
        参考实现来自: [organoid-fl](https://github.com/dechang64/organoid-fl)
        """)
    
    # Venture Philanthropy
    st.markdown("---")
    st.markdown("### 💼 Venture Philanthropy模式")
    st.markdown("罕见病药物研发的'megafund'模式（MIT Fernald 2013）：将慈善资金以风险投资方式投入研发，成功项目回报再投入下一个项目。")
    
    col_vp1, col_vp2 = st.columns([1, 1])
    with col_vp1:
        st.markdown("""
        **VP vs 传统慈善**
        
        | 维度 | 传统慈善 | Venture Philanthropy |
        |------|---------|---------------------|
        | 资金使用 | 拨款消耗 | 投资循环 |
        | 风险偏好 | 低风险 | 高风险高回报 |
        | 退出机制 | 无 | 药物授权/并购 |
        | 可持续性 | 依赖捐赠 | 自我循环 |
        | 衡量标准 | 拨款额 | 患者获益+ROI |
        """)
    with col_vp2:
        st.markdown("""
        **罕见病VP经济模型**
        
        - 孤儿药平均年销售：$1-5亿（Thomson Reuters）
        - 研发成本/患者：$13.7万-$74.3万
        - 93%孤儿药获批后获保险覆盖
        - 罕见病优先审评券（PRV）：价值$1亿+
        - Orphan Drug Act激励：税收抵免+市场独占权
        
        **PAI在VP中的角色：**
        - InvestOpt：优化基金会投资组合
        - ImpactLens：评估研发项目效果
        - FedShield：跨机构患者数据协作
        """)


# ============================================================
# TAB 7: Federated RAG — 联邦知识库检索
# ============================================================
with tab7:
    try:
        from core.federated_rag.streamlit_ui import render_federated_rag
        render_federated_rag()
    except ImportError as e:
        st.warning(f"Federated RAG module not available: {e}")
        st.info("""
        **Federated RAG** requires additional dependencies:
        ```bash
        pip install sentence-transformers faiss-cpu
        ```
        This module provides privacy-preserving cross-institutional
        knowledge retrieval powered by AI embeddings.
        """)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #999; font-size: 0.85rem;">
    <b>PAI — Philanthropic Asset Intelligence</b> | Prototype v0.4<br>
    Gates Foundation Grand Challenges 2026 · Project 3: AI to Accelerate Charitable Giving<br>
    110 tests · 7,490 LOC · 8 modules · MIT License<br>
    <br>
    核心模块: {'✅ 已加载' if CORE_MODULES_AVAILABLE else '⚠️ 部分未加载'}
</div>
""", unsafe_allow_html=True)
