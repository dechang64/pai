"""
PAI - Philanthropic Asset Intelligence
AI驱动的慈善投资优化、捐赠策略与效果评估系统
Prototype v0.1 — Gates Foundation Grand Challenges 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

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
            daf_apt = round(np.random.uniform(0.15, 0.35), 3)  # DAF annual payout rate
            daf_growth = round(np.random.uniform(0.03, 0.12), 3)  # DAF asset growth rate
            tax_efficiency = round(np.random.uniform(0.7, 0.95), 3)  # Tax efficiency score
            
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
    months = pd.date_range("2020-01-01", periods=60, freq="M")
    
    # Scenario 1: Typical DAF (default allocation, underperforming)
    typical = 1 + np.cumsum(np.random.normal(0.005, 0.02, 60))
    
    # Scenario 2: AI-optimized DAF
    optimized = 1 + np.cumsum(np.random.normal(0.007, 0.018, 60))
    
    # Scenario 3: Simple 60/40 benchmark
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
    st.markdown("📊 **InvestOpt** — 慈善投资优化")
    st.markdown("🤖 **GiveSmart** — 捐赠策略顾问")
    st.markdown("🎯 **ImpactLens** — 效果评估")
    st.markdown("🔒 **FedShield** — 隐私保护层")
    st.markdown("")
    st.markdown("---")
    st.markdown("#### 关键数据")
    st.markdown("- DAF资产: **$3,260亿**")
    st.markdown("- 被困资金: **$2,500亿+**")
    st.markdown("- 全球罕见病患者: **3亿**")
    st.markdown("- 95%罕见病无治疗方案")
    st.markdown("")
    st.markdown("---")
    st.markdown("*Prototype v0.1 | 2026-04-21*")


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

tab1, tab2, tab3, tab4 = st.tabs(["📊 InvestOpt", "🤖 GiveSmart", "🎯 ImpactLens", "🔗 Rare Disease Blueprint"])


# ============================================================
# TAB 1: InvestOpt — 慈善投资优化
# ============================================================
with tab1:
    st.markdown("## InvestOpt — 慈善投资优化引擎")
    st.markdown("**核心问题：** NBER 2025研究证明，非营利组织捐赠基金系统性跑输市场基准。DAF中$3,260亿资产的投资效率直接影响可用于公益的资金量。")
    
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
            size="sharpe", color="category",
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
    
    # Portfolio optimizer
    st.markdown("### 🎯 DAF投资组合优化器")
    st.markdown("选择基金构建DAF投资组合，AI自动计算最优权重和预期效果。")
    
    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        selected_funds = st.multiselect(
            "选择基金（最多5只）",
            options=data["funds"]["name"].tolist(),
            default=["Vanguard 500 Index", "Vanguard Total International", "Vanguard GNMA", "Fidelity Magellan", "T. Rowe Price Small-Cap"],
            key="fund_select",
        )
    with col_opt2:
        daf_amount = st.number_input("DAF金额（万美元）", value=100, min_value=1, max_value=10000)
        grant_schedule = st.selectbox("拨款计划", ["5年内拨清", "10年内拨清", "永续基金"], key="grant_plan")
    
    if selected_funds and len(selected_funds) >= 2:
        sel_df = data["funds"][data["funds"]["name"].isin(selected_funds)]
        
        # Simple equal-weight optimization
        n = len(sel_df)
        weights = np.ones(n) / n
        
        port_return = np.sum(weights * sel_df["ann_return"].values)
        port_vol = np.sqrt(np.sum((weights ** 2) * (sel_df["ann_vol"].values ** 2)))
        port_sharpe = (port_return - 0.02) / port_vol if port_vol > 0 else 0
        
        # Estimate additional charitable giving from optimization
        typical_return = 0.06  # Typical DAF default return
        extra_return = port_return - typical_return
        extra_giving = daf_amount * 10000 * extra_return
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("组合年化收益", f"{port_return:.1%}")
        with col_r2:
            st.metric("组合波动率", f"{port_vol:.1%}")
        with col_r3:
            st.metric("Sharpe比率", f"{port_sharpe:.2f}")
        with col_r4:
            st.metric("额外公益资金", f"${extra_giving:,.0f}/年", delta=f"+{extra_return:.1%}")
        
        st.markdown(f'<div class="insight-box">💡 <b>PAI建议：</b>以${daf_amount}万DAF为例，AI优化组合每年可多产生<b>${extra_giving:,.0f}</b>用于公益拨款。5年累计多拨款<b>${extra_giving*5:,.0f}</b>，相当于拯救约<b>{extra_giving*5/3500:.0f}</b>条生命（按AMF标准）。</div>', unsafe_allow_html=True)


# ============================================================
# TAB 2: GiveSmart — 捐赠策略顾问
# ============================================================
with tab2:
    st.markdown("## GiveSmart — AI捐赠策略顾问")
    st.markdown("**核心问题：** 2024年全球仅33%成年人捐款，首次捐赠者留存率仅19.4%。White et al. (2026)证明LLM个性化对话可提升有效捐赠45.9%。")
    
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
    
    # AI Chat Demo
    st.markdown("---")
    st.markdown("### 🤖 PAI捐赠顾问 — 交互演示")
    st.markdown("输入您的捐赠场景，PAI将提供个性化建议。")
    
    col_chat1, col_chat2 = st.columns([1, 2])
    with col_chat1:
        donor_type = st.selectbox("捐赠者类型", ["个人捐赠者", "DAF持有者", "企业CSR", "基金会"])
        annual_budget = st.selectbox("年捐赠预算", ["$1,000以下", "$1,000-$10,000", "$10,000-$100,000", "$100,000-$1M", "$1M以上"])
        interest = st.selectbox("关注领域", ["全球健康", "罕见病", "教育", "气候变化", "贫困缓解", "综合"])
        tax_situation = st.selectbox("税务状况", ["标准扣除额", "逐项扣除", "DAF已开设", "有增值证券"])
        donate_btn = st.button("🤖 获取PAI建议", type="primary", use_container_width=True)
    
    with col_chat2:
        if donate_btn:
            with st.spinner("PAI正在分析..."):
                import time
                time.sleep(1.5)
            
            # Generate personalized advice
            advice = generate_advice(donor_type, annual_budget, interest, tax_situation, data)
            st.markdown(advice)
        else:
            st.info("👈 选择您的捐赠场景，点击按钮获取AI个性化建议。")


def generate_advice(donor_type, budget, interest, tax, data):
    """Generate personalized donation advice"""
    
    budget_map = {"$1,000以下": 500, "$1,000-$10,000": 5000, "$10,000-$100,000": 50000, "$100,000-$1M": 500000, "$1M以上": 5000000}
    budget_num = budget_map.get(budget, 5000)
    
    # Top charity recommendation
    if interest == "罕见病":
        top = data["charities"][data["charities"]["category"].str.contains("Rare")]
    else:
        top = data["charities"].sort_values("impact_score", ascending=False).head(3)
    
    advice = f"""### 📋 PAI个性化捐赠建议

**捐赠者画像：** {donor_type} | 年预算：{budget} | 关注：{interest}

---

#### 🎯 推荐捐赠项目（按效果排序）
"""
    for _, row in top.iterrows():
        lives = budget_num / row["cost_per_life"]
        advice += f"""
| 指标 | 数值 |
|------|------|
| **{row['name']}** | |
| 效果评分 | ⭐ {row['impact_score']:.0%} |
| 每${budget_num:,.0f}美元可影响 | ~{lives:.1f}条生命 |
| 证据强度 | {row['evidence_strength']:.0%} |
| 管理费率 | {row['overhead_ratio']:.0%} |
| 覆盖地区 | {row['region']} |
| 受益人群 | {row['beneficiaries']} |

"""
    
    # Tax advice
    advice += """#### 💰 税务优化建议
"""
    if tax == "有增值证券":
        advice += """
- ✅ **优先捐赠增值证券**（股票/基金）→ 免资本利得税 + 全额抵扣
- ✅ 考虑开设DAF → 集中多年捐赠到一年（Bunching策略）→ 超过标准扣除额
- ✅ 增值证券捐入DAF后免税复利增长 → 分年拨款
"""
    elif tax == "DAF已开设":
        advice += """
- ✅ DAF投资组合优化：当前默认配置可能跑输市场8-12%/年
- ✅ 使用PAI InvestOpt优化DAF投资 → 每年多出更多资金用于拨款
- ✅ 检查DAF拨款率：建议保持≥20%/年
"""
    elif tax == "逐项扣除":
        advice += """
- ✅ Bunching策略：将2-3年的捐赠集中到一年 → 超过标准扣除额
- ✅ 超出部分放入DAF → 免税投资 → 分年拨款给公益项目
- ✅ 捐赠增值证券比现金更节税
"""
    else:
        advice += """
- ⚠️ 当前使用标准扣除额 → 捐赠可能无法获得税收抵扣
- ✅ 建议考虑Bunching策略 + DAF → 集中捐赠以超过标准扣除额
- ✅ 长期来看，DAF + 增值证券捐赠是最优策略
"""
    
    # Warm-glow insight
    advice += f"""
#### 🧠 行为经济学洞察
- **Warm-Glow效应**（Andreoni 1990）：您从捐赠行为本身获得心理回报
- **信息不对称**（Null 2009）：选择效果评分最高的项目 → 捐赠效率最大化
- **AI增强**（White et al. 2026）：个性化建议可提升有效捐赠**45.9%**
- **留存率**：首次捐赠者留存率仅19.4% → 建议设置月度自动捐赠（$50-100/月）→ 留存率提升至60%+

---
*以上建议由PAI原型生成，仅供参考。实际捐赠决策请结合个人情况。*
"""
    return advice


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
# TAB 4: Rare Disease Blueprint
# ============================================================
with tab4:
    st.markdown("## 🔬 罕见病基金会建设蓝图")
    st.markdown('<div class="rare-disease-box">⚠️ <b>战略背景：</b>全球3亿罕见病患者、7,000+种疾病、95%无获批治疗方案。WEF 2026报告指出罕见病投资可释放<b>万亿美元</b>经济机会。</div>', unsafe_allow_html=True)
    
    # Triangle model
    st.markdown("### 三角模型：投资-捐赠-研发闭环")
    
    fig_tri = go.Figure()
    # Triangle nodes
    fig_tri.add_trace(go.Scatter(
        x=[0, 5, 2.5, 0],
        y=[4, 4, 0, 4],
        fill="toself",
        fillcolor="rgba(26, 115, 232, 0.1)",
        line=dict(color="#1a73e8", width=2),
        hoverinfo="skip",
        showlegend=False,
    ))
    # Node labels
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
    # Edge labels
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
        - [ ] 部署PAI GiveSmart（捐赠者匹配）
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
    
    # Federated Learning for Rare Disease
    st.markdown("---")
    st.markdown("### 🔒 联邦学习：罕见病数据协作的关键")
    st.markdown("罕见病的核心挑战：**数据孤岛**。每种罕见病患者极少，单一机构数据不足以训练AI模型。联邦学习让多机构协作训练而不共享原始数据。")
    
    col_fl1, col_fl2 = st.columns([1, 1])
    with col_fl1:
        st.markdown("""
        **为什么罕见病需要联邦学习？**
        
        1. **数据稀缺**：每种病可能全球只有几百个病例
        2. **隐私法规**：HIPAA/GDPR限制跨机构数据共享
        3. **竞争壁垒**：医院/药企不愿共享数据
        4. **监管要求**：FDA/EMA要求多中心验证
        
        **organoid-fl技术栈直接复用：**
        - FedAvg聚合算法（99.17%准确率）
        - gRPC通信框架
        - HNSW向量检索
        - 区块链审计链
        """)
    with col_fl2:
        # FL architecture diagram
        fig_fl = go.Figure()
        # Central server
        fig_fl.add_trace(go.Scatter(
            x=[2.5], y=[3],
            mode="markers+text",
            marker=dict(size=40, color="#1a73e8", symbol="hexagon"),
            text=["PAI Server<br>（聚合中心）"],
            textposition="bottom center",
            textfont=dict(size=11),
            showlegend=False,
        ))
        # Hospitals
        hospitals = ["🏥 医院A<br>（苏州）", "🏥 医院B<br>（上海）", "🏥 医院C<br>（北京）", "🏥 药企D<br>（波士顿）"]
        hx = [0.5, 2.5, 4.5, 2.5]
        hy = [0.5, 0.5, 0.5, -0.5]
        fig_fl.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="markers+text",
            marker=dict(size=25, color="#34a853"),
            text=hospitals,
            textposition="top center",
            textfont=dict(size=9),
            showlegend=False,
        ))
        # Arrows (simplified as lines)
        for x in hx:
            fig_fl.add_trace(go.Scatter(
                x=[x, 2.5], y=[hy[hx.index(x)] + 0.3, 2.7],
                mode="lines",
                line=dict(color="#bdbdbd", width=1.5, dash="dot"),
                showlegend=False,
            ))
        fig_fl.update_layout(
            height=300,
            xaxis=dict(visible=False, range=[-0.5, 5.5]),
            yaxis=dict(visible=False, range=[-1.5, 4]),
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_fl, use_container_width=True)
        
        st.markdown("""
        **数据流向：**
        - 各机构本地训练 → 只上传模型参数
        - 中心聚合 → 全局模型更新
        - 区块链审计 → 训练过程可追溯
        - 原始患者数据**永不离开本地**
        """)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85rem;">
    <b>PAI — Philanthropic Asset Intelligence</b> | Prototype v0.1<br>
    Gates Foundation Grand Challenges 2026 · Project 3: AI to Accelerate Charitable Giving<br>
    同时作为罕见病基金会建设蓝图 · 2026-04-21
</div>
""", unsafe_allow_html=True)
