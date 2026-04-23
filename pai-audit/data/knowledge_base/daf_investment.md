# DAF Investment Optimization and Portfolio Management

## The DAF Investment Problem

Donor-Advised Funds hold over $326 billion in assets, but research shows nonprofit endowments and DAFs significantly underperform market benchmarks (Lo et al., 2025, NBER). This investment underperformance represents billions in lost potential charitable impact. The key challenge: DAFs have unique constraints that standard portfolio theory doesn't address.

## DAF-Specific Investment Constraints

Unlike individual investors, DAFs face unique constraints:
1. **Giving Timeline**: Funds must be eventually granted to charities, creating a known or estimated distribution schedule
2. **Liquidity Needs**: Grant recommendations require periodic liquidity
3. **Fiduciary Duty**: DAF sponsors must act in donors' best interests
4. **Tax Optimization**: Investment returns grow tax-free inside the DAF
5. **Mission Alignment**: Some donors prefer ESG or impact-aligned investments

## Markowitz Mean-Variance Optimization for DAFs

PAI's InvestOpt applies Modern Portfolio Theory (MPT) with DAF-specific modifications:
- **Time-Horizon Adjustment**: Portfolio allocation shifts based on planned giving timeline
- **Liquidity Buffer**: Reserve 5-15% in liquid assets for near-term grants
- **Impact Overlay**: Optional ESG screening without significantly reducing expected returns
- **Tax-Free Growth**: Since DAF investments grow tax-free, after-tax optimization is simplified

## Efficient Frontier for Charitable Portfolios

The efficient frontier represents the set of portfolios offering maximum expected return for each level of risk. For DAF portfolios:
- Short-term DAFs (grants within 1-2 years): Conservative allocation (60% bonds, 30% equities, 10% cash)
- Medium-term DAFs (3-7 years): Balanced allocation (40% bonds, 55% equities, 5% alternatives)
- Long-term DAFs (7+ years): Growth allocation (20% bonds, 70% equities, 10% alternatives)

## Risk Metrics for DAF Portfolios

PAI tracks multiple risk metrics beyond standard deviation:
1. **Sharpe Ratio**: Return per unit of total risk (target: >0.5)
2. **Sortino Ratio**: Return per unit of downside risk (target: >0.8)
3. **Maximum Drawdown**: Largest peak-to-trough decline (target: <20%)
4. **Calmar Ratio**: Return per unit of maximum drawdown (target: >0.5)
5. **Value-at-Risk (VaR)**: Maximum expected loss at 95% confidence

## Fund Selection Methodology

PAI evaluates 63 mutual funds across 7 risk metrics:
- Expense ratio (target: <0.50%)
- 3-year and 5-year annualized returns
- Standard deviation and beta
- Sharpe and Sortino ratios
- Maximum drawdown
- Correlation with other portfolio holdings

## Impact of Investment Optimization on Charitable Giving

A DAF with $1M that improves investment returns by 2% annually generates an additional $20,000 per year in grant-making capacity. Over 10 years, this compounds to approximately $240,000 in additional charitable grants — equivalent to funding 48 additional GiveDirectly recipients or 48,000 additional malaria nets through AMF.

## FundFL Integration

PAI's InvestOpt is built on the FundFL platform, which provides:
- Real-time fund data for 63 mutual funds
- 7 risk metrics calculated from historical returns
- Interactive efficient frontier visualization
- Portfolio comparison and backtesting tools
- Open-source code available at github.com/dechang64/FundFL
