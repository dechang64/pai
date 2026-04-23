"""
Portfolio Optimization Module for PAI
Implements Mean-Variance Optimization (Markowitz) with SciPy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, List
import warnings


class PortfolioOptimizer:
    """
    Mean-Variance Portfolio Optimization based on Markowitz Modern Portfolio Theory.
    
    Features:
    - Minimum variance portfolio
    - Maximum Sharpe ratio portfolio
    - Efficient frontier calculation
    - Risk parity portfolio
    - Black-Litterman model support (planned)
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize optimizer with return data.
        
        Args:
            returns: DataFrame of asset returns (columns = assets, rows = periods)
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)
        
        # Pre-compute statistics
        self.mean_returns = returns.mean() * 12  # Annualized
        self.cov_matrix = returns.cov() * 12     # Annualized
        
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return."""
        return np.dot(weights, self.mean_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        if vol == 0:
            return 0
        return (ret - self.risk_free_rate) / vol
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe for minimization."""
        return -self.portfolio_sharpe(weights)
    
    def portfolio_variance(self, weights: np.ndarray) -> float:
        """Calculate portfolio variance."""
        return np.dot(weights.T, np.dot(self.cov_matrix, weights))
    
    def min_variance_portfolio(self) -> Dict:
        """
        Find minimum variance portfolio.
        
        Returns:
            Dict with weights, return, volatility, sharpe
        """
        # Objective: minimize variance
        def objective(w):
            return self.portfolio_variance(w)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: each weight between 0 and 1 (long only)
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Initial guess: equal weights
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
        
        weights = result.x
        return {
            'weights': dict(zip(self.asset_names, weights.round(4))),
            'expected_return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe_ratio': self.portfolio_sharpe(weights),
            'success': result.success
        }
    
    def max_sharpe_portfolio(self) -> Dict:
        """
        Find maximum Sharpe ratio portfolio.
        
        Returns:
            Dict with weights, return, volatility, sharpe
        """
        # Objective: minimize negative Sharpe
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            self.negative_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
        
        weights = result.x
        return {
            'weights': dict(zip(self.asset_names, weights.round(4))),
            'expected_return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe_ratio': self.portfolio_sharpe(weights),
            'success': result.success
        }
    
    def efficient_frontier(
        self, 
        n_points: int = 50,
        target_returns: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate efficient frontier.
        
        Args:
            n_points: Number of points on frontier
            target_returns: Optional specific target returns
            
        Returns:
            Tuple of (returns, volatilities, weights_list)
        """
        if target_returns is None:
            # Generate target returns from min to max possible
            min_ret = self.mean_returns.min()
            max_ret = self.mean_returns.max()
            target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontiers_returns = []
        frontiers_vols = []
        frontiers_weights = []
        
        for target in target_returns:
            # Objective: minimize variance
            def objective(w):
                return self.portfolio_variance(w)
            
            # Constraints: sum to 1, achieve target return
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, t=target: self.portfolio_return(w) - t}
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            w0 = np.ones(self.n_assets) / self.n_assets
            
            try:
                result = minimize(
                    objective,
                    w0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500}
                )
                
                if result.success:
                    frontiers_returns.append(target)
                    frontiers_vols.append(self.portfolio_volatility(result.x))
                    frontiers_weights.append(result.x)
            except Exception:
                continue
        
        return np.array(frontiers_returns), np.array(frontiers_vols), frontiers_weights
    
    def risk_parity_portfolio(self) -> Dict:
        """
        Calculate risk parity portfolio (equal risk contribution).
        
        Returns:
            Dict with weights, return, volatility, sharpe
        """
        def risk_contribution(w):
            """Calculate risk contribution of each asset."""
            vol = self.portfolio_volatility(w)
            marginal_risk = np.dot(self.cov_matrix, w)
            risk_contrib = w * marginal_risk / vol if vol > 0 else np.zeros_like(w)
            return risk_contrib
        
        def objective(w):
            """Minimize deviation from equal risk contribution."""
            rc = risk_contribution(w)
            target_rc = np.ones(self.n_assets) * (self.portfolio_volatility(w) / self.n_assets)
            return np.sum((rc - target_rc) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0.01, 1) for _ in range(self.n_assets))  # At least 1% each
        w0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        weights = result.x
        return {
            'weights': dict(zip(self.asset_names, weights.round(4))),
            'expected_return': self.portfolio_return(weights),
            'volatility': self.portfolio_volatility(weights),
            'sharpe_ratio': self.portfolio_sharpe(weights),
            'success': result.success
        }
    
    def daf_optimized_portfolio(
        self,
        daf_size: float,
        payout_rate: float = 0.05,
        investment_horizon: int = 10
    ) -> Dict:
        """
        Optimize DAF (Donor-Advised Fund) portfolio.
        
        Args:
            daf_size: DAF size in dollars
            payout_rate: Annual payout rate (default 5%)
            investment_horizon: Investment horizon in years
            
        Returns:
            Dict with optimized portfolio and recommendations
        """
        # Get max Sharpe portfolio
        max_sharpe = self.max_sharpe_portfolio()
        
        # Calculate additional charitable giving from optimization
        # vs typical DAF default return of 4-6%
        typical_return = 0.05  # Conservative typical DAF return
        optimized_return = max_sharpe['expected_return']
        extra_return = optimized_return - typical_return
        
        # Calculate impact metrics
        annual_extra = daf_size * extra_return
        annual_grant = daf_size * payout_rate
        extra_grants = annual_extra
        
        # Lives saved calculation (based on GiveWell AMF: $3,500/life)
        lives_saved_5yr = (annual_grant + extra_grants) * 5 / 3500
        
        return {
            'weights': max_sharpe['weights'],
            'expected_return': optimized_return,
            'volatility': max_sharpe['volatility'],
            'sharpe_ratio': max_sharpe['sharpe_ratio'],
            'daf_size': daf_size,
            'annual_payout': annual_grant,
            'extra_annual_grants': extra_grants,
            'extra_grants_5yr': extra_grants * 5,
            'lives_saved_5yr': lives_saved_5yr,
            'investment_horizon': investment_horizon,
            'asset_allocation': self._format_allocation(max_sharpe['weights'])
        }
    
    def _format_allocation(self, weights: Dict) -> str:
        """Format weights into readable allocation string."""
        # Group by category (if available)
        formatted = []
        for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
            if weight > 0.01:  # Only show >1%
                formatted.append(f"{name}: {weight*100:.1f}%")
        return "\n".join(formatted)


def optimize_daf_portfolio(
    funds_df: pd.DataFrame,
    selected_funds: List[str],
    daf_amount: float
) -> Dict:
    """
    Convenience function to optimize a DAF portfolio.
    
    Args:
        funds_df: DataFrame with fund data including monthly returns
        selected_funds: List of fund names to include
        daf_amount: DAF size in dollars
        
    Returns:
        Optimization results
    """
    # Filter to selected funds
    selected = funds_df[funds_df['name'].isin(selected_funds)].copy()
    
    if len(selected) < 2:
        return {'error': 'Need at least 2 funds for optimization'}
    
    # Extract monthly returns
    returns_df = pd.DataFrame(
        selected['monthly_returns'].tolist(),
        index=selected['name']
    ).T
    
    # Run optimization
    optimizer = PortfolioOptimizer(returns_df)
    
    results = {
        'min_variance': optimizer.min_variance_portfolio(),
        'max_sharpe': optimizer.max_sharpe_portfolio(),
        'risk_parity': optimizer.risk_parity_portfolio(),
        'daf_recommendation': optimizer.daf_optimized_portfolio(daf_amount)
    }
    
    return results
