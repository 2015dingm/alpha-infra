"""
Alpha Generation Pipeline Infrastructure

A modular pipeline for transforming stock returns into trading signals:
1. Windowing: Raw returns -> Windowed returns
2. Pre-processing: Windowed returns -> Processed returns  
3. Reduction: Processed returns -> Reduced return vector
4. Post-processing: Reduced vector -> Trading signal
5. Position sizing: Signal -> Portfolio weights
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional, Union
import warnings

# =============================================================================
# STEP 1: WINDOWING FUNCTIONS (R -> R^w)
# =============================================================================

def rolling_window(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Create rolling window of returns."""
    return returns.rolling(window=window)

def expanding_window(returns: pd.DataFrame, min_periods: int = 1) -> pd.DataFrame:
    """Create expanding window of returns."""
    return returns.expanding(min_periods=min_periods)

def fixed_window(returns: pd.DataFrame, window: int, step: int = 1) -> list:
    """Create fixed-size non-overlapping windows."""
    windowed_data = []
    for i in range(0, len(returns) - window + 1, step):
        windowed_data.append(returns.iloc[i:i+window])
    return windowed_data

def exponential_window(returns: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """Create exponentially weighted window."""
    return returns.ewm(alpha=alpha)

# =============================================================================
# STEP 2: PRE-PROCESSING FUNCTIONS (R^w -> Ř^w)
# =============================================================================

def clip_outliers(data: pd.DataFrame, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """Clip extreme values to percentile bounds."""
    return data.clip(
        lower=data.quantile(lower_pct, axis=1), 
        upper=data.quantile(upper_pct, axis=1),
        axis=1
    )

def threshold_filter(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Set values below threshold to zero."""
    return data.where(np.abs(data) >= threshold, 0)

def z_score_normalize(data: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
    """Standardize data using z-score."""
    if window:
        mean = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
    else:
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        # Convert to DataFrame for proper broadcasting
        mean = pd.DataFrame([mean] * len(data.columns)).T
        mean.columns = data.columns
        mean.index = data.index
        std = pd.DataFrame([std] * len(data.columns)).T
        std.columns = data.columns
        std.index = data.index
    
    return (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

def robust_normalize(data: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
    """Normalize using median and MAD (robust to outliers)."""
    if window:
        median = data.rolling(window=window).median()
        mad = data.rolling(window=window).apply(lambda x: np.median(np.abs(x - np.median(x))))
    else:
        median = data.median(axis=1)
        mad = data.apply(lambda row: np.median(np.abs(row - np.median(row))), axis=1)
        # Convert to DataFrame for proper broadcasting
        median = pd.DataFrame([median] * len(data.columns)).T
        median.columns = data.columns
        median.index = data.index
        mad = pd.DataFrame([mad] * len(data.columns)).T
        mad.columns = data.columns
        mad.index = data.index
    
    return (data - median) / (mad + 1e-8)

def winsorize(data: pd.DataFrame, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
    """Winsorize extreme values."""
    from scipy.stats import mstats
    result = data.copy()
    
    # Apply winsorize to each row (time period)
    for i in range(len(data)):
        row_data = data.iloc[i].values
        if not np.isnan(row_data).all():  # Skip if all NaN
            winsorized = mstats.winsorize(row_data, limits=limits)
            result.iloc[i] = winsorized
    
    return result

# =============================================================================
# STEP 3: REDUCTION FUNCTIONS (Ř^w -> r)
# =============================================================================

def rolling_mean(windowed_data: pd.core.window.rolling.Rolling) -> pd.Series:
    """Compute rolling mean."""
    return windowed_data.mean()

def ewma_reduction(windowed_data: pd.core.window.ewm.ExponentialMovingWindow) -> pd.Series:
    """Compute exponentially weighted moving average."""
    return windowed_data.mean()

def rolling_std(windowed_data: pd.core.window.rolling.Rolling) -> pd.Series:
    """Compute rolling standard deviation."""
    return windowed_data.std()

def rolling_skewness(windowed_data: pd.core.window.rolling.Rolling) -> pd.Series:
    """Compute rolling skewness."""
    return windowed_data.skew()

def rolling_kurtosis(windowed_data: pd.core.window.rolling.Rolling) -> pd.Series:
    """Compute rolling kurtosis."""
    return windowed_data.kurt()

def cross_sectional_mean(data: pd.DataFrame) -> pd.Series:
    """Compute cross-sectional mean across assets."""
    return data.mean(axis=1)

def cross_sectional_rank(data: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional rank."""
    return data.rank(axis=1, pct=True)

def principal_component(data: pd.DataFrame, n_components: int = 1) -> pd.Series:
    """Extract first principal component."""
    from sklearn.decomposition import PCA
    
    # Handle NaN values - drop rows with any NaN
    data_clean = data.dropna()
    if len(data_clean) == 0:
        return pd.Series(index=data.index, dtype=float)
    
    # PCA expects samples (dates) as rows and features (assets) as columns
    # So we use data_clean directly (dates x assets)
    pca = PCA(n_components=n_components)
    
    try:
        # Fit and transform the data (dates x assets)
        pc_scores = pca.fit_transform(data_clean)  # Shape: (n_dates, n_components)
        
        # Create result series with first principal component
        result = pd.Series(index=data.index, dtype=float)
        result.loc[data_clean.index] = pc_scores[:, 0]  # First component for each date
        
        return result
    
    except Exception:
        # If PCA fails, return cross-sectional mean as fallback
        return data_clean.mean(axis=1)

# =============================================================================
# STEP 4: POST-PROCESSING FUNCTIONS (r -> s)
# =============================================================================

def clip_signal(signal: pd.Series, lower: float = -3, upper: float = 3) -> pd.Series:
    """Clip signal to bounds."""
    return signal.clip(lower=lower, upper=upper)

def rank_signal(signal: pd.Series) -> pd.Series:
    """Convert signal to ranks."""
    return signal.rank(pct=True) - 0.5  # Center around 0

def sign_signal(signal: pd.Series) -> pd.Series:
    """Convert to binary signal."""
    return np.sign(signal)

def smooth_signal(signal: pd.Series, window: int) -> pd.Series:
    """Smooth signal using rolling mean."""
    return signal.rolling(window=window, center=True).mean()

def threshold_signal(signal: pd.Series, threshold: float) -> pd.Series:
    """Apply threshold to signal."""
    return signal.where(np.abs(signal) >= threshold, 0)

def zscore_signal(signal: pd.Series, window: Optional[int] = None) -> pd.Series:
    """Normalize signal using z-score."""
    if window:
        mean = signal.rolling(window=window).mean()
        std = signal.rolling(window=window).std()
    else:
        mean = signal.mean()
        std = signal.std()
    
    return (signal - mean) / (std + 1e-8)

# =============================================================================
# STEP 5: POSITION SIZING / HEDGING FUNCTIONS (s -> w)
# =============================================================================

def dollar_neutral(signal: pd.Series) -> pd.Series:
    """Make positions dollar neutral."""
    return signal - signal.mean()

def market_neutral(signal: pd.Series, market_cap: Optional[pd.Series] = None) -> pd.Series:
    """Make positions market neutral."""
    if market_cap is not None:
        # Weight by market cap for proper market neutrality
        weights = market_cap / market_cap.sum()
        market_exposure = (signal * weights).sum()
        return signal - market_exposure
    else:
        # Simple equal-weight market neutral
        return signal - signal.mean()

def volatility_target(signal: pd.Series, target_vol: float = 0.1, 
                     realized_vol: Optional[pd.Series] = None) -> pd.Series:
    """Scale positions to target volatility."""
    if realized_vol is None:
        realized_vol = signal.rolling(window=252).std() * np.sqrt(252)
    
    vol_scalar = target_vol / (realized_vol + 1e-8)
    return signal * vol_scalar

def position_limits(signal: pd.Series, max_position: float = 0.05) -> pd.Series:
    """Apply position size limits."""
    return signal.clip(-max_position, max_position)

def leverage_constraint(positions: pd.Series, max_leverage: float = 1.0) -> pd.Series:
    """Ensure leverage doesn't exceed limit."""
    current_leverage = positions.abs().sum()
    if current_leverage > max_leverage:
        return positions * (max_leverage / current_leverage)
    return positions

# =============================================================================
# PIPELINE ORCHESTRATION FUNCTIONS
# =============================================================================

def build_pipeline_config() -> Dict[str, Dict[str, Callable]]:
    """Return available functions for each pipeline step."""
    return {
        'windowing': {
            'rolling': rolling_window,
            'expanding': expanding_window,
            'fixed': fixed_window,
            'exponential': exponential_window,
        },
        'preprocessing': {
            'clip_outliers': clip_outliers,
            'threshold_filter': threshold_filter,
            'z_score': z_score_normalize,
            'robust_normalize': robust_normalize,
            'winsorize': winsorize,
        },
        'reduction': {
            'rolling_mean': rolling_mean,
            'ewma': ewma_reduction,
            'rolling_std': rolling_std,
            'rolling_skewness': rolling_skewness,
            'rolling_kurtosis': rolling_kurtosis,
            'cross_sectional_mean': cross_sectional_mean,
            'cross_sectional_rank': cross_sectional_rank,
            'principal_component': principal_component,
        },
        'postprocessing': {
            'clip': clip_signal,
            'rank': rank_signal,
            'sign': sign_signal,
            'smooth': smooth_signal,
            'threshold': threshold_signal,
            'zscore': zscore_signal,
        },
        'position_sizing': {
            'dollar_neutral': dollar_neutral,
            'market_neutral': market_neutral,
            'volatility_target': volatility_target,
            'position_limits': position_limits,
            'leverage_constraint': leverage_constraint,
        }
    }

def run_alpha_pipeline(returns: pd.DataFrame, 
                      pipeline_config: Dict[str, Dict[str, Any]]) -> pd.Series:
    """
    Execute the full alpha generation pipeline.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Raw returns data (dates x assets)
    pipeline_config : dict
        Configuration for each pipeline step with function name and parameters
        
    Example:
    --------
    config = {
        'windowing': {'function': 'rolling', 'params': {'window': 20}},
        'preprocessing': {'function': 'z_score', 'params': {}},
        'reduction': {'function': 'cross_sectional_mean', 'params': {}},
        'postprocessing': {'function': 'rank', 'params': {}},
        'position_sizing': {'function': 'dollar_neutral', 'params': {}}
    }
    """
    
    available_functions = build_pipeline_config()
    data = returns.copy()
    
    # Step 1: Windowing
    windowed_data = None
    if 'windowing' in pipeline_config:
        func_name = pipeline_config['windowing']['function']
        params = pipeline_config['windowing']['params']
        windowing_func = available_functions['windowing'][func_name]
        
        if func_name in ['rolling', 'expanding', 'exponential']:
            windowed_data = windowing_func(data, **params)
        else:
            data = windowing_func(data, **params)
    
    # Step 2: Pre-processing
    if 'preprocessing' in pipeline_config:
        func_name = pipeline_config['preprocessing']['function']
        params = pipeline_config['preprocessing']['params']
        preprocessing_func = available_functions['preprocessing'][func_name]
        
        if windowed_data is not None:
            # For rolling/expanding windows, we can't easily apply preprocessing
            # So we skip this step and work with original data
            data = preprocessing_func(data, **params)
        else:
            data = preprocessing_func(data, **params)
    
    # Step 3: Reduction
    if 'reduction' in pipeline_config:
        func_name = pipeline_config['reduction']['function']
        params = pipeline_config['reduction']['params']
        reduction_func = available_functions['reduction'][func_name]
        
        if windowed_data is not None and func_name in ['rolling_mean', 'ewma', 'rolling_std', 'rolling_skewness', 'rolling_kurtosis']:
            # Apply reduction to windowed data
            data = reduction_func(windowed_data, **params)
            # Take mean across assets to get single signal
            if isinstance(data, pd.DataFrame):
                data = data.mean(axis=1)
        elif func_name in ['cross_sectional_mean', 'cross_sectional_rank', 'principal_component', 'dispersion_factor']:
            data = reduction_func(data, **params)
        else:
            # For other reduction functions, work with current data
            if isinstance(data, pd.DataFrame):
                data = data.mean(axis=1)  # Simple mean as fallback
    
    # Ensure we have a Series at this point
    if isinstance(data, pd.DataFrame):
        data = data.mean(axis=1)
    
    # Step 4: Post-processing
    if 'postprocessing' in pipeline_config:
        func_name = pipeline_config['postprocessing']['function']
        params = pipeline_config['postprocessing']['params']
        postprocessing_func = available_functions['postprocessing'][func_name]
        data = postprocessing_func(data, **params)
    
    # Step 5: Position sizing
    if 'position_sizing' in pipeline_config:
        func_name = pipeline_config['position_sizing']['function']
        params = pipeline_config['position_sizing']['params']
        position_func = available_functions['position_sizing'][func_name]
        data = position_func(data, **params)
    
    return data

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_available_functions() -> None:
    """Print all available functions by category."""
    functions = build_pipeline_config()
    
    for step, funcs in functions.items():
        print(f"\n{step.upper()}:")
        for name in funcs.keys():
            print(f"  - {name}")

def validate_pipeline_config(config: Dict[str, Dict[str, Any]]) -> bool:
    """Validate pipeline configuration."""
    available_functions = build_pipeline_config()
    
    for step, step_config in config.items():
        if step not in available_functions:
            print(f"Unknown pipeline step: {step}")
            return False
        
        func_name = step_config.get('function')
        if func_name not in available_functions[step]:
            print(f"Unknown function '{func_name}' for step '{step}'")
            return False
    
    return True

# =============================================================================
# BACKTESTING FUNCTIONS
# =============================================================================

def calculate_portfolio_returns(signal: pd.Series, 
                               returns_data: pd.DataFrame,
                               transaction_cost: float = 0.001,
                               max_position: float = 0.1,
                               leverage_limit: float = 1.0) -> pd.Series:
    """Calculate portfolio returns from signal and asset returns."""
    
    # Align signal with returns data
    common_dates = signal.index.intersection(returns_data.index)
    signal_aligned = signal.loc[common_dates]
    returns_aligned = returns_data.loc[common_dates]
    
    if len(signal_aligned) == 0:
        return pd.Series(dtype=float)
    
    # Generate positions from signal
    # Simple strategy: signal strength determines position allocation
    n_assets = len(returns_aligned.columns)
    positions = pd.DataFrame(index=signal_aligned.index, columns=returns_aligned.columns, dtype=float)
    
    for date in signal_aligned.index:
        signal_strength = signal_aligned.loc[date]
        
        if pd.notna(signal_strength):
            # Distribute signal across assets
            if signal_strength > 0:
                weight_per_asset = min(abs(signal_strength), max_position) / n_assets
                positions.loc[date] = weight_per_asset
            elif signal_strength < 0:
                weight_per_asset = min(abs(signal_strength), max_position) / n_assets
                positions.loc[date] = -weight_per_asset
            else:
                positions.loc[date] = 0.0
        else:
            positions.loc[date] = 0.0
        
        # Apply leverage limit
        total_leverage = positions.loc[date].abs().sum()
        if total_leverage > leverage_limit:
            positions.loc[date] *= leverage_limit / total_leverage
    
    positions = positions.fillna(0)
    
    # Calculate portfolio returns
    portfolio_returns = (positions.shift(1) * returns_aligned).sum(axis=1).fillna(0)
    
    # Apply transaction costs
    position_changes = positions.diff().abs().sum(axis=1)
    transaction_costs = position_changes * transaction_cost
    
    # Net returns after costs
    net_returns = portfolio_returns - transaction_costs
    
    return net_returns

def calculate_performance_metrics(portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series = None) -> dict:
    """Calculate comprehensive performance metrics."""
    
    if len(portfolio_returns) == 0:
        return {}
    
    # Remove NaN values
    returns = portfolio_returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = returns.cumsum().iloc[-1]
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0.0
    else:
        sortino_ratio = float('inf') if annual_return > 0 else 0.0
    
    # Drawdown analysis
    cumulative_returns = returns.cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # Value at Risk
    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)
    
    # Win rate and trade statistics (simplified)
    winning_periods = (returns > 0).sum()
    total_periods = len(returns)
    win_rate = winning_periods / total_periods if total_periods > 0 else 0.0
    
    avg_win = returns[returns > 0].mean() if winning_periods > 0 else 0.0
    avg_loss = returns[returns < 0].mean() if (total_periods - winning_periods) > 0 else 0.0
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'var_99': var_99,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_periods': total_periods,
        'portfolio_returns': returns,
        'cumulative_returns': cumulative_returns,
        'drawdown': drawdown
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None:
        bench_aligned = benchmark_returns.reindex(returns.index).fillna(0)
        if len(bench_aligned) > 0:
            active_returns = returns - bench_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0.0
            
            # Beta calculation
            if bench_aligned.std() > 0:
                correlation = returns.corr(bench_aligned)
                beta = correlation * (returns.std() / bench_aligned.std()) if not pd.isna(correlation) else 0.0
                alpha = returns.mean() - beta * bench_aligned.mean()
            else:
                beta = 0.0
                alpha = 0.0
            
            metrics.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha * 252  # Annualized alpha
            })
    
    return metrics

def backtest_signal(signal: pd.Series,
                   returns_data: pd.DataFrame,
                   transaction_cost: float = 0.001,
                   max_position: float = 0.1,
                   leverage_limit: float = 1.0,
                   benchmark_returns: pd.Series = None) -> dict:
    """
    Backtest a trading signal against historical returns.
    
    Parameters:
    -----------
    signal : pd.Series
        Trading signal (dates)
    returns_data : pd.DataFrame  
        Asset returns data (dates x assets)
    transaction_cost : float
        Transaction cost as percentage
    max_position : float
        Maximum position size per asset
    leverage_limit : float
        Maximum portfolio leverage
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics and results
    """
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(
        signal=signal,
        returns_data=returns_data,
        transaction_cost=transaction_cost,
        max_position=max_position,
        leverage_limit=leverage_limit
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns)
    
    return metrics

def print_backtest_results(metrics: dict, strategy_name: str = "Strategy"):
    """Print formatted backtest results."""
    
    if not metrics:
        print(f"No results available for {strategy_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {strategy_name.upper()}")
    print(f"{'='*60}")
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'─'*40}")
    print(f"Total Return:           {metrics.get('total_return', 0):8.2%}")
    print(f"Annual Return:          {metrics.get('annual_return', 0):8.2%}")
    print(f"Annual Volatility:      {metrics.get('annual_volatility', 0):8.2%}")
    print(f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):8.2f}")
    print(f"Sortino Ratio:          {metrics.get('sortino_ratio', 0):8.2f}")
    print(f"Calmar Ratio:           {metrics.get('calmar_ratio', 0):8.2f}")
    
    print(f"\n⚠️  RISK METRICS")
    print(f"{'─'*40}")
    print(f"Maximum Drawdown:       {metrics.get('max_drawdown', 0):8.2%}")
    print(f"VaR (95%):             {metrics.get('var_95', 0):8.2%}")
    print(f"VaR (99%):             {metrics.get('var_99', 0):8.2%}")
    
    print(f"\n💼 TRADING STATISTICS")
    print(f"{'─'*40}")
    print(f"Win Rate:               {metrics.get('win_rate', 0):8.2%}")
    print(f"Average Win:            {metrics.get('avg_win', 0):8.4f}")
    print(f"Average Loss:           {metrics.get('avg_loss', 0):8.4f}")
    print(f"Total Periods:          {metrics.get('total_periods', 0):8d}")
    
    # Benchmark comparison if available
    if 'information_ratio' in metrics:
        print(f"\n📈 BENCHMARK COMPARISON")
        print(f"{'─'*40}")
        print(f"Information Ratio:      {metrics.get('information_ratio', 0):8.2f}")
        print(f"Beta:                   {metrics.get('beta', 0):8.2f}")
        print(f"Alpha (Annual):         {metrics.get('alpha', 0):8.2%}")
        print(f"Tracking Error:         {metrics.get('tracking_error', 0):8.2%}")

def plot_backtest_results(metrics: dict, 
                         benchmark_returns: pd.Series = None,
                         strategy_name: str = "Strategy",
                         save_path: str = None):
    """Plot backtest results."""
    
    if not metrics or 'portfolio_returns' not in metrics:
        print("No data to plot")
        return
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{strategy_name} - Backtest Results', fontsize=16, fontweight='bold')
        
        portfolio_returns = metrics['portfolio_returns']
        cumulative_returns = metrics['cumulative_returns']
        drawdown = metrics['drawdown']
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                 label='Strategy', linewidth=2, color='blue')
        
        if benchmark_returns is not None:
            bench_cum = benchmark_returns.reindex(cumulative_returns.index).fillna(0).cumsum()
            ax1.plot(bench_cum.index, bench_cum.values, 
                    label='Benchmark', linewidth=2, color='red', alpha=0.7)
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe (60-day)
        ax3 = axes[1, 0]
        if len(portfolio_returns) >= 60:
            rolling_sharpe = portfolio_returns.rolling(window=60).mean() / \
                           portfolio_returns.rolling(window=60).std() * np.sqrt(252)
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_title('60-Day Rolling Sharpe Ratio')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data\nfor rolling Sharpe', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('60-Day Rolling Sharpe Ratio')
        
        # 4. Return Distribution
        ax4 = axes[1, 1]
        returns_clean = portfolio_returns.dropna()
        if len(returns_clean) > 0:
            ax4.hist(returns_clean, bins=min(30, len(returns_clean)//2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax4.axvline(returns_clean.mean(), color='red', linestyle='--', 
                       label=f'Mean: {returns_clean.mean():.4f}')
            ax4.set_title('Daily Returns Distribution')
            ax4.set_xlabel('Daily Return')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")

def compare_strategies(strategy_results: dict, save_path: str = None):
    """Compare multiple strategy results."""
    
    if len(strategy_results) < 2:
        print("Need at least 2 strategies to compare")
        return
    
    # Create comparison table
    comparison_data = []
    for name, metrics in strategy_results.items():
        if metrics:  # Check if metrics exist
            comparison_data.append({
                'Strategy': name,
                'Annual Return': f"{metrics.get('annual_return', 0):.2%}",
                'Annual Vol': f"{metrics.get('annual_volatility', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Calmar Ratio': f"{metrics.get('calmar_ratio', 0):.2f}",
                'Win Rate': f"{metrics.get('win_rate', 0):.2%}"
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n{'='*80}")
        print("STRATEGY COMPARISON")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        
        # Plot comparison if matplotlib available
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Risk-Return scatter
            ax1 = axes[0]
            for name, metrics in strategy_results.items():
                if metrics:
                    ax1.scatter(metrics.get('annual_volatility', 0), 
                              metrics.get('annual_return', 0), 
                              s=100, label=name, alpha=0.7)
            
            ax1.set_xlabel('Annual Volatility')
            ax1.set_ylabel('Annual Return')
            ax1.set_title('Risk-Return Profile')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Cumulative returns comparison
            ax2 = axes[1]
            for name, metrics in strategy_results.items():
                if metrics and 'cumulative_returns' in metrics:
                    cum_ret = metrics['cumulative_returns']
                    ax2.plot(cum_ret.index, cum_ret.values, label=name, linewidth=2)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Return')
            ax2.set_title('Cumulative Returns Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating comparison plots: {str(e)}")

# =============================================================================
# COMPLETE PIPELINE WITH BACKTESTING
# =============================================================================

def run_alpha_pipeline_with_backtest(returns_data: pd.DataFrame,
                                    pipeline_config: Dict[str, Dict[str, Any]],
                                    transaction_cost: float = 0.001,
                                    max_position: float = 0.1,
                                    leverage_limit: float = 1.0,
                                    benchmark_returns: pd.Series = None,
                                    strategy_name: str = "Alpha Strategy",
                                    print_results: bool = True,
                                    plot_results: bool = True,
                                    save_plots: bool = False) -> tuple:
    """
    Run complete alpha pipeline with integrated backtesting.
    
    Parameters:
    -----------
    returns_data : pd.DataFrame
        Asset returns data (dates x assets)
    pipeline_config : dict
        Pipeline configuration
    transaction_cost : float
        Transaction costs
    max_position : float
        Maximum position per asset
    leverage_limit : float
        Maximum leverage
    benchmark_returns : pd.Series, optional
        Benchmark for comparison
    strategy_name : str
        Strategy name
    print_results : bool
        Whether to print results
    plot_results : bool
        Whether to create plots
    save_plots : bool
        Whether to save plots
        
    Returns:
    --------
    tuple
        (signal, backtest_metrics)
    """
    
    # Step 1: Generate alpha signal
    signal = run_alpha_pipeline(returns_data, pipeline_config)
    
    # Step 2: Backtest the signal
    backtest_metrics = backtest_signal(
        signal=signal,
        returns_data=returns_data,
        transaction_cost=transaction_cost,
        max_position=max_position,
        leverage_limit=leverage_limit,
        benchmark_returns=benchmark_returns
    )
    
    # Step 3: Display results
    if print_results:
        print_backtest_results(backtest_metrics, strategy_name)
    
    if plot_results and backtest_metrics:
        save_path = f"{strategy_name.lower().replace(' ', '_')}_backtest.png" if save_plots else None
        plot_backtest_results(backtest_metrics, benchmark_returns, strategy_name, save_path)
    
    return signal, backtest_metrics
def create_sample_pipeline_configs() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return some example pipeline configurations."""
    return {
        'momentum': {
            'windowing': {'function': 'rolling', 'params': {'window': 20}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'rolling_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'dollar_neutral', 'params': {}}
        },
        'mean_reversion': {
            'windowing': {'function': 'rolling', 'params': {'window': 10}},
            'preprocessing': {'function': 'clip_outliers', 'params': {'lower_pct': 0.05, 'upper_pct': 0.95}},
            'reduction': {'function': 'cross_sectional_rank', 'params': {}},
            'postprocessing': {'function': 'sign', 'params': {}},
            'position_sizing': {'function': 'market_neutral', 'params': {}}
        },
        'volatility': {
            'windowing': {'function': 'rolling', 'params': {'window': 252}},
            'preprocessing': {'function': 'robust_normalize', 'params': {}},
            'reduction': {'function': 'rolling_std', 'params': {}},
            'postprocessing': {'function': 'threshold', 'params': {'threshold': 0.5}},
            'position_sizing': {'function': 'volatility_target', 'params': {'target_vol': 0.15}}
        }
    }

if __name__ == "__main__":
    # Example usage
    print("Alpha Generation Pipeline Infrastructure")
    print("=" * 50)
    list_available_functions()
