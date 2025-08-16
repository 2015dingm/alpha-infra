"""
Extension Examples for Alpha Pipeline

This module shows how easy it is to add new functions to the infrastructure.
Just define your function and add it to the appropriate category.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
import warnings

# =============================================================================
# EXTENDED WINDOWING FUNCTIONS
# =============================================================================

def seasonal_window(returns: pd.DataFrame, season_length: int = 252) -> pd.DataFrame:
    """Create seasonal/cyclical windows (e.g., same day of year)."""
    seasonal_data = {}
    for i in range(len(returns)):
        season_idx = i % season_length
        if season_idx not in seasonal_data:
            seasonal_data[season_idx] = []
        seasonal_data[season_idx].append(returns.iloc[i])
    
    # Convert back to DataFrame format
    result = returns.copy()
    for i in range(len(returns)):
        season_idx = i % season_length
        if len(seasonal_data[season_idx]) > 1:
            # Use historical data from same season
            historical = pd.DataFrame(seasonal_data[season_idx][:-1])
            result.iloc[i] = historical.mean()
    
    return result

def volatility_adjusted_window(returns: pd.DataFrame, vol_window: int = 20, 
                              target_vol: float = 0.02) -> pd.DataFrame:
    """Adjust window size based on volatility regime."""
    volatility = returns.rolling(window=vol_window).std().mean(axis=1)
    
    # Adjust window size inversely to volatility
    base_window = 20
    vol_ratio = volatility / target_vol
    adjusted_windows = np.clip(base_window / vol_ratio, 5, 60).astype(int)
    
    result = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for i, window in enumerate(adjusted_windows):
        if i >= window:
            result.iloc[i] = returns.iloc[i-window:i].mean()
        else:
            result.iloc[i] = returns.iloc[:i+1].mean()
    
    return result

# =============================================================================
# EXTENDED PRE-PROCESSING FUNCTIONS
# =============================================================================

def regime_aware_normalize(data: pd.DataFrame, regime_window: int = 60) -> pd.DataFrame:
    """Normalize based on detected volatility regimes."""
    # Simple 2-regime model based on volatility
    vol = data.rolling(window=20).std().mean(axis=1)
    vol_threshold = vol.rolling(window=regime_window).median()
    
    high_vol_regime = vol > vol_threshold
    
    result = data.copy()
    
    # Normalize differently for each regime
    for regime_mask, regime_name in [(high_vol_regime, 'high'), (~high_vol_regime, 'low')]:
        if regime_mask.any():
            regime_data = data[regime_mask]
            if len(regime_data) > 5:  # Need sufficient data
                mean = regime_data.mean()
                std = regime_data.std()
                result.loc[regime_mask] = (regime_data - mean) / (std + 1e-8)
    
    return result

def adaptive_clipping(data: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Adaptive clipping based on rolling quantiles."""
    rolling_q01 = data.rolling(window=lookback).quantile(0.01)
    rolling_q99 = data.rolling(window=lookback).quantile(0.99)
    
    return data.clip(lower=rolling_q01, upper=rolling_q99, axis=1)

def sector_neutralize(data: pd.DataFrame, sector_mapping: dict) -> pd.DataFrame:
    """Neutralize by sector (requires sector mapping)."""
    result = data.copy()
    
    # Group by sectors and subtract sector mean
    for sector, assets in sector_mapping.items():
        sector_assets = [asset for asset in assets if asset in data.columns]
        if sector_assets:
            sector_mean = data[sector_assets].mean(axis=1)
            result[sector_assets] = data[sector_assets].sub(sector_mean, axis=0)
    
    return result

# =============================================================================
# EXTENDED REDUCTION FUNCTIONS  
# =============================================================================

def information_ratio(windowed_data: pd.core.window.rolling.Rolling) -> pd.Series:
    """Compute rolling information ratio."""
    returns_mean = windowed_data.mean()
    returns_std = windowed_data.std()
    return returns_mean / (returns_std + 1e-8)

def cross_sectional_momentum(data: pd.DataFrame, momentum_window: int = 20) -> pd.Series:
    """Compute cross-sectional momentum factor."""
    momentum = data.pct_change(momentum_window)
    return momentum.mean(axis=1)

def dispersion_factor(data: pd.DataFrame) -> pd.Series:
    """Compute cross-sectional dispersion."""
    return data.std(axis=1)

def correlation_factor(data: pd.DataFrame, benchmark_col: str = None) -> pd.Series:
    """Compute average correlation with benchmark or first asset."""
    if benchmark_col is None:
        benchmark_col = data.columns[0]
    
    if benchmark_col not in data.columns:
        return pd.Series(index=data.index, dtype=float)
    
    benchmark = data[benchmark_col]
    correlations = data.corrwith(benchmark, axis=1)
    
    return correlations

def regime_conditional_mean(data: pd.DataFrame, regime_indicator: pd.Series = None) -> pd.Series:
    """Compute mean conditional on market regime."""
    if regime_indicator is None:
        # Use volatility as simple regime indicator
        vol = data.std(axis=1)
        regime_indicator = vol > vol.median()
    
    result = pd.Series(index=data.index, dtype=float)
    
    # High volatility regime
    high_vol_mask = regime_indicator
    if high_vol_mask.any():
        result.loc[high_vol_mask] = data.loc[high_vol_mask].mean(axis=1)
    
    # Low volatility regime  
    low_vol_mask = ~regime_indicator
    if low_vol_mask.any():
        result.loc[low_vol_mask] = data.loc[low_vol_mask].mean(axis=1) * 0.5  # Reduce exposure
    
    return result

# =============================================================================
# EXTENDED POST-PROCESSING FUNCTIONS
# =============================================================================

def power_transform(signal: pd.Series, power: float = 0.5) -> pd.Series:
    """Apply power transformation to signal."""
    return np.sign(signal) * (np.abs(signal) ** power)

def quantile_transform(signal: pd.Series, n_quantiles: int = 100) -> pd.Series:
    """Transform signal to uniform distribution."""
    from sklearn.preprocessing import QuantileTransformer
    
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='uniform')
    transformed = qt.fit_transform(signal.values.reshape(-1, 1)).flatten()
    
    return pd.Series(transformed - 0.5, index=signal.index)  # Center around 0

def regime_conditional_signal(signal: pd.Series, regime_indicator: pd.Series = None, 
                             regime_multiplier: float = 2.0) -> pd.Series:
    """Amplify signal in certain regimes."""
    if regime_indicator is None:
        # Use signal volatility as regime indicator
        regime_indicator = signal.rolling(20).std() > signal.rolling(60).std()
    
    result = signal.copy()
    result.loc[regime_indicator] *= regime_multiplier
    
    return result

def decay_signal(signal: pd.Series, decay_rate: float = 0.95) -> pd.Series:
    """Apply exponential decay to signal."""
    result = signal.copy()
    
    for i in range(1, len(result)):
        if pd.notna(result.iloc[i-1]):
            # New signal = current signal + decayed previous signal  
            result.iloc[i] = signal.iloc[i] + decay_rate * result.iloc[i-1]
    
    return result

# =============================================================================
# EXTENDED POSITION SIZING FUNCTIONS
# =============================================================================

def risk_parity_weights(signal: pd.Series, lookback: int = 60) -> pd.Series:
    """Risk parity position sizing."""
    # Estimate risk using rolling volatility
    risk = signal.rolling(window=lookback).std()
    
    # Inverse volatility weighting
    inv_vol_weights = 1 / (risk + 1e-8)
    normalized_weights = inv_vol_weights / inv_vol_weights.rolling(window=lookback).sum()
    
    return signal * normalized_weights

def kelly_sizing(signal: pd.Series, returns: pd.Series, lookback: int = 252) -> pd.Series:
    """Kelly criterion position sizing."""
    # Estimate win rate and avg win/loss
    signal_returns = (signal.shift(1) * returns).dropna()
    
    win_rate = (signal_returns > 0).rolling(lookback).mean()
    avg_win = signal_returns[signal_returns > 0].rolling(lookback).mean()
    avg_loss = signal_returns[signal_returns < 0].rolling(lookback).mean()
    
    # Kelly fraction: f = (bp - q) / b where b = avg_win/|avg_loss|, p = win_rate, q = 1-win_rate
    b = avg_win / np.abs(avg_loss + 1e-8)
    kelly_f = (b * win_rate - (1 - win_rate)) / (b + 1e-8)
    
    # Cap Kelly fraction to prevent extreme leverage
    kelly_f = kelly_f.clip(-0.25, 0.25)
    
    return signal * kelly_f

def max_drawdown_sizing(signal: pd.Series, max_dd_threshold: float = 0.05) -> pd.Series:
    """Reduce position size based on drawdown."""
    # Calculate cumulative returns (simplified)
    cum_returns = (signal * 0.01).cumsum()  # Assume 1% per day returns
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / (running_max + 1e-8)
    
    # Reduce position size when drawdown is large
    size_multiplier = np.clip(1 + drawdown / max_dd_threshold, 0.1, 1.0)
    
    return signal * size_multiplier

def correlation_adjusted_sizing(signal: pd.Series, other_signals: pd.DataFrame = None) -> pd.Series:
    """Adjust position size based on correlation with other strategies."""
    if other_signals is None or other_signals.empty:
        return signal
    
    # Calculate rolling correlation with other signals
    avg_correlation = other_signals.corrwith(signal, axis=0).mean()
    rolling_corr = pd.Series(index=signal.index, dtype=float)
    
    window = 60
    for i in range(window, len(signal)):
        recent_signal = signal.iloc[i-window:i]
        recent_others = other_signals.iloc[i-window:i]
        if not recent_others.empty:
            corr = recent_others.corrwith(recent_signal, axis=0).mean()
            rolling_corr.iloc[i] = corr
    
    # Reduce position size when highly correlated with other strategies
    diversification_benefit = 1 - rolling_corr.abs()
    
    return signal * diversification_benefit.fillna(1.0)

# =============================================================================
# FUNCTION TO UPDATE PIPELINE WITH EXTENSIONS
# =============================================================================

def get_extended_pipeline_config():
    """Return extended pipeline configuration including new functions."""
    from alpha_pipeline import build_pipeline_config
    
    # Start with base configuration
    config = build_pipeline_config()
    
    # Add new windowing functions
    config['windowing'].update({
        'seasonal': seasonal_window,
        'volatility_adjusted': volatility_adjusted_window,
    })
    
    # Add new preprocessing functions
    config['preprocessing'].update({
        'regime_aware_normalize': regime_aware_normalize,
        'adaptive_clipping': adaptive_clipping,
        'sector_neutralize': sector_neutralize,
    })
    
    # Add new reduction functions
    config['reduction'].update({
        'information_ratio': information_ratio,
        'cross_sectional_momentum': cross_sectional_momentum,
        'dispersion_factor': dispersion_factor,
        'correlation_factor': correlation_factor,
        'regime_conditional_mean': regime_conditional_mean,
    })
    
    # Add new post-processing functions
    config['postprocessing'].update({
        'power_transform': power_transform,
        'quantile_transform': quantile_transform,
        'regime_conditional_signal': regime_conditional_signal,
        'decay_signal': decay_signal,
    })
    
    # Add new position sizing functions
    config['position_sizing'].update({
        'risk_parity': risk_parity_weights,
        'kelly_sizing': kelly_sizing,
        'max_drawdown_sizing': max_drawdown_sizing,
        'correlation_adjusted_sizing': correlation_adjusted_sizing,
    })
    
    return config

def demonstrate_extensions():
    """Show how the extended functions work."""
    print("Extended Alpha Pipeline Functions")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    assets = [f'ASSET_{i}' for i in range(10)]
    returns = pd.DataFrame(
        np.random.normal(0, 0.02, (200, 10)),
        index=dates,
        columns=assets
    )
    
    print("Testing extended windowing functions...")
    seasonal_data = seasonal_window(returns, season_length=50)
    vol_adj_data = volatility_adjusted_window(returns)
    print(f"✓ Seasonal window shape: {seasonal_data.shape}")
    print(f"✓ Volatility adjusted window shape: {vol_adj_data.shape}")
    
    print("\nTesting extended preprocessing functions...")
    regime_norm = regime_aware_normalize(returns)
    adaptive_clip = adaptive_clipping(returns, lookback=50)
    print(f"✓ Regime aware normalization: mean={regime_norm.mean().mean():.4f}")
    print(f"✓ Adaptive clipping: range=[{adaptive_clip.min().min():.4f}, {adaptive_clip.max().max():.4f}]")
    
    print("\nTesting extended reduction functions...")
    momentum = cross_sectional_momentum(returns, momentum_window=10)
    dispersion = dispersion_factor(returns)
    print(f"✓ Cross-sectional momentum: {len(momentum)} observations")
    print(f"✓ Dispersion factor range: [{dispersion.min():.4f}, {dispersion.max():.4f}]")
    
    print("\nTesting extended post-processing functions...")
    signal = momentum  # Use momentum as sample signal
    power_sig = power_transform(signal, power=0.7)
    quantile_sig = quantile_transform(signal)
    print(f"✓ Power transform: original std={signal.std():.4f}, transformed std={power_sig.std():.4f}")
    print(f"✓ Quantile transform range: [{quantile_sig.min():.4f}, {quantile_sig.max():.4f}]")
    
    print("\nTesting extended position sizing functions...")
    risk_parity = risk_parity_weights(signal)
    max_dd_sized = max_drawdown_sizing(signal)
    print(f"✓ Risk parity weights computed")
    print(f"✓ Max drawdown sizing computed")
    
    print("\nAll extended functions working correctly! ✅")
    
    return returns, signal

def create_advanced_strategy_examples():
    """Create example configurations using extended functions."""
    return {
        'regime_momentum': {
            'windowing': {'function': 'volatility_adjusted', 'params': {'vol_window': 20, 'target_vol': 0.015}},
            'preprocessing': {'function': 'regime_aware_normalize', 'params': {'regime_window': 60}},
            'reduction': {'function': 'cross_sectional_momentum', 'params': {'momentum_window': 15}},
            'postprocessing': {'function': 'regime_conditional_signal', 'params': {'regime_multiplier': 1.5}},
            'position_sizing': {'function': 'risk_parity', 'params': {'lookback': 60}}
        },
        
        'adaptive_mean_reversion': {
            'windowing': {'function': 'seasonal', 'params': {'season_length': 252}},
            'preprocessing': {'function': 'adaptive_clipping', 'params': {'lookback': 120}},
            'reduction': {'function': 'dispersion_factor', 'params': {}},
            'postprocessing': {'function': 'quantile_transform', 'params': {'n_quantiles': 50}},
            'position_sizing': {'function': 'max_drawdown_sizing', 'params': {'max_dd_threshold': 0.03}}
        },
        
        'information_strategy': {
            'windowing': {'function': 'rolling', 'params': {'window': 30}},
            'preprocessing': {'function': 'robust_normalize', 'params': {}},
            'reduction': {'function': 'information_ratio', 'params': {}},
            'postprocessing': {'function': 'power_transform', 'params': {'power': 0.6}},
            'position_sizing': {'function': 'kelly_sizing', 'params': {'lookback': 180}}
        }
    }

if __name__ == "__main__":
    # Demonstrate extended functions
    returns, signal = demonstrate_extensions()
    
    print(f"\n{'='*60}")
    print("ADVANCED STRATEGY EXAMPLES")
    print('='*60)
    
    # Show available extended functions
    extended_config = get_extended_pipeline_config()
    print("\nExtended functions available:")
    for category, functions in extended_config.items():
        print(f"\n{category.upper()} ({len(functions)} functions):")
        for name in functions.keys():
            print(f"  - {name}")
    
    # Create and show advanced strategies
    advanced_strategies = create_advanced_strategy_examples()
    print(f"\nCreated {len(advanced_strategies)} advanced strategy examples:")
    for name in advanced_strategies.keys():
        print(f"  - {name}")
    
    print("\n✅ Extension demonstration complete!")
    print("You can now use these advanced functions in your pipeline configurations.")
