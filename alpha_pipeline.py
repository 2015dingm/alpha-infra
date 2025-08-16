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
    
    # Handle NaN values
    data_clean = data.dropna()
    if len(data_clean) == 0:
        return pd.Series(index=data.index, dtype=float)
    
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(data_clean.T)
    
    result = pd.Series(index=data.index, dtype=float)
    result.loc[data_clean.index] = pc[:, 0]
    
    return result

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

# Example usage and testing functions
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
