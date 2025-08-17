"""
Alpha Strategy Turnover Decomposition

This module implements methods to decompose an alpha strategy into:
1. Low turnover component (maintains most of the Sharpe ratio)  
2. High turnover component (captures remaining alpha)

Based on research by:
- Garleanu & Pedersen (2013): Dynamic Trading with Predictable Returns and Transaction Costs
- Almgren & Chriss (1999): Optimal Execution of Portfolio Transactions
- Grinold & Kahn (1999): Active Portfolio Management
"""

import sys
import os
sys.path.append('/home/mingd/Documents/Projects/alpha/alpha_infra')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.decomposition import PCA
from alpha_pipeline import *

def compute_signal_turnover(signal: pd.Series, method: str = 'absolute') -> pd.Series:
    """
    Compute turnover of a trading signal.
    
    Parameters:
    -----------
    signal : pd.Series
        Trading signal
    method : str
        'absolute': sum of absolute position changes
        'squared': sum of squared position changes
        
    Returns:
    --------
    pd.Series
        Rolling turnover
    """
    
    position_changes = signal.diff().abs() if method == 'absolute' else signal.diff() ** 2
    return position_changes

def signal_autocorrelation(signal: pd.Series, max_lags: int = 20) -> pd.Series:
    """Compute autocorrelation of signal to understand persistence."""
    
    autocorr = pd.Series(index=range(max_lags + 1), dtype=float)
    
    for lag in range(max_lags + 1):
        if lag == 0:
            autocorr[lag] = 1.0
        else:
            shifted = signal.shift(lag)
            valid_idx = signal.index.intersection(shifted.index)
            if len(valid_idx) > lag + 10:  # Need sufficient overlap
                autocorr[lag] = signal.loc[valid_idx].corr(shifted.loc[valid_idx])
            else:
                autocorr[lag] = np.nan
    
    return autocorr.dropna()

def frequency_domain_decomposition(signal: pd.Series, 
                                 cutoff_frequency: float = 0.1) -> tuple:
    """
    Decompose signal using frequency domain filtering.
    
    Parameters:
    -----------
    signal : pd.Series
        Original trading signal
    cutoff_frequency : float
        Cutoff frequency for low-pass filter (0 to 0.5)
        
    Returns:
    --------
    tuple
        (low_frequency_signal, high_frequency_signal)
    """
    
    try:
        from scipy.signal import butter, filtfilt
        
        # Design low-pass Butterworth filter
        nyquist = 0.5  # Normalized frequency
        normal_cutoff = cutoff_frequency / nyquist
        b, a = butter(2, normal_cutoff, btype='low', analog=False)
        
        # Apply filter (forward and backward to avoid phase shift)
        signal_values = signal.dropna().values
        low_freq_values = filtfilt(b, a, signal_values)
        
        # Create low frequency signal
        low_freq_signal = pd.Series(low_freq_values, index=signal.dropna().index)
        
        # High frequency is the residual
        high_freq_signal = signal - low_freq_signal.reindex(signal.index, fill_value=0)
        
        return low_freq_signal, high_freq_signal
        
    except ImportError:
        # Fallback to simple moving average approach
        return moving_average_decomposition(signal, window=int(1/cutoff_frequency))

def moving_average_decomposition(signal: pd.Series, 
                               window: int = 20) -> tuple:
    """
    Simple moving average based decomposition.
    
    Parameters:
    -----------
    signal : pd.Series
        Original trading signal
    window : int
        Window for moving average (low frequency component)
        
    Returns:
    --------
    tuple
        (low_frequency_signal, high_frequency_signal)
    """
    
    # Low frequency: smoothed signal
    low_freq_signal = signal.rolling(window=window, center=True).mean()
    
    # High frequency: residual
    high_freq_signal = signal - low_freq_signal
    
    return low_freq_signal.fillna(0), high_freq_signal.fillna(0)

def pca_based_decomposition(signal: pd.Series, 
                          returns_data: pd.DataFrame,
                          n_components: int = 2) -> tuple:
    """
    Use PCA on signal and returns to find low/high turnover components.
    
    Parameters:
    -----------
    signal : pd.Series
        Original trading signal
    returns_data : pd.DataFrame
        Asset returns data
    n_components : int
        Number of principal components
        
    Returns:
    --------
    tuple
        (low_turnover_signal, high_turnover_signal)
    """
    
    # Align signal with returns
    common_dates = signal.index.intersection(returns_data.index)
    signal_aligned = signal.loc[common_dates]
    returns_aligned = returns_data.loc[common_dates]
    
    # Create feature matrix: [signal, lagged_signal, market_returns]
    features = pd.DataFrame(index=common_dates)
    features['signal'] = signal_aligned
    features['signal_lag1'] = signal_aligned.shift(1)
    features['signal_lag2'] = signal_aligned.shift(2)
    features['market_return'] = returns_aligned.mean(axis=1)
    features['market_vol'] = returns_aligned.std(axis=1)
    
    # Remove NaN
    features_clean = features.dropna()
    
    if len(features_clean) < 10:
        # Fallback to simple decomposition
        return moving_average_decomposition(signal)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(features_clean.values)
    
    # First component usually captures persistent behavior (low turnover)
    # Second component captures transient behavior (high turnover)
    low_turnover_weights = pca_components[:, 0]
    
    # Reconstruct low turnover signal
    low_turnover_signal = pd.Series(
        low_turnover_weights * np.std(signal_aligned) / np.std(low_turnover_weights),
        index=features_clean.index
    )
    
    # High turnover is residual
    high_turnover_signal = signal_aligned.reindex(low_turnover_signal.index) - low_turnover_signal
    
    # Extend to full index
    low_full = low_turnover_signal.reindex(signal.index, fill_value=0)
    high_full = high_turnover_signal.reindex(signal.index, fill_value=0)
    
    return low_full, high_full

def optimization_based_decomposition(signal: pd.Series,
                                   returns_data: pd.DataFrame,
                                   target_turnover_ratio: float = 0.3,
                                   sharpe_preservation: float = 0.8) -> tuple:
    """
    Optimize decomposition to achieve target turnover while preserving Sharpe ratio.
    
    Parameters:
    -----------
    signal : pd.Series
        Original trading signal
    returns_data : pd.DataFrame
        Asset returns data
    target_turnover_ratio : float
        Target ratio of low-turnover component turnover to original turnover
    sharpe_preservation : float
        Minimum fraction of original Sharpe ratio to preserve in low-turnover component
        
    Returns:
    --------
    tuple
        (optimized_low_turnover_signal, high_turnover_signal)
    """
    
    # Compute original metrics
    original_backtest = backtest_signal(signal, returns_data)
    if not original_backtest or 'portfolio_returns' not in original_backtest:
        return moving_average_decomposition(signal)
    
    original_sharpe = original_backtest.get('sharpe_ratio', 0)
    original_turnover = compute_signal_turnover(signal).sum()
    
    target_turnover = target_turnover_ratio * original_turnover
    target_sharpe = sharpe_preservation * original_sharpe
    
    # Define optimization objective
    def objective(smoothing_params):
        alpha, window = smoothing_params
        window = max(2, int(window))
        
        # Create low turnover signal using exponential smoothing + moving average
        ema_signal = signal.ewm(alpha=alpha).mean()
        smooth_signal = ema_signal.rolling(window=window).mean().fillna(ema_signal)
        
        # Compute metrics for low turnover component
        low_turnover_backtest = backtest_signal(smooth_signal, returns_data)
        if not low_turnover_backtest:
            return 1e6  # Large penalty for failed backtest
        
        low_sharpe = low_turnover_backtest.get('sharpe_ratio', 0)
        low_turnover = compute_signal_turnover(smooth_signal).sum()
        
        # Multi-objective: minimize turnover while maintaining Sharpe ratio
        turnover_penalty = abs(low_turnover - target_turnover) / (target_turnover + 1e-6)
        sharpe_penalty = max(0, target_sharpe - low_sharpe) / (target_sharpe + 1e-6)
        
        return turnover_penalty + 10 * sharpe_penalty  # Weight Sharpe preservation heavily
    
    # Optimization constraints
    bounds = [(0.01, 0.5), (2, 50)]  # (alpha, window)
    
    try:
        # Use multiple starting points
        best_result = None
        best_value = float('inf')
        
        for alpha_start in [0.05, 0.1, 0.2]:
            for window_start in [5, 10, 20]:
                initial_guess = [alpha_start, window_start]
                
                result = optimize.minimize(
                    objective, 
                    initial_guess, 
                    bounds=bounds, 
                    method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_value:
                    best_result = result
                    best_value = result.fun
        
        if best_result is not None:
            alpha_opt, window_opt = best_result.x
            window_opt = max(2, int(window_opt))
            
            # Create optimized low turnover signal
            ema_signal = signal.ewm(alpha=alpha_opt).mean()
            low_turnover_signal = ema_signal.rolling(window=window_opt).mean().fillna(ema_signal)
            
            # High turnover is residual
            high_turnover_signal = signal - low_turnover_signal
            
            return low_turnover_signal, high_turnover_signal
        
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    # Fallback to simple method
    return moving_average_decomposition(signal)

def analyze_decomposition_quality(original_signal: pd.Series,
                                low_turnover_signal: pd.Series,
                                high_turnover_signal: pd.Series,
                                returns_data: pd.DataFrame) -> dict:
    """
    Analyze the quality of signal decomposition.
    
    Returns:
    --------
    dict
        Analysis results including turnover ratios, Sharpe ratios, etc.
    """
    
    # Backtest all three signals
    original_backtest = backtest_signal(original_signal, returns_data)
    low_backtest = backtest_signal(low_turnover_signal, returns_data)
    high_backtest = backtest_signal(high_turnover_signal, returns_data)
    
    # Compute turnovers
    original_turnover = compute_signal_turnover(original_signal).sum()
    low_turnover = compute_signal_turnover(low_turnover_signal).sum()
    high_turnover = compute_signal_turnover(high_turnover_signal).sum()
    
    # Extract metrics
    def safe_get(backtest, metric, default=0):
        return backtest.get(metric, default) if backtest else default
    
    analysis = {
        # Performance metrics
        'original_sharpe': safe_get(original_backtest, 'sharpe_ratio'),
        'low_sharpe': safe_get(low_backtest, 'sharpe_ratio'),
        'high_sharpe': safe_get(high_backtest, 'sharpe_ratio'),
        
        'original_return': safe_get(original_backtest, 'annual_return'),
        'low_return': safe_get(low_backtest, 'annual_return'),
        'high_return': safe_get(high_backtest, 'annual_return'),
        
        # Turnover metrics
        'original_turnover': original_turnover,
        'low_turnover': low_turnover,
        'high_turnover': high_turnover,
        
        # Ratios
        'turnover_reduction_ratio': low_turnover / (original_turnover + 1e-8),
        'sharpe_preservation_ratio': safe_get(low_backtest, 'sharpe_ratio') / (safe_get(original_backtest, 'sharpe_ratio') + 1e-8),
        
        # Signal properties
        'signal_correlation': original_signal.corr(low_turnover_signal + high_turnover_signal),
        'low_high_correlation': low_turnover_signal.corr(high_turnover_signal),
        'low_weight': low_turnover_signal.var() / (low_turnover_signal.var() + high_turnover_signal.var() + 1e-8),
    }
    
    return analysis

def comprehensive_decomposition_study(signal: pd.Series,
                                    returns_data: pd.DataFrame,
                                    methods: list = None) -> dict:
    """
    Compare multiple decomposition methods.
    
    Parameters:
    -----------
    signal : pd.Series
        Original trading signal
    returns_data : pd.DataFrame
        Asset returns data
    methods : list, optional
        List of methods to compare
        
    Returns:
    --------
    dict
        Results from all decomposition methods
    """
    
    if methods is None:
        methods = ['moving_average', 'frequency_domain', 'pca_based', 'optimization_based']
    
    results = {}
    
    print("Comprehensive Signal Decomposition Study")
    print("=" * 60)
    print(f"Original signal: {len(signal)} observations")
    
    # Compute original signal statistics
    original_backtest = backtest_signal(signal, returns_data)
    original_turnover = compute_signal_turnover(signal).sum()
    
    if original_backtest:
        print(f"Original Sharpe ratio: {original_backtest.get('sharpe_ratio', 0):.3f}")
        print(f"Original turnover: {original_turnover:.4f}")
    
    print(f"\nTesting {len(methods)} decomposition methods:")
    print("-" * 40)
    
    for method in methods:
        print(f"\n{method.upper().replace('_', ' ')}:")
        
        try:
            if method == 'moving_average':
                low_sig, high_sig = moving_average_decomposition(signal, window=15)
            elif method == 'frequency_domain':
                low_sig, high_sig = frequency_domain_decomposition(signal, cutoff_frequency=0.08)
            elif method == 'pca_based':
                low_sig, high_sig = pca_based_decomposition(signal, returns_data)
            elif method == 'optimization_based':
                low_sig, high_sig = optimization_based_decomposition(signal, returns_data)
            else:
                print(f"Unknown method: {method}")
                continue
            
            # Analyze decomposition quality
            analysis = analyze_decomposition_quality(signal, low_sig, high_sig, returns_data)
            results[method] = {
                'low_turnover_signal': low_sig,
                'high_turnover_signal': high_sig,
                'analysis': analysis
            }
            
            # Print key metrics
            print(f"  Sharpe preservation: {analysis['sharpe_preservation_ratio']:.2%}")
            print(f"  Turnover reduction: {analysis['turnover_reduction_ratio']:.2%}")
            print(f"  Low-high correlation: {analysis['low_high_correlation']:.3f}")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return results

def plot_decomposition_results(original_signal: pd.Series,
                             decomposition_results: dict,
                             save_path: str = None):
    """Plot comparison of decomposition methods."""
    
    try:
        n_methods = len(decomposition_results)
        fig, axes = plt.subplots(n_methods + 1, 2, figsize=(15, 4 * (n_methods + 1)))
        
        if n_methods == 0:
            return
        
        # Ensure axes is 2D
        if n_methods == 1:
            axes = axes.reshape(2, 2)
        
        # Plot original signal
        axes[0, 0].plot(original_signal.index, original_signal.values, 'k-', linewidth=2, label='Original')
        axes[0, 0].set_title('Original Signal')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot original turnover
        original_turnover = compute_signal_turnover(original_signal)
        axes[0, 1].plot(original_turnover.index, original_turnover.values, 'k-', linewidth=1)
        axes[0, 1].set_title('Original Signal Turnover')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot each decomposition method
        for i, (method, result) in enumerate(decomposition_results.items()):
            row = i + 1
            
            low_sig = result['low_turnover_signal']
            high_sig = result['high_turnover_signal']
            analysis = result['analysis']
            
            # Plot signals
            axes[row, 0].plot(original_signal.index, original_signal.values, 'k--', alpha=0.5, label='Original')
            axes[row, 0].plot(low_sig.index, low_sig.values, 'b-', linewidth=2, label='Low Turnover')
            axes[row, 0].plot(high_sig.index, high_sig.values, 'r-', linewidth=1, alpha=0.7, label='High Turnover')
            axes[row, 0].set_title(f'{method.title().replace("_", " ")} Decomposition\n'
                                 f'Sharpe: {analysis["sharpe_preservation_ratio"]:.1%}, '
                                 f'Turnover: {analysis["turnover_reduction_ratio"]:.1%}')
            axes[row, 0].legend()
            axes[row, 0].grid(True, alpha=0.3)
            
            # Plot turnover comparison
            low_turnover = compute_signal_turnover(low_sig)
            high_turnover = compute_signal_turnover(high_sig)
            
            axes[row, 1].plot(low_turnover.index, low_turnover.values, 'b-', label='Low Turnover')
            axes[row, 1].plot(high_turnover.index, high_turnover.values, 'r-', alpha=0.7, label='High Turnover')
            axes[row, 1].set_title(f'Turnover Components\n'
                                 f'Low: {analysis["low_turnover"]:.3f}, High: {analysis["high_turnover"]:.3f}')
            axes[row, 1].legend()
            axes[row, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    print("Alpha Strategy Turnover Decomposition")
    print("=" * 60)
    
    # This will be populated with actual examples when imported
    print("This module provides:")
    print("• Moving average based decomposition")  
    print("• Frequency domain decomposition")
    print("• PCA-based decomposition") 
    print("• Optimization-based decomposition")
    print("• Comprehensive quality analysis")
    print("• Visualization tools")
    
    print(f"\nKey research references:")
    print("• Garleanu & Pedersen (2013): Dynamic Trading with Predictable Returns")
    print("• Almgren & Chriss (1999): Optimal Execution of Portfolio Transactions") 
    print("• Grinold & Kahn (1999): Active Portfolio Management")
    print("• Lynch & Balduzzi (2000): Transaction Costs and Predictability")
