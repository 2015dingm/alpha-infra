"""
Minimal Alpha Pipeline Starter Template

This is a minimal, step-by-step implementation following the development roadmap.
Start here and build incrementally according to the roadmap.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

# =============================================================================
# PHASE 1: FOUNDATION - START HERE
# =============================================================================

def load_returns_data(data_source, date_col='date', validate=True):
    """
    STEP 1: Load and validate returns data
    
    This is your starting point. Get this working first!
    """
    if isinstance(data_source, str):
        # Load from file
        if data_source.endswith('.csv'):
            data = pd.read_csv(data_source, index_col=date_col, parse_dates=True)
        else:
            raise ValueError("Only CSV files supported initially")
    elif isinstance(data_source, pd.DataFrame):
        # Use provided DataFrame
        data = data_source.copy()
    else:
        raise ValueError("data_source must be file path or DataFrame")
    
    if validate:
        validation_report = validate_data(data)
        if not validation_report['is_valid']:
            warnings.warn(f"Data validation issues: {validation_report['issues']}")
    
    return data

def validate_data(data, min_periods=50, max_missing_pct=0.2):
    """
    STEP 2: Data validation - critical for data quality
    """
    issues = []
    
    # Check basic structure
    if len(data) < min_periods:
        issues.append(f"Insufficient data: {len(data)} < {min_periods}")
    
    # Check for missing data
    missing_pct = data.isnull().sum() / len(data)
    problematic_assets = missing_pct[missing_pct > max_missing_pct]
    if len(problematic_assets) > 0:
        issues.append(f"High missing data in {len(problematic_assets)} assets")
    
    # Check for extreme values (basic check)
    extreme_returns = (data.abs() > 0.5).sum()  # >50% daily return
    if extreme_returns.sum() > 0:
        issues.append(f"Extreme returns detected: {extreme_returns.sum()} occurrences")
    
    # Check data types
    if not all(data.dtypes == 'float64'):
        issues.append("Non-numeric data detected")
    
    return {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'n_assets': len(data.columns),
        'n_periods': len(data),
        'missing_pct': missing_pct.mean()
    }

def rolling_window_basic(data, window):
    """
    STEP 3: Basic rolling window - start simple
    """
    if window <= 0 or window > len(data):
        raise ValueError(f"Invalid window size: {window}")
    
    return data.rolling(window=window, min_periods=window//2)

def z_score_normalize_basic(data):
    """
    STEP 4: Basic normalization - cross-sectional z-score
    """
    # Cross-sectional (across assets at each time point)
    mean = data.mean(axis=1)
    std = data.std(axis=1)
    
    # Broadcast to match data shape
    normalized = data.sub(mean, axis=0).div(std + 1e-8, axis=0)
    
    return normalized.fillna(0)

# =============================================================================
# TESTING FUNCTIONS - IMPLEMENT THESE ALONGSIDE MAIN FUNCTIONS
# =============================================================================

def test_basic_functions():
    """
    Basic tests for Phase 1 functions
    Run this after implementing each function!
    """
    print("Testing Phase 1 Functions")
    print("=" * 40)
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    assets = ['AAPL', 'MSFT', 'GOOGL']
    test_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (100, 3)),
        index=dates,
        columns=assets
    )
    
    # Test 1: Data loading
    print("Test 1: Data Loading")
    try:
        loaded_data = load_returns_data(test_data)
        print(f"✓ Loaded data: {loaded_data.shape}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Data validation
    print("\nTest 2: Data Validation")
    try:
        validation = validate_data(test_data)
        print(f"✓ Validation: {validation['is_valid']}")
        print(f"  Assets: {validation['n_assets']}, Periods: {validation['n_periods']}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 3: Rolling window
    print("\nTest 3: Rolling Window")
    try:
        windowed = rolling_window_basic(test_data, window=20)
        windowed_mean = windowed.mean()
        print(f"✓ Windowed data: {windowed_mean.shape}")
        print(f"  Non-null values: {windowed_mean.count().sum()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 4: Normalization
    print("\nTest 4: Z-Score Normalization")
    try:
        normalized = z_score_normalize_basic(test_data)
        print(f"✓ Normalized data: {normalized.shape}")
        print(f"  Cross-sectional mean: {normalized.mean(axis=1).abs().mean():.6f} (should be ~0)")
        print(f"  Cross-sectional std: {normalized.std(axis=1).mean():.3f} (should be ~1)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    return test_data

# =============================================================================
# PHASE 2: SIGNAL GENERATION - ADD THESE NEXT
# =============================================================================

def cross_sectional_mean_basic(data):
    """
    STEP 5: Cross-sectional mean - market-relative signal
    Add this in Phase 2
    """
    return data.mean(axis=1)

def cross_sectional_rank_basic(data):
    """
    STEP 6: Cross-sectional ranking
    Add this in Phase 2
    """
    return data.rank(axis=1, pct=True) - 0.5  # Center around 0

def simple_pipeline_v1(data, window=20):
    """
    STEP 7: First complete pipeline
    Add this in Phase 2 - combines all Phase 1 & 2 functions
    """
    # Validate input
    validation = validate_data(data)
    if not validation['is_valid']:
        raise ValueError(f"Invalid data: {validation['issues']}")
    
    # Step 1: Windowing
    windowed_data = rolling_window_basic(data, window)
    
    # Step 2: Preprocessing (on original data for now)
    normalized_data = z_score_normalize_basic(data)
    
    # Step 3: Reduction
    signal = cross_sectional_mean_basic(normalized_data)
    
    # Step 4: Position sizing (simple dollar neutral)
    positions = signal - signal.mean()
    
    return positions

def test_phase2_functions(test_data):
    """
    Tests for Phase 2 functions
    """
    print("\n" + "=" * 40)
    print("Testing Phase 2 Functions")
    print("=" * 40)
    
    # Test cross-sectional functions
    print("Test 5: Cross-Sectional Mean")
    try:
        cs_mean = cross_sectional_mean_basic(test_data)
        print(f"✓ Cross-sectional mean: {len(cs_mean)} observations")
        print(f"  Range: [{cs_mean.min():.4f}, {cs_mean.max():.4f}]")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 6: Cross-Sectional Rank")
    try:
        cs_rank = cross_sectional_rank_basic(test_data)
        print(f"✓ Cross-sectional rank: {cs_rank.shape}")
        print(f"  Mean rank: {cs_rank.mean().mean():.6f} (should be ~0)")
        print(f"  Rank range: [{cs_rank.min().min():.3f}, {cs_rank.max().max():.3f}]")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTest 7: Simple Pipeline v1")
    try:
        signal = simple_pipeline_v1(test_data, window=20)
        print(f"✓ Pipeline signal: {len(signal)} observations")
        print(f"  Signal mean: {signal.mean():.6f} (should be ~0)")
        print(f"  Signal std: {signal.std():.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        
    return signal

# =============================================================================
# PHASE 3: BASIC BACKTESTING - ADD THESE THIRD
# =============================================================================

def calculate_portfolio_returns_basic(signal, returns_data, transaction_cost=0.001):
    """
    STEP 8: Basic portfolio return calculation
    Add this in Phase 3
    """
    # Align signal with returns
    common_dates = signal.index.intersection(returns_data.index)
    if len(common_dates) < 10:
        raise ValueError("Insufficient overlapping dates")
    
    signal_aligned = signal.loc[common_dates]
    returns_aligned = returns_data.loc[common_dates]
    
    # Simple equal-weight allocation based on signal
    n_assets = len(returns_aligned.columns)
    positions = pd.DataFrame(
        index=signal_aligned.index, 
        columns=returns_aligned.columns,
        dtype=float
    )
    
    # Distribute signal equally across assets
    for date in signal_aligned.index:
        signal_strength = signal_aligned.loc[date]
        if pd.notna(signal_strength) and signal_strength != 0:
            weight_per_asset = signal_strength / n_assets
            positions.loc[date] = weight_per_asset
        else:
            positions.loc[date] = 0.0
    
    # Calculate returns
    portfolio_returns = (positions.shift(1) * returns_aligned).sum(axis=1)
    
    # Apply transaction costs (simplified)
    position_changes = positions.diff().abs().sum(axis=1)
    costs = position_changes * transaction_cost
    
    net_returns = portfolio_returns - costs
    
    return net_returns.fillna(0)

def calculate_basic_metrics_basic(portfolio_returns):
    """
    STEP 9: Basic performance metrics
    Add this in Phase 3
    """
    if len(portfolio_returns) == 0:
        return {}
    
    returns = portfolio_returns.dropna()
    
    # Basic metrics
    total_return = returns.sum()
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'n_periods': len(returns)
    }

def simple_backtest_v1(signal, returns_data, transaction_cost=0.001):
    """
    STEP 10: First complete backtest
    Add this in Phase 3
    """
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns_basic(
        signal, returns_data, transaction_cost
    )
    
    # Calculate metrics
    metrics = calculate_basic_metrics_basic(portfolio_returns)
    
    # Add the return series for plotting
    metrics['portfolio_returns'] = portfolio_returns
    metrics['cumulative_returns'] = portfolio_returns.cumsum()
    
    return metrics

def test_phase3_functions(signal, test_data):
    """
    Tests for Phase 3 functions
    """
    print("\n" + "=" * 40)
    print("Testing Phase 3 Functions")
    print("=" * 40)
    
    print("Test 8: Portfolio Returns")
    try:
        portfolio_returns = calculate_portfolio_returns_basic(signal, test_data)
        print(f"✓ Portfolio returns: {len(portfolio_returns)} observations")
        print(f"  Mean daily return: {portfolio_returns.mean():.6f}")
        print(f"  Daily volatility: {portfolio_returns.std():.4f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\nTest 9: Basic Metrics")
    try:
        metrics = calculate_basic_metrics_basic(portfolio_returns)
        print(f"✓ Calculated {len(metrics)} metrics")
        print(f"  Annual return: {metrics.get('annual_return', 0):.2%}")
        print(f"  Annual volatility: {metrics.get('annual_volatility', 0):.2%}")
        print(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Win rate: {metrics.get('win_rate', 0):.2%}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\nTest 10: Complete Backtest")
    try:
        backtest_results = simple_backtest_v1(signal, test_data)
        print(f"✓ Backtest complete: {backtest_results.get('n_periods', 0)} periods")
        print(f"  Sharpe ratio: {backtest_results.get('sharpe_ratio', 0):.3f}")
        
        # Simple plot
        if 'cumulative_returns' in backtest_results:
            plt.figure(figsize=(10, 4))
            plt.plot(backtest_results['cumulative_returns'])
            plt.title('Cumulative Returns')
            plt.grid(True)
            plt.show()
            
    except Exception as e:
        print(f"✗ Error: {e}")

# =============================================================================
# MAIN EXECUTION - YOUR DEVELOPMENT WORKFLOW
# =============================================================================

def main():
    """
    Main function - your development workflow
    
    Run this to test your implementation as you build it!
    """
    print("Alpha Pipeline Starter Template")
    print("Follow the development roadmap step by step!")
    print("=" * 60)
    
    # Phase 1: Test basic functions
    test_data = test_basic_functions()
    
    # Phase 2: Test signal generation (uncomment when Phase 2 is complete)
    signal = test_phase2_functions(test_data)
    
    # Phase 3: Test backtesting (uncomment when Phase 3 is complete)
    test_phase3_functions(signal, test_data)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. Make sure all tests pass ✓")
    print("2. Test with your real data")
    print("3. Move to Phase 4: Advanced features")
    print("4. See development_roadmap.py for detailed guidance")

if __name__ == "__main__":
    main()
