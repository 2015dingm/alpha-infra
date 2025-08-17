"""
Testing Utilities and Data Generators

Comprehensive testing support for alpha pipeline development.
Use these utilities to test your functions as you build them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_realistic_returns(n_assets: int = 50, 
                              n_days: int = 500, 
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic stock return data for testing.
    
    Features:
    - Market factor with varying volatility regimes
    - Sector effects
    - Fat tails and volatility clustering
    - Missing data patterns
    """
    np.random.seed(seed)
    
    # Create date range (business days only)
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
    
    # Create asset names with sector structure
    sectors = ['TECH', 'FINL', 'HLTH', 'CONS', 'ENGY']
    assets = []
    sector_map = {}
    
    assets_per_sector = n_assets // len(sectors)
    for i, sector in enumerate(sectors):
        sector_assets = [f'{sector}_{j:03d}' for j in range(assets_per_sector)]
        assets.extend(sector_assets)
        sector_map[sector] = sector_assets
    
    # Add remaining assets to last sector
    remaining = n_assets - len(assets)
    for j in range(remaining):
        asset_name = f'{sectors[-1]}_{len(sector_map[sectors[-1]]) + j:03d}'
        assets.append(asset_name)
        sector_map[sectors[-1]].append(asset_name)
    
    # Generate market factors
    # 1. Market factor with regime switching
    market_vol_regimes = np.random.choice([0.01, 0.025], n_days, p=[0.8, 0.2])
    market_returns = np.random.normal(0.0005, market_vol_regimes)
    
    # 2. Sector factors
    sector_factors = {}
    for sector in sectors:
        sector_factors[sector] = np.random.normal(0, 0.008, n_days)
    
    # Generate individual asset returns
    returns_matrix = np.zeros((n_days, n_assets))
    
    for i, asset in enumerate(assets):
        # Determine asset's sector
        asset_sector = next(s for s, assets_list in sector_map.items() if asset in assets_list)
        
        # Random factor loadings
        market_beta = np.random.uniform(0.5, 1.5)
        sector_beta = np.random.uniform(0.3, 0.8)
        
        # Base returns from factors
        base_returns = (market_beta * market_returns + 
                       sector_beta * sector_factors[asset_sector])
        
        # Add idiosyncratic returns with volatility clustering
        idio_vol = np.random.uniform(0.015, 0.025)
        garch_vol = np.ones(n_days) * idio_vol
        
        # Simple GARCH-like volatility clustering
        for t in range(1, n_days):
            if abs(base_returns[t-1]) > 0.02:  # Shock condition
                garch_vol[t] = min(garch_vol[t-1] * 1.2, idio_vol * 2)
            else:
                garch_vol[t] = max(garch_vol[t-1] * 0.98, idio_vol * 0.8)
        
        idiosyncratic = np.random.normal(0, garch_vol)
        
        # Combine all effects
        returns_matrix[:, i] = base_returns + idiosyncratic
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns_matrix, index=dates, columns=assets)
    
    # Add some missing data patterns (realistic)
    # 1. Random missing data (holidays, suspensions)
    missing_mask = np.random.random((n_days, n_assets)) < 0.005
    
    # 2. Consecutive missing periods (IPOs, delistings)
    for asset_idx in range(n_assets):
        if np.random.random() < 0.1:  # 10% of assets have missing periods
            start_missing = np.random.randint(0, n_days//2)
            end_missing = start_missing + np.random.randint(5, 20)
            missing_mask[start_missing:end_missing, asset_idx] = True
    
    returns_df[missing_mask] = np.nan
    
    return returns_df, sector_map

def generate_simple_test_data(n_assets: int = 10, 
                             n_days: int = 100) -> pd.DataFrame:
    """
    Generate simple test data for unit testing.
    """
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    assets = [f'ASSET_{i:02d}' for i in range(n_assets)]
    
    # Simple random returns with known properties
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (n_days, n_assets)),
        index=dates,
        columns=assets
    )
    
    return returns

def generate_known_pattern_data(pattern_type: str = 'momentum') -> pd.DataFrame:
    """
    Generate data with known patterns for validation testing.
    
    Patterns:
    - 'momentum': Strong momentum pattern
    - 'mean_reversion': Mean reversion pattern  
    - 'volatility': Volatility clustering
    - 'trending': Strong trending pattern
    """
    np.random.seed(42)
    n_days = 200
    n_assets = 5
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    assets = [f'STOCK_{i}' for i in range(n_assets)]
    
    if pattern_type == 'momentum':
        # Create momentum pattern: past winners keep winning
        returns = np.zeros((n_days, n_assets))
        
        # Initialize first period
        returns[0, :] = np.random.normal(0, 0.02, n_assets)
        
        # Create momentum: positive autocorrelation
        for t in range(1, n_days):
            momentum_component = 0.3 * returns[t-1, :]  # 30% momentum
            noise = np.random.normal(0, 0.015, n_assets)
            returns[t, :] = momentum_component + noise
            
    elif pattern_type == 'mean_reversion':
        # Create mean reversion pattern
        returns = np.zeros((n_days, n_assets))
        prices = np.ones((n_days, n_assets))
        
        for t in range(1, n_days):
            # Mean reversion toward price of 1.0
            reversion_component = -0.1 * (prices[t-1, :] - 1.0)
            noise = np.random.normal(0, 0.02, n_assets)
            returns[t, :] = reversion_component + noise
            prices[t, :] = prices[t-1, :] * (1 + returns[t, :])
            
    elif pattern_type == 'trending':
        # Create strong trending pattern
        returns = np.zeros((n_days, n_assets))
        trends = np.random.choice([-0.002, 0.002], n_assets)  # Random up/down trends
        
        for t in range(n_days):
            returns[t, :] = trends + np.random.normal(0, 0.01, n_assets)
            
    else:
        # Default to simple random
        returns = np.random.normal(0, 0.02, (n_days, n_assets))
    
    return pd.DataFrame(returns, index=dates, columns=assets)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_signal_properties(signal: pd.Series, 
                              expected_mean: float = 0.0,
                              expected_std: Optional[float] = None,
                              tolerance: float = 0.01) -> Dict:
    """
    Validate statistical properties of generated signals.
    """
    if len(signal) == 0:
        return {'valid': False, 'error': 'Empty signal'}
    
    signal_clean = signal.dropna()
    if len(signal_clean) == 0:
        return {'valid': False, 'error': 'All NaN signal'}
    
    results = {
        'valid': True,
        'n_observations': len(signal_clean),
        'mean': signal_clean.mean(),
        'std': signal_clean.std(),
        'min': signal_clean.min(),
        'max': signal_clean.max(),
        'n_zeros': (signal_clean == 0).sum(),
        'checks': {}
    }
    
    # Check mean
    mean_diff = abs(results['mean'] - expected_mean)
    results['checks']['mean_ok'] = mean_diff < tolerance
    
    # Check for reasonable range
    results['checks']['range_ok'] = (abs(results['min']) < 10 and 
                                   abs(results['max']) < 10)
    
    # Check for non-constant signal
    results['checks']['non_constant'] = results['std'] > 1e-6
    
    # Check expected std if provided
    if expected_std is not None:
        std_diff = abs(results['std'] - expected_std)
        results['checks']['std_ok'] = std_diff < tolerance
    
    # Overall validation
    results['valid'] = all(results['checks'].values())
    
    return results

def validate_backtest_results(backtest_results: Dict,
                             min_sharpe: float = -2.0,
                             max_sharpe: float = 5.0) -> Dict:
    """
    Validate backtest results for reasonableness.
    """
    validation = {
        'valid': True,
        'checks': {},
        'warnings': []
    }
    
    # Check required fields
    required_fields = ['annual_return', 'annual_volatility', 'sharpe_ratio']
    for field in required_fields:
        if field not in backtest_results:
            validation['checks'][f'{field}_present'] = False
            validation['valid'] = False
        else:
            validation['checks'][f'{field}_present'] = True
    
    if not validation['valid']:
        return validation
    
    # Check reasonableness of metrics
    annual_return = backtest_results['annual_return']
    annual_vol = backtest_results['annual_volatility']
    sharpe = backtest_results['sharpe_ratio']
    
    # Annual return checks
    validation['checks']['return_reasonable'] = -1.0 < annual_return < 2.0
    if not validation['checks']['return_reasonable']:
        validation['warnings'].append(f"Extreme annual return: {annual_return:.2%}")
    
    # Volatility checks
    validation['checks']['vol_reasonable'] = 0.01 < annual_vol < 2.0
    if not validation['checks']['vol_reasonable']:
        validation['warnings'].append(f"Extreme volatility: {annual_vol:.2%}")
    
    # Sharpe ratio checks
    validation['checks']['sharpe_reasonable'] = min_sharpe < sharpe < max_sharpe
    if not validation['checks']['sharpe_reasonable']:
        validation['warnings'].append(f"Extreme Sharpe ratio: {sharpe:.2f}")
    
    # Portfolio returns checks
    if 'portfolio_returns' in backtest_results:
        portfolio_returns = backtest_results['portfolio_returns']
        daily_max = portfolio_returns.abs().max()
        validation['checks']['daily_returns_reasonable'] = daily_max < 0.5
        if not validation['checks']['daily_returns_reasonable']:
            validation['warnings'].append(f"Extreme daily return: {daily_max:.2%}")
    
    return validation

# =============================================================================
# COMPARISON AND BENCHMARKING FUNCTIONS
# =============================================================================

def create_benchmark_signals(returns_data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Create benchmark signals for comparison.
    """
    benchmarks = {}
    
    # Random signal
    np.random.seed(42)
    benchmarks['random'] = pd.Series(
        np.random.normal(0, 0.1, len(returns_data)),
        index=returns_data.index
    )
    
    # Momentum signal (simple)
    rolling_returns = returns_data.rolling(20).mean().mean(axis=1)
    benchmarks['momentum'] = rolling_returns.rank(pct=True) - 0.5
    
    # Mean reversion signal
    short_ma = returns_data.rolling(5).mean().mean(axis=1)
    long_ma = returns_data.rolling(20).mean().mean(axis=1)
    benchmarks['mean_reversion'] = -(short_ma - long_ma)
    
    # Volatility signal
    vol_signal = returns_data.rolling(20).std().mean(axis=1)
    benchmarks['volatility'] = (vol_signal - vol_signal.mean()) / vol_signal.std()
    
    return benchmarks

def compare_with_benchmarks(signal: pd.Series,
                          returns_data: pd.DataFrame,
                          backtest_func) -> pd.DataFrame:
    """
    Compare a signal with benchmark strategies.
    """
    benchmarks = create_benchmark_signals(returns_data)
    
    results = []
    
    # Test main signal
    try:
        main_result = backtest_func(signal, returns_data)
        results.append({
            'Strategy': 'Main Signal',
            'Sharpe': main_result.get('sharpe_ratio', 0),
            'Annual Return': main_result.get('annual_return', 0),
            'Annual Vol': main_result.get('annual_volatility', 0),
            'Status': 'Success'
        })
    except Exception as e:
        results.append({
            'Strategy': 'Main Signal',
            'Sharpe': 0,
            'Annual Return': 0,
            'Annual Vol': 0,
            'Status': f'Error: {str(e)[:50]}'
        })
    
    # Test benchmarks
    for name, benchmark_signal in benchmarks.items():
        try:
            result = backtest_func(benchmark_signal, returns_data)
            results.append({
                'Strategy': name.replace('_', ' ').title(),
                'Sharpe': result.get('sharpe_ratio', 0),
                'Annual Return': result.get('annual_return', 0),
                'Annual Vol': result.get('annual_volatility', 0),
                'Status': 'Success'
            })
        except Exception as e:
            results.append({
                'Strategy': name.replace('_', ' ').title(),
                'Sharpe': 0,
                'Annual Return': 0,
                'Annual Vol': 0,
                'Status': f'Error: {str(e)[:50]}'
            })
    
    return pd.DataFrame(results)

# =============================================================================
# VISUAL DEBUGGING TOOLS
# =============================================================================

def plot_signal_analysis(signal: pd.Series, 
                        returns_data: pd.DataFrame = None,
                        title: str = "Signal Analysis"):
    """
    Create comprehensive signal analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # 1. Signal over time
    ax1 = axes[0, 0]
    ax1.plot(signal.index, signal.values, linewidth=1)
    ax1.set_title('Signal Over Time')
    ax1.set_ylabel('Signal Strength')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Signal distribution
    ax2 = axes[0, 1]
    signal_clean = signal.dropna()
    ax2.hist(signal_clean, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(signal_clean.mean(), color='red', linestyle='--', 
               label=f'Mean: {signal_clean.mean():.4f}')
    ax2.set_title('Signal Distribution')
    ax2.set_xlabel('Signal Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Signal autocorrelation
    ax3 = axes[1, 0]
    if len(signal_clean) > 50:
        lags = range(1, min(21, len(signal_clean)//4))
        autocorr = [signal_clean.autocorr(lag=lag) for lag in lags]
        ax3.plot(lags, autocorr, 'o-')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title('Signal Autocorrelation')
        ax3.set_xlabel('Lag (days)')
        ax3.set_ylabel('Autocorrelation')
        ax3.grid(True, alpha=0.3)
    
    # 4. Signal vs Market (if returns provided)
    ax4 = axes[1, 1]
    if returns_data is not None:
        market_returns = returns_data.mean(axis=1)
        common_dates = signal.index.intersection(market_returns.index)
        if len(common_dates) > 10:
            signal_aligned = signal.loc[common_dates]
            market_aligned = market_returns.loc[common_dates]
            
            ax4.scatter(market_aligned, signal_aligned, alpha=0.6)
            ax4.set_xlabel('Market Return')
            ax4.set_ylabel('Signal')
            ax4.set_title('Signal vs Market Returns')
            ax4.grid(True, alpha=0.3)
            
            # Add correlation
            corr = signal_aligned.corr(market_aligned)
            ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=ax4.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def plot_development_progress(test_results: List[Dict]):
    """
    Plot development progress over time.
    """
    if not test_results:
        print("No test results to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(test_results)
    
    if 'date' not in df.columns:
        df['date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Development Progress', fontsize=16)
    
    # Tests passing over time
    ax1 = axes[0, 0]
    if 'tests_passed' in df.columns:
        ax1.plot(df['date'], df['tests_passed'], 'o-', color='green')
        ax1.set_title('Tests Passing')
        ax1.set_ylabel('Number of Tests')
        ax1.grid(True, alpha=0.3)
    
    # Sharpe ratio improvement
    ax2 = axes[0, 1]
    if 'sharpe_ratio' in df.columns:
        ax2.plot(df['date'], df['sharpe_ratio'], 'o-', color='blue')
        ax2.set_title('Sharpe Ratio Progress')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
    
    # Performance metrics
    ax3 = axes[1, 0]
    if 'execution_time' in df.columns:
        ax3.plot(df['date'], df['execution_time'], 'o-', color='orange')
        ax3.set_title('Execution Time')
        ax3.set_ylabel('Seconds')
        ax3.grid(True, alpha=0.3)
    
    # Code coverage
    ax4 = axes[1, 1]
    if 'code_coverage' in df.columns:
        ax4.plot(df['date'], df['code_coverage'], 'o-', color='purple')
        ax4.set_title('Code Coverage')
        ax4.set_ylabel('Coverage %')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo_testing_utilities():
    """
    Demonstrate how to use the testing utilities.
    """
    print("Testing Utilities Demo")
    print("=" * 40)
    
    # Generate test data
    print("1. Generating test data...")
    returns_data, sector_map = generate_realistic_returns(n_assets=20, n_days=200)
    print(f"✓ Generated {returns_data.shape} returns data")
    print(f"✓ Sectors: {list(sector_map.keys())}")
    
    # Create a simple signal for testing
    print("\n2. Creating test signal...")
    signal = returns_data.rolling(20).mean().mean(axis=1).rank(pct=True) - 0.5
    signal_validation = validate_signal_properties(signal)
    print(f"✓ Signal validation: {signal_validation['valid']}")
    if signal_validation['valid']:
        print(f"  Mean: {signal_validation['mean']:.4f}")
        print(f"  Std: {signal_validation['std']:.4f}")
    
    # Visual analysis
    print("\n3. Creating visual analysis...")
    plot_signal_analysis(signal, returns_data, "Test Signal Analysis")
    
    # Benchmark comparison (would need backtest function)
    print("\n4. Benchmark signals created:")
    benchmarks = create_benchmark_signals(returns_data)
    for name in benchmarks.keys():
        print(f"  ✓ {name}")
    
    print("\n✅ Testing utilities demo complete!")
    
    return returns_data, signal, benchmarks

if __name__ == "__main__":
    demo_testing_utilities()
