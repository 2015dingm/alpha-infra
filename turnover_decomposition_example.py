"""
Practical Example: Alpha Strategy Turnover Decomposition

This example demonstrates how to decompose an alpha strategy into low and high turnover components.
"""

import sys
import os
sys.path.append('/home/mingd/Documents/Projects/alpha/alpha_infra')

import numpy as np
import pandas as pd
from alpha_pipeline import *
from turnover_decomposition import *

def generate_realistic_alpha_signal(returns_data: pd.DataFrame, 
                                  signal_type: str = 'momentum_with_noise') -> pd.Series:
    """Generate a realistic alpha signal with both persistent and transient components."""
    
    dates = returns_data.index
    
    if signal_type == 'momentum_with_noise':
        # Base momentum signal (persistent)
        momentum_window = 20
        market_returns = returns_data.mean(axis=1)
        momentum_signal = market_returns.rolling(window=momentum_window).mean()
        
        # Add high-frequency noise (transient)
        noise_factor = 0.3
        high_freq_noise = pd.Series(
            np.random.normal(0, 0.1, len(dates)) * noise_factor,
            index=dates
        )
        
        # Add some mean reversion component
        mean_reversion = -market_returns.rolling(window=5).mean() * 0.2
        
        # Combine components
        combined_signal = momentum_signal + high_freq_noise + mean_reversion
        
    elif signal_type == 'trend_following':
        # Trend following with occasional sharp reversals
        market_returns = returns_data.mean(axis=1)
        trend_signal = market_returns.rolling(window=30).mean()
        
        # Add sharp reversal signals (high turnover)
        reversal_probability = 0.05  # 5% chance per day
        reversal_signal = pd.Series(0.0, index=dates)
        
        for i, date in enumerate(dates[10:], 10):
            if np.random.random() < reversal_probability:
                reversal_signal.iloc[i:i+3] = -2.0 * trend_signal.iloc[i]
        
        combined_signal = trend_signal + reversal_signal
        
    else:
        # Simple noisy signal
        base_signal = pd.Series(np.random.normal(0, 0.5, len(dates)), index=dates)
        combined_signal = base_signal.rolling(window=10).mean() + np.random.normal(0, 0.2, len(dates))
    
    return combined_signal.fillna(0)

def demonstrate_decomposition_methods():
    """Demonstrate different decomposition methods with realistic data."""
    
    print("Alpha Strategy Turnover Decomposition Example")
    print("=" * 70)
    
    # Generate realistic market data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    assets = [f'STOCK_{i:02d}' for i in range(12)]
    
    # Create correlated returns
    market_factor = np.random.normal(0.0005, 0.015, 200)  # Slight positive drift
    returns_data = []
    
    for i in range(12):
        beta = np.random.uniform(0.8, 1.2)
        idiosyncratic = np.random.normal(0, 0.012, 200)
        asset_returns = beta * market_factor + idiosyncratic
        returns_data.append(asset_returns)
    
    returns_df = pd.DataFrame(
        np.array(returns_data).T,
        index=dates,
        columns=assets
    )
    
    print(f"Generated {returns_df.shape[0]} days of returns for {returns_df.shape[1]} assets")
    
    # Generate alpha signal with both persistent and transient components
    alpha_signal = generate_realistic_alpha_signal(returns_df, 'momentum_with_noise')
    
    print(f"Generated alpha signal with {len(alpha_signal)} observations")
    
    # Analyze original signal properties
    original_autocorr = signal_autocorrelation(alpha_signal, max_lags=10)
    original_backtest = backtest_signal(alpha_signal, returns_df)
    original_turnover = compute_signal_turnover(alpha_signal).sum()
    
    print(f"\nOriginal Signal Analysis:")
    print(f"  Sharpe ratio: {original_backtest.get('sharpe_ratio', 0):.3f}")
    print(f"  Annual return: {original_backtest.get('annual_return', 0):.2%}")
    print(f"  Total turnover: {original_turnover:.4f}")
    print(f"  Autocorrelation (lag 1): {original_autocorr.get(1, 0):.3f}")
    print(f"  Autocorrelation (lag 5): {original_autocorr.get(5, 0):.3f}")
    
    # Run comprehensive decomposition study
    decomposition_results = comprehensive_decomposition_study(
        alpha_signal, 
        returns_df,
        methods=['moving_average', 'frequency_domain', 'optimization_based']
    )
    
    return alpha_signal, returns_df, decomposition_results

def detailed_method_comparison():
    """Detailed comparison of a specific decomposition method."""
    
    print(f"\n{'='*70}")
    print("DETAILED METHOD ANALYSIS: OPTIMIZATION-BASED DECOMPOSITION")
    print(f"{'='*70}")
    
    # Generate data
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    assets = [f'ASSET_{i:02d}' for i in range(8)]
    
    returns_df = pd.DataFrame(
        np.random.normal(0.0003, 0.018, (150, 8)),
        index=dates,
        columns=assets
    )
    
    # Create signal with known structure
    persistent_component = pd.Series(
        np.sin(np.arange(150) * 0.1) + np.random.normal(0, 0.1, 150),
        index=dates
    )
    
    transient_component = pd.Series(
        np.random.normal(0, 0.5, 150),
        index=dates
    )
    
    # Combine with different weights to test decomposition
    weights = [0.8, 0.2]  # 80% persistent, 20% transient
    original_signal = weights[0] * persistent_component + weights[1] * transient_component
    
    print(f"Created synthetic signal:")
    print(f"  True persistent weight: {weights[0]:.1%}")
    print(f"  True transient weight: {weights[1]:.1%}")
    
    # Test optimization-based decomposition with different parameters
    parameter_sets = [
        {'target_turnover_ratio': 0.2, 'sharpe_preservation': 0.9},
        {'target_turnover_ratio': 0.3, 'sharpe_preservation': 0.8},
        {'target_turnover_ratio': 0.4, 'sharpe_preservation': 0.7},
    ]
    
    print(f"\nTesting optimization with different parameter sets:")
    print(f"{'Target T/O':<12} {'Sharpe Pres':<12} {'Actual T/O':<12} {'Actual Sharpe':<12} {'Quality':<8}")
    print("-" * 65)
    
    best_decomposition = None
    best_quality = 0
    
    for params in parameter_sets:
        low_sig, high_sig = optimization_based_decomposition(
            original_signal, returns_df, **params
        )
        
        analysis = analyze_decomposition_quality(
            original_signal, low_sig, high_sig, returns_df
        )
        
        # Quality score: balance of turnover reduction and Sharpe preservation
        quality_score = (
            analysis['sharpe_preservation_ratio'] * 
            (1 - analysis['turnover_reduction_ratio'])  # Reward lower turnover
        )
        
        print(f"{params['target_turnover_ratio']:<12.1%} "
              f"{params['sharpe_preservation']:<12.1%} "
              f"{analysis['turnover_reduction_ratio']:<12.1%} "
              f"{analysis['sharpe_preservation_ratio']:<12.1%} "
              f"{quality_score:<8.3f}")
        
        if quality_score > best_quality:
            best_quality = quality_score
            best_decomposition = (low_sig, high_sig, analysis)
    
    if best_decomposition:
        low_sig, high_sig, analysis = best_decomposition
        
        print(f"\nBest decomposition results:")
        print(f"  Low turnover Sharpe: {analysis['low_sharpe']:.3f}")
        print(f"  High turnover Sharpe: {analysis['high_sharpe']:.3f}")
        print(f"  Turnover reduction: {analysis['turnover_reduction_ratio']:.1%}")
        print(f"  Sharpe preservation: {analysis['sharpe_preservation_ratio']:.1%}")
        print(f"  Component correlation: {analysis['low_high_correlation']:.3f}")
    
    return best_decomposition

def practical_implementation_example():
    """Show how to implement this in practice."""
    
    print(f"\n{'='*70}")
    print("PRACTICAL IMPLEMENTATION GUIDE")
    print(f"{'='*70}")
    
    # Step 1: Start with existing alpha strategy
    print("STEP 1: Create your alpha strategy")
    
    # Generate sample data
    np.random.seed(789)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    returns_df = pd.DataFrame(
        np.random.normal(0.0008, 0.02, (100, 5)),
        index=dates,
        columns=assets
    )
    
    # Create alpha strategy using pipeline
    strategy_config = {
        'windowing': {'function': 'rolling', 'params': {'window': 15}},
        'preprocessing': {'function': 'z_score', 'params': {}},
        'reduction': {'function': 'cross_sectional_mean', 'params': {}},
        'postprocessing': {'function': 'rank', 'params': {}},
        'position_sizing': {'function': 'dollar_neutral', 'params': {}}
    }
    
    original_signal = run_alpha_pipeline(returns_df, strategy_config)
    print(f"✓ Created alpha signal with {len(original_signal)} observations")
    
    # Step 2: Analyze current strategy
    print("\nSTEP 2: Analyze your current strategy")
    original_metrics = backtest_signal(original_signal, returns_df)
    original_turnover = compute_signal_turnover(original_signal).sum()
    
    print(f"✓ Current Sharpe ratio: {original_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"✓ Current turnover: {original_turnover:.4f}")
    
    # Step 3: Apply decomposition
    print("\nSTEP 3: Apply turnover decomposition")
    low_turnover_signal, high_turnover_signal = optimization_based_decomposition(
        original_signal, 
        returns_df,
        target_turnover_ratio=0.25,  # Target 25% of original turnover
        sharpe_preservation=0.85     # Keep 85% of Sharpe ratio
    )
    
    # Step 4: Analyze results
    print("\nSTEP 4: Analyze decomposition results")
    analysis = analyze_decomposition_quality(
        original_signal, low_turnover_signal, high_turnover_signal, returns_df
    )
    
    print(f"✓ Low turnover component:")
    print(f"    Sharpe ratio: {analysis['low_sharpe']:.3f} (was {original_metrics.get('sharpe_ratio', 0):.3f})")
    print(f"    Turnover: {analysis['low_turnover']:.4f} (was {original_turnover:.4f})")
    print(f"    Reduction: {analysis['turnover_reduction_ratio']:.1%}")
    
    print(f"✓ High turnover component:")
    print(f"    Sharpe ratio: {analysis['high_sharpe']:.3f}")
    print(f"    Turnover: {analysis['high_turnover']:.4f}")
    
    # Step 5: Implementation strategy
    print("\nSTEP 5: Implementation strategy")
    print("✓ Deploy low turnover component with:")
    print("    • Lower transaction costs")
    print("    • Larger position sizes")
    print("    • Less frequent rebalancing")
    
    print("✓ Deploy high turnover component with:")
    print("    • Higher expected returns to offset costs")
    print("    • Smaller position sizes")
    print("    • More frequent rebalancing")
    print("    • Consider if net Sharpe > cost threshold")
    
    # Step 6: Monitoring
    print("\nSTEP 6: Ongoing monitoring")
    print("✓ Track performance of both components separately")
    print("✓ Adjust decomposition parameters based on market conditions")
    print("✓ Rebalance allocation between components based on capacity")
    
    return {
        'original_signal': original_signal,
        'low_turnover_signal': low_turnover_signal,
        'high_turnover_signal': high_turnover_signal,
        'analysis': analysis
    }

if __name__ == "__main__":
    print("Practical Alpha Strategy Turnover Decomposition")
    print("=" * 80)
    
    try:
        # Run demonstrations
        print("🚀 DEMONSTRATION OF DECOMPOSITION METHODS")
        original_signal, returns_data, decomp_results = demonstrate_decomposition_methods()
        
        print("\n🔍 DETAILED METHOD ANALYSIS") 
        detailed_results = detailed_method_comparison()
        
        print("\n💡 PRACTICAL IMPLEMENTATION GUIDE")
        practical_results = practical_implementation_example()
        
        # Create visualization
        if decomp_results:
            print(f"\n📊 Creating visualization...")
            plot_decomposition_results(
                original_signal,
                decomp_results,
                save_path='/home/mingd/Documents/Projects/alpha/alpha_infra/turnover_decomposition_results.png'
            )
        
        print(f"\n{'='*80}")
        print("TURNOVER DECOMPOSITION STUDY COMPLETE")
        print(f"{'='*80}")
        
        print(f"\n📚 Research Background:")
        print(f"This approach is based on the following key insights:")
        print(f"• Most alpha signals contain both persistent and transient components")
        print(f"• High transaction costs can erode profits from high-turnover strategies")
        print(f"• Optimal decomposition balances alpha capture vs. transaction costs")
        print(f"• Low turnover component captures most of the Sharpe ratio")
        print(f"• High turnover component may not be economically viable after costs")
        
        print(f"\n🎯 Key Findings:")
        if 'analysis' in practical_results:
            analysis = practical_results['analysis']
            print(f"• Turnover can be reduced by {analysis['turnover_reduction_ratio']:.0%}")
            print(f"• While maintaining {analysis['sharpe_preservation_ratio']:.0%} of Sharpe ratio")
            print(f"• Components have correlation of {analysis['low_high_correlation']:.2f}")
        
        print(f"\n📄 Generated Files:")
        print(f"• turnover_decomposition_results.png - Visual comparison of methods")
        
    except Exception as e:
        print(f"Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
