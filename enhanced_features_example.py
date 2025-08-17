"""
Enhanced Pipeline Features Example

This example demonstrates:
1) Windowing with shift parameter to avoid immediate reversal
2) Periodic rebalancing in backtesting
"""

import sys
import os
sys.path.append('/home/mingd/Documents/Projects/alpha/alpha_infra')

import numpy as np
import pandas as pd
from alpha_pipeline import *

def test_windowing_with_shift():
    """Demonstrate windowing with shift parameter."""
    
    print("="*70)
    print("WINDOWING WITH SHIFT PARAMETER")
    print("="*70)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    assets = [f'STOCK_{i:02d}' for i in range(8)]
    
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (100, 8)),
        index=dates,
        columns=assets
    )
    
    print(f"Generated {returns_data.shape[0]} days of returns for {returns_data.shape[1]} assets")
    
    # Compare strategies with different shift parameters
    shift_configs = {
        'no_shift': {
            'windowing': {'function': 'rolling', 'params': {'window': 20, 'shift': 0}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'dollar_neutral', 'params': {}}
        },
        
        'shift_1day': {
            'windowing': {'function': 'rolling', 'params': {'window': 20, 'shift': 1}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'dollar_neutral', 'params': {}}
        },
        
        'shift_3days': {
            'windowing': {'function': 'rolling', 'params': {'window': 20, 'shift': 3}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'dollar_neutral', 'params': {}}
        }
    }
    
    shift_results = {}
    benchmark_returns = returns_data.mean(axis=1)
    
    for config_name, config in shift_configs.items():
        print(f"\n{'─'*50}")
        print(f"TESTING: {config_name.upper().replace('_', ' ')}")
        print(f"{'─'*50}")
        
        try:
            signal, metrics = run_alpha_pipeline_with_backtest(
                returns_data=returns_data,
                pipeline_config=config,
                transaction_cost=0.001,
                benchmark_returns=benchmark_returns,
                strategy_name=config_name,
                print_results=True,
                plot_results=False
            )
            
            shift_results[config_name] = metrics
            
        except Exception as e:
            print(f"Error with {config_name}: {str(e)}")
    
    # Compare results
    if len(shift_results) > 1:
        print(f"\n{'='*70}")
        print("SHIFT PARAMETER COMPARISON")
        print(f"{'='*70}")
        compare_strategies(shift_results)
    
    return shift_results

def test_rebalancing_frequencies():
    """Demonstrate different rebalancing frequencies."""
    
    print(f"\n{'='*70}")
    print("PERIODIC REBALANCING COMPARISON")
    print(f"{'='*70}")
    
    # Generate sample data
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', periods=120, freq='D')  # ~4 months
    assets = [f'ASSET_{i:02d}' for i in range(12)]
    
    returns_data = pd.DataFrame(
        np.random.normal(0.0005, 0.018, (120, 12)),
        index=dates,
        columns=assets
    )
    
    print(f"Generated {returns_data.shape[0]} days of returns for {returns_data.shape[1]} assets")
    
    # Strategy configuration
    strategy_config = {
        'windowing': {'function': 'rolling', 'params': {'window': 15, 'shift': 1}},
        'preprocessing': {'function': 'robust_normalize', 'params': {}},
        'reduction': {'function': 'cross_sectional_rank', 'params': {}},
        'postprocessing': {'function': 'smooth', 'params': {'window': 3}},
        'position_sizing': {'function': 'dollar_neutral', 'params': {}}
    }
    
    # Test different rebalancing frequencies
    rebalancing_frequencies = ['D', 'W', 'M', '5', '10']  # Daily, Weekly, Monthly, Every 5 days, Every 10 days
    rebalance_results = {}
    benchmark_returns = returns_data.mean(axis=1)
    
    for freq in rebalancing_frequencies:
        freq_name = {
            'D': 'Daily',
            'W': 'Weekly', 
            'M': 'Monthly',
            '5': 'Every_5_Days',
            '10': 'Every_10_Days'
        }.get(freq, f'Every_{freq}_Days')
        
        print(f"\n{'─'*60}")
        print(f"REBALANCING FREQUENCY: {freq_name.upper().replace('_', ' ')}")
        print(f"{'─'*60}")
        
        try:
            signal, metrics = run_alpha_pipeline_with_backtest(
                returns_data=returns_data,
                pipeline_config=strategy_config,
                transaction_cost=0.0015,  # Higher transaction cost to show impact
                rebalance_frequency=freq,
                benchmark_returns=benchmark_returns,
                strategy_name=f"Rebal_{freq_name}",
                print_results=True,
                plot_results=False
            )
            
            rebalance_results[freq_name] = metrics
            
        except Exception as e:
            print(f"Error with {freq_name}: {str(e)}")
    
    # Compare rebalancing frequencies
    if len(rebalance_results) > 1:
        print(f"\n{'='*80}")
        print("REBALANCING FREQUENCY IMPACT ANALYSIS")
        print(f"{'='*80}")
        
        compare_strategies(rebalance_results)
        
        # Create detailed comparison table
        comparison_data = []
        for name, metrics in rebalance_results.items():
            if metrics:
                comparison_data.append({
                    'Strategy': name,
                    'Rebalance_Events': metrics.get('rebalancing_events', 0),
                    'Avg_Days_Between': f"{metrics.get('avg_days_between_rebalance', 0):.1f}",
                    'Sharpe_Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                    'Annual_Return': f"{metrics.get('annual_return', 0):.2%}",
                    'Max_Drawdown': f"{metrics.get('max_drawdown', 0):.2%}"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(f"\nDetailed Rebalancing Analysis:")
            print(comparison_df.to_string(index=False))
    
    return rebalance_results

def combined_features_example():
    """Example combining both shift and rebalancing features."""
    
    print(f"\n{'='*70}")
    print("COMBINED FEATURES: SHIFT + REBALANCING")
    print(f"{'='*70}")
    
    # Generate sample data
    np.random.seed(456)
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    assets = [f'STOCK_{i:02d}' for i in range(10)]
    
    returns_data = pd.DataFrame(
        np.random.normal(0.0008, 0.019, (150, 10)),
        index=dates,
        columns=assets
    )
    
    # Optimal configuration combining both features
    optimal_config = {
        'windowing': {'function': 'rolling', 'params': {'window': 25, 'shift': 2}},  # 25-day window, 2-day shift
        'preprocessing': {'function': 'winsorize', 'params': {'limits': (0.05, 0.05)}},
        'reduction': {'function': 'cross_sectional_rank', 'params': {}},
        'postprocessing': {'function': 'smooth', 'params': {'window': 3}},
        'position_sizing': {'function': 'position_limits', 'params': {'max_position': 0.12}}
    }
    
    benchmark_returns = returns_data.mean(axis=1)
    
    print(f"Testing optimal configuration:")
    print(f"• Window: 25 days with 2-day shift (avoids immediate reversal)")
    print(f"• Rebalancing: Weekly (reduces transaction costs)")
    print(f"• Transaction costs: 20 bps (realistic)")
    
    signal, metrics = run_alpha_pipeline_with_backtest(
        returns_data=returns_data,
        pipeline_config=optimal_config,
        transaction_cost=0.002,          # 20 bps transaction cost
        max_position=0.12,               # 12% max position per asset
        leverage_limit=1.0,              # 100% leverage limit
        rebalance_frequency='W',         # Weekly rebalancing
        benchmark_returns=benchmark_returns,
        strategy_name="Optimal Combined Strategy",
        print_results=True,
        plot_results=True,
        save_plots=True
    )
    
    print(f"\n✅ Combined features example completed!")
    
    return signal, metrics

if __name__ == "__main__":
    print("Enhanced Alpha Pipeline Features")
    print("=" * 80)
    
    try:
        # Test windowing with shift
        print("\n🔄 TESTING WINDOWING WITH SHIFT PARAMETER")
        shift_results = test_windowing_with_shift()
        
        # Test periodic rebalancing
        print("\n⏰ TESTING PERIODIC REBALANCING")
        rebalance_results = test_rebalancing_frequencies()
        
        # Combined features example
        print("\n🚀 TESTING COMBINED FEATURES")
        combined_signal, combined_metrics = combined_features_example()
        
        print(f"\n{'='*80}")
        print("ENHANCED FEATURES TESTING COMPLETE!")
        print(f"{'='*80}")
        
        print(f"\nKey insights:")
        print(f"✓ Shift parameter helps avoid immediate reversal signals")
        print(f"✓ Less frequent rebalancing can improve risk-adjusted returns")
        print(f"✓ Transaction costs have significant impact on performance")
        print(f"✓ Combined optimization can lead to better strategies")
        
        # Find best shift configuration
        if shift_results:
            best_shift = max([name for name, metrics in shift_results.items() if metrics], 
                           key=lambda x: shift_results[x].get('sharpe_ratio', -999))
            print(f"✓ Best shift configuration: {best_shift} (Sharpe: {shift_results[best_shift].get('sharpe_ratio', 0):.3f})")
        
        # Find best rebalancing frequency
        if rebalance_results:
            best_rebalance = max([name for name, metrics in rebalance_results.items() if metrics], 
                               key=lambda x: rebalance_results[x].get('sharpe_ratio', -999))
            print(f"✓ Best rebalancing frequency: {best_rebalance} (Sharpe: {rebalance_results[best_rebalance].get('sharpe_ratio', 0):.3f})")
        
    except Exception as e:
        print(f"Error running enhanced features test: {str(e)}")
        import traceback
        traceback.print_exc()
