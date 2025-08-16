"""
Simple Example: Alpha Pipeline with Integrated Backtesting

This example shows how to use the alpha pipeline with integrated backtesting
functionality - all using functions, no classes required!
"""

import sys
import os
sys.path.append('/home/mingd/Documents/Projects/alpha/alpha_infra')

import numpy as np
import pandas as pd
from alpha_pipeline import *

def simple_backtest_example():
    """Simple example showing integrated backtesting."""
    
    print("Alpha Pipeline with Integrated Backtesting")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample market data...")
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    assets = [f'STOCK_{i:02d}' for i in range(10)]
    
    # Create realistic returns with some correlation
    market_factor = np.random.normal(0, 0.012, 150)
    returns_data = []
    
    for i in range(10):
        beta = np.random.uniform(0.7, 1.3)
        idiosyncratic = np.random.normal(0, 0.015, 150)
        asset_returns = beta * market_factor + idiosyncratic
        returns_data.append(asset_returns)
    
    returns_df = pd.DataFrame(
        np.array(returns_data).T,
        index=dates,
        columns=assets
    )
    
    print(f"✓ Generated {returns_df.shape[0]} days of returns for {returns_df.shape[1]} assets")
    
    # 2. Define strategy configuration
    print("\n2. Defining alpha strategy...")
    
    strategy_config = {
        'windowing': {'function': 'rolling', 'params': {'window': 20}},
        'preprocessing': {'function': 'z_score', 'params': {}},
        'reduction': {'function': 'cross_sectional_mean', 'params': {}},
        'postprocessing': {'function': 'rank', 'params': {}},
        'position_sizing': {'function': 'dollar_neutral', 'params': {}}
    }
    
    print("✓ Strategy defined: 20-day momentum with cross-sectional ranking")
    
    # 3. Run pipeline with integrated backtesting (one function call!)
    print("\n3. Running complete pipeline with backtesting...")
    print("-" * 50)
    
    signal, backtest_metrics = run_alpha_pipeline_with_backtest(
        returns_data=returns_df,
        pipeline_config=strategy_config,
        transaction_cost=0.001,      # 0.1% transaction cost
        max_position=0.12,           # Max 12% per asset
        leverage_limit=1.0,          # 100% max leverage
        benchmark_returns=returns_df.mean(axis=1),  # Equal-weight benchmark
        strategy_name="Momentum Strategy",
        print_results=True,          # Show detailed results
        plot_results=True,           # Create plots
        save_plots=True             # Save plots to files
    )
    
    print(f"\n✅ Complete pipeline and backtesting finished!")
    print(f"Signal generated: {len(signal)} observations")
    print(f"Backtest metrics calculated: {len(backtest_metrics)} metrics")
    
    return signal, backtest_metrics, returns_df

def multiple_strategies_example():
    """Example comparing multiple strategies."""
    
    print(f"\n{'='*80}")
    print("COMPARING MULTIPLE ALPHA STRATEGIES")
    print(f"{'='*80}")
    
    # Generate data
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    assets = [f'ASSET_{i:02d}' for i in range(15)]
    
    returns_data = pd.DataFrame(
        np.random.normal(0.0003, 0.018, (200, 15)),  # Slight positive drift
        index=dates,
        columns=assets
    )
    
    benchmark_returns = returns_data.mean(axis=1)
    
    # Define multiple strategies
    strategies = {
        'short_momentum': {
            'windowing': {'function': 'rolling', 'params': {'window': 10}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_rank', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'dollar_neutral', 'params': {}}
        },
        
        'mean_reversion': {
            'windowing': {'function': 'rolling', 'params': {'window': 15}},
            'preprocessing': {'function': 'robust_normalize', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'sign', 'params': {}},
            'position_sizing': {'function': 'market_neutral', 'params': {}}
        },
        
        'volatility_signal': {
            'windowing': {'function': 'rolling', 'params': {'window': 25}},
            'preprocessing': {'function': 'clip_outliers', 'params': {'lower_pct': 0.1, 'upper_pct': 0.9}},
            'reduction': {'function': 'rolling_std', 'params': {}},
            'postprocessing': {'function': 'threshold', 'params': {'threshold': 0.5}},
            'position_sizing': {'function': 'position_limits', 'params': {'max_position': 0.08}}
        }
    }
    
    print(f"Testing {len(strategies)} different alpha strategies...")
    
    # Run backtests for each strategy
    all_results = {}
    all_signals = {}
    
    for strategy_name, config in strategies.items():
        print(f"\n{'─'*50}")
        print(f"STRATEGY: {strategy_name.upper().replace('_', ' ')}")
        print(f"{'─'*50}")
        
        try:
            signal, metrics = run_alpha_pipeline_with_backtest(
                returns_data=returns_data,
                pipeline_config=config,
                transaction_cost=0.0015,  # 15 bps
                benchmark_returns=benchmark_returns,
                strategy_name=strategy_name,
                print_results=True,
                plot_results=False,  # Don't plot individual strategies
                save_plots=False
            )
            
            all_results[strategy_name] = metrics
            all_signals[strategy_name] = signal
            
        except Exception as e:
            print(f"Error with {strategy_name}: {str(e)}")
    
    # Compare all strategies
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("FINAL STRATEGY COMPARISON")
        print(f"{'='*80}")
        
        compare_strategies(all_results, 
                         save_path='/home/mingd/Documents/Projects/alpha/alpha_infra/multi_strategy_comparison.png')
        
        # Find and analyze best strategy
        valid_strategies = {name: metrics for name, metrics in all_results.items() if metrics}
        if valid_strategies:
            best_strategy = max(valid_strategies.keys(), 
                              key=lambda x: valid_strategies[x].get('sharpe_ratio', -999))
            
            print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy.upper()}")
            print(f"{'─'*60}")
            
            best_metrics = valid_strategies[best_strategy]
            print(f"Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"Annual Return: {best_metrics.get('annual_return', 0):.2%}")
            print(f"Max Drawdown: {best_metrics.get('max_drawdown', 0):.2%}")
            
            # Plot best strategy in detail
            plot_backtest_results(
                best_metrics,
                benchmark_returns,
                strategy_name=f"Best Strategy: {best_strategy}",
                save_path=f'/home/mingd/Documents/Projects/alpha/alpha_infra/best_{best_strategy}.png'
            )
    
    return all_results, all_signals

def custom_backtest_parameters_example():
    """Example showing custom backtest parameters."""
    
    print(f"\n{'='*80}")
    print("CUSTOM BACKTEST PARAMETERS EXAMPLE")
    print(f"{'='*80}")
    
    # Generate data
    np.random.seed(789)
    dates = pd.date_range('2023-01-01', periods=120, freq='D')
    assets = [f'STOCK_{i:02d}' for i in range(8)]
    
    returns_data = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (120, 8)),
        index=dates,
        columns=assets
    )
    
    # Strategy configuration
    config = {
        'windowing': {'function': 'rolling', 'params': {'window': 15}},
        'preprocessing': {'function': 'winsorize', 'params': {'limits': (0.05, 0.05)}},
        'reduction': {'function': 'principal_component', 'params': {'n_components': 1}},
        'postprocessing': {'function': 'smooth', 'params': {'window': 3}},
        'position_sizing': {'function': 'leverage_constraint', 'params': {'max_leverage': 0.8}}
    }
    
    # Test different transaction cost scenarios
    cost_scenarios = [0.0005, 0.001, 0.002]  # 5bps, 10bps, 20bps
    
    print("Testing different transaction cost scenarios:")
    
    cost_results = {}
    
    for cost in cost_scenarios:
        print(f"\n--- Transaction Cost: {cost*100:.1f} bps ---")
        
        signal, metrics = run_alpha_pipeline_with_backtest(
            returns_data=returns_data,
            pipeline_config=config,
            transaction_cost=cost,
            max_position=0.15,
            leverage_limit=0.8,
            strategy_name=f"Cost_{cost*10000:.0f}bps",
            print_results=False,  # Don't print individual results
            plot_results=False
        )
        
        cost_results[f"{cost*10000:.0f}bps"] = metrics
        
        if metrics:
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    
    # Compare transaction cost impact
    print(f"\n{'='*60}")
    print("TRANSACTION COST IMPACT ANALYSIS")
    print(f"{'='*60}")
    
    compare_strategies(cost_results)
    
    return cost_results

if __name__ == "__main__":
    print("Alpha Pipeline with Integrated Backtesting - Examples")
    print("=" * 80)
    
    try:
        # Run examples
        print("\n🚀 SIMPLE BACKTEST EXAMPLE")
        signal, metrics, returns_data = simple_backtest_example()
        
        print("\n🚀 MULTIPLE STRATEGIES EXAMPLE")
        multi_results, multi_signals = multiple_strategies_example()
        
        print("\n🚀 CUSTOM PARAMETERS EXAMPLE")
        cost_results = custom_backtest_parameters_example()
        
        print(f"\n{'='*80}")
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        print(f"\nKey takeaways:")
        print(f"✓ Use run_alpha_pipeline_with_backtest() for complete pipeline + backtesting")
        print(f"✓ Use backtest_signal() for backtesting existing signals")
        print(f"✓ Use compare_strategies() to compare multiple approaches")
        print(f"✓ All functions return detailed metrics and create visualizations")
        print(f"✓ No classes needed - everything is function-based!")
        
        print(f"\nGenerated files:")
        print(f"• momentum_strategy_backtest.png - Detailed backtest results")
        print(f"• multi_strategy_comparison.png - Strategy comparison")
        print(f"• best_*.png - Best strategy detailed analysis")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
