"""
Example Usage of Alpha Pipeline Infrastructure

This script demonstrates how to use the alpha pipeline with sample data.
"""

import sys
import os
sys.path.append('/home/mingd/Documents/Projects/alpha/alpha_infra')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_pipeline import *

def generate_sample_data(n_assets=50, n_days=1000, seed=42):
    """Generate sample stock return data."""
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Create asset names
    assets = [f'ASSET_{i:03d}' for i in range(n_assets)]
    
    # Generate correlated returns with some market factor
    market_return = np.random.normal(0, 0.015, n_days)
    
    returns = []
    for i in range(n_assets):
        # Each asset has some exposure to market + idiosyncratic noise
        beta = np.random.uniform(0.5, 1.5)
        idiosyncratic = np.random.normal(0, 0.02, n_days)
        asset_return = beta * market_return + idiosyncratic
        returns.append(asset_return)
    
    return pd.DataFrame(returns, columns=dates, index=assets).T

def demonstrate_pipeline_steps():
    """Demonstrate each step of the pipeline."""
    print("Generating sample data...")
    returns = generate_sample_data(n_assets=20, n_days=500)
    print(f"Generated returns data: {returns.shape}")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    print("\n" + "="*60)
    print("STEP-BY-STEP DEMONSTRATION")
    print("="*60)
    
    # Step 1: Windowing
    print("\n1. WINDOWING: Raw returns -> Windowed returns")
    windowed = rolling_window(returns, window=20)
    print(f"Created rolling window with 20-day lookback")
    
    # Step 2: Pre-processing
    print("\n2. PRE-PROCESSING: Remove outliers")
    # For demonstration, let's work with the raw data and apply preprocessing
    processed = z_score_normalize(returns)
    print(f"Applied z-score normalization")
    print(f"Original std: {returns.std().mean():.4f}, Normalized std: {processed.std().mean():.4f}")
    
    # Step 3: Reduction
    print("\n3. REDUCTION: Cross-sectional ranking")
    ranked = cross_sectional_rank(processed)
    print(f"Applied cross-sectional ranking")
    print(f"Rank range: [{ranked.min().min():.3f}, {ranked.max().max():.3f}]")
    
    # Step 4: Post-processing
    print("\n4. POST-PROCESSING: Convert to signal")
    signal = rank_signal(ranked.mean(axis=1))  # Average rank across assets
    print(f"Created mean-reversion signal")
    print(f"Signal range: [{signal.min():.3f}, {signal.max():.3f}]")
    
    # Step 5: Position sizing
    print("\n5. POSITION SIZING: Dollar neutral")
    positions = dollar_neutral(signal)
    print(f"Applied dollar neutral constraint")
    print(f"Position sum: {positions.sum():.6f} (should be ~0)")
    
    return returns, positions

def run_complete_pipelines():
    """Run several complete pipeline examples."""
    print("\n" + "="*60)
    print("COMPLETE PIPELINE EXAMPLES")
    print("="*60)
    
    returns = generate_sample_data(n_assets=30, n_days=400)
    
    # Get example configurations
    configs = create_sample_pipeline_configs()
    
    results = {}
    
    for strategy_name, config in configs.items():
        print(f"\nRunning {strategy_name.upper()} strategy...")
        
        try:
            # Validate configuration
            if not validate_pipeline_config(config):
                print(f"Invalid configuration for {strategy_name}")
                continue
            
            # Run pipeline
            signal = run_alpha_pipeline(returns, config)
            results[strategy_name] = signal
            
            print(f"Signal statistics:")
            print(f"  Mean: {signal.mean():.6f}")
            print(f"  Std: {signal.std():.4f}")
            print(f"  Sharpe: {signal.mean()/signal.std():.4f}")
            print(f"  Non-zero signals: {(signal != 0).sum()}/{len(signal)}")
            
        except Exception as e:
            print(f"Error running {strategy_name}: {str(e)}")
    
    return results

def analyze_signal_performance(signal, returns, name="Strategy"):
    """Basic performance analysis of a signal."""
    print(f"\nPerformance Analysis: {name}")
    print("-" * 40)
    
    # Calculate signal returns (simplified)
    # In practice, you'd use proper backtesting with transaction costs, etc.
    portfolio_return = (signal.shift(1) * returns.mean(axis=1)).dropna()
    
    # Basic statistics
    total_return = portfolio_return.cumsum().iloc[-1]
    annual_return = portfolio_return.mean() * 252
    annual_vol = portfolio_return.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    print(f"Total Return: {total_return:.4f}")
    print(f"Annual Return: {annual_return:.4f}")
    print(f"Annual Volatility: {annual_vol:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    return portfolio_return

def plot_signals(results):
    """Plot the generated signals."""
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
    if len(results) == 1:
        axes = [axes]
    
    for i, (name, signal) in enumerate(results.items()):
        axes[i].plot(signal.index, signal.values)
        axes[i].set_title(f"{name.capitalize()} Signal")
        axes[i].set_ylabel("Signal Strength")
        axes[i].grid(True, alpha=0.3)
        
        # Add zero line
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('/home/mingd/Documents/Projects/alpha/alpha_infra/signals_example.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def custom_pipeline_example():
    """Show how to create a custom pipeline."""
    print("\n" + "="*60)
    print("CUSTOM PIPELINE EXAMPLE")
    print("="*60)
    
    returns = generate_sample_data(n_assets=25, n_days=300)
    
    # Create custom configuration
    custom_config = {
        'windowing': {
            'function': 'rolling', 
            'params': {'window': 15}
        },
        'preprocessing': {
            'function': 'winsorize', 
            'params': {'limits': (0.02, 0.02)}
        },
        'reduction': {
            'function': 'principal_component', 
            'params': {'n_components': 1}
        },
        'postprocessing': {
            'function': 'smooth', 
            'params': {'window': 5}
        },
        'position_sizing': {
            'function': 'position_limits', 
            'params': {'max_position': 0.1}
        }
    }
    
    print("Custom configuration:")
    for step, config in custom_config.items():
        print(f"  {step}: {config['function']} with {config['params']}")
    
    # Run custom pipeline
    custom_signal = run_alpha_pipeline(returns, custom_config)
    
    print(f"\nCustom signal statistics:")
    print(f"  Range: [{custom_signal.min():.4f}, {custom_signal.max():.4f}]")
    print(f"  Mean: {custom_signal.mean():.6f}")
    print(f"  Std: {custom_signal.std():.4f}")
    
    return custom_signal

if __name__ == "__main__":
    print("Alpha Pipeline Infrastructure - Example Usage")
    print("=" * 60)
    
    # Show available functions
    print("\nAvailable functions in each category:")
    list_available_functions()
    
    # Run step-by-step demonstration
    returns, positions = demonstrate_pipeline_steps()
    
    # Run complete pipeline examples
    results = run_complete_pipelines()
    
    # Show custom pipeline
    custom_signal = custom_pipeline_example()
    
    # Add custom signal to results
    if custom_signal is not None:
        results['custom'] = custom_signal
    
    # Analyze performance for each strategy
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    for name, signal in results.items():
        analyze_signal_performance(signal, returns, name)
    
    # Plot all signals
    if results:
        print(f"\nPlotting {len(results)} signals...")
        plot_signals(results)
    
    print("\nDone! Check 'signals_example.png' for visualizations.")
