"""
Risk Management Example for Alpha Pipeline

This example demonstrates the risk management capabilities including:
1. Spectral risk adjustment with different eigenvalue methods
2. Covariance matrix computation and adjustment
3. Integration with the alpha pipeline
"""

import sys
import os
sys.path.append('/home/mingd/Documents/Projects/alpha/alpha_infra')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_pipeline import *

def generate_correlated_returns(n_assets=10, n_days=300, seed=42):
    """Generate returns data with realistic correlation structure."""
    np.random.seed(seed)
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Create sector structure
    sectors = ['Tech', 'Finance', 'Healthcare', 'Energy']
    assets_per_sector = n_assets // len(sectors)
    
    asset_names = []
    all_returns = []
    
    for sector_idx, sector in enumerate(sectors):
        # Common sector factor
        sector_factor = np.random.normal(0, 0.01, n_days)
        
        for asset_idx in range(assets_per_sector):
            asset_name = f'{sector}_{asset_idx:02d}'
            asset_names.append(asset_name)
            
            # Asset specific parameters
            market_beta = np.random.uniform(0.8, 1.2)
            sector_beta = np.random.uniform(0.5, 1.0)
            volatility = np.random.uniform(0.15, 0.25)
            
            # Market factor
            market_factor = np.random.normal(0, 0.012, n_days)
            
            # Idiosyncratic noise
            idiosyncratic = np.random.normal(0, volatility * 0.7, n_days)
            
            # Combine factors
            asset_returns = (market_beta * market_factor + 
                           sector_beta * sector_factor + 
                           idiosyncratic) / np.sqrt(252)  # Daily volatility
            
            all_returns.append(asset_returns)
    
    # Add remaining assets to last sector if needed
    while len(asset_names) < n_assets:
        asset_name = f'Other_{len(asset_names):02d}'
        asset_names.append(asset_name)
        returns = np.random.normal(0, 0.02/np.sqrt(252), n_days)
        all_returns.append(returns)
    
    returns_df = pd.DataFrame(
        np.array(all_returns).T,
        index=dates,
        columns=asset_names[:n_assets]
    )
    
    return returns_df

def demonstrate_risk_methods():
    """Demonstrate different risk adjustment methods."""
    
    print("Risk Management Methods Demonstration")
    print("=" * 60)
    
    # Generate sample data
    returns_data = generate_correlated_returns(n_assets=8, n_days=200)
    print(f"Generated {returns_data.shape[0]} days of returns for {returns_data.shape[1]} assets")
    
    # Compute sample covariance matrix
    cov_matrix = compute_covariance_matrix(returns_data, window=100)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    
    # Create sample weights
    signal_weights = pd.Series([0.2, -0.1, 0.15, 0.0, -0.05, 0.1, 0.3, -0.2], 
                              index=returns_data.columns)
    print(f"Original signal weights sum: {signal_weights.sum():.4f}")
    
    # Test different risk adjustment methods
    risk_methods = {
        'ridge': {'lambda_reg': 0.01},
        'shrinkage': {'alpha': 0.2},
        'sqrt': {},
        'threshold': {'drop_largest': True}
    }
    
    print(f"\n{'Method':<12} {'Sum':<8} {'Volatility':<12} {'Max Weight':<12}")
    print("-" * 50)
    
    results = {}
    
    for method_name, params in risk_methods.items():
        risk_adjusted = spectral_risk_adjustment(
            cov_matrix, 
            signal_weights, 
            method=method_name, 
            **params
        )
        
        # Compute portfolio risk
        risk_metrics = portfolio_risk_metrics(risk_adjusted, cov_matrix)
        portfolio_vol = risk_metrics['portfolio_volatility']
        
        results[method_name] = risk_adjusted
        
        print(f"{method_name:<12} {risk_adjusted.sum():<8.4f} {portfolio_vol:<12.4f} {risk_adjusted.abs().max():<12.4f}")
    
    return results, cov_matrix, signal_weights

def test_risk_managed_pipeline():
    """Test the complete pipeline with risk management."""
    
    print(f"\n{'='*80}")
    print("RISK-MANAGED ALPHA PIPELINE")
    print(f"{'='*80}")
    
    # Generate market data
    returns_data = generate_correlated_returns(n_assets=12, n_days=250)
    print(f"Generated market data: {returns_data.shape}")
    
    # Define strategies with different risk management approaches
    strategies = {
        'basic_momentum': {
            'windowing': {'function': 'rolling', 'params': {'window': 20}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'dollar_neutral', 'params': {}}
        },
        
        'ridge_risk_managed': {
            'windowing': {'function': 'rolling', 'params': {'window': 20}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'risk_managed_positions', 
                              'params': {'risk_method': 'ridge', 'lambda_reg': 0.02, 'lookback_window': 60}}
        },
        
        'shrinkage_risk_managed': {
            'windowing': {'function': 'rolling', 'params': {'window': 20}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'risk_managed_positions', 
                              'params': {'risk_method': 'shrinkage', 'alpha': 0.3, 'lookback_window': 60}}
        },
        
        'threshold_risk_managed': {
            'windowing': {'function': 'rolling', 'params': {'window': 20}},
            'preprocessing': {'function': 'z_score', 'params': {}},
            'reduction': {'function': 'cross_sectional_mean', 'params': {}},
            'postprocessing': {'function': 'rank', 'params': {}},
            'position_sizing': {'function': 'risk_managed_positions', 
                              'params': {'risk_method': 'threshold', 'drop_largest': True, 'lookback_window': 60}}
        }
    }
    
    # Run backtests for each strategy
    all_results = {}
    benchmark_returns = returns_data.mean(axis=1)
    
    for strategy_name, config in strategies.items():
        print(f"\n{'─'*50}")
        print(f"TESTING: {strategy_name.upper()}")
        print(f"{'─'*50}")
        
        try:
            signal, metrics = run_alpha_pipeline_with_backtest(
                returns_data=returns_data,
                pipeline_config=config,
                transaction_cost=0.001,
                benchmark_returns=benchmark_returns,
                strategy_name=strategy_name,
                print_results=True,
                plot_results=False
            )
            
            all_results[strategy_name] = metrics
            
        except Exception as e:
            print(f"Error with {strategy_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Compare all strategies
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("RISK MANAGEMENT COMPARISON")
        print(f"{'='*80}")
        
        compare_strategies(all_results)
        
        # Create detailed comparison
        comparison_data = []
        for name, metrics in all_results.items():
            if metrics:
                comparison_data.append({
                    'Strategy': name,
                    'Sharpe': metrics.get('sharpe_ratio', 0),
                    'Return': metrics.get('annual_return', 0),
                    'Volatility': metrics.get('annual_volatility', 0),
                    'Max_DD': metrics.get('max_drawdown', 0),
                    'Risk_Adjusted': metrics.get('sharpe_ratio', 0) / (abs(metrics.get('max_drawdown', 0.01)) + 0.01)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Risk_Adjusted', ascending=False)
            
            print(f"\nDetailed Risk-Adjusted Performance Ranking:")
            print(comparison_df.round(4).to_string(index=False))
    
    return all_results

def analyze_eigenvalue_adjustments():
    """Analyze the impact of different eigenvalue adjustments."""
    
    print(f"\n{'='*80}")
    print("EIGENVALUE ADJUSTMENT ANALYSIS")
    print(f"{'='*80}")
    
    # Generate data with known correlation structure
    returns_data = generate_correlated_returns(n_assets=6, n_days=150)
    
    # Compute covariance matrix
    cov_matrix = compute_covariance_matrix(returns_data, window=100)
    
    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix.values)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    
    print(f"Original eigenvalues: {eigenvalues}")
    print(f"Condition number: {eigenvalues[0] / eigenvalues[-1]:.2f}")
    
    # Test different adjustment methods
    adjustment_methods = {
        'original': eigenvalues,
        'ridge_0.01': eigenvalues + 0.01,
        'ridge_0.05': eigenvalues + 0.05,
        'shrinkage_0.1': 0.9 * eigenvalues + 0.1 * np.mean(eigenvalues),
        'shrinkage_0.3': 0.7 * eigenvalues + 0.3 * np.mean(eigenvalues),
        'sqrt': np.sqrt(eigenvalues),
        'threshold': np.where(eigenvalues == eigenvalues[0], eigenvalues[1], eigenvalues)
    }
    
    print(f"\n{'Method':<15} {'Condition #':<12} {'Min Eigenval':<12} {'Max Eigenval':<12}")
    print("-" * 60)
    
    for method_name, adj_eigenvals in adjustment_methods.items():
        condition_num = adj_eigenvals[0] / adj_eigenvals[-1]
        print(f"{method_name:<15} {condition_num:<12.2f} {adj_eigenvals[-1]:<12.4f} {adj_eigenvals[0]:<12.4f}")
    
    # Visualize eigenvalue spectra
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot eigenvalue spectra
        ax1.plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', label='Original')
        ax1.plot(range(1, len(eigenvalues)+1), adjustment_methods['ridge_0.01'], 's-', label='Ridge 0.01')
        ax1.plot(range(1, len(eigenvalues)+1), adjustment_methods['shrinkage_0.3'], '^-', label='Shrinkage 0.3')
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalue Spectra Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot condition numbers
        methods = list(adjustment_methods.keys())
        condition_numbers = [adj_eigenvals[0] / adj_eigenvals[-1] for adj_eigenvals in adjustment_methods.values()]
        
        ax2.bar(methods, condition_numbers)
        ax2.set_xlabel('Adjustment Method')
        ax2.set_ylabel('Condition Number')
        ax2.set_title('Condition Number by Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/mingd/Documents/Projects/alpha/alpha_infra/eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return adjustment_methods

if __name__ == "__main__":
    print("Risk Management for Alpha Pipeline")
    print("=" * 80)
    
    try:
        # Run demonstrations
        print("\n🔧 RISK METHODS DEMONSTRATION")
        risk_results, cov_matrix, weights = demonstrate_risk_methods()
        
        print("\n🚀 RISK-MANAGED PIPELINE TEST")
        pipeline_results = test_risk_managed_pipeline()
        
        print("\n📊 EIGENVALUE ADJUSTMENT ANALYSIS")
        eigenvalue_analysis = analyze_eigenvalue_adjustments()
        
        print(f"\n{'='*80}")
        print("RISK MANAGEMENT DEMONSTRATION COMPLETE")
        print(f"{'='*80}")
        
        print(f"\nKey Features Demonstrated:")
        print(f"✓ Spectral decomposition with multiple eigenvalue adjustments")
        print(f"✓ Ridge regularization (C + λI)")
        print(f"✓ Shrinkage estimation ((1-α)C + αtr(C)/n I)")
        print(f"✓ Square root covariance adjustment")
        print(f"✓ Eigenvalue thresholding")
        print(f"✓ Integration with alpha pipeline")
        print(f"✓ Portfolio risk metrics computation")
        
        print(f"\nGenerated files:")
        print(f"• eigenvalue_analysis.png - Eigenvalue spectrum analysis")
        print(f"• Various backtest plots for risk-managed strategies")
        
        # Summary of best performing method
        if pipeline_results:
            valid_results = {name: metrics for name, metrics in pipeline_results.items() if metrics}
            if valid_results:
                best_strategy = max(valid_results.keys(), 
                                  key=lambda x: valid_results[x].get('sharpe_ratio', -999))
                best_sharpe = valid_results[best_strategy].get('sharpe_ratio', 0)
                
                print(f"\n🏆 BEST RISK MANAGEMENT METHOD: {best_strategy}")
                print(f"   Sharpe Ratio: {best_sharpe:.3f}")
        
    except Exception as e:
        print(f"Error in risk management demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
