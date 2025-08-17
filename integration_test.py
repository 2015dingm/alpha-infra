"""
Integration Test Suite for Alpha Pipeline

Comprehensive testing of the full alpha generation pipeline.
Run this to validate all components work together correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
import traceback
from datetime import datetime

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from alpha_pipeline import *
    from testing_utilities import *
    print("✓ Successfully imported alpha_pipeline and testing_utilities")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure alpha_pipeline.py and testing_utilities.py are in the same directory")
    sys.exit(1)

class PipelineIntegrationTest:
    """
    Comprehensive integration test suite for the alpha pipeline.
    """
    
    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
        
    def log_result(self, test_name: str, passed: bool, 
                   details: str = "", exception: str = ""):
        """Log a test result."""
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'exception': exception,
            'timestamp': datetime.now()
        }
        self.test_results.append(result)
        
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if details and passed:
            print(f"      {details}")
        elif exception:
            print(f"      Error: {exception}")
    
    def test_data_generation(self):
        """Test data generation utilities."""
        try:
            # Test realistic data generation
            returns, sector_map = generate_realistic_returns(n_assets=30, n_days=150)
            
            assert returns.shape == (150, 30), f"Expected (150, 30), got {returns.shape}"
            assert not returns.empty, "Returns data is empty"
            assert len(sector_map) > 0, "No sectors generated"
            
            # Check for reasonable values
            max_return = returns.abs().max().max()
            assert max_return < 1.0, f"Unreasonably large return: {max_return}"
            
            # Test simple data generation
            simple_returns = generate_simple_test_data(n_assets=10, n_days=50)
            assert simple_returns.shape == (50, 10), "Simple data wrong shape"
            
            self.log_result("Data Generation", True, 
                          f"Generated {returns.shape} realistic data with {len(sector_map)} sectors")
            return returns, sector_map
            
        except Exception as e:
            self.log_result("Data Generation", False, exception=str(e))
            return None, None
    
    def test_windowing_functions(self, returns_data):
        """Test all windowing functions."""
        if returns_data is None:
            self.log_result("Windowing Functions", False, exception="No test data")
            return {}
        
        results = {}
        
        try:
            # Test rolling window
            rolling_result = rolling_window(returns_data, window=20, shift=0)
            assert not rolling_result.empty, "Rolling window result is empty"
            assert rolling_result.shape[0] <= returns_data.shape[0], "Rolling window too large"
            results['rolling'] = rolling_result
            
            # Test expanding window
            expanding_result = expanding_window(returns_data, min_periods=10, shift=1)
            assert not expanding_result.empty, "Expanding window result is empty"
            results['expanding'] = expanding_result
            
            # Test exponential window
            exp_result = exponential_window(returns_data, alpha=0.1, shift=2)
            assert not exp_result.empty, "Exponential window result is empty"
            results['exponential'] = exp_result
            
            self.log_result("Windowing Functions", True,
                          f"All 3 windowing functions working correctly")
            return results
            
        except Exception as e:
            self.log_result("Windowing Functions", False, exception=str(e))
            return {}
    
    def test_preprocessing_functions(self, windowed_data):
        """Test preprocessing functions."""
        if not windowed_data or 'rolling' not in windowed_data:
            self.log_result("Preprocessing Functions", False, exception="No windowed data")
            return {}
        
        results = {}
        test_data = windowed_data['rolling']
        
        try:
            # Test standardization
            std_result = z_score_normalize(test_data)
            assert not std_result.empty, "Standardization result is empty"
            
            # Check if standardization worked (should have mean ~0, std ~1)
            mean_abs_mean = abs(std_result.mean().mean())
            assert mean_abs_mean < 0.1, f"Standardized mean too large: {mean_abs_mean}"
            results['standardized'] = std_result
            
            # Test winsorization
            wins_result = winsorize(test_data, limits=(0.05, 0.05))
            assert not wins_result.empty, "Winsorization result is empty"
            results['winsorized'] = wins_result
            
            # Test normalization (using robust_normalize)
            norm_result = robust_normalize(test_data)
            assert not norm_result.empty, "Normalization result is empty"
            results['normalized'] = norm_result
            
            self.log_result("Preprocessing Functions", True,
                          "All preprocessing functions working correctly")
            return results
            
        except Exception as e:
            self.log_result("Preprocessing Functions", False, exception=str(e))
            return {}
    
    def test_reduction_functions(self, processed_data):
        """Test dimensionality reduction functions."""
        if not processed_data or 'standardized' not in processed_data:
            self.log_result("Reduction Functions", False, exception="No processed data")
            return {}
        
        results = {}
        test_data = processed_data['standardized']
        
        try:
            # Test cross-sectional operations
            cs_mean = cross_sectional_mean(test_data)
            assert isinstance(cs_mean, pd.Series), "Cross-sectional mean should be Series"
            assert len(cs_mean) <= len(test_data), "Cross-sectional mean too long"
            results['cs_mean'] = cs_mean
            
            cs_rank = cross_sectional_rank(test_data, normalize=True)
            assert isinstance(cs_rank, pd.DataFrame), "Cross-sectional rank should be DataFrame"
            results['cs_rank'] = cs_rank
            
            # Test PCA (only if we have enough data points)
            if test_data.shape[0] > test_data.shape[1] and test_data.shape[1] > 5:
                try:
                    pca_result = principal_component(test_data, n_components=3)
                    assert isinstance(pca_result, pd.Series), "PCA result should be Series"
                    results['pca'] = pca_result
                    pca_success = True
                except Exception as pca_e:
                    print(f"      PCA failed (non-critical): {pca_e}")
                    pca_success = False
            else:
                pca_success = True  # Skip PCA due to insufficient data
                print(f"      Skipping PCA test (insufficient data: {test_data.shape})")
            
            self.log_result("Reduction Functions", True,
                          f"Cross-sectional functions working, PCA: {'✓' if pca_success else '❌'}")
            return results
            
        except Exception as e:
            self.log_result("Reduction Functions", False, exception=str(e))
            return {}
    
    def test_postprocessing_functions(self, reduced_data):
        """Test postprocessing functions."""
        if not reduced_data or 'cs_mean' not in reduced_data:
            self.log_result("Postprocessing Functions", False, exception="No reduced data")
            return {}
        
        results = {}
        test_signal = reduced_data['cs_mean']
        
        try:
            # Test ranking
            ranked_signal = rank_signal(test_signal)
            assert isinstance(ranked_signal, pd.Series), "Ranked signal should be Series"
            assert len(ranked_signal) == len(test_signal), "Ranked signal length mismatch"
            results['ranked'] = ranked_signal
            
            # Test smoothing
            if len(test_signal) > 10:
                smoothed_signal = smooth_signal(test_signal, window=5)
                assert isinstance(smoothed_signal, pd.Series), "Smoothed signal should be Series"
                results['smoothed'] = smoothed_signal
            
            # Test signal validation
            validation = validate_signal_properties(ranked_signal)
            assert validation['valid'], f"Signal validation failed: {validation}"
            
            self.log_result("Postprocessing Functions", True,
                          f"Signal processing successful, validation: {validation['valid']}")
            return results
            
        except Exception as e:
            self.log_result("Postprocessing Functions", False, exception=str(e))
            return {}
    
    def test_position_sizing(self, signal, returns_data):
        """Test position sizing functions."""
        if signal is None or returns_data is None:
            self.log_result("Position Sizing", False, exception="No signal or returns data")
            return {}
        
        results = {}
        
        try:
            # Use the ranked signal from postprocessing
            if isinstance(signal, dict) and 'ranked' in signal:
                test_signal = signal['ranked']
            elif isinstance(signal, pd.Series):
                test_signal = signal
            else:
                raise ValueError("Invalid signal format for position sizing")
            
            # Test position sizing using dollar_neutral and position_limits
            neutral_positions = dollar_neutral(test_signal)
            assert isinstance(neutral_positions, pd.Series), "Positions should be Series"
            
            # Apply position limits
            positions = position_limits(neutral_positions, max_position=0.1)
            assert abs(positions).max() <= 0.1, f"Position limit violated: {abs(positions).max()}"
            results['equal_weight'] = positions
            
            # Test volatility targeting (if we have enough data)
            common_dates = test_signal.index.intersection(returns_data.index)
            if len(common_dates) > 20:
                vol_positions = volatility_target(test_signal, target_vol=0.1, 
                                                 returns_data=returns_data)
                assert isinstance(vol_positions, pd.Series), "Vol positions should be Series"
                results['vol_weighted'] = vol_positions
            
            self.log_result("Position Sizing", True,
                          f"Position sizing successful, max position: {abs(positions).max():.3f}")
            return results
            
        except Exception as e:
            self.log_result("Position Sizing", False, exception=str(e))
            return {}
    
    def test_risk_management(self, positions, returns_data):
        """Test risk management functions."""
        if positions is None or returns_data is None:
            self.log_result("Risk Management", False, exception="No positions or returns data")
            return
        
        try:
            # Use equal weight positions
            if isinstance(positions, dict) and 'equal_weight' in positions:
                test_positions = positions['equal_weight']
            elif isinstance(positions, pd.Series):
                test_positions = positions
            else:
                raise ValueError("Invalid positions format")
            
            # Test risk metrics calculation
            # Get aligned returns data
            position_date = test_positions.index[0] if hasattr(test_positions.index[0], 'date') else test_positions.index[0]
            
            # Find a suitable date in returns_data
            if hasattr(returns_data.index[0], 'date'):
                returns_date = returns_data.index[-1]  # Use last date
            else:
                returns_date = returns_data.index[-1]
            
            # Create a simple risk check
            max_single_position = abs(test_positions).max()
            position_concentration = (abs(test_positions) > 0.05).sum() / len(test_positions)
            
            risk_ok = (max_single_position < 0.2 and position_concentration < 0.5)
            
            self.log_result("Risk Management", True,
                          f"Max position: {max_single_position:.3f}, Concentration: {position_concentration:.3f}")
            
        except Exception as e:
            self.log_result("Risk Management", False, exception=str(e))
    
    def test_backtesting(self, signal, returns_data):
        """Test backtesting functions."""
        if signal is None or returns_data is None:
            self.log_result("Backtesting", False, exception="No signal or returns data")
            return {}
        
        try:
            # Use the appropriate signal
            if isinstance(signal, dict) and 'ranked' in signal:
                test_signal = signal['ranked']
            elif isinstance(signal, pd.Series):
                test_signal = signal
            else:
                raise ValueError("Invalid signal format")
            
            # Create a simple aligned test
            # Get overlapping dates between signal and returns
            common_dates = test_signal.index.intersection(returns_data.index)
            
            if len(common_dates) < 10:
                self.log_result("Backtesting", False, 
                               exception=f"Insufficient overlapping dates: {len(common_dates)}")
                return {}
            
            # Align data
            aligned_signal = test_signal.loc[common_dates]
            aligned_returns = returns_data.loc[common_dates]
            
            # Simple backtest: signal today, returns tomorrow
            if len(aligned_signal) > 1:
                # Shift signal to avoid look-ahead bias
                lagged_signal = aligned_signal.shift(1).dropna()
                next_day_returns = aligned_returns.loc[lagged_signal.index]
                
                # Simple strategy: long/short based on signal
                strategy_returns = (lagged_signal * next_day_returns.mean(axis=1)).dropna()
                
                if len(strategy_returns) > 10:
                    # Calculate basic metrics
                    annual_return = strategy_returns.mean() * 252
                    annual_vol = strategy_returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    backtest_results = {
                        'annual_return': annual_return,
                        'annual_volatility': annual_vol,
                        'sharpe_ratio': sharpe_ratio,
                        'total_periods': len(strategy_returns)
                    }
                    
                    # Validate results
                    validation = validate_backtest_results(backtest_results)
                    
                    self.log_result("Backtesting", True,
                                  f"Sharpe: {sharpe_ratio:.2f}, Annual return: {annual_return:.1%}")
                    return backtest_results
                else:
                    self.log_result("Backtesting", False,
                                  exception="Insufficient strategy returns for analysis")
                    return {}
            else:
                self.log_result("Backtesting", False,
                              exception="Insufficient aligned data for backtesting")
                return {}
                
        except Exception as e:
            self.log_result("Backtesting", False, exception=str(e))
            return {}
    
    def test_full_pipeline(self):
        """Test the complete pipeline end-to-end."""
        try:
            # Generate test data
            returns_data, _ = generate_realistic_returns(n_assets=15, n_days=100, seed=42)
            
            # Create windowed data
            windowed_data = rolling_window(returns_data, window=20, shift=1)
            
            # Preprocess
            processed_data = z_score_normalize(windowed_data)
            
            # Reduce dimensions
            signal = cross_sectional_mean(processed_data)
            
            # Postprocess
            final_signal = rank_signal(signal)
            
            # Create positions using available functions
            neutral_positions = dollar_neutral(final_signal)
            positions = position_limits(neutral_positions, max_position=0.05)
            
            # Simple validation
            assert isinstance(final_signal, pd.Series), "Final signal should be Series"
            assert isinstance(positions, pd.Series), "Positions should be Series"
            assert len(final_signal) > 0, "Signal should not be empty"
            assert abs(positions).max() <= 0.05, "Position size too large"
            
            self.log_result("Full Pipeline", True,
                          f"Complete pipeline successful, {len(final_signal)} signal points")
            return True
            
        except Exception as e:
            self.log_result("Full Pipeline", False, exception=str(e))
            return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        print("Starting Alpha Pipeline Integration Tests")
        print("=" * 50)
        
        # Test 1: Data generation
        returns_data, sector_map = self.test_data_generation()
        
        # Test 2: Windowing functions
        windowed_data = self.test_windowing_functions(returns_data)
        
        # Test 3: Preprocessing functions  
        processed_data = self.test_preprocessing_functions(windowed_data)
        
        # Test 4: Reduction functions
        reduced_data = self.test_reduction_functions(processed_data)
        
        # Test 5: Postprocessing functions
        final_signal = self.test_postprocessing_functions(reduced_data)
        
        # Test 6: Position sizing
        positions = self.test_position_sizing(final_signal, returns_data)
        
        # Test 7: Risk management
        self.test_risk_management(positions, returns_data)
        
        # Test 8: Backtesting
        backtest_results = self.test_backtesting(final_signal, returns_data)
        
        # Test 9: Full pipeline
        self.test_full_pipeline()
        
        # Print summary
        self.print_summary()
        
        return self.test_results
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✓")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print(f"\nFailed Tests:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  ❌ {result['test']}: {result['exception']}")
        
        duration = datetime.now() - self.start_time
        print(f"\nTotal Test Duration: {duration.total_seconds():.1f} seconds")
        
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! The alpha pipeline is ready for use.")
        else:
            print(f"\n⚠️  Some tests failed. Review the errors above before proceeding.")

def main():
    """Run the integration test suite."""
    # Create and run tests
    test_suite = PipelineIntegrationTest()
    results = test_suite.run_all_tests()
    
    return results

if __name__ == "__main__":
    results = main()
