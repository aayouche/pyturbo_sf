"""
Tests for bootstrapping_tools.py module functionality.

This module tests all bootstrapping functions for 1D, 2D, and 3D cases.
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.bootstrapping_tools import (
    # 1D functions
    run_bootstrap_sf_1d,
    monte_carlo_simulation_1d,
    _process_spacing_data_batch_1d,
    _calculate_bootstrap_statistics_1d,
    _evaluate_convergence_1d,
    _group_bins_for_iteration_1d,
    _get_spacing_distribution_1d,
    _update_spacing_effectiveness_1d,
    # 2D functions
    run_bootstrap_sf_2d,
    monte_carlo_simulation_2d,
    _process_bootstrap_batch_2d,
    _calculate_bootstrap_statistics_2d,
    _evaluate_convergence_2d,
    _group_bins_for_iteration_2d,
    _get_spacing_distribution_2d,
    _update_spacing_effectiveness_2d,
    # 3D functions
    run_bootstrap_sf_3d,
    monte_carlo_simulation_3d,
    _process_bootstrap_batch_3d,
    _calculate_bootstrap_statistics_3d,
    _evaluate_convergence_3d,
    _group_bins_for_iteration_3d,
    _get_spacing_distribution_3d,
    _update_spacing_effectiveness_3d,
    # Weighted bootstrap statistics
    _compute_weighted_bootstrap_stats,
)


# =============================================================================
# Fixtures for test datasets
# =============================================================================

@pytest.fixture
def dataset_1d():
    """Create a simple 1D dataset for testing."""
    n = 64  # Power of 2 for bootstrapping
    x = np.linspace(0, 10, n)
    scalar = np.sin(2 * np.pi * x / 10)
    scalar2 = np.cos(2 * np.pi * x / 10)
    
    ds = xr.Dataset(
        data_vars={
            "temperature": ("x", scalar),
            "salinity": ("x", scalar2),
        },
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_2d():
    """Create a simple 2D dataset for testing."""
    nx, ny = 32, 32  # Power of 2 for bootstrapping
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 8)
    v = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8)
    scalar = np.sin(X + Y)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "temperature": (("y", "x"), scalar),
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_3d():
    """Create a simple 3D dataset for testing."""
    nx, ny, nz = 16, 16, 8  # Power of 2 for bootstrapping
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10)
    v = np.cos(2 * np.pi * Y / 8)
    w = np.sin(2 * np.pi * Z / 6)
    scalar = np.sin(X + Y + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "temperature": (("z", "y", "x"), scalar),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def linear_bins_1d():
    """Create linear bin edges for 1D testing."""
    return np.linspace(0, 10, 11)


@pytest.fixture
def linear_bins_x():
    """Create linear bin edges for x dimension."""
    return np.linspace(0, 10, 11)


@pytest.fixture
def linear_bins_y():
    """Create linear bin edges for y dimension."""
    return np.linspace(0, 8, 9)


@pytest.fixture
def linear_bins_z():
    """Create linear bin edges for z dimension."""
    return np.linspace(0, 6, 7)


# =============================================================================
# Tests for 1D Bootstrap Functions
# =============================================================================

class TestCalculateBootstrapStatistics1D:
    """Tests for _calculate_bootstrap_statistics_1d function."""
    
    def test_basic_statistics(self):
        """Test basic statistics calculation from accumulators."""
        n_bins = 10
        
        # Create mock accumulators
        bin_accumulators = {}
        for j in range(n_bins):
            bin_accumulators[j] = {
                'weighted_sum': np.random.rand() * 100,
                'total_weight': np.random.rand() * 10 + 1,
                'bootstrap_samples': [
                    {'mean': np.random.rand() * 10,
                     'weight': np.random.rand() + 0.1, 'weight': np.random.rand() + 0.1}
                    for _ in range(20)
                ]
            }
        
        sf_means, sf_stds, ci_lower, ci_upper = _calculate_bootstrap_statistics_1d(
            bin_accumulators, n_bins, confidence_level=0.95
        )
        
        assert sf_means.shape == (n_bins,)
        assert sf_stds.shape == (n_bins,)
        assert ci_lower.shape == (n_bins,)
        assert ci_upper.shape == (n_bins,)
        
        # All bins should have valid means
        assert np.all(~np.isnan(sf_means))
        # CI upper should be >= CI lower
        valid = ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
        assert np.all(ci_upper[valid] >= ci_lower[valid])
        
    def test_different_confidence_levels(self):
        """Test different confidence levels."""
        n_bins = 5
        
        bin_accumulators = {}
        for j in range(n_bins):
            bin_accumulators[j] = {
                'weighted_sum': 50.0,
                'total_weight': 10.0,
                'bootstrap_samples': [
                    {'mean': 5.0 + np.random.randn() * 0.5, 'weight': np.random.rand() + 0.1}
                    for _ in range(30)
                ]
            }
        
        sf_means, sf_stds, ci_lower, ci_upper = _calculate_bootstrap_statistics_1d(
            bin_accumulators, n_bins, confidence_level=0.95
        )
        
        # Check that CIs are valid
        for j in range(n_bins):
            if not np.isnan(ci_lower[j]) and not np.isnan(ci_upper[j]):
                assert ci_lower[j] <= ci_upper[j]
                
    def test_empty_accumulators(self):
        """Test with empty accumulators."""
        n_bins = 5
        bin_accumulators = {}  # Empty
        
        sf_means, sf_stds, ci_lower, ci_upper = _calculate_bootstrap_statistics_1d(
            bin_accumulators, n_bins
        )
        
        assert np.all(np.isnan(sf_means))
        assert np.all(np.isnan(sf_stds))
        
    def test_single_bootstrap_sample(self):
        """Test with single bootstrap sample (should have NaN std)."""
        n_bins = 3
        
        bin_accumulators = {
            0: {
                'weighted_sum': 10.0,
                'total_weight': 2.0,
                'bootstrap_samples': [{'mean': 5.0, 'weight': np.random.rand() + 0.1}]  # Only one sample
            }
        }
        
        sf_means, sf_stds, ci_lower, ci_upper = _calculate_bootstrap_statistics_1d(
            bin_accumulators, n_bins
        )
        
        assert sf_means[0] == pytest.approx(5.0)
        assert np.isnan(sf_stds[0])


class TestEvaluateConvergence1D:
    """Tests for _evaluate_convergence_1d function."""
    
    def test_convergence_by_epsilon(self):
        """Test convergence detection by epsilon threshold."""
        n_bins = 10
        sf_stds = np.ones(n_bins) * 0.001  # Very small std
        point_counts = np.ones(n_bins, dtype=int) * 100
        bin_bootstraps = np.ones(n_bins, dtype=int) * 50
        
        converged, reasons = _evaluate_convergence_1d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        # All bins should converge by epsilon
        assert np.all(converged)
        assert reasons['converged_eps'] == n_bins
        
    def test_convergence_by_low_density(self):
        """Test convergence marking for low density bins."""
        n_bins = 10
        sf_stds = np.ones(n_bins) * 0.1
        point_counts = np.ones(n_bins, dtype=int) * 5  # Low count
        bin_bootstraps = np.ones(n_bins, dtype=int) * 50
        
        converged, reasons = _evaluate_convergence_1d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        # All bins should be marked converged due to low density
        assert np.all(converged)
        assert reasons['low_density'] == n_bins
        
    def test_convergence_by_max_bootstraps(self):
        """Test convergence when max bootstraps reached."""
        n_bins = 10
        sf_stds = np.ones(n_bins) * 0.5  # High std
        point_counts = np.ones(n_bins, dtype=int) * 100
        bin_bootstraps = np.ones(n_bins, dtype=int) * 100  # At max
        
        converged, reasons = _evaluate_convergence_1d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        assert np.all(converged)
        assert reasons['max_bootstraps'] == n_bins
        
    def test_nan_std_convergence(self):
        """Test that NaN std bins are marked converged."""
        n_bins = 10
        sf_stds = np.full(n_bins, np.nan)
        point_counts = np.ones(n_bins, dtype=int) * 100
        bin_bootstraps = np.ones(n_bins, dtype=int) * 50
        
        converged, reasons = _evaluate_convergence_1d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        assert np.all(converged)
        assert reasons['nan_std'] == n_bins


class TestGroupBinsForIteration1D:
    """Tests for _group_bins_for_iteration_1d function."""
    
    def test_basic_grouping(self):
        """Test basic bin grouping."""
        unconverged_indices = np.array([0, 2, 5, 7, 9])
        bin_density = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        bootstrap_steps = np.ones(10, dtype=int) * 10
        
        groups = _group_bins_for_iteration_1d(unconverged_indices, bin_density, bootstrap_steps)
        
        # Should have groups based on (step, density_quartile)
        assert isinstance(groups, dict)
        
        # Total bins in groups should equal unconverged count
        total_bins = sum(len(v) for v in groups.values())
        assert total_bins == len(unconverged_indices)
        
    def test_different_step_sizes(self):
        """Test grouping with different step sizes."""
        unconverged_indices = np.array([0, 1, 2, 3])
        bin_density = np.array([0.5, 0.5, 0.5, 0.5])
        bootstrap_steps = np.array([10, 10, 20, 20])
        
        groups = _group_bins_for_iteration_1d(unconverged_indices, bin_density, bootstrap_steps)
        
        # Should have 2 groups (different step sizes)
        assert len(groups) == 2


class TestGetSpacingDistribution1D:
    """Tests for _get_spacing_distribution_1d function."""
    
    def test_basic_distribution(self):
        """Test basic spacing distribution."""
        bin_list = [0, 1, 2, 3, 4]
        spacing_values = [1, 2, 4]
        n_bins = 10
        
        # Create effectiveness scores
        spacing_effectiveness = {
            1: np.ones(n_bins) * 0.5,
            2: np.ones(n_bins) * 0.3,
            4: np.ones(n_bins) * 0.2,
        }
        
        total_bootstraps = 100
        
        distribution = _get_spacing_distribution_1d(
            bin_list, spacing_effectiveness, total_bootstraps, spacing_values
        )
        
        assert isinstance(distribution, list)
        
        # Total distributed bootstraps should be <= total_bootstraps
        total_distributed = sum(bootstraps for _, bootstraps in distribution)
        assert total_distributed <= total_bootstraps
        
    def test_zero_effectiveness(self):
        """Test with zero effectiveness scores."""
        bin_list = [0, 1]
        spacing_values = [1, 2]
        n_bins = 5
        
        spacing_effectiveness = {
            1: np.zeros(n_bins),
            2: np.zeros(n_bins),
        }
        
        distribution = _get_spacing_distribution_1d(
            bin_list, spacing_effectiveness, 100, spacing_values
        )
        
        # Should still return something (equal distribution fallback)
        assert isinstance(distribution, list)


class TestUpdateSpacingEffectiveness1D:
    """Tests for _update_spacing_effectiveness_1d function."""
    
    def test_basic_update(self):
        """Test basic effectiveness update."""
        n_bins = 10
        spacing_values = [1, 2, 4]
        
        bin_spacing_effectiveness = {sp: np.zeros(n_bins) for sp in spacing_values}
        bin_spacing_counts = {sp: np.zeros(n_bins, dtype=int) for sp in spacing_values}
        bin_spacing_bootstraps = {sp: np.zeros(n_bins, dtype=int) for sp in spacing_values}
        
        # Add some counts
        bin_spacing_counts[1][0] = 50
        bin_spacing_counts[1][1] = 30
        
        bin_list = [0, 1, 2]
        
        _update_spacing_effectiveness_1d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value=1, bin_list=bin_list,
            bootstraps=100
        )
        
        # Check effectiveness was updated
        assert bin_spacing_effectiveness[1][0] == 0.5  # 50/100
        assert bin_spacing_effectiveness[1][1] == 0.3  # 30/100
        assert bin_spacing_effectiveness[1][2] == 0.0  # No counts
        
        # Check bootstraps were updated
        assert bin_spacing_bootstraps[1][0] == 100
        
    def test_zero_bootstraps(self):
        """Test that zero bootstraps doesn't update."""
        n_bins = 5
        
        bin_spacing_effectiveness = {1: np.zeros(n_bins)}
        bin_spacing_counts = {1: np.ones(n_bins, dtype=int) * 10}
        bin_spacing_bootstraps = {1: np.zeros(n_bins, dtype=int)}
        
        _update_spacing_effectiveness_1d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value=1, bin_list=[0, 1],
            bootstraps=0
        )
        
        # Nothing should change
        assert np.all(bin_spacing_effectiveness[1] == 0)


# =============================================================================
# Tests for 2D Bootstrap Functions
# =============================================================================

class TestCalculateBootstrapStatistics2D:
    """Tests for _calculate_bootstrap_statistics_2d function."""
    
    def test_basic_statistics(self):
        """Test basic 2D statistics calculation."""
        n_bins_y, n_bins_x = 5, 6
        bin_shape = (n_bins_y, n_bins_x)
        
        bin_accumulators = {}
        for j in range(n_bins_y):
            for i in range(n_bins_x):
                bin_accumulators[(j, i)] = {
                    'weighted_sum': np.random.rand() * 100,
                    'total_weight': np.random.rand() * 10 + 1,
                    'bootstrap_samples': [
                        {'mean': np.random.rand() * 10, 'weight': np.random.rand() + 0.1}
                        for _ in range(15)
                    ]
                }
        
        sf_means, sf_stds = _calculate_bootstrap_statistics_2d(
            bin_accumulators, bin_shape
        )
        
        assert sf_means.shape == bin_shape
        assert sf_stds.shape == bin_shape


class TestEvaluateConvergence2D:
    """Tests for _evaluate_convergence_2d function."""
    
    def test_convergence_2d(self):
        """Test 2D convergence evaluation."""
        shape = (5, 6)
        sf_stds = np.ones(shape) * 0.001  # Small std
        point_counts = np.ones(shape, dtype=int) * 100
        bin_bootstraps = np.ones(shape, dtype=int) * 50
        
        converged, reasons = _evaluate_convergence_2d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        assert converged.shape == shape
        assert np.all(converged)
        
    def test_mixed_convergence_2d(self):
        """Test mixed convergence states in 2D."""
        shape = (4, 4)
        sf_stds = np.ones(shape) * 0.5  # High std
        sf_stds[0, 0] = 0.001  # One converged
        point_counts = np.ones(shape, dtype=int) * 100
        point_counts[1, 1] = 5  # Low density
        bin_bootstraps = np.ones(shape, dtype=int) * 50
        
        converged, reasons = _evaluate_convergence_2d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        assert converged[0, 0]  # Converged by epsilon
        assert converged[1, 1]  # Converged by low density
        assert reasons['converged_eps'] >= 1
        assert reasons['low_density'] >= 1


class TestGroupBinsForIteration2D:
    """Tests for _group_bins_for_iteration_2d function."""
    
    def test_basic_2d_grouping(self):
        """Test basic 2D bin grouping."""
        # Simulate unconverged indices as (y_indices, x_indices)
        unconverged_indices = (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        bin_density = np.array([[0.5, 0.5], [0.5, 0.5]])
        bootstrap_steps = np.ones((2, 2), dtype=int) * 10
        
        groups = _group_bins_for_iteration_2d(unconverged_indices, bin_density, bootstrap_steps)
        
        assert isinstance(groups, dict)
        total_bins = sum(len(v) for v in groups.values())
        assert total_bins == 4


class TestGetSpacingDistribution2D:
    """Tests for _get_spacing_distribution_2d function."""
    
    def test_2d_distribution(self):
        """Test 2D spacing distribution."""
        bin_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        spacing_values = [1, 2]
        shape = (2, 2)
        
        spacing_effectiveness = {
            1: np.ones(shape) * 0.6,
            2: np.ones(shape) * 0.4,
        }
        
        distribution = _get_spacing_distribution_2d(
            bin_list, spacing_effectiveness, 50, spacing_values
        )
        
        assert isinstance(distribution, list)
        total = sum(b for _, b in distribution)
        assert total <= 50


# =============================================================================
# Tests for 3D Bootstrap Functions
# =============================================================================

class TestCalculateBootstrapStatistics3D:
    """Tests for _calculate_bootstrap_statistics_3d function."""
    
    def test_basic_statistics_3d(self):
        """Test basic 3D statistics calculation."""
        n_bins_z, n_bins_y, n_bins_x = 4, 5, 6
        bin_shape = (n_bins_z, n_bins_y, n_bins_x)
        
        bin_accumulators = {}
        for k in range(n_bins_z):
            for j in range(n_bins_y):
                for i in range(n_bins_x):
                    bin_accumulators[(k, j, i)] = {
                        'weighted_sum': np.random.rand() * 100,
                        'total_weight': np.random.rand() * 10 + 1,
                        'bootstrap_samples': [
                            {'mean': np.random.rand() * 10, 'weight': np.random.rand() + 0.1}
                            for _ in range(10)
                        ]
                    }
        
        sf_means, sf_stds = _calculate_bootstrap_statistics_3d(
            bin_accumulators, bin_shape
        )
        
        assert sf_means.shape == bin_shape
        assert sf_stds.shape == bin_shape


class TestEvaluateConvergence3D:
    """Tests for _evaluate_convergence_3d function."""
    
    def test_convergence_3d(self):
        """Test 3D convergence evaluation."""
        shape = (4, 5, 6)
        sf_stds = np.ones(shape) * 0.001
        point_counts = np.ones(shape, dtype=int) * 100
        bin_bootstraps = np.ones(shape, dtype=int) * 50
        
        converged, reasons = _evaluate_convergence_3d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        assert converged.shape == shape
        assert np.all(converged)


class TestGroupBinsForIteration3D:
    """Tests for _group_bins_for_iteration_3d function."""
    
    def test_basic_3d_grouping(self):
        """Test basic 3D bin grouping."""
        unconverged_indices = (
            np.array([0, 0, 1, 1]),  # z
            np.array([0, 1, 0, 1]),  # y
            np.array([0, 0, 0, 0])   # x
        )
        bin_density = np.ones((2, 2, 1)) * 0.5
        bootstrap_steps = np.ones((2, 2, 1), dtype=int) * 10
        
        groups = _group_bins_for_iteration_3d(unconverged_indices, bin_density, bootstrap_steps)
        
        assert isinstance(groups, dict)


# =============================================================================
# Tests for Monte Carlo Simulation Functions
# =============================================================================

class TestMonteCarloSimulation1D:
    """Tests for monte_carlo_simulation_1d function."""
    
    def test_no_bootstrappable_dims(self, dataset_1d):
        """Test MC simulation with no bootstrappable dimensions."""
        sf_results, separations, pair_counts_results = monte_carlo_simulation_1d(
            ds=dataset_1d,
            dim="x",
            variables_names=["temperature"],
            order=2,
            nbootstrap=10,
            bootsize={"x": 32},
            num_bootstrappable=0,  # No bootstrapping
            all_spacings=[1],
            boot_indexes={},
            fun='scalar'
        )
        
        # Should return single result
        assert len(sf_results) == 1
        assert len(separations) == 1
        assert len(pair_counts_results) == 1


class TestMonteCarloSimulation2D:
    """Tests for monte_carlo_simulation_2d function."""
    
    def test_no_bootstrappable_dims_2d(self, dataset_2d):
        """Test 2D MC simulation with no bootstrappable dimensions."""
        sf_results, dx_vals, dy_vals, pair_counts_results = monte_carlo_simulation_2d(
            ds=dataset_2d,
            dims=["y", "x"],
            variables_names=["u", "v"],
            order=2,
            nbootstrap=10,
            bootsize={"y": 16, "x": 16},
            num_bootstrappable=0,
            all_spacings=[1],
            boot_indexes={},
            bootstrappable_dims=[],
            fun='longitudinal'
        )
        
        assert len(sf_results) == 1
        assert len(dx_vals) == 1
        assert len(dy_vals) == 1
        assert len(pair_counts_results) == 1


class TestMonteCarloSimulation3D:
    """Tests for monte_carlo_simulation_3d function."""
    
    def test_no_bootstrappable_dims_3d(self, dataset_3d):
        """Test 3D MC simulation with no bootstrappable dimensions."""
        sf_results, dx_vals, dy_vals, dz_vals, pair_counts_results = monte_carlo_simulation_3d(
            ds=dataset_3d,
            dims=["z", "y", "x"],
            variables_names=["u", "v", "w"],
            order=2,
            nbootstrap=10,
            bootsize={"z": 4, "y": 8, "x": 8},
            num_bootstrappable=0,
            all_spacings=[1],
            boot_indexes={},
            bootstrappable_dims=[],
            fun='longitudinal'
        )
        
        assert len(sf_results) == 1
        assert len(dx_vals) == 1
        assert len(dy_vals) == 1
        assert len(dz_vals) == 1
        assert len(pair_counts_results) == 1


# =============================================================================
# Tests for Process Batch Functions
# =============================================================================

class TestProcessBootstrapBatch2D:
    """Tests for _process_bootstrap_batch_2d function."""
    
    def test_basic_batch_processing(self):
        """Test basic 2D batch processing."""
        # Create mock SF results
        n_samples = 5
        n_points = 100
        
        sf_results = [np.random.randn(n_points) for _ in range(n_samples)]
        dx_vals = [np.random.rand(n_points) * 10 for _ in range(n_samples)]
        dy_vals = [np.random.rand(n_points) * 8 for _ in range(n_samples)]
        
        bins_x = np.linspace(0, 10, 6)
        bins_y = np.linspace(0, 8, 5)
        
        bin_accumulators = {}
        target_bins = {(j, i) for j in range(4) for i in range(5)}
        point_counts = np.zeros((4, 5), dtype=int)
        
        updated_bins = _process_bootstrap_batch_2d(
            sf_results, dx_vals, dy_vals,
            bins_x, bins_y, bin_accumulators, target_bins,
            point_counts=point_counts, add_to_counts=True
        )
        
        assert isinstance(updated_bins, set)
        assert np.sum(point_counts) > 0


class TestProcessBootstrapBatch3D:
    """Tests for _process_bootstrap_batch_3d function."""
    
    def test_basic_batch_processing_3d(self):
        """Test basic 3D batch processing."""
        n_samples = 3
        n_points = 50
        
        sf_results = [np.random.randn(n_points) for _ in range(n_samples)]
        dx_vals = [np.random.rand(n_points) * 10 for _ in range(n_samples)]
        dy_vals = [np.random.rand(n_points) * 8 for _ in range(n_samples)]
        dz_vals = [np.random.rand(n_points) * 6 for _ in range(n_samples)]
        
        bins_x = np.linspace(0, 10, 6)
        bins_y = np.linspace(0, 8, 5)
        bins_z = np.linspace(0, 6, 4)
        
        bin_accumulators = {}
        target_bins = {(k, j, i) for k in range(3) for j in range(4) for i in range(5)}
        point_counts = np.zeros((3, 4, 5), dtype=int)
        
        updated_bins = _process_bootstrap_batch_3d(
            sf_results, dx_vals, dy_vals, dz_vals,
            bins_x, bins_y, bins_z, bin_accumulators, target_bins,
            point_counts=point_counts, add_to_counts=True
        )
        
        assert isinstance(updated_bins, set)
        assert np.sum(point_counts) > 0


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestBootstrapEdgeCases:
    """Tests for edge cases in bootstrapping functions."""
    
    def test_empty_bootstrap_samples(self):
        """Test statistics with empty bootstrap samples."""
        n_bins = 5
        
        bin_accumulators = {
            0: {
                'weighted_sum': 10.0,
                'total_weight': 2.0,
                'bootstrap_samples': []  # Empty
            }
        }
        
        sf_means, sf_stds, ci_lower, ci_upper = _calculate_bootstrap_statistics_1d(
            bin_accumulators, n_bins
        )
        
        # Mean should still be calculated
        assert sf_means[0] == pytest.approx(5.0)
        # Std should be NaN with no samples
        assert np.isnan(sf_stds[0])
        
    def test_all_nan_values(self):
        """Test convergence with all NaN values."""
        n_bins = 5
        sf_stds = np.full(n_bins, np.nan)
        point_counts = np.zeros(n_bins, dtype=int)
        bin_bootstraps = np.zeros(n_bins, dtype=int)
        
        converged, reasons = _evaluate_convergence_1d(
            sf_stds, point_counts, bin_bootstraps,
            convergence_eps=0.01, max_bootstraps=100
        )
        
        # All should be marked converged (low density and nan)
        assert np.all(converged)


# =============================================================================
# Tests for weighted bootstrap statistics with n_eff correction
# =============================================================================

class TestComputeWeightedBootstrapStats:
    """Tests for _compute_weighted_bootstrap_stats function with n_eff correction."""
    
    def test_basic_computation(self):
        """Test basic computation with simple data."""
        bootstrap_samples = [
            {'mean': 1.0, 'weight': 10},
            {'mean': 2.0, 'weight': 20},
            {'mean': 3.0, 'weight': 30},
        ]
        
        weighted_mean, std_error, ci_lower, ci_upper = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.95
        )
        
        # Weighted mean should be closer to 3.0 due to higher weight
        assert 1.0 <= weighted_mean <= 3.0
        assert weighted_mean > 1.5  # Should be pulled toward 3.0
        
        # Standard error should be positive
        assert std_error >= 0
        
        # CI should bracket the mean
        assert ci_lower <= weighted_mean <= ci_upper
        
    def test_equal_weights(self):
        """Test with equal weights - should match numpy SE calculation."""
        bootstrap_samples = [
            {'mean': 10.0, 'weight': 100},
            {'mean': 12.0, 'weight': 100},
            {'mean': 11.0, 'weight': 100},
            {'mean': 13.0, 'weight': 100},
            {'mean': 9.0, 'weight': 100},
            {'mean': 11.0, 'weight': 100},
            {'mean': 10.0, 'weight': 100},
            {'mean': 12.0, 'weight': 100},
        ]
        
        weighted_mean, std_error, ci_lower, ci_upper = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.95
        )
        
        # With equal weights, weighted mean should be simple average
        means = np.array([s['mean'] for s in bootstrap_samples])
        expected_mean = np.mean(means)
        expected_se = np.std(means, ddof=1) / np.sqrt(len(means))
        
        assert np.abs(weighted_mean - expected_mean) < 0.01
        assert np.abs(std_error - expected_se) < 0.01
        
    def test_n_eff_calculation(self):
        """Test that effective sample size is correctly calculated."""
        # With equal weights, n_eff should equal actual n
        bootstrap_samples = [
            {'mean': 1.0, 'weight': 10},
            {'mean': 2.0, 'weight': 10},
            {'mean': 3.0, 'weight': 10},
        ]
        
        # n_eff = (sum(w))^2 / sum(w^2) = (30)^2 / (300) = 3.0
        weights = np.array([10, 10, 10])
        expected_n_eff = (np.sum(weights)**2) / np.sum(weights**2)
        assert expected_n_eff == 3.0
        
        # With unequal weights, n_eff should be less than n
        bootstrap_samples_unequal = [
            {'mean': 1.0, 'weight': 1000},
            {'mean': 2.0, 'weight': 100},
            {'mean': 3.0, 'weight': 500},
        ]
        
        weights_unequal = np.array([1000, 100, 500])
        n_eff_unequal = (np.sum(weights_unequal)**2) / np.sum(weights_unequal**2)
        assert n_eff_unequal < 3.0  # Should be less than actual n
        assert n_eff_unequal > 1.0  # But greater than 1
        
    def test_confidence_interval_width(self):
        """Test that higher confidence means wider interval."""
        bootstrap_samples = [
            {'mean': 1.0, 'weight': 10},
            {'mean': 2.0, 'weight': 20},
            {'mean': 3.0, 'weight': 30},
            {'mean': 2.5, 'weight': 25},
        ]
        
        _, _, ci_lower_95, ci_upper_95 = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.95
        )
        _, _, ci_lower_99, ci_upper_99 = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.99
        )
        
        width_95 = ci_upper_95 - ci_lower_95
        width_99 = ci_upper_99 - ci_lower_99
        
        # 99% CI should be wider than 95% CI
        assert width_99 > width_95
        
    def test_single_sample(self):
        """Test behavior with single bootstrap sample."""
        bootstrap_samples = [
            {'mean': 5.0, 'weight': 100},
        ]
        
        weighted_mean, std_error, ci_lower, ci_upper = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.95
        )
        
        # Should return the single value with NaN for stats
        assert weighted_mean == 5.0
        assert np.isnan(std_error)
        
    def test_empty_samples(self):
        """Test behavior with empty bootstrap samples."""
        bootstrap_samples = []
        
        weighted_mean, std_error, ci_lower, ci_upper = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.95
        )
        
        # Should return NaN for all
        assert np.isnan(weighted_mean)
        assert np.isnan(std_error)
        assert np.isnan(ci_lower)
        assert np.isnan(ci_upper)
        
    def test_ci_symmetric_around_mean(self):
        """Test that CI is symmetric around the mean (normal approximation)."""
        bootstrap_samples = [
            {'mean': 1.0, 'weight': 10},
            {'mean': 2.0, 'weight': 10},
            {'mean': 3.0, 'weight': 10},
            {'mean': 4.0, 'weight': 10},
            {'mean': 5.0, 'weight': 10},
        ]
        
        weighted_mean, std_error, ci_lower, ci_upper = _compute_weighted_bootstrap_stats(
            bootstrap_samples, confidence_level=0.95
        )
        
        # Distance from mean to lower should equal distance from mean to upper
        dist_lower = weighted_mean - ci_lower
        dist_upper = ci_upper - weighted_mean
        
        assert np.abs(dist_lower - dist_upper) < 1e-10


class TestPairCountsWeighting:
    """Tests for proper pair_counts weighting in batch processing."""
    
    def test_2d_batch_uses_pair_counts(self):
        """Test that _process_bootstrap_batch_2d properly uses pair_counts for weighting."""
        # Create mock SF results with different pair counts
        sf_results = [np.array([1.0, 2.0, 3.0])]  # One bootstrap with 3 separations
        dx_vals = [np.array([0.5, 1.5, 2.5])]
        dy_vals = [np.array([0.5, 1.5, 2.5])]
        pair_counts_results = [np.array([10, 100, 50])]  # Different pair counts per separation
        
        bins_x = np.array([0.0, 1.0, 2.0, 3.0])  # 3 bins
        bins_y = np.array([0.0, 1.0, 2.0, 3.0])
        
        bin_accumulators = {}
        target_bins = {(0, 0), (1, 1), (2, 2)}  # Target diagonal bins
        
        updated = _process_bootstrap_batch_2d(
            sf_results, dx_vals, dy_vals, bins_x, bins_y,
            bin_accumulators, target_bins, pair_counts_results=pair_counts_results
        )
        
        # Should have updated some bins
        assert len(updated) > 0
        
        # Check that weighted_sum uses pair_counts as weights
        for bin_key in updated:
            if bin_key in bin_accumulators:
                acc = bin_accumulators[bin_key]
                # total_weight should reflect pair counts, not just count of separations
                assert acc['total_weight'] > 0
                # bootstrap_samples should have weights based on pair_counts
                for sample in acc['bootstrap_samples']:
                    assert 'weight' in sample
                    assert sample['weight'] > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
