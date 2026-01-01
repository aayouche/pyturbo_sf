"""
Tests for one_dimensional.py module functionality.

This module tests the bin_sf_1d function which is the main entry point
for 1D structure function calculations with binning.
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from pyturbo_sf.one_dimensional import bin_sf_1d


# =============================================================================
# Fixtures for test datasets
# =============================================================================

@pytest.fixture
def dataset_1d_scalar():
    """Create a 1D dataset with scalar field for testing."""
    n = 64  # Power of 2 for good bootstrapping
    x = np.linspace(0, 10, n)
    
    # Create sinusoidal data
    scalar1 = np.sin(2 * np.pi * x / 10)
    scalar2 = np.cos(2 * np.pi * x / 10)
    
    ds = xr.Dataset(
        data_vars={
            "scalar1": ("x", scalar1),
            "scalar2": ("x", scalar2),
        },
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_1d_with_mask():
    """Create a 1D dataset with mask for testing."""
    n = 64
    x = np.linspace(0, 10, n)
    
    scalar = np.sin(2 * np.pi * x / 10)
    # Mask: 1 for valid, 0 for invalid
    mask = np.ones(n, dtype=int)
    mask[::4] = 0  # Mark every 4th point as invalid
    
    ds = xr.Dataset(
        data_vars={
            "scalar1": ("x", scalar),
            "mask": ("x", mask),
        },
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_1d_time():
    """Create a 1D dataset with time dimension for testing."""
    n = 64
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    times = np.array([base_time + timedelta(hours=i) for i in range(n)])
    
    # Create data with trend
    t_numeric = np.arange(n)
    scalar1 = np.sin(t_numeric * 0.1)
    
    ds = xr.Dataset(
        data_vars={
            "scalar1": ("time", scalar1),
        },
        coords={"time": times}
    )
    return ds


@pytest.fixture
def dataset_1d_small():
    """Create a small 1D dataset for edge case testing."""
    n = 20
    x = np.linspace(0, 10, n)
    scalar = np.sin(2 * np.pi * x / 10)
    
    ds = xr.Dataset(
        data_vars={"scalar1": ("x", scalar)},
        coords={"x": x}
    )
    return ds


@pytest.fixture
def linear_bins():
    """Create linear bin edges."""
    return np.linspace(0.1, 5, 10)


@pytest.fixture
def log_bins():
    """Create logarithmic bin edges."""
    return np.logspace(-1, 1, 10)


# =============================================================================
# Tests for bin_sf_1d function
# =============================================================================

class TestBinSF1DBasic:
    """Basic tests for bin_sf_1d function."""
    
    def test_basic_scalar_function(self, dataset_1d_scalar, linear_bins):
        """Test basic scalar structure function binning."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            step_nbootstrap=5,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert 'std_error' in result.data_vars
        assert 'ci_upper' in result.data_vars
        assert 'ci_lower' in result.data_vars
        assert 'point_counts' in result.data_vars
        
        # Check coordinates
        assert 'bin' in result.coords
        assert len(result.bin) == len(linear_bins) - 1
        
        # Check attributes
        assert result.attrs['function_type'] == 'scalar'
        assert result.attrs['order'] == '2'
        
    def test_scalar_scalar_function(self, dataset_1d_scalar, linear_bins):
        """Test scalar-scalar structure function binning."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1", "scalar2"],
            order=(2, 1),
            bins=bins,
            bootsize=16,
            fun='scalar_scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            step_nbootstrap=5,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert result.attrs['function_type'] == 'scalar_scalar'
        
    def test_logarithmic_bins(self, dataset_1d_scalar, log_bins):
        """Test with logarithmic bin spacing."""
        bins = {"x": log_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['bin_type'] == 'logarithmic'
        
    def test_linear_bins_attribute(self, dataset_1d_scalar, linear_bins):
        """Test that linear bins are correctly identified."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['bin_type'] == 'linear'


class TestBinSF1DWithConditioning:
    """Tests for bin_sf_1d with conditioning."""
    
    def test_with_conditioning_bins(self, dataset_1d_with_mask, linear_bins):
        """Test binning with conditioning variable and bin range."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_with_mask,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            conditioning_var="mask",
            conditioning_bins=[0.5, 1.5],  # Select points where 0.5 <= mask < 1.5 (i.e., mask == 1)
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        
    def test_with_different_conditioning_range(self, dataset_1d_with_mask, linear_bins):
        """Test binning with different conditioning range."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_with_mask,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            conditioning_var="mask",
            conditioning_bins=[-0.5, 0.5],  # Select points where mask == 0
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)


class TestBinSF1DNoBootstrap:
    """Tests for bin_sf_1d with no bootstrappable dimensions."""
    
    def test_no_bootstrap_case(self, dataset_1d_small, linear_bins):
        """Test when bootsize equals data size (no bootstrapping)."""
        bins = {"x": linear_bins}
        n = len(dataset_1d_small.x)
        
        result = bin_sf_1d(
            ds=dataset_1d_small,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=n,  # Equal to data size - no bootstrapping possible
            fun='scalar',
            initial_nbootstrap=10,
            max_nbootstrap=20,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert 'std_error' in result.data_vars
        assert 'ci_lower' in result.data_vars
        assert 'ci_upper' in result.data_vars
        assert 'point_counts' in result.data_vars
        assert result.attrs['bootstrappable_dimensions'] == 'none'
        
        # CI should bracket the mean where both are valid
        sf = result['sf'].values
        ci_lower = result['ci_lower'].values
        ci_upper = result['ci_upper'].values
        
        valid = ~np.isnan(sf) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
        if np.any(valid):
            assert np.all(ci_lower[valid] <= sf[valid] + 1e-10)
            assert np.all(sf[valid] <= ci_upper[valid] + 1e-10)


class TestBinSF1DTimeDimension:
    """Tests for bin_sf_1d with time dimension."""
    



class TestBinSF1DErrorHandling:
    """Tests for error handling in bin_sf_1d."""
    
    def test_invalid_bins_type(self, dataset_1d_scalar):
        """Test that non-dict bins raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            bin_sf_1d(
                ds=dataset_1d_scalar,
                variables_names=["scalar1"],
                order=2,
                bins=np.linspace(0, 10, 10),  # Array instead of dict
                bootsize=16,
                n_jobs=1
            )
            
    def test_missing_dimension_in_bins(self, dataset_1d_scalar):
        """Test that missing dimension in bins raises error."""
        with pytest.raises(ValueError, match="Bins must be provided"):
            bin_sf_1d(
                ds=dataset_1d_scalar,
                variables_names=["scalar1"],
                order=2,
                bins={"y": np.linspace(0, 10, 10)},  # Wrong dimension
                bootsize=16,
                n_jobs=1
            )


class TestBinSF1DDifferentOrders:
    """Tests for different orders in bin_sf_1d."""
    
    def test_order_1(self, dataset_1d_scalar, linear_bins):
        """Test first-order structure function."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=1,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['order'] == '1'
        
    def test_order_3(self, dataset_1d_scalar, linear_bins):
        """Test third-order structure function."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=3,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['order'] == '3'
        
    def test_fractional_order(self, dataset_1d_scalar, linear_bins):
        """Test fractional order structure function."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2/3,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        # Order stored as string
        assert 'order' in result.attrs


class TestBinSF1DBootstrapParameters:
    """Tests for bootstrap parameter handling in bin_sf_1d."""
    
    def test_convergence_parameters(self, dataset_1d_scalar, linear_bins):
        """Test different convergence parameters."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=20,
            max_nbootstrap=50,
            step_nbootstrap=10,
            convergence_eps=0.05,
            n_jobs=1
        )
        
        assert 'convergence_eps' in result.attrs
        assert result.attrs['convergence_eps'] == 0.05
        
    def test_parallel_processing(self, dataset_1d_scalar, linear_bins):
        """Test with parallel processing enabled."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=2,  # Use 2 jobs
            backend='threading'
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['backend'] == 'threading'


class TestBinSF1DOutputStructure:
    """Tests for output dataset structure of bin_sf_1d."""
    
    def test_output_variables(self, dataset_1d_scalar, linear_bins):
        """Test that output contains all expected variables."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        # Check data variables
        expected_vars = ['sf', 'std_error', 'ci_upper', 'ci_lower', 
                        'point_counts', 'density', 'nbootstraps', 'converged']
        for var in expected_vars:
            assert var in result.data_vars, f"Missing variable: {var}"
            
    def test_output_attributes(self, dataset_1d_scalar, linear_bins):
        """Test that output contains all expected attributes."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            confidence_interval=0.95,
            n_jobs=1
        )
        
        expected_attrs = ['bin_type', 'order', 'function_type', 'variables',
                         'confidence_level', 'convergence_eps',
                         'max_nbootstrap', 'initial_nbootstrap', 'backend']
        for attr in expected_attrs:
            assert attr in result.attrs, f"Missing attribute: {attr}"
            
    def test_bin_edges_stored(self, dataset_1d_scalar, linear_bins):
        """Test that bin edges are stored in output."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        # Check that bin edges are stored
        assert 'x_bins' in result.data_vars or 'x_bins' in result.coords


class TestBinSF1DNumericalResults:
    """Tests for numerical properties of bin_sf_1d results."""
    
    def test_second_order_non_negative(self, dataset_1d_scalar, linear_bins):
        """Test that second-order SF is non-negative."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        sf_values = result['sf'].values
        valid_values = sf_values[~np.isnan(sf_values)]
        assert np.all(valid_values >= 0), "Second-order SF should be non-negative"
        
    def test_ci_bounds_relationship(self, dataset_1d_scalar, linear_bins):
        """Test that CI lower <= mean <= CI upper."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=10,
            max_nbootstrap=20,
            n_jobs=1
        )
        
        sf = result['sf'].values
        ci_lower = result['ci_lower'].values
        ci_upper = result['ci_upper'].values
        
        # Check only where all values are valid
        valid = ~np.isnan(sf) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
        
        assert np.all(ci_lower[valid] <= sf[valid] + 1e-10), "CI lower should be <= mean"
        assert np.all(sf[valid] <= ci_upper[valid] + 1e-10), "Mean should be <= CI upper"


# =============================================================================
# Tests for seed parameter (reproducibility)
# =============================================================================

class TestSeedParameter:
    """Tests for seed parameter functionality."""
    
    def test_bin_sf_1d_with_seed(self, dataset_1d_scalar, linear_bins):
        """Test that bin_sf_1d accepts seed parameter."""
        bins = {"x": linear_bins}
        
        result = bin_sf_1d(
            ds=dataset_1d_scalar,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=16,
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1,
            seed=42
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars


if __name__ == "__main__":
    pytest.main(["-v", __file__])
