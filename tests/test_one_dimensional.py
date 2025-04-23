"""
Tests for one_dimensional.py module functionality.
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from pyturbo_sf.one_dimensional import (
    calc_scalar_1d, calc_scalar_scalar_1d,
    calculate_structure_function_1d,
    monte_carlo_simulation_1d,
    bin_sf_1d
)


@pytest.fixture
def dataset_1d_scalar():
    """Create a 1D dataset with scalar field for testing."""
    # Create coordinates
    x = np.linspace(0, 10, 100)
    
    # Create some sample data
    scalar1 = np.sin(x)
    scalar2 = np.cos(x)
    
    ds = xr.Dataset(
        data_vars={
            "scalar1": ("x", scalar1),
            "scalar2": ("x", scalar2)
        },
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_1d_time():
    """Create a 1D dataset with time dimension for testing."""
    # Create time coordinates
    base_time = datetime(2023, 1, 1, 0, 0, 0)
    times = np.array([base_time + timedelta(hours=i) for i in range(100)])
    
    # Create some sample data with trend and noise
    t_numeric = np.arange(100)
    scalar1 = np.sin(t_numeric * 0.1) + 0.1 * np.random.randn(100)
    scalar2 = np.cos(t_numeric * 0.1) + 0.1 * np.random.randn(100)
    
    ds = xr.Dataset(
        data_vars={
            "scalar1": ("time", scalar1),
            "scalar2": ("time", scalar2)
        },
        coords={"time": times}
    )
    return ds


class TestCalcFunctions:
    
    def test_calc_scalar_1d(self, dataset_1d_scalar):
        """Test scalar structure function calculation in 1D."""
        # Get a subset of the dataset for faster testing
        subset = dataset_1d_scalar.isel(x=slice(0, 20))
        n_points = len(subset.x)
        
        # Calculate the structure function
        results, separations = calc_scalar_1d(
            subset=subset,
            dim="x", 
            variable_name="scalar1",
            order=2,
            n_points=n_points
        )
        
        # Check shape and basic properties
        assert len(results) == n_points
        assert len(separations) == n_points
        
        # First element should be NaN (self-correlation)
        assert np.isnan(results[0])
        
        # The rest should have finite values
        assert np.all(np.isfinite(results[1:]))
        assert np.all(np.isfinite(separations[1:]))
        
        # Separation distances should be increasing
        assert np.all(np.diff(separations[1:]) >= 0)
        
    def test_calc_scalar_scalar_1d(self, dataset_1d_scalar):
        """Test scalar-scalar structure function calculation in 1D."""
        # Get a subset of the dataset for faster testing
        subset = dataset_1d_scalar.isel(x=slice(0, 20))
        n_points = len(subset.x)
        
        # Calculate the structure function
        results, separations = calc_scalar_scalar_1d(
            subset=subset,
            dim="x", 
            variables_names=["scalar1", "scalar2"],
            order=(2, 1),  # First variable order 2, second order 1
            n_points=n_points
        )
        
        # Check shape and basic properties
        assert len(results) == n_points
        assert len(separations) == n_points
        
        # First element should be NaN (self-correlation)
        assert np.isnan(results[0])
        
        # The rest should have finite values
        assert np.all(np.isfinite(results[1:]))
        assert np.all(np.isfinite(separations[1:]))


class TestCalculateStructureFunction:
    
    def test_calculate_structure_function_1d_scalar(self, dataset_1d_scalar):
        """Test calculating structure function for scalar fields."""
        # Calculate structure function
        results, separations = calculate_structure_function_1d(
            ds=dataset_1d_scalar,
            dim="x",
            variables_names=["scalar1"],
            order=2,
            fun="scalar"
        )
        
        # Check shape and basic properties
        assert len(results) == len(dataset_1d_scalar.x)
        assert len(separations) == len(dataset_1d_scalar.x)
        
        # First element should be NaN (self-correlation)
        assert np.isnan(results[0])
        
        # Most values should be finite
        assert np.sum(np.isfinite(results)) > len(results) * 0.5
        assert np.sum(np.isfinite(separations)) > len(separations) * 0.5
        
    def test_calculate_structure_function_1d_scalar_scalar(self, dataset_1d_scalar):
        """Test calculating structure function for two scalar fields."""
        # Calculate structure function
        results, separations = calculate_structure_function_1d(
            ds=dataset_1d_scalar,
            dim="x",
            variables_names=["scalar1", "scalar2"],
            order=(1, 1),  # Same order for both
            fun="scalar_scalar"
        )
        
        # Check shape and basic properties
        assert len(results) == len(dataset_1d_scalar.x)
        assert len(separations) == len(dataset_1d_scalar.x)
        
        # First element should be NaN (self-correlation)
        assert np.isnan(results[0])
        
        # Most values should be finite
        assert np.sum(np.isfinite(results)) > len(results) * 0.5
        assert np.sum(np.isfinite(separations)) > len(separations) * 0.5
        
    def test_calculate_structure_function_1d_time(self, dataset_1d_time):
        """Test calculating structure function with time dimension."""
        # Calculate structure function
        results, separations = calculate_structure_function_1d(
            ds=dataset_1d_time,
            dim="time",
            variables_names=["scalar1"],
            order=2,
            fun="scalar"
        )
        
        # Check shape and basic properties
        assert len(results) == len(dataset_1d_time.time)
        assert len(separations) == len(dataset_1d_time.time)
        
        # First element should be NaN (self-correlation)
        assert np.isnan(results[0])
        
        # Most values should be finite
        assert np.sum(np.isfinite(results)) > len(results) * 0.5
        assert np.sum(np.isfinite(separations)) > len(separations) * 0.5
        
    def test_calculate_structure_function_1d_error(self, dataset_1d_scalar):
        """Test error cases for structure function calculation."""
        # Test unsupported function type
        with pytest.raises(ValueError):
            calculate_structure_function_1d(
                ds=dataset_1d_scalar,
                dim="x",
                variables_names=["scalar1"],
                order=2,
                fun="unsupported_type"
            )
            
        # Test wrong number of variables for scalar function
        with pytest.raises(ValueError):
            calculate_structure_function_1d(
                ds=dataset_1d_scalar,
                dim="x",
                variables_names=["scalar1", "scalar2"],
                order=2,
                fun="scalar"
            )
            
        # Test non-existent variable
        with pytest.raises(ValueError):
            calculate_structure_function_1d(
                ds=dataset_1d_scalar,
                dim="x",
                variables_names=["nonexistent"],
                order=2,
                fun="scalar"
            )


class TestMonteCarloSimulation:
    
    def test_monte_carlo_simulation_1d(self, dataset_1d_scalar):
        """Test Monte Carlo simulation for 1D structure functions."""
        # Reduce dataset size for faster testing
        dataset = dataset_1d_scalar.isel(x=slice(0, 30))
        
        # Setup parameters
        dim = "x"
        bootsize = 10
        nbootstrap = 3  # Small number for faster testing
        
        # Calculate adaptive spacings (needed for Monte Carlo simulation)
        data_shape = {dim: len(dataset[dim])}
        bootsize_dict = {dim: bootsize}
        
        from pyturbo_sf.core import (
            calculate_adaptive_spacings_1d,
            compute_boot_indexes_1d
        )
        
        # Single bootstrappable dimension
        num_bootstrappable = 1
        
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, num_bootstrappable
        )
        
        # Run Monte Carlo simulation with minimal iterations
        results, separations = monte_carlo_simulation_1d(
            ds=dataset,
            dim=dim,
            variables_names=["scalar1"],
            order=2,
            nbootstrap=nbootstrap,
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            fun="scalar",
            spacing=1,
            n_jobs=1  # Sequential processing for testing
        )
        
        # Check results
        assert len(results) == nbootstrap
        assert len(separations) == nbootstrap
        



class TestBinSF:
    
    def test_bin_sf_1d(self, dataset_1d_scalar):
        """Test the main binning function for 1D structure functions."""
        # Reduce dataset size for faster testing
        dataset = dataset_1d_scalar.isel(x=slice(0, 30))
        
        # Setup bin edges (logarithmic spacing)
        bin_edges = np.logspace(-2, 1, 10)
        bins = {"x": bin_edges}
        
        # Test with minimal number of bootstraps for speed
        binned_ds = bin_sf_1d(
            ds=dataset,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize=10,
            initial_nbootstrap=3,
            max_nbootstrap=5,
            step_nbootstrap=2,
            convergence_eps=0.2,  # Larger value for faster convergence
            n_jobs=1  # Sequential processing for testing
        )
        
        # Verify the result has expected structure
        assert isinstance(binned_ds, xr.Dataset)
        assert "sf" in binned_ds.data_vars
        assert "sf_std" in binned_ds.data_vars
        assert "bin" in binned_ds.coords
        
        # Bin centers should have the right length (one less than bin edges)
        assert len(binned_ds.bin) == len(bin_edges) - 1
        
        # Check that binned structure function contains valid values
        assert not np.all(np.isnan(binned_ds.sf))
        
        # Check that attributes are correctly set
        assert "bin_type" in binned_ds.attrs
        assert "order" in binned_ds.attrs
        assert "function_type" in binned_ds.attrs
        assert "variables" in binned_ds.attrs
        
        # Test with scalar-scalar function
        binned_ds = bin_sf_1d(
            ds=dataset,
            variables_names=["scalar1", "scalar2"],
            fun="scalar_scalar",
            order=(1, 1),
            bins=bins,
            bootsize=10,
            initial_nbootstrap=3,
            max_nbootstrap=5,
            step_nbootstrap=2,
            convergence_eps=0.2,  # Larger value for faster convergence
            n_jobs=1
        )
        
        # Verify scalar-scalar results
        assert isinstance(binned_ds, xr.Dataset)
        assert "sf" in binned_ds.data_vars
        assert not np.all(np.isnan(binned_ds.sf))


if __name__ == "__main__":
    pytest.main(["-v", "test_one_dimensional.py"])
