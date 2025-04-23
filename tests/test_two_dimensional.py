"""
Tests for two_dimensional.py module functionality.
"""

import pytest
import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view

from pyturbo_sf.two_dimensional import (
    calc_longitudinal_2d, calc_transverse_2d, calc_default_vel_2d,
    calc_scalar_2d, calc_scalar_scalar_2d, calc_longitudinal_transverse_2d,
    calc_longitudinal_scalar_2d, calc_transverse_scalar_2d,
    calculate_structure_function_2d,
    monte_carlo_simulation_2d,
    bin_sf_2d,
    get_isotropic_sf_2d
)


@pytest.fixture
def dataset_2d():
    """Create a 2D dataset for testing."""
    # Create coordinates (smaller grid for faster tests)
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 15)
    X, Y = np.meshgrid(x, y)
    
    # Create velocity components and scalars
    u = np.sin(X) * np.cos(Y)
    v = np.cos(X) * np.sin(Y)
    scalar1 = np.sin(X + Y)
    scalar2 = np.cos(X - Y)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "scalar1": (("y", "x"), scalar1),
            "scalar2": (("y", "x"), scalar2)
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_zx():
    """Create a 2D dataset with (z,x) dimensions for testing."""
    # Create coordinates (smaller grid for faster tests)
    x = np.linspace(0, 10, 20)
    z = np.linspace(0, 10, 15)
    X, Z = np.meshgrid(x, z)
    
    # Create velocity components and scalar
    u = np.sin(X) * np.cos(Z)
    w = np.cos(X) * np.sin(Z)
    scalar = np.sin(X + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "x"), u),
            "w": (("z", "x"), w),
            "scalar": (("z", "x"), scalar)
        },
        coords={
            "x": (["z", "x"], X),
            "z": (["z", "x"], Z),
        }
    )
    return ds




class TestCalcFunctions:
    
    def test_calc_longitudinal_2d(self, dataset_2d):
        """Test longitudinal structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.u.shape
        
        # Calculate longitudinal structure function
        results, dx, dy = calc_longitudinal_2d(
            subset=subset,
            variables_names=["u", "v"],
            order=2,
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite (some may be NaN from boundary effects)
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_transverse_2d(self, dataset_2d):
        """Test transverse structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.u.shape
        
        # Calculate transverse structure function
        results, dx, dy = calc_transverse_2d(
            subset=subset,
            variables_names=["u", "v"],
            order=2,
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_default_vel_2d(self, dataset_2d):
        """Test default velocity structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.u.shape
        
        # Calculate default velocity structure function
        results, dx, dy = calc_default_vel_2d(
            subset=subset,
            variables_names=["u", "v"],
            order=2,
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_scalar_2d(self, dataset_2d):
        """Test scalar structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.scalar1.shape
        
        # Calculate scalar structure function
        results, dx, dy = calc_scalar_2d(
            subset=subset,
            variables_names=["scalar1"],
            order=2,
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_scalar_scalar_2d(self, dataset_2d):
        """Test scalar-scalar structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.scalar1.shape
        
        # Calculate scalar-scalar structure function
        results, dx, dy = calc_scalar_scalar_2d(
            subset=subset,
            variables_names=["scalar1", "scalar2"],
            order=(2, 1),  # Order 2 for first scalar, 1 for second
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_longitudinal_transverse_2d(self, dataset_2d):
        """Test longitudinal-transverse structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.u.shape
        
        # Calculate longitudinal-transverse structure function
        results, dx, dy = calc_longitudinal_transverse_2d(
            subset=subset,
            variables_names=["u", "v"],
            order=(2, 1),  # Order 2 for longitudinal, 1 for transverse
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_longitudinal_scalar_2d(self, dataset_2d):
        """Test longitudinal-scalar structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.u.shape
        
        # Calculate longitudinal-scalar structure function
        results, dx, dy = calc_longitudinal_scalar_2d(
            subset=subset,
            variables_names=["u", "v", "scalar1"],
            order=(2, 1),  # Order 2 for longitudinal, 1 for scalar
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calc_transverse_scalar_2d(self, dataset_2d):
        """Test transverse-scalar structure function calculation in 2D."""
        # Get a small subset for faster testing
        subset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        ny, nx = subset.u.shape
        
        # Calculate transverse-scalar structure function
        results, dx, dy = calc_transverse_scalar_2d(
            subset=subset,
            variables_names=["u", "v", "scalar1"],
            order=(2, 1),  # Order 2 for transverse, 1 for scalar
            dims=["y", "x"],
            ny=ny, 
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0


class TestCalculateStructureFunction:
    
    def test_calculate_structure_function_2d(self, dataset_2d):
        """Test main structure function calculation in 2D."""
        # Test with longitudinal function
        results, dx, dy = calculate_structure_function_2d(
            ds=dataset_2d,
            dims=["y", "x"],
            variables_names=["u", "v"],
            order=2,
            fun="longitudinal"
        )
        
        # Check shapes
        ny, nx = dataset_2d.u.shape
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calculate_structure_function_2d_different_plane(self, dataset_2d_zx):
        """Test structure function calculation for (z,x) plane."""
        # Test with longitudinal function
        results, dx, dy = calculate_structure_function_2d(
            ds=dataset_2d_zx,
            dims=["z", "x"],
            variables_names=["u", "w"],
            order=2,
            fun="longitudinal"
        )
        
        # Check shapes
        nz, nx = dataset_2d_zx.u.shape
        assert results.shape == (nz * nx,)
        assert dx.shape == (nz * nx,)
        assert dy.shape == (nz * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calculate_structure_function_2d_scalar(self, dataset_2d):
        """Test structure function calculation for scalar field."""
        # Test with scalar function
        results, dx, dy = calculate_structure_function_2d(
            ds=dataset_2d,
            dims=["y", "x"],
            variables_names=["scalar1"],
            order=2,
            fun="scalar"
        )
        
        # Check shapes
        ny, nx = dataset_2d.scalar1.shape
        assert results.shape == (ny * nx,)
        assert dx.shape == (ny * nx,)
        assert dy.shape == (ny * nx,)
        
        # Check that most values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        
    def test_calculate_structure_function_2d_errors(self, dataset_2d):
        """Test error cases for structure function calculation."""
        # Test with unsupported function type
        with pytest.raises(ValueError):
            calculate_structure_function_2d(
                ds=dataset_2d,
                dims=["y", "x"],
                variables_names=["u", "v"],
                order=2,
                fun="unsupported_type"
            )
            
        # Test with non-existent variable
        with pytest.raises(ValueError):
            calculate_structure_function_2d(
                ds=dataset_2d,
                dims=["y", "x"],
                variables_names=["nonexistent"],
                order=2,
                fun="scalar"
            )


class TestMonteCarloSimulation:
    
    def test_monte_carlo_simulation_2d(self, dataset_2d):
        """Test Monte Carlo simulation for 2D structure functions."""
        # Use a very small subset for faster tests
        dataset = dataset_2d.isel(x=slice(0, 5), y=slice(0, 5))
        
        # Setup parameters
        dims = ["y", "x"]
        bootsize = {"y": 3, "x": 3}
        nbootstrap = 2  # Small number for faster testing
        
        # Calculate adaptive spacings (needed for Monte Carlo simulation)
        from pyturbo_sf.core import (
            setup_bootsize_2d,
            calculate_adaptive_spacings_2d,
            compute_boot_indexes_2d
        )
        
        data_shape = dict(dataset.sizes)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape, bootsize)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_2d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        # Run Monte Carlo simulation with minimal iterations
        results, dx_vals, dy_vals = monte_carlo_simulation_2d(
            ds=dataset,
            dims=dims,
            variables_names=["u", "v"],
            order=2,
            nbootstrap=nbootstrap,
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims,
            fun="longitudinal",
            spacing=1,
            n_jobs=1  # Sequential processing for testing
        )
        
        # Check results
        assert len(results) == nbootstrap
        assert len(dx_vals) == nbootstrap
        assert len(dy_vals) == nbootstrap
        
            
        # At least some values should be finite
        assert np.any(np.isfinite(results[0]))
        assert np.any(np.isfinite(dx_vals[0]))
        assert np.any(np.isfinite(dy_vals[0]))


class TestBinSF:
    
    def test_bin_sf_2d(self, dataset_2d):
        """Test the binning function for 2D structure functions."""
        # Use a small subset for faster tests
        dataset = dataset_2d.isel(x=slice(0, 8), y=slice(0, 6))
        
        # Setup bin edges
        bins_x = np.linspace(0, 5, 6) + 1.0e-6
        bins_y = np.linspace(0, 5, 6) + 1.0e-6
        bins = {"x": bins_x, "y": bins_y}
        
        # Test with minimal number of bootstraps for speed
        binned_ds = bin_sf_2d(
            ds=dataset,
            variables_names=["u", "v"],
            order=2,
            bins=bins,
            bootsize={"x": 3, "y": 3},
            fun="longitudinal",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            convergence_eps=0.5,  # Large value for faster convergence
            n_jobs=1  # Sequential processing for testing
        )
        
        # Verify the result has expected structure
        assert isinstance(binned_ds, xr.Dataset)
        assert "sf" in binned_ds.data_vars
        assert "sf_std" in binned_ds.data_vars
        assert "x" in binned_ds.coords
        assert "y" in binned_ds.coords
        
        # Check that binned structure function contains valid values
        assert not np.all(np.isnan(binned_ds.sf))
        
        # Check that attributes are correctly set
        assert "bin_type_x" in binned_ds.attrs
        assert "bin_type_y" in binned_ds.attrs
        assert "order" in binned_ds.attrs
        assert "function_type" in binned_ds.attrs
        assert "variables" in binned_ds.attrs
        
        # Test with a scalar function
        binned_ds = bin_sf_2d(
            ds=dataset,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize={"x": 3, "y": 3},
            fun="scalar",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        # Verify scalar results
        assert isinstance(binned_ds, xr.Dataset)
        assert "sf" in binned_ds.data_vars
        assert not np.all(np.isnan(binned_ds.sf))


class TestIsotropicSF:
    
    def test_get_isotropic_sf_2d(self, dataset_2d):
        """Test the isotropic structure function calculation for 2D."""
        # Use a small subset for faster tests
        dataset = dataset_2d.isel(x=slice(0, 8), y=slice(0, 8))
        
        # Setup radial bin edges
        r_bins = np.linspace(0, 5, 2) + 1e-6
        bins = {"r": r_bins}
        
        # Test with minimal number of bootstraps and angular bins for speed
        isotropic_ds = get_isotropic_sf_2d(
            ds=dataset,
            variables_names=["u", "v"],
            order=2,
            bins=bins,
            bootsize={"x": 4, "y": 4},
            fun="longitudinal",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            n_bins_theta=8,  # Small number of angular bins
            window_size_theta=3,
            window_size_r=2,
            convergence_eps=0.5,  # Large value for faster convergence
            n_jobs=1  # Sequential processing for testing
        )
        
        # Verify the result has expected structure
        assert isinstance(isotropic_ds, xr.Dataset)
        assert "sf" in isotropic_ds.data_vars
        assert "sf_polar" in isotropic_ds.data_vars
        assert "r" in isotropic_ds.coords
        assert "theta" in isotropic_ds.coords
        
        # Check dimensions of polar results
        assert isotropic_ds.sf_polar.dims == ('theta', 'r')
        assert len(isotropic_ds.r) == len(r_bins) - 1
        assert len(isotropic_ds.theta) == 8
        
        # Check that isotropic structure function contains valid values
        assert not np.all(np.isnan(isotropic_ds.sf))
        
        # Check that attributes are correctly set
        assert "bin_type" in isotropic_ds.attrs
        assert "order" in isotropic_ds.attrs
        assert "function_type" in isotropic_ds.attrs
        assert "variables" in isotropic_ds.attrs
        assert "window_size_theta" in isotropic_ds.attrs
        assert "window_size_r" in isotropic_ds.attrs
        
        # Test with a scalar function
        isotropic_ds = get_isotropic_sf_2d(
            ds=dataset,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize={"x": 4, "y": 4},
            fun="scalar",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            n_bins_theta=8,
            window_size_theta=3,
            window_size_r=2,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        # Verify scalar results
        assert isinstance(isotropic_ds, xr.Dataset)
        assert "sf" in isotropic_ds.data_vars
        assert not np.all(np.isnan(isotropic_ds.sf))


if __name__ == "__main__":
    pytest.main(["-v", "test_two_dimensional.py"])
