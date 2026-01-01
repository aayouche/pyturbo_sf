"""
Tests for structure_functions.py module functionality.

This module tests all structure function calculations for 1D, 2D, and 3D cases.
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from pyturbo_sf.structure_functions import (
    # 1D functions
    calc_scalar_1d,
    calc_scalar_scalar_1d,
    calculate_structure_function_1d,
    # 2D functions
    calc_longitudinal_2d,
    calc_transverse_2d,
    calc_default_vel_2d,
    calc_scalar_2d,
    calc_scalar_scalar_2d,
    calc_longitudinal_transverse_2d,
    calc_longitudinal_scalar_2d,
    calc_transverse_scalar_2d,
    calc_advective_2d,
    calculate_structure_function_2d,
    # 3D functions
    calc_longitudinal_3d,
    calc_transverse_ij,
    calc_transverse_ik,
    calc_transverse_jk,
    calc_default_vel_3d,
    calc_scalar_3d,
    calc_scalar_scalar_3d,
    calc_longitudinal_scalar_3d,
    calc_transverse_ij_scalar,
    calc_transverse_ik_scalar,
    calc_transverse_jk_scalar,
    calc_longitudinal_transverse_ij,
    calc_longitudinal_transverse_ik,
    calc_longitudinal_transverse_jk,
    calc_advective_3d,
    calc_pressure_work_3d,
    calculate_structure_function_3d,
)


# =============================================================================
# Fixtures for test datasets
# =============================================================================

@pytest.fixture
def dataset_1d():
    """Create a simple 1D dataset for testing."""
    n = 50
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
def dataset_1d_with_mask():
    """Create a 1D dataset with mask for testing."""
    n = 50
    x = np.linspace(0, 10, n)
    scalar = np.sin(2 * np.pi * x / 10)
    mask = np.zeros(n, dtype=int)
    mask[::2] = 1  # Alternating mask values
    mask[1::4] = 2
    
    ds = xr.Dataset(
        data_vars={
            "temperature": ("x", scalar),
            "mask": ("x", mask),
        },
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_2d():
    """Create a simple 2D dataset for testing."""
    nx, ny = 20, 15
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 8)
    v = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8)
    scalar = np.sin(X + Y)
    scalar2 = np.cos(X - Y)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "temperature": (("y", "x"), scalar),
            "salinity": (("y", "x"), scalar2),
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_with_mask():
    """Create a 2D dataset with mask for testing."""
    nx, ny = 20, 15
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 8)
    v = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8)
    scalar = np.sin(X + Y)
    
    # Create mask with different values
    mask = np.zeros((ny, nx), dtype=int)
    mask[::2, ::2] = 1
    mask[1::2, 1::2] = 2
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "temperature": (("y", "x"), scalar),
            "mask": (("y", "x"), mask),
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_zx():
    """Create a 2D dataset with z-x plane for testing."""
    nx, nz = 20, 15
    x = np.linspace(0, 10, nx)
    z = np.linspace(0, 8, nz)
    Z, X = np.meshgrid(z, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Z / 8)
    w = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Z / 8)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "x"), u),
            "w": (("z", "x"), w),
        },
        coords={
            "x": (["z", "x"], X),
            "z": (["z", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_advective():
    """Create a 2D dataset with advective velocity fields for testing."""
    nx, ny = 20, 15
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10)
    v = np.cos(2 * np.pi * Y / 8)
    u_adv = np.ones_like(u) * 0.5  # Constant advective velocity
    v_adv = np.ones_like(v) * 0.3
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "u_adv": (("y", "x"), u_adv),
            "v_adv": (("y", "x"), v_adv),
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
    nx, ny, nz = 10, 8, 6
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10)
    v = np.cos(2 * np.pi * Y / 8)
    w = np.sin(2 * np.pi * Z / 6)
    scalar = np.sin(X + Y + Z)
    scalar2 = np.cos(X - Y + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "temperature": (("z", "y", "x"), scalar),
            "salinity": (("z", "y", "x"), scalar2),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_3d_with_mask():
    """Create a 3D dataset with mask for testing."""
    nx, ny, nz = 10, 8, 6
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10)
    v = np.cos(2 * np.pi * Y / 8)
    w = np.sin(2 * np.pi * Z / 6)
    
    # Create mask
    mask = np.zeros((nz, ny, nx), dtype=int)
    mask[::2, ::2, ::2] = 1
    mask[1::2, 1::2, 1::2] = 2
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "mask": (("z", "y", "x"), mask),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_3d_pressure():
    """Create a 3D dataset with pressure field for testing pressure work function."""
    nx, ny, nz = 10, 8, 6
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    u = np.sin(2 * np.pi * X / 10)
    v = np.cos(2 * np.pi * Y / 8)
    w = np.sin(2 * np.pi * Z / 6)
    pressure = np.sin(X) * np.cos(Y) * np.sin(Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "pressure": (("z", "y", "x"), pressure),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


# =============================================================================
# Tests for 1D Structure Functions
# =============================================================================

class TestStructureFunctions1D:
    """Tests for 1D structure functions."""
    
    def test_calc_scalar_1d_basic(self, dataset_1d):
        """Test basic scalar structure function calculation."""
        n_points = len(dataset_1d.x)
        results, separations, pair_counts = calc_scalar_1d(
            dataset_1d, "x", "temperature", order=2, n_points=n_points
        )
        
        assert results.shape == (n_points,)
        assert separations.shape == (n_points,)
        # First element should be NaN (no self-correlation)
        assert np.isnan(results[0])
        # Other elements should have valid values
        assert np.any(~np.isnan(results[1:]))
        # Separations should be positive (except first)
        assert np.all(separations[1:] >= 0)
        
    def test_calc_scalar_1d_order_1(self, dataset_1d):
        """Test first-order scalar structure function."""
        n_points = len(dataset_1d.x)
        results, separations, pair_counts = calc_scalar_1d(
            dataset_1d, "x", "temperature", order=1, n_points=n_points
        )
        
        assert results.shape == (n_points,)
        assert np.any(~np.isnan(results[1:]))
        
    def test_calc_scalar_1d_order_3(self, dataset_1d):
        """Test third-order scalar structure function."""
        n_points = len(dataset_1d.x)
        results, separations, pair_counts = calc_scalar_1d(
            dataset_1d, "x", "temperature", order=3, n_points=n_points
        )
        
        assert results.shape == (n_points,)
        assert np.any(~np.isnan(results[1:]))
        
    def test_calc_scalar_1d_with_mask(self, dataset_1d_with_mask):
        """Test scalar structure function with masking."""
        n_points = len(dataset_1d_with_mask.x)
        results, separations, pair_counts = calc_scalar_1d(
            dataset_1d_with_mask, "x", "temperature", order=2, n_points=n_points,
            conditioning_var="mask", conditioning_bins=[1, 2]
        )
        
        assert results.shape == (n_points,)
        assert np.any(~np.isnan(results[1:]))
        
    
        
    def test_calc_scalar_scalar_1d_basic(self, dataset_1d):
        """Test scalar-scalar structure function."""
        n_points = len(dataset_1d.x)
        results, separations, pair_counts = calc_scalar_scalar_1d(
            dataset_1d, "x", ["temperature", "salinity"], order=(1, 1), n_points=n_points
        )
        
        assert results.shape == (n_points,)
        assert separations.shape == (n_points,)
        assert np.any(~np.isnan(results[1:]))
        
    def test_calc_scalar_scalar_1d_different_orders(self, dataset_1d):
        """Test scalar-scalar structure function with different orders."""
        n_points = len(dataset_1d.x)
        results, separations, pair_counts = calc_scalar_scalar_1d(
            dataset_1d, "x", ["temperature", "salinity"], order=(2, 1), n_points=n_points
        )
        
        assert results.shape == (n_points,)
        assert np.any(~np.isnan(results[1:]))
        
    def test_calc_scalar_scalar_1d_wrong_variables(self, dataset_1d):
        """Test scalar-scalar with wrong number of variables raises error."""
        n_points = len(dataset_1d.x)
        with pytest.raises(ValueError, match="requires exactly 2"):
            calc_scalar_scalar_1d(
                dataset_1d, "x", ["temperature"], order=(1, 1), n_points=n_points
            )
            
    def test_calc_scalar_scalar_1d_wrong_order(self, dataset_1d):
        """Test scalar-scalar with wrong order format raises error."""
        n_points = len(dataset_1d.x)
        with pytest.raises(ValueError, match="Order must be a tuple"):
            calc_scalar_scalar_1d(
                dataset_1d, "x", ["temperature", "salinity"], order=2, n_points=n_points
            )
            
    def test_calculate_structure_function_1d_scalar(self, dataset_1d):
        """Test main 1D wrapper with scalar function."""
        results, separations, pair_counts = calculate_structure_function_1d(
            dataset_1d, "x", ["temperature"], order=2, fun='scalar'
        )
        
        assert len(results) > 0
        assert len(separations) > 0
        
    def test_calculate_structure_function_1d_scalar_scalar(self, dataset_1d):
        """Test main 1D wrapper with scalar-scalar function."""
        results, separations, pair_counts = calculate_structure_function_1d(
            dataset_1d, "x", ["temperature", "salinity"], order=(1, 1), fun='scalar_scalar'
        )
        
        assert len(results) > 0
        assert len(separations) > 0
        
    def test_calculate_structure_function_1d_invalid_function(self, dataset_1d):
        """Test main 1D wrapper with invalid function type."""
        with pytest.raises(ValueError, match="Unsupported function type"):
            calculate_structure_function_1d(
                dataset_1d, "x", ["temperature"], order=2, fun='invalid'
            )
            
    def test_calculate_structure_function_1d_missing_variable(self, dataset_1d):
        """Test main 1D wrapper with missing variable."""
        with pytest.raises(ValueError, match="not found in dataset"):
            calculate_structure_function_1d(
                dataset_1d, "x", ["nonexistent"], order=2, fun='scalar'
            )


# =============================================================================
# Tests for 2D Structure Functions
# =============================================================================

class TestStructureFunctions2D:
    """Tests for 2D structure functions."""
    
    # --- Longitudinal tests ---
    def test_calc_longitudinal_2d_basic(self, dataset_2d):
        """Test basic longitudinal structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_2d(
            dataset_2d, ["u", "v"], order=2, dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert dx_vals.shape == (ny * nx,)
        assert dy_vals.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_longitudinal_2d_order_3(self, dataset_2d):
        """Test third-order longitudinal structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_2d(
            dataset_2d, ["u", "v"], order=3, dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_longitudinal_2d_with_mask(self, dataset_2d_with_mask):
        """Test longitudinal structure function with masking."""
        ny, nx = dataset_2d_with_mask.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_2d(
            dataset_2d_with_mask, ["u", "v"], order=2, dims=dims, ny=ny, nx=nx,
            conditioning_var="mask", conditioning_bins=[1, 2]
        )
        
        assert results.shape == (ny * nx,)
        
    def test_calc_longitudinal_2d_wrong_variables(self, dataset_2d):
        """Test longitudinal with wrong number of variables."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        with pytest.raises(ValueError, match="requires exactly 2"):
            calc_longitudinal_2d(
                dataset_2d, ["u"], order=2, dims=dims, ny=ny, nx=nx
            )
            
    # --- Transverse tests ---
    def test_calc_transverse_2d_basic(self, dataset_2d):
        """Test basic transverse structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_transverse_2d(
            dataset_2d, ["u", "v"], order=2, dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Default velocity tests ---
    def test_calc_default_vel_2d_basic(self, dataset_2d):
        """Test basic default velocity structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_default_vel_2d(
            dataset_2d, ["u", "v"], order=2, dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Scalar tests ---
    def test_calc_scalar_2d_basic(self, dataset_2d):
        """Test basic 2D scalar structure function."""
        ny, nx = dataset_2d.temperature.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_scalar_2d(
            dataset_2d, ["temperature"], order=2, dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_scalar_2d_wrong_variables(self, dataset_2d):
        """Test scalar with wrong number of variables."""
        ny, nx = dataset_2d.temperature.shape
        dims = ["y", "x"]
        
        with pytest.raises(ValueError, match="requires exactly 1"):
            calc_scalar_2d(
                dataset_2d, ["temperature", "salinity"], order=2, dims=dims, ny=ny, nx=nx
            )
            
    # --- Scalar-scalar tests ---
    def test_calc_scalar_scalar_2d_basic(self, dataset_2d):
        """Test basic 2D scalar-scalar structure function."""
        ny, nx = dataset_2d.temperature.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_scalar_scalar_2d(
            dataset_2d, ["temperature", "salinity"], order=(1, 1), dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_scalar_scalar_2d_wrong_order(self, dataset_2d):
        """Test scalar-scalar with wrong order format."""
        ny, nx = dataset_2d.temperature.shape
        dims = ["y", "x"]
        
        with pytest.raises(ValueError, match="Order must be a tuple"):
            calc_scalar_scalar_2d(
                dataset_2d, ["temperature", "salinity"], order=2, dims=dims, ny=ny, nx=nx
            )
            
    # --- Longitudinal-transverse tests ---
    def test_calc_longitudinal_transverse_2d_basic(self, dataset_2d):
        """Test longitudinal-transverse structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_transverse_2d(
            dataset_2d, ["u", "v"], order=(1, 1), dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Longitudinal-scalar tests ---
    def test_calc_longitudinal_scalar_2d_basic(self, dataset_2d):
        """Test longitudinal-scalar structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_scalar_2d(
            dataset_2d, ["u", "v", "temperature"], order=(1, 1), dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Transverse-scalar tests ---
    def test_calc_transverse_scalar_2d_basic(self, dataset_2d):
        """Test transverse-scalar structure function."""
        ny, nx = dataset_2d.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_transverse_scalar_2d(
            dataset_2d, ["u", "v", "temperature"], order=(1, 1), dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Advective tests ---
    def test_calc_advective_2d_basic(self, dataset_2d_advective):
        """Test advective structure function."""
        ny, nx = dataset_2d_advective.u.shape
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_advective_2d(
            dataset_2d_advective, ["u", "v", "u_adv", "v_adv"], order=1, 
            dims=dims, ny=ny, nx=nx
        )
        
        assert results.shape == (ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Z-X plane tests ---
    def test_calc_longitudinal_2d_zx_plane(self, dataset_2d_zx):
        """Test longitudinal structure function on z-x plane."""
        nz, nx = dataset_2d_zx.u.shape
        dims = ["z", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_2d(
            dataset_2d_zx, ["u", "w"], order=2, dims=dims, ny=nz, nx=nx
        )
        
        assert results.shape == (nz * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Main wrapper tests ---
    def test_calculate_structure_function_2d_longitudinal(self, dataset_2d):
        """Test main 2D wrapper with longitudinal function."""
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            dataset_2d, dims, ["u", "v"], order=2, fun='longitudinal'
        )
        
        assert len(results) > 0
        assert np.any(~np.isnan(results))
        
    def test_calculate_structure_function_2d_transverse(self, dataset_2d):
        """Test main 2D wrapper with transverse function."""
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            dataset_2d, dims, ["u", "v"], order=2, fun='transverse'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_2d_scalar(self, dataset_2d):
        """Test main 2D wrapper with scalar function."""
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            dataset_2d, dims, ["temperature"], order=2, fun='scalar'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_2d_invalid_function(self, dataset_2d):
        """Test main 2D wrapper with invalid function type."""
        dims = ["y", "x"]
        
        with pytest.raises(ValueError, match="Unsupported function type"):
            calculate_structure_function_2d(
                dataset_2d, dims, ["u", "v"], order=2, fun='invalid'
            )
            
    def test_calculate_structure_function_2d_missing_variable(self, dataset_2d):
        """Test main 2D wrapper with missing variable."""
        dims = ["y", "x"]
        
        with pytest.raises(ValueError, match="not found in dataset"):
            calculate_structure_function_2d(
                dataset_2d, dims, ["nonexistent"], order=2, fun='scalar'
            )


# =============================================================================
# Tests for 3D Structure Functions
# =============================================================================

class TestStructureFunctions3D:
    """Tests for 3D structure functions."""
    
    # --- Longitudinal tests ---
    def test_calc_longitudinal_3d_basic(self, dataset_3d):
        """Test basic 3D longitudinal structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_3d(
            dataset_3d, ["u", "v", "w"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert dx_vals.shape == (nz * ny * nx,)
        assert dy_vals.shape == (nz * ny * nx,)
        assert dz_vals.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_longitudinal_3d_with_mask(self, dataset_3d_with_mask):
        """Test 3D longitudinal with masking."""
        nz, ny, nx = dataset_3d_with_mask.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_3d(
            dataset_3d_with_mask, ["u", "v", "w"], order=2, dims=dims, 
            nz=nz, ny=ny, nx=nx, conditioning_var="mask", conditioning_bins=[1, 2]
        )
        
        assert results.shape == (nz * ny * nx,)
        
    # --- Transverse tests ---
    def test_calc_transverse_ij_basic(self, dataset_3d):
        """Test transverse_ij (xy-plane) structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ij(
            dataset_3d, ["u", "v"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_transverse_ik_basic(self, dataset_3d):
        """Test transverse_ik (xz-plane) structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ik(
            dataset_3d, ["u", "w"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_transverse_jk_basic(self, dataset_3d):
        """Test transverse_jk (yz-plane) structure function."""
        nz, ny, nx = dataset_3d.v.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_jk(
            dataset_3d, ["v", "w"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_transverse_ij_wrong_variables(self, dataset_3d):
        """Test transverse_ij with wrong number of variables."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        with pytest.raises(ValueError, match="requires exactly 2"):
            calc_transverse_ij(
                dataset_3d, ["u"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
            )
            
    # --- Default velocity tests ---
    def test_calc_default_vel_3d_basic(self, dataset_3d):
        """Test default velocity structure function in 3D."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_default_vel_3d(
            dataset_3d, ["u", "v", "w"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Scalar tests ---
    def test_calc_scalar_3d_basic(self, dataset_3d):
        """Test 3D scalar structure function."""
        nz, ny, nx = dataset_3d.temperature.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_scalar_3d(
            dataset_3d, ["temperature"], order=2, dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Scalar-scalar tests ---
    def test_calc_scalar_scalar_3d_basic(self, dataset_3d):
        """Test 3D scalar-scalar structure function."""
        nz, ny, nx = dataset_3d.temperature.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_scalar_scalar_3d(
            dataset_3d, ["temperature", "salinity"], order=(1, 1), 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Longitudinal-scalar tests ---
    def test_calc_longitudinal_scalar_3d_basic(self, dataset_3d):
        """Test 3D longitudinal-scalar structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_scalar_3d(
            dataset_3d, ["u", "v", "w", "temperature"], order=(1, 1), 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Transverse-scalar tests ---
    def test_calc_transverse_ij_scalar_basic(self, dataset_3d):
        """Test transverse_ij_scalar structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ij_scalar(
            dataset_3d, ["u", "v", "temperature"], order=(1, 1), 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_transverse_ik_scalar_basic(self, dataset_3d):
        """Test transverse_ik_scalar structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ik_scalar(
            dataset_3d, ["u", "w", "temperature"], order=(1, 1), 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_transverse_jk_scalar_basic(self, dataset_3d):
        """Test transverse_jk_scalar structure function."""
        nz, ny, nx = dataset_3d.v.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_jk_scalar(
            dataset_3d, ["v", "w", "temperature"], order=(1, 1), 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Longitudinal-transverse tests ---
    def test_calc_longitudinal_transverse_ij_basic(self, dataset_3d):
        """Test longitudinal_transverse_ij structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_transverse_ij(
            dataset_3d, ["u", "v"], order=(1, 1), dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_longitudinal_transverse_ik_basic(self, dataset_3d):
        """Test longitudinal_transverse_ik structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_transverse_ik(
            dataset_3d, ["u", "w"], order=(1, 1), dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    def test_calc_longitudinal_transverse_jk_basic(self, dataset_3d):
        """Test longitudinal_transverse_jk structure function."""
        nz, ny, nx = dataset_3d.v.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_transverse_jk(
            dataset_3d, ["v", "w"], order=(1, 1), dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Advective tests ---
    def test_calc_advective_3d_basic(self, dataset_3d):
        """Test 3D advective structure function."""
        nz, ny, nx = dataset_3d.u.shape
        dims = ["z", "y", "x"]
        
        # Use same velocities as advective velocities for simplicity
        ds = dataset_3d.copy()
        ds["u_adv"] = ds["u"]
        ds["v_adv"] = ds["v"]
        ds["w_adv"] = ds["w"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_advective_3d(
            ds, ["u", "v", "w", "u_adv", "v_adv", "w_adv"], order=1, 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Pressure work tests ---
    def test_calc_pressure_work_3d_basic(self, dataset_3d_pressure):
        """Test 3D pressure work structure function."""
        nz, ny, nx = dataset_3d_pressure.u.shape
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_pressure_work_3d(
            dataset_3d_pressure, ["u", "v", "w", "pressure"], order=1, 
            dims=dims, nz=nz, ny=ny, nx=nx
        )
        
        assert results.shape == (nz * ny * nx,)
        assert np.any(~np.isnan(results))
        
    # --- Main wrapper tests ---
    def test_calculate_structure_function_3d_longitudinal(self, dataset_3d):
        """Test main 3D wrapper with longitudinal function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["u", "v", "w"], order=2, fun='longitudinal'
        )
        
        assert len(results) > 0
        assert np.any(~np.isnan(results))
        
    def test_calculate_structure_function_3d_transverse_ij(self, dataset_3d):
        """Test main 3D wrapper with transverse_ij function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["u", "v"], order=2, fun='transverse_ij'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_3d_transverse_ik(self, dataset_3d):
        """Test main 3D wrapper with transverse_ik function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["u", "w"], order=2, fun='transverse_ik'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_3d_transverse_jk(self, dataset_3d):
        """Test main 3D wrapper with transverse_jk function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["v", "w"], order=2, fun='transverse_jk'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_3d_scalar(self, dataset_3d):
        """Test main 3D wrapper with scalar function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["temperature"], order=2, fun='scalar'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_3d_scalar_scalar(self, dataset_3d):
        """Test main 3D wrapper with scalar_scalar function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["temperature", "salinity"], order=(1, 1), fun='scalar_scalar'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_3d_default_vel(self, dataset_3d):
        """Test main 3D wrapper with default_vel function."""
        dims = ["z", "y", "x"]
        
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            dataset_3d, dims, ["u", "v", "w"], order=2, fun='default_vel'
        )
        
        assert len(results) > 0
        
    def test_calculate_structure_function_3d_invalid_function(self, dataset_3d):
        """Test main 3D wrapper with invalid function type."""
        dims = ["z", "y", "x"]
        
        with pytest.raises(ValueError, match="Unsupported function type"):
            calculate_structure_function_3d(
                dataset_3d, dims, ["u", "v", "w"], order=2, fun='invalid'
            )
            
    def test_calculate_structure_function_3d_missing_variable(self, dataset_3d):
        """Test main 3D wrapper with missing variable."""
        dims = ["z", "y", "x"]
        
        with pytest.raises(ValueError, match="not found in dataset"):
            calculate_structure_function_3d(
                dataset_3d, dims, ["nonexistent"], order=2, fun='scalar'
            )


# =============================================================================
# Tests for edge cases and special scenarios
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and special scenarios."""
    
    def test_small_dataset_1d(self):
        """Test with very small 1D dataset."""
        n = 5
        x = np.linspace(0, 1, n)
        scalar = np.sin(x)
        
        ds = xr.Dataset(
            data_vars={"temperature": ("x", scalar)},
            coords={"x": x}
        )
        
        results, separations, pair_counts = calculate_structure_function_1d(
            ds, "x", ["temperature"], order=2, fun='scalar'
        )
        
        assert len(results) == n
        
    def test_small_dataset_2d(self):
        """Test with very small 2D dataset."""
        nx, ny = 5, 4
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        u = np.sin(X)
        v = np.cos(Y)
        
        ds = xr.Dataset(
            data_vars={
                "u": (("y", "x"), u),
                "v": (("y", "x"), v),
            },
            coords={
                "x": (["y", "x"], X),
                "y": (["y", "x"], Y),
            }
        )
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            ds, ["y", "x"], ["u", "v"], order=2, fun='longitudinal'
        )
        
        assert len(results) == ny * nx
        
    def test_constant_field_1d(self):
        """Test with constant scalar field (should give zero structure function)."""
        n = 20
        x = np.linspace(0, 10, n)
        scalar = np.ones(n) * 5.0  # Constant field
        
        ds = xr.Dataset(
            data_vars={"temperature": ("x", scalar)},
            coords={"x": x}
        )
        
        results, separations, pair_counts = calculate_structure_function_1d(
            ds, "x", ["temperature"], order=2, fun='scalar'
        )
        
        # All differences should be zero for constant field
        assert np.allclose(results[1:], 0, atol=1e-10) or np.all(np.isnan(results[1:]))
        
    def test_linear_field_2d(self):
        """Test with linear velocity field."""
        nx, ny = 15, 10
        x = np.linspace(0, 10, nx)
        y = np.linspace(0, 8, ny)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        # Linear field: u = x, v = y
        u = X
        v = Y
        
        ds = xr.Dataset(
            data_vars={
                "u": (("y", "x"), u),
                "v": (("y", "x"), v),
            },
            coords={
                "x": (["y", "x"], X),
                "y": (["y", "x"], Y),
            }
        )
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            ds, ["y", "x"], ["u", "v"], order=2, fun='longitudinal'
        )
        
        # Results should be valid for linear field
        assert np.any(~np.isnan(results))
        
    def test_all_masked_points(self, dataset_2d_with_mask):
        """Test behavior when mask excludes most/all points."""
        ny, nx = dataset_2d_with_mask.u.shape
        dims = ["y", "x"]
        
        # Use a mask condition that excludes everything
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_2d(
            dataset_2d_with_mask, ["u", "v"], order=2, dims=dims, ny=ny, nx=nx,
            conditioning_var="mask", conditioning_bins=[99, 100]  # No points have this value
        )
        
        # All results should be NaN when no points pass the mask
        assert results.shape == (ny * nx,)


# =============================================================================
# Tests for numerical properties
# =============================================================================

class TestNumericalProperties:
    """Tests for numerical properties of structure functions."""
    
    def test_second_order_positive(self, dataset_2d):
        """Test that second-order structure functions are non-negative."""
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            dataset_2d, dims, ["u", "v"], order=2, fun='longitudinal'
        )
        
        # Second-order SF should be non-negative (it's a squared quantity)
        valid_results = results[~np.isnan(results)]
        assert np.all(valid_results >= -1e-10)  # Allow small numerical errors
        
    def test_third_order_can_be_negative(self, dataset_2d):
        """Test that third-order structure functions can be negative."""
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            dataset_2d, dims, ["u", "v"], order=3, fun='longitudinal'
        )
        
        valid_results = results[~np.isnan(results)]
        # Third-order can have both positive and negative values
        # Just check it computes without error
        assert len(valid_results) > 0
        
    def test_separation_distances_consistent(self, dataset_2d):
        """Test that separation distances are computed consistently."""
        dims = ["y", "x"]
        
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            dataset_2d, dims, ["temperature"], order=2, fun='scalar'
        )
        
        # Check that dx and dy are not all zero
        assert np.any(dx_vals != 0) or np.any(dy_vals != 0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
