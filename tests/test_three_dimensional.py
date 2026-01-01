"""
Tests for three_dimensional.py module functionality.

This module tests the main entry point functions for 3D structure function calculations:
- bin_sf_3d: 3D Cartesian binning
- get_isotropic_sf_3d: Spherical (radial) binning with isotropy metrics

Note: 3D energy flux (Bessel) decomposition is not yet implemented.
Only 2D energy flux is available via two_dimensional.get_energy_flux_2d().
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.three_dimensional import (
    bin_sf_3d,
    get_isotropic_sf_3d
)


# =============================================================================
# Fixtures for test datasets
# =============================================================================

@pytest.fixture
def dataset_3d():
    """Create a 3D dataset for testing with 3D coordinates."""
    nx, ny, nz = 16, 12, 10
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = np.transpose(X, (2, 1, 0))
    Y = np.transpose(Y, (2, 1, 0))
    Z = np.transpose(Z, (2, 1, 0))
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 8) * np.sin(2 * np.pi * Z / 6)
    v = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8) * np.cos(2 * np.pi * Z / 6)
    w = np.sin(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8) * np.sin(2 * np.pi * Z / 6)
    scalar1 = np.sin(X + Y + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "scalar1": (("z", "y", "x"), scalar1),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_3d_simple():
    """Create a simple 3D dataset for binning tests with 3D coordinates."""
    nx, ny, nz = 12, 10, 8
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    z = np.arange(nz, dtype=float)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = np.transpose(X, (2, 1, 0))
    Y = np.transpose(Y, (2, 1, 0))
    Z = np.transpose(Z, (2, 1, 0))
    
    u = 1.0 + 0.1 * X + 0.05 * Y + 0.02 * Z
    v = 0.5 + 0.05 * X + 0.1 * Y + 0.02 * Z
    w = 0.3 + 0.02 * X + 0.02 * Y + 0.1 * Z
    scalar1 = X + Y + Z
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "scalar1": (("z", "y", "x"), scalar1),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_3d_with_conditioning():
    """Create a 3D dataset with conditioning variable for testing."""
    nx, ny, nz = 12, 10, 8
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = np.transpose(X, (2, 1, 0))
    Y = np.transpose(Y, (2, 1, 0))
    Z = np.transpose(Z, (2, 1, 0))
    
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    v = np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.sin(X) * np.sin(Y) * np.sin(Z)
    vorticity = np.random.randn(nz, ny, nx) * 0.5 + 1.0
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "vorticity": (("z", "y", "x"), vorticity),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_3d_small():
    """Create a small 3D dataset for edge case testing with 3D coordinates."""
    nx, ny, nz = 8, 6, 5
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    z = np.linspace(0, 6, nz)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = np.transpose(X, (2, 1, 0))
    Y = np.transpose(Y, (2, 1, 0))
    Z = np.transpose(Z, (2, 1, 0))
    
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    v = np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.sin(X) * np.sin(Y) * np.sin(Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds



@pytest.fixture
def linear_bins_3d():
    """Create linear bin edges for 3D."""
    return {
        "x": np.linspace(0.1, 5, 5),
        "y": np.linspace(0.1, 4, 4),
        "z": np.linspace(0.1, 3, 4),
    }


@pytest.fixture
def radial_bins():
    """Create radial bin edges for isotropic functions."""
    return {"r": np.linspace(0.1, 5, 6)}


@pytest.fixture
def log_radial_bins():
    """Create logarithmic radial bin edges."""
    return {"r": np.logspace(-0.5, 0.7, 6)}


# =============================================================================
# Tests for bin_sf_3d function
# =============================================================================

class TestBinSF3DBasic:
    """Basic tests for bin_sf_3d function."""
    
    def test_longitudinal_function(self, dataset_3d_simple, linear_bins_3d):
        """Test longitudinal structure function binning."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert 'std_error' in result.data_vars
        assert 'ci_lower' in result.data_vars
        assert 'ci_upper' in result.data_vars
        assert 'x' in result.coords
        assert 'y' in result.coords
        assert 'z' in result.coords
        assert result['sf'].dims == ('z', 'y', 'x')
        assert result.attrs['function_type'] == 'longitudinal'
        
    def test_transverse_ij_function(self, dataset_3d_simple, linear_bins_3d):
        """Test transverse_ij structure function binning."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='transverse_ij',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'transverse_ij'
        
    def test_scalar_function(self, dataset_3d_simple, linear_bins_3d):
        """Test scalar structure function binning."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["scalar1"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'scalar'


class TestBinSF3DWithConditioning:
    """Tests for bin_sf_3d with conditioning."""
    
    def test_with_conditioning_single_bin(self, dataset_3d_with_conditioning, linear_bins_3d):
        """Test binning with conditioning variable and single bin."""
        result = bin_sf_3d(
            ds=dataset_3d_with_conditioning,
            variables_names=["u", "v", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            conditioning_var="vorticity",
            conditioning_bins=[0.5, 1.5],
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars


class TestBinSF3DErrorHandling:
    """Tests for error handling in bin_sf_3d."""
    
    def test_invalid_bins_type(self, dataset_3d_simple):
        """Test that non-dict bins raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            bin_sf_3d(
                ds=dataset_3d_simple,
                variables_names=["u", "v", "w"],
                order=2,
                bins=np.linspace(0, 10, 10),
                bootsize={"x": 4, "y": 4, "z": 3},
                n_jobs=1
            )
            
    def test_missing_dimension_in_bins(self, dataset_3d_simple):
        """Test that missing dimension in bins raises error."""
        with pytest.raises(ValueError, match="must be a dictionary with all dimensions"):
            bin_sf_3d(
                ds=dataset_3d_simple,
                variables_names=["u", "v", "w"],
                order=2,
                bins={"x": np.linspace(0, 10, 5), "y": np.linspace(0, 8, 4)},
                bootsize={"x": 4, "y": 4, "z": 3},
                n_jobs=1
            )


class TestBinSF3DOutputStructure:
    """Tests for output structure of bin_sf_3d."""
    
    def test_output_variables(self, dataset_3d_simple, linear_bins_3d):
        """Test that output contains expected variables."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        expected_vars = ['sf', 'std_error', 'ci_lower', 'ci_upper', 'point_counts', 
                        'density', 'nbootstraps', 'converged']
        for var in expected_vars:
            assert var in result.data_vars, f"Missing variable: {var}"


# =============================================================================
# Tests for get_isotropic_sf_3d function
# =============================================================================

class TestIsotropicSF3DBasic:
    """Basic tests for get_isotropic_sf_3d function."""
    
    def test_longitudinal_function(self, dataset_3d_simple, radial_bins):
        """Test isotropic longitudinal structure function."""
        result = get_isotropic_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v", "w"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=6,
            n_bins_phi=4,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert 'sf_spherical' in result.data_vars
        assert 'r' in result.coords
        assert 'theta' in result.coords
        assert 'phi' in result.coords
        
    def test_scalar_function(self, dataset_3d_simple, radial_bins):
        """Test isotropic scalar structure function."""
        result = get_isotropic_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["scalar1"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=6,
            n_bins_phi=4,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'scalar'


class TestIsotropicSF3DConditioning:
    """Tests for conditioning in get_isotropic_sf_3d."""
    
    def test_single_conditioning_bin(self, dataset_3d_with_conditioning, radial_bins):
        """Test isotropic SF with single conditioning bin."""
        result = get_isotropic_sf_3d(
            ds=dataset_3d_with_conditioning,
            variables_names=["u", "v", "w"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=6,
            n_bins_phi=4,
            conditioning_var="vorticity",
            conditioning_bins=[0.5, 1.5],
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars


class TestNumericalProperties:
    """Tests for numerical properties of results."""
    
    def test_second_order_non_negative_3d(self, dataset_3d_simple, linear_bins_3d):
        """Test that second-order SF is non-negative for 3D bins."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        sf_values = result['sf'].values
        valid_values = sf_values[~np.isnan(sf_values)]
        assert np.all(valid_values >= -1e-10), "Second-order SF should be non-negative"


class TestTransverseComponents:
    """Tests for different transverse structure function components in 3D."""
    
    def test_transverse_ik(self, dataset_3d_simple, linear_bins_3d):
        """Test transverse_ik structure function."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='transverse_ik',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['function_type'] == 'transverse_ik'
        
    def test_transverse_jk(self, dataset_3d_simple, linear_bins_3d):
        """Test transverse_jk structure function."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["v", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='transverse_jk',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['function_type'] == 'transverse_jk'


# =============================================================================
# Tests for seed parameter (reproducibility)
# =============================================================================

class TestSeedParameter:
    """Tests for seed parameter functionality."""
    
    def test_bin_sf_3d_with_seed(self, dataset_3d_simple, linear_bins_3d):
        """Test that bin_sf_3d accepts seed parameter."""
        result = bin_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v", "w"],
            order=2,
            bins=linear_bins_3d,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1,
            seed=42
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        
    def test_isotropic_sf_3d_with_seed(self, dataset_3d_simple, radial_bins):
        """Test that get_isotropic_sf_3d accepts seed parameter."""
        result = get_isotropic_sf_3d(
            ds=dataset_3d_simple,
            variables_names=["u", "v", "w"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 4, "y": 4, "z": 3},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=6,
            n_bins_phi=4,
            n_jobs=1,
            seed=42
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars


if __name__ == "__main__":
    pytest.main(["-v", __file__])
