"""
Tests for two_dimensional.py module functionality.

This module tests the main entry point functions for 2D structure function calculations:
- bin_sf_2d: 2D Cartesian binning
- get_isotropic_sf_2d: Radial (polar) binning with isotropy metrics
- get_energy_flux_2d: Energy flux calculations (advective SF only)
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.two_dimensional import (
    bin_sf_2d,
    get_isotropic_sf_2d,
    get_energy_flux_2d
)


# =============================================================================
# Fixtures for test datasets
# =============================================================================

@pytest.fixture
def dataset_2d():
    """Create a 2D dataset for testing."""
    nx, ny = 32, 24
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y)
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 8)
    v = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8)
    # Advective velocities (for energy flux calculations)
    adv_u = u * 0.5 + np.random.randn(ny, nx) * 0.1
    adv_v = v * 0.5 + np.random.randn(ny, nx) * 0.1
    scalar1 = np.sin(X + Y)
    scalar2 = np.cos(X - Y)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "adv_u": (("y", "x"), adv_u),
            "adv_v": (("y", "x"), adv_v),
            "scalar1": (("y", "x"), scalar1),
            "scalar2": (("y", "x"), scalar2),
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_zx():
    """Create a 2D dataset with (z,x) dimensions."""
    nx, nz = 32, 24
    x = np.linspace(0, 10, nx)
    z = np.linspace(0, 8, nz)
    X, Z = np.meshgrid(x, z)
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Z / 8)
    w = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Z / 8)
    scalar = np.sin(X + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "x"), u),
            "w": (("z", "x"), w),
            "scalar": (("z", "x"), scalar),
        },
        coords={
            "x": (["z", "x"], X),
            "z": (["z", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_with_conditioning():
    """Create a 2D dataset with conditioning variable."""
    nx, ny = 32, 24
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y)
    
    u = np.sin(2 * np.pi * X / 10) * np.cos(2 * np.pi * Y / 8)
    v = np.cos(2 * np.pi * X / 10) * np.sin(2 * np.pi * Y / 8)
    vorticity = np.random.randn(ny, nx) * 0.5 + 1.0
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "vorticity": (("y", "x"), vorticity),
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_small():
    """Create a small 2D dataset for edge case testing."""
    nx, ny = 16, 12
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 8, ny)
    X, Y = np.meshgrid(x, y)
    
    u = np.sin(X) * np.cos(Y)
    v = np.cos(X) * np.sin(Y)
    
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
    return ds


@pytest.fixture
def linear_bins_2d():
    """Create linear bin edges for 2D."""
    return {
        "x": np.linspace(0.1, 5, 8),
        "y": np.linspace(0.1, 4, 6),
    }


@pytest.fixture
def log_bins_2d():
    """Create logarithmic bin edges for 2D."""
    return {
        "x": np.logspace(-1, 0.7, 8),
        "y": np.logspace(-1, 0.6, 6),
    }


@pytest.fixture
def radial_bins():
    """Create radial bin edges for isotropic functions."""
    return {"r": np.linspace(0.1, 5, 8)}


@pytest.fixture
def log_radial_bins():
    """Create logarithmic radial bin edges."""
    return {"r": np.logspace(-1, 0.7, 8)}


# =============================================================================
# Tests for bin_sf_2d function
# =============================================================================

class TestBinSF2DBasic:
    """Basic tests for bin_sf_2d function."""
    
    def test_longitudinal_function(self, dataset_2d, linear_bins_2d):
        """Test longitudinal structure function binning."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            step_nbootstrap=5,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert 'std_error' in result.data_vars
        assert 'ci_lower' in result.data_vars
        assert 'ci_upper' in result.data_vars
        assert 'x' in result.coords
        assert 'y' in result.coords
        assert result['sf'].dims == ('y', 'x')
        assert result.attrs['function_type'] == 'longitudinal'
        
    def test_transverse_function(self, dataset_2d, linear_bins_2d):
        """Test transverse structure function binning."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='transverse',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'transverse'
        
    def test_scalar_function(self, dataset_2d, linear_bins_2d):
        """Test scalar structure function binning."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["scalar1"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'scalar'
        
    def test_scalar_scalar_function(self, dataset_2d, linear_bins_2d):
        """Test scalar-scalar structure function binning."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["scalar1", "scalar2"],
            order=(2, 1),
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='scalar_scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'scalar_scalar'
        
    def test_zx_dimensions(self, dataset_2d_zx):
        """Test with (z, x) dimensions."""
        bins = {
            "x": np.linspace(0.1, 5, 6),
            "z": np.linspace(0.1, 4, 5),
        }
        
        result = bin_sf_2d(
            ds=dataset_2d_zx,
            variables_names=["u", "w"],
            order=2,
            bins=bins,
            bootsize={"x": 8, "z": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'x' in result.coords
        assert 'z' in result.coords


class TestBinSF2DWithConditioning:
    """Tests for bin_sf_2d with conditioning."""
    
    def test_with_conditioning_single_bin(self, dataset_2d_with_conditioning, linear_bins_2d):
        """Test binning with conditioning variable and single bin."""
        result = bin_sf_2d(
            ds=dataset_2d_with_conditioning,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            conditioning_var="vorticity",
            conditioning_bins=[0.5, 1.5],
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars


class TestBinSF2DErrorHandling:
    """Tests for error handling in bin_sf_2d."""
    
    def test_invalid_bins_type(self, dataset_2d):
        """Test that non-dict bins raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            bin_sf_2d(
                ds=dataset_2d,
                variables_names=["u", "v"],
                order=2,
                bins=np.linspace(0, 10, 10),
                bootsize={"x": 8, "y": 6},
                n_jobs=1
            )
            
    def test_missing_dimension_in_bins(self, dataset_2d):
        """Test that missing dimension in bins raises error."""
        with pytest.raises(ValueError, match="must be a dictionary with all dimensions"):
            bin_sf_2d(
                ds=dataset_2d,
                variables_names=["u", "v"],
                order=2,
                bins={"x": np.linspace(0, 10, 10)},
                bootsize={"x": 8, "y": 6},
                n_jobs=1
            )


class TestBinSF2DOutputStructure:
    """Tests for output structure of bin_sf_2d."""
    
    def test_output_variables(self, dataset_2d, linear_bins_2d):
        """Test that output contains expected variables."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
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
# Tests for get_isotropic_sf_2d function
# =============================================================================

class TestIsotropicSF2DBasic:
    """Basic tests for get_isotropic_sf_2d function."""
    
    def test_longitudinal_function(self, dataset_2d, radial_bins):
        """Test isotropic longitudinal structure function."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            window_size_theta=3,
            window_size_r=2,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        assert 'sf_polar' in result.data_vars
        assert 'error_isotropy' in result.data_vars
        assert 'error_homogeneity' in result.data_vars
        assert 'r' in result.coords
        assert 'theta' in result.coords
        assert result['sf_polar'].dims == ('theta', 'r')
        assert len(result.theta) == 8
        
    def test_scalar_function(self, dataset_2d, radial_bins):
        """Test isotropic scalar structure function."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["scalar1"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            fun='scalar',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert result.attrs['function_type'] == 'scalar'
        
    def test_log_radial_bins(self, dataset_2d, log_radial_bins):
        """Test with logarithmic radial bins."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=log_radial_bins,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1
        )
        
        assert result.attrs['bin_type'] == 'logarithmic'


class TestIsotropicSF2DConditioning:
    """Tests for conditioning in get_isotropic_sf_2d."""
    
    def test_single_conditioning_bin(self, dataset_2d_with_conditioning, radial_bins):
        """Test isotropic SF with single conditioning bin."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d_with_conditioning,
            variables_names=["u", "v"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            conditioning_var="vorticity",
            conditioning_bins=[0.5, 1.5],
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars


# =============================================================================
# Tests for get_energy_flux_2d function
# =============================================================================

class TestEnergyFlux2DBasic:
    """Basic tests for get_energy_flux_2d function."""
    
    def test_advective_function(self, dataset_2d):
        """Test energy flux with advective structure function."""
        wavenumbers = np.logspace(-0.5, 0.5, 10)
        
        result = get_energy_flux_2d(
            ds=dataset_2d,
            variables_names=["u", "v", "adv_u", "adv_v"],
            order=1,
            wavenumbers=wavenumbers,
            bootsize={"x": 8, "y": 6},
            fun='advective',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            window_size_theta=3,
            window_size_k=2,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'energy_flux' in result.data_vars
        assert 'k' in result.coords
        
    def test_auto_wavenumbers(self, dataset_2d):
        """Test automatic wavenumber generation."""
        result = get_energy_flux_2d(
            ds=dataset_2d,
            variables_names=["u", "v", "adv_u", "adv_v"],
            order=1,
            wavenumbers=None,
            bootsize={"x": 8, "y": 6},
            fun='advective',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'k' in result.coords
        
    def test_dict_wavenumbers(self, dataset_2d):
        """Test with dictionary wavenumber specification."""
        wavenumbers = {'k': np.logspace(-0.5, 0.5, 15)}
        
        result = get_energy_flux_2d(
            ds=dataset_2d,
            variables_names=["u", "v", "adv_u", "adv_v"],
            order=1,
            wavenumbers=wavenumbers,
            bootsize={"x": 8, "y": 6},
            fun='advective',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1
        )
        
        assert len(result.k) == 15
        
    def test_invalid_function_raises_error(self, dataset_2d):
        """Test that invalid function type raises ValueError."""
        with pytest.raises(ValueError, match="Energy flux decomposition requires"):
            get_energy_flux_2d(
                ds=dataset_2d,
                variables_names=["u", "v"],
                order=2,
                wavenumbers=np.logspace(-0.5, 0.5, 8),
                bootsize={"x": 8, "y": 6},
                fun='longitudinal',
                n_jobs=1
            )


# =============================================================================
# Tests for numerical properties
# =============================================================================

class TestNumericalProperties:
    """Tests for numerical properties of results."""
    
    def test_second_order_non_negative_2d(self, dataset_2d, linear_bins_2d):
        """Test that second-order SF is non-negative for 2D bins."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        sf_values = result['sf'].values
        valid_values = sf_values[~np.isnan(sf_values)]
        assert np.all(valid_values >= -1e-10), "Second-order SF should be non-negative"
        
    def test_second_order_non_negative_isotropic(self, dataset_2d, radial_bins):
        """Test that second-order SF is non-negative for isotropic."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1
        )
        
        sf_values = result['sf'].values
        valid_values = sf_values[~np.isnan(sf_values)]
        assert np.all(valid_values >= -1e-10), "Second-order SF should be non-negative"


# =============================================================================
# Tests for different orders
# =============================================================================

class TestDifferentOrders:
    """Tests for different structure function orders."""
    
    def test_order_1(self, dataset_2d, radial_bins):
        """Test first-order structure function."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=1,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            n_bins_theta=8,
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['order'] == '1'
        
    def test_order_3(self, dataset_2d, radial_bins):
        """Test third-order structure function."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=3,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            n_bins_theta=8,
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1
        )
        
        assert result.attrs['order'] == '3'


# =============================================================================
# Tests for seed parameter (reproducibility)
# =============================================================================

class TestSeedParameter:
    """Tests for seed parameter functionality."""
    
    def test_bin_sf_2d_with_seed(self, dataset_2d, linear_bins_2d):
        """Test that bin_sf_2d accepts seed parameter."""
        result = bin_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=linear_bins_2d,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_jobs=1,
            seed=42
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        
    def test_isotropic_sf_2d_with_seed(self, dataset_2d, radial_bins):
        """Test that get_isotropic_sf_2d accepts seed parameter."""
        result = get_isotropic_sf_2d(
            ds=dataset_2d,
            variables_names=["u", "v"],
            order=2,
            bins=radial_bins,
            bootsize={"x": 8, "y": 6},
            fun='longitudinal',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1,
            seed=42
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'sf' in result.data_vars
        
    def test_energy_flux_2d_with_seed(self, dataset_2d):
        """Test that get_energy_flux_2d accepts seed parameter."""
        wavenumbers = np.logspace(-0.5, 0.5, 8)
        
        result = get_energy_flux_2d(
            ds=dataset_2d,
            variables_names=["u", "v", "adv_u", "adv_v"],
            order=1,
            wavenumbers=wavenumbers,
            bootsize={"x": 8, "y": 6},
            fun='advective',
            initial_nbootstrap=5,
            max_nbootstrap=10,
            n_bins_theta=8,
            n_jobs=1,
            seed=42
        )
        
        assert isinstance(result, xr.Dataset)
        assert 'energy_flux' in result.data_vars


if __name__ == "__main__":
    pytest.main(["-v", __file__])
