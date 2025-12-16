"""
Tests for bessel_tools.py module functionality.

This module tests the Bessel/flux decomposition functions for 2D cases.
Note: 3D Bessel decomposition is not yet implemented.
"""

import pytest
import numpy as np
import xarray as xr
from scipy.special import jv

from pyturbo_sf.bessel_tools import (
    _validate_flux_function,
    _initialize_wavenumbers_2d,
    _initialize_r_bins_2d,
    _initialize_flux_config_2d,
    _compute_energy_flux_2d,
    _calculate_wavenumber_density_2d,
    VALID_FLUX_FUNCTIONS
)


# =============================================================================
# Fixtures for test datasets
# =============================================================================

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
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "scalar": (("y", "x"), scalar),
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def sample_wavenumbers():
    """Create sample wavenumbers for testing."""
    return np.logspace(-1, 1, 20)


@pytest.fixture
def sample_r_bins():
    """Create sample radial bins for testing."""
    return np.linspace(0.1, 5, 50)


# =============================================================================
# Tests for _validate_flux_function
# =============================================================================

class TestValidateFluxFunction:
    """Tests for _validate_flux_function."""
    
    def test_valid_advective(self):
        """Test validation of 'advective' function type (no exception)."""
        # Should not raise
        _validate_flux_function('advective')
        
    def test_valid_scalar_scalar(self):
        """Test validation of 'scalar_scalar' function type (no exception)."""
        # Should not raise
        _validate_flux_function('scalar_scalar')
        
    def test_invalid_function(self):
        """Test that invalid function type raises ValueError."""
        with pytest.raises(ValueError, match="Energy flux decomposition requires fun in"):
            _validate_flux_function('invalid_function')
            
    def test_longitudinal_not_valid(self):
        """Test that 'longitudinal' is not valid for flux calculations."""
        with pytest.raises(ValueError):
            _validate_flux_function('longitudinal')
            
    def test_valid_flux_functions_constant(self):
        """Test that VALID_FLUX_FUNCTIONS contains expected values."""
        assert 'advective' in VALID_FLUX_FUNCTIONS
        assert 'scalar_scalar' in VALID_FLUX_FUNCTIONS


# =============================================================================
# Tests for _initialize_wavenumbers_2d
# =============================================================================

class TestInitializeWavenumbers2D:
    """Tests for _initialize_wavenumbers_2d."""
    
    def test_array_input(self, dataset_2d):
        """Test with array input."""
        wavenumbers = np.logspace(-1, 1, 15)
        result = _initialize_wavenumbers_2d(wavenumbers, dataset_2d, ['y', 'x'])
        
        assert isinstance(result, dict)
        assert 'k' in result
        assert len(result['k']) == 15
        np.testing.assert_array_almost_equal(result['k'], wavenumbers)
        
    def test_dict_input(self, dataset_2d):
        """Test with dictionary input."""
        wavenumbers = {'k': np.logspace(-1, 1, 15)}
        result = _initialize_wavenumbers_2d(wavenumbers, dataset_2d, ['y', 'x'])
        
        assert isinstance(result, dict)
        assert 'k' in result
        assert len(result['k']) == 15
        
    def test_none_input(self, dataset_2d):
        """Test auto-generation when None is provided."""
        result = _initialize_wavenumbers_2d(None, dataset_2d, ['y', 'x'])
        
        assert isinstance(result, dict)
        assert 'k' in result
        assert len(result['k']) > 0
        
    def test_wavenumbers_are_positive(self, dataset_2d):
        """Test that generated wavenumbers are positive."""
        result = _initialize_wavenumbers_2d(None, dataset_2d, ['y', 'x'])
        assert np.all(result['k'] > 0)
        
    def test_result_contains_metadata(self, dataset_2d):
        """Test that result contains expected metadata."""
        result = _initialize_wavenumbers_2d(None, dataset_2d, ['y', 'x'])
        assert 'n_k' in result
        assert 'log_spaced' in result
        assert 'k_min' in result
        assert 'k_max' in result


# =============================================================================
# Tests for _initialize_r_bins_2d
# =============================================================================

class TestInitializeRBins2D:
    """Tests for _initialize_r_bins_2d."""
    
    def test_array_input(self, dataset_2d):
        """Test with array input."""
        r_bins = np.linspace(0.1, 5, 30)
        result = _initialize_r_bins_2d(r_bins, dataset_2d, ['y', 'x'])
        
        assert isinstance(result, dict)
        assert 'r_edges' in result
        assert 'r_centers' in result
        assert 'dr' in result
        
    def test_none_input(self, dataset_2d):
        """Test auto-generation when None is provided."""
        result = _initialize_r_bins_2d(None, dataset_2d, ['y', 'x'], n_r_bins=50)
        
        assert isinstance(result, dict)
        assert 'r_edges' in result
        assert len(result['r_edges']) == 51  # n_r_bins + 1 edges
        
    def test_r_centers_computed(self, dataset_2d):
        """Test that r_centers are properly computed."""
        r_bins = np.linspace(0.1, 5, 11)
        result = _initialize_r_bins_2d(r_bins, dataset_2d, ['y', 'x'])
        
        assert len(result['r_centers']) == 10  # n_edges - 1


# =============================================================================
# Tests for _initialize_flux_config_2d
# =============================================================================

class TestInitializeFluxConfig2D:
    """Tests for _initialize_flux_config_2d."""
    
    def test_basic_config(self, dataset_2d):
        """Test basic configuration creation."""
        wavenumbers = np.logspace(-1, 1, 10)
        k_config = _initialize_wavenumbers_2d(wavenumbers, dataset_2d, ['y', 'x'])
        r_config = _initialize_r_bins_2d(None, dataset_2d, ['y', 'x'], n_r_bins=50)
        
        result = _initialize_flux_config_2d(
            k_config['k'], r_config, n_bins_theta=36
        )
        
        assert isinstance(result, dict)
        assert 'k' in result
        assert 'r_edges' in result
        assert 'r_centers' in result
        assert 'dr' in result
        assert 'theta_bins' in result
        assert 'theta_centers' in result
        assert 'n_bins_theta' in result
        
    def test_theta_bins_count(self, dataset_2d):
        """Test that theta bins have correct count."""
        wavenumbers = np.logspace(-1, 1, 10)
        k_config = _initialize_wavenumbers_2d(wavenumbers, dataset_2d, ['y', 'x'])
        r_config = _initialize_r_bins_2d(None, dataset_2d, ['y', 'x'], n_r_bins=50)
        
        n_bins_theta = 24
        result = _initialize_flux_config_2d(
            k_config['k'], r_config, n_bins_theta=n_bins_theta
        )
        
        assert len(result['theta_centers']) == n_bins_theta


# =============================================================================
# Tests for _compute_energy_flux_2d
# =============================================================================

class TestComputeEnergyFlux2D:
    """Tests for _compute_energy_flux_2d."""
    
    def test_basic_computation(self):
        """Test basic energy flux computation."""
        # Create mock SF(r) data
        r_centers = np.linspace(0.5, 4.5, 20)
        dr = np.full(len(r_centers), r_centers[1] - r_centers[0])
        sf_r = r_centers ** (2/3)  # K41-like scaling
        
        k = np.logspace(-1, 0.5, 10)
        
        result = _compute_energy_flux_2d(sf_r, r_centers, dr, k)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(k)
        
    def test_flux_shape(self):
        """Test that flux has correct shape."""
        r_centers = np.linspace(0.5, 4.5, 30)
        dr = np.full(len(r_centers), r_centers[1] - r_centers[0])
        sf_r = np.sin(r_centers)
        
        k = np.logspace(-1, 1, 15)
        
        result = _compute_energy_flux_2d(sf_r, r_centers, dr, k)
        
        assert result.shape == (15,)


# =============================================================================
# Tests for _calculate_wavenumber_density_2d
# =============================================================================

class TestWavenumberDensity2D:
    """Tests for _calculate_wavenumber_density_2d."""
    
    def test_basic_computation(self):
        """Test basic wavenumber density computation."""
        point_counts = np.array([100, 150, 200, 180, 120])
        k = np.logspace(-1, 0.5, 5)
        
        result = _calculate_wavenumber_density_2d(point_counts, k)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == point_counts.shape
        
    def test_density_is_non_negative(self):
        """Test that density is non-negative."""
        point_counts = np.array([100, 150, 200, 180, 120])
        k = np.logspace(-1, 0.5, 5)
        
        result = _calculate_wavenumber_density_2d(point_counts, k)
        
        assert np.all(result >= 0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
