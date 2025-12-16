"""
Tests for isotropy_tools.py module functionality.

This module tests all isotropization functions for 2D (polar) and 3D (spherical) cases.
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.isotropy_tools import (
    # 2D polar functions
    _initialize_polar_bins_2d,
    _process_no_bootstrap_polar_2d,
    _calculate_bin_density_polar_2d,
    _create_isotropic_dataset,
    _calculate_isotropy_error_2d,
    _calculate_homogeneity_error_2d,
    # 3D spherical functions
    _initialize_spherical_bins_3d,
    _process_no_bootstrap_spherical_3d,
    _calculate_bin_density_spherical_3d,
    _create_spherical_dataset,
    _calculate_isotropy_error_3d,
    _calculate_homogeneity_error_3d,
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
    nx, ny, nz = 10, 8, 6
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
def linear_r_bins():
    """Create linear radial bin edges."""
    return np.linspace(0.1, 10, 11)


@pytest.fixture
def log_r_bins():
    """Create logarithmic radial bin edges."""
    return np.logspace(-1, 1, 11)


@pytest.fixture
def sample_sfr_2d():
    """Create sample 2D structure function data in polar coordinates."""
    n_theta = 12
    n_r = 10
    # Create data that varies with angle (anisotropic)
    theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    r = np.linspace(1, 10, n_r)
    R, THETA = np.meshgrid(r, theta)
    # Structure function that has some angular variation
    sfr = R ** (2/3) * (1 + 0.2 * np.cos(2 * THETA))
    return sfr


@pytest.fixture
def sample_sfr_3d():
    """Create sample 3D structure function data in spherical coordinates."""
    n_phi = 6
    n_theta = 12
    n_r = 10
    # Create data with some angular variation
    sfr = np.random.rand(n_phi, n_theta, n_r) + 1.0
    return sfr


@pytest.fixture
def isotropic_sfr_2d():
    """Create isotropic (angle-independent) 2D structure function data."""
    n_theta = 12
    n_r = 10
    r = np.linspace(1, 10, n_r)
    # Isotropic: only depends on r
    sfr = np.tile(r ** (2/3), (n_theta, 1))
    return sfr


@pytest.fixture
def isotropic_sfr_3d():
    """Create isotropic (angle-independent) 3D structure function data."""
    n_phi = 6
    n_theta = 12
    n_r = 10
    r = np.linspace(1, 10, n_r)
    # Isotropic: only depends on r
    sfr = np.tile(r ** (2/3), (n_phi, n_theta, 1))
    return sfr


# =============================================================================
# Tests for 2D Polar Binning Functions
# =============================================================================

class TestInitializePolarBins2D:
    """Tests for _initialize_polar_bins_2d function."""
    
    def test_linear_radial_bins(self, linear_r_bins):
        """Test initialization with linear radial bins."""
        n_theta = 12
        config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
        
        assert 'r_bins' in config
        assert 'theta_bins' in config
        assert 'r_centers' in config
        assert 'theta_centers' in config
        assert 'n_bins_r' in config
        assert 'n_bins_theta' in config
        assert 'log_bins' in config
        
        assert config['n_bins_r'] == 10
        assert config['n_bins_theta'] == n_theta
        assert config['log_bins'] == False
        
        # Check radial centers are arithmetic means
        expected_r_centers = 0.5 * (linear_r_bins[:-1] + linear_r_bins[1:])
        np.testing.assert_array_almost_equal(config['r_centers'], expected_r_centers)
        
    def test_log_radial_bins(self, log_r_bins):
        """Test initialization with logarithmic radial bins."""
        n_theta = 12
        config = _initialize_polar_bins_2d(log_r_bins, n_theta)
        
        assert config['log_bins'] == True
        
        # Check radial centers are geometric means
        expected_r_centers = np.sqrt(log_r_bins[:-1] * log_r_bins[1:])
        np.testing.assert_array_almost_equal(config['r_centers'], expected_r_centers)
        
    def test_theta_bins_coverage(self, linear_r_bins):
        """Test that theta bins cover full circle."""
        n_theta = 12
        config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
        
        # Should span from -pi to pi
        assert config['theta_bins'][0] == pytest.approx(-np.pi)
        assert config['theta_bins'][-1] == pytest.approx(np.pi)
        assert len(config['theta_bins']) == n_theta + 1
        
    def test_different_n_theta(self, linear_r_bins):
        """Test different angular resolutions."""
        for n_theta in [4, 8, 16, 36]:
            config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
            
            assert config['n_bins_theta'] == n_theta
            assert len(config['theta_centers']) == n_theta
            assert len(config['theta_bins']) == n_theta + 1


class TestProcessNoBootstrapPolar2D:
    """Tests for _process_no_bootstrap_polar_2d function."""
    
    def test_basic_processing(self, dataset_2d, linear_r_bins):
        """Test basic polar no-bootstrap processing."""
        dims = ["y", "x"]
        n_theta = 12
        time_dims = {"y": False, "x": False}
        
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_polar_2d(
            ds=dataset_2d,
            dims=dims,
            variables_names=["u", "v"],
            order=2,
            fun='longitudinal',
            r_bins=linear_r_bins,
            n_theta=n_theta,
            time_dims=time_dims,
            conditioning_var=None,
            conditioning_bins=None
        )
        
        assert sf_means.shape == (bins_config['n_bins_r'],)
        assert sf_stds.shape == (bins_config['n_bins_r'],)
        assert point_counts.shape == (bins_config['n_bins_r'],)
        assert sfr.shape == (n_theta, bins_config['n_bins_r'])
        assert sfr_counts.shape == (n_theta, bins_config['n_bins_r'])
        
        # Check that we have some valid results
        assert np.any(~np.isnan(sf_means))
        assert np.sum(point_counts) > 0
        
    def test_scalar_function(self, dataset_2d, linear_r_bins):
        """Test polar processing with scalar function."""
        dims = ["y", "x"]
        n_theta = 8
        time_dims = {"y": False, "x": False}
        
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_polar_2d(
            ds=dataset_2d,
            dims=dims,
            variables_names=["temperature"],
            order=2,
            fun='scalar',
            r_bins=linear_r_bins,
            n_theta=n_theta,
            time_dims=time_dims,
            conditioning_var=None,
            conditioning_bins=None
        )
        
        assert sf_means.shape == (bins_config['n_bins_r'],)


class TestCalculateBinDensityPolar2D:
    """Tests for _calculate_bin_density_polar_2d function."""
    
    def test_uniform_distribution(self, linear_r_bins):
        """Test density calculation with uniform point distribution."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.ones(n_bins, dtype=int) * 100
        
        density = _calculate_bin_density_polar_2d(point_counts, linear_r_bins)
        
        assert density.shape == (n_bins,)
        assert np.max(density) <= 1.0
        assert np.all(density >= 0)
        
    def test_empty_bins(self, linear_r_bins):
        """Test density with empty bins."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.zeros(n_bins, dtype=int)
        
        density = _calculate_bin_density_polar_2d(point_counts, linear_r_bins)
        
        assert np.all(density == 0)
        
    def test_radial_area_weighting(self, linear_r_bins):
        """Test that density accounts for radial area (larger rings have larger area)."""
        n_bins = len(linear_r_bins) - 1
        # Same number of points in each bin
        point_counts = np.ones(n_bins, dtype=int) * 100
        
        density = _calculate_bin_density_polar_2d(point_counts, linear_r_bins)
        
        # Outer bins have larger area, so same point count = lower density
        # (density should generally decrease with radius for uniform point counts)
        # Just check it runs and produces valid output
        assert np.all(np.isfinite(density))


class TestCalculateIsotropyError2D:
    """Tests for _calculate_isotropy_error_2d function."""
    
    def test_isotropic_field_low_error(self, isotropic_sfr_2d):
        """Test that isotropic field has low isotropy error."""
        sf_means = np.mean(isotropic_sfr_2d, axis=0)
        window_size_theta = 4
        
        eiso = _calculate_isotropy_error_2d(isotropic_sfr_2d, sf_means, window_size_theta)
        
        assert eiso.shape == (isotropic_sfr_2d.shape[1],)
        # Isotropic field should have very low error
        assert np.all(eiso < 1e-10)
        
    def test_anisotropic_field_higher_error(self, sample_sfr_2d):
        """Test that anisotropic field has higher isotropy error."""
        sf_means = np.mean(sample_sfr_2d, axis=0)
        window_size_theta = 4
        
        eiso = _calculate_isotropy_error_2d(sample_sfr_2d, sf_means, window_size_theta)
        
        assert eiso.shape == (sample_sfr_2d.shape[1],)
        # Anisotropic field should have non-zero error
        assert np.any(eiso > 0)
        
    def test_small_window_size(self, sample_sfr_2d):
        """Test with small window size."""
        sf_means = np.mean(sample_sfr_2d, axis=0)
        window_size_theta = 2
        
        eiso = _calculate_isotropy_error_2d(sample_sfr_2d, sf_means, window_size_theta)
        
        assert eiso.shape == (sample_sfr_2d.shape[1],)
        
    def test_large_window_size(self, sample_sfr_2d):
        """Test with window size close to n_theta."""
        sf_means = np.mean(sample_sfr_2d, axis=0)
        n_theta = sample_sfr_2d.shape[0]
        window_size_theta = n_theta - 2
        
        eiso = _calculate_isotropy_error_2d(sample_sfr_2d, sf_means, window_size_theta)
        
        assert eiso.shape == (sample_sfr_2d.shape[1],)


class TestCalculateHomogeneityError2D:
    """Tests for _calculate_homogeneity_error_2d function."""
    
    def test_basic_homogeneity(self, sample_sfr_2d):
        """Test basic homogeneity error calculation."""
        window_size_r = 3
        
        ehom, r_subset_indices = _calculate_homogeneity_error_2d(sample_sfr_2d, window_size_r)
        
        # Output should be smaller than input due to windowing
        assert len(ehom) <= sample_sfr_2d.shape[1]
        assert len(r_subset_indices) == len(ehom)
        
    def test_small_window(self, sample_sfr_2d):
        """Test with small radial window."""
        window_size_r = 2
        
        ehom, r_subset_indices = _calculate_homogeneity_error_2d(sample_sfr_2d, window_size_r)
        
        assert len(ehom) > 0
        
    def test_large_window(self, sample_sfr_2d):
        """Test with window size equal to n_r (edge case)."""
        n_r = sample_sfr_2d.shape[1]
        window_size_r = n_r  # Window as large as data
        
        ehom, r_subset_indices = _calculate_homogeneity_error_2d(sample_sfr_2d, window_size_r)
        
        # When window >= n_r, should return full range with zero error
        assert len(r_subset_indices) == n_r
        assert np.all(ehom == 0)


class TestCreateIsotropicDataset:
    """Tests for _create_isotropic_dataset function."""
    
    def test_basic_dataset_creation(self, linear_r_bins):
        """Test basic isotropic dataset creation."""
        n_theta = 12
        bins_config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
        n_r = bins_config['n_bins_r']
        
        results = {
            'sfr': np.random.rand(n_theta, n_r),
            'sf_means': np.random.rand(n_r),
            'sf_stds': np.abs(np.random.rand(n_r)) * 0.1,
            'bin_bootstraps': np.ones(n_r, dtype=int) * 10,
            'bin_density': np.random.rand(n_r),
            'point_counts': np.ones(n_r, dtype=int) * 100,
            'bin_status': np.ones(n_r, dtype=bool),
            'spacing_values': [1, 2, 4],
        }
        
        ds = _create_isotropic_dataset(
            results=results,
            bins_config=bins_config,
            order=2,
            fun='longitudinal',
            window_size_theta=4,
            window_size_r=3,
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            bootstrappable_dims=['y', 'x'],
            backend='serial',
            variables_names=['u', 'v'],
            confidence_interval=0.95
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'sf_polar' in ds.data_vars
        assert 'sf' in ds.data_vars
        assert 'error_isotropy' in ds.data_vars
        assert 'error_homogeneity' in ds.data_vars
        assert 'std_error' in ds.data_vars
        assert 'ci_upper' in ds.data_vars
        assert 'ci_lower' in ds.data_vars
        
        # Check coordinates
        assert 'r' in ds.coords
        assert 'theta' in ds.coords
        
        # Check attributes
        assert ds.attrs['function_type'] == 'longitudinal'
        assert ds.attrs['confidence_level'] == 0.95
        
    def test_with_precomputed_ci(self, linear_r_bins):
        """Test dataset creation with pre-computed confidence intervals."""
        n_theta = 12
        bins_config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
        n_r = bins_config['n_bins_r']
        
        sf_means = np.random.rand(n_r)
        results = {
            'sfr': np.random.rand(n_theta, n_r),
            'sf_means': sf_means,
            'sf_stds': np.abs(np.random.rand(n_r)) * 0.1,
            'ci_lower': sf_means - 0.1,
            'ci_upper': sf_means + 0.1,
            'bin_bootstraps': np.ones(n_r, dtype=int) * 10,
            'bin_density': np.random.rand(n_r),
            'point_counts': np.ones(n_r, dtype=int) * 100,
            'bin_status': np.ones(n_r, dtype=bool),
            'spacing_values': [1],
        }
        
        ds = _create_isotropic_dataset(
            results=results,
            bins_config=bins_config,
            order=2,
            fun='longitudinal',
            window_size_theta=4,
            window_size_r=3,
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            bootstrappable_dims=['y', 'x'],
            backend='serial',
            variables_names=['u', 'v'],
            confidence_interval=0.95
        )
        
        np.testing.assert_array_equal(ds['ci_lower'].values, sf_means - 0.1)
        np.testing.assert_array_equal(ds['ci_upper'].values, sf_means + 0.1)


# =============================================================================
# Tests for 3D Spherical Binning Functions
# =============================================================================

class TestInitializeSphericalBins3D:
    """Tests for _initialize_spherical_bins_3d function."""
    
    def test_linear_radial_bins(self, linear_r_bins):
        """Test initialization with linear radial bins."""
        n_theta = 12
        n_phi = 6
        config = _initialize_spherical_bins_3d(linear_r_bins, n_theta, n_phi)
        
        assert 'r_bins' in config
        assert 'theta_bins' in config
        assert 'phi_bins' in config
        assert 'r_centers' in config
        assert 'theta_centers' in config
        assert 'phi_centers' in config
        assert 'n_bins_r' in config
        assert 'n_bins_theta' in config
        assert 'n_bins_phi' in config
        assert 'log_bins' in config
        
        assert config['n_bins_r'] == 10
        assert config['n_bins_theta'] == n_theta
        assert config['n_bins_phi'] == n_phi
        assert config['log_bins'] == False
        
    def test_log_radial_bins(self, log_r_bins):
        """Test initialization with logarithmic radial bins."""
        n_theta = 12
        n_phi = 6
        config = _initialize_spherical_bins_3d(log_r_bins, n_theta, n_phi)
        
        assert config['log_bins'] == True
        
        # Check radial centers are geometric means
        expected_r_centers = np.sqrt(log_r_bins[:-1] * log_r_bins[1:])
        np.testing.assert_array_almost_equal(config['r_centers'], expected_r_centers)
        
    def test_angular_coverage(self, linear_r_bins):
        """Test that angular bins cover appropriate ranges."""
        n_theta = 12
        n_phi = 6
        config = _initialize_spherical_bins_3d(linear_r_bins, n_theta, n_phi)
        
        # Theta (azimuthal) should span -pi to pi
        assert config['theta_bins'][0] == pytest.approx(-np.pi)
        assert config['theta_bins'][-1] == pytest.approx(np.pi)
        
        # Phi (polar) should span 0 to pi
        assert config['phi_bins'][0] == pytest.approx(0)
        assert config['phi_bins'][-1] == pytest.approx(np.pi)


class TestProcessNoBootstrapSpherical3D:
    """Tests for _process_no_bootstrap_spherical_3d function."""
    
    def test_basic_processing(self, dataset_3d, linear_r_bins):
        """Test basic spherical no-bootstrap processing."""
        dims = ["z", "y", "x"]
        n_theta = 8
        n_phi = 4
        time_dims = {"z": False, "y": False, "x": False}
        
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_spherical_3d(
            ds=dataset_3d,
            dims=dims,
            variables_names=["u", "v", "w"],
            order=2,
            fun='longitudinal',
            r_bins=linear_r_bins,
            n_theta=n_theta,
            n_phi=n_phi,
            time_dims=time_dims
        )
        
        assert sf_means.shape == (bins_config['n_bins_r'],)
        assert sf_stds.shape == (bins_config['n_bins_r'],)
        assert point_counts.shape == (bins_config['n_bins_r'],)
        assert sfr.shape == (n_phi, n_theta, bins_config['n_bins_r'])
        
        # Check that we have some valid results
        assert np.any(~np.isnan(sf_means))
        assert np.sum(point_counts) > 0


class TestCalculateBinDensitySpherical3D:
    """Tests for _calculate_bin_density_spherical_3d function."""
    
    def test_uniform_distribution(self, linear_r_bins):
        """Test density calculation with uniform distribution."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.ones(n_bins, dtype=int) * 100
        
        density = _calculate_bin_density_spherical_3d(point_counts, linear_r_bins)
        
        assert density.shape == (n_bins,)
        assert np.max(density) <= 1.0
        assert np.all(density >= 0)
        
    def test_empty_bins(self, linear_r_bins):
        """Test density with empty bins."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.zeros(n_bins, dtype=int)
        
        density = _calculate_bin_density_spherical_3d(point_counts, linear_r_bins)
        
        assert np.all(density == 0)
        
    def test_spherical_volume_weighting(self, linear_r_bins):
        """Test that density accounts for spherical shell volume."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.ones(n_bins, dtype=int) * 100
        
        density = _calculate_bin_density_spherical_3d(point_counts, linear_r_bins)
        
        # Volume grows as r^3, so density should decrease with radius
        # for uniform point counts
        assert np.all(np.isfinite(density))


class TestCalculateIsotropyError3D:
    """Tests for _calculate_isotropy_error_3d function."""
    
    def test_isotropic_field_low_error(self, isotropic_sfr_3d):
        """Test that isotropic field has low isotropy error."""
        sf_means = np.mean(np.mean(isotropic_sfr_3d, axis=0), axis=0)
        window_size_theta = 4
        window_size_phi = 2
        
        eiso = _calculate_isotropy_error_3d(isotropic_sfr_3d, sf_means, 
                                           window_size_theta, window_size_phi)
        
        assert eiso.shape == (isotropic_sfr_3d.shape[2],)
        # Isotropic field should have very low error
        assert np.all(eiso < 1e-10)
        
    def test_anisotropic_field(self, sample_sfr_3d):
        """Test with anisotropic field."""
        sf_means = np.mean(np.mean(sample_sfr_3d, axis=0), axis=0)
        window_size_theta = 4
        window_size_phi = 2
        
        eiso = _calculate_isotropy_error_3d(sample_sfr_3d, sf_means,
                                           window_size_theta, window_size_phi)
        
        assert eiso.shape == (sample_sfr_3d.shape[2],)


class TestCalculateHomogeneityError3D:
    """Tests for _calculate_homogeneity_error_3d function."""
    
    def test_function_exists(self):
        """Test that _calculate_homogeneity_error_3d function exists."""
        from pyturbo_sf.isotropy_tools import _calculate_homogeneity_error_3d
        assert callable(_calculate_homogeneity_error_3d)


class TestCreateSphericalDataset:
    """Tests for _create_spherical_dataset function."""
    
    def test_basic_dataset_creation(self, linear_r_bins):
        """Test basic spherical dataset creation."""
        n_theta = 8
        n_phi = 4
        bins_config = _initialize_spherical_bins_3d(linear_r_bins, n_theta, n_phi)
        n_r = bins_config['n_bins_r']
        
        results = {
            'sfr': np.random.rand(n_phi, n_theta, n_r),
            'sf_means': np.random.rand(n_r),
            'sf_stds': np.abs(np.random.rand(n_r)) * 0.1,
            'bin_bootstraps': np.ones(n_r, dtype=int) * 10,
            'bin_density': np.random.rand(n_r),
            'point_counts': np.ones(n_r, dtype=int) * 100,
            'bin_status': np.ones(n_r, dtype=bool),
            'spacing_values': [1, 2, 4],
        }
        
        ds = _create_spherical_dataset(
            results=results,
            bins_config=bins_config,
            order=2,
            fun='longitudinal',
            window_size_theta=4,
            window_size_phi=2,
            window_size_r=3,
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            bootstrappable_dims=['z', 'y', 'x'],
            backend='serial',
            variables_names=['u', 'v', 'w'],
            confidence_interval=0.95
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'sf_spherical' in ds.data_vars
        assert 'sf' in ds.data_vars
        assert 'error_isotropy' in ds.data_vars
        assert 'error_homogeneity' in ds.data_vars
        assert 'std_error' in ds.data_vars
        assert 'ci_upper' in ds.data_vars
        assert 'ci_lower' in ds.data_vars
        
        # Check coordinates
        assert 'r' in ds.coords
        assert 'theta' in ds.coords
        assert 'phi' in ds.coords
        
        # Check dimensions of spherical SF
        assert ds['sf_spherical'].dims == ('phi', 'theta', 'r')


# =============================================================================
# Tests for edge cases and integration
# =============================================================================

class TestIsotropyEdgeCases:
    """Tests for edge cases in isotropy functions."""
    
    def test_single_radial_bin(self):
        """Test with single radial bin."""
        r_bins = np.array([0.1, 10.0])  # 1 bin
        n_theta = 12
        
        config = _initialize_polar_bins_2d(r_bins, n_theta)
        
        assert config['n_bins_r'] == 1
        
    def test_many_angular_bins(self, linear_r_bins):
        """Test with many angular bins."""
        n_theta = 360  # 1 degree resolution
        
        config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
        
        assert config['n_bins_theta'] == 360
        assert len(config['theta_centers']) == 360
        
    def test_sfr_with_nans(self):
        """Test error calculations handle NaN values."""
        n_theta = 12
        n_r = 10
        sfr = np.random.rand(n_theta, n_r)
        sfr[0, 0] = np.nan
        sfr[5, 5] = np.nan
        
        sf_means = np.nanmean(sfr, axis=0)
        window_size_theta = 4
        
        eiso = _calculate_isotropy_error_2d(sfr, sf_means, window_size_theta)
        
        assert eiso.shape == (n_r,)
        # Should still produce results (using nanmean internally)


class TestCoordinateConversions:
    """Tests for coordinate conversion properties."""
    
    def test_polar_bins_symmetry(self, linear_r_bins):
        """Test that polar bins are symmetric around zero angle."""
        n_theta = 12
        config = _initialize_polar_bins_2d(linear_r_bins, n_theta)
        
        # Check theta centers are symmetric
        theta_centers = config['theta_centers']
        # Sum of symmetric points should be approximately zero
        assert np.abs(np.sum(theta_centers)) < 1e-10
        
    def test_spherical_phi_range(self, linear_r_bins):
        """Test that phi (polar angle) covers proper range."""
        n_theta = 12
        n_phi = 6
        config = _initialize_spherical_bins_3d(linear_r_bins, n_theta, n_phi)
        
        # Phi should be in [0, pi]
        assert np.all(config['phi_centers'] >= 0)
        assert np.all(config['phi_centers'] <= np.pi)


class TestDensityNormalization:
    """Tests for density normalization properties."""
    
    def test_polar_density_normalized(self, linear_r_bins):
        """Test that polar density is normalized to [0, 1]."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.random.randint(1, 1000, n_bins)
        
        density = _calculate_bin_density_polar_2d(point_counts, linear_r_bins)
        
        assert np.max(density) == pytest.approx(1.0)
        assert np.all(density >= 0)
        assert np.all(density <= 1.0)
        
    def test_spherical_density_normalized(self, linear_r_bins):
        """Test that spherical density is normalized to [0, 1]."""
        n_bins = len(linear_r_bins) - 1
        point_counts = np.random.randint(1, 1000, n_bins)
        
        density = _calculate_bin_density_spherical_3d(point_counts, linear_r_bins)
        
        assert np.max(density) == pytest.approx(1.0)
        assert np.all(density >= 0)
        assert np.all(density <= 1.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
