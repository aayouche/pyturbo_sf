"""
Tests for binning_tools.py module functionality.

This module tests all binning functions for 1D, 2D, and 3D cases.
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.binning_tools import (
    # 1D functions
    _initialize_1d_bins,
    _process_no_bootstrap_1d,
    _calculate_bin_density_1d,
    _create_1d_dataset,
    # 2D functions
    _initialize_2d_bins,
    _process_no_bootstrap_2d,
    _calculate_bin_density_2d,
    _create_2d_dataset,
    # 3D functions
    _initialize_3d_bins,
    _process_no_bootstrap_3d,
    _calculate_bin_density_3d,
    _create_3d_dataset,
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
    
    ds = xr.Dataset(
        data_vars={"temperature": ("x", scalar)},
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
    mask[::2] = 1
    
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
def linear_bins_1d():
    """Create linear bin edges for 1D testing."""
    return np.linspace(0, 10, 11)  # 10 bins


@pytest.fixture
def log_bins_1d():
    """Create logarithmic bin edges for 1D testing."""
    return np.logspace(-1, 1, 11)  # 10 bins from 0.1 to 10


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


@pytest.fixture
def log_bins_x():
    """Create logarithmic bin edges for x dimension."""
    return np.logspace(-1, 1, 11)


@pytest.fixture
def log_bins_y():
    """Create logarithmic bin edges for y dimension."""
    return np.logspace(-1, 0.9, 9)


@pytest.fixture
def log_bins_z():
    """Create logarithmic bin edges for z dimension."""
    return np.logspace(-1, 0.8, 7)


# =============================================================================
# Tests for 1D Binning Functions
# =============================================================================

class TestInitialize1DBins:
    """Tests for _initialize_1d_bins function."""
    
    def test_linear_bins(self, linear_bins_1d):
        """Test initialization with linear bins."""
        config = _initialize_1d_bins(linear_bins_1d, "x")
        
        assert 'bin_edges' in config
        assert 'bin_centers' in config
        assert 'n_bins' in config
        assert 'log_bins' in config
        assert 'dim_name' in config
        
        assert config['n_bins'] == 10
        assert config['log_bins'] == False
        assert config['dim_name'] == "x"
        assert len(config['bin_centers']) == 10
        
        # Check that bin centers are arithmetic means for linear bins
        expected_centers = 0.5 * (linear_bins_1d[:-1] + linear_bins_1d[1:])
        np.testing.assert_array_almost_equal(config['bin_centers'], expected_centers)
        
    def test_log_bins(self, log_bins_1d):
        """Test initialization with logarithmic bins."""
        config = _initialize_1d_bins(log_bins_1d, "x")
        
        assert config['n_bins'] == 10
        assert config['log_bins'] == True
        
        # Check that bin centers are geometric means for log bins
        expected_centers = np.sqrt(log_bins_1d[:-1] * log_bins_1d[1:])
        np.testing.assert_array_almost_equal(config['bin_centers'], expected_centers)
        
    def test_too_few_bin_edges(self):
        """Test that too few bin edges raises error."""
        with pytest.raises(ValueError, match="at least 2 values"):
            _initialize_1d_bins(np.array([1.0]), "x")
            
    def test_negative_bin_edges(self):
        """Test bins with negative values default to linear."""
        bins = np.linspace(-5, 5, 11)
        config = _initialize_1d_bins(bins, "x")
        
        assert config['log_bins'] == False
        
    def test_irregular_bins(self):
        """Test irregular bin spacing defaults to linear."""
        bins = np.array([0, 1, 3, 6, 10, 15])  # Irregular spacing
        config = _initialize_1d_bins(bins, "x")
        
        assert config['log_bins'] == False
        assert config['n_bins'] == 5
        
    def test_two_bin_edges(self):
        """Test minimum valid bin configuration (2 edges = 1 bin)."""
        bins = np.array([0.0, 10.0])
        config = _initialize_1d_bins(bins, "x")
        
        assert config['n_bins'] == 1
        assert len(config['bin_centers']) == 1
        assert config['bin_centers'][0] == 5.0


class TestProcessNoBootstrap1D:
    """Tests for _process_no_bootstrap_1d function."""
    
    def test_basic_processing(self, dataset_1d, linear_bins_1d):
        """Test basic no-bootstrap processing."""
        bins_config = _initialize_1d_bins(linear_bins_1d, "x")
        
        sf_means, sf_stds, point_counts = _process_no_bootstrap_1d(
            ds=dataset_1d,
            dim_name="x",
            variables_names=["temperature"],
            order=2,
            fun='scalar',
            bins_config=bins_config
        )
        
        assert sf_means.shape == (bins_config['n_bins'],)
        assert sf_stds.shape == (bins_config['n_bins'],)
        assert point_counts.shape == (bins_config['n_bins'],)
        
        # Check that we have some valid results
        assert np.any(~np.isnan(sf_means))
        assert np.sum(point_counts) > 0
        
    def test_with_mask(self, dataset_1d_with_mask, linear_bins_1d):
        """Test no-bootstrap processing with mask."""
        bins_config = _initialize_1d_bins(linear_bins_1d, "x")
        
        sf_means, sf_stds, point_counts = _process_no_bootstrap_1d(
            ds=dataset_1d_with_mask,
            dim_name="x",
            variables_names=["temperature"],
            order=2,
            fun='scalar',
            bins_config=bins_config,
            conditioning_var="mask",
            conditioning_bins=[0.5, 1.5]
        )
        
        assert sf_means.shape == (bins_config['n_bins'],)


class TestCalculateBinDensity1D:
    """Tests for _calculate_bin_density_1d function."""
    
    def test_uniform_distribution(self, linear_bins_1d):
        """Test density calculation with uniform point distribution."""
        n_bins = len(linear_bins_1d) - 1
        point_counts = np.ones(n_bins, dtype=int) * 10  # 10 points per bin
        
        density = _calculate_bin_density_1d(point_counts, linear_bins_1d)
        
        assert density.shape == (n_bins,)
        # Uniform distribution should have roughly equal density
        assert np.max(density) <= 1.0
        assert np.all(density >= 0)
        
    def test_empty_bins(self, linear_bins_1d):
        """Test density calculation with empty bins."""
        n_bins = len(linear_bins_1d) - 1
        point_counts = np.zeros(n_bins, dtype=int)
        
        density = _calculate_bin_density_1d(point_counts, linear_bins_1d)
        
        assert np.all(density == 0)
        
    def test_single_populated_bin(self, linear_bins_1d):
        """Test density with single populated bin."""
        n_bins = len(linear_bins_1d) - 1
        point_counts = np.zeros(n_bins, dtype=int)
        point_counts[5] = 100  # Only one bin has points
        
        density = _calculate_bin_density_1d(point_counts, linear_bins_1d)
        
        # The populated bin should have maximum density
        assert density[5] == 1.0
        # Other bins should have zero density
        assert np.sum(density) == 1.0
        
    def test_normalized_output(self, linear_bins_1d):
        """Test that density is normalized to [0, 1]."""
        n_bins = len(linear_bins_1d) - 1
        point_counts = np.random.randint(1, 100, n_bins)
        
        density = _calculate_bin_density_1d(point_counts, linear_bins_1d)
        
        assert np.max(density) == 1.0
        assert np.all(density >= 0)
        assert np.all(density <= 1.0)


class TestCreate1DDataset:
    """Tests for _create_1d_dataset function."""
    
    def test_basic_dataset_creation(self, linear_bins_1d):
        """Test basic 1D dataset creation."""
        bins_config = _initialize_1d_bins(linear_bins_1d, "x")
        n_bins = bins_config['n_bins']
        
        results = {
            'sf_means': np.random.randn(n_bins),
            'sf_stds': np.abs(np.random.randn(n_bins)),
            'bin_bootstraps': np.ones(n_bins, dtype=int) * 10,
            'bin_density': np.random.rand(n_bins),
            'point_counts': np.ones(n_bins, dtype=int) * 100,
            'bin_status': np.ones(n_bins, dtype=bool),
            'spacing_values': [1, 2, 4],
            'variables_names': ['temperature'],
        }
        
        ds = _create_1d_dataset(
            results=results,
            bins_config=bins_config,
            dim_name="x",
            order=2,
            fun='scalar',
            bootstrappable_dims=['x'],
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            confidence_level=0.95,
            backend='serial'
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'sf' in ds.data_vars
        assert 'std_error' in ds.data_vars
        assert 'ci_upper' in ds.data_vars
        assert 'ci_lower' in ds.data_vars
        assert 'nbootstraps' in ds.data_vars
        assert 'density' in ds.data_vars
        assert 'point_counts' in ds.data_vars
        assert 'converged' in ds.data_vars
        
        # Check attributes
        assert 'bin_type' in ds.attrs
        assert ds.attrs['function_type'] == 'scalar'
        assert ds.attrs['confidence_level'] == 0.95
        
    def test_with_precomputed_ci(self, linear_bins_1d):
        """Test dataset creation with pre-computed confidence intervals."""
        bins_config = _initialize_1d_bins(linear_bins_1d, "x")
        n_bins = bins_config['n_bins']
        
        sf_means = np.random.randn(n_bins)
        results = {
            'sf_means': sf_means,
            'sf_stds': np.abs(np.random.randn(n_bins)),
            'ci_lower': sf_means - 0.5,
            'ci_upper': sf_means + 0.5,
            'bin_bootstraps': np.ones(n_bins, dtype=int) * 10,
            'bin_density': np.random.rand(n_bins),
            'point_counts': np.ones(n_bins, dtype=int) * 100,
            'bin_status': np.ones(n_bins, dtype=bool),
            'spacing_values': [1, 2, 4],
        }
        
        ds = _create_1d_dataset(
            results=results,
            bins_config=bins_config,
            dim_name="x",
            order=2,
            fun='scalar',
            bootstrappable_dims=['x'],
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            confidence_level=0.95,
            backend='serial'
        )
        
        # Check that pre-computed CIs are used
        np.testing.assert_array_equal(ds['ci_lower'].values, sf_means - 0.5)
        np.testing.assert_array_equal(ds['ci_upper'].values, sf_means + 0.5)
        
    def test_different_confidence_levels(self, linear_bins_1d):
        """Test dataset creation with different confidence levels."""
        bins_config = _initialize_1d_bins(linear_bins_1d, "x")
        n_bins = bins_config['n_bins']
        
        results = {
            'sf_means': np.random.randn(n_bins),
            'sf_stds': np.abs(np.random.randn(n_bins)) * 0.1,
            'bin_bootstraps': np.ones(n_bins, dtype=int) * 10,
            'bin_density': np.random.rand(n_bins),
            'point_counts': np.ones(n_bins, dtype=int) * 100,
            'bin_status': np.ones(n_bins, dtype=bool),
            'spacing_values': [1],
        }
        
        ds_95 = _create_1d_dataset(
            results=results,
            bins_config=bins_config,
            dim_name="x",
            order=2,
            fun='scalar',
            bootstrappable_dims=['x'],
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            confidence_level=0.95,
            backend='serial'
        )
        
        ds_99 = _create_1d_dataset(
            results=results,
            bins_config=bins_config,
            dim_name="x",
            order=2,
            fun='scalar',
            bootstrappable_dims=['x'],
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            confidence_level=0.99,
            backend='serial'
        )
        
        # 99% CI should be wider than 95% CI
        ci_width_95 = ds_95['ci_upper'].values - ds_95['ci_lower'].values
        ci_width_99 = ds_99['ci_upper'].values - ds_99['ci_lower'].values
        
        valid_mask = ~np.isnan(ci_width_95) & ~np.isnan(ci_width_99)
        assert np.all(ci_width_99[valid_mask] >= ci_width_95[valid_mask])


# =============================================================================
# Tests for 2D Binning Functions
# =============================================================================

class TestInitialize2DBins:
    """Tests for _initialize_2d_bins function."""
    
    def test_linear_bins(self, linear_bins_x, linear_bins_y):
        """Test initialization with linear bins."""
        dims_order = ["y", "x"]
        config = _initialize_2d_bins(linear_bins_x, linear_bins_y, dims_order)
        
        assert 'bins_x' in config
        assert 'bins_y' in config
        assert 'x_centers' in config
        assert 'y_centers' in config
        assert 'n_bins_x' in config
        assert 'n_bins_y' in config
        assert 'log_bins_x' in config
        assert 'log_bins_y' in config
        
        assert config['n_bins_x'] == 10
        assert config['n_bins_y'] == 8
        assert config['log_bins_x'] == False
        assert config['log_bins_y'] == False
        
    def test_log_bins(self, log_bins_x, log_bins_y):
        """Test initialization with logarithmic bins."""
        dims_order = ["y", "x"]
        config = _initialize_2d_bins(log_bins_x, log_bins_y, dims_order)
        
        assert config['log_bins_x'] == True
        assert config['log_bins_y'] == True
        
        # Check geometric mean centers
        expected_x_centers = np.sqrt(log_bins_x[:-1] * log_bins_x[1:])
        expected_y_centers = np.sqrt(log_bins_y[:-1] * log_bins_y[1:])
        np.testing.assert_array_almost_equal(config['x_centers'], expected_x_centers)
        np.testing.assert_array_almost_equal(config['y_centers'], expected_y_centers)
        
    def test_mixed_bins(self, linear_bins_x, log_bins_y):
        """Test initialization with mixed bin types."""
        dims_order = ["y", "x"]
        config = _initialize_2d_bins(linear_bins_x, log_bins_y, dims_order)
        
        assert config['log_bins_x'] == False
        assert config['log_bins_y'] == True
        
    def test_dims_order_preserved(self, linear_bins_x, linear_bins_y):
        """Test that dims_order is stored in config."""
        dims_order = ["y", "x"]
        config = _initialize_2d_bins(linear_bins_x, linear_bins_y, dims_order)
        
        assert config['dims_order'] == dims_order


class TestProcessNoBootstrap2D:
    """Tests for _process_no_bootstrap_2d function."""
    
    def test_basic_processing(self, dataset_2d, linear_bins_x, linear_bins_y):
        """Test basic 2D no-bootstrap processing."""
        dims = ["y", "x"]
        bins = {"x": linear_bins_x, "y": linear_bins_y}
        time_dims = {"y": False, "x": False}
        
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_2d(
            ds=dataset_2d,
            dims=dims,
            variables_names=["u", "v"],
            order=2,
            fun='longitudinal',
            bins=bins,
            time_dims=time_dims,
            conditioning_var=None,
            conditioning_bins=None
        )
        
        expected_shape = (bins_config['n_bins_y'], bins_config['n_bins_x'])
        assert sf_means.shape == expected_shape
        assert sf_stds.shape == expected_shape
        assert point_counts.shape == expected_shape
        
        # Check that we have some valid results
        assert np.any(~np.isnan(sf_means))
        assert np.sum(point_counts) > 0
        
    def test_scalar_function(self, dataset_2d, linear_bins_x, linear_bins_y):
        """Test 2D no-bootstrap with scalar function."""
        dims = ["y", "x"]
        bins = {"x": linear_bins_x, "y": linear_bins_y}
        time_dims = {"y": False, "x": False}
        
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_2d(
            ds=dataset_2d,
            dims=dims,
            variables_names=["temperature"],
            order=2,
            fun='scalar',
            bins=bins,
            time_dims=time_dims,
            conditioning_var=None,
            conditioning_bins=None
        )
        
        assert sf_means.shape == (bins_config['n_bins_y'], bins_config['n_bins_x'])


class TestCalculateBinDensity2D:
    """Tests for _calculate_bin_density_2d function."""
    
    def test_uniform_distribution(self, linear_bins_x, linear_bins_y):
        """Test 2D density calculation with uniform distribution."""
        n_bins_x = len(linear_bins_x) - 1
        n_bins_y = len(linear_bins_y) - 1
        point_counts = np.ones((n_bins_y, n_bins_x), dtype=int) * 10
        
        density = _calculate_bin_density_2d(point_counts, linear_bins_x, linear_bins_y)
        
        assert density.shape == (n_bins_y, n_bins_x)
        assert np.max(density) <= 1.0
        assert np.all(density >= 0)
        
    def test_empty_bins(self, linear_bins_x, linear_bins_y):
        """Test 2D density with empty bins."""
        n_bins_x = len(linear_bins_x) - 1
        n_bins_y = len(linear_bins_y) - 1
        point_counts = np.zeros((n_bins_y, n_bins_x), dtype=int)
        
        density = _calculate_bin_density_2d(point_counts, linear_bins_x, linear_bins_y)
        
        assert np.all(density == 0)
        
    def test_single_populated_bin(self, linear_bins_x, linear_bins_y):
        """Test 2D density with single populated bin."""
        n_bins_x = len(linear_bins_x) - 1
        n_bins_y = len(linear_bins_y) - 1
        point_counts = np.zeros((n_bins_y, n_bins_x), dtype=int)
        point_counts[4, 5] = 100
        
        density = _calculate_bin_density_2d(point_counts, linear_bins_x, linear_bins_y)
        
        assert density[4, 5] == 1.0


class TestCreate2DDataset:
    """Tests for _create_2d_dataset function."""
    
    def test_basic_dataset_creation(self, linear_bins_x, linear_bins_y):
        """Test basic 2D dataset creation."""
        dims = ["y", "x"]
        bins_config = _initialize_2d_bins(linear_bins_x, linear_bins_y, dims)
        
        shape = (bins_config['n_bins_y'], bins_config['n_bins_x'])
        
        results = {
            'sf_means': np.random.randn(*shape),
            'sf_stds': np.abs(np.random.randn(*shape)),
            'bin_bootstraps': np.ones(shape, dtype=int) * 10,
            'bin_density': np.random.rand(*shape),
            'point_counts': np.ones(shape, dtype=int) * 100,
            'bin_status': np.ones(shape, dtype=bool),
            'spacing_values': [1, 2, 4],
            'variables_names': ['u', 'v'],
        }
        
        time_dims = {"y": False, "x": False}
        
        ds = _create_2d_dataset(
            results=results,
            bins_config=bins_config,
            dims=dims,
            order=2,
            fun='longitudinal',
            bootstrappable_dims=['y', 'x'],
            time_dims=time_dims,
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            backend='serial'
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'sf' in ds.data_vars
        assert 'std_error' in ds.data_vars
        assert 'nbootstraps' in ds.data_vars
        assert 'density' in ds.data_vars
        assert 'point_counts' in ds.data_vars
        assert 'converged' in ds.data_vars
        
        # Check dimensions
        assert ds['sf'].dims == ('y', 'x')
        
        # Check attributes
        assert 'bin_type_x' in ds.attrs
        assert 'bin_type_y' in ds.attrs
        assert ds.attrs['function_type'] == 'longitudinal'
        
    def test_bin_edges_stored(self, linear_bins_x, linear_bins_y):
        """Test that bin edges are stored in dataset."""
        dims = ["y", "x"]
        bins_config = _initialize_2d_bins(linear_bins_x, linear_bins_y, dims)
        
        shape = (bins_config['n_bins_y'], bins_config['n_bins_x'])
        
        results = {
            'sf_means': np.random.randn(*shape),
            'sf_stds': np.abs(np.random.randn(*shape)),
            'bin_bootstraps': np.ones(shape, dtype=int) * 10,
            'bin_density': np.random.rand(*shape),
            'point_counts': np.ones(shape, dtype=int) * 100,
            'bin_status': np.ones(shape, dtype=bool),
            'spacing_values': [1],
            'variables_names': ['u', 'v'],
        }
        
        time_dims = {"y": False, "x": False}
        
        ds = _create_2d_dataset(
            results=results,
            bins_config=bins_config,
            dims=dims,
            order=2,
            fun='longitudinal',
            bootstrappable_dims=['y', 'x'],
            time_dims=time_dims,
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            backend='serial'
        )
        
        # Check bin edges are stored
        assert 'x_bins' in ds.data_vars
        assert 'y_bins' in ds.data_vars


# =============================================================================
# Tests for 3D Binning Functions
# =============================================================================

class TestInitialize3DBins:
    """Tests for _initialize_3d_bins function."""
    
    def test_linear_bins(self, linear_bins_x, linear_bins_y, linear_bins_z):
        """Test 3D initialization with linear bins."""
        dims_order = ["z", "y", "x"]
        config = _initialize_3d_bins(linear_bins_x, linear_bins_y, linear_bins_z, dims_order)
        
        assert 'bins_x' in config
        assert 'bins_y' in config
        assert 'bins_z' in config
        assert 'x_centers' in config
        assert 'y_centers' in config
        assert 'z_centers' in config
        assert 'n_bins_x' in config
        assert 'n_bins_y' in config
        assert 'n_bins_z' in config
        
        assert config['n_bins_x'] == 10
        assert config['n_bins_y'] == 8
        assert config['n_bins_z'] == 6
        assert config['log_bins_x'] == False
        assert config['log_bins_y'] == False
        assert config['log_bins_z'] == False
        
    def test_log_bins(self, log_bins_x, log_bins_y, log_bins_z):
        """Test 3D initialization with logarithmic bins."""
        dims_order = ["z", "y", "x"]
        config = _initialize_3d_bins(log_bins_x, log_bins_y, log_bins_z, dims_order)
        
        assert config['log_bins_x'] == True
        assert config['log_bins_y'] == True
        assert config['log_bins_z'] == True
        
    def test_mixed_bins(self, linear_bins_x, log_bins_y, linear_bins_z):
        """Test 3D initialization with mixed bin types."""
        dims_order = ["z", "y", "x"]
        config = _initialize_3d_bins(linear_bins_x, log_bins_y, linear_bins_z, dims_order)
        
        assert config['log_bins_x'] == False
        assert config['log_bins_y'] == True
        assert config['log_bins_z'] == False


class TestProcessNoBootstrap3D:
    """Tests for _process_no_bootstrap_3d function."""
    
    def test_basic_processing(self, dataset_3d, linear_bins_x, linear_bins_y, linear_bins_z):
        """Test basic 3D no-bootstrap processing."""
        dims = ["z", "y", "x"]
        bins = {"x": linear_bins_x, "y": linear_bins_y, "z": linear_bins_z}
        time_dims = {"z": False, "y": False, "x": False}
        
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_3d(
            ds=dataset_3d,
            dims=dims,
            variables_names=["u", "v", "w"],
            order=2,
            fun='longitudinal',
            bins=bins,
            time_dims=time_dims
        )
        
        expected_shape = (bins_config['n_bins_z'], bins_config['n_bins_y'], bins_config['n_bins_x'])
        assert sf_means.shape == expected_shape
        assert sf_stds.shape == expected_shape
        assert point_counts.shape == expected_shape
        
        # Check that we have some valid results
        assert np.any(~np.isnan(sf_means))
        assert np.sum(point_counts) > 0


class TestCalculateBinDensity3D:
    """Tests for _calculate_bin_density_3d function."""
    
    def test_uniform_distribution(self, linear_bins_x, linear_bins_y, linear_bins_z):
        """Test 3D density calculation with uniform distribution."""
        n_bins_x = len(linear_bins_x) - 1
        n_bins_y = len(linear_bins_y) - 1
        n_bins_z = len(linear_bins_z) - 1
        point_counts = np.ones((n_bins_z, n_bins_y, n_bins_x), dtype=int) * 10
        
        density = _calculate_bin_density_3d(point_counts, linear_bins_x, linear_bins_y, linear_bins_z)
        
        assert density.shape == (n_bins_z, n_bins_y, n_bins_x)
        assert np.max(density) <= 1.0
        assert np.all(density >= 0)
        
    def test_empty_bins(self, linear_bins_x, linear_bins_y, linear_bins_z):
        """Test 3D density with empty bins."""
        n_bins_x = len(linear_bins_x) - 1
        n_bins_y = len(linear_bins_y) - 1
        n_bins_z = len(linear_bins_z) - 1
        point_counts = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=int)
        
        density = _calculate_bin_density_3d(point_counts, linear_bins_x, linear_bins_y, linear_bins_z)
        
        assert np.all(density == 0)
        
    def test_single_populated_bin(self, linear_bins_x, linear_bins_y, linear_bins_z):
        """Test 3D density with single populated bin."""
        n_bins_x = len(linear_bins_x) - 1
        n_bins_y = len(linear_bins_y) - 1
        n_bins_z = len(linear_bins_z) - 1
        point_counts = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=int)
        point_counts[3, 4, 5] = 100
        
        density = _calculate_bin_density_3d(point_counts, linear_bins_x, linear_bins_y, linear_bins_z)
        
        assert density[3, 4, 5] == 1.0


class TestCreate3DDataset:
    """Tests for _create_3d_dataset function."""
    
    def test_basic_dataset_creation(self, linear_bins_x, linear_bins_y, linear_bins_z):
        """Test basic 3D dataset creation."""
        dims = ["z", "y", "x"]
        bins_config = _initialize_3d_bins(linear_bins_x, linear_bins_y, linear_bins_z, dims)
        
        shape = (bins_config['n_bins_z'], bins_config['n_bins_y'], bins_config['n_bins_x'])
        
        results = {
            'sf_means': np.random.randn(*shape),
            'sf_stds': np.abs(np.random.randn(*shape)),
            'bin_bootstraps': np.ones(shape, dtype=int) * 10,
            'bin_density': np.random.rand(*shape),
            'point_counts': np.ones(shape, dtype=int) * 100,
            'bin_status': np.ones(shape, dtype=bool),
            'spacing_values': [1, 2, 4],
            'variables_names': ['u', 'v', 'w'],
        }
        
        time_dims = {"z": False, "y": False, "x": False}
        
        ds = _create_3d_dataset(
            results=results,
            bins_config=bins_config,
            dims=dims,
            order=2,
            fun='longitudinal',
            bootstrappable_dims=['z', 'y', 'x'],
            time_dims=time_dims,
            convergence_eps=0.01,
            max_nbootstrap=100,
            initial_nbootstrap=10,
            backend='serial',
            variables_names=['u', 'v', 'w']
        )
        
        assert isinstance(ds, xr.Dataset)
        assert 'sf' in ds.data_vars
        assert 'std_error' in ds.data_vars
        assert 'nbootstraps' in ds.data_vars
        assert 'density' in ds.data_vars
        assert 'point_counts' in ds.data_vars
        assert 'converged' in ds.data_vars
        
        # Check dimensions
        assert ds['sf'].dims == ('z', 'y', 'x')
        
        # Check attributes
        assert 'bin_type_x' in ds.attrs
        assert 'bin_type_y' in ds.attrs
        assert 'bin_type_z' in ds.attrs
        assert ds.attrs['function_type'] == 'longitudinal'


# =============================================================================
# Tests for edge cases and integration
# =============================================================================

class TestBinningEdgeCases:
    """Tests for edge cases in binning functions."""
    
    def test_very_small_bins_1d(self):
        """Test with very few bins in 1D."""
        bins = np.array([0, 5, 10])  # 2 bins
        config = _initialize_1d_bins(bins, "x")
        
        assert config['n_bins'] == 2
        
    def test_many_bins_1d(self):
        """Test with many bins in 1D."""
        bins = np.linspace(0, 10, 101)  # 100 bins
        config = _initialize_1d_bins(bins, "x")
        
        assert config['n_bins'] == 100
        
    def test_bin_density_normalization(self, linear_bins_1d):
        """Test that bin density is properly normalized."""
        n_bins = len(linear_bins_1d) - 1
        
        # Create non-uniform distribution
        point_counts = np.zeros(n_bins, dtype=int)
        point_counts[0] = 1000
        point_counts[5] = 500
        point_counts[9] = 100
        
        density = _calculate_bin_density_1d(point_counts, linear_bins_1d)
        
        # Maximum should be normalized to 1
        assert np.isclose(np.max(density), 1.0)
        assert np.all(density >= 0)
        assert np.all(density <= 1.0)


class TestBinCenterCalculations:
    """Tests for bin center calculations."""
    
    def test_linear_bin_centers(self):
        """Test arithmetic mean for linear bin centers."""
        bins = np.array([0, 2, 4, 6, 8, 10])
        config = _initialize_1d_bins(bins, "x")
        
        expected = np.array([1, 3, 5, 7, 9])
        np.testing.assert_array_almost_equal(config['bin_centers'], expected)
        
    def test_log_bin_centers(self):
        """Test geometric mean for log bin centers."""
        bins = np.array([1, 10, 100, 1000])
        config = _initialize_1d_bins(bins, "x")
        
        # Geometric means
        expected = np.array([np.sqrt(10), np.sqrt(1000), np.sqrt(100000)])
        np.testing.assert_array_almost_equal(config['bin_centers'], expected)


class TestBinningWithTimeDimensions:
    """Tests for binning with time dimensions."""
    
    def test_2d_with_time_dimension(self, dataset_2d, linear_bins_x, linear_bins_y):
        """Test 2D processing with one time dimension."""
        dims = ["y", "x"]
        bins = {"x": linear_bins_x, "y": linear_bins_y}
        time_dims = {"y": True, "x": False}  # y is time
        
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_2d(
            ds=dataset_2d,
            dims=dims,
            variables_names=["temperature"],
            order=2,
            fun='scalar',
            bins=bins,
            time_dims=time_dims,
            conditioning_var=None,
            conditioning_bins=None
        )
        
        # Should still produce valid output
        assert sf_means.shape == (bins_config['n_bins_y'], bins_config['n_bins_x'])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
