"""
Tests for core.py module functionality.
"""

import pytest
import numpy as np
import xarray as xr
import warnings
from datetime import datetime

from pyturbo_sf.core import (
    is_time_dimension,
    _check_bootsize_power_of_2,
    validate_dataset_1d, validate_dataset_2d, validate_dataset_3d,
    setup_bootsize_1d, setup_bootsize_2d, setup_bootsize_3d,
    calculate_adaptive_spacings_1d, calculate_adaptive_spacings_2d, calculate_adaptive_spacings_3d,
    _get_simplified_adaptive_spacings_2d, _get_simplified_adaptive_spacings_3d,
    compute_boot_indexes_1d, compute_boot_indexes_2d, compute_boot_indexes_3d,
    get_boot_indexes_1d, get_boot_indexes_2d, get_boot_indexes_3d
)


# =============================================================================
# Fixtures for test datasets
# =============================================================================

@pytest.fixture
def dataset_1d():
    """Create a simple 1D dataset for testing."""
    x = np.linspace(0, 10, 100)
    data = np.sin(x)
    
    ds = xr.Dataset(
        data_vars={"data": ("x", data)},
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_1d_small():
    """Create a small 1D dataset for edge case testing."""
    x = np.linspace(0, 10, 16)
    data = np.sin(x)
    
    ds = xr.Dataset(
        data_vars={"data": ("x", data)},
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_2d():
    """Create a simple 2D dataset for testing."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 80)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Y)
    v = np.cos(X) * np.sin(Y)
    scalar = np.sin(X + Y)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("y", "x"), u),
            "v": (("y", "x"), v),
            "scalar": (("y", "x"), scalar)
        },
        coords={
            "x": (["y", "x"], X),
            "y": (["y", "x"], Y),
        }
    )
    return ds


@pytest.fixture
def dataset_2d_reversed():
    """Create a 2D dataset with reversed dimension order for transposition testing."""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 80)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    u = np.sin(X) * np.cos(Y)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("x", "y"), u.T),
        },
        coords={
            "x": x,
            "y": y,
        }
    )
    return ds


@pytest.fixture
def dataset_2d_with_time():
    """Create a 2D dataset with time dimension for testing."""
    time = np.array(['2023-01-01', '2023-01-02', '2023-01-03'], dtype='datetime64[D]')
    x = np.linspace(0, 10, 100)
    
    data = np.random.randn(len(time), len(x))
    
    ds = xr.Dataset(
        data_vars={"data": (("time", "x"), data)},
        coords={
            "time": time,
            "x": x
        }
    )
    return ds


@pytest.fixture
def dataset_2d_zx():
    """Create a 2D dataset with z-x dimensions."""
    z = np.linspace(0, 10, 50)
    x = np.linspace(0, 10, 100)
    Z, X = np.meshgrid(z, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Z)
    w = np.cos(X) * np.sin(Z)
    
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
def dataset_3d():
    """Create a simple 3D dataset for testing."""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 15)
    z = np.linspace(0, 10, 10)
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    v = np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.sin(X) * np.sin(Y) * np.sin(Z)
    scalar = np.sin(X + Y + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "scalar": (("z", "y", "x"), scalar)
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


@pytest.fixture
def dataset_3d_reordered():
    """Create a 3D dataset with non-standard dimension order."""
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 15)
    z = np.linspace(0, 10, 10)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("x", "y", "z"), u),
        },
        coords={
            "x": x,
            "y": y,
            "z": z,
        }
    )
    return ds


@pytest.fixture
def dataset_3d_with_time():
    """Create a 3D dataset with time dimension for testing."""
    time = np.array(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'], dtype='datetime64[D]')
    y = np.linspace(0, 10, 15)
    x = np.linspace(0, 10, 20)
    
    data = np.random.randn(len(time), len(y), len(x))
    
    ds = xr.Dataset(
        data_vars={"data": (("time", "y", "x"), data)},
        coords={
            "time": time,
            "y": y,
            "x": x
        }
    )
    return ds


@pytest.fixture
def dataset_3d_large():
    """Create a larger 3D dataset for spacing tests."""
    x = np.linspace(0, 10, 64)
    y = np.linspace(0, 10, 64)
    z = np.linspace(0, 10, 32)
    
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
    
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


# =============================================================================
# Tests for is_time_dimension
# =============================================================================

class TestIsTimeDimension:
    """Test the is_time_dimension helper function."""
    
    def test_time_named_dimension(self):
        """Test detection of dimension named 'time'."""
        ds = xr.Dataset(
            data_vars={"data": ("time", [1, 2, 3])},
            coords={"time": [1, 2, 3]}
        )
        assert is_time_dimension("time", ds) == True
        assert is_time_dimension("Time", ds) == True  # Case insensitive
        
    def test_datetime64_dimension(self, dataset_2d_with_time):
        """Test detection of datetime64 dimension."""
        assert is_time_dimension("time", dataset_2d_with_time) == True
        assert is_time_dimension("x", dataset_2d_with_time) == False
        
    def test_python_datetime_dimension(self):
        """Test detection of Python datetime dimension."""
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        ds = xr.Dataset(
            data_vars={"data": ("date", [1, 2, 3])},
            coords={"date": dates}
        )
        assert is_time_dimension("date", ds) == True
        
    def test_non_time_dimension(self, dataset_2d):
        """Test that spatial dimensions are not detected as time."""
        assert is_time_dimension("x", dataset_2d) == False
        assert is_time_dimension("y", dataset_2d) == False
        
    def test_dimension_not_in_coords(self):
        """Test behavior when dimension is not in coords."""
        ds = xr.Dataset(
            data_vars={"data": ("x", [1, 2, 3])}
        )
        # Should not raise error, just return False
        assert is_time_dimension("x", ds) == False


# =============================================================================
# Tests for _check_bootsize_power_of_2
# =============================================================================

class TestCheckBootsizePowerOf2:
    """Test the _check_bootsize_power_of_2 helper function."""
    
    def test_valid_bootsize_1d(self):
        """Test with valid power-of-2 bootsize for 1D."""
        dims = "x"
        data_shape = {"x": 64}
        bootsize_dict = {"x": 32}  # 64/32 = 2 = 2^1, valid
        
        # Should not raise warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_bootsize_power_of_2(dims, data_shape, bootsize_dict)
            assert len(w) == 0
            
    def test_invalid_bootsize_1d(self):
        """Test with invalid bootsize for 1D."""
        dims = "x"
        data_shape = {"x": 64}
        bootsize_dict = {"x": 30}  # Not a power of 2 divisor
        
        # Should raise warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_bootsize_power_of_2(dims, data_shape, bootsize_dict)
            assert len(w) == 1
            assert "Warning" in str(w[0].message)
            
    def test_valid_bootsize_2d(self):
        """Test with valid power-of-2 bootsizes for 2D."""
        dims = ["y", "x"]
        data_shape = {"y": 64, "x": 128}
        bootsize_dict = {"y": 32, "x": 32}  # Both valid
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_bootsize_power_of_2(dims, data_shape, bootsize_dict)
            assert len(w) == 0
            
    def test_non_bootstrappable_dimension(self):
        """Test when bootsize >= data_size (not bootstrappable)."""
        dims = ["y", "x"]
        data_shape = {"y": 32, "x": 128}
        bootsize_dict = {"y": 32, "x": 32}  # y is not bootstrappable
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _check_bootsize_power_of_2(dims, data_shape, bootsize_dict)
            # Should only return bootstrappable dimensions
            assert "y" not in result or len(result.get("y", [])) == 1


# =============================================================================
# Tests for validate_dataset functions
# =============================================================================

class TestValidateDataset:
    """Test dataset validation functions."""
    
    # 1D Tests
    def test_validate_dataset_1d(self, dataset_1d):
        """Test validation of 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        assert dim == "x"
        assert isinstance(data_shape, dict)
        assert "x" in data_shape
        assert data_shape["x"] == 100
        
    def test_validate_dataset_1d_error(self, dataset_2d):
        """Test that validation fails for dataset with wrong dimensions."""
        with pytest.raises(ValueError, match="must have exactly 1 dimension"):
            validate_dataset_1d(dataset_2d)
            
    # 2D Tests
    def test_validate_dataset_2d(self, dataset_2d):
        """Test validation of 2D dataset."""
        dims, data_shape, ds, time_dims = validate_dataset_2d(dataset_2d)
        
        assert dims == ["y", "x"]
        assert isinstance(data_shape, dict)
        assert "y" in data_shape and "x" in data_shape
        assert data_shape["y"] == 80
        assert data_shape["x"] == 100
        assert isinstance(ds, xr.Dataset)
        assert isinstance(time_dims, dict)
        assert time_dims["y"] == False
        assert time_dims["x"] == False
        
    def test_validate_dataset_2d_transposition(self, dataset_2d_reversed):
        """Test that 2D dataset with reversed dims gets validated."""
        dims, data_shape, ds, time_dims = validate_dataset_2d(dataset_2d_reversed)
        
        # Should return valid results - either transposed or original order
        # depending on how xarray handles the transpose
        assert len(dims) == 2
        assert 'x' in dims and 'y' in dims
        assert isinstance(data_shape, dict)
        assert isinstance(ds, xr.Dataset)
        
    def test_validate_dataset_2d_zx_plane(self, dataset_2d_zx):
        """Test validation of 2D dataset with z-x dimensions."""
        dims, data_shape, ds, time_dims = validate_dataset_2d(dataset_2d_zx)
        
        assert dims == ["z", "x"]
        assert data_shape["z"] == 50
        assert data_shape["x"] == 100
        
    def test_validate_dataset_2d_with_time(self, dataset_2d_with_time):
        """Test validation of 2D dataset with time dimension."""
        dims, data_shape, ds, time_dims = validate_dataset_2d(dataset_2d_with_time)
        
        assert dims == ["time", "x"]
        assert data_shape["time"] == 3
        assert data_shape["x"] == 100
        assert time_dims["time"] == True
        assert time_dims["x"] == False
        
    def test_validate_dataset_2d_error(self, dataset_1d):
        """Test that validation fails for dataset with wrong dimensions."""
        with pytest.raises(ValueError, match="Expected exactly 2 dimensions"):
            validate_dataset_2d(dataset_1d)
            
    def test_validate_dataset_2d_invalid_dims(self):
        """Test validation fails for incompatible dimension names."""
        ds = xr.Dataset(
            data_vars={"data": (("a", "b"), np.random.randn(10, 10))},
            coords={"a": np.arange(10), "b": np.arange(10)}
        )
        with pytest.raises(ValueError, match="not compatible"):
            validate_dataset_2d(ds)
            
    # 3D Tests
    def test_validate_dataset_3d(self, dataset_3d):
        """Test validation of 3D dataset."""
        dims, data_shape, ds, time_dims = validate_dataset_3d(dataset_3d)
        
        assert dims == ["z", "y", "x"]
        assert isinstance(data_shape, dict)
        assert "z" in data_shape and "y" in data_shape and "x" in data_shape
        assert data_shape["z"] == 10
        assert data_shape["y"] == 15
        assert data_shape["x"] == 20
        assert isinstance(ds, xr.Dataset)
        assert time_dims["z"] == False
        assert time_dims["y"] == False
        assert time_dims["x"] == False
        
    def test_validate_dataset_3d_transposition(self, dataset_3d_reordered):
        """Test that 3D dataset with non-standard order gets validated."""
        dims, data_shape, ds, time_dims = validate_dataset_3d(dataset_3d_reordered)
        
        # Should return valid results - either transposed or original order
        assert len(dims) == 3
        assert 'x' in dims and 'y' in dims and 'z' in dims
        assert isinstance(data_shape, dict)
        assert isinstance(ds, xr.Dataset)
        
    def test_validate_dataset_3d_with_time(self, dataset_3d_with_time):
        """Test validation of 3D dataset with time dimension."""
        dims, data_shape, ds, time_dims = validate_dataset_3d(dataset_3d_with_time)
        
        assert dims == ["time", "y", "x"]
        assert data_shape["time"] == 4
        assert data_shape["y"] == 15
        assert data_shape["x"] == 20
        assert time_dims["time"] == True
        assert time_dims["y"] == False
        assert time_dims["x"] == False
        
    def test_validate_dataset_3d_error(self, dataset_2d):
        """Test that validation fails for dataset with wrong dimensions."""
        with pytest.raises(ValueError, match="Expected exactly 3 dimensions"):
            validate_dataset_3d(dataset_2d)


# =============================================================================
# Tests for setup_bootsize functions
# =============================================================================

class TestSetupBootsize:
    """Test bootsize setup functions."""
    
    # 1D Tests
    def test_setup_bootsize_1d_default(self, dataset_1d):
        """Test default bootsize setup for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=None
        )
        
        assert isinstance(bootsize_dict, dict)
        assert dim in bootsize_dict
        assert bootsize_dict[dim] <= min(32, data_shape[dim] // 2)
        assert bootstrappable_dims == [dim]
        assert num_bootstrappable == 1
        
    def test_setup_bootsize_1d_custom_int(self, dataset_1d):
        """Test custom integer bootsize for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        custom_bootsize = 20
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=custom_bootsize
        )
        
        assert bootsize_dict[dim] == custom_bootsize
        
    def test_setup_bootsize_1d_custom_dict(self, dataset_1d):
        """Test custom dictionary bootsize for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        custom_bootsize_dict = {dim: 25}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=custom_bootsize_dict
        )
        
        assert bootsize_dict[dim] == 25
        
    def test_setup_bootsize_1d_non_bootstrappable(self, dataset_1d_small):
        """Test when bootsize >= data_size."""
        dim, data_shape = validate_dataset_1d(dataset_1d_small)
        
        # Bootsize equal to data size
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=data_shape[dim]
        )
        
        assert num_bootstrappable == 0
        assert bootstrappable_dims == []
        
    # 2D Tests
    def test_setup_bootsize_2d_default(self, dataset_2d):
        """Test default bootsize setup for 2D dataset."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(
            dims, data_shape, bootsize=None
        )
        
        assert isinstance(bootsize_dict, dict)
        assert all(dim in bootsize_dict for dim in dims)
        assert set(bootstrappable_dims) == set(dims)
        assert num_bootstrappable == 2
        
    def test_setup_bootsize_2d_custom(self, dataset_2d):
        """Test custom bootsize for 2D dataset."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        
        custom_bootsize_dict = {"x": 25, "y": 20}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(
            dims, data_shape, bootsize=custom_bootsize_dict
        )
        
        assert bootsize_dict["x"] == 25
        assert bootsize_dict["y"] == 20
        
    def test_setup_bootsize_2d_partial_bootstrappable(self, dataset_2d):
        """Test when only one dimension is bootstrappable."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        
        # Make y non-bootstrappable
        bootsize_dict = {"x": 25, "y": data_shape["y"]}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(
            dims, data_shape, bootsize=bootsize_dict
        )
        
        assert num_bootstrappable == 1
        assert "x" in bootstrappable_dims
        assert "y" not in bootstrappable_dims
        
    # 3D Tests
    def test_setup_bootsize_3d_default(self, dataset_3d):
        """Test default bootsize setup for 3D dataset."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(
            dims, data_shape, bootsize=None
        )
        
        assert isinstance(bootsize_dict, dict)
        assert all(dim in bootsize_dict for dim in dims)
        assert set(bootstrappable_dims) == set(dims)
        assert num_bootstrappable == 3
        
    def test_setup_bootsize_3d_custom(self, dataset_3d):
        """Test custom bootsize for 3D dataset."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        
        custom_bootsize_dict = {"x": 5, "y": 5, "z": 5}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(
            dims, data_shape, bootsize=custom_bootsize_dict
        )
        
        assert bootsize_dict["x"] == 5
        assert bootsize_dict["y"] == 5
        assert bootsize_dict["z"] == 5


# =============================================================================
# Tests for calculate_adaptive_spacings functions
# =============================================================================

class TestCalculateAdaptiveSpacings:
    """Test adaptive spacing calculation functions."""
    
    # 1D Tests
    def test_calculate_adaptive_spacings_1d(self, dataset_1d):
        """Test calculation of adaptive spacings for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim, data_shape)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        
        assert isinstance(spacings_info, dict)
        assert 'spacings' in spacings_info
        assert isinstance(all_spacings, list)
        assert len(all_spacings) > 0
        assert 1 in all_spacings
        # Spacings should be powers of 2
        for sp in all_spacings:
            assert (sp & (sp - 1)) == 0 or sp == 0  # Check power of 2
        
    def test_calculate_adaptive_spacings_1d_no_bootstrappable(self, dataset_1d):
        """Test spacings calculation with no bootstrappable dimensions."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, {dim: data_shape[dim]}, 0
        )
        
        assert spacings_info['spacings'] == [1]
        assert all_spacings == [1]
        
    # 2D Tests
    def test_calculate_adaptive_spacings_2d(self, dataset_2d):
        """Test calculation of adaptive spacings for 2D dataset."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        assert isinstance(spacings_info, dict)
        assert 'shared_spacings' in spacings_info
        assert isinstance(all_spacings, list)
        assert len(all_spacings) > 0
        assert 1 in all_spacings
        
    def test_calculate_adaptive_spacings_2d_single_bootstrappable(self, dataset_2d):
        """Test spacings with single bootstrappable dimension."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        
        # Only x is bootstrappable
        bootsize_dict = {"x": 25, "y": data_shape["y"]}
        bootstrappable_dims = ["x"]
        
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, 1
        )
        
        assert 'shared_spacings' in spacings_info
        assert len(all_spacings) > 0
        
    def test_get_simplified_adaptive_spacings_2d(self):
        """Test the simplified 2D spacing calculation."""
        data_shape = {"y": 128, "x": 128}
        bootsize = {"y": 32, "x": 32}
        
        result = _get_simplified_adaptive_spacings_2d(data_shape, bootsize)
        
        assert 'shared_spacings' in result
        assert 1 in result['shared_spacings']
        # Should include powers of 2 up to the ratio
        assert 2 in result['shared_spacings']
        
    def test_get_simplified_adaptive_spacings_2d_unbalanced(self):
        """Test spacing calculation with unbalanced dimension ratios."""
        data_shape = {"y": 32, "x": 128}
        bootsize = {"y": 16, "x": 16}  # Ratios: y=2, x=8
        
        result = _get_simplified_adaptive_spacings_2d(data_shape, bootsize)
        
        # Should be limited by the smaller ratio
        assert 'shared_spacings' in result
        assert max(result['shared_spacings']) <= 2
        
    # 3D Tests
    def test_calculate_adaptive_spacings_3d(self, dataset_3d):
        """Test calculation of adaptive spacings for 3D dataset."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        assert isinstance(spacings_info, dict)
        assert 'shared_spacings' in spacings_info
        assert isinstance(all_spacings, list)
        assert len(all_spacings) > 0
        assert 1 in all_spacings
        
    def test_calculate_adaptive_spacings_3d_no_bootstrappable(self, dataset_3d):
        """Test 3D spacings with no bootstrappable dimensions."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, {d: data_shape[d] for d in dims}, [], 0
        )
        
        assert spacings_info['shared_spacings'] == [1]
        
    def test_get_simplified_adaptive_spacings_3d(self):
        """Test the simplified 3D spacing calculation."""
        data_shape = {"z": 64, "y": 64, "x": 64}
        bootsize = {"z": 16, "y": 16, "x": 16}
        bootstrappable_dims = ["z", "y", "x"]
        
        result = _get_simplified_adaptive_spacings_3d(data_shape, bootsize, bootstrappable_dims)
        
        assert 'shared_spacings' in result
        assert 1 in result['shared_spacings']
        # Ratio is 4, so we should have spacings [1, 2, 4]
        assert 2 in result['shared_spacings']
        assert 4 in result['shared_spacings']


# =============================================================================
# Tests for boot_indexes functions
# =============================================================================

class TestBootIndexes:
    """Test boot index computation and retrieval functions."""
    
    # 1D Tests
    def test_compute_boot_indexes_1d(self, dataset_1d):
        """Test computation of boot indexes for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, num_bootstrappable
        )
        
        assert isinstance(boot_indexes, dict)
        assert len(boot_indexes) > 0
        assert 1 in boot_indexes
        assert dim in boot_indexes[1]
        
    def test_compute_boot_indexes_1d_no_bootstrappable(self, dataset_1d):
        """Test boot indexes with no bootstrappable dimensions."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, {dim: data_shape[dim]}, [1], 0
        )
        
        assert boot_indexes == {}
        
    def test_get_boot_indexes_1d(self, dataset_1d):
        """Test getting boot indexes for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, num_bootstrappable
        )
        
        # Default spacing
        indexes = get_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, boot_indexes, num_bootstrappable
        )
        
        assert isinstance(indexes, dict)
        assert dim in indexes
        
    def test_get_boot_indexes_1d_explicit_spacing(self, dataset_1d):
        """Test getting boot indexes with explicit spacing."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, num_bootstrappable
        )
        
        # Explicit spacing as int
        if 2 in boot_indexes:
            indexes = get_boot_indexes_1d(
                dim, data_shape, bootsize_dict, all_spacings, boot_indexes, 
                num_bootstrappable, spacing=2
            )
            assert dim in indexes
            
    def test_get_boot_indexes_1d_dict_spacing(self, dataset_1d):
        """Test getting boot indexes with dictionary spacing."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, num_bootstrappable
        )
        
        # Spacing as dict
        indexes = get_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, boot_indexes,
            num_bootstrappable, spacing={dim: 1}
        )
        assert dim in indexes
        
    def test_get_boot_indexes_1d_no_bootstrappable(self, dataset_1d):
        """Test get_boot_indexes_1d with no bootstrappable dimensions."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        indexes = get_boot_indexes_1d(
            dim, data_shape, {dim: data_shape[dim]}, [1], {}, 0
        )
        
        assert indexes == {}
        
    # 2D Tests
    def test_compute_boot_indexes_2d(self, dataset_2d):
        """Test computation of boot indexes for 2D dataset."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_2d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        assert isinstance(boot_indexes, dict)
        assert len(boot_indexes) > 0
        assert 1 in boot_indexes
        assert all(dim in boot_indexes[1] for dim in bootstrappable_dims)
        
    def test_get_boot_indexes_2d(self, dataset_2d):
        """Test getting boot indexes for 2D dataset."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_2d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        indexes = get_boot_indexes_2d(
            dims, data_shape, bootsize_dict, all_spacings, boot_indexes,
            bootstrappable_dims, num_bootstrappable
        )
        
        assert isinstance(indexes, dict)
        
    def test_get_boot_indexes_2d_no_bootstrappable(self, dataset_2d):
        """Test get_boot_indexes_2d with no bootstrappable dimensions."""
        dims, data_shape, _, _ = validate_dataset_2d(dataset_2d)
        
        indexes = get_boot_indexes_2d(
            dims, data_shape, {d: data_shape[d] for d in dims}, [1], {}, [], 0
        )
        
        assert indexes == {}
        
    # 3D Tests
    def test_compute_boot_indexes_3d(self, dataset_3d):
        """Test computation of boot indexes for 3D dataset."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        assert isinstance(boot_indexes, dict)
        assert len(boot_indexes) > 0
        assert 1 in boot_indexes
        assert all(dim in boot_indexes[1] for dim in bootstrappable_dims)
        
    def test_get_boot_indexes_3d(self, dataset_3d):
        """Test getting boot indexes for 3D dataset."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        indexes = get_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, boot_indexes,
            bootstrappable_dims, num_bootstrappable
        )
        
        assert isinstance(indexes, dict)
        
    def test_get_boot_indexes_3d_dict_spacing(self, dataset_3d):
        """Test getting 3D boot indexes with dictionary spacing."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        # Spacing as dict
        indexes = get_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, boot_indexes,
            bootstrappable_dims, num_bootstrappable, spacing={"x": 1, "y": 1, "z": 1}
        )
        
        assert isinstance(indexes, dict)
        
    def test_get_boot_indexes_3d_no_bootstrappable(self, dataset_3d):
        """Test get_boot_indexes_3d with no bootstrappable dimensions."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d)
        
        indexes = get_boot_indexes_3d(
            dims, data_shape, {d: data_shape[d] for d in dims}, [1], {}, [], 0
        )
        
        assert indexes == {}
        
    def test_boot_indexes_3d_large_dataset(self, dataset_3d_large):
        """Test boot indexes computation on larger dataset."""
        dims, data_shape, _, _ = validate_dataset_3d(dataset_3d_large)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        # Should have multiple spacing levels
        assert len(boot_indexes) > 1
        

# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_1d_pipeline(self, dataset_1d):
        """Test full 1D setup pipeline."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, bootsize_dict, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, num_bootstrappable
        )
        
        # Verify we can get indexes for all spacings
        for sp in all_spacings:
            indexes = get_boot_indexes_1d(
                dim, data_shape, bootsize_dict, all_spacings, boot_indexes,
                num_bootstrappable, spacing=sp
            )
            if sp in boot_indexes:
                assert dim in indexes
                
    def test_full_2d_pipeline(self, dataset_2d):
        """Test full 2D setup pipeline."""
        dims, data_shape, ds, time_dims = validate_dataset_2d(dataset_2d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_2d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        # Verify pipeline produces valid results
        assert len(boot_indexes) > 0
        assert isinstance(ds, xr.Dataset)
        
    def test_full_3d_pipeline(self, dataset_3d):
        """Test full 3D setup pipeline."""
        dims, data_shape, ds, time_dims = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        # Verify pipeline produces valid results
        assert len(boot_indexes) > 0
        assert isinstance(ds, xr.Dataset)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
