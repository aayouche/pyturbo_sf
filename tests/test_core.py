"""
Tests for core.py module functionality.
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.core import (
    validate_dataset_1d, validate_dataset_2d, validate_dataset_3d,
    setup_bootsize_1d, setup_bootsize_2d, setup_bootsize_3d,
    calculate_adaptive_spacings_1d, calculate_adaptive_spacings_2d, calculate_adaptive_spacings_3d,
    compute_boot_indexes_1d, compute_boot_indexes_2d, compute_boot_indexes_3d,
    get_boot_indexes_1d, get_boot_indexes_2d, get_boot_indexes_3d
)


# Fixtures for test datasets
@pytest.fixture
def dataset_1d():
    """Create a simple 1D dataset for testing."""
    # Create a simple 1D dataset
    x = np.linspace(0, 10, 100)
    data = np.sin(x)
    
    ds = xr.Dataset(
        data_vars={"data": ("x", data)},
        coords={"x": x}
    )
    return ds


@pytest.fixture
def dataset_2d():
    """Create a simple 2D dataset for testing."""
    # Create a simple 2D dataset
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 80)
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    # Create some sample data (u, v components and a scalar)
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
def dataset_3d():
    """Create a simple 3D dataset for testing."""
    # Create a simple 3D dataset (smaller size for faster tests)
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 15)
    z = np.linspace(0, 10, 10)
    
    # Create a meshgrid
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
	    
    # Create some sample data (u, v, w components and a scalar)
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


class TestValidateDataset:
    
    def test_validate_dataset_1d(self, dataset_1d):
        """Test validation of 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        # Check that the dimension is correctly identified
        assert dim == "x"
        
        # Check that the data shape is correctly extracted
        assert isinstance(data_shape, dict)
        assert "x" in data_shape
        assert data_shape["x"] == 100
        
    def test_validate_dataset_1d_error(self, dataset_2d):
        """Test that validation fails for dataset with wrong dimensions."""
        with pytest.raises(ValueError):
            validate_dataset_1d(dataset_2d)
            
    def test_validate_dataset_2d(self, dataset_2d):
        """Test validation of 2D dataset."""
        dims, data_shape, ds = validate_dataset_2d(dataset_2d)
        
        # Check that dimensions are correctly identified
        assert dims == ["y", "x"]
        
        # Check that the data shape is correctly extracted
        assert isinstance(data_shape, dict)
        assert "y" in data_shape and "x" in data_shape
        assert data_shape["y"] == 80
        assert data_shape["x"] == 100
        
        # Check that the dataset is returned correctly
        assert isinstance(ds, xr.Dataset)
        
    def test_validate_dataset_2d_error(self, dataset_1d):
        """Test that validation fails for dataset with wrong dimensions."""
        with pytest.raises(ValueError):
            validate_dataset_2d(dataset_1d)
            
    def test_validate_dataset_3d(self, dataset_3d):
        """Test validation of 3D dataset."""
        dims, data_shape, ds = validate_dataset_3d(dataset_3d)
        
        # Check that dimensions are correctly identified
        assert dims == ["z", "y", "x"]
        
        # Check that the data shape is correctly extracted
        assert isinstance(data_shape, dict)
        assert "z" in data_shape and "y" in data_shape and "x" in data_shape
        assert data_shape["z"] == 10
        assert data_shape["y"] == 15
        assert data_shape["x"] == 20
        
        # Check that the dataset is returned correctly
        assert isinstance(ds, xr.Dataset)
        
    def test_validate_dataset_3d_error(self, dataset_2d):
        """Test that validation fails for dataset with wrong dimensions."""
        with pytest.raises(ValueError):
            validate_dataset_3d(dataset_2d)


class TestSetupBootsize:
    
    def test_setup_bootsize_1d(self, dataset_1d):
        """Test setup of bootsize for 1D dataset."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        # Test default bootsize
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=None
        )
        
        assert isinstance(bootsize_dict, dict)
        assert dim in bootsize_dict
        assert bootstrappable_dims == [dim]
        assert num_bootstrappable == 1
        
        # Test custom bootsize (integer)
        custom_bootsize = 20
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=custom_bootsize
        )
        
        assert bootsize_dict[dim] == custom_bootsize
        
        # Test custom bootsize (dictionary)
        custom_bootsize_dict = {dim: 25}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(
            dim, data_shape, bootsize=custom_bootsize_dict
        )
        
        assert bootsize_dict[dim] == 25
        
    def test_setup_bootsize_2d(self, dataset_2d):
        """Test setup of bootsize for 2D dataset."""
        dims, data_shape, _ = validate_dataset_2d(dataset_2d)
        
        # Test default bootsize
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(
            dims, data_shape, bootsize=None
        )
        
        assert isinstance(bootsize_dict, dict)
        assert all(dim in bootsize_dict for dim in dims)
        assert set(bootstrappable_dims) == set(dims)
        assert num_bootstrappable == 2
        
        # Test custom bootsize (dictionary)
        custom_bootsize_dict = {"x": 25, "y": 20}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(
            dims, data_shape, bootsize=custom_bootsize_dict
        )
        
        assert bootsize_dict["x"] == 25
        assert bootsize_dict["y"] == 20
        
    def test_setup_bootsize_3d(self, dataset_3d):
        """Test setup of bootsize for 3D dataset."""
        dims, data_shape, _ = validate_dataset_3d(dataset_3d)
        
        # Test default bootsize
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(
            dims, data_shape, bootsize=None
        )
        
        assert isinstance(bootsize_dict, dict)
        assert all(dim in bootsize_dict for dim in dims)
        assert set(bootstrappable_dims) == set(dims)
        assert num_bootstrappable == 3
        
        # Test custom bootsize (dictionary)
        custom_bootsize_dict = {"x": 5, "y": 5, "z": 5}
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(
            dims, data_shape, bootsize=custom_bootsize_dict
        )
        
        assert bootsize_dict["x"] == 5
        assert bootsize_dict["y"] == 5
        assert bootsize_dict["z"] == 5


class TestCalculateAdaptiveSpacings:
    
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
        assert 1 in all_spacings  # Default spacing of 1 should be included
        
    def test_calculate_adaptive_spacings_1d_no_bootstrappable(self, dataset_1d):
        """Test spacings calculation with no bootstrappable dimensions."""
        dim, data_shape = validate_dataset_1d(dataset_1d)
        
        # Force no bootstrappable dimensions
        spacings_info, all_spacings = calculate_adaptive_spacings_1d(
            dim, data_shape, {dim: data_shape[dim]}, 0
        )
        
        assert isinstance(spacings_info, dict)
        assert 'spacings' in spacings_info
        assert spacings_info['spacings'] == [1]  # Only default spacing for no bootstrapping
        assert all_spacings == [1]
        
    def test_calculate_adaptive_spacings_2d(self, dataset_2d):
        """Test calculation of adaptive spacings for 2D dataset."""
        dims, data_shape, _ = validate_dataset_2d(dataset_2d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        assert isinstance(spacings_info, dict)
        assert 'shared_spacings' in spacings_info
        assert isinstance(all_spacings, list)
        assert len(all_spacings) > 0
        assert 1 in all_spacings  # Default spacing of 1 should be included
        
    def test_calculate_adaptive_spacings_3d(self, dataset_3d):
        """Test calculation of adaptive spacings for 3D dataset."""
        dims, data_shape, _ = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        assert isinstance(spacings_info, dict)
        assert 'shared_spacings' in spacings_info
        assert isinstance(all_spacings, list)
        assert len(all_spacings) > 0
        assert 1 in all_spacings  # Default spacing of 1 should be included


class TestBootIndexes:
    
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
        
        # Check that indexes for default spacing (1) exist
        assert 1 in boot_indexes
        assert dim in boot_indexes[1]
        
        # Check the shape of the indexes
        window_size = bootsize_dict[dim]
        assert boot_indexes[1][dim].shape[1] > 0
        
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
        
        # Get indexes for default spacing
        indexes = get_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, boot_indexes, num_bootstrappable
        )
        
        assert isinstance(indexes, dict)
        assert dim in indexes
        assert indexes[dim].shape[1] > 0
        
        # Test with explicit spacing
        indexes_sp2 = get_boot_indexes_1d(
            dim, data_shape, bootsize_dict, all_spacings, boot_indexes, num_bootstrappable, spacing=2
        )
        
        # If spacing 2 is valid, it should return valid indexes
        if 2 in boot_indexes:
            assert dim in indexes_sp2
        
    def test_compute_boot_indexes_2d(self, dataset_2d):
        """Test computation of boot indexes for 2D dataset."""
        dims, data_shape, _ = validate_dataset_2d(dataset_2d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_2d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_2d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        assert isinstance(boot_indexes, dict)
        assert len(boot_indexes) > 0
        
        # Check that indexes for default spacing (1) exist
        assert 1 in boot_indexes
        assert all(dim in boot_indexes[1] for dim in bootstrappable_dims)
        
    def test_compute_boot_indexes_3d(self, dataset_3d):
        """Test computation of boot indexes for 3D dataset."""
        dims, data_shape, _ = validate_dataset_3d(dataset_3d)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape)
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        assert isinstance(boot_indexes, dict)
        assert len(boot_indexes) > 0
        
        # Check that indexes for default spacing (1) exist
        assert 1 in boot_indexes
        assert all(dim in boot_indexes[1] for dim in bootstrappable_dims)


if __name__ == "__main__":
    pytest.main(["-v", "test_core.py"])
