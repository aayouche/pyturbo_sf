"""
Tests for utils.py module functionality.
"""

import pytest
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

from pyturbo_sf.utils import (
    fast_shift_1d, fast_shift_2d, fast_shift_3d,
    calculate_time_diff_1d,
    map_variables_by_pattern_2d, map_variables_by_pattern_3d,
    check_and_reorder_variables_2d, check_and_reorder_variables_3d
)


class TestShiftFunctions:
    
    def test_fast_shift_1d(self):
        """Test 1D array shifting."""
        # Create a test array
        input_array = np.array([1, 2, 3, 4, 5])
        
        # Test shift by 0 (no shift)
        shifted = fast_shift_1d(input_array, shift=0)
        np.testing.assert_array_equal(shifted, input_array)
        
        # Test positive shift
        shifted = fast_shift_1d(input_array, shift=2)
        
        # Check that the first part contains the correct values
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # Check the second part of the array (implementation dependent)
        # Some implementations might use NaN, others might use other values
        # We just ensure these values don't equal the original array values
        for i in range(3, len(shifted)):
            # Either it's NaN or it doesn't equal the original value
            assert np.isnan(shifted[i]) or shifted[i] != input_array[i]
        
        # Verify that the original array is untouched
        np.testing.assert_array_equal(input_array, np.array([1, 2, 3, 4, 5]))
        
    def test_fast_shift_1d_datetime(self):
        """Test 1D array shifting with datetime objects."""
        # Test with Python datetime array
        base_date = datetime(2023, 1, 1)
        dates = np.array([base_date + timedelta(days=i) for i in range(5)])
        
        # Test shift by 0 (no shift)
        shifted_dates = fast_shift_1d(dates, shift=0)
        for i in range(len(dates)):
            assert shifted_dates[i] == dates[i]
        
        # Test shift by 2
        shifted_dates = fast_shift_1d(dates, shift=2)
        
        # Check that the first part contains the correct values
        for i in range(3):
            assert shifted_dates[i] == dates[i+2]
        
        # For Python datetime objects, NA values should be None
        for i in range(3, len(shifted_dates)):
            assert shifted_dates[i] is None
            
    def test_fast_shift_1d_datetime64(self):
        """Test 1D array shifting with numpy datetime64 arrays."""
        # Test with numpy datetime64 array
        base_date = datetime(2023, 1, 1)
        dates_np = np.array([np.datetime64(base_date + timedelta(days=i)) for i in range(5)])
        
        # Test shift by 2
        shifted_dates_np = fast_shift_1d(dates_np, shift=2)
        
        # Check that the first part contains the correct values
        for i in range(3):
            assert shifted_dates_np[i] == dates_np[i+2]
        
        # For datetime64 arrays, invalid values should be NaT
        for i in range(3, len(shifted_dates_np)):
            assert np.isnat(shifted_dates_np[i])
            
    def test_fast_shift_1d_integer(self):
        """Test 1D array shifting with integer arrays."""
        # Create an integer array
        input_array = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        
        # Test shift by 2
        shifted = fast_shift_1d(input_array, shift=2)
        
        # Check that the first part contains the correct values
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # For integer arrays, NA values should be 0
        for i in range(3, len(shifted)):
            assert shifted[i] == 0
            
    def test_fast_shift_1d_boolean(self):
        """Test 1D array shifting with boolean arrays."""
        # Create a boolean array
        input_array = np.array([True, False, True, False, True], dtype=np.bool_)
        
        # Test shift by 2
        shifted = fast_shift_1d(input_array, shift=2)
        
        # Check that the first part contains the correct values
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # For boolean arrays, NA values should be False
        for i in range(3, len(shifted)):
            assert shifted[i] == False
            
    def test_fast_shift_1d_float(self):
        """Test 1D array shifting with float arrays."""
        # Create a float array
        input_array = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
        
        # Test shift by 2
        shifted = fast_shift_1d(input_array, shift=2)
        
        # Check that the first part contains the correct values
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # For float arrays, NA values should be NaN
        for i in range(3, len(shifted)):
            assert np.isnan(shifted[i])
        
    def test_fast_shift_2d(self):
        """Test 2D array shifting."""
        # Create a test array
        input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Test shift by 0 in both dimensions (no shift)
        shifted = fast_shift_2d(input_array, y_shift=0, x_shift=0)
        np.testing.assert_array_equal(shifted, input_array)
        
        # Test shift in x direction
        shifted = fast_shift_2d(input_array, y_shift=0, x_shift=1)
        expected_x = np.array([[2, 3, np.nan], [5, 6, np.nan], [8, 9, np.nan]])
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:, :2], input_array[:, 1:])
        assert np.all(np.isnan(shifted[:, 2]))
        
        # Test shift in y direction
        shifted = fast_shift_2d(input_array, y_shift=1, x_shift=0)
        expected_y = np.array([[4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]])
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:2, :], input_array[1:, :])
        assert np.all(np.isnan(shifted[2, :]))
        
        # Test shift in both directions
        shifted = fast_shift_2d(input_array, y_shift=1, x_shift=1)
        expected_xy = np.array([[5, 6, np.nan], [8, 9, np.nan], [np.nan, np.nan, np.nan]])
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:2, :2], input_array[1:, 1:])
        assert np.all(np.isnan(shifted[2, :]))
        assert np.all(np.isnan(shifted[:, 2]))
        
    def test_fast_shift_3d(self):
        """Test 3D array shifting."""
        # Create a test array (2x2x2 for simplicity)
        input_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        # Test shift by 0 in all dimensions (no shift)
        shifted = fast_shift_3d(input_array, z_shift=0, y_shift=0, x_shift=0)
        np.testing.assert_array_equal(shifted, input_array)
        
        # Test shift in x direction
        shifted = fast_shift_3d(input_array, z_shift=0, y_shift=0, x_shift=1)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:, :, 0], input_array[:, :, 1])
        assert np.all(np.isnan(shifted[:, :, 1]))
        
        # Test shift in y direction
        shifted = fast_shift_3d(input_array, z_shift=0, y_shift=1, x_shift=0)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:, 0, :], input_array[:, 1, :])
        assert np.all(np.isnan(shifted[:, 1, :]))
        
        # Test shift in z direction
        shifted = fast_shift_3d(input_array, z_shift=1, y_shift=0, x_shift=0)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[0, :, :], input_array[1, :, :])
        assert np.all(np.isnan(shifted[1, :, :]))
        
        # Test shift in all directions
        shifted = fast_shift_3d(input_array, z_shift=1, y_shift=1, x_shift=1)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[0, 0, 0], input_array[1, 1, 1])
        assert np.all(np.isnan(shifted[1, :, :]))
        assert np.all(np.isnan(shifted[0, 1, :]))
        assert np.all(np.isnan(shifted[0, 0, 1]))


class TestTimeDifference:
    
    def test_calculate_time_diff_1d_datetime(self):
        """Test time difference calculation with datetime objects."""
        # Create an array of datetime objects
        base_date = datetime(2023, 1, 1, 12, 0, 0)
        dates = np.array([base_date + timedelta(hours=i) for i in range(5)])
        
        # Calculate time difference with shift=1
        diff = calculate_time_diff_1d(dates, shift=1)

        # Expected difference is 3600 seconds (1 hour)
        expected = np.array([3600.0, 3600.0, 3600.0, 3600.0, np.nan])
        
        # Check shape and values
        assert diff.shape == dates.shape
        np.testing.assert_almost_equal(diff[:-1], expected[:-1])
        assert np.isnan(diff[-1])
        
    def test_calculate_time_diff_1d_datetime64(self):
        """Test time difference calculation with numpy datetime64 values."""
        # Create an array of datetime64 objects with hourly intervals
        dates = np.array(['2023-01-01T00:00:00', '2023-01-01T01:00:00', 
                         '2023-01-01T02:00:00', '2023-01-01T03:00:00', 
                         '2023-01-01T04:00:00'], dtype='datetime64[s]')
        
        # Calculate time difference with shift=1
        diff = calculate_time_diff_1d(dates, shift=1)
        
        # Expected difference is 3600 seconds (1 hour)
        expected = np.array([3600.0, 3600.0, 3600.0, 3600.0, np.nan])
        
        # Check shape and values
        assert diff.shape == dates.shape
        np.testing.assert_almost_equal(diff[:-1], expected[:-1])
        assert np.isnan(diff[-1])
        
    def test_calculate_time_diff_1d_numeric(self):
        """Test time difference calculation with numeric values."""
        # Create a numeric array (already in seconds)
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        # Calculate time difference with shift=1
        diff = calculate_time_diff_1d(times, shift=1)
        
        # Expected difference is 1.0
        expected = np.array([1.0, 1.0, 1.0, 1.0, np.nan])
        
        # Check shape and values
        assert diff.shape == times.shape
        np.testing.assert_almost_equal(diff[:-1], expected[:-1])
        assert np.isnan(diff[-1])
        
    def test_calculate_time_diff_1d_zero_shift(self):
        """Test time difference calculation with zero shift."""
        # Create an array of datetime objects
        base_date = datetime(2023, 1, 1, 12, 0, 0)
        dates = np.array([base_date + timedelta(hours=i) for i in range(5)])
        
        # Calculate time difference with shift=0
        diff = calculate_time_diff_1d(dates, shift=0)
        
        # Expected difference is all zeros
        expected = np.zeros(5, dtype=float)
        
        # Check shape and values
        assert diff.shape == dates.shape
        np.testing.assert_array_equal(diff, expected)


class TestVariableReordering:
    
    def test_map_variables_by_pattern_2d(self):
        """Test variable name mapping by pattern for 2D."""
        # Test for (y, x) plane with u, v components
        provided = ["velocity_x", "v_vel"]
        expected = ["u", "v"]
        plane_tuple = ("y", "x")
        
        mapped = map_variables_by_pattern_2d(provided, expected, plane_tuple)
        assert mapped == tuple(provided)
        
        # Test for no match
        provided = ["temp", "pressure"]
        mapped = map_variables_by_pattern_2d(provided, expected, plane_tuple)
        assert mapped is None
        
    def test_check_and_reorder_variables_2d(self):
        """Test checking and reordering variables for 2D structure functions."""
        # Test longitudinal function with correct order
        variables = ["u", "v"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        # Test longitudinal function with reversed order
        variables = ["v", "u"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == ("u", "v")
        
        # Test with alternative variable names
        variables = ["velocity_x", "velocity_y"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        # Test scalar function
        variables = ["temperature"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="scalar")
        assert result == tuple(variables)
        
        # Test scalar-scalar function
        variables = ["temperature", "pressure"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="scalar_scalar")
        assert result == tuple(variables)
        
        # Test for other planes
        variables = ["u", "w"]
        dims = ["z", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        variables = ["v", "w"]
        dims = ["z", "y"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_2d_error(self):
        """Test error cases for variable reordering in 2D."""
        # Test wrong number of variables
        variables = ["u", "v", "w"]
        dims = ["y", "x"]
        with pytest.raises(ValueError):
            check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
            
        # Test unsupported dimension combination
        variables = ["u", "v"]
        dims = ["a", "b"]
        with pytest.raises(ValueError):
            check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
            
    def test_check_and_reorder_variables_3d(self):
        """Test checking and reordering variables for 3D structure functions."""
        # Test longitudinal function with correct order
        variables = ["u", "v", "w"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        # Test longitudinal function with different order
        variables = ["w", "u", "v"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
        assert result == ("u", "v", "w")
        
        # Test with alternative variable names
        variables = ["velocity_x", "velocity_y", "velocity_z"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
        assert len(result) == 3
        
        # Test transverse function
        variables = ["u", "w"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="transverse_ik")
        assert result == tuple(variables)
        
        # Test scalar function
        variables = ["temperature"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="scalar")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_3d_error(self):
        """Test error cases for variable reordering in 3D."""
        # Test wrong number of variables
        variables = ["u", "v"]
        dims = ["z", "y", "x"]
        with pytest.raises(ValueError):
            check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
            
        # Test unsupported dimension order
        variables = ["u", "v", "w"]
        dims = ["x", "y", "z"]  # Wrong order
        with pytest.raises(ValueError):
            check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
            

if __name__ == "__main__":
    pytest.main(["-v", "test_utils.py"])
