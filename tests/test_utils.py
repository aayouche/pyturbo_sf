"""
Tests for utils.py module functionality.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

from pyturbo_sf.utils import (
    _is_log_spaced,
    _calculate_confidence_intervals,
    _calculate_quality_mask,
    fast_shift_1d, fast_shift_2d, fast_shift_3d,
    calculate_time_diff_1d,
    map_variables_by_pattern_2d, map_variables_by_pattern_3d,
    check_and_reorder_variables_2d, check_and_reorder_variables_3d
)


class TestIsLogSpaced:
    """Tests for _is_log_spaced function."""
    
    def test_log_spaced_array(self):
        """Test detection of logarithmically spaced array."""
        # Create a logarithmically spaced array
        log_array = np.logspace(0, 3, 20)  # 1 to 1000 in 20 steps
        result = _is_log_spaced(log_array)
        assert result, f"Expected True for log-spaced array, got {result}"
        
    def test_linear_spaced_array(self):
        """Test detection of linearly spaced array."""
        # Create a linearly spaced array
        linear_array = np.linspace(1, 100, 20)
        result = _is_log_spaced(linear_array)
        assert not result, f"Expected False for linear array, got {result}"
        
    def test_short_array(self):
        """Test behavior with array too short to determine spacing."""
        short_array = np.array([1.0])
        result = _is_log_spaced(short_array)
        assert not result, f"Expected False for short array, got {result}"
        
        # Edge case: exactly 2 elements
        two_elem = np.array([1.0, 2.0])
        # Should not raise an error
        result = _is_log_spaced(two_elem)
        assert isinstance(result, (bool, np.bool_))
        
    def test_empty_array(self):
        """Test behavior with empty array."""
        empty_array = np.array([])
        result = _is_log_spaced(empty_array)
        assert not result, f"Expected False for empty array, got {result}"
        
    def test_constant_spacing_ratio(self):
        """Test with constant ratio (geometric sequence)."""
        # Geometric sequence with ratio 2
        geom_array = np.array([1, 2, 4, 8, 16, 32, 64])
        result = _is_log_spaced(geom_array)
        assert result, f"Expected True for geometric sequence, got {result}"
        
    def test_nearly_linear_array(self):
        """Test array with ratio close to 1 (nearly linear)."""
        # Array where ratio is very close to 1
        nearly_linear = np.array([1.0, 1.001, 1.002, 1.003, 1.004])
        result = _is_log_spaced(nearly_linear)
        assert not result, f"Expected False for nearly linear array, got {result}"


class TestCalculateConfidenceIntervals:
    """Tests for _calculate_confidence_intervals function."""
    
    def test_basic_confidence_interval(self):
        """Test basic confidence interval calculation."""
        means = np.array([10.0, 20.0, 30.0])
        stds = np.array([1.0, 2.0, 3.0])
        counts = np.array([100, 100, 100])
        
        ci_upper, ci_lower = _calculate_confidence_intervals(means, stds, counts)
        
        # Check shapes
        assert ci_upper.shape == means.shape
        assert ci_lower.shape == means.shape
        
        # Check that upper > mean > lower for valid values
        assert np.all(ci_upper > means)
        assert np.all(ci_lower < means)
        
    def test_confidence_interval_with_nan(self):
        """Test confidence interval with NaN values."""
        means = np.array([10.0, np.nan, 30.0])
        stds = np.array([1.0, 2.0, np.nan])
        counts = np.array([100, 100, 100])
        
        ci_upper, ci_lower = _calculate_confidence_intervals(means, stds, counts)
        
        # Valid indices should have computed values
        assert not np.isnan(ci_upper[0])
        assert not np.isnan(ci_lower[0])
        
        # NaN indices should remain NaN
        assert np.isnan(ci_upper[1])
        assert np.isnan(ci_lower[1])
        assert np.isnan(ci_upper[2])
        assert np.isnan(ci_lower[2])
        
    def test_confidence_interval_low_counts(self):
        """Test confidence interval with low counts."""
        means = np.array([10.0, 20.0, 30.0])
        stds = np.array([1.0, 2.0, 3.0])
        counts = np.array([100, 1, 0])  # Second and third have insufficient counts
        
        ci_upper, ci_lower = _calculate_confidence_intervals(means, stds, counts)
        
        # First should be valid
        assert not np.isnan(ci_upper[0])
        assert not np.isnan(ci_lower[0])
        
        # Counts <= 1 should be NaN
        assert np.isnan(ci_upper[1])
        assert np.isnan(ci_lower[1])
        assert np.isnan(ci_upper[2])
        assert np.isnan(ci_lower[2])
        
    def test_confidence_level_parameter(self):
        """Test different confidence levels."""
        means = np.array([10.0])
        stds = np.array([1.0])
        counts = np.array([100])
        
        # 95% confidence level (default)
        ci_upper_95, ci_lower_95 = _calculate_confidence_intervals(
            means, stds, counts, confidence_level=0.95
        )
        
        # 99% confidence level (wider interval)
        ci_upper_99, ci_lower_99 = _calculate_confidence_intervals(
            means, stds, counts, confidence_level=0.99
        )
        
        # 99% CI should be wider than 95% CI
        assert ci_upper_99[0] > ci_upper_95[0]
        assert ci_lower_99[0] < ci_lower_95[0]
        
    def test_z_score_calculation(self):
        """Test that z-score is correctly applied."""
        means = np.array([0.0])
        stds = np.array([1.0])
        counts = np.array([100])
        
        ci_upper, ci_lower = _calculate_confidence_intervals(
            means, stds, counts, confidence_level=0.95
        )
        
        # For 95% CI, z-score ≈ 1.96
        expected_z = stats.norm.ppf(0.975)
        np.testing.assert_almost_equal(ci_upper[0], expected_z, decimal=5)
        np.testing.assert_almost_equal(ci_lower[0], -expected_z, decimal=5)


class TestCalculateQualityMask:
    """Tests for _calculate_quality_mask function."""
    
    def test_basic_quality_mask(self):
        """Test basic quality mask calculation."""
        sf_bessel = np.array([1.0, 2.0, 3.0, np.nan])
        sf_stds = np.array([0.1, 0.2, 0.3, 0.1])
        point_counts = np.array([100, 50, 5, 100])
        eiso = np.array([0.01, 0.05, 0.1, 0.01])
        converged = np.array([True, True, True, True])
        
        mask = _calculate_quality_mask(
            sf_bessel, sf_stds, point_counts, eiso, converged,
            min_points=10
        )
        
        # Check that NaN values are excluded
        assert not mask[3], f"Expected False for NaN value, got {mask[3]}"
        
        # Check that low point counts are excluded
        assert not mask[2], f"Expected False for low point count, got {mask[2]}"
        
        # Valid values should be included
        assert mask[0], f"Expected True for valid value, got {mask[0]}"
        assert mask[1], f"Expected True for valid value, got {mask[1]}"
        
    def test_isotropy_error_threshold(self):
        """Test max_isotropy_error parameter."""
        sf_bessel = np.array([1.0, 2.0, 3.0])
        sf_stds = np.array([0.1, 0.2, 0.3])
        point_counts = np.array([100, 100, 100])
        eiso = np.array([0.01, 0.1, 0.5])
        converged = np.array([True, True, True])
        
        mask = _calculate_quality_mask(
            sf_bessel, sf_stds, point_counts, eiso, converged,
            min_points=10, max_isotropy_error=0.2
        )
        
        # First two should pass, third should fail due to high isotropy error
        assert mask[0], f"Expected True for low isotropy error, got {mask[0]}"
        assert mask[1], f"Expected True for medium isotropy error, got {mask[1]}"
        assert not mask[2], f"Expected False for high isotropy error, got {mask[2]}"
        
    def test_std_ratio_threshold(self):
        """Test max_std_ratio parameter."""
        sf_bessel = np.array([1.0, 1.0, 1.0])
        sf_stds = np.array([0.1, 0.5, 2.0])  # std/mean ratios: 0.1, 0.5, 2.0
        point_counts = np.array([100, 100, 100])
        eiso = np.array([0.01, 0.01, 0.01])
        converged = np.array([True, True, True])
        
        mask = _calculate_quality_mask(
            sf_bessel, sf_stds, point_counts, eiso, converged,
            min_points=10, max_std_ratio=1.0
        )
        
        # First two should pass, third should fail due to high std ratio
        assert mask[0], f"Expected True for low std ratio, got {mask[0]}"
        assert mask[1], f"Expected True for medium std ratio, got {mask[1]}"
        assert not mask[2], f"Expected False for high std ratio, got {mask[2]}"
        
    def test_all_filters_combined(self):
        """Test with all filters applied."""
        sf_bessel = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sf_stds = np.array([0.1, 0.2, 0.3, 4.0, 0.5])
        point_counts = np.array([100, 5, 100, 100, 100])
        eiso = np.array([0.01, 0.01, 0.01, 0.01, 0.5])
        converged = np.array([True, True, True, True, True])
        
        mask = _calculate_quality_mask(
            sf_bessel, sf_stds, point_counts, eiso, converged,
            min_points=10, max_isotropy_error=0.2, max_std_ratio=0.5
        )
        
        # Only first should pass all filters
        assert mask[0], f"Expected True for valid value, got {mask[0]}"
        assert not mask[1], f"Expected False (low point count), got {mask[1]}"
        assert not mask[2], f"Expected False (NaN value), got {mask[2]}"
        assert not mask[3], f"Expected False (high std ratio), got {mask[3]}"
        assert not mask[4], f"Expected False (high isotropy error), got {mask[4]}"
        
    def test_nan_isotropy_handling(self):
        """Test handling of NaN isotropy values."""
        sf_bessel = np.array([1.0, 2.0])
        sf_stds = np.array([0.1, 0.2])
        point_counts = np.array([100, 100])
        eiso = np.array([0.01, np.nan])
        converged = np.array([True, True])
        
        mask = _calculate_quality_mask(
            sf_bessel, sf_stds, point_counts, eiso, converged,
            min_points=10, max_isotropy_error=0.2
        )
        
        # NaN isotropy should not exclude point (treated as valid)
        assert mask[0], f"Expected True for valid value, got {mask[0]}"
        assert mask[1], f"Expected True for NaN isotropy (treated as valid), got {mask[1]}"


class TestShiftFunctions:
    """Tests for array shifting functions."""
    
    def test_fast_shift_1d(self):
        """Test 1D array shifting."""
        input_array = np.array([1, 2, 3, 4, 5])
        
        # Test shift by 0 (no shift)
        shifted = fast_shift_1d(input_array, shift=0)
        np.testing.assert_array_equal(shifted, input_array)
        
        # Test positive shift
        shifted = fast_shift_1d(input_array, shift=2)
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # Check fill values
        for i in range(3, len(shifted)):
            assert np.isnan(shifted[i]) or shifted[i] != input_array[i]
        
        # Verify original array is untouched
        np.testing.assert_array_equal(input_array, np.array([1, 2, 3, 4, 5]))
        
    def test_fast_shift_1d_datetime(self):
        """Test 1D array shifting with datetime objects."""
        base_date = datetime(2023, 1, 1)
        dates = np.array([base_date + timedelta(days=i) for i in range(5)])
        
        # Test shift by 0
        shifted_dates = fast_shift_1d(dates, shift=0)
        for i in range(len(dates)):
            assert shifted_dates[i] == dates[i]
        
        # Test shift by 2
        shifted_dates = fast_shift_1d(dates, shift=2)
        for i in range(3):
            assert shifted_dates[i] == dates[i+2]
        for i in range(3, len(shifted_dates)):
            assert shifted_dates[i] is None
            
    def test_fast_shift_1d_datetime64(self):
        """Test 1D array shifting with numpy datetime64 arrays."""
        base_date = datetime(2023, 1, 1)
        dates_np = np.array([np.datetime64(base_date + timedelta(days=i)) for i in range(5)])
        
        shifted_dates_np = fast_shift_1d(dates_np, shift=2)
        
        for i in range(3):
            assert shifted_dates_np[i] == dates_np[i+2]
        for i in range(3, len(shifted_dates_np)):
            assert np.isnat(shifted_dates_np[i])
            
    def test_fast_shift_1d_integer(self):
        """Test 1D array shifting with integer arrays."""
        input_array = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        
        shifted = fast_shift_1d(input_array, shift=2)
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # For integer arrays, NA values should be 0
        for i in range(3, len(shifted)):
            assert shifted[i] == 0
            
    def test_fast_shift_1d_boolean(self):
        """Test 1D array shifting with boolean arrays."""
        input_array = np.array([True, False, True, False, True], dtype=np.bool_)
        
        shifted = fast_shift_1d(input_array, shift=2)
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # For boolean arrays, NA values should be False
        for i in range(3, len(shifted)):
            assert shifted[i] == False
            
    def test_fast_shift_1d_float(self):
        """Test 1D array shifting with float arrays."""
        input_array = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=np.float64)
        
        shifted = fast_shift_1d(input_array, shift=2)
        np.testing.assert_array_equal(shifted[:3], input_array[2:])
        
        # For float arrays, NA values should be NaN
        for i in range(3, len(shifted)):
            assert np.isnan(shifted[i])
        
    def test_fast_shift_2d(self):
        """Test 2D array shifting."""
        input_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Test no shift
        shifted = fast_shift_2d(input_array, y_shift=0, x_shift=0)
        np.testing.assert_array_equal(shifted, input_array)
        
        # Test x shift
        shifted = fast_shift_2d(input_array, y_shift=0, x_shift=1)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:, :2], input_array[:, 1:])
        assert np.all(np.isnan(shifted[:, 2]))
        
        # Test y shift
        shifted = fast_shift_2d(input_array, y_shift=1, x_shift=0)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:2, :], input_array[1:, :])
        assert np.all(np.isnan(shifted[2, :]))
        
        # Test both shifts
        shifted = fast_shift_2d(input_array, y_shift=1, x_shift=1)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:2, :2], input_array[1:, 1:])
        assert np.all(np.isnan(shifted[2, :]))
        assert np.all(np.isnan(shifted[:, 2]))
        
    def test_fast_shift_3d(self):
        """Test 3D array shifting."""
        input_array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        # Test no shift
        shifted = fast_shift_3d(input_array, z_shift=0, y_shift=0, x_shift=0)
        np.testing.assert_array_equal(shifted, input_array)
        
        # Test x shift
        shifted = fast_shift_3d(input_array, z_shift=0, y_shift=0, x_shift=1)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:, :, 0], input_array[:, :, 1])
        assert np.all(np.isnan(shifted[:, :, 1]))
        
        # Test y shift
        shifted = fast_shift_3d(input_array, z_shift=0, y_shift=1, x_shift=0)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[:, 0, :], input_array[:, 1, :])
        assert np.all(np.isnan(shifted[:, 1, :]))
        
        # Test z shift
        shifted = fast_shift_3d(input_array, z_shift=1, y_shift=0, x_shift=0)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[0, :, :], input_array[1, :, :])
        assert np.all(np.isnan(shifted[1, :, :]))
        
        # Test all shifts
        shifted = fast_shift_3d(input_array, z_shift=1, y_shift=1, x_shift=1)
        assert shifted.shape == input_array.shape
        np.testing.assert_array_equal(shifted[0, 0, 0], input_array[1, 1, 1])
        assert np.all(np.isnan(shifted[1, :, :]))
        assert np.all(np.isnan(shifted[0, 1, :]))


class TestTimeDifference:
    """Tests for calculate_time_diff_1d function."""
    
    def test_calculate_time_diff_1d_datetime(self):
        """Test time difference calculation with datetime objects."""
        base_date = datetime(2023, 1, 1, 12, 0, 0)
        dates = np.array([base_date + timedelta(hours=i) for i in range(5)])
        
        diff = calculate_time_diff_1d(dates, shift=1)
        expected = np.array([3600.0, 3600.0, 3600.0, 3600.0, np.nan])
        
        assert diff.shape == dates.shape
        np.testing.assert_almost_equal(diff[:-1], expected[:-1])
        assert np.isnan(diff[-1])
        
    def test_calculate_time_diff_1d_datetime64(self):
        """Test time difference calculation with numpy datetime64 values."""
        dates = np.array(['2023-01-01T00:00:00', '2023-01-01T01:00:00', 
                         '2023-01-01T02:00:00', '2023-01-01T03:00:00', 
                         '2023-01-01T04:00:00'], dtype='datetime64[s]')
        
        diff = calculate_time_diff_1d(dates, shift=1)
        expected = np.array([3600.0, 3600.0, 3600.0, 3600.0, np.nan])
        
        assert diff.shape == dates.shape
        np.testing.assert_almost_equal(diff[:-1], expected[:-1])
        assert np.isnan(diff[-1])
        
    def test_calculate_time_diff_1d_numeric(self):
        """Test time difference calculation with numeric values."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        diff = calculate_time_diff_1d(times, shift=1)
        expected = np.array([1.0, 1.0, 1.0, 1.0, np.nan])
        
        assert diff.shape == times.shape
        np.testing.assert_almost_equal(diff[:-1], expected[:-1])
        assert np.isnan(diff[-1])
        
    def test_calculate_time_diff_1d_zero_shift(self):
        """Test time difference calculation with zero shift."""
        base_date = datetime(2023, 1, 1, 12, 0, 0)
        dates = np.array([base_date + timedelta(hours=i) for i in range(5)])
        
        diff = calculate_time_diff_1d(dates, shift=0)
        expected = np.zeros(5, dtype=float)
        
        assert diff.shape == dates.shape
        np.testing.assert_array_equal(diff, expected)
        
    def test_calculate_time_diff_1d_larger_shift(self):
        """Test time difference calculation with larger shift."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        diff = calculate_time_diff_1d(times, shift=3)
        
        # Only first two values should have valid differences
        np.testing.assert_almost_equal(diff[0], 3.0)
        np.testing.assert_almost_equal(diff[1], 3.0)
        assert np.isnan(diff[2])
        assert np.isnan(diff[3])
        assert np.isnan(diff[4])


class TestVariableReordering:
    """Tests for variable name mapping and reordering functions."""
    
    def test_map_variables_by_pattern_2d(self):
        """Test variable name mapping by pattern for 2D."""
        provided = ["velocity_x", "v_vel"]
        expected = ["u", "v"]
        plane_tuple = ("y", "x")
        
        mapped = map_variables_by_pattern_2d(provided, expected, plane_tuple)
        assert mapped == tuple(provided)
        
        # Test for no match
        provided = ["temp", "pressure"]
        mapped = map_variables_by_pattern_2d(provided, expected, plane_tuple)
        assert mapped is None
        
    def test_map_variables_by_pattern_2d_exact_match(self):
        """Test exact variable name matching for 2D."""
        provided = ["u", "v"]
        expected = ["u", "v"]
        plane_tuple = ("y", "x")
        
        mapped = map_variables_by_pattern_2d(provided, expected, plane_tuple)
        assert mapped == ("u", "v")
        
    def test_map_variables_by_pattern_3d(self):
        """Test variable name mapping by pattern for 3D."""
        provided = ["velocity_x", "v_vel", "velocity_z"]
        expected = ["u", "v", "w"]
        
        mapped = map_variables_by_pattern_3d(provided, expected)
        assert mapped is not None
        assert len(mapped) == 3
        
    def test_map_variables_by_pattern_3d_no_match(self):
        """Test 3D mapping with no matching patterns."""
        provided = ["temp", "pressure", "humidity"]
        expected = ["u", "v", "w"]
        
        mapped = map_variables_by_pattern_3d(provided, expected)
        assert mapped is None
        
    def test_check_and_reorder_variables_2d_longitudinal(self):
        """Test checking and reordering variables for 2D longitudinal."""
        # Correct order
        variables = ["u", "v"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        # Reversed order
        variables = ["v", "u"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == ("u", "v")
        
    def test_check_and_reorder_variables_2d_transverse(self):
        """Test 2D transverse function variable reordering."""
        variables = ["u", "v"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="transverse")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_2d_scalar(self):
        """Test 2D scalar function."""
        variables = ["temperature"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="scalar")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_2d_scalar_scalar(self):
        """Test 2D scalar-scalar function."""
        variables = ["temperature", "pressure"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="scalar_scalar")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_2d_longitudinal_scalar(self):
        """Test 2D longitudinal-scalar function."""
        variables = ["u", "v", "temperature"]
        dims = ["y", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal_scalar")
        assert len(result) == 3
        # Scalar should be last
        assert result[2] == "temperature"
        
    def test_check_and_reorder_variables_2d_other_planes(self):
        """Test 2D with different plane combinations."""
        # (z, x) plane
        variables = ["u", "w"]
        dims = ["z", "x"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        # (z, y) plane
        variables = ["v", "w"]
        dims = ["z", "y"]
        result = check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_2d_error_wrong_count(self):
        """Test error for wrong number of variables."""
        variables = ["u", "v", "w"]
        dims = ["y", "x"]
        with pytest.raises(ValueError):
            check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
            
    def test_check_and_reorder_variables_2d_error_unsupported_dims(self):
        """Test error for unsupported dimension combination."""
        variables = ["u", "v"]
        dims = ["a", "b"]
        with pytest.raises(ValueError):
            check_and_reorder_variables_2d(variables, dims, fun="longitudinal")
            
    def test_check_and_reorder_variables_3d_longitudinal(self):
        """Test 3D longitudinal function."""
        # Correct order
        variables = ["u", "v", "w"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
        assert result == tuple(variables)
        
        # Different order
        variables = ["w", "u", "v"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
        assert result == ("u", "v", "w")
        
    def test_check_and_reorder_variables_3d_transverse(self):
        """Test 3D transverse functions for different planes."""
        dims = ["z", "y", "x"]
        
        # ij plane (u, v)
        variables = ["u", "v"]
        result = check_and_reorder_variables_3d(variables, dims, fun="transverse_ij")
        assert result == tuple(variables)
        
        # ik plane (u, w)
        variables = ["u", "w"]
        result = check_and_reorder_variables_3d(variables, dims, fun="transverse_ik")
        assert result == tuple(variables)
        
        # jk plane (v, w)
        variables = ["v", "w"]
        result = check_and_reorder_variables_3d(variables, dims, fun="transverse_jk")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_3d_scalar(self):
        """Test 3D scalar function."""
        variables = ["temperature"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="scalar")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_3d_scalar_scalar(self):
        """Test 3D scalar-scalar function."""
        variables = ["temperature", "salinity"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="scalar_scalar")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_3d_longitudinal_scalar(self):
        """Test 3D longitudinal-scalar function."""
        variables = ["u", "v", "w", "temperature"]
        dims = ["z", "y", "x"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal_scalar")
        assert len(result) == 4
        # Scalar should be last
        assert result[3] == "temperature"
        

        
    def test_check_and_reorder_variables_3d_longitudinal_transverse(self):
        """Test 3D longitudinal-transverse functions."""
        dims = ["z", "y", "x"]
        
        # ij plane
        variables = ["u", "v"]
        result = check_and_reorder_variables_3d(variables, dims, fun="longitudinal_transverse_ij")
        assert result == tuple(variables)
        
    def test_check_and_reorder_variables_3d_error_wrong_count(self):
        """Test error for wrong number of variables in 3D."""
        variables = ["u", "v"]
        dims = ["z", "y", "x"]
        with pytest.raises(ValueError):
            check_and_reorder_variables_3d(variables, dims, fun="longitudinal")
            
    def test_check_and_reorder_variables_3d_error_wrong_dims(self):
        """Test error for wrong dimension order in 3D."""
        variables = ["u", "v", "w"]
        dims = ["x", "y", "z"]  # Wrong order
        with pytest.raises(ValueError):
            check_and_reorder_variables_3d(variables, dims, fun="longitudinal")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
