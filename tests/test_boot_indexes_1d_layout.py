"""Regression tests for the (bootsize, num_windows) layout of 1D bootstrap indexes.

The 1D indexes must match the 2D/3D convention, since structure_functions.py
selects a bootstrap sample with `indexes[dim][:, nb]`.
"""
import numpy as np
import pytest

from pyturbo_sf.core import (
    compute_boot_indexes_1d,
    compute_boot_indexes_2d,
    get_boot_indexes_1d,
)


@pytest.mark.parametrize("spacing", [1, 2, 3, 4])
def test_1d_index_shape_is_bootsize_by_nwindows(spacing):
    n, bootsize = 64, 8
    idx = compute_boot_indexes_1d(
        "x", {"x": n}, {"x": bootsize}, [spacing], 1
    )[spacing]["x"]
    expected_windows = n - bootsize * spacing + 1
    assert idx.shape == (bootsize, expected_windows)


@pytest.mark.parametrize("spacing", [1, 2, 3, 4])
def test_1d_column_is_a_single_strided_window(spacing):
    n, bootsize = 64, 8
    idx = compute_boot_indexes_1d(
        "x", {"x": n}, {"x": bootsize}, [spacing], 1
    )[spacing]["x"]
    for nb in (0, 1, idx.shape[1] - 1):
        col = idx[:, nb]
        assert len(col) == bootsize
        np.testing.assert_array_equal(
            col, nb + spacing * np.arange(bootsize)
        )
        assert col.max() < n


@pytest.mark.parametrize("spacing", [1, 2, 3, 4])
def test_1d_matches_2d_convention(spacing):
    n, bootsize = 64, 8
    a = compute_boot_indexes_1d(
        "x", {"x": n}, {"x": bootsize}, [spacing], 1
    )[spacing]["x"]
    b = compute_boot_indexes_2d(
        ["y", "x"], {"x": n, "y": n}, {"x": bootsize, "y": bootsize},
        [spacing], ["y", "x"],
    )[spacing]["x"]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("spacing", [1, 2, 3])
def test_on_the_fly_path_matches_precomputed(spacing):
    """get_boot_indexes_1d must agree with compute_boot_indexes_1d."""
    n, bootsize = 64, 8
    pre = compute_boot_indexes_1d(
        "x", {"x": n}, {"x": bootsize}, [spacing], 1
    )[spacing]["x"]
    fly = get_boot_indexes_1d(
        "x", {"x": n}, {"x": bootsize}, [spacing], {}, 1, spacing
    )["x"]
    np.testing.assert_array_equal(pre, fly)
