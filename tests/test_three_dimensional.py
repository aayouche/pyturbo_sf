"""
Tests for three_dimensional.py module functionality.
"""

import pytest
import numpy as np
import xarray as xr

from pyturbo_sf.three_dimensional import (
    calc_longitudinal_3d, calc_transverse_ij, calc_transverse_ik, calc_transverse_jk,
    calc_scalar_3d, calc_scalar_scalar_3d, calc_longitudinal_scalar_3d,
    calc_transverse_ij_scalar, calc_transverse_ik_scalar, calc_transverse_jk_scalar,
    calc_longitudinal_transverse_ij, calc_longitudinal_transverse_ik, calc_longitudinal_transverse_jk,
    calculate_structure_function_3d,
    monte_carlo_simulation_3d,
    bin_sf_3d,
    get_isotropic_sf_3d
)


@pytest.fixture
def dataset_3d():
    """Create a 3D dataset for testing."""
    # Create coordinates (small grid for faster tests)
    x = np.linspace(0, 10, 8)
    y = np.linspace(0, 10, 6)
    z = np.linspace(0, 10, 5)
    
    # Create a meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create velocity components and scalars
    u = np.sin(X) * np.cos(Y) * np.sin(Z)
    v = np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.sin(X) * np.sin(Y) * np.sin(Z)
    scalar1 = np.sin(X + Y + Z)
    scalar2 = np.cos(X - Y + Z)
    
    ds = xr.Dataset(
        data_vars={
            "u": (("z", "y", "x"), u),
            "v": (("z", "y", "x"), v),
            "w": (("z", "y", "x"), w),
            "scalar1": (("z", "y", "x"), scalar1),
            "scalar2": (("z", "y", "x"), scalar2)
        },
        coords={
            "x": (["z", "y", "x"], X),
            "y": (["z", "y", "x"], Y),
            "z": (["z", "y", "x"], Z),
        }
    )
    return ds


class TestCalcFunctions:
    
    def test_calc_longitudinal_3d(self, dataset_3d):
        """Test longitudinal structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate longitudinal structure function
        results, dx, dy, dz = calc_longitudinal_3d(
            subset=subset,
            variables_names=["u", "v", "w"],
            order=2,
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_transverse_ij(self, dataset_3d):
        """Test transverse (xy-plane) structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate transverse structure function in xy-plane
        results, dx, dy, dz = calc_transverse_ij(
            subset=subset,
            variables_names=["u", "v"],
            order=2,
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_transverse_ik(self, dataset_3d):
        """Test transverse (xz-plane) structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate transverse structure function in xz-plane
        results, dx, dy, dz = calc_transverse_ik(
            subset=subset,
            variables_names=["u", "w"],
            order=2,
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_transverse_jk(self, dataset_3d):
        """Test transverse (yz-plane) structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate transverse structure function in yz-plane
        results, dx, dy, dz = calc_transverse_jk(
            subset=subset,
            variables_names=["v", "w"],
            order=2,
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_scalar_3d(self, dataset_3d):
        """Test scalar structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.scalar1.shape
        dims = ["z", "y", "x"]
        
        # Calculate scalar structure function
        results, dx, dy, dz = calc_scalar_3d(
            subset=subset,
            variables_names=["scalar1"],
            order=2,
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_scalar_scalar_3d(self, dataset_3d):
        """Test scalar-scalar structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.scalar1.shape
        dims = ["z", "y", "x"]
        
        # Calculate scalar-scalar structure function
        results, dx, dy, dz = calc_scalar_scalar_3d(
            subset=subset,
            variables_names=["scalar1", "scalar2"],
            order=(2, 1),  # Order 2 for first scalar, 1 for second
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_longitudinal_scalar_3d(self, dataset_3d):
        """Test longitudinal-scalar structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate longitudinal-scalar structure function
        results, dx, dy, dz = calc_longitudinal_scalar_3d(
            subset=subset,
            variables_names=["u", "v", "w", "scalar1"],
            order=(2, 1),  # Order 2 for longitudinal, 1 for scalar
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_transverse_ij_scalar(self, dataset_3d):
        """Test transverse-scalar (xy-plane) structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate transverse-scalar structure function in xy-plane
        results, dx, dy, dz = calc_transverse_ij_scalar(
            subset=subset,
            variables_names=["u", "v", "scalar1"],
            order=(2, 1),  # Order 2 for transverse, 1 for scalar
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calc_longitudinal_transverse_ij(self, dataset_3d):
        """Test longitudinal-transverse (xy-plane) structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        nz, ny, nx = subset.u.shape
        dims = ["z", "y", "x"]
        
        # Calculate longitudinal-transverse structure function in xy-plane
        results, dx, dy, dz = calc_longitudinal_transverse_ij(
            subset=subset,
            variables_names=["u", "v"],
            order=(2, 1),  # Order 2 for longitudinal, 1 for transverse
            dims=dims,
            nz=nz, 
            ny=ny,
            nx=nx
        )
        
        # Check shapes
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0


class TestCalculateStructureFunction:
    
    def test_calculate_structure_function_3d(self, dataset_3d):
        """Test main structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Test with longitudinal function
        results, dx, dy, dz = calculate_structure_function_3d(
            ds=subset,
            dims=["z", "y", "x"],
            variables_names=["u", "v", "w"],
            order=2,
            fun="longitudinal"
        )
        
        # Check shapes
        nz, ny, nx = subset.u.shape
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calculate_structure_function_3d_scalar(self, dataset_3d):
        """Test structure function calculation for scalar field in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Test with scalar function
        results, dx, dy, dz = calculate_structure_function_3d(
            ds=subset,
            dims=["z", "y", "x"],
            variables_names=["scalar1"],
            order=2,
            fun="scalar"
        )
        
        # Check shapes
        nz, ny, nx = subset.scalar1.shape
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calculate_structure_function_3d_transverse(self, dataset_3d):
        """Test transverse structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Test with transverse_ij function (xy-plane)
        results, dx, dy, dz = calculate_structure_function_3d(
            ds=subset,
            dims=["z", "y", "x"],
            variables_names=["u", "v"],
            order=2,
            fun="transverse_ij"
        )
        
        # Check shapes
        nz, ny, nx = subset.u.shape
        assert results.shape == (nz * ny * nx,)
        assert dx.shape == (nz * ny * nx,)
        assert dy.shape == (nz * ny * nx,)
        assert dz.shape == (nz * ny * nx,)
        
        # Check that some values are finite
        assert np.sum(np.isfinite(results)) > 0
        assert np.sum(np.isfinite(dx)) > 0
        assert np.sum(np.isfinite(dy)) > 0
        assert np.sum(np.isfinite(dz)) > 0
        
    def test_calculate_structure_function_3d_error(self, dataset_3d):
        """Test error cases for structure function calculation in 3D."""
        # Get a small subset for faster testing
        subset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Test with unsupported function type
        with pytest.raises(ValueError):
            calculate_structure_function_3d(
                ds=subset,
                dims=["z", "y", "x"],
                variables_names=["u", "v", "w"],
                order=2,
                fun="unsupported_type"
            )
            
        # Test with non-existent variable
        with pytest.raises(ValueError):
            calculate_structure_function_3d(
                ds=subset,
                dims=["z", "y", "x"],
                variables_names=["nonexistent"],
                order=2,
                fun="scalar"
            )


class TestMonteCarloSimulation:
    
    def test_monte_carlo_simulation_3d(self, dataset_3d):
        """Test Monte Carlo simulation for 3D structure functions."""
        # Use a very small subset for faster tests
        dataset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Setup parameters
        dims = ["z", "y", "x"]
        bootsize = {"z": 2, "y": 2, "x": 2}
        nbootstrap = 2  # Small number for faster testing
        
        # Calculate adaptive spacings (needed for Monte Carlo simulation)
        from pyturbo_sf.core import (
            setup_bootsize_3d,
            calculate_adaptive_spacings_3d,
            compute_boot_indexes_3d
        )
        
        data_shape = dict(dataset.sizes)
        bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
        
        spacings_info, all_spacings = calculate_adaptive_spacings_3d(
            dims, data_shape, bootsize_dict, bootstrappable_dims, num_bootstrappable
        )
        
        boot_indexes = compute_boot_indexes_3d(
            dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims
        )
        
        # Run Monte Carlo simulation with minimal iterations
        results, dx_vals, dy_vals, dz_vals = monte_carlo_simulation_3d(
            ds=dataset,
            dims=dims,
            variables_names=["u", "v", "w"],
            order=2,
            nbootstrap=nbootstrap,
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims,
            fun="longitudinal",
            spacing=1,
            n_jobs=1  # Sequential processing for testing
        )
        
        # Check results
        assert len(results) == nbootstrap
        assert len(dx_vals) == nbootstrap
        assert len(dy_vals) == nbootstrap
        assert len(dz_vals) == nbootstrap
        
        # At least some values should be finite
        assert np.any(np.isfinite(results[0]))
        assert np.any(np.isfinite(dx_vals[0]))
        assert np.any(np.isfinite(dy_vals[0]))
        assert np.any(np.isfinite(dz_vals[0]))


class TestBinSF:
    
    def test_bin_sf_3d(self, dataset_3d):
        """Test the binning function for 3D structure functions."""
        # Use a very small subset for faster tests
        dataset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Setup bin edges
        bins_x = np.linspace(0, 3, 4) + 1.0e-6
        bins_y = np.linspace(0, 3, 4) + 1.0e-6
        bins_z = np.linspace(0, 3, 4) + 1.0e-6
        bins = {"x": bins_x, "y": bins_y, "z": bins_z}
        
        # Test with minimal number of bootstraps for speed
        binned_ds = bin_sf_3d(
            ds=dataset,
            variables_names=["u", "v", "w"],
            order=2,
            bins=bins,
            bootsize={"x": 2, "y": 2, "z": 2},
            fun="longitudinal",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            convergence_eps=0.5,  # Large value for faster convergence
            n_jobs=1  # Sequential processing for testing
        )
        
        # Verify the result has expected structure
        assert isinstance(binned_ds, xr.Dataset)
        assert "sf" in binned_ds.data_vars
        assert "sf_std" in binned_ds.data_vars
        assert "x" in binned_ds.coords
        assert "y" in binned_ds.coords
        assert "z" in binned_ds.coords
        
        # Check that binned structure function contains valid values
        assert not np.all(np.isnan(binned_ds.sf))
        
        # Check that attributes are correctly set
        assert "bin_type_x" in binned_ds.attrs
        assert "bin_type_y" in binned_ds.attrs
        assert "bin_type_z" in binned_ds.attrs
        assert "order" in binned_ds.attrs
        assert "function_type" in binned_ds.attrs
        assert "variables" in binned_ds.attrs
        
        # Test with a scalar function
        binned_ds = bin_sf_3d(
            ds=dataset,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize={"x": 2, "y": 2, "z": 2},
            fun="scalar",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        # Verify scalar results
        assert isinstance(binned_ds, xr.Dataset)
        assert "sf" in binned_ds.data_vars
        assert not np.all(np.isnan(binned_ds.sf))


class TestIsotropicSF:
    
    def test_get_isotropic_sf_3d(self, dataset_3d):
        """Test the isotropic structure function calculation for 3D."""
        # Use a very small subset for faster tests
        dataset = dataset_3d.isel(x=slice(0, 3), y=slice(0, 3), z=slice(0, 2))
        
        # Setup radial bin edges
        r_bins = np.linspace(0, 3, 4) + 1.0e-6
        bins = {"r": r_bins}
        
        # Test with minimal number of bootstraps and angular bins for speed
        isotropic_ds = get_isotropic_sf_3d(
            ds=dataset,
            variables_names=["u", "v", "w"],
            order=2,
            bins=bins,
            bootsize={"x": 2, "y": 2, "z": 2},
            fun="longitudinal",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            n_bins_theta=6,  # Small number of angular bins
            n_bins_phi=4,    # Small number of angular bins
            window_size_theta=2,
            window_size_phi=2,
            window_size_r=2,
            convergence_eps=0.5,  # Large value for faster convergence
            n_jobs=1  # Sequential processing for testing
        )
        
        # Verify the result has expected structure
        assert isinstance(isotropic_ds, xr.Dataset)
        assert "sf" in isotropic_ds.data_vars
        assert "sf_spherical" in isotropic_ds.data_vars
        assert "r" in isotropic_ds.coords
        assert "theta" in isotropic_ds.coords
        assert "phi" in isotropic_ds.coords
        
        # Check dimensions of spherical results
        assert isotropic_ds.sf_spherical.dims == ('phi', 'theta', 'r')
        assert len(isotropic_ds.r) == len(r_bins) - 1
        assert len(isotropic_ds.theta) == 6
        assert len(isotropic_ds.phi) == 4
        
        # Check that isotropic structure function contains valid values
        assert not np.all(np.isnan(isotropic_ds.sf))
        
        # Check that attributes are correctly set
        assert "bin_type" in isotropic_ds.attrs
        assert "order" in isotropic_ds.attrs
        assert "function_type" in isotropic_ds.attrs
        assert "variables" in isotropic_ds.attrs
        assert "window_size_theta" in isotropic_ds.attrs
        assert "window_size_phi" in isotropic_ds.attrs
        assert "window_size_r" in isotropic_ds.attrs
        
        # Test with a scalar function
        isotropic_ds = get_isotropic_sf_3d(
            ds=dataset,
            variables_names=["scalar1"],
            order=2,
            bins=bins,
            bootsize={"x": 2, "y": 2, "z": 2},
            fun="scalar",
            initial_nbootstrap=2,
            max_nbootstrap=3,
            step_nbootstrap=1,
            n_bins_theta=6,
            n_bins_phi=4,
            window_size_theta=2,
            window_size_phi=2,
            window_size_r=2,
            convergence_eps=0.5,
            n_jobs=1
        )
        
        # Verify scalar results
        assert isinstance(isotropic_ds, xr.Dataset)
        assert "sf" in isotropic_ds.data_vars
        assert not np.all(np.isnan(isotropic_ds.sf))


if __name__ == "__main__":
    pytest.main(["-v", "test_three_dimensional.py"])
