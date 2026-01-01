"""Three-dimensional structure function calculations.

Note: 3D energy flux (Bessel) decomposition is not yet implemented.
Only 2D energy flux is available via two_dimensional.get_energy_flux_2d().
"""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import bottleneck as bn
import gc
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

from .core import (
    validate_dataset_3d, 
    setup_bootsize_3d, 
    calculate_adaptive_spacings_3d,
    compute_boot_indexes_3d, 
    get_boot_indexes_3d, 
    is_time_dimension
)
from .utils import (
    fast_shift_3d, 
    check_and_reorder_variables_3d, 
    map_variables_by_pattern_3d,
    calculate_time_diff_1d, 
    _calculate_confidence_intervals
)
from .structure_functions import calculate_structure_function_3d
from .binning_tools import (
    _initialize_3d_bins,
    _process_no_bootstrap_3d,
    _create_3d_dataset
)
from .bootstrapping_tools import (
    _run_adaptive_bootstrap_loop_3d,
)
from .isotropy_tools import (
    _initialize_spherical_bins_3d,
    _process_no_bootstrap_spherical_3d,
    _create_spherical_dataset
)

#####################################3D Binning###############################################################

def bin_sf_3d(ds, variables_names, order, bins, bootsize=None, fun='longitudinal', 
            initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
            convergence_eps=0.1, n_jobs=-1, backend='threading',
            conditioning_var=None, conditioning_bins=None, confidence_interval=0.95, seed=None):
    """
    Bin 3D structure function with proper volume element weighting.
    
    Uses the same modular structure as 2D binning with helper functions.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with velocity/scalar fields.
    variables_names : list
        Names of variables to use.
    order : float or tuple
        Order of the structure function.
    bins : dict
        Dictionary with dimensions as keys and bin edges as values.
    bootsize : dict, optional
        Bootstrap block sizes for each dimension.
    fun : str
        Structure function type. Default is 'longitudinal'.
    initial_nbootstrap : int
        Initial number of bootstrap iterations. Default is 100.
    max_nbootstrap : int
        Maximum number of bootstrap iterations. Default is 1000.
    step_nbootstrap : int
        Bootstrap step size for adaptive convergence. Default is 100.
    convergence_eps : float
        Convergence epsilon for bootstrap. Default is 0.1.
    n_jobs : int
        Number of parallel jobs. Default is -1 (all cores).
    backend : str
        Parallel backend. Default is 'threading'.
    conditioning_var : str, optional
        Name of variable to condition on.
    conditioning_bins : array-like, optional
        Bin edges for conditioning variable.
    confidence_interval : float
        Confidence level for intervals (0-1). Default is 0.95.
    seed : int, optional
        Random seed for reproducibility. Use same seed for conditioned and 
        unconditioned runs to ensure point_counts partition correctly.
    
    Returns
    -------
    xarray.Dataset
        Dataset with binned structure function results.
    
    Note: This function produces 3D output where confidence intervals are computed
    using the standard normal approximation. For percentile-based CIs, use
    get_isotropic_sf_3d which produces 1D radial output.
    """
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_3d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
    spacings_info, all_spacings = calculate_adaptive_spacings_3d(dims, data_shape, bootsize_dict, 
                                                               bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_3d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING BIN_SF_3D WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    if seed is not None:
        print(f"Using seed {seed} for reproducible bootstrap sampling")
    print("="*60 + "\n")
    
    # Validate bins
    if not isinstance(bins, dict) or not all(dim in bins for dim in dims):
        raise ValueError("'bins' must be a dictionary with all dimensions as keys")
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_3d(
            valid_ds, dims, variables_names, order, fun, bins, time_dims, conditioning_var, conditioning_bins
        )
        
        results = {
            'sf_means': sf_means,
            'sf_stds': sf_stds,
            'point_counts': point_counts,
            'bin_bootstraps': np.zeros_like(sf_means),
            'bin_density': np.zeros_like(sf_means),
            'bin_status': np.ones_like(sf_means, dtype=bool),
            'spacing_values': []
        }
        
        return _create_3d_dataset(results, bins_config, dims, order, fun,
                                bootstrappable_dims, time_dims, convergence_eps,
                                max_nbootstrap, initial_nbootstrap, backend, variables_names,
                                confidence_level=confidence_interval)
    
    # Initialize bins
    bins_config = _initialize_3d_bins(bins[dims[2]], bins[dims[1]], bins[dims[0]], dims)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_3d(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, is_3d=True, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins,
        seed=seed, confidence_level=confidence_interval
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_binned = _create_3d_dataset(results, bins_config, dims, order, fun,
                                 bootstrappable_dims, time_dims, convergence_eps,
                                 max_nbootstrap, initial_nbootstrap, backend, variables_names,
                                 confidence_level=confidence_interval)
    
    print("3D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned


def get_isotropic_sf_3d(ds, variables_names, order=2.0, bins=None, bootsize=None,
                       initial_nbootstrap=100, max_nbootstrap=1000, 
                       step_nbootstrap=100, fun='longitudinal', 
                       n_bins_theta=36, n_bins_phi=18, 
                       window_size_theta=None, window_size_phi=None, window_size_r=None,
                       convergence_eps=0.1, n_jobs=-1, backend='threading',
                       conditioning_var=None, conditioning_bins=None, confidence_interval=0.95,
                       seed=None):
    """
    Get isotropic (spherically binned) structure function with volume element weighting.
    
    Uses the same modular structure as 2D isotropic binning with helper functions.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with velocity fields.
    variables_names : list
        Names of velocity components to use.
    order : float
        Order of the structure function. Default is 2.0.
    bins : dict
        Dictionary with 'r' key for radial bin edges.
    bootsize : dict, optional
        Bootstrap block sizes for each dimension.
    initial_nbootstrap : int
        Initial number of bootstrap iterations. Default is 100.
    max_nbootstrap : int
        Maximum number of bootstrap iterations. Default is 1000.
    step_nbootstrap : int
        Bootstrap step size for adaptive convergence. Default is 100.
    fun : str
        Structure function type. Default is 'longitudinal'.
    n_bins_theta : int
        Number of azimuthal angle bins. Default is 36.
    n_bins_phi : int
        Number of polar angle bins. Default is 18.
    window_size_theta : int, optional
        Window size for azimuthal isotropy error calculation.
    window_size_phi : int, optional
        Window size for polar isotropy error calculation.
    window_size_r : int, optional
        Window size for radial homogeneity error calculation.
    convergence_eps : float
        Convergence epsilon for bootstrap. Default is 0.1.
    n_jobs : int
        Number of parallel jobs. Default is -1 (all cores).
    backend : str
        Parallel backend. Default is 'threading'.
    conditioning_var : str, optional
        Name of variable to condition on (e.g., 'vorticity', 'temperature').
    conditioning_bins : array-like, optional
        Bin edges for conditioning variable. Can be:
        - [T_lo, T_hi]: Single bin
        - np.linspace(...) or np.logspace(...): Multiple bins (N+1 edges for N bins)
    confidence_interval : float
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility. Use same seed for conditioned and 
        unconditioned runs to ensure point_counts partition correctly.
        
    Returns
    -------
    xarray.Dataset
        Dataset with isotropic structure function results.
        If conditioning_bins has >2 elements, output has 'cond_bin' dimension.
    """
    # Check for multiple conditioning bins
    if conditioning_bins is not None and len(conditioning_bins) > 2:
        # Multiple bins case - loop and concatenate
        conditioning_bins = np.asarray(conditioning_bins)
        n_cond_bins = len(conditioning_bins) - 1
        
        print(f"\n{'='*60}")
        print(f"MULTI-BIN CONDITIONING: {n_cond_bins} bins for {conditioning_var}")
        print(f"Bin edges: {conditioning_bins}")
        if seed is not None:
            print(f"Using seed {seed} for reproducible bootstrap sampling")
        print(f"{'='*60}\n")
        
        datasets = []
        for i in range(n_cond_bins):
            single_bin = [conditioning_bins[i], conditioning_bins[i+1]]
            print(f"\n--- Processing conditioning bin {i+1}/{n_cond_bins}: [{single_bin[0]:.4g}, {single_bin[1]:.4g}) ---")
            
            ds_single = _get_isotropic_sf_3d_single_bin(
                ds, variables_names, order, bins, bootsize,
                initial_nbootstrap, max_nbootstrap, step_nbootstrap,
                fun, n_bins_theta, n_bins_phi, window_size_theta, window_size_phi, window_size_r,
                convergence_eps, n_jobs, backend,
                conditioning_var, single_bin, confidence_interval,
                conditioning_info={'var_name': conditioning_var, 'bins': conditioning_bins, 'bin_idx': i},
                seed=seed
            )
            datasets.append(ds_single)
        
        # Concatenate along cond_bin dimension
        ds_combined = xr.concat(datasets, dim='cond_bin')
        
        # Update cond_bin coordinate to be the bin centers
        cond_bin_centers = 0.5 * (conditioning_bins[:-1] + conditioning_bins[1:])
        ds_combined = ds_combined.assign_coords(cond_bin=cond_bin_centers)
        
        # Add conditioning metadata
        ds_combined.attrs['conditioning_variable'] = conditioning_var
        ds_combined.attrs['conditioning_bin_edges'] = list(conditioning_bins)
        
        print(f"\n{'='*60}")
        print(f"MULTI-BIN CONDITIONING COMPLETE")
        print(f"Output dimensions: {dict(ds_combined.sizes)}")
        print(f"{'='*60}\n")
        
        return ds_combined
    
    # Single bin or no conditioning - use existing logic
    return _get_isotropic_sf_3d_single_bin(
        ds, variables_names, order, bins, bootsize,
        initial_nbootstrap, max_nbootstrap, step_nbootstrap,
        fun, n_bins_theta, n_bins_phi, window_size_theta, window_size_phi, window_size_r,
        convergence_eps, n_jobs, backend,
        conditioning_var, conditioning_bins, confidence_interval,
        seed=seed
    )


def _get_isotropic_sf_3d_single_bin(ds, variables_names, order, bins, bootsize,
                                     initial_nbootstrap, max_nbootstrap, step_nbootstrap,
                                     fun, n_bins_theta, n_bins_phi, window_size_theta, window_size_phi, window_size_r,
                                     convergence_eps, n_jobs, backend,
                                     conditioning_var, conditioning_bins, confidence_interval,
                                     conditioning_info=None, seed=None):
    """
    Internal function to compute 3D isotropic SF for a single conditioning bin.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_3d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
    spacings_info, all_spacings = calculate_adaptive_spacings_3d(dims, data_shape, bootsize_dict, 
                                                               bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_3d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING ISOTROPIC_SF_3D WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print(f"Confidence level: {confidence_interval}")
    if conditioning_var:
        print(f"Conditioning: {conditioning_var} in {conditioning_bins}")
    print("="*60 + "\n")
    
    # Validate bins
    if bins is None or 'r' not in bins:
        raise ValueError("'bins' must be a dictionary with 'r' as key")
    
    # Default window sizes
    if window_size_theta is None:
        window_size_theta = max(n_bins_theta // 3, 1)
    if window_size_phi is None:
        window_size_phi = max(n_bins_phi // 3, 1)
    if window_size_r is None:
        window_size_r = max((len(bins['r']) - 1) // 3, 1)
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_spherical_3d(
            valid_ds, dims, variables_names, order, fun, bins['r'], n_bins_theta, n_bins_phi, time_dims,
            conditioning_var, conditioning_bins
        )
        
        results = {
            'sf_means': sf_means,
            'sf_stds': sf_stds,
            'point_counts': point_counts,
            'sfr': sfr,
            'sfr_counts': sfr_counts,
            'bin_bootstraps': np.zeros_like(sf_means),
            'bin_density': np.zeros_like(sf_means),
            'bin_status': np.ones_like(sf_means, dtype=bool),
            'spacing_values': []
        }
        
        return _create_spherical_dataset(results, bins_config, order, fun,
                                       window_size_theta, window_size_phi, window_size_r,
                                       convergence_eps, max_nbootstrap,
                                       initial_nbootstrap, bootstrappable_dims,
                                       backend, variables_names, confidence_interval,
                                       conditioning_info)
    
    # Initialize bins
    bins_config = _initialize_spherical_bins_3d(bins['r'], n_bins_theta, n_bins_phi)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_3d(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, is_3d=False, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins,
        confidence_level=confidence_interval, seed=seed
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_iso = _create_spherical_dataset(
        results, bins_config, order, fun,
        window_size_theta, window_size_phi, window_size_r,
        convergence_eps, max_nbootstrap,
        initial_nbootstrap, bootstrappable_dims,
        backend, variables_names, confidence_interval,
        conditioning_info
    )
    
    print("ISOTROPIC SF 3D COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_iso

##############################################################################################################
