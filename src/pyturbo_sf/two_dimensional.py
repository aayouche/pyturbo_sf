"""Two-dimensional structure function calculations."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import bottleneck as bn
import gc
from scipy import stats
from scipy.special import jv
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime

from .core import (
    validate_dataset_2d, 
    setup_bootsize_2d, 
    calculate_adaptive_spacings_2d,
    compute_boot_indexes_2d, 
    get_boot_indexes_2d, 
    is_time_dimension
)
from .utils import (
    fast_shift_2d, 
    check_and_reorder_variables_2d, 
    map_variables_by_pattern_2d,
    calculate_time_diff_1d, 
    _calculate_confidence_intervals
)
from .structure_functions import calculate_structure_function_2d
from .binning_tools import (
    _initialize_2d_bins,
    _process_no_bootstrap_2d,
    _create_2d_dataset
)
from .bootstrapping_tools import (
    _run_adaptive_bootstrap_loop_2d,
    _run_adaptive_bootstrap_loop_flux_2d
)
from .isotropy_tools import (
    _initialize_polar_bins_2d,
    _process_no_bootstrap_polar_2d,
    _create_isotropic_dataset
)
from .bessel_tools import (
    _initialize_wavenumbers_2d,
    _initialize_r_bins_2d,
    _initialize_flux_config_2d,
    _process_no_bootstrap_flux_2d,
    _create_flux_dataset_2d,
    _validate_flux_function,
    VALID_FLUX_FUNCTIONS
)

###################################################################Main Function for 2D Pyturbo###########################################################################################

def bin_sf_2d(ds, variables_names, order, bins, bootsize=None, fun='longitudinal', 
            initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
            convergence_eps=0.1, n_jobs=-1, backend='threading',
            conditioning_var=None, conditioning_bins=None, confidence_interval=0.95,
            seed=None):
    """
    Bin structure function with proper volume element weighting.
    
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
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility. Use same seed for conditioned and 
        unconditioned runs to ensure point_counts partition correctly.
    
    Returns
    -------
    xarray.Dataset
        Dataset with binned structure function results.
    
    Note: This function produces 2D output where confidence intervals are computed
    using weighted percentile bootstrap method.
    """
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_2d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape, bootsize)
    
    
    spacings_info, all_spacings = calculate_adaptive_spacings_2d(dims, data_shape, bootsize_dict, 
                                                               bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_2d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING BIN_SF WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    if seed is not None:
        print(f"Using seed {seed} for reproducible bootstrap sampling")
    print("="*60 + "\n")
    
    # Validate bins
    if not isinstance(bins, dict) or not all(dim in bins for dim in dims):
        raise ValueError("'bins' must be a dictionary with all dimensions as keys")
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_2d(
            valid_ds, dims, variables_names, order, fun, bins, time_dims, conditioning_var, conditioning_bins
        )
        
        results = {
            'sf_means': sf_means,
            'sf_stds': sf_stds,
            'point_counts': point_counts,
            'bin_bootstraps': np.zeros_like(sf_means),
            'bin_density': np.zeros_like(sf_means),
            'bin_status': np.ones_like(sf_means, dtype=bool),
            'spacing_values': [],
            'variables_names': variables_names
        }
        
        return _create_2d_dataset(results, bins_config, dims, order, fun,
                                bootstrappable_dims, time_dims, convergence_eps,
                                max_nbootstrap, initial_nbootstrap, backend,
                                confidence_level=confidence_interval)
    
    # Initialize bins
    bins_config = _initialize_2d_bins(bins[dims[1]], bins[dims[0]], dims)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_2d(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, conditioning_var, conditioning_bins, is_2d=True,
        confidence_level=confidence_interval, seed=seed
    )
    
    results['variables_names'] = variables_names
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_binned = _create_2d_dataset(results, bins_config, dims, order, fun,
                                 bootstrappable_dims, time_dims, convergence_eps,
                                 max_nbootstrap, initial_nbootstrap, backend,
                                 confidence_level=confidence_interval)
    
    print("2D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned

#############################################################################################################################################################################

###################################################################Main Function for 2D Isotropic Pyturbo####################################################################

def get_isotropic_sf_2d(ds, variables_names, order=2.0, bins=None, bootsize=None,
                      initial_nbootstrap=100, max_nbootstrap=1000, 
                      step_nbootstrap=100, fun='longitudinal', 
                      n_bins_theta=36, window_size_theta=None, window_size_r=None,
                      convergence_eps=0.1, n_jobs=-1, backend='threading',
                      conditioning_var=None, conditioning_bins=None, confidence_interval=0.95,
                      seed=None):
    """
    Get isotropic (radially binned) structure function with volume element weighting.
    
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
        Number of angular bins. Default is 36.
    window_size_theta : int, optional
        Window size for isotropy error calculation.
    window_size_r : int, optional
        Window size for homogeneity error calculation.
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
            
            ds_single = _get_isotropic_sf_2d_single_bin(
                ds, variables_names, order, bins, bootsize,
                initial_nbootstrap, max_nbootstrap, step_nbootstrap,
                fun, n_bins_theta, window_size_theta, window_size_r,
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
    return _get_isotropic_sf_2d_single_bin(
        ds, variables_names, order, bins, bootsize,
        initial_nbootstrap, max_nbootstrap, step_nbootstrap,
        fun, n_bins_theta, window_size_theta, window_size_r,
        convergence_eps, n_jobs, backend,
        conditioning_var, conditioning_bins, confidence_interval,
        seed=seed
    )


def _get_isotropic_sf_2d_single_bin(ds, variables_names, order, bins, bootsize,
                                    initial_nbootstrap, max_nbootstrap, step_nbootstrap,
                                    fun, n_bins_theta, window_size_theta, window_size_r,
                                    convergence_eps, n_jobs, backend,
                                    conditioning_var, conditioning_bins, confidence_interval,
                                    conditioning_info=None, seed=None):
    """
    Internal function to compute isotropic SF for a single conditioning bin.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_2d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape, bootsize)
        
    spacings_info, all_spacings = calculate_adaptive_spacings_2d(dims, data_shape, bootsize_dict, 
                                                               bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_2d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING ISOTROPIC_SF WITH FUNCTION TYPE: {fun}")
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
    if window_size_r is None:
        window_size_r = max((len(bins['r']) - 1) // 3, 1)
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_polar_2d(
            valid_ds, dims, variables_names, order, fun, bins['r'], n_bins_theta, time_dims, conditioning_var, conditioning_bins
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
        
        return _create_isotropic_dataset(results, bins_config, order, fun,
                                       window_size_theta, window_size_r,
                                       convergence_eps, max_nbootstrap,
                                       initial_nbootstrap, bootstrappable_dims,
                                       backend, variables_names, confidence_interval,
                                       conditioning_info)
    
    # Initialize bins
    bins_config = _initialize_polar_bins_2d(bins['r'], n_bins_theta)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_2d(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, conditioning_var, conditioning_bins, is_2d=False,
        confidence_level=confidence_interval, seed=seed
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_iso = _create_isotropic_dataset(
        results, bins_config, order, fun,
        window_size_theta, window_size_r,
        convergence_eps, max_nbootstrap,
        initial_nbootstrap, bootstrappable_dims,
        backend, variables_names, confidence_interval,
        conditioning_info
    )
    
    print("ISOTROPIC SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_iso

###################################################################################################################################################################################


###################################################################Main Function for 2D Energy Flux####################################################################

def get_energy_flux_2d(ds, variables_names, order=1.0, wavenumbers=None, r_bins=None,
                       bootsize=None, initial_nbootstrap=100, max_nbootstrap=1000,
                       step_nbootstrap=100, fun='advective',
                       n_bins_theta=36, n_r_bins=100, window_size_theta=None, window_size_k=None,
                       convergence_eps=0.1, n_jobs=-1, backend='threading',
                       conditioning_var=None, conditioning_bins=None, confidence_interval=0.95,
                       seed=None):
    """
    Compute spectral energy flux from advective structure function.
    
    Uses the Bessel J₁ transform to compute energy flux:
    
        Π(K) = -K/2 ∫₀^∞ SF̃(r) J₁(Kr) dr
    
    where SF̃(r) is the angle-averaged advective structure function.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with velocity fields.
    variables_names : list
        Names of velocity components to use. For advective SF, typically
        [u, v, u] or similar for δu·δu·δr/|δr|.
    order : float
        Order of the structure function. Default is 1.0 (third-order for advective).
        Note: The advective SF is inherently odd-order.
    wavenumbers : array-like or dict, optional
        If array-like: wavenumber values to evaluate at.
        If dict with 'k': uses those wavenumbers.
        If None: automatically generates logarithmically-spaced wavenumbers.
    r_bins : array-like, optional
        Radial bin edges for angle-averaging. If None, auto-generated.
    bootsize : dict, optional
        Bootstrap block sizes for each dimension.
    initial_nbootstrap : int
        Initial number of bootstrap iterations. Default is 100.
    max_nbootstrap : int
        Maximum number of bootstrap iterations. Default is 1000.
    step_nbootstrap : int
        Bootstrap step size for adaptive convergence. Default is 100.
    fun : str
        Structure function type. Must be 'advective' or 'scalar_scalar'.
        Default is 'advective'.
    n_bins_theta : int
        Number of angular bins for isotropy diagnostics. Default is 36.
    n_r_bins : int
        Number of radial bins if auto-generating r_bins. Default is 100.
    window_size_theta : int, optional
        Window size for isotropy error calculation.
    window_size_k : int, optional
        Window size for homogeneity error calculation.
    convergence_eps : float
        Convergence epsilon for bootstrap. Default is 0.1.
    n_jobs : int
        Number of parallel jobs. Default is -1 (all cores).
    backend : str
        Parallel backend. Default is 'threading'.
    conditioning_var : str, optional
        Name of variable to condition on (e.g., 'vorticity', 'temperature').
    conditioning_bins : array-like, optional
        Bin edges for conditioning variable.
    confidence_interval : float
        Confidence level for intervals. Default is 0.95.
        
    Returns
    -------
    xarray.Dataset
        Dataset containing:
        - energy_flux: Spectral energy flux Π(K) at each wavenumber
        - flux_polar: Angular distribution of flux (theta, k)
        - sf_r: Angle-averaged structure function SF̃(r)
        - error_isotropy: Isotropy error at each wavenumber
        - error_homogeneity: Homogeneity error at subset of wavenumbers
        - std_error: Standard error from bootstrap
        - ci_upper, ci_lower: Confidence interval bounds
        - mask_quality: Quality mask (True for reliable estimates)
        - n_bootstrap: Number of bootstrap iterations per wavenumber
        - point_counts: Number of points per radial bin
        - converged: Convergence status
        
    Notes
    -----
    The energy flux Π(K) represents the rate of energy transfer across 
    wavenumber K. Positive values indicate forward cascade (energy flowing
    to smaller scales), negative values indicate inverse cascade.
    
    This function only accepts 'advective' or 'scalar_scalar' structure 
    function types, as the J₁ Bessel transform is only physically meaningful
    for these quantities.
    
    References
    ----------
    Derived from Plancherel theorem relating spectral energy flux to 
    real-space advective structure function.
    """
    # Validate function type
    _validate_flux_function(fun)
    
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
            
            ds_single = _get_energy_flux_2d_single_bin(
                ds, variables_names, order, wavenumbers, r_bins, bootsize,
                initial_nbootstrap, max_nbootstrap, step_nbootstrap,
                fun, n_bins_theta, n_r_bins, window_size_theta, window_size_k,
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
    return _get_energy_flux_2d_single_bin(
        ds, variables_names, order, wavenumbers, r_bins, bootsize,
        initial_nbootstrap, max_nbootstrap, step_nbootstrap,
        fun, n_bins_theta, n_r_bins, window_size_theta, window_size_k,
        convergence_eps, n_jobs, backend,
        conditioning_var, conditioning_bins, confidence_interval,
        seed=seed
    )


def _get_energy_flux_2d_single_bin(ds, variables_names, order, wavenumbers, r_bins, bootsize,
                                    initial_nbootstrap, max_nbootstrap, step_nbootstrap,
                                    fun, n_bins_theta, n_r_bins, window_size_theta, window_size_k,
                                    convergence_eps, n_jobs, backend,
                                    conditioning_var, conditioning_bins, confidence_interval,
                                    conditioning_info=None, seed=None):
    """
    Internal function to compute energy flux for a single conditioning bin.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """
    # Validate function type
    _validate_flux_function(fun)
    
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_2d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_2d(dims, data_shape, bootsize)
    
    spacings_info, all_spacings = calculate_adaptive_spacings_2d(dims, data_shape, bootsize_dict,
                                                                  bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_2d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING ENERGY FLUX COMPUTATION")
    print(f"Function type: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print(f"Confidence level: {confidence_interval}")
    print(f"Formula: Π(K) = -K/2 ∫ SF̃(r) J₁(Kr) dr")
    if conditioning_var:
        print(f"Conditioning: {conditioning_var} in {conditioning_bins}")
    print("="*60 + "\n")
    
    # Setup wavenumbers
    wavenumbers_config = _initialize_wavenumbers_2d(wavenumbers, ds, dims)
    k = wavenumbers_config['k']
    
    # Setup radial bins
    r_config = _initialize_r_bins_2d(r_bins, ds, dims, n_r_bins)
    
    # Default window sizes
    if window_size_theta is None:
        window_size_theta = max(n_bins_theta // 3, 1)
    if window_size_k is None:
        window_size_k = max(len(k) // 3, 1)
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        energy_flux, flux_stds, point_counts, flux_theta_k, config = _process_no_bootstrap_flux_2d(
            valid_ds, dims, variables_names, order, fun, 
            k, r_config, n_bins_theta, time_dims, conditioning_var, conditioning_bins
        )
        
        results = {
            'energy_flux': energy_flux,
            'flux_stds': flux_stds,
            'point_counts': point_counts,
            'flux_theta_k': flux_theta_k,
            'sf_r': config['sf_r'],
            'bin_bootstraps': np.zeros_like(energy_flux),
            'bin_density': np.zeros_like(energy_flux),
            'bin_status': np.ones_like(energy_flux, dtype=bool),
            'spacing_values': []
        }
        
        return _create_flux_dataset_2d(results, config, order, fun,
                                        window_size_theta, window_size_k,
                                        convergence_eps, max_nbootstrap,
                                        initial_nbootstrap, bootstrappable_dims,
                                        backend, variables_names, confidence_interval,
                                        conditioning_info)
    
    # Initialize configuration
    config = _initialize_flux_config_2d(k, r_config, n_bins_theta)
    
    # Check if bootstrap function is available
    if _run_adaptive_bootstrap_loop_flux_2d is None:
        raise NotImplementedError(
            "The bootstrapping_tools module needs to be updated with "
            "_run_adaptive_bootstrap_loop_flux_2d function. "
            "Please see the migration notes in bessel_tools.py. "
            "For now, use num_bootstrappable=0 (no bootstrap) by ensuring "
            "your data dimensions don't support bootstrapping, or use "
            "the structure function directly with _process_no_bootstrap_flux_2d."
        )
    
    # Run adaptive bootstrap loop for energy flux
    results = _run_adaptive_bootstrap_loop_flux_2d(
        valid_ds, dims, variables_names, order, fun,
        config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, conditioning_var, conditioning_bins,
        confidence_level=confidence_interval, seed=seed
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_flux = _create_flux_dataset_2d(
        results, config, order, fun,
        window_size_theta, window_size_k,
        convergence_eps, max_nbootstrap,
        initial_nbootstrap, bootstrappable_dims,
        backend, variables_names, confidence_interval,
        conditioning_info
    )
    
    print("ENERGY FLUX COMPUTATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_flux


#############################################################################################################################
