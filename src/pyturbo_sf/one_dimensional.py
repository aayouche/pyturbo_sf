"""One-dimensional structure function calculations."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import gc
from .core import (validate_dataset_1d, setup_bootsize_1d, calculate_adaptive_spacings_1d, 
                  compute_boot_indexes_1d, get_boot_indexes_1d)
from .utils import (fast_shift_1d, calculate_time_diff_1d, _calculate_confidence_intervals)
from .structure_functions import calculate_structure_function_1d
from .binning_tools import (
    _initialize_1d_bins,
    _process_no_bootstrap_1d,
    _create_1d_dataset
)
from .bootstrapping_tools import _run_adaptive_bootstrap_loop_1d

#####################################################################################################################

def bin_sf_1d(ds, variables_names, order, bins, bootsize=None, fun='scalar', 
             initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
             convergence_eps=0.1, n_jobs=-1, backend='threading',
             conditioning_var=None, conditioning_bins=None, confidence_interval=0.95,
             seed=None):
    """
    Bin structure function results with improved weighted statistics and memory efficiency.
    
    Parameters
    -----------
    ds : xarray.Dataset
        Dataset containing scalar fields
    variables_names : list
        List of variable names to use, depends on function type
    order : float or tuple
        Order(s) of the structure function
    bins : dict
        Dictionary with dimension as key and bin edges as values
    bootsize : dict or int, optional
        Bootsize for the dimension
    fun : str, optional
        Type of structure function: ['scalar', 'scalar_scalar']
    initial_nbootstrap : int, optional
        Initial number of bootstrap samples
    max_nbootstrap : int, optional
        Maximum number of bootstrap samples
    step_nbootstrap : int, optional
        Step size for increasing bootstrap samples
    convergence_eps : float, optional
        Convergence threshold for bin standard deviation
    n_jobs : int, optional
        Number of jobs for parallel processing
    backend : str, optional
        Backend for joblib: 'threading', 'multiprocessing', or 'loky'. Default is 'threading'.
    mask : str, optional
        Name of mask variable in dataset
    conditioning_bins : tuple, optional
        Conditions for masking. If dict with 'array' and 'shifted' keys,
        creates separate indicators I_α and I_β. If list, applies same
        condition to both.
    confidence_interval : float, optional
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility. Use same seed for conditioned and 
        unconditioned runs to ensure point_counts partition correctly.
        
    Returns
    --------
    xarray.Dataset
        Dataset with binned structure function results
    """
    # Validate dataset
    dim_name, data_shape = validate_dataset_1d(ds)
    
    # Setup bootsize
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_1d(dim_name, data_shape, bootsize)
    
    # Calculate spacings
    spacings_info, all_spacings = calculate_adaptive_spacings_1d(dim_name, data_shape, bootsize_dict, num_bootstrappable)
    
    # Compute boot indexes
    boot_indexes = compute_boot_indexes_1d(dim_name, data_shape, bootsize_dict, all_spacings, num_bootstrappable)
    
    print("\n" + "="*60)
    print(f"STARTING BIN_SF WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print(f"Bootstrap parameters: initial={initial_nbootstrap}, max={max_nbootstrap}, step={step_nbootstrap}")
    print(f"Convergence threshold: {convergence_eps}")
    print(f"Confidence level: {confidence_interval}")
    print(f"Bootstrappable dimensions: {bootstrappable_dims} (count: {num_bootstrappable})")
    print("Using volume element weighting: |dx|")
    print("="*60 + "\n")
    
    # Validate bins
    if not isinstance(bins, dict):
        raise ValueError("'bins' must be a dictionary with dimension as key and bin edges as values")
    
    if dim_name not in bins:
        raise ValueError(f"Bins must be provided for dimension '{dim_name}'")
    
    # Initialize bins
    bins_config = _initialize_1d_bins(bins[dim_name], dim_name)
    
    # Special case: no bootstrappable dimensions
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts = _process_no_bootstrap_1d(
            ds, dim_name, variables_names, order, fun, bins_config, conditioning_var, conditioning_bins
        )
        
        # Calculate confidence intervals (standard method - no bootstrap samples available)
        ci_upper, ci_lower = _calculate_confidence_intervals(sf_means, sf_stds, point_counts, confidence_interval)
        
        # Create minimal dataset
        ds_binned = xr.Dataset(
            data_vars={
                'sf': (('bin'), sf_means),
                'std_error': (('bin'), sf_stds),
                'ci_upper': (('bin'), ci_upper),
                'ci_lower': (('bin'), ci_lower),
                'point_counts': (('bin'), point_counts)
            },
            coords={
                'bin': bins_config['bin_centers'],
                f'{dim_name}_bins': ((f'{dim_name}_edges'), bins_config['bin_edges'])
            },
            attrs={
                'bin_type': 'logarithmic' if bins_config['log_bins'] else 'linear',
                'order': str(order),
                'function_type': fun,
                'variables': variables_names,
                'dimension': dim_name,
                'confidence_level': confidence_interval,                'bootstrappable_dimensions': 'none',
                'weighting': 'volume_element'
            }
        )
        
        print("1D SF COMPLETED SUCCESSFULLY (no bootstrapping)!")
        print("="*60)
        
        return ds_binned
    
    # Normal bootstrapping case
    spacing_values = all_spacings
    print(f"Available spacings: {spacing_values}")
    gc.collect()
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_1d(
        ds, dim_name, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, spacing_values,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, n_jobs, backend, conditioning_var, conditioning_bins,
        confidence_level=confidence_interval, seed=seed
    )
    
    # Add variables_names to results for dataset creation
    results['variables_names'] = variables_names
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_binned = _create_1d_dataset(
        results, bins_config, dim_name, order, fun,
        bootstrappable_dims, convergence_eps, max_nbootstrap,
        initial_nbootstrap, confidence_interval, backend
    )
    
    print("1D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned
#####################################################################################################################
