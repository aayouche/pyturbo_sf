"""Binning Tools"""

import numpy as np
import xarray as xr
from scipy import stats

from .structure_functions import (
    calculate_structure_function_1d,
    calculate_structure_function_2d,
    calculate_structure_function_3d
)

from .utils import _is_log_spaced


##################################################
# HELPER FUNCTION FOR PROPER BINNING
##################################################

def _get_bin_indices_with_range_check(values, bin_edges, n_bins):
    """
    Get bin indices with proper handling of edge cases.
    
    Unlike np.clip(np.digitize(...) - 1, 0, n_bins - 1) which forces
    out-of-range values into edge bins, this function:
    1. Properly handles values exactly at the last bin edge (includes them)
    2. Returns a mask indicating which values are in range
    3. Returns bin indices (only valid where in_range_mask is True)
    
    Parameters
    ----------
    values : array
        Values to bin
    bin_edges : array
        Bin edge values
    n_bins : int
        Number of bins (len(bin_edges) - 1)
        
    Returns
    -------
    bin_idx : array
        Bin indices (only valid where in_range_mask is True)
    in_range_mask : array
        Boolean mask indicating which values are within bin range
    """
    # Get raw bin indices
    bin_idx = np.digitize(values, bin_edges) - 1
    
    # Handle last edge: values exactly at last edge should go to last bin
    # (digitize treats last edge as outside, we want it included)
    at_last_edge = np.isclose(values, bin_edges[-1], rtol=1e-10, atol=1e-10)
    bin_idx = np.where(at_last_edge, n_bins - 1, bin_idx)
    
    # Create range mask: include values >= first edge AND <= last edge
    in_range_mask = (values >= bin_edges[0]) & (values <= bin_edges[-1])
    
    # Clip indices to valid range (for safety, though they should be correct now)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    return bin_idx, in_range_mask


##################################################################################################1D###############################################################################################################
def _initialize_1d_bins(bin_edges, dim_name):
    """
    Initialize 1D bin configuration.
    
    Parameters
    ----------
    bin_edges : array
        Bin edges
    dim_name : str
        Dimension name
        
    Returns
    -------
    config : dict
        Dictionary with bin configuration including:
        - bin_edges: bin edges
        - bin_centers: bin centers  
        - n_bins: number of bins
        - log_bins: whether bins are logarithmic
    """
    n_bins = len(bin_edges) - 1
    
    if len(bin_edges) < 2:
        raise ValueError(f"Bin edges must have at least 2 values")
    
    # Check if bins are logarithmic or linear
    log_bins = False
    
    if np.all(bin_edges > 0):  # Only check log bins if all values are positive
        ratios = bin_edges[1:] / bin_edges[:-1]
        ratio_std = np.std(ratios)
        ratio_mean = np.mean(ratios)
        
        # Determine bin type
        if ratio_std / ratio_mean < 0.01:
            if np.abs(ratio_mean - 1.0) < 0.01:
                log_bins = False  # Linear bins
                print(f"Detected linear binning for dimension '{dim_name}'")
            else:
                log_bins = True  # Log bins
                print(f"Detected logarithmic binning for dimension '{dim_name}'")
        else:
            log_bins = False  # Default to linear if irregular spacing
            print(f"Detected irregular bin spacing for dimension '{dim_name}', treating as linear")
    else:
        log_bins = False
        print(f"Bins contain negative or zero values, using linear binning")
    
    # Calculate bin centers based on bin type
    if log_bins:
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log bins
    else:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Arithmetic mean for linear bins
    
    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'n_bins': n_bins,
        'log_bins': log_bins,
        'dim_name': dim_name
    }

def _process_no_bootstrap_1d(ds, dim_name, variables_names, order, fun, bins_config, conditioning_var=None, conditioning_bins=None):
    """
    Handle the special case of no bootstrappable dimensions for 1D.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing scalar fields
    dim_name : str
        Name of the dimension
    variables_names : list
        List of variable names
    order : float or tuple
        Order(s) of the structure function
    fun : str
        Type of structure function
    bins_config : dict
        Bin configuration from _initialize_1d_bins
    conditioning_var : str, optional
        Name of conditioning variable in dataset
    conditioning_bins : tuple, optional
        (T_lo, T_hi) bounds for conditioning
        
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Standard deviations
    point_counts : array
        Point counts per bin
    """
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function once with the entire dataset
    results, separations, pair_counts = calculate_structure_function_1d(
        ds=ds,
        dim=dim_name,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        conditioning_var=conditioning_var,
        conditioning_bins=conditioning_bins
    )
    
    # Filter out invalid values
    valid_mask = ~np.isnan(results) & ~np.isnan(separations)
    valid_results = results[valid_mask]
    valid_separations = separations[valid_mask]
    if len(valid_results) == 0:
        raise ValueError("No valid results found to bin")
    
    # Get bin indices WITH PROPER RANGE CHECKING
    n_bins = bins_config['n_bins']
    bin_indices, in_range_mask = _get_bin_indices_with_range_check(
        valid_separations, bins_config['bin_edges'], n_bins
    )
    
    # Apply range mask - only use values within bin range
    valid_results = valid_results[in_range_mask]
    valid_separations = valid_separations[in_range_mask]
    bin_indices = bin_indices[in_range_mask]
    
    if len(valid_results) == 0:
        raise ValueError("No valid results found within bin range")
    
    # Initialize arrays for binning
    sf_means = np.full(n_bins, np.nan)
    sf_stds = np.full(n_bins, np.nan)
    point_counts = np.zeros(n_bins, dtype=np.int_)
    
    # Calculate weights (using separation distance)
    weights = np.abs(valid_separations)
    weights = np.maximum(weights, 1e-10)  # Avoid zero weights
    
    # Bin the data using unique bin IDs for vectorization
    unique_bins, inverse_indices, counts = np.unique(bin_indices, return_inverse=True, return_counts=True)
    
    # Process each unique bin
    for i, bin_id in enumerate(unique_bins):
        if bin_id < 0 or bin_id >= n_bins:
            continue
            
        # Get mask for this bin
        bin_mask = inverse_indices == i
        bin_count = counts[i]
        
        # Extract values for this bin
        bin_sf = valid_results[bin_mask]
        bin_weights = weights[bin_mask]
        
        # Update counts
        point_counts[bin_id] = bin_count
        
        # Calculate weighted mean and std
        if bin_count > 0:
            # Normalize weights to sum to number of points
            normalized_weights = bin_weights / np.sum(bin_weights) * bin_count
            sf_means[bin_id] = np.average(bin_sf, weights=normalized_weights)
            if bin_count > 1:
                # Weighted standard deviation
                weighted_var = np.average((bin_sf - sf_means[bin_id])**2, weights=normalized_weights)
                sf_stds[bin_id] = np.sqrt(weighted_var)
    
    return sf_means, sf_stds, point_counts

def _calculate_bin_density_1d(point_counts, bin_edges):
    """
    Calculate normalized bin density for 1D case.
    
    Parameters
    ----------
    point_counts : array
        Number of points in each bin
    bin_edges : array
        Bin edges
        
    Returns
    -------
    bin_density : array
        Normalized density (0 to 1)
    """
    total_points = np.sum(point_counts)
    if total_points == 0:
        return np.zeros_like(point_counts, dtype=np.float32)
    
    # Calculate all bin widths at once
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Vectorized density calculation
    bin_density = np.divide(point_counts, bin_widths * total_points, 
                          out=np.zeros_like(point_counts, dtype=np.float32), 
                          where=bin_widths > 0)
    
    # Normalize density
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
        
    return bin_density

def _create_1d_dataset(results, bins_config, dim_name, order, fun,
                     bootstrappable_dims, convergence_eps, max_nbootstrap,
                     initial_nbootstrap, confidence_level, backend):
    """
    Create output dataset for 1D binning.
    
    Parameters
    ----------
    results : dict
        Results from adaptive bootstrap loop
    bins_config : dict
        Bin configuration
    dim_name : str
        Dimension name
    order : str
        Order of structure function
    fun : str
        Function type
    bootstrappable_dims : list
        List of bootstrappable dimensions
    convergence_eps : float
        Convergence epsilon
    max_nbootstrap : int
        Maximum bootstraps
    initial_nbootstrap : int
        Initial bootstraps
    confidence_level : float
        Confidence level for intervals
    backend : str
        Backend used
        
    Returns
    -------
    ds_binned : xarray.Dataset
        Binned structure function dataset
    """
    # Use pre-computed CIs if available
    if 'ci_lower' in results and 'ci_upper' in results:
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
    else:
        # Fallback to standard method (original behavior)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        ci_upper = np.full(bins_config['n_bins'], np.nan)
        ci_lower = np.full(bins_config['n_bins'], np.nan)
        
        # Calculate confidence intervals for valid bins
        valid_bins = ~np.isnan(results['sf_means']) & ~np.isnan(results['sf_stds'])
        if np.any(valid_bins):
            ci_upper[valid_bins] = results['sf_means'][valid_bins] + z_score * results['sf_stds'][valid_bins]
            ci_lower[valid_bins] = results['sf_means'][valid_bins] - z_score * results['sf_stds'][valid_bins]
    
    # Create output dataset
    ds_binned = xr.Dataset(
        data_vars={
            'sf': (('bin'), results['sf_means']),
            'std_error': (('bin'), results['sf_stds']),
            'ci_upper': (('bin'), ci_upper),
            'ci_lower': (('bin'), ci_lower),
            'nbootstraps': (('bin'), results['bin_bootstraps']),
            'density': (('bin'), results['bin_density']),
            'point_counts': (('bin'), results['point_counts']),
            'converged': (('bin'), results['bin_status'])
        },
        coords={
            'bin': bins_config['bin_centers'],
            f'{dim_name}_bins': ((f'{dim_name}_edges'), bins_config['bin_edges'])
        },
        attrs={
            'bin_type': 'logarithmic' if bins_config['log_bins'] else 'linear',
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'order': str(order),
            'function_type': fun,
            'spacing_values': list(results['spacing_values']),
            'variables': results.get('variables_names', []),
            'dimension': dim_name,
            'confidence_level': confidence_level,
            'bootstrappable_dimensions': ','.join(bootstrappable_dims),
            'backend': backend,
            'weighting': 'volume_element',
            'bootstrap_se_method': 'n_eff_correction'
        }
    )
    
    return ds_binned    
###################################################################################################################################################################################################################

##################################################################################################2D###############################################################################################################
def _initialize_2d_bins(bins_x, bins_y, dims_order):
    """
    Initialize 2D bin configuration.
    
    Returns
    -------
    config : dict
        Dictionary with bin configuration including:
        - bins_x, bins_y: bin edges
        - x_centers, y_centers: bin centers
        - n_bins_x, n_bins_y: number of bins
        - log_bins_x, log_bins_y: whether bins are logarithmic
    """
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1
        
    log_bins_x = _is_log_spaced(bins_x)
    log_bins_y = _is_log_spaced(bins_y)
    
    # Calculate bin centers
    if log_bins_x:
        x_centers = np.sqrt(bins_x[:-1] * bins_x[1:])
    else:
        x_centers = 0.5 * (bins_x[:-1] + bins_x[1:])
        
    if log_bins_y:
        y_centers = np.sqrt(bins_y[:-1] * bins_y[1:])
    else:
        y_centers = 0.5 * (bins_y[:-1] + bins_y[1:])
    
    return {
        'bins_x': bins_x,
        'bins_y': bins_y,
        'x_centers': x_centers,
        'y_centers': y_centers,
        'n_bins_x': n_bins_x,
        'n_bins_y': n_bins_y,
        'log_bins_x': log_bins_x,
        'log_bins_y': log_bins_y,
        'dims_order': dims_order
    }

def _process_no_bootstrap_2d(ds, dims, variables_names, order, fun, bins, time_dims, conditioning_var=None, conditioning_bins=None):
    """
    Handle the special case of no bootstrappable dimensions for 2D.
    
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Standard deviations
    point_counts : array
        Point counts per bin
    bins_config : dict
        Bin configuration
    """
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function once
    results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
        ds=ds,
        dims=dims,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        time_dims=time_dims,
        conditioning_var=conditioning_var,
        conditioning_bins=conditioning_bins
    )
    
    # Initialize bins
    bins_config = _initialize_2d_bins(bins[dims[1]], bins[dims[0]], dims)
    
    # Bin the results
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    
    # Get bin indices WITH PROPER RANGE CHECKING
    x_bins_idx, x_in_range = _get_bin_indices_with_range_check(
        valid_dx, bins_config['bins_x'], bins_config['n_bins_x']
    )
    y_bins_idx, y_in_range = _get_bin_indices_with_range_check(
        valid_dy, bins_config['bins_y'], bins_config['n_bins_y']
    )
    
    # Combined range mask - both x and y must be in range
    in_range_mask = x_in_range & y_in_range
    
    # Apply range mask
    valid_results = valid_results[in_range_mask]
    valid_dx = valid_dx[in_range_mask]
    valid_dy = valid_dy[in_range_mask]
    x_bins_idx = x_bins_idx[in_range_mask]
    y_bins_idx = y_bins_idx[in_range_mask]
    
    # Volume element weights
    weights = np.abs(valid_dx * valid_dy)
    weights = np.maximum(weights, 1e-10)
    
    # Initialize result arrays
    sf_means = np.full((bins_config['n_bins_y'], bins_config['n_bins_x']), np.nan)
    sf_stds = np.full((bins_config['n_bins_y'], bins_config['n_bins_x']), np.nan)
    point_counts = np.zeros((bins_config['n_bins_y'], bins_config['n_bins_x']), dtype=np.int_)
    
    # Bin the data using unique bin IDs
    bin_ids = y_bins_idx * bins_config['n_bins_x'] + x_bins_idx
    unique_bins = np.unique(bin_ids)
    
    for bin_id in unique_bins:
        j, i = divmod(bin_id, bins_config['n_bins_x'])
        
        bin_mask = bin_ids == bin_id
        bin_sf = valid_results[bin_mask]
        bin_weights = weights[bin_mask]
        
        point_counts[j, i] = len(bin_sf)
        
        if len(bin_sf) > 0:
            normalized_weights = bin_weights / np.sum(bin_weights) * len(bin_weights)
            sf_means[j, i] = np.average(bin_sf, weights=normalized_weights)
            if len(bin_sf) > 1:
                weighted_var = np.average((bin_sf - sf_means[j, i])**2, weights=normalized_weights)
                sf_stds[j, i] = np.sqrt(weighted_var)
    
    return sf_means, sf_stds, point_counts, bins_config

def _calculate_bin_density_2d(point_counts, bins_x, bins_y):
    """Calculate normalized bin density for 2D case."""
    total_points = np.sum(point_counts)
    if total_points == 0:
        return np.zeros_like(point_counts, dtype=np.float32)
    
    x_widths = bins_x[1:] - bins_x[:-1]
    y_widths = bins_y[1:] - bins_y[:-1]
    bin_areas = np.outer(y_widths, x_widths)
    
    bin_density = np.divide(point_counts, bin_areas * total_points,
                          out=np.zeros_like(point_counts, dtype=np.float32),
                          where=bin_areas > 0)
    
    # Normalize
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
        
    return bin_density

def _create_2d_dataset(results, bins_config, dims, order, fun, 
                      bootstrappable_dims, time_dims, convergence_eps,
                      max_nbootstrap, initial_nbootstrap, backend,
                      confidence_level=0.95):
    """Create output dataset for 2D binning."""
    
    # Use pre-computed CIs if available
    if 'ci_lower' in results and 'ci_upper' in results:
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
    else:
        # Fallback to normal approximation
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        ci_upper = np.full_like(results['sf_means'], np.nan)
        ci_lower = np.full_like(results['sf_means'], np.nan)
        
        valid_bins = ~np.isnan(results['sf_means']) & ~np.isnan(results['sf_stds'])
        if np.any(valid_bins):
            ci_upper[valid_bins] = results['sf_means'][valid_bins] + z_score * results['sf_stds'][valid_bins]
            ci_lower[valid_bins] = results['sf_means'][valid_bins] - z_score * results['sf_stds'][valid_bins]
    
    ds_binned = xr.Dataset(
        data_vars={
            'sf': ((dims[0], dims[1]), results['sf_means']),
            'std_error': ((dims[0], dims[1]), results['sf_stds']),
            'ci_upper': ((dims[0], dims[1]), ci_upper),
            'ci_lower': ((dims[0], dims[1]), ci_lower),
            'nbootstraps': ((dims[0], dims[1]), results['bin_bootstraps']),
            'density': ((dims[0], dims[1]), results['bin_density']),
            'point_counts': ((dims[0], dims[1]), results['point_counts']),
            'converged': ((dims[0], dims[1]), results['bin_status'])
        },
        coords={
            dims[1]: bins_config['x_centers'],
            dims[0]: bins_config['y_centers']
        },
        attrs={
            'bin_type_x': 'logarithmic' if bins_config['log_bins_x'] else 'linear',
            'bin_type_y': 'logarithmic' if bins_config['log_bins_y'] else 'linear',
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'order': str(order),
            'function_type': fun,
            'spacing_values': list(results['spacing_values']),
            'variables': ','.join(results.get('variables_names', [])),
            'bootstrappable_dimensions': ','.join(bootstrappable_dims),
            'time_dimensions': ','.join([dim for dim, is_time in time_dims.items() if is_time]),
            'backend': backend,
            'confidence_level': confidence_level,
            'weighting': 'volume_element',
            'bootstrap_se_method': 'n_eff_correction'
        }
    )
    
    # Add bin edges
    ds_binned[f'{dims[1]}_bins'] = ((dims[1], 'edge'), 
                                   np.column_stack([bins_config['bins_x'][:-1], 
                                                   bins_config['bins_x'][1:]]))
    ds_binned[f'{dims[0]}_bins'] = ((dims[0], 'edge'), 
                                   np.column_stack([bins_config['bins_y'][:-1], 
                                                   bins_config['bins_y'][1:]]))
    
    return ds_binned

###################################################################################################################################################################################################################

##################################################################################################3D###############################################################################################################
def _initialize_3d_bins(bins_x, bins_y, bins_z, dims_order):
    """
    Initialize 3D bin configuration.
    
    Returns
    -------
    config : dict
        Dictionary with bin configuration including:
        - bins_x, bins_y, bins_z: bin edges
        - x_centers, y_centers, z_centers: bin centers
        - n_bins_x, n_bins_y, n_bins_z: number of bins
        - log_bins_x, log_bins_y, log_bins_z: whether bins are logarithmic
    """
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1
    n_bins_z = len(bins_z) - 1
        
    log_bins_x = _is_log_spaced(bins_x)
    log_bins_y = _is_log_spaced(bins_y)
    log_bins_z = _is_log_spaced(bins_z)
    
    # Calculate bin centers
    if log_bins_x:
        x_centers = np.sqrt(bins_x[:-1] * bins_x[1:])
    else:
        x_centers = 0.5 * (bins_x[:-1] + bins_x[1:])
        
    if log_bins_y:
        y_centers = np.sqrt(bins_y[:-1] * bins_y[1:])
    else:
        y_centers = 0.5 * (bins_y[:-1] + bins_y[1:])
        
    if log_bins_z:
        z_centers = np.sqrt(bins_z[:-1] * bins_z[1:])
    else:
        z_centers = 0.5 * (bins_z[:-1] + bins_z[1:])
    
    return {
        'bins_x': bins_x,
        'bins_y': bins_y,
        'bins_z': bins_z,
        'x_centers': x_centers,
        'y_centers': y_centers,
        'z_centers': z_centers,
        'n_bins_x': n_bins_x,
        'n_bins_y': n_bins_y,
        'n_bins_z': n_bins_z,
        'log_bins_x': log_bins_x,
        'log_bins_y': log_bins_y,
        'log_bins_z': log_bins_z,
        'dims_order': dims_order
    }

def _process_no_bootstrap_3d(ds, dims, variables_names, order, fun, bins, time_dims, conditioning_var=None, conditioning_bins=None):
    """
    Handle the special case of no bootstrappable dimensions for 3D.
    
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Standard deviations
    point_counts : array
        Point counts per bin
    bins_config : dict
        Bin configuration
    """
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function once
    results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
        ds=ds,
        dims=dims,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        time_dims=time_dims,
        conditioning_var=conditioning_var,
        conditioning_bins=conditioning_bins
    )
    
    # Initialize bins
    bins_config = _initialize_3d_bins(bins[dims[2]], bins[dims[1]], bins[dims[0]], dims)
    
    # Bin the results
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals) & ~np.isnan(dz_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    valid_dz = dz_vals[valid_mask]
    
    # Get bin indices WITH PROPER RANGE CHECKING
    x_bins_idx, x_in_range = _get_bin_indices_with_range_check(
        valid_dx, bins_config['bins_x'], bins_config['n_bins_x']
    )
    y_bins_idx, y_in_range = _get_bin_indices_with_range_check(
        valid_dy, bins_config['bins_y'], bins_config['n_bins_y']
    )
    z_bins_idx, z_in_range = _get_bin_indices_with_range_check(
        valid_dz, bins_config['bins_z'], bins_config['n_bins_z']
    )
    
    # Combined range mask - all dimensions must be in range
    in_range_mask = x_in_range & y_in_range & z_in_range
    
    # Apply range mask
    valid_results = valid_results[in_range_mask]
    valid_dx = valid_dx[in_range_mask]
    valid_dy = valid_dy[in_range_mask]
    valid_dz = valid_dz[in_range_mask]
    x_bins_idx = x_bins_idx[in_range_mask]
    y_bins_idx = y_bins_idx[in_range_mask]
    z_bins_idx = z_bins_idx[in_range_mask]
    
    # Volume element weights
    weights = np.abs(valid_dx * valid_dy * valid_dz)
    weights = np.maximum(weights, 1e-10)
    
    # Initialize result arrays
    sf_means = np.full((bins_config['n_bins_z'], bins_config['n_bins_y'], bins_config['n_bins_x']), np.nan)
    sf_stds = np.full((bins_config['n_bins_z'], bins_config['n_bins_y'], bins_config['n_bins_x']), np.nan)
    point_counts = np.zeros((bins_config['n_bins_z'], bins_config['n_bins_y'], bins_config['n_bins_x']), dtype=np.int_)
    
    # Bin the data using unique bin IDs
    bin_ids = z_bins_idx * bins_config['n_bins_y'] * bins_config['n_bins_x'] + y_bins_idx * bins_config['n_bins_x'] + x_bins_idx
    unique_bins = np.unique(bin_ids)
    
    for bin_id in unique_bins:
        k = bin_id // (bins_config['n_bins_y'] * bins_config['n_bins_x'])
        j = (bin_id % (bins_config['n_bins_y'] * bins_config['n_bins_x'])) // bins_config['n_bins_x']
        i = bin_id % bins_config['n_bins_x']
        
        bin_mask = bin_ids == bin_id
        bin_sf = valid_results[bin_mask]
        bin_weights = weights[bin_mask]
        
        point_counts[k, j, i] = len(bin_sf)
        
        if len(bin_sf) > 0:
            normalized_weights = bin_weights / np.sum(bin_weights) * len(bin_weights)
            sf_means[k, j, i] = np.average(bin_sf, weights=normalized_weights)
            if len(bin_sf) > 1:
                weighted_var = np.average((bin_sf - sf_means[k, j, i])**2, weights=normalized_weights)
                sf_stds[k, j, i] = np.sqrt(weighted_var)
    
    return sf_means, sf_stds, point_counts, bins_config

def _calculate_bin_density_3d(point_counts, bins_x, bins_y, bins_z):
    """Calculate normalized bin density for 3D case."""
    total_points = np.sum(point_counts)
    if total_points == 0:
        return np.zeros_like(point_counts, dtype=np.float32)
    
    # Calculate bin volumes
    x_widths = bins_x[1:] - bins_x[:-1]
    y_widths = bins_y[1:] - bins_y[:-1]
    z_widths = bins_z[1:] - bins_z[:-1]
    
    # Create meshgrid of widths
    Z, Y, X = np.meshgrid(z_widths, y_widths, x_widths, indexing='ij')
    bin_volumes = Z * Y * X
    
    bin_density = np.divide(point_counts, bin_volumes * total_points,
                          out=np.zeros_like(point_counts, dtype=np.float32),
                          where=bin_volumes > 0)
    
    # Normalize
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
        
    return bin_density

def _create_3d_dataset(results, bins_config, dims, order, fun, 
                      bootstrappable_dims, time_dims, convergence_eps,
                      max_nbootstrap, initial_nbootstrap, backend, variables_names,
                      confidence_level=0.95):
    """Create output dataset for 3D binning."""
    
    # Use pre-computed CIs if available
    if 'ci_lower' in results and 'ci_upper' in results:
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
    else:
        # Fallback to normal approximation
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        ci_upper = np.full_like(results['sf_means'], np.nan)
        ci_lower = np.full_like(results['sf_means'], np.nan)
        
        valid_bins = ~np.isnan(results['sf_means']) & ~np.isnan(results['sf_stds'])
        if np.any(valid_bins):
            ci_upper[valid_bins] = results['sf_means'][valid_bins] + z_score * results['sf_stds'][valid_bins]
            ci_lower[valid_bins] = results['sf_means'][valid_bins] - z_score * results['sf_stds'][valid_bins]
    
    ds_binned = xr.Dataset(
        data_vars={
            'sf': ((dims[0], dims[1], dims[2]), results['sf_means']),
            'std_error': ((dims[0], dims[1], dims[2]), results['sf_stds']),
            'ci_upper': ((dims[0], dims[1], dims[2]), ci_upper),
            'ci_lower': ((dims[0], dims[1], dims[2]), ci_lower),
            'nbootstraps': ((dims[0], dims[1], dims[2]), results['bin_bootstraps']),
            'density': ((dims[0], dims[1], dims[2]), results['bin_density']),
            'point_counts': ((dims[0], dims[1], dims[2]), results['point_counts']),
            'converged': ((dims[0], dims[1], dims[2]), results['bin_status'])
        },
        coords={
            dims[2]: bins_config['x_centers'],
            dims[1]: bins_config['y_centers'],
            dims[0]: bins_config['z_centers']
        },
        attrs={
            'bin_type_x': 'logarithmic' if bins_config['log_bins_x'] else 'linear',
            'bin_type_y': 'logarithmic' if bins_config['log_bins_y'] else 'linear',
            'bin_type_z': 'logarithmic' if bins_config['log_bins_z'] else 'linear',
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'order': str(order),
            'function_type': fun,
            'spacing_values': list(results['spacing_values']),
            'variables': ','.join(variables_names),
            'bootstrappable_dimensions': ','.join(bootstrappable_dims),
            'time_dimensions': ','.join([dim for dim, is_time in time_dims.items() if is_time]),
            'backend': backend,
            'confidence_level': confidence_level,
            'weighting': 'volume_element',
            'bootstrap_se_method': 'n_eff_correction'
        }
    )
    
    # Add bin edges
    ds_binned[f'{dims[2]}_bins'] = ((dims[2], 'edge'), 
                                   np.column_stack([bins_config['bins_x'][:-1], 
                                                   bins_config['bins_x'][1:]]))
    ds_binned[f'{dims[1]}_bins'] = ((dims[1], 'edge'), 
                                   np.column_stack([bins_config['bins_y'][:-1], 
                                                   bins_config['bins_y'][1:]]))
    ds_binned[f'{dims[0]}_bins'] = ((dims[0], 'edge'), 
                                   np.column_stack([bins_config['bins_z'][:-1], 
                                                   bins_config['bins_z'][1:]]))
    
    return ds_binned

###################################################################################################################################################################################################################
