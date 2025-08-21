"""One-dimensional structure function calculations."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import gc
from scipy import stats



from .core import (validate_dataset_1d, setup_bootsize_1d, calculate_adaptive_spacings_1d, 
                  compute_boot_indexes_1d, get_boot_indexes_1d)
from .utils import (fast_shift_1d, calculate_time_diff_1d)

##################################Structure Functions Types########################################

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import as_strided

def calc_scalar_1d(subset, dim, variable_name, order, n_points):
    """
    Calculate scalar structure function: (dscalar^n)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    dim : str
        Name of the dimension
    variable_name : str
        Name of the scalar variable
    order : int
        Order of the structure function
    n_points : int
        Number of points
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Structure function values, separation values
    """
    # Arrays to store results
    results = np.full(n_points, np.nan)
    separations = np.full(n_points, 0.0)
    
    # Get the scalar variable
    scalar_var = subset[variable_name].values
    
    # Get coordinate variable
    coord_var = subset[dim].values
    
    # Loop through all points
    for i in range(1, n_points):  # Start from 1 to avoid self-correlation
        # Calculate scalar difference
        dscalar = fast_shift_1d(scalar_var, i) - scalar_var
        
        # Calculate separation distance
        if dim == 'time':
            # Special handling for time dimension
            dt = calculate_time_diff_1d(coord_var, i)
            separation = dt
        else:
            # For spatial dimensions
            separation = fast_shift_1d(coord_var, i) - coord_var
        
        # Store the separation distance (mean of all valid separations)
        valid_sep = ~np.isnan(separation)
        if np.any(valid_sep):
            separations[i] = np.mean(np.abs(separation[valid_sep]))
        
        # Calculate scalar structure function: dscalar^n
        sf_val = dscalar ** order
        
        # Store the mean of all valid values
        valid_sf = ~np.isnan(sf_val)
        if np.any(valid_sf):
            results[i] = np.mean(sf_val[valid_sf])
    
    return results, separations


def calc_scalar_scalar_1d(subset, dim, variables_names, order, n_points):
    """
    Calculate scalar-scalar structure function: (dscalar1^n * dscalar2^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    dim : str
        Name of the dimension
    variables_names : list
        List of variable names (should contain two scalar variables)
    order : tuple
        Tuple of orders (n, k) for the structure function
    n_points : int
        Number of points
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Structure function values, separation values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for scalar-scalar structure function, got {order}")
    
    # Unpack order tuple
    n, k = order
    
    # Get variable names
    var1, var2 = variables_names
    
    # Arrays to store results
    results = np.full(n_points, np.nan)
    separations = np.full(n_points, 0.0)
    
    # Get the scalar variables
    scalar_var1 = subset[var1].values
    scalar_var2 = subset[var2].values
    
    # Get coordinate variable
    coord_var = subset[dim].values
    
    # Loop through all points
    for i in range(1, n_points):  # Start from 1 to avoid self-correlation
        # Calculate scalars difference
        dscalar1 = fast_shift_1d(scalar_var1, i) - scalar_var1
        dscalar2 = fast_shift_1d(scalar_var2, i) - scalar_var2
        
        # Calculate separation distance
        if dim == 'time':
            # Special handling for time dimension
            dt = calculate_time_diff_1d(coord_var, i)
            separation = dt
        else:
            # For spatial dimensions
            separation = fast_shift_1d(coord_var, i) - coord_var
        
        # Store the separation distance (mean of all valid separations)
        valid_sep = ~np.isnan(separation)
        if np.any(valid_sep):
            separations[i] = np.mean(np.abs(separation[valid_sep]))
        
        # Calculate scalar-scalar structure function: dscalar1^n * dscalar2^k
        sf_val = (dscalar1 ** n) * (dscalar2 ** k)
        
        # Store the mean of all valid values
        valid_sf = ~np.isnan(sf_val)
        if np.any(valid_sf):
            results[i] = np.mean(sf_val[valid_sf])
    
    return results, separations
#####################################################################################################################

################################Main SF Function#####################################################################

def calculate_structure_function_1d(ds, dim, variables_names, order, fun='scalar', nb=0, 
                                   spacing=None, num_bootstrappable=0, boot_indexes=None, bootsize=None):
    """
    Main function to calculate structure functions based on specified type.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing scalar fields
    dim : str
        Name of the dimension
    variables_names : list
        List of variable names to use, depends on function type
    order : int or tuple
        Order(s) of the structure function
    fun : str, optional
        Type of structure function: ['scalar', 'scalar_scalar']
    nb : int, optional
        Bootstrap index
    spacing : dict or int, optional
        Spacing value to use
    num_bootstrappable : int, optional
        Number of bootstrappable dimensions
    boot_indexes : dict, optional
        Dictionary with spacing values as keys and boot indexes as values
    bootsize : dict, optional
        Dictionary with dimension name as key and bootsize as value
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Structure function values, separation values
    """
    # If no bootstrappable dimensions, use the full dataset
    if num_bootstrappable == 0:
        subset = ds
    else:
        # Get data shape
        data_shape = dict(ds.sizes)
        
        # Use default spacing of 1 if None provided
        if spacing is None:
            sp_value = 1
        # Convert dict spacing to single value if needed
        elif isinstance(spacing, dict):
            # Get the spacing for the bootstrappable dimension
            if dim in spacing:
                sp_value = spacing[dim]
            else:
                sp_value = 1  # Default if dimension not found
        else:
            sp_value = spacing
        
        # Get boot indexes
        if boot_indexes is None or sp_value not in boot_indexes:
            # Calculate boot indexes on-the-fly
            indexes = get_boot_indexes_1d(dim, data_shape, bootsize, [sp_value], {}, num_bootstrappable, sp_value)
        else:
            indexes = boot_indexes[sp_value]
        
        # Check if we have valid indexes
        if not indexes or dim not in indexes or indexes[dim].shape[1] <= nb:
            print(f"Warning: No valid indexes for bootstrapping. Using the full dataset.")
            subset = ds
        else:
            # Extract the subset based on bootstrap index
            subset = ds.isel({dim: indexes[dim][:, nb]})
    
    # Check if the required variables exist in the dataset
    for var_name in variables_names:
        if var_name not in subset:
            raise ValueError(f"Variable {var_name} not found in dataset")
    
    # Get dimension of the subset
    n_points = len(subset[variables_names[0]])
    
    # Calculate structure function based on specified type
    if fun == 'scalar':
        if len(variables_names) != 1:
            raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(variables_names)}")
        
        variable_name = variables_names[0]
        results, separations = calc_scalar_1d(subset, dim, variable_name, order, n_points)
        
    elif fun == 'scalar_scalar':
        results, separations = calc_scalar_scalar_1d(subset, dim, variables_names, order, n_points)
        
    else:
        raise ValueError(f"Unsupported function type: {fun}. Only 'scalar' and 'scalar_scalar' are supported.")
        
    return results, separations
#####################################################################################################################

#####################################Bootstrapping Monte Carlo#######################################################
def run_bootstrap_sf(args):
    """Standalone bootstrap function for parallel processing."""
    ds, dim, variables_names, order, fun, nb, spacing, num_bootstrappable, boot_indexes, bootsize = args
    return calculate_structure_function_1d(
        ds=ds, dim=dim, variables_names=variables_names, order=order, fun=fun,
        nb=nb, spacing=spacing, num_bootstrappable=num_bootstrappable,
        boot_indexes=boot_indexes, bootsize=bootsize
    )

def monte_carlo_simulation_1d(ds, dim, variables_names, order, nbootstrap, bootsize, 
                             num_bootstrappable, all_spacings, boot_indexes,
                             fun='scalar', spacing=None, n_jobs=-1, backend='threading'):
    """
    Run Monte Carlo simulation for structure function calculation with multiple bootstrap samples.
    """
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, separations = calculate_structure_function_1d(
            ds=ds,
            dim=dim,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable
        )
        return [results], [separations]
    
    # Use default spacing of 1 if None provided
    if spacing is None:
        sp_value = 1
    # Convert dict spacing to single value if needed
    elif isinstance(spacing, dict):
        # Get the spacing for the bootstrappable dimension
        if dim in spacing:
            sp_value = spacing[dim]
        else:
            sp_value = 1  # Default if dimension not found
    else:
        sp_value = spacing
    
    
    # Get boot indexes for the specified spacing
    if sp_value in boot_indexes:
        indexes = boot_indexes[sp_value]
    else:
        # Calculate boot indexes on-the-fly
        indexes = get_boot_indexes_1d(dim, dict(ds.sizes), bootsize, all_spacings, boot_indexes, num_bootstrappable, sp_value)
    
    # Check if we have valid indexes
    if not indexes or dim not in indexes or indexes[dim].shape[1] == 0:
        print(f"Warning: No valid indices for dimension {dim} with spacing {sp_value}.")
        # Fall back to calculating once with full dataset
        results, separations = calculate_structure_function_1d(
            ds=ds,
            dim=dim,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable
        )
        return [results], [separations]
    
    # Generate random indices for the bootstrappable dimension
    random_indices = np.random.choice(indexes[dim].shape[1], size=nbootstrap)
    
    
    # Calculate optimal batch size based on number of jobs and bootstraps
    if n_jobs < 0:  # All negative n_jobs values
        import os
        total_cpus = os.cpu_count()
        if n_jobs == -1:  # Special case: use all CPUs
            n_workers = total_cpus
        else:  # Use (all CPUs - |n_jobs| - 1)
            n_workers = max(1, total_cpus + n_jobs + 1)  # +1 because -2 means all except 1
    else:
        n_workers = n_jobs
    
    batch_size = max(10, nbootstrap//(n_workers))
    
    # Create all argument tuples in advance for parallel processing
    all_args = []
    for j in range(nbootstrap):
        args = (
            ds, dim, variables_names, order, fun, 
            random_indices[j], sp_value, num_bootstrappable, 
            boot_indexes, bootsize
        )
        all_args.append(args)
    
    # Run simulations in parallel using the module-level function
    results = Parallel(n_jobs=n_jobs, verbose=0,  batch_size=batch_size, backend=backend)(
        delayed(run_bootstrap_sf)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    separations = [r[1] for r in results]
    
    return sf_results, separations
#####################################################################################################################

#################################Main Binning Function###############################################################
# Helper functions at module level
def _process_spacing_data_batch_1d(sf_results, separations, bin_edges, n_bins, 
                                   bin_accumulators, point_counts, bin_spacing_counts,
                                   sp_value, bin_list, add_to_counts=True):
    """Process structure function data for a specific spacing value with batch processing."""
    # Create a set of target bins for fast lookup
    target_bins = set(bin_list)
    
    # Function to calculate bin indices
    def bin_idx_func(values):
        return np.clip(np.digitize(values, bin_edges) - 1, 0, n_bins - 1)
    
    # Process all bootstrap samples
    for b in range(len(sf_results)):
        sf = sf_results[b]
        sep = separations[b]
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(sep)
        sf_valid = sf[valid]
        sep_valid = sep[valid]
        
        if len(sf_valid) == 0:
            continue
            
        # Volume element weights
        weights = np.abs(sep_valid)
        weights = np.maximum(weights, 1e-10)
        
        # Find bin indices
        bin_idx = bin_idx_func(sep_valid)
        
        # Process each point
        for idx in range(len(sf_valid)):
            bin_id = bin_idx[idx]
            if bin_id not in target_bins:
                continue
            
            # Initialize accumulator if needed
            if bin_id not in bin_accumulators:
                bin_accumulators[bin_id] = {
                    'weighted_sum': 0.0,
                    'total_weight': 0.0,
                    'bootstrap_samples': []
                }
            
            # Accumulate weighted values
            weight = weights[idx]
            bin_accumulators[bin_id]['weighted_sum'] += sf_valid[idx] * weight
            bin_accumulators[bin_id]['total_weight'] += weight
            
            # Update counts
            if add_to_counts:
                point_counts[bin_id] += 1
                bin_spacing_counts[sp_value][bin_id] += 1
        
        # Store bootstrap contributions
        for bin_id in bin_accumulators:
            if bin_id in target_bins:
                current_weighted_sum = bin_accumulators[bin_id]['weighted_sum']
                current_total_weight = bin_accumulators[bin_id]['total_weight']
                
                if len(bin_accumulators[bin_id]['bootstrap_samples']) > 0:
                    prev_sum = sum(s['weighted_sum'] for s in bin_accumulators[bin_id]['bootstrap_samples'])
                    prev_weight = sum(s['total_weight'] for s in bin_accumulators[bin_id]['bootstrap_samples'])
                    
                    bootstrap_sum = current_weighted_sum - prev_sum
                    bootstrap_weight = current_total_weight - prev_weight
                else:
                    bootstrap_sum = current_weighted_sum
                    bootstrap_weight = current_total_weight
                
                if bootstrap_weight > 0:
                    bin_accumulators[bin_id]['bootstrap_samples'].append({
                        'weighted_sum': bootstrap_sum,
                        'total_weight': bootstrap_weight,
                        'mean': bootstrap_sum / bootstrap_weight
                    })
    
    return bin_accumulators, point_counts, bin_spacing_counts


def _update_statistics_1d(bin_list, bin_accumulators, sf_means, sf_stds):
    """Update statistics for processed bins."""
    for j in bin_list:
        if j not in bin_accumulators:
            continue
            
        acc = bin_accumulators[j]
        if acc['total_weight'] > 0:
            # Overall weighted mean
            sf_means[j] = acc['weighted_sum'] / acc['total_weight']
            
            # Bootstrap standard error
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                sf_stds[j] = np.std(boot_means, ddof=1)
            else:
                sf_stds[j] = np.nan
    
    return sf_means, sf_stds


def _update_spacing_effectiveness_1d(sp_value, bootstraps, bin_list, 
                                    bin_spacing_counts, bin_spacing_effectiveness, 
                                    bin_spacing_bootstraps):
    """Update spacing effectiveness metrics."""
    if bootstraps > 0:
        for j in bin_list:
            if bin_spacing_counts[sp_value][j] > 0:
                bin_spacing_effectiveness[sp_value][j] = bin_spacing_counts[sp_value][j] / bootstraps
                bin_spacing_bootstraps[sp_value][j] += bootstraps
    
    return bin_spacing_effectiveness, bin_spacing_bootstraps


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


def _process_no_bootstrap_1d(ds, dim_name, variables_names, order, fun, bins_config):
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
    results, separations = calculate_structure_function_1d(
        ds=ds,
        dim=dim_name,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0
    )
    
    # Filter out invalid values
    valid_mask = ~np.isnan(results) & ~np.isnan(separations)
    valid_results = results[valid_mask]
    valid_separations = separations[valid_mask]
    
    if len(valid_results) == 0:
        raise ValueError("No valid results found to bin")
    
    # Create bin indices using numpy's digitize
    bin_indices = np.clip(np.digitize(valid_separations, bins_config['bin_edges']) - 1, 
                         0, bins_config['n_bins'] - 1)
    
    # Initialize arrays for binning
    n_bins = bins_config['n_bins']
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


def _calculate_bootstrap_statistics_1d(bin_accumulators, n_bins):
    """
    Calculate weighted means and bootstrap standard errors for 1D bins.
    
    Parameters
    ----------
    bin_accumulators : dict
        Accumulator dictionary with bin indices as keys
    n_bins : int
        Number of bins
        
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Bootstrap standard errors
    """
    sf_means = np.full(n_bins, np.nan)
    sf_stds = np.full(n_bins, np.nan)
    
    for j, acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            # Overall weighted mean
            sf_means[j] = acc['weighted_sum'] / acc['total_weight']
            
            # Bootstrap standard error
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                sf_stds[j] = np.std(boot_means, ddof=1)
            else:
                sf_stds[j] = np.nan
    
    return sf_means, sf_stds


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


def _evaluate_convergence_1d(sf_stds, point_counts, bin_bootstraps, 
                           convergence_eps, max_bootstraps):
    """
    Evaluate which bins have converged.
    
    Parameters
    ----------
    sf_stds : array
        Standard deviations
    point_counts : array
        Point counts
    bin_bootstraps : array
        Number of bootstraps per bin
    convergence_eps : float
        Convergence threshold
    max_bootstraps : int
        Maximum number of bootstraps
        
    Returns
    -------
    converged : array
        Boolean array indicating converged bins
    convergence_reasons : dict
        Dictionary mapping reason to count
    """
    converged = np.zeros_like(sf_stds, dtype=bool)
    reasons = {
        'low_density': 0,
        'nan_std': 0,
        'converged_eps': 0,
        'max_bootstraps': 0
    }
    
    # Low density bins
    low_density = (point_counts <= 10) & ~converged
    converged |= low_density
    reasons['low_density'] = np.sum(low_density)
    
    # NaN standard deviations
    nan_std = np.isnan(sf_stds) & ~converged
    converged |= nan_std
    reasons['nan_std'] = np.sum(nan_std)
    
    # Converged by epsilon
    eps_converged = (sf_stds <= convergence_eps) & ~converged & (point_counts > 10)
    converged |= eps_converged
    reasons['converged_eps'] = np.sum(eps_converged)
    
    # Max bootstraps reached
    max_boot = (bin_bootstraps >= max_bootstraps) & ~converged
    converged |= max_boot
    reasons['max_bootstraps'] = np.sum(max_boot)
    
    return converged, reasons


def _group_bins_for_iteration_1d(unconverged_indices, bin_density, bootstrap_steps):
    """
    Group unconverged bins by similar characteristics.
    
    Parameters
    ----------
    unconverged_indices : array
        Indices of unconverged bins
    bin_density : array
        Normalized bin density
    bootstrap_steps : array
        Step sizes for each bin
        
    Returns
    -------
    groups : dict
        Dictionary mapping (step, density_quartile) to list of bin indices
    """
    groups = {}
    
    for j in unconverged_indices:
        step = bootstrap_steps[j]
        density_quartile = int(bin_density[j] * 4)
        group_key = (step, density_quartile)
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(j)
    
    return groups


def _get_spacing_distribution_1d(bin_list, spacing_effectiveness, total_bootstraps, 
                               spacing_values):
    """
    Determine optimal distribution of bootstraps across spacings.
    
    Parameters
    ----------
    bin_list : list
        List of bins to process
    spacing_effectiveness : dict
        Effectiveness scores for each spacing
    total_bootstraps : int
        Total bootstraps to distribute
    spacing_values : list
        Available spacing values
        
    Returns
    -------
    distribution : list
        List of (spacing, bootstraps) tuples
    """
    # Calculate average effectiveness for this group
    group_effectiveness = {}
    for sp in spacing_values:
        total_eff = sum(spacing_effectiveness[sp][j] for j in bin_list)
        group_effectiveness[sp] = total_eff / len(bin_list) if len(bin_list) > 0 else 0
    
    # Sort spacings by effectiveness
    sorted_spacings = sorted(group_effectiveness.items(), key=lambda x: x[1], reverse=True)
    
    # Distribute bootstraps
    total_effectiveness = sum(eff for _, eff in sorted_spacings if eff > 0)
    distribution = []
    remaining = total_bootstraps
    
    for sp_value, effectiveness in sorted_spacings:
        if effectiveness <= 0 or remaining <= 0:
            continue
            
        if total_effectiveness > 0:
            proportion = effectiveness / total_effectiveness
            sp_bootstraps = min(int(total_bootstraps * proportion), remaining)
        else:
            # Equal distribution if no effectiveness data
            sp_bootstraps = remaining // len([s for s, e in sorted_spacings if e >= 0])
        
        if sp_bootstraps > 0:
            distribution.append((sp_value, sp_bootstraps))
            remaining -= sp_bootstraps
    
    return distribution


def _update_spacing_effectiveness_1d(bin_spacing_effectiveness, bin_spacing_counts,
                                   bin_spacing_bootstraps, sp_value, bin_list, 
                                   bootstraps):
    """
    Update spacing effectiveness metrics.
    
    Parameters
    ----------
    bin_spacing_effectiveness : dict
        Effectiveness scores
    bin_spacing_counts : dict
        Point counts
    bin_spacing_bootstraps : dict
        Bootstrap counts
    sp_value : int
        Current spacing value
    bin_list : list
        Bins that were processed
    bootstraps : int
        Number of bootstraps run
    """
    if bootstraps <= 0:
        return
        
    for j in bin_list:
        if bin_spacing_counts[sp_value][j] > 0:
            bin_spacing_effectiveness[sp_value][j] = (
                bin_spacing_counts[sp_value][j] / bootstraps
            )
            bin_spacing_bootstraps[sp_value][j] += bootstraps


def _run_adaptive_bootstrap_loop_1d(ds, dim_name, variables_names, order, fun,
                                  bins_config, initial_nbootstrap, max_nbootstrap,
                                  step_nbootstrap, convergence_eps, spacing_values,
                                  bootsize_dict, num_bootstrappable, all_spacings,
                                  boot_indexes, n_jobs, backend):
    """
    Run adaptive bootstrap loop for 1D structure function binning.
    
    This is the main workhorse function that handles the iterative
    bootstrap refinement process.
    """
    n_bins = bins_config['n_bins']
    
    # Initialize result arrays
    sf_means = np.full(n_bins, np.nan)
    sf_stds = np.full(n_bins, np.nan)
    point_counts = np.zeros(n_bins, dtype=np.int_)
    bin_density = np.zeros(n_bins, dtype=np.float32)
    bin_status = np.zeros(n_bins, dtype=bool)
    bin_bootstraps = np.ones(n_bins, dtype=np.int_) * initial_nbootstrap
    bootstrap_steps = np.ones(n_bins, dtype=np.int_) * step_nbootstrap
    
    # Accumulator for weighted statistics
    bin_accumulators = {}
    
    # Initialize spacing effectiveness tracking
    bin_spacing_effectiveness = {sp: np.zeros(n_bins, dtype=np.float32) for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros(n_bins, dtype=np.int_) for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros(n_bins, dtype=np.int_) for sp in spacing_values}
    
    # Process initial bootstraps
    print("\nINITIAL BOOTSTRAP PHASE")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    all_bins = list(range(n_bins))
    
    for sp_value in spacing_values:
        if init_samples_per_spacing <= 0:
            continue
            
        print(f"  Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Run Monte Carlo simulation
        sf_results, separations = monte_carlo_simulation_1d(
            ds=ds,
            dim=dim_name,
            variables_names=variables_names,
            order=order, 
            nbootstrap=init_samples_per_spacing, 
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            fun=fun, 
            spacing=sp_value,
            n_jobs=n_jobs,
            backend=backend
        )
        
        # Process the results
        _process_spacing_data_batch_1d(
            sf_results, separations, bins_config['bin_edges'], n_bins,
            bin_accumulators, point_counts, bin_spacing_counts,
            sp_value, all_bins, add_to_counts=True
        )
        
        # Update effectiveness
        _update_spacing_effectiveness_1d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value, all_bins,
            init_samples_per_spacing
        )
        
        # Clean memory
        del sf_results, separations
        gc.collect()
    
    # Calculate statistics from accumulators
    sf_means, sf_stds = _calculate_bootstrap_statistics_1d(bin_accumulators, n_bins)
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    bin_density = _calculate_bin_density_1d(point_counts, bins_config['bin_edges'])
    
    print(f"Total points collected: {np.sum(point_counts)}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins}")
    print(f"Maximum density bin has {np.max(point_counts)} points")
    
    # Initial convergence check
    bin_status, convergence_reasons = _evaluate_convergence_1d(
        sf_stds, point_counts, bin_bootstraps, convergence_eps, max_nbootstrap
    )
    
    for reason, count in convergence_reasons.items():
        if count > 0:
            print(f"Marked {count} bins as converged ({reason})")
    
    # Main convergence loop
    iteration = 1
    print("\nSTARTING ADAPTIVE CONVERGENCE LOOP")
    
    while True:
        # Find unconverged bins
        unconverged = ~bin_status & (point_counts > 10) & (bin_bootstraps < max_nbootstrap)
        if not np.any(unconverged):
            print("All bins have converged or reached max bootstraps!")
            break
            
        print(f"\nIteration {iteration} - {np.sum(unconverged)} unconverged bins")
        
        # Group bins by similar bootstrap requirements
        unconverged_indices = np.where(unconverged)[0]
        groups = _group_bins_for_iteration_1d(unconverged_indices, bin_density, bootstrap_steps)
        
        print(f"Grouped unconverged bins into {len(groups)} groups")
        
        # Process each group
        for (step, density_q), bin_list in sorted(groups.items(), 
                                                 key=lambda x: (x[0][1], x[0][0]), 
                                                 reverse=True):
            print(f"\nProcessing {len(bin_list)} bins with step size {step} in density quartile {density_q}")
            
            # Get optimal spacing distribution
            distribution = _get_spacing_distribution_1d(
                bin_list, bin_spacing_effectiveness, step, spacing_values
            )
            
            # Process each spacing
            for sp_value, sp_bootstraps in distribution:
                if sp_bootstraps <= 0:
                    continue
                    
                print(f"  Batch processing spacing {sp_value} with {sp_bootstraps} bootstraps for {len(bin_list)} bins")
                
                # Run Monte Carlo simulation
                sf_results, separations = monte_carlo_simulation_1d(
                    ds=ds,
                    dim=dim_name,
                    variables_names=variables_names,
                    order=order, 
                    nbootstrap=sp_bootstraps, 
                    bootsize=bootsize_dict,
                    num_bootstrappable=num_bootstrappable,
                    all_spacings=all_spacings,
                    boot_indexes=boot_indexes,
                    fun=fun, 
                    spacing=sp_value,
                    n_jobs=n_jobs,
                    backend=backend
                )
                
                # Process the results (no count updates)
                _process_spacing_data_batch_1d(
                    sf_results, separations, bins_config['bin_edges'], n_bins,
                    bin_accumulators, point_counts, bin_spacing_counts,
                    sp_value, bin_list, add_to_counts=False
                )
                
                # Update effectiveness
                _update_spacing_effectiveness_1d(
                    bin_spacing_effectiveness, bin_spacing_counts,
                    bin_spacing_bootstraps, sp_value, bin_list,
                    sp_bootstraps
                )
                
                # Clean memory
                del sf_results, separations
                gc.collect()
            
            # Update bootstrap counts and check convergence
            for j in bin_list:
                bin_bootstraps[j] += step
                
                # Recalculate statistics for this bin
                if j in bin_accumulators:
                    acc = bin_accumulators[j]
                    if acc['total_weight'] > 0:
                        sf_means[j] = acc['weighted_sum'] / acc['total_weight']
                        if len(acc['bootstrap_samples']) > 1:
                            boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                            sf_stds[j] = np.std(boot_means, ddof=1)
                
                # Check convergence
                if sf_stds[j] <= convergence_eps:
                    bin_status[j] = True
                    print(f"  Bin {j} (separation={bins_config['bin_centers'][j]:.4f}) CONVERGED with std {sf_stds[j]:.6f}")
                elif bin_bootstraps[j] >= max_nbootstrap:
                    bin_status[j] = True
                    print(f"  Bin {j} (separation={bins_config['bin_centers'][j]:.4f}) reached MAX BOOTSTRAPS")
        
        # Next iteration
        iteration += 1
        gc.collect()
    
    # Final convergence statistics
    converged_bins = np.sum(bin_status & (point_counts > 10))
    unconverged_bins = np.sum(~bin_status & (point_counts > 10))
    max_bootstrap_bins = np.sum((bin_bootstraps >= max_nbootstrap) & (point_counts > 10))
    
    print("\nFINAL CONVERGENCE STATISTICS:")
    print(f"  Total bins with data (>10 points): {np.sum(point_counts > 10)}")
    print(f"  Converged bins: {converged_bins}")
    print(f"  Unconverged bins: {unconverged_bins}")
    print(f"  Bins at max bootstraps: {max_bootstrap_bins}")
    
    # Return all results
    return {
        'sf_means': sf_means,
        'sf_stds': sf_stds,
        'point_counts': point_counts,
        'bin_density': bin_density,
        'bin_status': bin_status,
        'bin_bootstraps': bin_bootstraps,
        'spacing_values': spacing_values
    }


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
    # Calculate confidence intervals
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
            'sf_std': (('bin'), results['sf_stds']),
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
            'bootstrap_se_method': 'unweighted_std'
        }
    )
    
    return ds_binned


def bin_sf_1d(ds, variables_names, order, bins, bootsize=None, fun='scalar', 
             initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
             convergence_eps=0.1, n_jobs=-1, backend='threading'):
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
            ds, dim_name, variables_names, order, fun, bins_config
        )
        
        # Calculate confidence intervals
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_upper = np.full(bins_config['n_bins'], np.nan)
        ci_lower = np.full(bins_config['n_bins'], np.nan)
        
        # Only calculate CIs for bins with data
        valid_bins = ~np.isnan(sf_means)
        if np.any(valid_bins):
            bins_with_points = (point_counts[valid_bins] > 0)
            if np.any(bins_with_points):
                indices = np.where(valid_bins)[0][bins_with_points]
                std_error = sf_stds[indices] / np.sqrt(point_counts[indices])
                ci_upper[indices] = sf_means[indices] + z_score * std_error
                ci_lower[indices] = sf_means[indices] - z_score * std_error
        
        # Create minimal dataset
        ds_binned = xr.Dataset(
            data_vars={
                'sf': (('bin'), sf_means),
                'sf_std': (('bin'), sf_stds),
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
                'confidence_level': confidence_level,
                'bootstrappable_dimensions': 'none',
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
        boot_indexes, n_jobs, backend
    )
    
    # Add variables_names to results for dataset creation
    results['variables_names'] = variables_names
    
    # Create output dataset
    print("\nCreating output dataset...")
    confidence_level = 0.95
    ds_binned = _create_1d_dataset(
        results, bins_config, dim_name, order, fun,
        bootstrappable_dims, convergence_eps, max_nbootstrap,
        initial_nbootstrap, confidence_level, backend
    )
    
    print("1D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned
#####################################################################################################################
