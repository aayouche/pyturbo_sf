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

def monte_carlo_simulation_1d(ds, dim, variables_names, order, nbootstrap, bootsize, 
                             num_bootstrappable, all_spacings, boot_indexes,
                             fun='scalar', spacing=None, n_jobs=-1):
    """
    Run Monte Carlo simulation for structure function calculation with multiple bootstrap samples.
    
    Parameters
    -----------
    ds : xarray.Dataset
        Dataset containing scalar fields
    dim : str
        Name of the dimension
    variables_names : list
        List of variable names to use, depends on function type
    order : int or tuple
        Order(s) of the structure function
    nbootstrap : int
        Number of bootstrap samples
    bootsize : dict
        Dictionary with dimension name as key and bootsize as value
    num_bootstrappable : int
        Number of bootstrappable dimensions
    all_spacings : list
        List of all spacing values
    boot_indexes : dict
        Dictionary with spacing values as keys and boot indexes as values
    fun : str, optional
        Type of structure function: ['scalar', 'scalar_scalar']
    spacing : dict or int, optional
        Spacing value to use
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1 (all cores).
    
    Returns
    --------
    list
        Raw structure function values for all bootstrap samples
    list
        Separation values for all bootstrap samples
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
    
    # Set the seed for reproducibility
    np.random.seed(10000000)
    
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
    
    # Prepare a function to run in parallel
    def simulate_bootstrap(j):
        return calculate_structure_function_1d(
            ds=ds,
            dim=dim,
            variables_names=variables_names,
            order=order,
            fun=fun,
            nb=random_indices[j], 
            spacing=sp_value,
            num_bootstrappable=num_bootstrappable,
            boot_indexes=boot_indexes,
            bootsize=bootsize
        )
    
    # Run simulations in parallel
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(simulate_bootstrap)(j) for j in range(nbootstrap)
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    separations = [r[1] for r in results]
    
    return sf_results, separations
#####################################################################################################################

#################################Main Binning Function###############################################################

def bin_sf_1d(ds, variables_names, order, bins, bootsize=None, fun='scalar', 
             initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
             convergence_eps=0.1, n_jobs=-1):
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
    print("="*60 + "\n")
    
    # Validate bins
    if not isinstance(bins, dict):
        raise ValueError("'bins' must be a dictionary with dimension as key and bin edges as values")
    
    if dim_name not in bins:
        raise ValueError(f"Bins must be provided for dimension '{dim_name}'")
    
    bin_edges = np.array(bins[dim_name])
    
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
    
    # Extract bin array and calculate number of bins
    n_bins = len(bin_centers)
    
    # Create function to calculate bin indices
    def bin_idx_func(values):
        return np.clip(np.digitize(values, bin_edges) - 1, 0, n_bins - 1)
    
    # Special case: no bootstrappable dimensions
    if num_bootstrappable == 0:
        print("\nNo bootstrappable dimensions available. "
              "Calculating structure function once with full dataset.")
        
        # Calculate structure function once with the entire dataset
        results, separations = calculate_structure_function_1d(
            ds=ds,
            dim=dim_name,
            variables_names=variables_names,
            order=order,
            fun=fun,
            num_bootstrappable=num_bootstrappable
        )
        
        # Filter out invalid values
        valid_mask = ~np.isnan(results) & ~np.isnan(separations)
        valid_results = results[valid_mask]
        valid_separations = separations[valid_mask]
        
        if len(valid_results) == 0:
            raise ValueError("No valid results found to bin")
        
        # Create bin indices using numpy's digitize
        bin_indices = bin_idx_func(valid_separations)
        
        # Initialize arrays for binning
        sf_means = np.full(n_bins, np.nan)
        sf_stds = np.full(n_bins, np.nan)
        point_counts = np.zeros(n_bins, dtype=np.int32)
        
        # Calculate weights (using separation distance)
        weights = np.abs(valid_separations)
        
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
            weight_sum = np.sum(bin_weights)
            if weight_sum > 0:
                norm_weights = bin_weights / weight_sum
                sf_means[bin_id] = np.sum(bin_sf * norm_weights)
                sf_stds[bin_id] = np.sqrt(np.sum(norm_weights * (bin_sf - sf_means[bin_id])**2))
        
        # Calculate confidence intervals
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_upper = np.full(n_bins, np.nan)
        ci_lower = np.full(n_bins, np.nan)
        
        # Only calculate CIs for bins with data
        valid_bins = ~np.isnan(sf_means)
        if np.any(valid_bins):
            bins_with_points = (point_counts[valid_bins] > 0)
            if np.any(bins_with_points):
                indices = np.where(valid_bins)[0][bins_with_points]
                ci_upper[indices] = sf_means[indices] + z_score * sf_stds[indices] / np.sqrt(point_counts[indices])
                ci_lower[indices] = sf_means[indices] - z_score * sf_stds[indices] / np.sqrt(point_counts[indices])
        
        # Create output dataset
        ds_binned = xr.Dataset(
            data_vars={
                'sf': (('bin'), sf_means),
                'sf_std': (('bin'), sf_stds),
                'ci_upper': (('bin'), ci_upper),
                'ci_lower': (('bin'), ci_lower),
                'point_counts': (('bin'), point_counts)
            },
            coords={
                'bin': bin_centers,
                f'{dim_name}_bins': ((f'{dim_name}_edges'), bin_edges)
            },
            attrs={
                'bin_type': 'logarithmic' if log_bins else 'linear',
                'order': str(order),
                'function_type': fun,
                'variables': variables_names,
                'dimension': dim_name,
                'confidence_level': confidence_level,
                'bootstrappable_dimensions': 'none'
            }
        )
        
        print("1D SF COMPLETED SUCCESSFULLY (no bootstrapping)!")
        print("="*60)
        
        return ds_binned
    
    # Normal bootstrapping case
    spacing_values = all_spacings
    print(f"Available spacings: {spacing_values}")
    gc.collect()
    
    # Initialize accumulators for weighted statistics
    sf_totals = np.zeros(n_bins, dtype=np.float64)          # Sum(sf * weight)
    weight_totals = np.zeros(n_bins, dtype=np.int32)        # Sum(weight)
    sf_sq_totals = np.zeros(n_bins, dtype=np.float64)       # Sum(sf^2 * weight)
    point_counts = np.zeros(n_bins, dtype=np.int32)         # Points per bin
    bin_density = np.zeros(n_bins, dtype=np.float32)        # Bin density distribution
    bin_status = np.zeros(n_bins, dtype=bool)               # Convergence status
    bin_bootstraps = np.ones(n_bins, dtype=np.int32) * initial_nbootstrap  # Bootstraps per bin
    
    # Initialize spacing effectiveness tracking for adaptive sampling
    bin_spacing_effectiveness = {sp: np.zeros(n_bins, dtype=np.float32) for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros(n_bins, dtype=np.int32) for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros(n_bins, dtype=np.int32) for sp in spacing_values}
    
    # Define process_spacing_data as a nested function for consistency with 2D approach
    def process_spacing_data(sp_value, bootstraps, add_to_counts=True):
        """Process structure function data for a specific spacing value."""
        if bootstraps <= 0:
            return np.zeros(n_bins, dtype=int)
                
        print(f"  Processing spacing {sp_value} with {bootstraps} bootstraps")
                
        # Run Monte Carlo simulation
        sf_results, separations = monte_carlo_simulation_1d(
            ds=ds,
            dim=dim_name,
            variables_names=variables_names,
            order=order, 
            nbootstrap=bootstraps, 
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            fun=fun, 
            spacing=sp_value,
            n_jobs=n_jobs
        )
        
        # Bin tracking for this spacing
        bin_points_added = np.zeros(n_bins, dtype=np.int32)
        
        # Process all bootstrap samples
        for b in range(len(sf_results)):
            sf = sf_results[b]
            sep = separations[b]
            
            # Create mask for valid values (not NaN)
            valid = ~np.isnan(sf) & ~np.isnan(sep)
            sf_valid = sf[valid]
            sep_valid = sep[valid]
            
            # Skip if no valid data in this bootstrap
            if len(sf_valid) == 0:
                continue
                
            # Use separation as weights
            weights = np.abs(sep_valid)
            
            # PERFORMANCE OPTIMIZATION: Use vectorized bin assignment
            # Find bin indices for each point
            bin_idx = bin_idx_func(sep_valid)
            
            # Only include points that fall within the bin boundaries
            valid_bins = (bin_idx >= 0) & (bin_idx < n_bins)
            
            # Process valid points
            if np.any(valid_bins):
                # Extract valid data
                valid_sf = sf_valid[valid_bins]
                valid_weights = weights[valid_bins]
                valid_bin_idx = bin_idx[valid_bins]
                
                # Use numpy's unique with counts for fast aggregation
                unique_bins, inverse_indices, counts = np.unique(valid_bin_idx, return_inverse=True, return_counts=True)
                
                # Process each unique bin
                for i, bin_id in enumerate(unique_bins):
                    # Valid bin check
                    if bin_id < 0 or bin_id >= n_bins:
                        continue
                    
                    # Get mask for this bin
                    bin_mask = inverse_indices == i
                    bin_count = counts[i]
                    
                    # Extract values for this bin
                    bin_sf = valid_sf[bin_mask]
                    bin_weights = valid_weights[bin_mask]
                    
                    # Count points for density calculation
                    if add_to_counts:
                        point_counts[bin_id] += bin_count
                        bin_points_added[bin_id] += bin_count
                        bin_spacing_counts[sp_value][bin_id] += bin_count
                    
                    # Calculate weighted statistics
                    weight_sum = np.sum(bin_weights)
                    if weight_sum > 0:
                        norm_weights = bin_weights / weight_sum
                        
                        # Update accumulators
                        sf_totals[bin_id] += np.sum(bin_sf * norm_weights)
                        sf_sq_totals[bin_id] += np.sum((bin_sf**2) * norm_weights)
                        weight_totals[bin_id] += 1
        
        # Clean memory
        del sf_results, separations
        gc.collect()
        
        return bin_points_added

    # Define calculate_bin_statistics as a nested function for consistency
    def calculate_bin_statistics():
        """
        Calculate current weighted means and standard deviations for all bins
        
        Returns
        -------
        tuple
            (means, stds) arrays of weighted means and standard deviations
        """
        means = np.full(n_bins, np.nan)
        stds = np.full(n_bins, np.nan)
        
        # Only calculate for bins with data
        valid_bins = weight_totals > 0
        if np.any(valid_bins):
            means[valid_bins] = sf_totals[valid_bins] / weight_totals[valid_bins]
        
        # Calculate variance and std for bins with enough samples
        valid_var_bins = weight_totals > 1
        if np.any(valid_var_bins):
            variance = np.zeros_like(sf_totals)
            variance[valid_var_bins] = (sf_sq_totals[valid_var_bins] / weight_totals[valid_var_bins]) - (means[valid_var_bins]**2)
            stds[valid_var_bins] = np.sqrt(np.maximum(0, variance[valid_var_bins]))
        
        return means, stds
    
    # Process initial bootstraps for all spacings
    print("\nINITIAL BOOTSTRAP PHASE")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    for sp_value in spacing_values:
        bin_points_added = process_spacing_data(sp_value, init_samples_per_spacing, True)
        
        # Update spacing effectiveness after processing all bootstraps
        if init_samples_per_spacing > 0:
            # Vectorized update of effectiveness
            mask = bin_points_added > 0
            if np.any(mask):
                bin_spacing_effectiveness[sp_value][mask] = bin_points_added[mask] / init_samples_per_spacing
                bin_spacing_bootstraps[sp_value][mask] += init_samples_per_spacing
    
    # Calculate bin density for adaptive sampling
    print("\nCALCULATING BIN DENSITIES")
    total_points = np.sum(point_counts)
    if total_points > 0:
        # Calculate all bin widths at once
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        
        # Vectorized density calculation
        bin_density = np.divide(point_counts, bin_widths * total_points, 
                              out=np.zeros_like(bin_density, dtype=np.float32), 
                              where=bin_widths > 0)
    
    # Normalize density
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
    
    print(f"Total points collected: {total_points}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins}")
    print(f"Maximum density bin has {np.max(point_counts)} points")
    
    # Calculate adaptive step sizes based on density
    bootstrap_steps = np.maximum(
        step_nbootstrap, 
        (step_nbootstrap * (1 + 2 * bin_density)).astype(int)
    )
    
    # Calculate initial statistics
    print("\nCALCULATING INITIAL STATISTICS")
    sf_means, sf_stds = calculate_bin_statistics()
    
    # Mark bins with too few points as converged
    low_density_mask = (point_counts <= 10) & ~bin_status
    bin_status |= low_density_mask
    print(f"Marked {np.sum(low_density_mask)} low-density bins (< 10 points) as converged")

    # Mark bins with NaN standard deviations as converged
    nan_std_mask = np.isnan(sf_stds) & ~bin_status
    bin_status |= nan_std_mask
    print(f"Marked {np.sum(nan_std_mask)} bins with NaN standard deviations as converged")

    # Mark early converged bins
    early_converged = (sf_stds <= convergence_eps) & ~bin_status & (point_counts > 10)
    bin_status |= early_converged
    print(f"Marked {np.sum(early_converged)} bins as early-converged (std <= {convergence_eps})")
    
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
        
        # Create density-ordered bin list
        bin_list = []
        bin_idxs = np.where(unconverged)[0]
        for i in bin_idxs:
            bin_list.append((i, bin_density[i]))
        
        # Sort by density (highest first)
        bin_list.sort(key=lambda x: x[1], reverse=True)
        
        # Track how many bins converged in this iteration
        bins_converged_in_iteration = 0
        max_reached_in_iteration = 0
        
        # Process bins in order of decreasing density
        for i, density in bin_list:
            # Skip if already converged
            if bin_status[i]:
                continue
                
            print(f"\nProcessing bin {i} (separation={bin_centers[i]:.4f}) - Density: {density:.4f} - " + 
                 f"Current bootstraps: {bin_bootstraps[i]} - " + 
                 f"Current std: {sf_stds[i]:.6f} - " + 
                 f"Points: {point_counts[i]}")
                 
            # Use exact bootstrap step value based on density
            step = bootstrap_steps[i]
            print(f"  Adding up to {step} more bootstraps to bin {i}")
            
            # Calculate spacing effectiveness for this bin
            spacing_effectiveness = {sp: bin_spacing_effectiveness[sp][i] for sp in spacing_values}
            
            # Sort spacings by effectiveness (highest first)
            sorted_spacings = sorted(spacing_effectiveness.items(), key=lambda x: x[1], reverse=True)
            
            # Use multi-spacing approach but more efficiently
            total_additional = 0
            remaining_step = step
            
            # Process all spacings based on their effectiveness
            total_effectiveness = sum(eff for _, eff in sorted_spacings if eff > 0)
            
            # Distribute bootstraps proportionally to effectiveness
            for sp_value, effectiveness in sorted_spacings:
                # Skip ineffective spacings
                if effectiveness <= 0: 
                    continue
                
                # Calculate proportion based on effectiveness
                if total_effectiveness > 0:
                    proportion = effectiveness / total_effectiveness
                    sp_additional = int(step * proportion)
                else:
                    sp_additional = 0
                
                sp_additional = min(sp_additional, remaining_step)
                if sp_additional <= 0:
                    continue
                
                # Process this spacing
                process_spacing_data(sp_value, sp_additional, False)
                
                # Update counters
                total_additional += sp_additional
                remaining_step -= sp_additional
                
                # Stop if we've allocated all bootstraps
                if remaining_step <= 0:
                    break
            
            # Update bootstrap counts
            bin_bootstraps[i] += total_additional
            
            # Recalculate statistics
            sf_means, sf_stds = calculate_bin_statistics()
            
            # Check for convergence or max bootstraps
            if sf_stds[i] <= convergence_eps:
                bin_status[i] = True
                print(f"  Bin {i} (separation={bin_centers[i]:.4f}) CONVERGED with std {sf_stds[i]:.6f} <= {convergence_eps}")
                bins_converged_in_iteration += 1
            elif bin_bootstraps[i] >= max_nbootstrap:
                bin_status[i] = True
                print(f"  Bin {i} (separation={bin_centers[i]:.4f}) reached MAX BOOTSTRAPS {max_nbootstrap}")
                max_reached_in_iteration += 1
        
        # Next iteration
        iteration += 1
        gc.collect()
    
    # Final convergence statistics
    converged_bins = np.sum(bin_status & (point_counts > 10))
    unconverged_bins = np.sum(~bin_status & (point_counts > 10))
    max_bootstrap_bins = np.sum((bin_bootstraps >= max_nbootstrap) & (point_counts > 10))
    
    print("\nFINAL CONVERGENCE STATISTICS:")
    print(f"  Total bins with data more than 10 points: {np.sum(point_counts > 10)}")
    print(f"  Converged bins: {converged_bins}")
    print(f"  Unconverged bins: {unconverged_bins}")
    print(f"  Bins at max bootstraps: {max_bootstrap_bins}")
    
    # Calculate confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Initialize arrays for confidence intervals
    ci_upper = np.full(n_bins, np.nan)
    ci_lower = np.full(n_bins, np.nan)
    
    # Calculate confidence intervals for valid bins
    valid_bins = ~np.isnan(sf_means) & (weight_totals > 0)
    if np.any(valid_bins):
        ci_upper[valid_bins] = sf_means[valid_bins] + z_score * sf_stds[valid_bins] / np.sqrt(weight_totals[valid_bins])
        ci_lower[valid_bins] = sf_means[valid_bins] - z_score * sf_stds[valid_bins] / np.sqrt(weight_totals[valid_bins])
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_binned = xr.Dataset(
        data_vars={
            'sf': (('bin'), sf_means),
            'sf_std': (('bin'), sf_stds),
            'ci_upper': (('bin'), ci_upper),
            'ci_lower': (('bin'), ci_lower),
            'nbootstraps': (('bin'), bin_bootstraps),
            'density': (('bin'), bin_density),
            'point_counts': (('bin'), point_counts),
            'converged': (('bin'), bin_status)
        },
        coords={
            'bin': bin_centers,
            f'{dim_name}_bins': ((f'{dim_name}_edges'), bin_edges)
        },
        attrs={
            'bin_type': 'logarithmic' if log_bins else 'linear',
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'order': str(order),
            'function_type': fun,
            'spacing_values': list(spacing_values),
            'variables': variables_names,
            'dimension': dim_name,
            'confidence_level': confidence_level,
            'bootstrappable_dimensions': ','.join(bootstrappable_dims)
        }
    )
    
    print("1D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned
#####################################################################################################################
