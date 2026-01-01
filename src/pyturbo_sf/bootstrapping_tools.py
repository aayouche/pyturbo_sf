"""Bootstrapping Tools"""

import numpy as np
import gc
import os
from joblib import Parallel, delayed
import bottleneck as bn
from scipy import stats

from .core import (
    get_boot_indexes_1d,
    get_boot_indexes_2d,
    get_boot_indexes_3d
)
from .structure_functions import (
    calculate_structure_function_1d,
    calculate_structure_function_2d,
    calculate_structure_function_3d
)
from .binning_tools import (
    _calculate_bin_density_1d,
    _calculate_bin_density_2d,
    _calculate_bin_density_3d
)
from .isotropy_tools import (
   _calculate_bin_density_polar_2d,
   _calculate_bin_density_spherical_3d
)
from .bessel_tools import (
   _bin_sf_by_radius_2d,
   _compute_energy_flux_2d,
   _calculate_wavenumber_density_2d
)
from scipy.special import jv  # Bessel function of the first kind


##################################################
# BOOTSTRAP STATISTICS
##################################################


def _compute_weighted_bootstrap_stats(bootstrap_samples, confidence_level=0.95):
    """
    Compute bootstrap statistics with proper effective sample size correction.
    
    Parameters
    ----------
    bootstrap_samples : list of dict
        Each dict contains 'mean' and 'weight' (number of points in that bootstrap)
    confidence_level : float
        Confidence level for intervals (default: 0.95)
        
    Returns
    -------
    theta_hat : float
        Point estimate (weighted mean of bootstrap means)
    std_error : float
        Bootstrap standard error with effective sample size correction
    ci_lower : float
        Lower confidence interval bound (theta_hat - z * SE)
    ci_upper : float
        Upper confidence interval bound (theta_hat + z * SE)
        
    Notes
    -----
    The standard error is computed using effective sample size:
    
    1. n_eff = (sum(w))^2 / sum(w^2)
    2. var_corrected = var_weighted * n_eff / (n_eff - 1)  [Bessel correction]
    3. SE = sqrt(var_corrected / n_eff)
    
    This properly accounts for:
    - Unequal weights in bootstrap samples
    - Bias correction (Bessel's correction)
    - Variance of the mean (not variance of data)
    """
    from scipy import stats
    
    boot_means = np.array([s['mean'] for s in bootstrap_samples])
    boot_weights = np.array([s['weight'] for s in bootstrap_samples], dtype=np.float64)
    
    # Handle edge cases
    if len(boot_means) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    if len(boot_means) == 1:
        return boot_means[0], np.nan, np.nan, np.nan
    
    # Point estimate: weighted mean
    sum_w = np.sum(boot_weights)
    theta_hat = np.sum(boot_weights * boot_means) / sum_w
    
    # Step 3.1: Effective sample size
    sum_w_sq = np.sum(boot_weights ** 2)
    n_eff = (sum_w ** 2) / sum_w_sq
    
    # Step 3.2: Corrected weighted variance
    # First compute weighted variance
    weighted_var = np.sum(boot_weights * (boot_means - theta_hat) ** 2) / sum_w
    
    # Apply Bessel correction: var_corrected = var_weighted * n_eff / (n_eff - 1)
    if n_eff > 1:
        var_corrected = weighted_var * n_eff / (n_eff - 1)
    else:
        var_corrected = weighted_var
    
    # Step 3.3: Standard error = sqrt(var_corrected / n_eff)
    if n_eff > 0:
        std_error = np.sqrt(var_corrected / n_eff)
    else:
        std_error = np.nan
    
    # Confidence intervals: theta_hat ± z * SE
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    ci_lower = theta_hat - z_score * std_error
    ci_upper = theta_hat + z_score * std_error
    
    return theta_hat, std_error, ci_lower, ci_upper


##################################################1D#####################################################################################
def run_bootstrap_sf_1d(args):
    """Standalone bootstrap function for parallel processing."""
    ds, dim, variables_names, order, fun, nb, spacing, num_bootstrappable, boot_indexes, bootsize, conditioning_var, conditioning_bins = args
    results, separations, pair_counts = calculate_structure_function_1d(
        ds=ds, dim=dim, variables_names=variables_names, order=order, fun=fun,
        nb=nb, spacing=spacing, num_bootstrappable=num_bootstrappable,
        boot_indexes=boot_indexes, bootsize=bootsize, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins
    )
    return results, separations, pair_counts

def monte_carlo_simulation_1d(ds, dim, variables_names, order, nbootstrap, bootsize, 
                             num_bootstrappable, all_spacings, boot_indexes,
                             fun='scalar', spacing=None, n_jobs=-1, backend='threading',
                             conditioning_var=None, conditioning_bins=None, seed=None):
    """
    Run Monte Carlo simulation for structure function calculation with multiple bootstrap samples.
    
    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. If None, uses random state.
    """
    # Create random generator (seeded if provided)
    rng = np.random.default_rng(seed)
    
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, separations, pair_counts = calculate_structure_function_1d(
            ds=ds,
            dim=dim,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            conditioning_var=conditioning_var,
            conditioning_bins=conditioning_bins
        )
        return [results], [separations], [pair_counts]
    
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
        results, separations, pair_counts = calculate_structure_function_1d(
            ds=ds,
            dim=dim,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            conditioning_var=conditioning_var,
            conditioning_bins=conditioning_bins
        )
        return [results], [separations], [pair_counts]
    
    # Generate random indices for the bootstrappable dimension (seeded)
    random_indices = rng.choice(indexes[dim].shape[1], size=nbootstrap)
    
    
    # Calculate optimal batch size based on number of jobs and bootstraps
    if n_jobs < 0:  # All negative n_jobs values
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
            boot_indexes, bootsize, conditioning_var, conditioning_bins
        )
        all_args.append(args)
    
    # Run simulations in parallel using the module-level function
    results = Parallel(n_jobs=n_jobs, verbose=0,  batch_size=batch_size, backend=backend)(
        delayed(run_bootstrap_sf_1d)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    separations = [r[1] for r in results]
    pair_counts_results = [r[2] for r in results]
    
    return sf_results, separations, pair_counts_results

def _process_spacing_data_batch_1d(sf_results, separations, bin_edges, n_bins, 
                                   bin_accumulators, point_counts, bin_spacing_counts,
                                   sp_value, bin_list, add_to_counts=True,
                                   pair_counts_results=None):
    """
    Process structure function data for a specific spacing value with batch processing.
    
    FIXED: Now records each bootstrap mean independently rather than incrementally.
    Each bootstrap iteration produces one mean estimate per bin.
    Uses pair_counts for proper weighting when combining separations into bins.
    """
    # Create a set of target bins for fast lookup
    target_bins = set(bin_list)
    
    # Function to calculate bin indices
    def bin_idx_func(values):
        return np.clip(np.digitize(values, bin_edges) - 1, 0, n_bins - 1)
    
    # Process each bootstrap sample INDEPENDENTLY
    for b in range(len(sf_results)):
        sf = sf_results[b]
        sep = separations[b]
        # Get pair counts for this bootstrap (if available)
        pc = pair_counts_results[b] if pair_counts_results is not None else None
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(sep)
        sf_valid = sf[valid]
        sep_valid = sep[valid]
        pc_valid = pc[valid] if pc is not None else None
        
        if len(sf_valid) == 0:
            continue
        
        # Find bin indices
        bin_idx = bin_idx_func(sep_valid)
        
        # Temporary accumulators for THIS bootstrap only
        boot_accum = {}
        
        # Accumulate data for this bootstrap
        for idx in range(len(sf_valid)):
            bin_id = bin_idx[idx]
            if bin_id not in target_bins:
                continue
            
            if bin_id not in boot_accum:
                boot_accum[bin_id] = {'weighted_sum': 0.0, 'total_weight': 0.0, 'count': 0}
            
            # Use pair_counts as weights if available, otherwise use 1
            weight = float(pc_valid[idx]) if pc_valid is not None else 1.0
            boot_accum[bin_id]['weighted_sum'] += sf_valid[idx] * weight
            boot_accum[bin_id]['total_weight'] += weight
            boot_accum[bin_id]['count'] += 1
        
        # Record the bootstrap mean for each bin that received data
        for bin_id, data in boot_accum.items():
            if data['total_weight'] > 0:
                boot_mean = data['weighted_sum'] / data['total_weight']
                
                # Initialize main accumulator if needed
                if bin_id not in bin_accumulators:
                    bin_accumulators[bin_id] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                
                # Add to global accumulator for overall mean
                bin_accumulators[bin_id]['weighted_sum'] += data['weighted_sum']
                bin_accumulators[bin_id]['total_weight'] += data['total_weight']
                bin_accumulators[bin_id]['bootstrap_samples'].append({
                    'mean': boot_mean,
                    'weight': data['total_weight']
                })
                
                # Update counts (only when add_to_counts is True)
                if add_to_counts:
                    point_counts[bin_id] += data['count']
                    bin_spacing_counts[sp_value][bin_id] += data['count']
    
    return bin_accumulators, point_counts, bin_spacing_counts

def _calculate_bootstrap_statistics_1d(bin_accumulators, n_bins,
                                       confidence_level=0.95):
    """
    Calculate weighted means, bootstrap standard errors, and CIs for 1D bins.
    
    Parameters
    ----------
    bin_accumulators : dict
        Accumulator dictionary with bin indices as keys
    n_bins : int
        Number of bins
    confidence_level : float
        Confidence level for intervals
        
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Bootstrap standard errors
    ci_lower : array
        Lower confidence interval bounds
    ci_upper : array
        Upper confidence interval bounds
    """
    sf_means = np.full(n_bins, np.nan)
    sf_stds = np.full(n_bins, np.nan)
    ci_lower = np.full(n_bins, np.nan)
    ci_upper = np.full(n_bins, np.nan)
    
    for j, acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            # Bootstrap standard error and CIs
            if len(acc['bootstrap_samples']) > 1:
                sf_means[j], sf_stds[j], ci_lower[j], ci_upper[j] = \
                    _compute_weighted_bootstrap_stats(
                        acc['bootstrap_samples'], 
                        confidence_level=confidence_level
                    )
            else:
                # Fall back to simple weighted mean if only one sample
                sf_means[j] = acc['weighted_sum'] / acc['total_weight']
                sf_stds[j] = np.nan
    
    return sf_means, sf_stds, ci_lower, ci_upper


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
            sp_bootstraps = 0 #remaining // len([s for s, e in sorted_spacings if e >= 0])
        
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
                                  boot_indexes, n_jobs, backend, conditioning_var=None, conditioning_bins=None,
                                  confidence_level=0.95, seed=None):
    """
    Run adaptive bootstrap loop for 1D structure function binning.
    
    This is the main workhorse function that handles the iterative
    bootstrap refinement process.
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility.
    """
    n_bins = bins_config['n_bins']
    
    # Initialize result arrays
    sf_means = np.full(n_bins, np.nan)
    sf_stds = np.full(n_bins, np.nan)
    ci_lower = np.full(n_bins, np.nan)
    ci_upper = np.full(n_bins, np.nan)
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
    
    for sp_idx, sp_value in enumerate(spacing_values):
        if init_samples_per_spacing <= 0:
            continue
            
        print(f"  Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Derive per-spacing seed for reproducibility
        sp_seed = (seed + sp_idx) if seed is not None else None
        
        # Run Monte Carlo simulation
        sf_results, separations, pair_counts_results = monte_carlo_simulation_1d(
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
            backend=backend,
            conditioning_var=conditioning_var,
            conditioning_bins=conditioning_bins,
            seed=sp_seed
        )
        
        # Process the results
        _process_spacing_data_batch_1d(
            sf_results, separations, bins_config['bin_edges'], n_bins,
            bin_accumulators, point_counts, bin_spacing_counts,
            sp_value, all_bins, add_to_counts=True,
            pair_counts_results=pair_counts_results
        )
        
        # Update effectiveness
        _update_spacing_effectiveness_1d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value, all_bins,
            init_samples_per_spacing
        )
        
        # Clean memory
        del sf_results, separations, pair_counts_results
        gc.collect()
    
    # Calculate statistics from accumulators
    sf_means, sf_stds, ci_lower, ci_upper = _calculate_bootstrap_statistics_1d(
        bin_accumulators, n_bins, confidence_level=confidence_level
    )
    
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
                sf_results, separations, pair_counts_results = monte_carlo_simulation_1d(
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
                    backend=backend,
                    conditioning_var=conditioning_var,
                    conditioning_bins=conditioning_bins
                )
                
                # Process the results (accumulate counts)
                _process_spacing_data_batch_1d(
                    sf_results, separations, bins_config['bin_edges'], n_bins,
                    bin_accumulators, point_counts, bin_spacing_counts,
                    sp_value, bin_list, add_to_counts=True,
                    pair_counts_results=pair_counts_results
                )
                
                # Update effectiveness
                _update_spacing_effectiveness_1d(
                    bin_spacing_effectiveness, bin_spacing_counts,
                    bin_spacing_bootstraps, sp_value, bin_list,
                    sp_bootstraps
                )
                
                # Clean memory
                del sf_results, separations, pair_counts_results
                gc.collect()
            
            # Update bootstrap counts and check convergence
            for j in bin_list:
                bin_bootstraps[j] += step
                
                # Recalculate statistics for this bin
                if j in bin_accumulators:
                    acc = bin_accumulators[j]
                    if acc['total_weight'] > 0:
                        if len(acc['bootstrap_samples']) > 1:
                            sf_means[j], sf_stds[j], ci_lower[j], ci_upper[j] = \
                                _compute_weighted_bootstrap_stats(
                                    acc['bootstrap_samples'],
                                    confidence_level=confidence_level,
                                    
                                )
                        else:
                            sf_means[j] = acc['weighted_sum'] / acc['total_weight']
                
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
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'point_counts': point_counts,
        'bin_density': bin_density,
        'bin_status': bin_status,
        'bin_bootstraps': bin_bootstraps,
        'spacing_values': spacing_values
    }

#########################################################################################################################################

##################################################2D#####################################################################################

def run_bootstrap_sf_2d(args):
    """Standalone bootstrap function for parallel processing in 2D."""
    ds, dims, variables_names, order, fun, nbx, nby, spacing, num_bootstrappable, bootstrappable_dims, boot_indexes, time_dims, conditioning_var, conditioning_bins = args
    results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
        ds=ds, dims=dims, variables_names=variables_names, order=order, fun=fun,
        nbx=nbx, nby=nby, spacing=spacing, num_bootstrappable=num_bootstrappable,
        bootstrappable_dims=bootstrappable_dims, boot_indexes=boot_indexes, time_dims=time_dims, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins
    )
    return results, dx_vals, dy_vals, pair_counts

def monte_carlo_simulation_2d(ds, dims, variables_names, order, nbootstrap, bootsize, 
                            num_bootstrappable, all_spacings, boot_indexes, bootstrappable_dims,
                            fun='longitudinal', spacing=None, n_jobs=-1, backend='threading', 
                            time_dims=None, conditioning_var=None, conditioning_bins=None, seed=None):
    """
    Run Monte Carlo simulation for structure function calculation with multiple bootstrap samples.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing velocity components and/or scalar fields
    dims : list
        List of dimension names
    variables_names : list
        List of variable names to use, depends on function type
    order : int or tuple
        Order(s) of the structure function
    nbootstrap : int
        Number of bootstrap samples
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    num_bootstrappable : int
        Number of bootstrappable dimensions
    all_spacings : list
        List of all spacing values
    boot_indexes : dict
        Dictionary with spacing values as keys and boot indexes as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
    fun : str, optional
        Type of structure function
    spacing : int or dict, optional
        Spacing value to use
    n_jobs : int, optional
        Number of jobs for parallel processing
    backend : str, optional
        Backend for parallel processing
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list, list, list
        Lists of structure function values, DX values, DY values
    """
    # Create random generator (seeded if provided)
    rng = np.random.default_rng(seed)
    
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
            ds=ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            time_dims=time_dims,
            conditioning_var=conditioning_var,
            conditioning_bins=conditioning_bins
        )
        return [results], [dx_vals], [dy_vals], [pair_counts]
    
    # Use default spacing of 1 if None provided
    if spacing is None:
        sp_value = 1
    # Convert dict spacing to single value if needed
    elif isinstance(spacing, dict):
        # Get the spacing for a bootstrappable dimension
        for dim in bootstrappable_dims:
            if dim in spacing:
                sp_value = spacing[dim]
                break
        else:
            sp_value = 1  # Default if no matching dimension found
    else:
        sp_value = spacing
    

    
    # Get boot indexes for the specified spacing
    if sp_value in boot_indexes:
        indexes = boot_indexes[sp_value]
    else:
        # Calculate boot indexes on-the-fly
        data_shape = dict(ds.sizes)
        indexes = get_boot_indexes_2d(dims, data_shape, bootsize, all_spacings, boot_indexes, 
                                     bootstrappable_dims, num_bootstrappable, sp_value)
    
    # Check if we have valid indexes
    if num_bootstrappable == 1:
        bootstrap_dim = bootstrappable_dims[0]
        valid_indices = bootstrap_dim in indexes and indexes[bootstrap_dim].shape[1] > 0
        if not valid_indices:
            print(f"Warning: No valid indices for dimension {bootstrap_dim} with spacing {sp_value}.")
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims,
                conditioning_var=conditioning_var,
                conditioning_bins=conditioning_bins
            )
            return [results], [dx_vals], [dy_vals], [pair_counts]
    else:
        # Two bootstrappable dimensions - check both
        valid_y_indices = dims[0] in indexes and indexes[dims[0]].shape[1] > 0
        valid_x_indices = dims[1] in indexes and indexes[dims[1]].shape[1] > 0
        
        if not valid_y_indices or not valid_x_indices:
            print("Warning: Not enough valid indices for bootstrapping with current spacing.")
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals, pair_counts = calculate_structure_function_2d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims,
                conditioning_var=conditioning_var,
                conditioning_bins=conditioning_bins
            )
            return [results], [dx_vals], [dy_vals], [pair_counts]
    
    # Create all argument arrays for parallel processing
    all_args = []
    
    # Prepare parameters based on bootstrappable dimensions
    if num_bootstrappable == 1:
        # One bootstrappable dimension - only randomize that dimension
        bootstrap_dim = bootstrappable_dims[0]
        
        # Generate random indices for the bootstrappable dimension (seeded)
        random_indices = rng.choice(indexes[bootstrap_dim].shape[1], size=nbootstrap)
        
        # Create arguments for all bootstrap iterations
        for j in range(nbootstrap):
            if bootstrap_dim == dims[1]:  # x-dimension
                args = (
                    ds, dims, variables_names, order, fun,
                    random_indices[j], 0, sp_value, num_bootstrappable,
                    bootstrappable_dims, boot_indexes, time_dims, conditioning_var, conditioning_bins
                )
            else:  # y-dimension
                args = (
                    ds, dims, variables_names, order, fun,
                    0, random_indices[j], sp_value, num_bootstrappable,
                    bootstrappable_dims, boot_indexes, time_dims, conditioning_var, conditioning_bins
                )
            all_args.append(args)
            
    else:
        # Two bootstrappable dimensions - randomize both
        # Generate random indices for both dimensions (seeded)
        nby = rng.choice(indexes[dims[0]].shape[1], size=nbootstrap) 
        nbx = rng.choice(indexes[dims[1]].shape[1], size=nbootstrap)
        
        # Create arguments for all bootstrap iterations
        for j in range(nbootstrap):
            args = (
                ds, dims, variables_names, order, fun,
                nbx[j], nby[j], sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims, conditioning_var, conditioning_bins
            )
            all_args.append(args)
    
    # Calculate optimal batch size based on number of jobs and bootstraps
    if n_jobs < 0:  # All negative n_jobs values
        total_cpus = os.cpu_count()
        if n_jobs == -1:  # Special case: use all CPUs
            n_workers = total_cpus
        else:  # Use (all CPUs - |n_jobs| - 1)
            n_workers = max(1, total_cpus + n_jobs + 1)  # +1 because -2 means all except 1
    else:
        n_workers = n_jobs
    
    batch_size = max(10, nbootstrap//(n_workers*2))
    
    # Run simulations in parallel using module-level function
    results = Parallel(n_jobs=n_jobs, verbose=0, batch_size=batch_size, backend=backend)(
        delayed(run_bootstrap_sf_2d)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    dx_vals = [r[1] for r in results]
    dy_vals = [r[2] for r in results]
    pair_counts_results = [r[3] for r in results]
    
    return sf_results, dx_vals, dy_vals, pair_counts_results    

def _process_bootstrap_batch_2d(sf_results, dx_vals, dy_vals, bins_x, bins_y, 
                               bin_accumulators, target_bins, point_counts=None,
                               spacing_counts=None, sp_value=None, add_to_counts=True,
                               pair_counts_results=None):
    """
    Process a batch of bootstrap results for 2D Cartesian binning.
    
    FIXED: Now records each bootstrap mean independently rather than incrementally.
    Each bootstrap iteration produces one mean estimate per bin.
    
    Parameters
    ----------
    sf_results : list
        Structure function results from monte carlo simulation
    dx_vals, dy_vals : list
        Separation distances for each bootstrap
    bins_x, bins_y : array
        Bin edges for x and y dimensions
    bin_accumulators : dict
        Accumulator dictionary with keys (j, i)
    target_bins : set
        Set of (j, i) tuples for bins to process
    point_counts : array, optional
        Array to update with point counts
    spacing_counts : dict, optional
        Dictionary of spacing counts to update
    sp_value : int, optional
        Current spacing value
    add_to_counts : bool
        Whether to update counts
    pair_counts_results : list, optional
        List of pair counts arrays from structure function calculations.
        
    Returns
    -------
    updated_bins : set
        Set of bins that were updated
    """
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1
    updated_bins = set()
    
    # Create set of target bin IDs for fast lookup
    target_bin_ids = {j * n_bins_x + i for j, i in target_bins}
    
    # Process each bootstrap sample INDEPENDENTLY
    for b in range(len(sf_results)):
        sf = sf_results[b]
        dx = dx_vals[b]
        dy = dy_vals[b]
        # Get pair counts for this bootstrap (if available)
        pc = pair_counts_results[b] if pair_counts_results is not None else None
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid):
            continue
            
        sf_valid = sf[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]
        pc_valid = pc[valid] if pc is not None else None
        
        # Vectorized bin assignment
        x_indices = np.clip(np.digitize(dx_valid, bins_x) - 1, 0, n_bins_x - 1)
        y_indices = np.clip(np.digitize(dy_valid, bins_y) - 1, 0, n_bins_y - 1)
        
        # Create unique bin IDs
        bin_ids = y_indices * n_bins_x + x_indices
        
        # Temporary accumulators for THIS bootstrap only
        boot_accum = {}
        
        # Accumulate data for this bootstrap
        for idx in range(len(sf_valid)):
            bin_id = bin_ids[idx]
            if bin_id not in target_bin_ids:
                continue
                
            j, i = divmod(bin_id, n_bins_x)
            bin_key = (j, i)
            value = sf_valid[idx]
            # Get actual pair count for this separation (or 1 if not available)
            # This is the weight for combining SF means from different separations
            pair_count = pc_valid[idx] if pc_valid is not None else 1
            
            if bin_key not in boot_accum:
                boot_accum[bin_key] = {'weighted_sum': 0.0, 'total_weight': 0.0, 'pair_count': 0}
            
            # Weight by pair_count since value is a mean over pair_count origins
            boot_accum[bin_key]['weighted_sum'] += value * pair_count
            boot_accum[bin_key]['total_weight'] += pair_count
            boot_accum[bin_key]['pair_count'] += pair_count
        
        # Record the bootstrap mean for each bin that received data
        for bin_key, data in boot_accum.items():
            if data['total_weight'] > 0:
                boot_mean = data['weighted_sum'] / data['total_weight']
                
                # Initialize main accumulator if needed
                if bin_key not in bin_accumulators:
                    bin_accumulators[bin_key] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                
                # Add to global accumulator for overall mean
                bin_accumulators[bin_key]['weighted_sum'] += data['weighted_sum']
                bin_accumulators[bin_key]['total_weight'] += data['total_weight']
                bin_accumulators[bin_key]['bootstrap_samples'].append({
                    'mean': boot_mean,
                    'weight': data['total_weight']
                })
                
                updated_bins.add(bin_key)
                
                # Update counts (only when add_to_counts is True)
                if add_to_counts:
                    j, i = bin_key
                    if point_counts is not None:
                        point_counts[j, i] += data['pair_count']  # Use actual pair count!
                    if spacing_counts is not None and sp_value is not None:
                        spacing_counts[sp_value][j, i] += data['pair_count']  # Use actual pair count!
        
    return updated_bins


def _process_bootstrap_batch_polar_2d(sf_results, dx_vals, dy_vals, r_bins, theta_bins,
                                 bin_accumulators, angular_accumulators, target_r_bins,
                                 point_counts=None, spacing_counts=None, sp_value=None,
                                 add_to_counts=True, pair_counts_results=None):
    """
    Process a batch of bootstrap results for polar binning.
    
    FIXED: Now records each bootstrap mean independently rather than incrementally.
    Each bootstrap iteration produces one mean estimate per radial bin.
    
    Parameters
    ----------
    sf_results : list
        Structure function results
    dx_vals, dy_vals : list
        Separation distances
    r_bins : array
        Radial bin edges
    theta_bins : array
        Angular bin edges
    bin_accumulators : dict
        Radial accumulator with keys as r_idx
    angular_accumulators : dict
        Angular accumulator with keys as (theta_idx, r_idx)
    target_r_bins : set
        Set of radial bin indices to process
    point_counts : array, optional
        Array to update with counts
    spacing_counts : dict, optional
        Dictionary of spacing counts
    sp_value : int, optional
        Current spacing value
    add_to_counts : bool
        Whether to update counts
    pair_counts_results : list, optional
        List of pair counts arrays from structure function calculations.
        Each element corresponds to a bootstrap iteration.
        
    Returns
    -------
    updated_r_bins : set
        Set of r bins that were updated
    """
    n_bins_r = len(r_bins) - 1
    n_bins_theta = len(theta_bins) - 1
    updated_r_bins = set()
    
    # Process each bootstrap sample INDEPENDENTLY
    for b in range(len(sf_results)):
        sf = sf_results[b]
        dx = dx_vals[b]
        dy = dy_vals[b]
        # Get pair counts for this bootstrap (if available)
        pc = pair_counts_results[b] if pair_counts_results is not None else None
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid):
            continue
        
        # Get original indices of valid entries (needed for pair_counts lookup)
        valid_indices = np.where(valid)[0]
            
        sf_valid = sf[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]
        pc_valid = pc[valid] if pc is not None else None
        
        # Convert to polar coordinates
        r_valid = np.sqrt(dx_valid**2 + dy_valid**2)
        theta_valid = np.arctan2(dy_valid, dx_valid)
        
        # Create bin indices
        r_indices = np.clip(np.digitize(r_valid, r_bins) - 1, 0, n_bins_r - 1)
        theta_indices = np.clip(np.digitize(theta_valid, theta_bins) - 1, 0, n_bins_theta - 1)
        
        # Temporary accumulators for THIS bootstrap only
        boot_accum_r = {}  # For radial bins
        boot_accum_angular = {}  # For angular bins
        
        # Accumulate data for this bootstrap
        for idx in range(len(sf_valid)):
            r_idx = r_indices[idx]
            if r_idx not in target_r_bins:
                continue
            
            theta_idx = theta_indices[idx]
            value = sf_valid[idx]
            # Get actual pair count for this separation (or 1 if not available)
            # This is the weight for combining SF means from different separations
            pair_count = pc_valid[idx] if pc_valid is not None else 1
            
            # Radial accumulator - weight by pair_count since value is a mean over pair_count origins
            if r_idx not in boot_accum_r:
                boot_accum_r[r_idx] = {'weighted_sum': 0.0, 'total_weight': 0.0, 'pair_count': 0}
            boot_accum_r[r_idx]['weighted_sum'] += value * pair_count  # Weight by pair count!
            boot_accum_r[r_idx]['total_weight'] += pair_count          # Weight by pair count!
            boot_accum_r[r_idx]['pair_count'] += pair_count
            
            # Angular accumulator - also weight by pair_count
            angular_key = (theta_idx, r_idx)
            if angular_key not in boot_accum_angular:
                boot_accum_angular[angular_key] = {'weighted_sum': 0.0, 'total_weight': 0.0}
            boot_accum_angular[angular_key]['weighted_sum'] += value * pair_count
            boot_accum_angular[angular_key]['total_weight'] += pair_count
        
        # Record the bootstrap mean for each radial bin that received data
        for r_idx, data in boot_accum_r.items():
            if data['total_weight'] > 0:
                boot_mean = data['weighted_sum'] / data['total_weight']
                
                # Initialize main accumulator if needed
                if r_idx not in bin_accumulators:
                    bin_accumulators[r_idx] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                
                # Add to global accumulator for overall mean
                bin_accumulators[r_idx]['weighted_sum'] += data['weighted_sum']
                bin_accumulators[r_idx]['total_weight'] += data['total_weight']
                bin_accumulators[r_idx]['bootstrap_samples'].append({
                    'mean': boot_mean,
                    'weight': data['total_weight']
                })
                
                updated_r_bins.add(r_idx)
                
                # Update counts (only when add_to_counts is True)
                if add_to_counts:
                    if point_counts is not None:
                        point_counts[r_idx] += data['pair_count']  # Use actual pair count!
                    if spacing_counts is not None and sp_value is not None:
                        spacing_counts[sp_value][r_idx] += data['pair_count']  # Use actual pair count!
        
        # Update angular accumulators (these don't need bootstrap samples)
        for angular_key, data in boot_accum_angular.items():
            if data['total_weight'] > 0:
                if angular_key not in angular_accumulators:
                    angular_accumulators[angular_key] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0
                    }
                angular_accumulators[angular_key]['weighted_sum'] += data['weighted_sum']
                angular_accumulators[angular_key]['total_weight'] += data['total_weight']
    
    return updated_r_bins
    
def _process_bootstrap_batch_flux_2d(sf_results, dx_vals, dy_vals,
                                      config, k_accumulators, angular_accumulators,
                                      r_accumulators, target_k_set, point_counts,
                                      spacing_counts, sp_value, update_counts):
    """
    Process a batch of bootstrap results for energy flux computation.
    
    This function:
    1. Bins SF values by radius to get angle-averaged SF̃(r)
    2. Computes energy flux Π(K) = -K/2 ∫ SF̃(r) J₁(Kr) dr
    
    Parameters
    ----------
    sf_results : list of arrays
        Structure function values from each bootstrap.
    dx_vals, dy_vals : list of arrays
        Separation distances from each bootstrap.
    config : dict
        Configuration with 'k', 'r_centers', 'dr', 'theta_bins', etc.
    k_accumulators : dict
        Accumulators for wavenumber statistics (energy flux).
    angular_accumulators : dict
        Accumulators for angular-wavenumber statistics.
    r_accumulators : dict
        Accumulators for radial SF statistics.
    target_k_set : set
        Set of wavenumber indices to process.
    point_counts : array or None
        Point counts to update (if update_counts is True).
    spacing_counts : dict
        Counts per spacing.
    sp_value : int
        Current spacing value.
    update_counts : bool
        Whether to update point counts.
    """
    k = config['k']
    n_k = len(k)
    r_edges = config['r_edges']
    r_centers = config['r_centers']
    dr = config['dr']
    n_r = config['n_r']
    theta_bins = config['theta_bins']
    n_theta = config['n_bins_theta']
    
    for boot_idx, (sf, dx, dy) in enumerate(zip(sf_results, dx_vals, dy_vals)):
        # Filter valid data
        valid_mask = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid_mask):
            continue
        
        valid_sf = sf[valid_mask]
        valid_dx = dx[valid_mask]
        valid_dy = dy[valid_mask]
        
        r = np.sqrt(valid_dx**2 + valid_dy**2)
        theta = np.arctan2(valid_dy, valid_dx)
        
        # Step 1: Bin SF by radius to get angle-averaged SF̃(r)
        r_indices = np.clip(np.digitize(r, r_edges) - 1, 0, n_r - 1)
        theta_indices = np.clip(np.digitize(theta, theta_bins) - 1, 0, n_theta - 1)
        
        # Compute SF̃(r) for each radial bin
        sf_r = np.full(n_r, np.nan)
        counts_r = np.zeros(n_r, dtype=np.int_)
        
        for r_idx in range(n_r):
            mask = r_indices == r_idx
            if np.sum(mask) > 0:
                sf_r[r_idx] = np.mean(valid_sf[mask])
                counts_r[r_idx] = np.sum(mask)
        
        # Store radial SF in accumulators
        for r_idx in range(n_r):
            if counts_r[r_idx] > 0:
                if r_idx not in r_accumulators:
                    r_accumulators[r_idx] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                r_accumulators[r_idx]['weighted_sum'] += sf_r[r_idx] * counts_r[r_idx]
                r_accumulators[r_idx]['total_weight'] += counts_r[r_idx]
                r_accumulators[r_idx]['bootstrap_samples'].append({
                    'mean': sf_r[r_idx],
                    'weight': counts_r[r_idx]
                })
        
        # Step 2: Compute energy flux Π(K) = -K/2 ∫ SF̃(r) J₁(Kr) dr
        valid_r_mask = ~np.isnan(sf_r)
        if not np.any(valid_r_mask):
            continue
        
        sf_valid = sf_r[valid_r_mask]
        r_valid = r_centers[valid_r_mask]
        dr_valid = dr[valid_r_mask]
        
        # Total valid points for this bootstrap (used for point counts per k)
        total_valid_points = int(np.sum(counts_r))
        
        # Compute J₁(kr) for all (k, r) pairs
        kr = np.outer(k, r_valid)  # (n_k, n_valid_r)
        J1_values = jv(1, kr)
        
        # Compute integral for each wavenumber
        # Π(K) = -K/2 Σᵢ SF̃(rᵢ) J₁(K·rᵢ) Δrᵢ
        integral = np.sum(J1_values * sf_valid * dr_valid, axis=1)
        energy_flux = -k / 2.0 * integral
        
        # Update flux accumulators
        for k_idx in target_k_set:
            if k_idx >= n_k:
                continue
            
            flux_val = energy_flux[k_idx]
            if np.isnan(flux_val):
                continue
            
            # Weight by number of valid radial bins
            weight = np.sum(valid_r_mask)
            
            if k_idx not in k_accumulators:
                k_accumulators[k_idx] = {
                    'weighted_sum': 0.0,
                    'total_weight': 0.0,
                    'bootstrap_samples': []
                }
            
            acc = k_accumulators[k_idx]
            acc['weighted_sum'] += flux_val * weight
            acc['total_weight'] += weight
            acc['bootstrap_samples'].append({
                'mean': flux_val,
                'weight': weight
            })
            
            # Update point counts per wavenumber
            if update_counts and point_counts is not None:
                point_counts[k_idx] += total_valid_points
            if spacing_counts is not None and sp_value is not None:
                spacing_counts[sp_value][k_idx] += int(weight)
        
        # Step 3: Compute angular flux distribution
        for theta_idx in range(n_theta):
            theta_mask_all = theta_indices == theta_idx
            if not np.any(theta_mask_all):
                continue
            
            # Bin by radius within this angular sector
            sf_r_theta = np.full(n_r, np.nan)
            for r_idx in range(n_r):
                combined_mask = theta_mask_all & (r_indices == r_idx)
                if np.sum(combined_mask) > 0:
                    sf_r_theta[r_idx] = np.mean(valid_sf[combined_mask])
            
            # Compute flux for this angular sector
            valid_r_theta = ~np.isnan(sf_r_theta)
            if not np.any(valid_r_theta):
                continue
            
            sf_theta_valid = sf_r_theta[valid_r_theta]
            r_theta_valid = r_centers[valid_r_theta]
            dr_theta_valid = dr[valid_r_theta]
            
            kr_theta = np.outer(k, r_theta_valid)
            J1_theta = jv(1, kr_theta)
            integral_theta = np.sum(J1_theta * sf_theta_valid * dr_theta_valid, axis=1)
            flux_theta = -k / 2.0 * integral_theta
            
            for k_idx in target_k_set:
                if k_idx >= n_k:
                    continue
                
                flux_val = flux_theta[k_idx]
                if np.isnan(flux_val):
                    continue
                
                key = (theta_idx, k_idx)
                if key not in angular_accumulators:
                    angular_accumulators[key] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                
                weight = np.sum(valid_r_theta)
                angular_accumulators[key]['weighted_sum'] += flux_val * weight
                angular_accumulators[key]['total_weight'] += weight
                angular_accumulators[key]['bootstrap_samples'].append({
                    'mean': flux_val,
                    'weight': weight
                })

def _calculate_bootstrap_statistics_2d(bin_accumulators, bin_shape):
    """
    Calculate weighted means and bootstrap standard errors for 2D bins.
    
    Parameters
    ----------
    bin_accumulators : dict
        Accumulator dictionary with keys (j, i)
    bin_shape : tuple
        Shape of output arrays (ny, nx)
        
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Bootstrap standard errors
    """
    ny, nx = bin_shape
    sf_means = np.full((ny, nx), np.nan)
    sf_stds = np.full((ny, nx), np.nan)
    
    for (j, i), acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            # Bootstrap standard error
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                boot_weights = np.array([s['weight'] for s in acc['bootstrap_samples']])
                
                # Weighted mean
                sf_means[j, i] = np.average(boot_means, weights=boot_weights)
                
                # Weighted std
                weighted_var = np.average((boot_means - sf_means[j, i])**2, weights=boot_weights)
                sf_stds[j, i] = np.sqrt(weighted_var)
            else:
                sf_means[j, i] = acc['weighted_sum'] / acc['total_weight']
                sf_stds[j, i] = np.nan
    
    return sf_means, sf_stds


def _calculate_bootstrap_statistics_polar_2d(bin_accumulators, angular_accumulators, 
                                             n_bins_r, n_bins_theta,
                                             confidence_level=0.95):
    """
    Calculate statistics for polar binning with CI support.
    
    Returns
    -------
    sf_means : array
        Radial means
    sf_stds : array
        Radial standard errors
    ci_lower : array
        Lower confidence interval bounds
    ci_upper : array
        Upper confidence interval bounds
    sfr : array
        Angular-radial structure function
    sfr_counts : array
        Counts for angular-radial bins
    """
    sf_means = np.full(n_bins_r, np.nan)
    sf_stds = np.full(n_bins_r, np.nan)
    ci_lower = np.full(n_bins_r, np.nan)
    ci_upper = np.full(n_bins_r, np.nan)
    sfr = np.full((n_bins_theta, n_bins_r), np.nan)
    sfr_counts = np.zeros((n_bins_theta, n_bins_r), dtype=np.int_)
    
    # Radial statistics
    for r_idx, acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            if len(acc['bootstrap_samples']) > 1:
                sf_means[r_idx], sf_stds[r_idx], ci_lower[r_idx], ci_upper[r_idx] = \
                    _compute_weighted_bootstrap_stats(
                        acc['bootstrap_samples'],
                        confidence_level=confidence_level
                    )
            else:
                sf_means[r_idx] = acc['weighted_sum'] / acc['total_weight']
                sf_stds[r_idx] = np.nan
    
    # Angular-radial matrix
    for (theta_idx, r_idx), acc in angular_accumulators.items():
        if acc['total_weight'] > 0:
            sfr[theta_idx, r_idx] = acc['weighted_sum'] / acc['total_weight']
            sfr_counts[theta_idx, r_idx] = int(acc['total_weight'])
    
    return sf_means, sf_stds, ci_lower, ci_upper, sfr, sfr_counts
    
    
def _calculate_bootstrap_statistics_flux_2d(k_accumulators, angular_accumulators, r_accumulators,
                                             n_k, n_theta, n_r,
                                             confidence_level=0.95):
    """
    Calculate statistics from energy flux accumulators with CI support.
    
    Parameters
    ----------
    k_accumulators : dict
        Accumulators for wavenumber (flux) statistics.
    angular_accumulators : dict
        Accumulators for angular-wavenumber statistics.
    r_accumulators : dict
        Accumulators for radial SF statistics.
    n_k : int
        Number of wavenumbers.
    n_theta : int
        Number of angular bins.
    n_r : int
        Number of radial bins.
    confidence_level : float
        Confidence level for intervals.
        
    Returns
    -------
    energy_flux : array
        Energy flux at each wavenumber.
    flux_stds : array
        Standard errors.
    ci_lower, ci_upper : array
        Confidence interval bounds.
    flux_theta_k : array
        Angular distribution of flux.
    flux_theta_k_counts : array
        Counts for angular-wavenumber bins.
    sf_r : array
        Angle-averaged structure function.
    """
    energy_flux = np.full(n_k, np.nan)
    flux_stds = np.full(n_k, np.nan)
    ci_lower = np.full(n_k, np.nan)
    ci_upper = np.full(n_k, np.nan)
    flux_theta_k = np.full((n_theta, n_k), np.nan)
    flux_theta_k_counts = np.zeros((n_theta, n_k), dtype=np.int_)
    sf_r = np.full(n_r, np.nan)
    
    # Wavenumber (flux) statistics
    for k_idx, acc in k_accumulators.items():
        if acc['total_weight'] > 0:
            if len(acc['bootstrap_samples']) > 1:
                energy_flux[k_idx], flux_stds[k_idx], ci_lower[k_idx], ci_upper[k_idx] = \
                    _compute_weighted_bootstrap_stats(
                        acc['bootstrap_samples'],
                        confidence_level=confidence_level
                    )
            else:
                energy_flux[k_idx] = acc['weighted_sum'] / acc['total_weight']
                flux_stds[k_idx] = np.nan
    
    # Angular-wavenumber matrix
    for (theta_idx, k_idx), acc in angular_accumulators.items():
        if acc['total_weight'] > 0:
            flux_theta_k[theta_idx, k_idx] = acc['weighted_sum'] / acc['total_weight']
            flux_theta_k_counts[theta_idx, k_idx] = int(acc['total_weight'])
    
    # Radial SF
    for r_idx, acc in r_accumulators.items():
        if acc['total_weight'] > 0:
            sf_r[r_idx] = acc['weighted_sum'] / acc['total_weight']
    
    return energy_flux, flux_stds, ci_lower, ci_upper, flux_theta_k, flux_theta_k_counts, sf_r


def _update_spacing_effectiveness_flux_2d(bin_spacing_effectiveness, bin_spacing_counts,
                                      bin_spacing_bootstraps, sp_value, k_indices,
                                      bootstraps):
    """Update spacing effectiveness for energy flux calculation."""
    if bootstraps <= 0:
        return
    
    for idx in k_indices:
        if bin_spacing_counts[sp_value][idx] > 0:
            bin_spacing_effectiveness[sp_value][idx] = (
                bin_spacing_counts[sp_value][idx] / bootstraps
            )
            bin_spacing_bootstraps[sp_value][idx] += bootstraps
            
            
def _update_spacing_effectiveness_2d(bin_spacing_effectiveness, bin_spacing_counts,
                                bin_spacing_bootstraps, sp_value, bin_indices, 
                                bootstraps):
    """
    Update spacing effectiveness metrics.
    
    Parameters
    ----------
    bin_spacing_effectiveness : dict
        Effectiveness scores for each spacing
    bin_spacing_counts : dict
        Point counts for each spacing
    bin_spacing_bootstraps : dict
        Bootstrap counts for each spacing
    sp_value : int
        Current spacing value
    bin_indices : list
        Bins that were processed
    bootstraps : int
        Number of bootstraps run
    """
    if bootstraps <= 0:
        return
        
    # For 2D case
    if isinstance(bin_indices[0], tuple):
        for j, i in bin_indices:
            if bin_spacing_counts[sp_value][j, i] > 0:
                bin_spacing_effectiveness[sp_value][j, i] = (
                    bin_spacing_counts[sp_value][j, i] / bootstraps
                )
                bin_spacing_bootstraps[sp_value][j, i] += bootstraps
    # For 1D case (polar)
    else:
        for idx in bin_indices:
            if bin_spacing_counts[sp_value][idx] > 0:
                bin_spacing_effectiveness[sp_value][idx] = (
                    bin_spacing_counts[sp_value][idx] / bootstraps
                )
                bin_spacing_bootstraps[sp_value][idx] += bootstraps

def _evaluate_convergence_2d(sf_stds, point_counts, bin_bootstraps,
                        convergence_eps, max_bootstraps):
    """
    Evaluate which bins have converged.
    
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

def _evaluate_convergence_flux_2d(sf_stds, point_counts, bin_bootstraps,
                              convergence_eps, max_bootstraps):
    """Evaluate convergence for energy flux (wavenumber) case."""
    converged = np.zeros_like(sf_stds, dtype=bool)
    reasons = {
        'low_density': 0,
        'nan_std': 0,
        'converged_eps': 0,
        'max_bootstraps': 0
    }
    
    # Low density
    low_density = (point_counts <= 10) & ~converged
    converged |= low_density
    reasons['low_density'] = np.sum(low_density)
    
    # NaN std
    nan_std = np.isnan(sf_stds) & ~converged
    converged |= nan_std
    reasons['nan_std'] = np.sum(nan_std)
    
    # Converged by epsilon
    eps_converged = (sf_stds <= convergence_eps) & ~converged & (point_counts > 10)
    converged |= eps_converged
    reasons['converged_eps'] = np.sum(eps_converged)
    
    # Max bootstraps
    max_boot = (bin_bootstraps >= max_bootstraps) & ~converged
    converged |= max_boot
    reasons['max_bootstraps'] = np.sum(max_boot)
    
    return converged, reasons
    
def _group_bins_for_iteration_2d(unconverged_indices, bin_density, bootstrap_steps):
    """
    Group unconverged bins by similar characteristics.
    
    Returns
    -------
    groups : dict
        Dictionary mapping (step, density_quartile) to list of bin indices
    """
    groups = {}
    
    # Handle both 2D and 1D cases
    if len(unconverged_indices) == 2:  # 2D case
        y_idxs, x_idxs = unconverged_indices
        for j, i in zip(y_idxs, x_idxs):
            step = bootstrap_steps[j, i]
            density_quartile = int(bin_density[j, i] * 4)
            group_key = (step, density_quartile)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((j, i))
    else:  # 1D case
        indices = unconverged_indices[0]
        for idx in indices:
            step = bootstrap_steps[idx]
            density_quartile = int(bin_density[idx] * 4)
            group_key = (step, density_quartile)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(idx)
    
    return groups

def _group_wavenumbers_for_iteration_2d(unconverged_indices, bin_density, bootstrap_steps):
    """Group unconverged wavenumbers by characteristics."""
    groups = {}
    
    for idx in unconverged_indices:
        step = bootstrap_steps[idx]
        density_quartile = int(bin_density[idx] * 4)
        group_key = (step, density_quartile)
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(idx)
    
    return groups

def _get_spacing_distribution_2d(bin_list, spacing_effectiveness, total_bootstraps,
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
        if isinstance(bin_list[0], tuple):  # 2D case
            total_eff = sum(spacing_effectiveness[sp][j, i] for j, i in bin_list)
        else:  # 1D case
            total_eff = sum(spacing_effectiveness[sp][idx] for idx in bin_list)
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
            sp_bootstraps = 0 #remaining // len([s for s, e in sorted_spacings if e >= 0])
        
        if sp_bootstraps > 0:
            distribution.append((sp_value, sp_bootstraps))
            remaining -= sp_bootstraps
    
    return distribution

def _get_spacing_distribution_flux_2d(k_list, spacing_effectiveness, total_bootstraps,
                                  spacing_values):
    """Determine spacing distribution for energy flux case."""
    group_effectiveness = {}
    for sp in spacing_values:
        total_eff = sum(spacing_effectiveness[sp][idx] for idx in k_list)
        group_effectiveness[sp] = total_eff / len(k_list) if len(k_list) > 0 else 0
    
    sorted_spacings = sorted(group_effectiveness.items(), key=lambda x: x[1], reverse=True)
    
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
            sp_bootstraps = 0 #remaining // len([s for s, e in sorted_spacings if e >= 0])
        
        if sp_bootstraps > 0:
            distribution.append((sp_value, sp_bootstraps))
            remaining -= sp_bootstraps
    
    return distribution
    
def _run_adaptive_bootstrap_loop_2d(valid_ds, dims, variables_names, order, fun,
                               bins_config, initial_nbootstrap, max_nbootstrap,
                               step_nbootstrap, convergence_eps, spacing_values,
                               bootsize_dict, num_bootstrappable, all_spacings,
                               boot_indexes, bootstrappable_dims, n_jobs, backend,
                               time_dims, conditioning_var, conditioning_bins, is_2d=True,
                               confidence_level=0.95, seed=None):
    """
    Generic adaptive bootstrap loop used by both 2D and isotropic functions.
    
    This function now handles both 2D and polar cases internally.
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility.
    """
    # Determine result shape and initialize arrays
    if is_2d:
        result_shape = (bins_config['n_bins_y'], bins_config['n_bins_x'])
        n_bins_total = bins_config['n_bins_y'] * bins_config['n_bins_x']
    else:
        result_shape = (bins_config['n_bins_r'],)
        n_bins_total = bins_config['n_bins_r']
    
    # Initialize result arrays based on shape
    if is_2d:
        sf_means = np.full(result_shape, np.nan)
        sf_stds = np.full(result_shape, np.nan)
        ci_lower = np.full(result_shape, np.nan)
        ci_upper = np.full(result_shape, np.nan)
        point_counts = np.zeros(result_shape, dtype=np.int_)
        bin_density = np.zeros(result_shape, dtype=np.float32)
        bin_status = np.zeros(result_shape, dtype=bool)
        bin_bootstraps = np.ones(result_shape, dtype=np.int_) * initial_nbootstrap
        bootstrap_steps = np.ones(result_shape, dtype=np.int_) * step_nbootstrap
    else:
        sf_means = np.full(result_shape[0], np.nan)
        sf_stds = np.full(result_shape[0], np.nan)
        ci_lower = np.full(result_shape[0], np.nan)
        ci_upper = np.full(result_shape[0], np.nan)
        point_counts = np.zeros(result_shape[0], dtype=np.int_)
        bin_density = np.zeros(result_shape[0], dtype=np.float32)
        bin_status = np.zeros(result_shape[0], dtype=bool)
        bin_bootstraps = np.ones(result_shape[0], dtype=np.int_) * initial_nbootstrap
        bootstrap_steps = np.ones(result_shape[0], dtype=np.int_) * step_nbootstrap
        # Additional arrays for polar
        sfr = np.full((bins_config['n_bins_theta'], bins_config['n_bins_r']), np.nan)
        sfr_counts = np.zeros((bins_config['n_bins_theta'], bins_config['n_bins_r']), dtype=np.int_)
    
    # Initialize accumulators
    bin_accumulators = {}
    angular_accumulators = {} if not is_2d else None
    
    # Initialize spacing effectiveness tracking
    shape_for_tracking = result_shape if is_2d else result_shape[0]
    bin_spacing_effectiveness = {sp: np.zeros(shape_for_tracking, dtype=np.float32) 
                               for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros(shape_for_tracking, dtype=np.int_) 
                            for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros(shape_for_tracking, dtype=np.int_) 
                        for sp in spacing_values}
    
    # Generate list of all bins
    if is_2d:
        all_bins = [(j, i) for j in range(result_shape[0]) for i in range(result_shape[1])]
    else:
        all_bins = list(range(result_shape[0]))
    
    # INITIAL BOOTSTRAP PHASE
    print("\nINITIAL BOOTSTRAP PHASE")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    
    for sp_idx, sp_value in enumerate(spacing_values):
        print(f"Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Derive per-spacing seed for reproducibility
        sp_seed = (seed + sp_idx) if seed is not None else None
        
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals, pair_counts_results = monte_carlo_simulation_2d(
            ds=valid_ds, dims=dims, variables_names=variables_names,
            order=order, nbootstrap=init_samples_per_spacing,
            bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings, boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims, fun=fun,
            spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims, 
            conditioning_var=conditioning_var, conditioning_bins=conditioning_bins,
            seed=sp_seed
        )
        
        # Process batch based on type
        if is_2d:
            _process_bootstrap_batch_2d(
                sf_results, dx_vals, dy_vals, 
                bins_config['bins_x'], bins_config['bins_y'],
                bin_accumulators, set(all_bins), point_counts,
                bin_spacing_counts, sp_value, True,
                pair_counts_results=pair_counts_results
            )
        else:
            _process_bootstrap_batch_polar_2d(
                sf_results, dx_vals, dy_vals, 
                bins_config['r_bins'], bins_config['theta_bins'],
                bin_accumulators, angular_accumulators, set(all_bins),
                point_counts, bin_spacing_counts, sp_value, True,
                pair_counts_results=pair_counts_results
            )
        
        # Update effectiveness
        _update_spacing_effectiveness_2d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value, all_bins,
            init_samples_per_spacing
        )
        
        del sf_results, dx_vals, dy_vals
        gc.collect()
    
    
    # Calculate initial statistics based on type
    if is_2d:
        sf_means[:], sf_stds[:] = _calculate_bootstrap_statistics_2d(
            bin_accumulators, result_shape
        )
    else:
        sf_means[:], sf_stds[:], ci_lower[:], ci_upper[:], sfr[:], sfr_counts[:] = _calculate_bootstrap_statistics_polar_2d(
            bin_accumulators, angular_accumulators,
            bins_config['n_bins_r'], bins_config['n_bins_theta'],
            confidence_level=confidence_level, 
        )
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    if is_2d:
        bin_density = _calculate_bin_density_2d(point_counts, bins_config['bins_x'], 
                                              bins_config['bins_y'])
    else:
        bin_density = _calculate_bin_density_polar_2d(point_counts, bins_config['r_bins'])
    
    print(f"Total points collected: {np.sum(point_counts)}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins_total}")
    
    # Initial convergence check
    bin_status, convergence_reasons = _evaluate_convergence_2d(
        sf_stds, point_counts, bin_bootstraps, convergence_eps, max_nbootstrap
    )
    
    for reason, count in convergence_reasons.items():
        if count > 0:
            print(f"Marked {count} bins as converged ({reason})")
    
    # MAIN CONVERGENCE LOOP
    iteration = 1
    print("\nSTARTING ADAPTIVE CONVERGENCE LOOP")
    
    while True:
        unconverged = ~bin_status & (point_counts > 10) & (bin_bootstraps < max_nbootstrap)
        if not np.any(unconverged):
            print("All bins have converged or reached max bootstraps!")
            break
            
        print(f"\nIteration {iteration} - {np.sum(unconverged)} unconverged bins")
        
        unconverged_indices = np.where(unconverged)
            
        groups = _group_bins_for_iteration_2d(unconverged_indices, bin_density, bootstrap_steps)
        print(f"Grouped unconverged bins into {len(groups)} groups")
        
        # Process each group
        for (step, density_q), bin_list in sorted(groups.items(),
                                                 key=lambda x: (x[0][1], x[0][0]),
                                                 reverse=True):
            print(f"\nProcessing {len(bin_list)} bins with step size {step} in density quartile {density_q}")
            
            # Get spacing distribution
            distribution = _get_spacing_distribution_2d(
                bin_list, bin_spacing_effectiveness, step, spacing_values
            )
            
            # Process each spacing
            for sp_value, sp_bootstraps in distribution:
                if sp_bootstraps <= 0:
                    continue
                    
                # Run Monte Carlo
                sf_results, dx_vals, dy_vals, pair_counts_results = monte_carlo_simulation_2d(
                    ds=valid_ds, dims=dims, variables_names=variables_names,
                    order=order, nbootstrap=sp_bootstraps,
                    bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
                    all_spacings=all_spacings, boot_indexes=boot_indexes,
                    bootstrappable_dims=bootstrappable_dims, fun=fun,
                    spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins
                )
                
                # Process batch based on type (accumulate counts)
                if is_2d:
                    _process_bootstrap_batch_2d(
                        sf_results, dx_vals, dy_vals, 
                        bins_config['bins_x'], bins_config['bins_y'],
                        bin_accumulators, set(bin_list), point_counts,
                        bin_spacing_counts, sp_value, True,
                        pair_counts_results=pair_counts_results
                    )
                else:
                    _process_bootstrap_batch_polar_2d(
                        sf_results, dx_vals, dy_vals, 
                        bins_config['r_bins'], bins_config['theta_bins'],
                        bin_accumulators, angular_accumulators, set(bin_list),
                        point_counts, bin_spacing_counts, sp_value, True,
                        pair_counts_results=pair_counts_results
                    )
                
                del sf_results, dx_vals, dy_vals, pair_counts_results
                gc.collect()
            
            # Update statistics and check convergence for this group
            for bin_idx in bin_list:
                # Update bootstrap count and recalculate statistics
                if is_2d:
                    j, i = bin_idx
                    bin_bootstraps[j, i] += step
                    
                    if (j, i) in bin_accumulators:
                        acc = bin_accumulators[(j, i)]
                        if acc['total_weight'] > 0:
                            if len(acc['bootstrap_samples']) > 1:
                                sf_means[j, i], sf_stds[j, i], ci_lower[j, i], ci_upper[j, i] = \
                                    _compute_weighted_bootstrap_stats(
                                        acc['bootstrap_samples'],
                                        confidence_level=confidence_level
                                    )
                            else:
                                sf_means[j, i] = acc['weighted_sum'] / acc['total_weight']
                        
                        if sf_stds[j, i] <= convergence_eps:
                            bin_status[j, i] = True
                            print(f"  Bin ({j},{i}) CONVERGED with std {sf_stds[j, i]:.6f}")
                        elif bin_bootstraps[j, i] >= max_nbootstrap:
                            bin_status[j, i] = True
                            print(f"  Bin ({j},{i}) reached MAX BOOTSTRAPS")
                else:
                    r_idx = bin_idx
                    bin_bootstraps[r_idx] += step
                    
                    if r_idx in bin_accumulators:
                        acc = bin_accumulators[r_idx]
                        if acc['total_weight'] > 0:
                            if len(acc['bootstrap_samples']) > 1:
                                sf_means[r_idx], sf_stds[r_idx], ci_lower[r_idx], ci_upper[r_idx] = \
                                    _compute_weighted_bootstrap_stats(
                                        acc['bootstrap_samples'],
                                        confidence_level=confidence_level
                                    )
                            else:
                                sf_means[r_idx] = acc['weighted_sum'] / acc['total_weight']
                        
                        if sf_stds[r_idx] <= convergence_eps:
                            bin_status[r_idx] = True
                            print(f"  Bin {r_idx} CONVERGED with std {sf_stds[r_idx]:.6f}")
                        elif bin_bootstraps[r_idx] >= max_nbootstrap:
                            bin_status[r_idx] = True
                            print(f"  Bin {r_idx} reached MAX BOOTSTRAPS")
        
        # Update angular-radial matrix if polar
        if not is_2d and angular_accumulators:
            for (theta_idx, r_idx), acc in angular_accumulators.items():
                if acc['total_weight'] > 0:
                    sfr[theta_idx, r_idx] = acc['weighted_sum'] / acc['total_weight']
        
        iteration += 1
        gc.collect()
    
    # Final statistics
    converged_bins = np.sum(bin_status & (point_counts > 10))
    unconverged_bins = np.sum(~bin_status & (point_counts > 10))
    max_bootstrap_bins = np.sum((bin_bootstraps >= max_nbootstrap) & (point_counts > 10))
    
    print("\nFINAL CONVERGENCE STATISTICS:")
    print(f"  Total bins with data (>10 points): {np.sum(point_counts > 10)}")
    print(f"  Converged bins: {converged_bins}")
    print(f"  Unconverged bins: {unconverged_bins}")
    print(f"  Bins at max bootstraps: {max_bootstrap_bins}")
    
    # Return all results
    results = {
        'sf_means': sf_means,
        'sf_stds': sf_stds,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'point_counts': point_counts,
        'bin_density': bin_density,
        'bin_status': bin_status,
        'bin_bootstraps': bin_bootstraps,
        'spacing_values': spacing_values
    }
    
    if not is_2d:
        results['sfr'] = sfr
        results['sfr_counts'] = sfr_counts
    
    return results

def _run_adaptive_bootstrap_loop_flux_2d(valid_ds, dims, variables_names, order, fun,
                                         config, initial_nbootstrap, max_nbootstrap,
                                         step_nbootstrap, convergence_eps, spacing_values,
                                         bootsize_dict, num_bootstrappable, all_spacings,
                                         boot_indexes, bootstrappable_dims, n_jobs, backend,
                                         time_dims, conditioning_var, conditioning_bins,
                                         confidence_level=0.95, seed=None):
    """
    Adaptive bootstrap loop for energy flux computation.
    
    Computes Π(K) = -K/2 ∫ SF̃(r) J₁(Kr) dr using:
    1. Radial binning to get angle-averaged SF̃(r)
    2. J₁ Bessel transform to get energy flux Π(K)
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility.
    """
    n_k = config['n_k']
    n_theta = config['n_bins_theta']
    n_r = config['n_r']
    k = config['k']
    
    # Initialize result arrays
    energy_flux = np.full(n_k, np.nan)
    flux_stds = np.full(n_k, np.nan)
    ci_lower = np.full(n_k, np.nan)
    ci_upper = np.full(n_k, np.nan)
    point_counts = np.zeros(n_k, dtype=np.int_)  # Counts per wavenumber
    bin_density = np.zeros(n_k, dtype=np.float32)
    bin_status = np.zeros(n_k, dtype=bool)
    bin_bootstraps = np.ones(n_k, dtype=np.int_) * initial_nbootstrap
    bootstrap_steps = np.ones(n_k, dtype=np.int_) * step_nbootstrap
    
    # Angular-wavenumber matrix for flux
    flux_theta_k = np.full((n_theta, n_k), np.nan)
    flux_theta_k_counts = np.zeros((n_theta, n_k), dtype=np.int_)
    
    # Angle-averaged SF
    sf_r = np.full(n_r, np.nan)
    
    # Accumulators for bootstrap statistics
    k_accumulators = {}  # For energy flux at each wavenumber
    angular_accumulators = {}  # For (theta, k) pairs
    r_accumulators = {}  # For radial SF
    
    # Initialize spacing tracking
    bin_spacing_effectiveness = {sp: np.zeros(n_k, dtype=np.float32) for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros(n_k, dtype=np.int_) for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros(n_k, dtype=np.int_) for sp in spacing_values}
    
    all_k_indices = list(range(n_k))
    
    # INITIAL BOOTSTRAP PHASE
    print("\nINITIAL BOOTSTRAP PHASE (Energy Flux)")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    
    for sp_idx, sp_value in enumerate(spacing_values):
        print(f"Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Derive per-spacing seed for reproducibility
        sp_seed = (seed + sp_idx) if seed is not None else None
        
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals, pair_counts_results = monte_carlo_simulation_2d(
            ds=valid_ds, dims=dims, variables_names=variables_names,
            order=order, nbootstrap=init_samples_per_spacing,
            bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings, boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims, fun=fun,
            spacing=sp_value, n_jobs=n_jobs, backend=backend, 
            time_dims=time_dims, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins,
            seed=sp_seed
        )
        
        # Process batch with energy flux weighting
        _process_bootstrap_batch_flux_2d(
            sf_results, dx_vals, dy_vals,
            config, k_accumulators, angular_accumulators, r_accumulators,
            set(all_k_indices), point_counts, bin_spacing_counts, sp_value, True
        )
        
        # Update effectiveness
        _update_spacing_effectiveness_flux_2d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value, all_k_indices,
            init_samples_per_spacing
        )
        
        del sf_results, dx_vals, dy_vals, pair_counts_results
        gc.collect()
    
    
    # Calculate initial statistics
    (energy_flux[:], flux_stds[:], ci_lower[:], ci_upper[:], 
     flux_theta_k[:], flux_theta_k_counts[:], sf_r[:]) = _calculate_bootstrap_statistics_flux_2d(
        k_accumulators, angular_accumulators, r_accumulators, n_k, n_theta, n_r,
        confidence_level=confidence_level, 
    )
    
    # Calculate density (effective samples at each wavenumber)
    print("\nCALCULATING WAVENUMBER DENSITIES")
    bin_density = _calculate_wavenumber_density_2d(point_counts, k)
    
    print(f"Total points collected: {np.sum(point_counts)}")
    print(f"Wavenumbers with flux estimates: {np.count_nonzero(~np.isnan(energy_flux))}/{n_k}")
    
    # Initial convergence check
    bin_status, convergence_reasons = _evaluate_convergence_flux_2d(
        flux_stds, point_counts, bin_bootstraps, convergence_eps, max_nbootstrap
    )
    
    for reason, count in convergence_reasons.items():
        if count > 0:
            print(f"Marked {count} wavenumbers as converged ({reason})")
    
    # MAIN CONVERGENCE LOOP
    iteration = 1
    print("\nSTARTING ADAPTIVE CONVERGENCE LOOP")
    
    while True:
        unconverged = ~bin_status & (bin_bootstraps < max_nbootstrap)
        if not np.any(unconverged):
            print("All wavenumbers have converged or reached max bootstraps!")
            break
        
        print(f"\nIteration {iteration} - {np.sum(unconverged)} unconverged wavenumbers")
        
        unconverged_indices = np.where(unconverged)[0]
        
        groups = _group_wavenumbers_for_iteration_2d(unconverged_indices, bin_density, bootstrap_steps)
        print(f"Grouped unconverged wavenumbers into {len(groups)} groups")
        
        # Process each group
        for (step, density_q), k_list in sorted(groups.items(),
                                                key=lambda x: (x[0][1], x[0][0]),
                                                reverse=True):
            print(f"\nProcessing {len(k_list)} wavenumbers with step size {step} in density quartile {density_q}")
            
            # Get spacing distribution
            distribution = _get_spacing_distribution_flux_2d(
                k_list, bin_spacing_effectiveness, step, spacing_values
            )
            
            # Process each spacing
            for sp_value, sp_bootstraps in distribution:
                if sp_bootstraps <= 0:
                    continue
                
                # Run Monte Carlo
                sf_results, dx_vals, dy_vals, pair_counts_results = monte_carlo_simulation_2d(
                    ds=valid_ds, dims=dims, variables_names=variables_names,
                    order=order, nbootstrap=sp_bootstraps,
                    bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
                    all_spacings=all_spacings, boot_indexes=boot_indexes,
                    bootstrappable_dims=bootstrappable_dims, fun=fun,
                    spacing=sp_value, n_jobs=n_jobs, backend=backend,
                    time_dims=time_dims, conditioning_var=conditioning_var, conditioning_bins=conditioning_bins
                )
                
                # Process batch (accumulate counts)
                _process_bootstrap_batch_flux_2d(
                    sf_results, dx_vals, dy_vals,
                    config, k_accumulators, angular_accumulators, r_accumulators,
                    set(k_list), point_counts, bin_spacing_counts, sp_value, True
                )
                
                del sf_results, dx_vals, dy_vals, pair_counts_results
                gc.collect()
            
            # Update statistics and check convergence for this group
            for k_idx in k_list:
                bin_bootstraps[k_idx] += step
                
                if k_idx in k_accumulators:
                    acc = k_accumulators[k_idx]
                    if acc['total_weight'] > 0:
                        if len(acc['bootstrap_samples']) > 1:
                            energy_flux[k_idx], flux_stds[k_idx], ci_lower[k_idx], ci_upper[k_idx] = \
                                _compute_weighted_bootstrap_stats(
                                    acc['bootstrap_samples'],
                                    confidence_level=confidence_level,
                                    
                                )
                        else:
                            energy_flux[k_idx] = acc['weighted_sum'] / acc['total_weight']
                    
                    if flux_stds[k_idx] <= convergence_eps:
                        bin_status[k_idx] = True
                        print(f"  Wavenumber {k_idx} CONVERGED with std {flux_stds[k_idx]:.6f}")
                    elif bin_bootstraps[k_idx] >= max_nbootstrap:
                        bin_status[k_idx] = True
                        print(f"  Wavenumber {k_idx} reached MAX BOOTSTRAPS")
        
        # Update angular-wavenumber matrix
        for (theta_idx, k_idx), acc in angular_accumulators.items():
            if acc['total_weight'] > 0:
                flux_theta_k[theta_idx, k_idx] = acc['weighted_sum'] / acc['total_weight']
        
        # Update radial SF
        for r_idx, acc in r_accumulators.items():
            if acc['total_weight'] > 0:
                sf_r[r_idx] = acc['weighted_sum'] / acc['total_weight']
        
        iteration += 1
        gc.collect()
    
    # Final statistics
    converged_k = np.sum(bin_status)
    unconverged_k = np.sum(~bin_status)
    max_bootstrap_k = np.sum(bin_bootstraps >= max_nbootstrap)
    
    print("\nFINAL CONVERGENCE STATISTICS:")
    print(f"  Total wavenumbers: {n_k}")
    print(f"  Converged wavenumbers: {converged_k}")
    print(f"  Unconverged wavenumbers: {unconverged_k}")
    print(f"  Wavenumbers at max bootstraps: {max_bootstrap_k}")
    
    return {
        'energy_flux': energy_flux,
        'flux_stds': flux_stds,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'point_counts': point_counts,
        'bin_density': bin_density,
        'bin_status': bin_status,
        'bin_bootstraps': bin_bootstraps,
        'flux_theta_k': flux_theta_k,
        'flux_theta_k_counts': flux_theta_k_counts,
        'sf_r': sf_r,
        'spacing_values': spacing_values
    }
##################################################################################################################################################################

################################################################3D################################################################################################
def run_bootstrap_sf_3d(args):
    """Standalone bootstrap function for parallel processing in 3D."""
    ds, dims, variables_names, order, fun, nbz, nby, nbx, spacing, num_bootstrappable, bootstrappable_dims, boot_indexes, time_dims, conditioning_var, conditioning_bins = args
    results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
        ds=ds, dims=dims, variables_names=variables_names, order=order, fun=fun,
        nbz=nbz, nby=nby, nbx=nbx, spacing=spacing, num_bootstrappable=num_bootstrappable, 
        bootstrappable_dims=bootstrappable_dims, boot_indexes=boot_indexes, time_dims=time_dims,
        conditioning_var=conditioning_var, conditioning_bins=conditioning_bins
    )
    return results, dx_vals, dy_vals, dz_vals, pair_counts

def monte_carlo_simulation_3d(ds, dims, variables_names, order, nbootstrap, bootsize, 
                            num_bootstrappable, all_spacings, boot_indexes, bootstrappable_dims,
                            fun='longitudinal', spacing=None, n_jobs=-1, backend='threading',
                            time_dims=None, conditioning_var=None, conditioning_bins=None, seed=None):
    """
    Run Monte Carlo simulation for structure function calculation with multiple bootstrap samples.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing velocity components and/or scalar fields
    dims : list
        List of dimension names
    variables_names : list
        List of variable names to use, depends on function type
    order : int or tuple
        Order(s) of the structure function
    nbootstrap : int
        Number of bootstrap samples
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    num_bootstrappable : int
        Number of bootstrappable dimensions
    all_spacings : list
        List of all spacing values
    boot_indexes : dict
        Dictionary with spacing values as keys and boot indexes as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
    fun : str, optional
        Type of structure function
    spacing : int or dict, optional
        Spacing value to use
    n_jobs : int, optional
        Number of jobs for parallel processing
    backend : str, optional
        Backend for parallel processing
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list, list, list, list, list
        Lists of structure function values, DX values, DY values, DZ values, pair_counts
    """
    # Create random generator (seeded if provided)
    rng = np.random.default_rng(seed)
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
            ds=ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            time_dims=time_dims  # Pass time_dims to calculate_structure_function_3d
        )
        return [results], [dx_vals], [dy_vals], [dz_vals], [pair_counts]
    
    # Use default spacing of 1 if None provided
    if spacing is None:
        sp_value = 1
    # Convert dict spacing to single value if needed
    elif isinstance(spacing, dict):
        # Get the spacing for a bootstrappable dimension
        for dim in bootstrappable_dims:
            if dim in spacing:
                sp_value = spacing[dim]
                break
        else:
            sp_value = 1  # Default if no matching dimension found
    else:
        sp_value = spacing
    
    
    # Get boot indexes for the specified spacing
    if sp_value in boot_indexes:
        indexes = boot_indexes[sp_value]
    else:
        # Calculate boot indexes on-the-fly
        data_shape = dict(ds.sizes)
        indexes = get_boot_indexes_3d(dims, data_shape, bootsize, all_spacings, boot_indexes, 
                                    bootstrappable_dims, num_bootstrappable, sp_value)
    
    # Create all argument arrays for parallel processing
    all_args = []
        
    # Prepare parameters based on bootstrappable dimensions
    if num_bootstrappable == 1:
        # Only one dimension is bootstrappable
        bootstrap_dim = bootstrappable_dims[0]
        
        if not indexes or bootstrap_dim not in indexes or indexes[bootstrap_dim].shape[1] == 0:
            print(f"Warning: No valid indices for dimension {bootstrap_dim} with spacing {sp_value}.")
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals], [pair_counts]
        
        # Generate random indices for the bootstrappable dimension (seeded)
        random_indices = rng.choice(indexes[bootstrap_dim].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            # Set values based on which dimension is bootstrappable
            nbz = random_indices[j] if bootstrap_dim == dims[0] else 0
            nby = random_indices[j] if bootstrap_dim == dims[1] else 0
            nbx = random_indices[j] if bootstrap_dim == dims[2] else 0
            
            args = (
                ds, dims, variables_names, order, fun, 
                nbz, nby, nbx, sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims,
                conditioning_var, conditioning_bins  # Add time_dims
            )
            all_args.append(args)
            
    elif num_bootstrappable == 2:
        # Two dimensions are bootstrappable
        # Check if we have valid indices for both dimensions
        valid_indexes = True
        for dim in bootstrappable_dims:
            if dim not in indexes or indexes[dim].shape[1] == 0:
                print(f"Warning: No valid indices for dimension {dim} with spacing {sp_value}.")
                valid_indexes = False
                break
        
        if not valid_indexes:
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals], [pair_counts]
        
        # Generate random indices for bootstrappable dimensions (seeded)
        nb_indices = {}
        for dim in bootstrappable_dims:
            nb_indices[dim] = rng.choice(indexes[dim].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            # Set values based on which dimensions are bootstrappable
            nbz = nb_indices[dims[0]][j] if dims[0] in bootstrappable_dims else 0
            nby = nb_indices[dims[1]][j] if dims[1] in bootstrappable_dims else 0
            nbx = nb_indices[dims[2]][j] if dims[2] in bootstrappable_dims else 0
            
            args = (
                ds, dims, variables_names, order, fun,
                nbz, nby, nbx, sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims,
                conditioning_var, conditioning_bins  # Add time_dims
            )
            all_args.append(args)
            
    else:  # num_bootstrappable == 3
        # All three dimensions are bootstrappable
        valid_indexes = True
        for dim in dims:
            if dim not in indexes or indexes[dim].shape[1] == 0:
                print(f"Warning: No valid indices for dimension {dim} with spacing {sp_value}.")
                valid_indexes = False
                break
        
        if not valid_indexes:
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals, dz_vals, pair_counts = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals], [pair_counts]
        
        # Generate random indices for all three dimensions (seeded)
        nbz = rng.choice(indexes[dims[0]].shape[1], size=nbootstrap) 
        nby = rng.choice(indexes[dims[1]].shape[1], size=nbootstrap)
        nbx = rng.choice(indexes[dims[2]].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            args = (
                ds, dims, variables_names, order, fun,
                nbz[j], nby[j], nbx[j], sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims,
                conditioning_var, conditioning_bins  # Add time_dims
            )
            all_args.append(args)
    
    # Calculate optimal batch size based on number of jobs and bootstraps
    if n_jobs < 0:  # All negative n_jobs values
        total_cpus = os.cpu_count()
        if n_jobs == -1:  # Special case: use all CPUs
            n_workers = total_cpus
        else:  # Use (all CPUs - |n_jobs| - 1)
            n_workers = max(1, total_cpus + n_jobs + 1)  # +1 because -2 means all except 1
    else:
        n_workers = n_jobs
    
    batch_size = max(10, nbootstrap//(n_workers*2))
    
    # Run simulations in parallel using module-level function
    results = Parallel(n_jobs=n_jobs, verbose=0, batch_size=batch_size, backend=backend)(
        delayed(run_bootstrap_sf_3d)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    dx_vals = [r[1] for r in results]
    dy_vals = [r[2] for r in results]
    dz_vals = [r[3] for r in results]
    pair_counts_results = [r[4] for r in results]
    
    
    return sf_results, dx_vals, dy_vals, dz_vals, pair_counts_results

def _process_bootstrap_batch_3d(sf_results, dx_vals, dy_vals, dz_vals, bins_x, bins_y, bins_z,
                               bin_accumulators, target_bins, point_counts=None,
                               spacing_counts=None, sp_value=None, add_to_counts=True,
                               pair_counts_results=None):
    """
    Process a batch of bootstrap results for 3D Cartesian binning.
    
    FIXED: Now records each bootstrap mean independently rather than incrementally.
    Each bootstrap iteration produces one mean estimate per bin.
    Uses pair_counts for proper weighting when combining separations into bins.
    
    Parameters
    ----------
    sf_results : list
        Structure function results from monte carlo simulation
    dx_vals, dy_vals, dz_vals : list
        Separation distances for each bootstrap
    bins_x, bins_y, bins_z : array
        Bin edges for x, y, and z dimensions
    bin_accumulators : dict
        Accumulator dictionary with keys (k, j, i)
    target_bins : set
        Set of (k, j, i) tuples for bins to process
    point_counts : array, optional
        Array to update with point counts
    spacing_counts : dict, optional
        Dictionary of spacing counts to update
    sp_value : int, optional
        Current spacing value
    add_to_counts : bool
        Whether to update counts
    pair_counts_results : list, optional
        List of pair counts arrays from structure function calculations.
        
    Returns
    -------
    updated_bins : set
        Set of bins that were updated
    """
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1
    n_bins_z = len(bins_z) - 1
    updated_bins = set()
    
    # Create set of target bin IDs for fast lookup
    target_bin_ids = {k * n_bins_y * n_bins_x + j * n_bins_x + i for k, j, i in target_bins}
    
    # Process each bootstrap sample INDEPENDENTLY
    for b in range(len(sf_results)):
        sf = sf_results[b]
        dx = dx_vals[b]
        dy = dy_vals[b]
        dz = dz_vals[b]
        # Get pair counts for this bootstrap (if available)
        pc = pair_counts_results[b] if pair_counts_results is not None else None
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy) & ~np.isnan(dz)
        if not np.any(valid):
            continue
            
        sf_valid = sf[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]
        dz_valid = dz[valid]
        pc_valid = pc[valid] if pc is not None else None
        
        # Vectorized bin assignment
        x_indices = np.clip(np.digitize(dx_valid, bins_x) - 1, 0, n_bins_x - 1)
        y_indices = np.clip(np.digitize(dy_valid, bins_y) - 1, 0, n_bins_y - 1)
        z_indices = np.clip(np.digitize(dz_valid, bins_z) - 1, 0, n_bins_z - 1)
        
        # Create unique bin IDs
        bin_ids = z_indices * n_bins_y * n_bins_x + y_indices * n_bins_x + x_indices
        
        # Temporary accumulators for THIS bootstrap only
        boot_accum = {}
        
        # Accumulate data for this bootstrap
        for idx in range(len(sf_valid)):
            bin_id = bin_ids[idx]
            if bin_id not in target_bin_ids:
                continue
                
            k = bin_id // (n_bins_y * n_bins_x)
            j = (bin_id % (n_bins_y * n_bins_x)) // n_bins_x
            i = bin_id % n_bins_x
            bin_key = (k, j, i)
            
            if bin_key not in boot_accum:
                boot_accum[bin_key] = {'weighted_sum': 0.0, 'total_weight': 0.0, 'count': 0}
            
            # Use pair_counts as weights if available, otherwise use 1
            weight = float(pc_valid[idx]) if pc_valid is not None else 1.0
            boot_accum[bin_key]['weighted_sum'] += sf_valid[idx] * weight
            boot_accum[bin_key]['total_weight'] += weight
            boot_accum[bin_key]['count'] += 1
        
        # Record the bootstrap mean for each bin that received data
        for bin_key, data in boot_accum.items():
            if data['total_weight'] > 0:
                boot_mean = data['weighted_sum'] / data['total_weight']
                
                # Initialize main accumulator if needed
                if bin_key not in bin_accumulators:
                    bin_accumulators[bin_key] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                
                # Add to global accumulator for overall mean
                bin_accumulators[bin_key]['weighted_sum'] += data['weighted_sum']
                bin_accumulators[bin_key]['total_weight'] += data['total_weight']
                bin_accumulators[bin_key]['bootstrap_samples'].append({
                    'mean': boot_mean,
                    'weight': data['total_weight']
                })
                
                updated_bins.add(bin_key)
                
                # Update counts (only when add_to_counts is True)
                if add_to_counts:
                    k, j, i = bin_key
                    if point_counts is not None:
                        point_counts[k, j, i] += data['count']
                    if spacing_counts is not None and sp_value is not None:
                        spacing_counts[sp_value][k, j, i] += data['count']
    
    return updated_bins


def _process_bootstrap_batch_spherical_3d(sf_results, dx_vals, dy_vals, dz_vals, r_bins, theta_bins, phi_bins,
                                     bin_accumulators, angular_accumulators, target_r_bins,
                                     point_counts=None, spacing_counts=None, sp_value=None,
                                     add_to_counts=True, pair_counts_results=None):
    """
    Process a batch of bootstrap results for spherical binning.
    
    FIXED: Now records each bootstrap mean independently rather than incrementally.
    Each bootstrap iteration produces one mean estimate per radial bin.
    Uses pair_counts for proper weighting when combining separations into bins.
    
    Parameters
    ----------
    sf_results : list
        Structure function results
    dx_vals, dy_vals, dz_vals : list
        Separation distances
    r_bins : array
        Radial bin edges
    theta_bins : array
        Azimuthal angular bin edges
    phi_bins : array
        Polar angular bin edges
    bin_accumulators : dict
        Radial accumulator with keys as r_idx
    angular_accumulators : dict
        Angular accumulator with keys as (phi_idx, theta_idx, r_idx)
    target_r_bins : set
        Set of radial bin indices to process
    point_counts : array, optional
        Array to update with counts
    spacing_counts : dict, optional
        Dictionary of spacing counts
    sp_value : int, optional
        Current spacing value
    add_to_counts : bool
        Whether to update counts
    pair_counts_results : list, optional
        List of pair counts arrays from structure function calculations.
        
    Returns
    -------
    updated_r_bins : set
        Set of r bins that were updated
    """
    n_bins_r = len(r_bins) - 1
    n_bins_theta = len(theta_bins) - 1
    n_bins_phi = len(phi_bins) - 1
    updated_r_bins = set()
    
    # Process each bootstrap sample INDEPENDENTLY
    for b in range(len(sf_results)):
        sf = sf_results[b]
        dx = dx_vals[b]
        dy = dy_vals[b]
        dz = dz_vals[b]
        # Get pair counts for this bootstrap (if available)
        pc = pair_counts_results[b] if pair_counts_results is not None else None
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy) & ~np.isnan(dz)
        if not np.any(valid):
            continue
            
        sf_valid = sf[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]
        dz_valid = dz[valid]
        pc_valid = pc[valid] if pc is not None else None
        
        # Convert to spherical coordinates
        r_valid = np.sqrt(dx_valid**2 + dy_valid**2 + dz_valid**2)
        theta_valid = np.arctan2(dy_valid, dx_valid)  # Azimuthal angle (-π to π)
        phi_valid = np.arccos(np.clip(dz_valid / np.maximum(r_valid, 1e-10), -1.0, 1.0))  # Polar angle (0 to π)
        
        # Create bin indices
        r_indices = np.clip(np.digitize(r_valid, r_bins) - 1, 0, n_bins_r - 1)
        theta_indices = np.clip(np.digitize(theta_valid, theta_bins) - 1, 0, n_bins_theta - 1)
        phi_indices = np.clip(np.digitize(phi_valid, phi_bins) - 1, 0, n_bins_phi - 1)
        
        # Temporary accumulators for THIS bootstrap only
        boot_accum_r = {}  # For radial bins
        boot_accum_angular = {}  # For angular bins
        
        # Accumulate data for this bootstrap
        for idx in range(len(sf_valid)):
            r_idx = r_indices[idx]
            if r_idx not in target_r_bins:
                continue
            
            theta_idx = theta_indices[idx]
            phi_idx = phi_indices[idx]
            # Use pair_counts as weights if available, otherwise use 1
            weight = float(pc_valid[idx]) if pc_valid is not None else 1.0
            value = sf_valid[idx]
            
            # Radial accumulator
            if r_idx not in boot_accum_r:
                boot_accum_r[r_idx] = {'weighted_sum': 0.0, 'total_weight': 0.0, 'count': 0}
            boot_accum_r[r_idx]['weighted_sum'] += value * weight
            boot_accum_r[r_idx]['total_weight'] += weight
            boot_accum_r[r_idx]['count'] += 1
            
            # Angular accumulator
            angular_key = (phi_idx, theta_idx, r_idx)
            if angular_key not in boot_accum_angular:
                boot_accum_angular[angular_key] = {'weighted_sum': 0.0, 'total_weight': 0.0}
            boot_accum_angular[angular_key]['weighted_sum'] += value * weight
            boot_accum_angular[angular_key]['total_weight'] += weight
        
        # Record the bootstrap mean for each radial bin that received data
        for r_idx, data in boot_accum_r.items():
            if data['total_weight'] > 0:
                boot_mean = data['weighted_sum'] / data['total_weight']
                
                # Initialize main accumulator if needed
                if r_idx not in bin_accumulators:
                    bin_accumulators[r_idx] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0,
                        'bootstrap_samples': []
                    }
                
                # Add to global accumulator for overall mean
                bin_accumulators[r_idx]['weighted_sum'] += data['weighted_sum']
                bin_accumulators[r_idx]['total_weight'] += data['total_weight']
                bin_accumulators[r_idx]['bootstrap_samples'].append({
                    'mean': boot_mean,
                    'weight': data['total_weight']
                })
                
                updated_r_bins.add(r_idx)
                
                # Update counts (only when add_to_counts is True)
                if add_to_counts:
                    if point_counts is not None:
                        point_counts[r_idx] += data['count']
                    if spacing_counts is not None and sp_value is not None:
                        spacing_counts[sp_value][r_idx] += data['count']
        
        # Update angular accumulators (these don't need bootstrap samples)
        for angular_key, data in boot_accum_angular.items():
            if data['total_weight'] > 0:
                if angular_key not in angular_accumulators:
                    angular_accumulators[angular_key] = {
                        'weighted_sum': 0.0,
                        'total_weight': 0.0
                    }
                angular_accumulators[angular_key]['weighted_sum'] += data['weighted_sum']
                angular_accumulators[angular_key]['total_weight'] += data['total_weight']
    
    return updated_r_bins



def _calculate_bootstrap_statistics_3d(bin_accumulators, bin_shape):
    """
    Calculate weighted means and bootstrap standard errors for 3D bins.
    
    Parameters
    ----------
    bin_accumulators : dict
        Accumulator dictionary with keys (k, j, i)
    bin_shape : tuple
        Shape of output arrays (nz, ny, nx)
        
    Returns
    -------
    sf_means : array
        Weighted means
    sf_stds : array
        Bootstrap standard errors
    """
    nz, ny, nx = bin_shape
    sf_means = np.full((nz, ny, nx), np.nan)
    sf_stds = np.full((nz, ny, nx), np.nan)
    
    for (k, j, i), acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            # Bootstrap standard error
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                boot_weights = np.array([s['weight'] for s in acc['bootstrap_samples']])
                
                # Weighted mean
                sf_means[k, j, i] = np.average(boot_means, weights=boot_weights)
                
                # Weighted std
                weighted_var = np.average((boot_means - sf_means[k, j, i])**2, weights=boot_weights)
                sf_stds[k, j, i] = np.sqrt(weighted_var)
            else:
                sf_means[k, j, i] = acc['weighted_sum'] / acc['total_weight']
                sf_stds[k, j, i] = np.nan
    
    return sf_means, sf_stds


def _calculate_bootstrap_statistics_spherical_3d(bin_accumulators, angular_accumulators, 
                                                 n_bins_r, n_bins_theta, n_bins_phi,
                                                 confidence_level=0.95):
    """
    Calculate statistics for spherical binning with CI support.
    
    Returns
    -------
    sf_means : array
        Radial means
    sf_stds : array
        Radial standard errors
    ci_lower : array
        Lower confidence interval bounds
    ci_upper : array
        Upper confidence interval bounds
    sfr : array
        Angular-radial structure function
    sfr_counts : array
        Counts for angular-radial bins
    """
    sf_means = np.full(n_bins_r, np.nan)
    sf_stds = np.full(n_bins_r, np.nan)
    ci_lower = np.full(n_bins_r, np.nan)
    ci_upper = np.full(n_bins_r, np.nan)
    sfr = np.full((n_bins_phi, n_bins_theta, n_bins_r), np.nan)
    sfr_counts = np.zeros((n_bins_phi, n_bins_theta, n_bins_r), dtype=np.int_)
    
    # Radial statistics
    for r_idx, acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            if len(acc['bootstrap_samples']) > 1:
                sf_means[r_idx], sf_stds[r_idx], ci_lower[r_idx], ci_upper[r_idx] = \
                    _compute_weighted_bootstrap_stats(
                        acc['bootstrap_samples'],
                        confidence_level=confidence_level
                    )
            else:
                sf_means[r_idx] = acc['weighted_sum'] / acc['total_weight']
                sf_stds[r_idx] = np.nan
    
    # Angular-radial matrix
    for (phi_idx, theta_idx, r_idx), acc in angular_accumulators.items():
        if acc['total_weight'] > 0:
            sfr[phi_idx, theta_idx, r_idx] = acc['weighted_sum'] / acc['total_weight']
            sfr_counts[phi_idx, theta_idx, r_idx] = int(acc['total_weight'])
    
    return sf_means, sf_stds, ci_lower, ci_upper, sfr, sfr_counts



def _update_spacing_effectiveness_3d(bin_spacing_effectiveness, bin_spacing_counts,
                                bin_spacing_bootstraps, sp_value, bin_indices, 
                                bootstraps):
    """
    Update spacing effectiveness metrics for 3D.
    
    Parameters
    ----------
    bin_spacing_effectiveness : dict
        Effectiveness scores for each spacing
    bin_spacing_counts : dict
        Point counts for each spacing
    bin_spacing_bootstraps : dict
        Bootstrap counts for each spacing
    sp_value : int
        Current spacing value
    bin_indices : list
        Bins that were processed
    bootstraps : int
        Number of bootstraps run
    """
    if bootstraps <= 0:
        return
        
    # For 3D case
    if isinstance(bin_indices[0], tuple):
        for k, j, i in bin_indices:
            if bin_spacing_counts[sp_value][k, j, i] > 0:
                bin_spacing_effectiveness[sp_value][k, j, i] = (
                    bin_spacing_counts[sp_value][k, j, i] / bootstraps
                )
                bin_spacing_bootstraps[sp_value][k, j, i] += bootstraps
    # For 1D case (spherical)
    else:
        for idx in bin_indices:
            if bin_spacing_counts[sp_value][idx] > 0:
                bin_spacing_effectiveness[sp_value][idx] = (
                    bin_spacing_counts[sp_value][idx] / bootstraps
                )
                bin_spacing_bootstraps[sp_value][idx] += bootstraps


def _evaluate_convergence_3d(sf_stds, point_counts, bin_bootstraps,
                        convergence_eps, max_bootstraps):
    """
    Evaluate which bins have converged for 3D.
    
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


def _group_bins_for_iteration_3d(unconverged_indices, bin_density, bootstrap_steps):
    """
    Group unconverged bins by similar characteristics for 3D.
    
    Returns
    -------
    groups : dict
        Dictionary mapping (step, density_quartile) to list of bin indices
    """
    groups = {}
    
    # Handle both 3D and 1D cases
    if len(unconverged_indices) == 3:  # 3D case
        z_idxs, y_idxs, x_idxs = unconverged_indices
        for k, j, i in zip(z_idxs, y_idxs, x_idxs):
            step = bootstrap_steps[k, j, i]
            density_quartile = int(bin_density[k, j, i] * 4)
            group_key = (step, density_quartile)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((k, j, i))
    else:  # 1D case (spherical)
        indices = unconverged_indices[0]
        for idx in indices:
            step = bootstrap_steps[idx]
            density_quartile = int(bin_density[idx] * 4)
            group_key = (step, density_quartile)
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(idx)
    
    return groups

def _group_wavenumbers_for_iteration_3d(unconverged_indices, bin_density, bootstrap_steps):
    """Group unconverged wavenumbers by characteristics for 3D."""
    groups = {}
    
    for idx in unconverged_indices:
        step = bootstrap_steps[idx]
        density_quartile = int(bin_density[idx] * 4)
        group_key = (step, density_quartile)
        
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(idx)
    
    return groups

def _get_spacing_distribution_3d(bin_list, spacing_effectiveness, total_bootstraps,
                            spacing_values):
    """
    Determine optimal distribution of bootstraps across spacings for 3D.
    
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
        if isinstance(bin_list[0], tuple):  # 3D case
            total_eff = sum(spacing_effectiveness[sp][k, j, i] for k, j, i in bin_list)
        else:  # 1D case (spherical)
            total_eff = sum(spacing_effectiveness[sp][idx] for idx in bin_list)
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
            sp_bootstraps = 0 #remaining // len([s for s, e in sorted_spacings if e >= 0])
        
        if sp_bootstraps > 0:
            distribution.append((sp_value, sp_bootstraps))
            remaining -= sp_bootstraps
    
    return distribution


def _run_adaptive_bootstrap_loop_3d(valid_ds, dims, variables_names, order, fun,
                                  bins_config, initial_nbootstrap, max_nbootstrap,
                                  step_nbootstrap, convergence_eps, spacing_values,
                                  bootsize_dict, num_bootstrappable, all_spacings,
                                  boot_indexes, bootstrappable_dims, n_jobs, backend,
                                  time_dims, is_3d=True, conditioning_var=None, conditioning_bins=None,
                                  confidence_level=0.95, seed=None):
    """
    Generic adaptive bootstrap loop used by both 3D and spherical functions.
    
    This function handles both 3D Cartesian and spherical cases internally.
    
    Parameters
    ----------
    confidence_level : float, optional
        Confidence level for intervals. Default is 0.95.
    seed : int, optional
        Random seed for reproducibility.
    """
    # Determine result shape and initialize arrays
    if is_3d:
        result_shape = (bins_config['n_bins_z'], bins_config['n_bins_y'], bins_config['n_bins_x'])
        n_bins_total = bins_config['n_bins_z'] * bins_config['n_bins_y'] * bins_config['n_bins_x']
    else:
        result_shape = (bins_config['n_bins_r'],)
        n_bins_total = bins_config['n_bins_r']
    
    # Initialize result arrays based on shape
    if is_3d:
        sf_means = np.full(result_shape, np.nan)
        sf_stds = np.full(result_shape, np.nan)
        ci_lower = np.full(result_shape, np.nan)
        ci_upper = np.full(result_shape, np.nan)
        point_counts = np.zeros(result_shape, dtype=np.int_)
        bin_density = np.zeros(result_shape, dtype=np.float32)
        bin_status = np.zeros(result_shape, dtype=bool)
        bin_bootstraps = np.ones(result_shape, dtype=np.int_) * initial_nbootstrap
        bootstrap_steps = np.ones(result_shape, dtype=np.int_) * step_nbootstrap
    else:
        sf_means = np.full(result_shape[0], np.nan)
        sf_stds = np.full(result_shape[0], np.nan)
        ci_lower = np.full(result_shape[0], np.nan)
        ci_upper = np.full(result_shape[0], np.nan)
        point_counts = np.zeros(result_shape[0], dtype=np.int_)
        bin_density = np.zeros(result_shape[0], dtype=np.float32)
        bin_status = np.zeros(result_shape[0], dtype=bool)
        bin_bootstraps = np.ones(result_shape[0], dtype=np.int_) * initial_nbootstrap
        bootstrap_steps = np.ones(result_shape[0], dtype=np.int_) * step_nbootstrap
        # Additional arrays for spherical
        sfr = np.full((bins_config['n_bins_phi'], bins_config['n_bins_theta'], bins_config['n_bins_r']), np.nan)
        sfr_counts = np.zeros((bins_config['n_bins_phi'], bins_config['n_bins_theta'], bins_config['n_bins_r']), dtype=np.int_)
    
    # Initialize accumulators
    bin_accumulators = {}
    angular_accumulators = {} if not is_3d else None
    
    # Initialize spacing effectiveness tracking
    shape_for_tracking = result_shape if is_3d else result_shape[0]
    bin_spacing_effectiveness = {sp: np.zeros(shape_for_tracking, dtype=np.float32) 
                               for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros(shape_for_tracking, dtype=np.int_) 
                            for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros(shape_for_tracking, dtype=np.int_) 
                        for sp in spacing_values}
    
    # Generate list of all bins
    if is_3d:
        all_bins = [(k, j, i) for k in range(result_shape[0]) 
                    for j in range(result_shape[1]) 
                    for i in range(result_shape[2])]
    else:
        all_bins = list(range(result_shape[0]))
    
    # INITIAL BOOTSTRAP PHASE
    print("\nINITIAL BOOTSTRAP PHASE")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    
    for sp_idx, sp_value in enumerate(spacing_values):
        print(f"Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Derive per-spacing seed for reproducibility
        sp_seed = (seed + sp_idx) if seed is not None else None
        
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals, dz_vals, pair_counts_results = monte_carlo_simulation_3d(
            ds=valid_ds, dims=dims, variables_names=variables_names,
            order=order, nbootstrap=init_samples_per_spacing,
            bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings, boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims, fun=fun,
            spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims,
            conditioning_var=conditioning_var, conditioning_bins=conditioning_bins,
            seed=sp_seed
        )
        
        # Process batch based on type
        if is_3d:
            _process_bootstrap_batch_3d(
                sf_results, dx_vals, dy_vals, dz_vals,
                bins_config['bins_x'], bins_config['bins_y'], bins_config['bins_z'],
                bin_accumulators, set(all_bins), point_counts,
                bin_spacing_counts, sp_value, True,
                pair_counts_results=pair_counts_results
            )
        else:
            _process_bootstrap_batch_spherical_3d(
                sf_results, dx_vals, dy_vals, dz_vals,
                bins_config['r_bins'], bins_config['theta_bins'], bins_config['phi_bins'],
                bin_accumulators, angular_accumulators, set(all_bins),
                point_counts, bin_spacing_counts, sp_value, True,
                pair_counts_results=pair_counts_results
            )
        
        # Update effectiveness
        _update_spacing_effectiveness_3d(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value, all_bins,
            init_samples_per_spacing
        )
        
        del sf_results, dx_vals, dy_vals, dz_vals, pair_counts_results
        gc.collect()
    
    
    # Calculate initial statistics based on type
    if is_3d:
        sf_means[:], sf_stds[:] = _calculate_bootstrap_statistics_3d(
            bin_accumulators, result_shape
        )
    else:
        sf_means[:], sf_stds[:], ci_lower[:], ci_upper[:], sfr[:], sfr_counts[:] = _calculate_bootstrap_statistics_spherical_3d(
            bin_accumulators, angular_accumulators,
            bins_config['n_bins_r'], bins_config['n_bins_theta'], bins_config['n_bins_phi'],
            confidence_level=confidence_level, 
        )
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    if is_3d:
        bin_density = _calculate_bin_density_3d(point_counts, bins_config['bins_x'], 
                                              bins_config['bins_y'], bins_config['bins_z'])
    else:
        bin_density = _calculate_bin_density_spherical_3d(point_counts, bins_config['r_bins'])
    
    print(f"Total points collected: {np.sum(point_counts)}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins_total}")
    
    # Initial convergence check
    bin_status, convergence_reasons = _evaluate_convergence_3d(
        sf_stds, point_counts, bin_bootstraps, convergence_eps, max_nbootstrap
    )
    
    for reason, count in convergence_reasons.items():
        if count > 0:
            print(f"Marked {count} bins as converged ({reason})")
    
    # MAIN CONVERGENCE LOOP
    iteration = 1
    print("\nSTARTING ADAPTIVE CONVERGENCE LOOP")
    
    while True:
        unconverged = ~bin_status & (point_counts > 10) & (bin_bootstraps < max_nbootstrap)
        if not np.any(unconverged):
            print("All bins have converged or reached max bootstraps!")
            break
            
        print(f"\nIteration {iteration} - {np.sum(unconverged)} unconverged bins")
        
        unconverged_indices = np.where(unconverged)
            
        groups = _group_bins_for_iteration_3d(unconverged_indices, bin_density, bootstrap_steps)
        print(f"Grouped unconverged bins into {len(groups)} groups")
        
        # Process each group
        for (step, density_q), bin_list in sorted(groups.items(),
                                                 key=lambda x: (x[0][1], x[0][0]),
                                                 reverse=True):
            print(f"\nProcessing {len(bin_list)} bins with step size {step} in density quartile {density_q}")
            
            # Get spacing distribution
            distribution = _get_spacing_distribution_3d(
                bin_list, bin_spacing_effectiveness, step, spacing_values
            )
            
            # Process each spacing
            for sp_value, sp_bootstraps in distribution:
                if sp_bootstraps <= 0:
                    continue
                    
                # Run Monte Carlo
                sf_results, dx_vals, dy_vals, dz_vals, pair_counts_results = monte_carlo_simulation_3d(
                    ds=valid_ds, dims=dims, variables_names=variables_names,
                    order=order, nbootstrap=sp_bootstraps,
                    bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
                    all_spacings=all_spacings, boot_indexes=boot_indexes,
                    bootstrappable_dims=bootstrappable_dims, fun=fun,
                    spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims,
                    conditioning_var=conditioning_var, conditioning_bins=conditioning_bins
                )
                
                # Process batch based on type (accumulate counts)
                if is_3d:
                    _process_bootstrap_batch_3d(
                        sf_results, dx_vals, dy_vals, dz_vals,
                        bins_config['bins_x'], bins_config['bins_y'], bins_config['bins_z'],
                        bin_accumulators, set(bin_list), point_counts,
                        bin_spacing_counts, sp_value, True,
                        pair_counts_results=pair_counts_results
                    )
                else:
                    _process_bootstrap_batch_spherical_3d(
                        sf_results, dx_vals, dy_vals, dz_vals,
                        bins_config['r_bins'], bins_config['theta_bins'], bins_config['phi_bins'],
                        bin_accumulators, angular_accumulators, set(bin_list),
                        point_counts, bin_spacing_counts, sp_value, True,
                        pair_counts_results=pair_counts_results
                    )
                
                del sf_results, dx_vals, dy_vals, dz_vals, pair_counts_results
                gc.collect()
            
            # Update statistics and check convergence for this group
            for bin_idx in bin_list:
                # Update bootstrap count and recalculate statistics
                if is_3d:
                    k, j, i = bin_idx
                    bin_bootstraps[k, j, i] += step
                    
                    if (k, j, i) in bin_accumulators:
                        acc = bin_accumulators[(k, j, i)]
                        if acc['total_weight'] > 0:
                            if len(acc['bootstrap_samples']) > 1:
                                sf_means[k, j, i], sf_stds[k, j, i], ci_lower[k, j, i], ci_upper[k, j, i] = \
                                    _compute_weighted_bootstrap_stats(
                                        acc['bootstrap_samples'],
                                        confidence_level=confidence_level
                                    )
                            else:
                                sf_means[k, j, i] = acc['weighted_sum'] / acc['total_weight']
                        
                        if sf_stds[k, j, i] <= convergence_eps:
                            bin_status[k, j, i] = True
                            print(f"  Bin ({k},{j},{i}) CONVERGED with std {sf_stds[k, j, i]:.6f}")
                        elif bin_bootstraps[k, j, i] >= max_nbootstrap:
                            bin_status[k, j, i] = True
                            print(f"  Bin ({k},{j},{i}) reached MAX BOOTSTRAPS")
                else:
                    r_idx = bin_idx
                    bin_bootstraps[r_idx] += step
                    
                    if r_idx in bin_accumulators:
                        acc = bin_accumulators[r_idx]
                        if acc['total_weight'] > 0:
                            if len(acc['bootstrap_samples']) > 1:
                                sf_means[r_idx], sf_stds[r_idx], ci_lower[r_idx], ci_upper[r_idx] = \
                                    _compute_weighted_bootstrap_stats(
                                        acc['bootstrap_samples'],
                                        confidence_level=confidence_level
                                    )
                            else:
                                sf_means[r_idx] = acc['weighted_sum'] / acc['total_weight']
                        
                        if sf_stds[r_idx] <= convergence_eps:
                            bin_status[r_idx] = True
                            print(f"  Bin {r_idx} CONVERGED with std {sf_stds[r_idx]:.6f}")
                        elif bin_bootstraps[r_idx] >= max_nbootstrap:
                            bin_status[r_idx] = True
                            print(f"  Bin {r_idx} reached MAX BOOTSTRAPS")
        
        # Update angular-radial matrix if spherical
        if not is_3d and angular_accumulators:
            for (phi_idx, theta_idx, r_idx), acc in angular_accumulators.items():
                if acc['total_weight'] > 0:
                    sfr[phi_idx, theta_idx, r_idx] = acc['weighted_sum'] / acc['total_weight']
        
        iteration += 1
        gc.collect()
    
    # Final statistics
    converged_bins = np.sum(bin_status & (point_counts > 10))
    unconverged_bins = np.sum(~bin_status & (point_counts > 10))
    max_bootstrap_bins = np.sum((bin_bootstraps >= max_nbootstrap) & (point_counts > 10))
    
    print("\nFINAL CONVERGENCE STATISTICS:")
    print(f"  Total bins with data (>10 points): {np.sum(point_counts > 10)}")
    print(f"  Converged bins: {converged_bins}")
    print(f"  Unconverged bins: {unconverged_bins}")
    print(f"  Bins at max bootstraps: {max_bootstrap_bins}")
    
    # Return all results
    results = {
        'sf_means': sf_means,
        'sf_stds': sf_stds,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'point_counts': point_counts,
        'bin_density': bin_density,
        'bin_status': bin_status,
        'bin_bootstraps': bin_bootstraps,
        'spacing_values': spacing_values
    }
    
    if not is_3d:
        results['sfr'] = sfr
        results['sfr_counts'] = sfr_counts
    
    return results

##################################################################################################################################################################
