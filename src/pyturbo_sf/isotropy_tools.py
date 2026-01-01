"""Isotropization Tools"""

import numpy as np
import xarray as xr
import bottleneck as bn
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

from .structure_functions import (
    calculate_structure_function_2d,
    calculate_structure_function_3d
)

from .utils import (
    _calculate_confidence_intervals,
     _is_log_spaced
)

###############################################################################################2D##############################################################################################
def _initialize_polar_bins_2d(r_bins, n_theta):
    """
    Initialize polar bin configuration.
    
    Returns
    -------
    config : dict
        Dictionary with polar bin configuration
    """
    # Determine if radial bins are log-spaced
    log_bins = _is_log_spaced(r_bins)
    
    if log_bins:
        r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    else:
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    # Set up angular bins
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    
    return {
        'r_bins': r_bins,
        'theta_bins': theta_bins,
        'r_centers': r_centers,
        'theta_centers': theta_centers,
        'n_bins_r': len(r_centers),
        'n_bins_theta': n_theta,
        'log_bins': log_bins
    }
    
def _process_no_bootstrap_polar_2d(ds, dims, variables_names, order, fun, r_bins, n_theta, time_dims, conditioning_var, conditioning_bins):
    """Handle the special case of no bootstrappable dimensions for polar."""
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function
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
    bins_config = _initialize_polar_bins_2d(r_bins, n_theta)
    
    # Filter and convert to polar
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    r_valid = np.sqrt(valid_dx**2 + valid_dy**2)
    theta_valid = np.arctan2(valid_dy, valid_dx)
    
    # Create bin indices
    r_indices = np.clip(np.digitize(r_valid, bins_config['r_bins']) - 1,
                       0, bins_config['n_bins_r'] - 1)
    theta_indices = np.clip(np.digitize(theta_valid, bins_config['theta_bins']) - 1,
                           0, bins_config['n_bins_theta'] - 1)
    
    # Initialize arrays
    sf_means = np.full(bins_config['n_bins_r'], np.nan)
    sf_stds = np.full(bins_config['n_bins_r'], np.nan)
    point_counts = np.zeros(bins_config['n_bins_r'], dtype=np.int_)
    sfr = np.full((bins_config['n_bins_theta'], bins_config['n_bins_r']), np.nan)
    sfr_counts = np.zeros((bins_config['n_bins_theta'], bins_config['n_bins_r']), dtype=np.int_)
    
    # Process radial bins
    for r_idx in range(bins_config['n_bins_r']):
        r_bin_mask = r_indices == r_idx
        if not np.any(r_bin_mask):
            continue
            
        bin_sf = valid_results[r_bin_mask]
        bin_theta_indices = theta_indices[r_bin_mask]
        
        point_counts[r_idx] = len(bin_sf)
        
        if len(bin_sf) > 0:
            # Simple unweighted mean - each estimate counts equally
            sf_means[r_idx] = np.mean(bin_sf)
            if len(bin_sf) > 1:
                sf_stds[r_idx] = np.std(bin_sf)
        
        # Process angular bins
        for theta_idx in range(bins_config['n_bins_theta']):
            theta_bin_mask = bin_theta_indices == theta_idx
            if not np.any(theta_bin_mask):
                continue
            
            theta_sf = bin_sf[theta_bin_mask]
            
            if len(theta_sf) > 0:
                sfr[theta_idx, r_idx] = np.mean(theta_sf)
                sfr_counts[theta_idx, r_idx] = len(theta_sf)
    
    return sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config

def _calculate_bin_density_polar_2d(point_counts, r_bins):
    """Calculate normalized bin density for polar case."""
    total_points = np.sum(point_counts)
    if total_points == 0:
        return np.zeros_like(point_counts, dtype=np.float32)
    
    # Calculate bin areas in polar coordinates
    bin_areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
    
    bin_density = np.divide(point_counts, bin_areas * total_points,
                          out=np.zeros_like(point_counts, dtype=np.float32),
                          where=bin_areas > 0)
    
    # Normalize
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
        
    return bin_density

def _create_isotropic_dataset(results, bins_config, order, fun, window_size_theta,
                            window_size_r, convergence_eps, max_nbootstrap,
                            initial_nbootstrap, bootstrappable_dims, backend,
                            variables_names, confidence_interval,
                            conditioning_info=None):
    """
    Create output dataset for isotropic binning.
    
    Parameters
    ----------
    results : dict
        Results dictionary from bootstrap loop
    bins_config : dict
        Bin configuration
    conditioning_info : dict, optional
        If provided, contains 'var_name', 'bins', and 'bin_idx' for the conditioning variable.
        When present, adds a conditioning dimension to the dataset.
    """
    # Calculate error metrics
    eiso = _calculate_isotropy_error_2d(results['sfr'], results['sf_means'], window_size_theta)
    ehom, r_subset_indices = _calculate_homogeneity_error_2d(results['sfr'], window_size_r)
    
    # Use pre-computed CIs if available
    if 'ci_lower' in results and 'ci_upper' in results:
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
    else:
        # Fallback to standard method (original behavior)
        ci_upper, ci_lower = _calculate_confidence_intervals(
            results['sf_means'], results['sf_stds'], results['point_counts'], confidence_interval
        )
    
    # Build coordinates
    coords = {
        'r': bins_config['r_centers'],
        'r_subset': bins_config['r_centers'][r_subset_indices],
        'theta': bins_config['theta_centers']
    }
    
    # Build attributes
    attrs = {
        'order': str(order),
        'function_type': fun,
        'window_size_theta': window_size_theta,
        'window_size_r': window_size_r,
        'convergence_eps': convergence_eps,
        'max_nbootstrap': max_nbootstrap,
        'initial_nbootstrap': initial_nbootstrap,
        'bin_type': 'logarithmic' if bins_config['log_bins'] else 'linear',
        'variables': variables_names,
        'confidence_level': confidence_interval,
        'bootstrappable_dimensions': ','.join(bootstrappable_dims),
        'backend': backend,
    }
    
    # Check if we have conditioning info
    if conditioning_info is not None:
        cond_var = conditioning_info['var_name']
        cond_bins = conditioning_info['bins']
        cond_bin_idx = conditioning_info.get('bin_idx', 0)
        
        # Add conditioning bin centers to coordinates
        cond_bin_centers = 0.5 * (cond_bins[:-1] + cond_bins[1:])
        coords['cond_bin'] = [cond_bin_centers[cond_bin_idx]]
        
        # Add conditioning info to attributes
        attrs['conditioning_variable'] = cond_var
        attrs['conditioning_bin_edges'] = list(cond_bins)
        attrs['conditioning_bin_idx'] = cond_bin_idx
        
        # Prepare data variables with conditioning dimension
        data_vars = {
            'sf_polar': (('theta', 'r', 'cond_bin'), results['sfr'][:, :, np.newaxis]),
            'sf': (('r', 'cond_bin'), results['sf_means'][:, np.newaxis]),
            'error_isotropy': (('r', 'cond_bin'), eiso[:, np.newaxis]),
            'std_error': (('r', 'cond_bin'), results['sf_stds'][:, np.newaxis]),
            'ci_upper': (('r', 'cond_bin'), ci_upper[:, np.newaxis]),
            'ci_lower': (('r', 'cond_bin'), ci_lower[:, np.newaxis]),
            'error_homogeneity': (('r_subset', 'cond_bin'), ehom[:, np.newaxis]),
            'n_bootstrap': (('r', 'cond_bin'), results['bin_bootstraps'][:, np.newaxis]),
            'bin_density': (('r', 'cond_bin'), results['bin_density'][:, np.newaxis]),
            'point_counts': (('r', 'cond_bin'), results['point_counts'][:, np.newaxis]),
            'converged': (('r', 'cond_bin'), results['bin_status'][:, np.newaxis])
        }
    else:
        # Standard case without conditioning
        data_vars = {
            'sf_polar': (('theta', 'r'), results['sfr']),
            'sf': (('r'), results['sf_means']),
            'error_isotropy': (('r'), eiso),
            'std_error': (('r'), results['sf_stds']),
            'ci_upper': (('r'), ci_upper),
            'ci_lower': (('r'), ci_lower),
            'error_homogeneity': (('r_subset'), ehom),
            'n_bootstrap': (('r'), results['bin_bootstraps']),
            'bin_density': (('r'), results['bin_density']),
            'point_counts': (('r'), results['point_counts']),
            'converged': (('r'), results['bin_status'])
        }
    
    ds_iso = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    # Add bin edges
    ds_iso['r_bins'] = (('r_edge'), bins_config['r_bins'])
    ds_iso['theta_bins'] = (('theta_edge'), bins_config['theta_bins'])
    
    return ds_iso
    
def _calculate_isotropy_error_2d(sfr, sf_means, window_size_theta):
    """Calculate error of isotropy using sliding windows."""
    n_bins_theta, n_bins_r = sfr.shape
    eiso = np.zeros(n_bins_r)
    
    if n_bins_theta > window_size_theta:
        indices_theta = sliding_window_view(
            np.arange(n_bins_theta),
            (n_bins_theta - window_size_theta + 1,),
            writeable=False
        )[::1]
        
        n_samples_theta = len(indices_theta)
        
        for i in range(n_samples_theta):
            idx = indices_theta[i]
            mean_sf = bn.nanmean(sfr[idx, :], axis=0)
            eiso += np.abs(mean_sf - sf_means)
        
        eiso /= max(1, n_samples_theta)
    
    return eiso


def _calculate_homogeneity_error_2d(sfr, window_size_r):
    """Calculate error of homogeneity."""
    n_bins_theta, n_bins_r = sfr.shape
    
    if n_bins_r > window_size_r:
        indices_r = sliding_window_view(
            np.arange(n_bins_r),
            (n_bins_r - window_size_r + 1,),
            writeable=False
        )[::1]
        
        n_samples_r = len(indices_r)
        r_subset_indices = indices_r[0]
        
        meanh = np.zeros(len(r_subset_indices))
        ehom = np.zeros(len(r_subset_indices))
        
        for i in range(n_samples_r):
            idx = indices_r[i]
            meanh += bn.nanmean(sfr[:, idx], axis=0)
        
        meanh /= max(1, n_samples_r)
        
        for i in range(n_samples_r):
            idx = indices_r[i]
            ehom += np.abs(bn.nanmean(sfr[:, idx], axis=0) - meanh)
        
        ehom /= max(1, n_samples_r)
    else:
        r_subset_indices = np.arange(n_bins_r)
        meanh = bn.nanmean(sfr, axis=0)
        ehom = np.zeros_like(meanh)
    
    return ehom, r_subset_indices

###############################################################################################################################################################################################

###############################################################################################3D##############################################################################################
def _initialize_spherical_bins_3d(r_bins, n_theta, n_phi):
    """
    Initialize spherical bin configuration.
    
    Returns
    -------
    config : dict
        Dictionary with spherical bin configuration
    """
    # Determine if radial bins are log-spaced
    log_bins = _is_log_spaced(r_bins)
    
    if log_bins:
        r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    else:
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    
    # Set up angular bins
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)    # Azimuthal angle
    phi_bins = np.linspace(0, np.pi, n_phi + 1)             # Polar angle
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    
    return {
        'r_bins': r_bins,
        'theta_bins': theta_bins,
        'phi_bins': phi_bins,
        'r_centers': r_centers,
        'theta_centers': theta_centers,
        'phi_centers': phi_centers,
        'n_bins_r': len(r_centers),
        'n_bins_theta': n_theta,
        'n_bins_phi': n_phi,
        'log_bins': log_bins
    }

def _process_no_bootstrap_spherical_3d(ds, dims, variables_names, order, fun, r_bins, n_theta, n_phi, time_dims, conditioning_var=None, conditioning_bins=None):
    """Handle the special case of no bootstrappable dimensions for spherical."""
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function
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
    bins_config = _initialize_spherical_bins_3d(r_bins, n_theta, n_phi)
    
    # Filter and convert to spherical
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals) & ~np.isnan(dz_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    valid_dz = dz_vals[valid_mask]
    r_valid = np.sqrt(valid_dx**2 + valid_dy**2 + valid_dz**2)
    theta_valid = np.arctan2(valid_dy, valid_dx)
    phi_valid = np.arccos(np.clip(valid_dz / np.maximum(r_valid, 1e-10), -1.0, 1.0))
    
    # Create bin indices
    r_indices = np.clip(np.digitize(r_valid, bins_config['r_bins']) - 1,
                       0, bins_config['n_bins_r'] - 1)
    theta_indices = np.clip(np.digitize(theta_valid, bins_config['theta_bins']) - 1,
                           0, bins_config['n_bins_theta'] - 1)
    phi_indices = np.clip(np.digitize(phi_valid, bins_config['phi_bins']) - 1,
                         0, bins_config['n_bins_phi'] - 1)
    
    # Initialize arrays
    sf_means = np.full(bins_config['n_bins_r'], np.nan)
    sf_stds = np.full(bins_config['n_bins_r'], np.nan)
    point_counts = np.zeros(bins_config['n_bins_r'], dtype=np.int_)
    sfr = np.full((bins_config['n_bins_phi'], bins_config['n_bins_theta'], bins_config['n_bins_r']), np.nan)
    sfr_counts = np.zeros((bins_config['n_bins_phi'], bins_config['n_bins_theta'], bins_config['n_bins_r']), dtype=np.int_)
    
    # Process radial bins
    for r_idx in range(bins_config['n_bins_r']):
        r_bin_mask = r_indices == r_idx
        if not np.any(r_bin_mask):
            continue
            
        bin_sf = valid_results[r_bin_mask]
        bin_theta_indices = theta_indices[r_bin_mask]
        bin_phi_indices = phi_indices[r_bin_mask]
        
        point_counts[r_idx] = len(bin_sf)
        
        if len(bin_sf) > 0:
            # Simple unweighted mean - each estimate counts equally
            sf_means[r_idx] = np.mean(bin_sf)
            if len(bin_sf) > 1:
                sf_stds[r_idx] = np.std(bin_sf)
        
        # Process angular bins
        for theta_idx in range(bins_config['n_bins_theta']):
            for phi_idx in range(bins_config['n_bins_phi']):
                angular_mask = (bin_theta_indices == theta_idx) & (bin_phi_indices == phi_idx)
                if not np.any(angular_mask):
                    continue
                
                angular_sf = bin_sf[angular_mask]
                
                if len(angular_sf) > 0:
                    sfr[phi_idx, theta_idx, r_idx] = np.mean(angular_sf)
                    sfr_counts[phi_idx, theta_idx, r_idx] = len(angular_sf)
    
    return sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config

def _calculate_bin_density_spherical_3d(point_counts, r_bins):
    """Calculate normalized bin density for spherical case."""
    total_points = np.sum(point_counts)
    if total_points == 0:
        return np.zeros_like(point_counts, dtype=np.float32)
    
    # Calculate bin volumes in spherical coordinates
    bin_volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    
    bin_density = np.divide(point_counts, bin_volumes * total_points,
                          out=np.zeros_like(point_counts, dtype=np.float32),
                          where=bin_volumes > 0)
    
    # Normalize
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
        
    return bin_density

def _create_spherical_dataset(results, bins_config, order, fun, window_size_theta,
                            window_size_phi, window_size_r, convergence_eps, max_nbootstrap,
                            initial_nbootstrap, bootstrappable_dims, backend,
                            variables_names, confidence_interval=0.95,
                            conditioning_info=None):
    """
    Create output dataset for spherical binning.
    
    Parameters
    ----------
    conditioning_info : dict, optional
        If provided, contains 'var_name', 'bins', and 'bin_idx' for the conditioning variable.
    """
    # Calculate error metrics
    eiso = _calculate_isotropy_error_3d(results['sfr'], results['sf_means'], 
                                       window_size_theta, window_size_phi)
    ehom, r_subset_indices = _calculate_homogeneity_error_3d(results['sfr'], window_size_r)
    
    # Use pre-computed CIs if available
    if 'ci_lower' in results and 'ci_upper' in results:
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
    else:
        ci_upper, ci_lower = _calculate_confidence_intervals(
            results['sf_means'], results['sf_stds'], results['point_counts'], confidence_interval
        )
    
    # Build coordinates
    coords = {
        'r': bins_config['r_centers'],
        'r_subset': bins_config['r_centers'][r_subset_indices],
        'theta': bins_config['theta_centers'],
        'phi': bins_config['phi_centers']
    }
    
    # Build attributes
    attrs = {
        'order': str(order),
        'function_type': fun,
        'window_size_theta': window_size_theta,
        'window_size_phi': window_size_phi,
        'window_size_r': window_size_r,
        'convergence_eps': convergence_eps,
        'max_nbootstrap': max_nbootstrap,
        'initial_nbootstrap': initial_nbootstrap,
        'bin_type': 'logarithmic' if bins_config['log_bins'] else 'linear',
        'variables': variables_names,
        'bootstrappable_dimensions': ','.join(bootstrappable_dims),
        'backend': backend,
        'weighting': 'r_squared',
        'bootstrap_se_method': 'unweighted_std',
        'confidence_level': confidence_interval
    }
    
    # Check if we have conditioning info
    if conditioning_info is not None:
        cond_var = conditioning_info['var_name']
        cond_bins = conditioning_info['bins']
        cond_bin_idx = conditioning_info.get('bin_idx', 0)
        
        # Add conditioning bin centers to coordinates
        cond_bin_centers = 0.5 * (cond_bins[:-1] + cond_bins[1:])
        coords['cond_bin'] = [cond_bin_centers[cond_bin_idx]]
        
        # Add conditioning info to attributes
        attrs['conditioning_variable'] = cond_var
        attrs['conditioning_bin_edges'] = list(cond_bins)
        attrs['conditioning_bin_idx'] = cond_bin_idx
        
        # Prepare data variables with conditioning dimension
        data_vars = {
            'sf_spherical': (('phi', 'theta', 'r', 'cond_bin'), results['sfr'][:, :, :, np.newaxis]),
            'sf': (('r', 'cond_bin'), results['sf_means'][:, np.newaxis]),
            'error_isotropy': (('r', 'cond_bin'), eiso[:, np.newaxis]),
            'std_error': (('r', 'cond_bin'), results['sf_stds'][:, np.newaxis]),
            'ci_upper': (('r', 'cond_bin'), ci_upper[:, np.newaxis]),
            'ci_lower': (('r', 'cond_bin'), ci_lower[:, np.newaxis]),
            'error_homogeneity': (('r_subset', 'cond_bin'), ehom[:, np.newaxis]),
            'n_bootstrap': (('r', 'cond_bin'), results['bin_bootstraps'][:, np.newaxis]),
            'bin_density': (('r', 'cond_bin'), results['bin_density'][:, np.newaxis]),
            'point_counts': (('r', 'cond_bin'), results['point_counts'][:, np.newaxis]),
            'converged': (('r', 'cond_bin'), results['bin_status'][:, np.newaxis])
        }
    else:
        # Standard case without conditioning
        data_vars = {
            'sf_spherical': (('phi', 'theta', 'r'), results['sfr']),
            'sf': (('r'), results['sf_means']),
            'error_isotropy': (('r'), eiso),
            'std_error': (('r'), results['sf_stds']),
            'ci_upper': (('r'), ci_upper),
            'ci_lower': (('r'), ci_lower),
            'error_homogeneity': (('r_subset'), ehom),
            'n_bootstrap': (('r'), results['bin_bootstraps']),
            'bin_density': (('r'), results['bin_density']),
            'point_counts': (('r'), results['point_counts']),
            'converged': (('r'), results['bin_status'])
        }
    
    ds_iso = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    # Add bin edges
    ds_iso['r_bins'] = (('r_edge'), bins_config['r_bins'])
    ds_iso['theta_bins'] = (('theta_edge'), bins_config['theta_bins'])
    ds_iso['phi_bins'] = (('phi_edge'), bins_config['phi_bins'])
    
    return ds_iso

def _calculate_isotropy_error_3d(sfr, sf_means, window_size_theta, window_size_phi):
    """Calculate error of isotropy using sliding windows for 3D."""
    n_bins_phi, n_bins_theta, n_bins_r = sfr.shape
    eiso = np.zeros(n_bins_r)
    
    if n_bins_theta > window_size_theta and n_bins_phi > window_size_phi:
        indices_theta = sliding_window_view(
            np.arange(n_bins_theta),
            (n_bins_theta - window_size_theta + 1,),
            writeable=False
        )[::1]
        
        indices_phi = sliding_window_view(
            np.arange(n_bins_phi),
            (n_bins_phi - window_size_phi + 1,),
            writeable=False
        )[::1]
        
        n_samples_theta = len(indices_theta)
        n_samples_phi = len(indices_phi)
        
        for j in range(n_bins_r):
            angle_vals = []
            
            # Bootstrap across both angles
            for i_phi in range(n_samples_phi):
                phi_idx = indices_phi[i_phi]
                for i_theta in range(n_samples_theta):
                    theta_idx = indices_theta[i_theta]
                    
                    # Get mean SF across these angular windows
                    mean_sf = bn.nanmean(sfr[np.ix_(phi_idx, theta_idx, [j])])
                    
                    if not np.isnan(mean_sf):
                        angle_vals.append(mean_sf)
            
            # Calculate error as angular standard deviation
            if angle_vals:
                eiso[j] = np.std(angle_vals)
    
    return eiso


def _calculate_homogeneity_error_3d(sfr, window_size_r):
    """Calculate error of homogeneity for 3D."""
    n_bins_phi, n_bins_theta, n_bins_r = sfr.shape
    
    if n_bins_r > window_size_r:
        indices_r = sliding_window_view(
            np.arange(n_bins_r),
            (n_bins_r - window_size_r + 1,),
            writeable=False
        )[::1]
        
        n_samples_r = len(indices_r)
        r_subset_indices = indices_r[0]
        
        meanh = np.zeros(len(r_subset_indices))
        ehom = np.zeros(len(r_subset_indices))
        
        for i in range(n_samples_r):
            idx = indices_r[i]
            meanh += bn.nanmean(sfr[:, :, idx])
        
        meanh /= max(1, n_samples_r)
        
        for i in range(n_samples_r):
            idx = indices_r[i]
            ehom += np.abs(bn.nanmean(sfr[:, :, idx]) - meanh)
        
        ehom /= max(1, n_samples_r)
    else:
        r_subset_indices = np.arange(n_bins_r)
        meanh = bn.nanmean(sfr, axis=(0, 1))
        ehom = np.zeros_like(meanh)
    
    return ehom, r_subset_indices

###############################################################################################################################################################################################
