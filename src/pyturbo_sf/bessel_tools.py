"""Energy Flux Decomposition Tools (2D only)

Computes spectral energy flux from advective structure functions using
Bessel function decomposition:

    Π(K) = -K/2 ∫₀^∞ SF̃(r) J₁(Kr) dr

where SF̃(r) is the angle-averaged advective structure function and J₁ is
the Bessel function of the first kind, order 1.

Reference: Plancherel theorem relating energy flux to real-space integral
involving advective SF and J₁ Bessel function.
"""

import numpy as np
import xarray as xr
import bottleneck as bn
from scipy import stats
from scipy.special import jv  # Bessel function of the first kind
from numpy.lib.stride_tricks import sliding_window_view

from .structure_functions import calculate_structure_function_2d

from .utils import (
    _calculate_confidence_intervals,
    _is_log_spaced,
    _calculate_quality_mask
)

# Valid function types for energy flux calculation
VALID_FLUX_FUNCTIONS = ['advective', 'scalar_scalar']


def _validate_flux_function(fun):
    """
    Validate that the structure function type is appropriate for energy flux.
    
    Parameters
    ----------
    fun : str
        Structure function type.
        
    Raises
    ------
    ValueError
        If fun is not in VALID_FLUX_FUNCTIONS.
    """
    if fun not in VALID_FLUX_FUNCTIONS:
        raise ValueError(
            f"Energy flux decomposition requires fun in {VALID_FLUX_FUNCTIONS}, "
            f"got '{fun}'. The Bessel J₁ transform is only physically meaningful "
            f"for advective-type structure functions."
        )


def _initialize_wavenumbers_2d(wavenumbers, ds, dims):
    """
    Initialize wavenumber configuration.
    
    Parameters
    ----------
    wavenumbers : array-like, dict, or None
        Wavenumber specification.
    ds : xarray.Dataset
        Input dataset for determining domain size.
    dims : list
        Dimension names.
        
    Returns
    -------
    dict
        Configuration with 'k' (wavenumbers) and metadata.
    """
    if wavenumbers is None:
        # Auto-generate based on domain size
        if dims == ['y', 'x']:
            Lx = float(ds.x.max() - ds.x.min())
            Ly = float(ds.y.max() - ds.y.min())
        elif dims == ['z', 'x']:
            Lx = float(ds.x.max() - ds.x.min())
            Ly = float(ds.z.max() - ds.z.min())
        elif dims == ['z', 'y']:
            Lx = float(ds.y.max() - ds.y.min())
            Ly = float(ds.z.max() - ds.z.min())
        else:
            Lx = float(ds[dims[1]].max() - ds[dims[1]].min())
            Ly = float(ds[dims[0]].max() - ds[dims[0]].min())
        
        L = np.sqrt(Lx * Ly)  # Characteristic length
        k_min = 2 * np.pi / L
        k_max = np.pi / min(Lx / ds.sizes[dims[1]], Ly / ds.sizes[dims[0]])
        
        # Logarithmically spaced wavenumbers
        k = np.logspace(np.log10(k_min), np.log10(k_max), 50)
        log_spaced = True
        
    elif isinstance(wavenumbers, dict):
        k = np.asarray(wavenumbers['k'])
        log_spaced = _is_log_spaced(k)
        
    else:
        k = np.asarray(wavenumbers)
        log_spaced = _is_log_spaced(k)
    
    return {
        'k': k,
        'n_k': len(k),
        'log_spaced': log_spaced,
        'k_min': k.min(),
        'k_max': k.max()
    }


def _initialize_r_bins_2d(r_bins, ds, dims, n_r_bins=100):
    """
    Initialize radial bin configuration for angle-averaging.
    
    Parameters
    ----------
    r_bins : array-like or None
        Radial bin edges. If None, auto-generate.
    ds : xarray.Dataset
        Input dataset for determining domain size.
    dims : list
        Dimension names.
    n_r_bins : int
        Number of radial bins if auto-generating. Default is 100.
        
    Returns
    -------
    dict
        Configuration with 'r_edges', 'r_centers', 'dr'.
    """
    if r_bins is None:
        # Auto-generate based on domain size
        if dims == ['y', 'x']:
            Lx = float(ds.x.max() - ds.x.min())
            Ly = float(ds.y.max() - ds.y.min())
            dx = Lx / ds.sizes['x']
            dy = Ly / ds.sizes['y']
        elif dims == ['z', 'x']:
            Lx = float(ds.x.max() - ds.x.min())
            Ly = float(ds.z.max() - ds.z.min())
            dx = Lx / ds.sizes['x']
            dy = Ly / ds.sizes['z']
        elif dims == ['z', 'y']:
            Lx = float(ds.y.max() - ds.y.min())
            Ly = float(ds.z.max() - ds.z.min())
            dx = Lx / ds.sizes['y']
            dy = Ly / ds.sizes['z']
        else:
            Lx = float(ds[dims[1]].max() - ds[dims[1]].min())
            Ly = float(ds[dims[0]].max() - ds[dims[0]].min())
            dx = Lx / ds.sizes[dims[1]]
            dy = Ly / ds.sizes[dims[0]]
        
        # r_min from grid spacing, r_max from domain diagonal
        r_min = np.sqrt(dx**2 + dy**2)
        r_max = np.sqrt(Lx**2 + Ly**2) / 2  # Half diagonal
        
        # Linear spacing for proper integration
        r_edges = np.linspace(r_min, r_max, n_r_bins + 1)
    else:
        r_edges = np.asarray(r_bins)
    
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = np.diff(r_edges)
    
    return {
        'r_edges': r_edges,
        'r_centers': r_centers,
        'dr': dr,
        'n_r': len(r_centers)
    }


def _initialize_flux_config_2d(k, r_config, n_bins_theta):
    """
    Initialize energy flux configuration.
    
    Parameters
    ----------
    k : array
        Wavenumber values.
    r_config : dict
        Radial bin configuration from _initialize_r_bins_2d.
    n_bins_theta : int
        Number of angular bins for isotropy diagnostics.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    n_k = len(k)
    
    # Angular bins for isotropy calculation
    theta_bins = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    
    return {
        'k': k,
        'n_k': n_k,
        'r_edges': r_config['r_edges'],
        'r_centers': r_config['r_centers'],
        'dr': r_config['dr'],
        'n_r': r_config['n_r'],
        'theta_bins': theta_bins,
        'theta_centers': theta_centers,
        'n_bins_theta': n_bins_theta,
        'log_spaced': _is_log_spaced(k)
    }


def _bin_sf_by_radius_2d(results, dx_vals, dy_vals, r_config, theta_bins=None):
    """
    Bin structure function values by radius to get angle-averaged SF(r).
    
    This computes SF̃(r) = (1/2π) ∫₀^{2π} SF(r,θ) dθ
    
    Parameters
    ----------
    results : array
        Structure function values (flattened).
    dx_vals : array
        X-separations (flattened).
    dy_vals : array
        Y-separations (flattened).
    r_config : dict
        Radial bin configuration.
    theta_bins : array, optional
        Angular bin edges for isotropy diagnostics.
        
    Returns
    -------
    sf_r : array
        Angle-averaged SF at each radial bin center (n_r,).
    sf_r_std : array
        Standard deviation in each radial bin (n_r,).
    counts_r : array
        Number of points in each radial bin (n_r,).
    sf_theta_r : array or None
        SF binned by (theta, r) if theta_bins provided, else None.
    """
    # Filter valid data
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    
    r_valid = np.sqrt(valid_dx**2 + valid_dy**2)
    
    r_edges = r_config['r_edges']
    n_r = r_config['n_r']
    
    # Initialize outputs
    sf_r = np.full(n_r, np.nan)
    sf_r_std = np.full(n_r, np.nan)
    counts_r = np.zeros(n_r, dtype=np.int_)
    
    if len(valid_results) == 0:
        sf_theta_r = None
        if theta_bins is not None:
            n_theta = len(theta_bins) - 1
            sf_theta_r = np.full((n_theta, n_r), np.nan)
        return sf_r, sf_r_std, counts_r, sf_theta_r
    
    # Bin by radius
    r_indices = np.digitize(r_valid, r_edges) - 1
    r_indices = np.clip(r_indices, 0, n_r - 1)
    
    for r_idx in range(n_r):
        mask = r_indices == r_idx
        if np.sum(mask) > 0:
            bin_values = valid_results[mask]
            sf_r[r_idx] = np.mean(bin_values)
            counts_r[r_idx] = len(bin_values)
            if len(bin_values) > 1:
                sf_r_std[r_idx] = np.std(bin_values, ddof=1)
    
    # Angular binning for isotropy diagnostics
    sf_theta_r = None
    if theta_bins is not None:
        theta_valid = np.arctan2(valid_dy, valid_dx)
        n_theta = len(theta_bins) - 1
        sf_theta_r = np.full((n_theta, n_r), np.nan)
        
        theta_indices = np.digitize(theta_valid, theta_bins) - 1
        theta_indices = np.clip(theta_indices, 0, n_theta - 1)
        
        for theta_idx in range(n_theta):
            for r_idx in range(n_r):
                mask = (theta_indices == theta_idx) & (r_indices == r_idx)
                if np.sum(mask) > 0:
                    sf_theta_r[theta_idx, r_idx] = np.mean(valid_results[mask])
    
    return sf_r, sf_r_std, counts_r, sf_theta_r


def _compute_energy_flux_2d(sf_r, r_centers, dr, k):
    """
    Compute energy flux using Bessel J₁ transform.
    
    Π(K) = -K/2 ∫₀^∞ SF̃(r) J₁(Kr) dr
         ≈ -K/2 Σᵢ SF̃(rᵢ) J₁(K·rᵢ) Δrᵢ
    
    Parameters
    ----------
    sf_r : array
        Angle-averaged structure function at radial bin centers (n_r,).
    r_centers : array
        Radial bin centers (n_r,).
    dr : array
        Radial bin widths (n_r,).
    k : array
        Wavenumbers to evaluate at (n_k,).
        
    Returns
    -------
    energy_flux : array
        Energy flux Π(K) at each wavenumber (n_k,).
    """
    n_k = len(k)
    energy_flux = np.full(n_k, np.nan)
    
    # Mask for valid (non-NaN) SF values
    valid_mask = ~np.isnan(sf_r)
    if not np.any(valid_mask):
        return energy_flux
    
    sf_valid = sf_r[valid_mask]
    r_valid = r_centers[valid_mask]
    dr_valid = dr[valid_mask]
    
    # Compute J₁(kr) for all (k, r) pairs: shape (n_k, n_valid_r)
    kr = np.outer(k, r_valid)
    J1_values = jv(1, kr)
    
    # Compute integral: Π(K) = -K/2 Σᵢ SF̃(rᵢ) J₁(K·rᵢ) Δrᵢ
    # Vectorized: sum over r dimension
    integral = np.sum(J1_values * sf_valid * dr_valid, axis=1)
    energy_flux = -k / 2.0 * integral
    
    return energy_flux


def _process_no_bootstrap_flux_2d(ds, dims, variables_names, order, fun, 
                                   k, r_config, n_theta, time_dims, 
                                   conditioning_var=None, conditioning_bins=None):
    """
    Handle the special case of no bootstrappable dimensions for energy flux.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    dims : list
        Dimension names.
    variables_names : list
        Variable names for SF calculation.
    order : float
        SF order (should be 1 for advective).
    fun : str
        Function type ('advective' or 'scalar_scalar').
    k : array
        Wavenumbers.
    r_config : dict
        Radial bin configuration.
    n_theta : int
        Number of angular bins.
    time_dims : list
        Time dimension names.
    conditioning_var : str, optional
        Conditioning variable name.
    conditioning_bins : array, optional
        Conditioning bin edges.
        
    Returns
    -------
    energy_flux : array
        Energy flux at each wavenumber (n_k,).
    flux_stds : array
        Standard deviation estimates (n_k,).
    point_counts : array
        Point counts per wavenumber (n_k,).
    flux_theta_k : array
        Angular distribution of flux (n_theta, n_k).
    config : dict
        Configuration dictionary.
    """
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
    
    # Initialize configuration
    config = _initialize_flux_config_2d(k, r_config, n_theta)
    
    # Bin SF by radius (angle-averaging)
    sf_r, sf_r_std, counts_r, sf_theta_r = _bin_sf_by_radius_2d(
        results, dx_vals, dy_vals, r_config, config['theta_bins']
    )
    
    # Compute energy flux
    energy_flux = _compute_energy_flux_2d(
        sf_r, config['r_centers'], config['dr'], k
    )
    
    # Store radial SF in config for diagnostics
    config['sf_r'] = sf_r
    config['sf_r_std'] = sf_r_std
    
    # Estimate flux uncertainty from SF uncertainty
    # Propagate error through the integral
    flux_stds = np.full(config['n_k'], np.nan)
    valid_r_mask = ~np.isnan(sf_r_std)
    if np.any(valid_r_mask):
        sf_std_valid = sf_r_std[valid_r_mask]
        r_valid = config['r_centers'][valid_r_mask]
        dr_valid = config['dr'][valid_r_mask]
        
        kr = np.outer(k, r_valid)
        J1_values = jv(1, kr)
        
        # Error propagation: σ_Π² = (K/2)² Σᵢ (J₁·Δr·σ_SF)²
        variance = np.sum((J1_values * dr_valid * sf_std_valid)**2, axis=1)
        flux_stds = (k / 2.0) * np.sqrt(variance)
    
    # Compute angular flux distribution
    flux_theta_k = np.full((config['n_bins_theta'], config['n_k']), np.nan)
    if sf_theta_r is not None:
        for theta_idx in range(config['n_bins_theta']):
            sf_theta = sf_theta_r[theta_idx, :]
            flux_theta_k[theta_idx, :] = _compute_energy_flux_2d(
                sf_theta, config['r_centers'], config['dr'], k
            )
    
    # Point counts per wavenumber (all wavenumbers use all valid points)
    total_valid_points = int(np.sum(counts_r))
    point_counts = np.full(config['n_k'], total_valid_points, dtype=np.int_)
    
    return energy_flux, flux_stds, point_counts, flux_theta_k, config


def _calculate_isotropy_error_flux_2d(flux_theta_k, energy_flux, window_size_theta):
    """
    Calculate isotropy error for energy flux.
    
    Parameters
    ----------
    flux_theta_k : array
        Angular distribution of flux (n_theta, n_k).
    energy_flux : array
        Angle-averaged flux (n_k,).
    window_size_theta : int
        Window size for sliding average.
        
    Returns
    -------
    eiso : array
        Isotropy error at each wavenumber (n_k,).
    """
    n_bins_theta, n_k = flux_theta_k.shape
    eiso = np.zeros(n_k)
    
    if n_bins_theta > window_size_theta:
        indices_theta = sliding_window_view(
            np.arange(n_bins_theta),
            (n_bins_theta - window_size_theta + 1,),
            writeable=False
        )[::1]
        
        n_samples_theta = len(indices_theta)
        
        for i in range(n_samples_theta):
            idx = indices_theta[i]
            mean_flux = bn.nanmean(flux_theta_k[idx, :], axis=0)
            eiso += np.abs(mean_flux - energy_flux)
        
        eiso /= max(1, n_samples_theta)
    
    return eiso


def _calculate_homogeneity_error_flux_2d(flux_theta_k, window_size_k):
    """
    Calculate homogeneity error for energy flux.
    
    Parameters
    ----------
    flux_theta_k : array
        Angular distribution of flux (n_theta, n_k).
    window_size_k : int
        Window size for sliding average.
        
    Returns
    -------
    ehom : array
        Homogeneity error at subset of wavenumbers.
    k_subset_indices : array
        Indices of wavenumbers in subset.
    """
    n_bins_theta, n_k = flux_theta_k.shape
    
    if n_k > window_size_k:
        indices_k = sliding_window_view(
            np.arange(n_k),
            (n_k - window_size_k + 1,),
            writeable=False
        )[::1]
        
        n_samples_k = len(indices_k)
        k_subset_indices = indices_k[0]
        
        meanh = np.zeros(len(k_subset_indices))
        ehom = np.zeros(len(k_subset_indices))
        
        for i in range(n_samples_k):
            idx = indices_k[i]
            meanh += bn.nanmean(flux_theta_k[:, idx], axis=0)
        
        meanh /= max(1, n_samples_k)
        
        for i in range(n_samples_k):
            idx = indices_k[i]
            ehom += np.abs(bn.nanmean(flux_theta_k[:, idx], axis=0) - meanh)
        
        ehom /= max(1, n_samples_k)
    else:
        k_subset_indices = np.arange(n_k)
        meanh = bn.nanmean(flux_theta_k, axis=0)
        ehom = np.zeros_like(meanh)
    
    return ehom, k_subset_indices


def _calculate_wavenumber_density_2d(point_counts, k):
    """Calculate normalized wavenumber density."""
    total_points = np.sum(point_counts)
    if total_points == 0:
        return np.zeros_like(k, dtype=np.float32)
    
    # For log-spaced wavenumbers, use dk/k as the "bin width"
    if _is_log_spaced(k):
        dk = np.diff(np.log(k))
        dk = np.append(dk, dk[-1])  # Extend for last bin
    else:
        dk = np.diff(k)
        dk = np.append(dk, dk[-1])
    
    # Point counts are per radial bin, replicate for k
    bin_density = np.full_like(k, total_points / len(k), dtype=np.float32)
    
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
    
    return bin_density


def _create_flux_dataset_2d(results, config, order, fun, window_size_theta,
                            window_size_k, convergence_eps, max_nbootstrap,
                            initial_nbootstrap, bootstrappable_dims, backend,
                            variables_names, confidence_interval,
                            conditioning_info=None):
    """
    Create output xarray Dataset for energy flux.
    
    Parameters
    ----------
    results : dict
        Results dictionary containing:
        - energy_flux: Energy flux at each wavenumber
        - flux_stds: Standard deviations
        - point_counts: Point counts per wavenumber
        - flux_theta_k: Angular distribution
        - bin_bootstraps, bin_status, etc.
    config : dict
        Configuration dictionary.
    order : float
        Structure function order.
    fun : str
        Function type.
    window_size_theta : int
        Window size for isotropy error.
    window_size_k : int
        Window size for homogeneity error.
    convergence_eps : float
        Convergence epsilon.
    max_nbootstrap : int
        Maximum bootstrap iterations.
    initial_nbootstrap : int
        Initial bootstrap iterations.
    bootstrappable_dims : list
        Bootstrappable dimension names.
    backend : str
        Parallel backend.
    variables_names : list
        Variable names.
    confidence_interval : float
        Confidence level.
    conditioning_info : dict, optional
        Conditioning information.
        
    Returns
    -------
    xarray.Dataset
        Dataset with energy flux results.
    """
    # Calculate error metrics
    eiso = _calculate_isotropy_error_flux_2d(
        results['flux_theta_k'], results['energy_flux'], window_size_theta
    )
    ehom, k_subset_indices = _calculate_homogeneity_error_flux_2d(
        results['flux_theta_k'], window_size_k
    )
    
    # Use pre-computed CIs if available
    if 'ci_lower' in results and 'ci_upper' in results:
        ci_lower = results['ci_lower']
        ci_upper = results['ci_upper']
    else:
        ci_upper, ci_lower = _calculate_confidence_intervals(
            results['energy_flux'], results['flux_stds'], 
            results['point_counts'], confidence_interval
        )
    
    # Calculate quality mask
    mask_quality = _calculate_quality_mask(
        results['energy_flux'], results['flux_stds'], results['point_counts'],
        eiso, results['bin_status'],
        min_points=10,
        max_isotropy_error=None,
        max_std_ratio=None
    )
    
    # Build coordinates
    coords = {
        'k': config['k'],
        'k_subset': config['k'][k_subset_indices],
        'theta': config['theta_centers'],
        'r': config['r_centers']
    }
    
    # Build attributes
    attrs = {
        'description': 'Spectral energy flux from advective structure function',
        'formula': 'Pi(K) = -K/2 * integral(SF_tilde(r) * J1(Kr) * dr)',
        'order': str(order),
        'function_type': fun,
        'window_size_theta': window_size_theta,
        'window_size_k': window_size_k,
        'convergence_eps': convergence_eps,
        'max_nbootstrap': max_nbootstrap,
        'initial_nbootstrap': initial_nbootstrap,
        'wavenumber_type': 'logarithmic' if config['log_spaced'] else 'linear',
        'variables': variables_names if isinstance(variables_names, str) else ','.join(variables_names),
        'bootstrappable_dimensions': ','.join(bootstrappable_dims),
        'backend': backend,
        'weighting': 'bessel_j1_flux',
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
            'energy_flux': (('k', 'cond_bin'), results['energy_flux'][:, np.newaxis]),
            'error_isotropy': (('k', 'cond_bin'), eiso[:, np.newaxis]),
            'std_error': (('k', 'cond_bin'), results['flux_stds'][:, np.newaxis]),
            'ci_upper': (('k', 'cond_bin'), ci_upper[:, np.newaxis]),
            'ci_lower': (('k', 'cond_bin'), ci_lower[:, np.newaxis]),
            'error_homogeneity': (('k_subset', 'cond_bin'), ehom[:, np.newaxis]),
            'mask_quality': (('k', 'cond_bin'), mask_quality[:, np.newaxis]),
            'n_bootstrap': (('k', 'cond_bin'), results['bin_bootstraps'][:, np.newaxis]),
            'bin_density': (('k', 'cond_bin'), results['bin_density'][:, np.newaxis]),
            'point_counts': (('k', 'cond_bin'), results['point_counts'][:, np.newaxis]),
            'converged': (('k', 'cond_bin'), results['bin_status'][:, np.newaxis])
        }
    else:
        # Standard case without conditioning
        data_vars = {
            'energy_flux': (('k',), results['energy_flux']),
            'error_isotropy': (('k',), eiso),
            'std_error': (('k',), results['flux_stds']),
            'ci_upper': (('k',), ci_upper),
            'ci_lower': (('k',), ci_lower),
            'error_homogeneity': (('k_subset',), ehom),
            'mask_quality': (('k',), mask_quality),
            'n_bootstrap': (('k',), results['bin_bootstraps']),
            'bin_density': (('k',), results['bin_density']),
            'point_counts': (('k',), results['point_counts']),
            'converged': (('k',), results['bin_status'])
        }
    
    ds_flux = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs
    )
    
    # Add bin edges
    ds_flux['theta_bins'] = (('theta_edge',), config['theta_bins'])
    ds_flux['r_bins'] = (('r_edge',), config['r_edges'])
    
    return ds_flux

