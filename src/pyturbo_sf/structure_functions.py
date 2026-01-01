"""List of All the structure functions."""

import numpy as np
import bottleneck as bn

from .utils import (
    fast_shift_1d,
    fast_shift_2d,
    fast_shift_3d,
    check_and_reorder_variables_2d,
    map_variables_by_pattern_2d,
    check_and_reorder_variables_3d,    
    calculate_time_diff_1d
)
from .core import get_boot_indexes_1d

##########################################################################1D######################################################################################
def calc_scalar_1d(subset, dim, variable_name, order, n_points, conditioning_var=None, conditioning_bins=None):
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
    conditioning_var : str, optional
        Name of variable to condition on (e.g., 'vorticity', 'temperature')
    conditioning_bins : list, optional
        Bin edges [T_lo, T_hi] for conditioning variable.
        
    Returns
    -------
    results : array
        Structure function values
    separations : array
        Separation values
    pair_counts : array
        Number of valid (origin, separation) pairs for each separation
    """
    # Arrays to store results
    results = np.full(n_points, np.nan)
    separations = np.full(n_points, 0.0)
    pair_counts = np.zeros(n_points, dtype=np.int64)
    
    # Get the scalar variable
    scalar_var = subset[variable_name].values
    
    # Get coordinate variable
    coord_var = subset[dim].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
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
        
        # Apply conditional averaging (on origin only) and count valid pairs
        if cond_mask is not None:
            sf_val_cond = np.where(cond_mask, sf_val, np.nan)
            valid_sf = ~np.isnan(sf_val_cond)
            if np.any(valid_sf):
                results[i] = np.mean(sf_val_cond[valid_sf])
            pair_counts[i] = np.sum(valid_sf)
        else:
            valid_sf = ~np.isnan(sf_val)
            if np.any(valid_sf):
                results[i] = np.mean(sf_val[valid_sf])
            pair_counts[i] = np.sum(valid_sf)
    
    return results, separations, pair_counts


def calc_scalar_scalar_1d(subset, dim, variables_names, order, n_points, conditioning_var=None, conditioning_bins=None):
    """
    Calculate scalar-scalar structure function: (dscalar1^n * dscalar2^k)
    
    With conditional masking: D_ss^(αβ)(x,r) = ⟨[δs1]^n [δs2]^k I_α(x)I_β(x+r)⟩ / P_αβ
    
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
    conditioning_var : str, optional
        Name of variable to condition on (e.g., 'vorticity', 'temperature')
    conditioning_bins : list, optional
        Bin edges [T_lo, T_hi] for conditioning variable.
        
    Returns
    -------
    results : array
        Structure function values
    separations : array
        Separation values
    pair_counts : array
        Number of valid (origin, separation) pairs for each separation
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
    pair_counts = np.zeros(n_points, dtype=np.int64)
    
    # Get the scalar variables
    scalar_var1 = subset[var1].values
    scalar_var2 = subset[var2].values
    
    # Get coordinate variable
    coord_var = subset[dim].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
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
        
        # Apply conditional averaging (on origin only) and count valid pairs
        if cond_mask is not None:
            sf_val_cond = np.where(cond_mask, sf_val, np.nan)
            valid_sf = ~np.isnan(sf_val_cond)
            if np.any(valid_sf):
                results[i] = np.mean(sf_val_cond[valid_sf])
            pair_counts[i] = np.sum(valid_sf)
        else:
            valid_sf = ~np.isnan(sf_val)
            if np.any(valid_sf):
                results[i] = np.mean(sf_val[valid_sf])
            pair_counts[i] = np.sum(valid_sf)
    
    return results, separations, pair_counts
    
def calculate_structure_function_1d(ds, dim, variables_names, order, fun='scalar', nb=0, 
                                   spacing=None, num_bootstrappable=0, boot_indexes=None, bootsize=None,
                                   conditioning_var=None, conditioning_bins=None):
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
    conditioning_var : str, optional
        Name of variable to condition on (e.g., 'vorticity', 'temperature')
    conditioning_bins : list, optional
        Conditions for masking
        
    Returns
    -------
    results : array
        Structure function values
    separations : array
        Separation values
    pair_counts : array
        Number of valid (origin, separation) pairs for each separation
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
        results, separations, pair_counts = calc_scalar_1d(subset, dim, variable_name, order, n_points, conditioning_var, conditioning_bins)
        
    elif fun == 'scalar_scalar':
        results, separations, pair_counts = calc_scalar_scalar_1d(subset, dim, variables_names, order, n_points, conditioning_var, conditioning_bins)
        
    else:
        raise ValueError(f"Unsupported function type: {fun}. Only 'scalar' and 'scalar_scalar' are supported.")
        
    return results, separations, pair_counts

##################################################################################################################################################################

##########################################################################2D######################################################################################

def calc_longitudinal_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate longitudinal structure function: (du*dx + dv*dy)^n / |r|^n
    or (du*dx + dw*dz)^n / |r|^n or (dv*dy + dw*dz)^n / |r|^n depending on the plane.
    
    With conditional masking: D_L^(αβ)(x,r) = ⟨[δu_L]^n I_α(x)I_β(x+r)⟩ / P_αβ
    
    Returns
    -------
    results : array
        Mean SF value for each separation
    dx_vals, dy_vals : array
        Mean separation distances
    pair_counts : array
        Number of valid (origin, separation) pairs for each separation
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check and reorder variables if needed based on plane
    var1, var2 = check_and_reorder_variables_2d(variables_names, dims)
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Project velocity difference onto separation direction (longitudinal)
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                delta_parallel = dcomp1 * (dx/norm)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                delta_parallel = dcomp2 * (dy/norm)
            else:
                delta_parallel = dcomp1 * (dx/norm) + dcomp2 * (dy/norm)
            
            # Compute structure function
            sf_val = (delta_parallel) ** order
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts

def calc_transverse_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate transverse structure function: (du*dy - dv*dx)^n / |r|^n
    or (du*dz - dw*dx)^n / |r|^n or (dv*dz - dw*dy)^n / |r|^n depending on the plane.
    
    With conditional masking: D_T^(αβ)(x,r) = ⟨[δu_T]^n I_α(x)I_β(x+r)⟩ / P_αβ
    
    Returns
    -------
    results, dx_vals, dy_vals, pair_counts
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check and reorder variables if needed based on plane
    var1, var2 = check_and_reorder_variables_2d(variables_names, dims, fun='transverse')
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate transverse component
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                delta_perp = dcomp2
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                delta_perp = dcomp1
            else:
                delta_perp = dcomp1 * (dy/norm) - dcomp2 * (dx/norm)
            
            # Compute structure function
            sf_val = (delta_perp) ** order
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts
    
def calc_default_vel_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate default velocity structure function with conditional masking.
    
    D_ij^(αβ)(x,r) = ⟨[u_i(x+r) - u_i(x)][u_j(x+r) - u_j(x)]I_α(x)I_β(x+r)⟩ / P_αβ
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain 2 or 3 velocity components)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
    conditioning_var : str, optional
        Name of variable to condition on (e.g., 'vorticity', 'temperature')
    conditioning_bins : list, optional
        Conditions for masking
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) not in [2, 3]:
        raise ValueError(f"Default velocity structure function requires 2 or 3 velocity components, got {len(variables_names)}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Handle 2D case (2 components)
    if len(variables_names) == 2:
        var1, var2 = check_and_reorder_variables_2d(variables_names, dims, fun='default_vel')
        var3 = None
    else:  # 3D case (3 components)
        var1, var2, var3 = variables_names
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    comp3_var = subset[var3].values if var3 is not None else None
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all separations
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Calculate structure function
            if var3 is None:
                sf_val = (dcomp1 ** order) + (dcomp2 ** order)
            else:
                dcomp3 = fast_shift_2d(comp3_var, iy, ix) - comp3_var
                sf_val = (dcomp1 ** order) + (dcomp2 ** order) + (dcomp3 ** order)
            
            # Store separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts
    
def calc_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate scalar structure function: (dscalar^n)
    
    With conditional masking: D_s^(αβ)(x,r) = ⟨[s(x+r) - s(x)]^n I_α(x)I_β(x+r)⟩ / P_αβ
    """
    if len(variables_names) != 1:
        raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(variables_names)}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    scalar_name = variables_names[0]
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the scalar variable
    scalar_var = subset[scalar_name].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate scalar difference
            dscalar = fast_shift_2d(scalar_var, iy, ix) - scalar_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate scalar structure function
            sf_val = dscalar ** order
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts
    
def calc_scalar_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate scalar-scalar structure function: (dscalar1^n * dscalar2^k)
    
    With conditional masking: D_s1s2^(αβ)(x,r) = ⟨[s1(x+r) - s1(x)]^n [s2(x+r) - s2(x)]^k I_α(x)I_β(x+r)⟩ / P_αβ
    """
    if len(variables_names) != 2:
        raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for scalar-scalar structure function, got {order}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    n, k = order
    var1, var2 = variables_names
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the scalar variables
    scalar_var1 = subset[var1].values
    scalar_var2 = subset[var2].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate scalars difference
            dscalar1 = fast_shift_2d(scalar_var1, iy, ix) - scalar_var1
            dscalar2 = fast_shift_2d(scalar_var2, iy, ix) - scalar_var2
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate scalar-scalar structure function
            sf_val = (dscalar1 ** n) * (dscalar2 ** k)
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts

def calc_longitudinal_transverse_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate cross longitudinal-transverse structure function: (du_longitudinal^n * du_transverse^k)
    
    With conditional masking: D_LT^(αβ)(x,r) = ⟨[δu_L]^n [δu_T]^k I_α(x)I_β(x+r)⟩ / P_αβ
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    n, k = order
    var1, var2 = check_and_reorder_variables_2d(variables_names, dims, fun='longitudinal_transverse')
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1.0e-10)
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Project velocity difference onto separation direction
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                delta_parallel = dcomp1 * (dx/norm)
                delta_perp = dcomp2
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                delta_parallel = dcomp2 * (dy/norm)
                delta_perp = dcomp1
            else:
                delta_parallel = dcomp1 * (dx/norm) + dcomp2 * (dy/norm)
                delta_perp = dcomp1 * (dy/norm) - dcomp2 * (dx/norm)
            
            # Calculate longitudinal-transverse structure function
            sf_val = (delta_parallel ** n) * (delta_perp ** k)
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts
    
def calc_longitudinal_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate cross longitudinal-scalar structure function: (du_longitudinal^n * dscalar^k)
    
    With conditional masking: D_Ls^(αβ)(x,r) = ⟨[δu_L]^n [δs]^k I_α(x)I_β(x+r)⟩ / P_αβ
    """
    if len(variables_names) != 3:
        raise ValueError(f"Longitudinal-scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-scalar structure function, got {order}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    n, k = order
    
    # Check and reorder variables if needed based on plane
    tmp = check_and_reorder_variables_2d(variables_names, dims, fun='longitudinal_scalar')
    vel_vars, scalar_var = tmp[:2], tmp[-1]
    var1, var2 = vel_vars
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components and scalar
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    scalar_var_values = subset[scalar_var].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
        print(f"Using (y, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
        print(f"Using (z, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
        print(f"Using (z, y) plane with components {var1}, {var2} and scalar {scalar_var}")
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
        print(f"Using {dims} with components {var1}, {var2} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1.0e-10)
            
            # Calculate velocity and scalar differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            dscalar = fast_shift_2d(scalar_var_values, iy, ix) - scalar_var_values
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Project velocity difference onto separation direction
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                delta_parallel = dcomp1 * (dx/norm)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                delta_parallel = dcomp2 * (dy/norm)
            else:
                delta_parallel = dcomp1 * (dx/norm) + dcomp2 * (dy/norm)
            
            # Calculate longitudinal-scalar structure function
            sf_val = (delta_parallel ** n) * (dscalar ** k)
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts
    
def calc_transverse_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate cross transverse-scalar structure function: (du_transverse^n * dscalar^k)
    
    With conditional masking: D_Ts^(αβ)(x,r) = ⟨[δu_T]^n [δs]^k I_α(x)I_β(x+r)⟩ / P_αβ
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse-scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    n, k = order
    
    # Check and reorder variables if needed based on plane
    tmp = check_and_reorder_variables_2d(variables_names, dims, fun='transverse_scalar')
    vel_vars, scalar_var = tmp[:2], tmp[-1]
    var1, var2 = vel_vars
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components and scalar
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    scalar_var_values = subset[scalar_var].values
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
        print(f"Using (y, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
        print(f"Using (z, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
        print(f"Using (z, y) plane with components {var1}, {var2} and scalar {scalar_var}")
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
        print(f"Using {dims} with components {var1}, {var2} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1.0e-10)
            
            # Calculate velocity and scalar differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            dscalar = fast_shift_2d(scalar_var_values, iy, ix) - scalar_var_values
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate transverse component
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                delta_perp = dcomp2
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                delta_perp = dcomp1
            else:
                delta_perp = dcomp1 * (dy/norm) - dcomp2 * (dx/norm)
            
            # Calculate transverse-scalar structure function
            sf_val = (delta_perp ** n) * (dscalar ** k)
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts
    
def calc_advective_2d(subset, variables_names, order, dims, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate advective structure function: 
    - 2D: (du*deltaadv_u + dv*deltaadv_v)^n
    - 3D: (du*deltaadv_u + dv*deltaadv_v + dw*deltaadv_w)^n
    
    With conditional masking: D_adv^(αβ)(x,r) = ⟨[advective_term]^n I_α(x)I_β(x+r)⟩ / P_αβ
    """
    if len(variables_names) not in [4, 6]:
        raise ValueError(f"Advective structure function requires 4 (2D) or 6 (3D) velocity components, got {len(variables_names)}")
    
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Determine if we're in 2D or 3D mode
    is_3d = len(variables_names) == 6
    
    # Extract regular and advective velocity components
    vel_vars = []
    adv_vars = []
    
    for var in variables_names:
        if var.startswith('adv_') or 'adv' in var.lower():
            adv_vars.append(var)
        else:
            vel_vars.append(var)
    
    # Check if we have the right number of components
    expected_vel_count = 3 if is_3d else 2
    if len(vel_vars) != expected_vel_count or len(adv_vars) != expected_vel_count:
        if is_3d:
            vel_vars = variables_names[:3]
            adv_vars = variables_names[3:]
        else:
            vel_vars = variables_names[:2]
            adv_vars = variables_names[2:]
    
    # Handle 2D case (4 components)
    if not is_3d:
        # Define expected components based on plane
        if dims == ['y', 'x']:
            expected_components = ['u', 'v']
        elif dims == ['z', 'x']:
            expected_components = ['u', 'w']
        elif dims == ['z', 'y']:
            expected_components = ['v', 'w']
        else:
            expected_components = ['comp1', 'comp2']
        
        def map_to_components(vars_list, expected):
            if len(vars_list) != len(expected):
                raise ValueError(f"Expected {len(expected)} components, got {len(vars_list)}")
                
            result = [None] * len(expected)
            
            for i, exp in enumerate(expected):
                for var in vars_list:
                    if exp in var.lower():
                        result[i] = var
                        break
            
            if None in result:
                return vars_list
                
            return result
        
        var1, var2 = map_to_components(vel_vars, expected_components)
        advvar1, advvar2 = map_to_components(adv_vars, expected_components)
        var3 = None
        advvar3 = None
    else:
        var1, var2, var3 = vel_vars
        advvar1, advvar2, advvar3 = adv_vars
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    pair_counts = np.zeros(ny * nx, dtype=np.int64)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    comp3_var = subset[var3].values if var3 is not None else None
    advcomp1_var = subset[advvar1].values
    advcomp2_var = subset[advvar2].values
    advcomp3_var = subset[advvar3].values if advvar3 is not None else None
    
    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        x_coord = subset.x.values
        y_coord = subset.z.values
    elif dims == ['z', 'y']:
        x_coord = subset.y.values
        y_coord = subset.z.values
    else:
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Skip zero separation
            if iy == 0 and ix == 0:
                idx += 1
                continue
            # Compute actual physical separation
            if time_dims[dims[1]]:
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Calculate advective velocity differences
            dadvcomp1 = fast_shift_2d(advcomp1_var, iy, ix) - advcomp1_var
            dadvcomp2 = fast_shift_2d(advcomp2_var, iy, ix) - advcomp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate advective structure function
            if not is_3d:
                advective_term = dcomp1 * dadvcomp1 + dcomp2 * dadvcomp2
            else:
                dcomp3 = fast_shift_2d(comp3_var, iy, ix) - comp3_var
                dadvcomp3 = fast_shift_2d(advcomp3_var, iy, ix) - advcomp3_var
                advective_term = dcomp1 * dadvcomp1 + dcomp2 * dadvcomp2 + dcomp3 * dadvcomp3
            
            sf_val = advective_term ** order
            
            # Apply conditional averaging (on origin only) and count valid pairs
            if cond_mask is not None:
                sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                results[idx] = bn.nanmean(sf_val_cond)
                pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
            else:
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
            
            idx += 1
            
    return results, dx_vals, dy_vals, pair_counts

def calculate_structure_function_2d(ds, dims, variables_names, order, fun='longitudinal', 
                                  nbx=0, nby=0, spacing=None, num_bootstrappable=0, 
                                  bootstrappable_dims=None, boot_indexes=None, time_dims=None,
                                  conditioning_var=None, conditioning_bins=None):
    """
    Main function to calculate structure functions based on specified type.
    
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
    fun : str, optional
        Type of structure function
    nbx, nby : int, optional
        Bootstrap indices for x and y dimensions
    spacing : dict or int, optional
        Spacing value to use
    num_bootstrappable : int, optional
        Number of bootstrappable dimensions
    bootstrappable_dims : list, optional
        List of bootstrappable dimensions
    boot_indexes : dict, optional
        Dictionary with spacing values as keys and boot indexes as values
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Start with the full dataset
    subset = ds
    
    # Only subset bootstrappable dimensions
    if num_bootstrappable > 0 and bootstrappable_dims:
        # Get boot indexes for bootstrappable dimensions
        if boot_indexes and spacing is not None:
            if isinstance(spacing, int):
                sp_value = spacing
            else:
                # Get the spacing for a bootstrappable dimension
                for dim in bootstrappable_dims:
                    if dim in spacing:
                        sp_value = spacing[dim]
                        break
                else:
                    sp_value = 1  # Default if no matching dimension found
                
            indexes = boot_indexes.get(sp_value, {}) if sp_value in boot_indexes else {}
        else:
            indexes = {}
        
        # Create subset selection
        subset_dict = {}
        
        if num_bootstrappable == 1:
            # Only one dimension is bootstrappable
            bootstrap_dim = bootstrappable_dims[0]
            # Determine which index (nbx or nby) to use based on which dimension is bootstrappable
            nb_index = nbx if bootstrap_dim == dims[1] else nby
            # Add only the bootstrappable dimension to subset dict
            if indexes and bootstrap_dim in indexes and indexes[bootstrap_dim].shape[1] > nb_index:
                subset_dict[bootstrap_dim] = indexes[bootstrap_dim][:, nb_index]
        else:
            # Both dimensions are bootstrappable
            for i, dim in enumerate(dims):
                nb_index = nby if i == 0 else nbx
                if indexes and dim in indexes and indexes[dim].shape[1] > nb_index:
                    subset_dict[dim] = indexes[dim][:, nb_index]
        
        # Apply subsetting if needed
        if subset_dict:
            subset = ds.isel(subset_dict)
    
    # Check if the required variables exist in the dataset
    for var_name in variables_names:
        if var_name not in subset:
            raise ValueError(f"Variable {var_name} not found in dataset")
    
    # Get dimensions of the first variable to determine array sizes
    ny, nx = subset[variables_names[0]].shape
    
    # Create results array for structure function
    results = np.full(ny * nx, np.nan)
    
    # Arrays to store separation distances
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Calculate structure function based on specified type, passing time_dims information
    if fun == 'longitudinal':
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_2d(subset, variables_names, order, 
                                                    dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse':
        results, dx_vals, dy_vals, pair_counts = calc_transverse_2d(subset, variables_names, order, 
                                                  dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'default_vel':
        results, dx_vals, dy_vals, pair_counts = calc_default_vel_2d(subset, variables_names, order, 
                                                   dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'scalar':
        results, dx_vals, dy_vals, pair_counts = calc_scalar_2d(subset, variables_names, order, 
                                             dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'scalar_scalar':
        results, dx_vals, dy_vals, pair_counts = calc_scalar_scalar_2d(subset, variables_names, order, 
                                                    dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'longitudinal_transverse':
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_transverse_2d(subset, variables_names, order, 
                                                              dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'longitudinal_scalar':
        results, dx_vals, dy_vals, pair_counts = calc_longitudinal_scalar_2d(subset, variables_names, order, 
                                                          dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_scalar':
        results, dx_vals, dy_vals, pair_counts = calc_transverse_scalar_2d(subset, variables_names, order, 
                                                        dims, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'advective':
        results, dx_vals, dy_vals, pair_counts = calc_advective_2d(subset, variables_names, order, 
                                                 dims, ny, nx, time_dims, conditioning_var, conditioning_bins)                                                                                                  
    else:
        raise ValueError(f"Unsupported function type: {fun}")
    
    return results, dx_vals, dy_vals, pair_counts

##################################################################################################################################################################

##########################################################################3D######################################################################################

def calc_default_vel_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate default velocity structure function in 3D: (du^n + dv^n + dw^n)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain velocity components matching number of spatial dimensions)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Count spatial dimensions
    spatial_dims_count = sum(1 for dim in dims if not time_dims.get(dim, False))
    
    # Check that number of variables matches number of spatial dimensions
    if len(variables_names) != spatial_dims_count:
        raise ValueError(f"Default velocity structure function requires exactly {spatial_dims_count} velocity components "
                         f"for {spatial_dims_count} spatial dimensions, got {len(variables_names)}")
    
    # We need at least one spatial dimension
    if spatial_dims_count == 0:
        raise ValueError("Default velocity structure function requires at least one spatial dimension")
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Map variables to appropriate dimensions based on which dimensions are spatial
    vel_components = []
    spatial_dim_indices = []
    
    # Identify which dimensions are spatial and map variables to them
    for i, dim in enumerate(dims):
        if not time_dims[dim]:
            spatial_dim_indices.append(i)
    
    # Check if we have the right number of components
    if len(spatial_dim_indices) != len(variables_names):
        raise ValueError(f"Expected {len(spatial_dim_indices)} velocity components for {len(spatial_dim_indices)} spatial dimensions, "
                         f"got {len(variables_names)}")
    
    # Map variables to components based on spatial dimensions
    vel_vars = variables_names.copy()  # Work with a copy to avoid modifying the original
    
    # Get the velocity components
    vel_components = [subset[var].values for var in vel_vars]
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate velocity differences for each component
                dvel = []
                for component in vel_components:
                    dvel.append(fast_shift_3d(component, iz, iy, ix) - component)
                
                # Calculate default velocity structure function: sum of dv^order for each spatial dimension
                sf_val = np.zeros_like(dvel[0])
                for i in range(len(dvel)):
                    sf_val += dvel[i] ** order
                
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts

def calc_longitudinal_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D longitudinal structure function: (du*dx + dv*dy + dw*dz)^n / |r|^n
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain velocity components matching number of spatial dimensions)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Count spatial dimensions
    spatial_dims_count = sum(1 for dim in dims if not time_dims.get(dim, False))
    
    # Check that number of variables matches number of spatial dimensions
    if len(variables_names) != spatial_dims_count:
        raise ValueError(f"Longitudinal structure function requires exactly {spatial_dims_count} velocity components "
                         f"for {spatial_dims_count} spatial dimensions, got {len(variables_names)}")
    
    # We need at least one spatial dimension for longitudinal calculation
    if spatial_dims_count == 0:
        raise ValueError("Longitudinal structure function requires at least one spatial dimension")
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Map variables to appropriate dimensions based on which dimensions are spatial
    vel_vars = variables_names.copy()
    vel_components = []
    
    # Dictionary mapping spatial dimension indices to velocity components
    vel_by_dim = {}
    var_idx = 0
    
    # Identify which dimensions are spatial and map variables to them
    for i, dim in enumerate(dims):
        if not time_dims[dim]:
            if var_idx < len(vel_vars):
                vel_by_dim[i] = vel_vars[var_idx]
                var_idx += 1
    
    # Get the velocity components
    vel_components = {idx: subset[var].values for idx, var in vel_by_dim.items()}
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector (only using spatial dimensions)
                spatial_components = []
                if not time_dims[dims[2]]:
                    spatial_components.append(dx**2)
                if not time_dims[dims[1]]:
                    spatial_components.append(dy**2)
                if not time_dims[dims[0]]:
                    spatial_components.append(dz**2)
                
                if spatial_components:
                    # Calculate norm using only spatial components
                    norm = np.maximum(np.sqrt(sum(spatial_components)), 1e-10)
                else:
                    # If all dimensions are time (shouldn't happen with validation), use a default
                    norm = np.ones_like(dx)
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate velocity differences and project onto separation direction
                delta_parallel = np.zeros_like(dx)
                
                # Compute dot product between velocity differences and separation vector
                for dim_idx, vel_var in vel_by_dim.items():
                    # Get velocity component
                    vel_comp = vel_components[dim_idx]
                    
                    # Calculate velocity difference
                    dvel = fast_shift_3d(vel_comp, iz, iy, ix) - vel_comp
                    
                    # Get appropriate coordinate difference
                    if dim_idx == 0:  # z dimension
                        r_component = dz
                    elif dim_idx == 1:  # y dimension
                        r_component = dy
                    else:  # x dimension
                        r_component = dx
                    
                    # Add to dot product
                    delta_parallel += dvel * (r_component / norm)
                
                # Compute structure function
                sf_val = (delta_parallel) ** order
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts
    

def calc_transverse_ij(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D transverse structure function in ij (xy) plane: 
    The component of velocity difference perpendicular to separation in xy-plane
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse_ij structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for transverse calculation
    if time_dims.get(dims[1], False) and time_dims.get(dims[2], False):
        raise ValueError("Transverse_ij calculation requires at least one spatial dimension in the xy-plane")
    
    # Check and reorder variables if needed - ensure we get u and v
    u, v = check_and_reorder_variables_3d(variables_names, dims, fun='transverse_ij')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xy-plane (handle time dimensions)
                spatial_components_xy = []
                if not time_dims[dims[2]]:
                    spatial_components_xy.append(dx**2)
                if not time_dims[dims[1]]:
                    spatial_components_xy.append(dy**2)
                
                if spatial_components_xy:
                    # Calculate norm using only spatial components in xy-plane
                    norm_xy = np.maximum(np.sqrt(sum(spatial_components_xy)), 1e-10)
                else:
                    # If both x and y are time (shouldn't happen after validation), use a default
                    norm_xy = np.ones_like(dx)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xy-plane)
                # Handle cases where one dimension is time
                if time_dims[dims[2]]:  # x is time, y is spatial
                    delta_perp_ij = du  # Only consider u component
                elif time_dims[dims[1]]:  # y is time, x is spatial
                    delta_perp_ij = dv  # Only consider v component
                else:  # Both are spatial
                    delta_perp_ij = du * (dy/norm_xy) - dv * (dx/norm_xy)
                
                # Compute structure function
                sf_val = (delta_perp_ij) ** order
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_transverse_ik(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D transverse structure function in ik (xz) plane: 
    The component of velocity difference perpendicular to separation in xz-plane
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse_ik structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for transverse calculation
    if time_dims.get(dims[0], False) and time_dims.get(dims[2], False):
        raise ValueError("Transverse_ik calculation requires at least one spatial dimension in the xz-plane")
    
    # Check and reorder variables if needed - ensure we get u and w
    u, w = check_and_reorder_variables_3d(variables_names, dims, fun='transverse_ik')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xz-plane (handle time dimensions)
                spatial_components_xz = []
                if not time_dims[dims[2]]:
                    spatial_components_xz.append(dx**2)
                if not time_dims[dims[0]]:
                    spatial_components_xz.append(dz**2)
                
                if spatial_components_xz:
                    # Calculate norm using only spatial components in xz-plane
                    norm_xz = np.maximum(np.sqrt(sum(spatial_components_xz)), 1e-10)
                else:
                    # If both x and z are time (shouldn't happen after validation), use a default
                    norm_xz = np.ones_like(dx)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xz-plane)
                # Handle cases where one dimension is time
                if time_dims[dims[2]]:  # x is time, z is spatial
                    delta_perp_ik = du  # Only consider u component
                elif time_dims[dims[0]]:  # z is time, x is spatial
                    delta_perp_ik = dw  # Only consider w component
                else:  # Both are spatial
                    delta_perp_ik = dw * (dx/norm_xz) - du * (dz/norm_xz)
                
                # Compute structure function
                sf_val = (delta_perp_ik) ** order
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_transverse_jk(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D transverse structure function in jk (yz) plane: 
    The component of velocity difference perpendicular to separation in yz-plane
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse_jk structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for transverse calculation
    if time_dims.get(dims[0], False) and time_dims.get(dims[1], False):
        raise ValueError("Transverse_jk calculation requires at least one spatial dimension in the yz-plane")
    
    # Check and reorder variables if needed - ensure we get v and w
    v, w = check_and_reorder_variables_3d(variables_names, dims, fun='transverse_jk')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in yz-plane (handle time dimensions)
                spatial_components_yz = []
                if not time_dims[dims[1]]:
                    spatial_components_yz.append(dy**2)
                if not time_dims[dims[0]]:
                    spatial_components_yz.append(dz**2)
                
                if spatial_components_yz:
                    # Calculate norm using only spatial components in yz-plane
                    norm_yz = np.maximum(np.sqrt(sum(spatial_components_yz)), 1e-10)
                else:
                    # If both y and z are time (shouldn't happen after validation), use a default
                    norm_yz = np.ones_like(dy)
                
                # Calculate velocity differences
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in yz-plane)
                # Handle cases where one dimension is time
                if time_dims[dims[1]]:  # y is time, z is spatial
                    delta_perp_jk = dv  # Only consider v component
                elif time_dims[dims[0]]:  # z is time, y is spatial
                    delta_perp_jk = dw  # Only consider w component
                else:  # Both are spatial
                    delta_perp_jk = dv * (dz/norm_yz) - dw * (dy/norm_yz)
                
                # Compute structure function
                sf_val = (delta_perp_jk) ** order
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D scalar structure function: (dscalar^n)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain one scalar variable)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 1:
        raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Get the scalar variable name
    scalar_name = variables_names[0]
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the scalar variable
    scalar_var = subset[scalar_name].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Calculate scalar difference
                dscalar = fast_shift_3d(scalar_var, iz, iy, ix) - scalar_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate scalar structure function: dscalar^n
                sf_val = dscalar ** order
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_longitudinal_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D longitudinal-scalar structure function: (du_longitudinal^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (velocity components matching spatial dimensions, plus one scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Count spatial dimensions
    spatial_dims_count = sum(1 for dim in dims if not time_dims.get(dim, False))
    
    # Check that number of variables matches number of spatial dimensions + 1 scalar
    if len(variables_names) != spatial_dims_count + 1:
        raise ValueError(f"Longitudinal-scalar structure function requires {spatial_dims_count} velocity components "
                         f"plus 1 scalar for {spatial_dims_count} spatial dimensions, got {len(variables_names)} total")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-scalar structure function, got {order}")
    
    # We need at least one spatial dimension for longitudinal calculation
    if spatial_dims_count == 0:
        raise ValueError("Longitudinal-scalar structure function requires at least one spatial dimension")
    
    # Unpack order tuple
    n, k = order
    
    # Get the scalar variable (last in the list)
    scalar_var = variables_names[-1]
    
    # Get velocity variables (all but the last one)
    vel_vars = variables_names[:-1]
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Dictionary mapping spatial dimension indices to velocity components
    vel_by_dim = {}
    var_idx = 0
    
    # Identify which dimensions are spatial and map variables to them
    for i, dim in enumerate(dims):
        if not time_dims[dim]:
            if var_idx < len(vel_vars):
                vel_by_dim[i] = vel_vars[var_idx]
                var_idx += 1
    
    # Get the velocity components and scalar
    vel_components = {idx: subset[var].values for idx, var in vel_by_dim.items()}
    scalar_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-scalar with {len(vel_vars)} velocity components and scalar {scalar_var}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector (only using spatial dimensions)
                spatial_components = []
                if not time_dims[dims[2]]:
                    spatial_components.append(dx**2)
                if not time_dims[dims[1]]:
                    spatial_components.append(dy**2)
                if not time_dims[dims[0]]:
                    spatial_components.append(dz**2)
                
                if spatial_components:
                    # Calculate norm using only spatial components
                    norm = np.maximum(np.sqrt(sum(spatial_components)), 1e-10)
                else:
                    # If all dimensions are time (shouldn't happen with validation), use a default
                    norm = np.ones_like(dx)
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate scalar difference
                dscalar = fast_shift_3d(scalar_values, iz, iy, ix) - scalar_values
                
                # Calculate velocity differences and project onto separation direction
                delta_parallel = np.zeros_like(dx)
                
                # Compute dot product between velocity differences and separation vector
                for dim_idx, vel_var in vel_by_dim.items():
                    # Get velocity component
                    vel_comp = vel_components[dim_idx]
                    
                    # Calculate velocity difference
                    dvel = fast_shift_3d(vel_comp, iz, iy, ix) - vel_comp
                    
                    # Get appropriate coordinate difference
                    if dim_idx == 0:  # z dimension
                        r_component = dz
                    elif dim_idx == 1:  # y dimension
                        r_component = dy
                    else:  # x dimension
                        r_component = dx
                    
                    # Add to dot product
                    delta_parallel += dvel * (r_component / norm)
                
                # Calculate longitudinal-scalar structure function: delta_parallel^n * dscalar^k
                sf_val = (delta_parallel ** n) * (dscalar ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts

def calc_transverse_ij_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D transverse-scalar structure function in ij (xy) plane: 
    (du_transverse_ij^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components and a scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse_ij_scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for transverse calculation
    if time_dims.get(dims[1], False) and time_dims.get(dims[2], False):
        raise ValueError("Transverse_ij_scalar calculation requires at least one spatial dimension in the xy-plane")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get u, v, and scalar
    vel_vars = variables_names[:2]
    scalar_var = variables_names[2]
    u, v = check_and_reorder_variables_3d(vel_vars, dims, fun='transverse_ij')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    v_var = subset[v].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_ij_scalar with components {u}, {v} and scalar {scalar_var}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xy-plane (handle time dimensions)
                spatial_components_xy = []
                if not time_dims[dims[2]]:
                    spatial_components_xy.append(dx**2)
                if not time_dims[dims[1]]:
                    spatial_components_xy.append(dy**2)
                
                if spatial_components_xy:
                    # Calculate norm using only spatial components in xy-plane
                    norm_xy = np.maximum(np.sqrt(sum(spatial_components_xy)), 1e-10)
                else:
                    # If both x and y are time (shouldn't happen after validation), use a default
                    norm_xy = np.ones_like(dx)
                
                # Calculate velocity and scalar differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xy-plane)
                # Handle cases where one dimension is time
                if time_dims[dims[2]]:  # x is time, y is spatial
                    delta_perp_ij = du  # Only consider u component
                elif time_dims[dims[1]]:  # y is time, x is spatial
                    delta_perp_ij = dv  # Only consider v component
                else:  # Both are spatial
                    delta_perp_ij = du * (dy/norm_xy) - dv * (dx/norm_xy)
                
                # Calculate transverse-scalar structure function: delta_perp_ij^n * dscalar^k
                sf_val = (delta_perp_ij ** n) * (dscalar ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_transverse_ik_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D transverse-scalar structure function in ik (xz) plane: 
    (du_transverse_ik^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components and a scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse_ik_scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for transverse calculation
    if time_dims.get(dims[0], False) and time_dims.get(dims[2], False):
        raise ValueError("Transverse_ik_scalar calculation requires at least one spatial dimension in the xz-plane")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get u, w, and scalar
    vel_vars = variables_names[:2]
    scalar_var = variables_names[2]
    u, w = check_and_reorder_variables_3d(vel_vars, dims, fun='transverse_ik')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_ik_scalar with components {u}, {w} and scalar {scalar_var}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xz-plane (handle time dimensions)
                spatial_components_xz = []
                if not time_dims[dims[2]]:
                    spatial_components_xz.append(dx**2)
                if not time_dims[dims[0]]:
                    spatial_components_xz.append(dz**2)
                
                if spatial_components_xz:
                    # Calculate norm using only spatial components in xz-plane
                    norm_xz = np.maximum(np.sqrt(sum(spatial_components_xz)), 1e-10)
                else:
                    # If both x and z are time (shouldn't happen after validation), use a default
                    norm_xz = np.ones_like(dx)
                
                # Calculate velocity and scalar differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xz-plane)
                # Handle cases where one dimension is time
                if time_dims[dims[2]]:  # x is time, z is spatial
                    delta_perp_ik = du  # Only consider u component
                elif time_dims[dims[0]]:  # z is time, x is spatial
                    delta_perp_ik = dw  # Only consider w component
                else:  # Both are spatial
                    delta_perp_ik = du * (dz/norm_xz) - dw * (dx/norm_xz)
                
                # Calculate transverse-scalar structure function: delta_perp_ik^n * dscalar^k
                sf_val = (delta_perp_ik ** n) * (dscalar ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_transverse_jk_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D transverse-scalar structure function in jk (yz) plane: 
    (du_transverse_jk^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components and a scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse_jk_scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for transverse calculation
    if time_dims.get(dims[0], False) and time_dims.get(dims[1], False):
        raise ValueError("Transverse_jk_scalar calculation requires at least one spatial dimension in the yz-plane")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get v, w, and scalar
    vel_vars = variables_names[:2]
    scalar_var = variables_names[2]
    v, w = check_and_reorder_variables_3d(vel_vars, dims, fun='transverse_jk')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components and scalar
    v_var = subset[v].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_jk_scalar with components {v}, {w} and scalar {scalar_var}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in yz-plane (handle time dimensions)
                spatial_components_yz = []
                if not time_dims[dims[1]]:
                    spatial_components_yz.append(dy**2)
                if not time_dims[dims[0]]:
                    spatial_components_yz.append(dz**2)
                
                if spatial_components_yz:
                    # Calculate norm using only spatial components in yz-plane
                    norm_yz = np.maximum(np.sqrt(sum(spatial_components_yz)), 1e-10)
                else:
                    # If both y and z are time (shouldn't happen after validation), use a default
                    norm_yz = np.ones_like(dy)
                
                # Calculate velocity and scalar differences
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in yz-plane)
                # Handle cases where one dimension is time
                if time_dims[dims[1]]:  # y is time, z is spatial
                    delta_perp_jk = dv  # Only consider v component
                elif time_dims[dims[0]]:  # z is time, y is spatial
                    delta_perp_jk = dw  # Only consider w component
                else:  # Both are spatial
                    delta_perp_jk = dv * (dz/norm_yz) - dw * (dy/norm_yz)
                
                # Calculate transverse-scalar structure function: delta_perp_jk^n * dscalar^k
                sf_val = (delta_perp_jk ** n) * (dscalar ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts

def calc_longitudinal_transverse_ij(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D longitudinal-transverse structure function in ij (xy) plane: 
    (du_longitudinal_ij^n * du_transverse_ij^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse_ij structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for longitudinal-transverse calculation
    if time_dims.get(dims[1], False) and time_dims.get(dims[2], False):
        raise ValueError("Longitudinal-transverse_ij calculation requires at least one spatial dimension in the xy-plane")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get u and v
    u, v = check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal_transverse_ij')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_ij with components {u}, {v}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xy-plane (handle time dimensions)
                spatial_components_xy = []
                if not time_dims[dims[2]]:
                    spatial_components_xy.append(dx**2)
                if not time_dims[dims[1]]:
                    spatial_components_xy.append(dy**2)
                
                if spatial_components_xy:
                    # Calculate norm using only spatial components in xy-plane
                    norm_xy = np.maximum(np.sqrt(sum(spatial_components_xy)), 1e-10)
                else:
                    # If both x and y are time (shouldn't happen after validation), use a default
                    norm_xy = np.ones_like(dx)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate longitudinal and transverse components with time handling
                if time_dims[dims[2]]:  # x is time, y is spatial
                    # Can't calculate proper longitudinal and transverse components
                    # Just use raw velocity differences
                    delta_parallel = du
                    delta_perp = dv
                elif time_dims[dims[1]]:  # y is time, x is spatial
                    # Can't calculate proper longitudinal and transverse components
                    # Just use raw velocity differences
                    delta_parallel = dv
                    delta_perp = du
                else:  # Both are spatial
                    # Project velocity difference onto separation direction in xy-plane (longitudinal)
                    delta_parallel = (du * dx + dv * dy) / norm_xy
                    
                    # Calculate transverse component (perpendicular to separation in xy-plane)
                    delta_perp = (du * dy - dv * dx) / norm_xy
                
                # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
                sf_val = (delta_parallel ** n) * (delta_perp ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_longitudinal_transverse_ik(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D longitudinal-transverse structure function in ik (xz) plane: 
    (du_longitudinal_ik^n * du_transverse_ik^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse_ik structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for longitudinal-transverse calculation
    if time_dims.get(dims[0], False) and time_dims.get(dims[2], False):
        raise ValueError("Longitudinal-transverse_ik calculation requires at least one spatial dimension in the xz-plane")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get u and w
    u, w = check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal_transverse_ik')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_ik with components {u}, {w}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xz-plane (handle time dimensions)
                spatial_components_xz = []
                if not time_dims[dims[2]]:
                    spatial_components_xz.append(dx**2)
                if not time_dims[dims[0]]:
                    spatial_components_xz.append(dz**2)
                
                if spatial_components_xz:
                    # Calculate norm using only spatial components in xz-plane
                    norm_xz = np.maximum(np.sqrt(sum(spatial_components_xz)), 1e-10)
                else:
                    # If both x and z are time (shouldn't happen after validation), use a default
                    norm_xz = np.ones_like(dx)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate longitudinal and transverse components with time handling
                if time_dims[dims[2]]:  # x is time, z is spatial
                    # Can't calculate proper longitudinal and transverse components
                    # Just use raw velocity differences
                    delta_parallel = du
                    delta_perp = dw
                elif time_dims[dims[0]]:  # z is time, x is spatial
                    # Can't calculate proper longitudinal and transverse components
                    # Just use raw velocity differences
                    delta_parallel = dw
                    delta_perp = du
                else:  # Both are spatial
                    # Project velocity difference onto separation direction in xz-plane (longitudinal)
                    delta_parallel = (du * dx + dw * dz) / norm_xz
                    
                    # Calculate transverse component (perpendicular to separation in xz-plane)
                    delta_perp = (du * dz - dw * dx) / norm_xz
                
                # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
                sf_val = (delta_parallel ** n) * (delta_perp ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_longitudinal_transverse_jk(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D longitudinal-transverse structure function in jk (yz) plane: 
    (du_longitudinal_jk^n * du_transverse_jk^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse_jk structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check if both dimensions are time - not suitable for longitudinal-transverse calculation
    if time_dims.get(dims[0], False) and time_dims.get(dims[1], False):
        raise ValueError("Longitudinal-transverse_jk calculation requires at least one spatial dimension in the yz-plane")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get v and w
    v, w = check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal_transverse_jk')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_jk with components {v}, {w}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in yz-plane (handle time dimensions)
                spatial_components_yz = []
                if not time_dims[dims[1]]:
                    spatial_components_yz.append(dy**2)
                if not time_dims[dims[0]]:
                    spatial_components_yz.append(dz**2)
                
                if spatial_components_yz:
                    # Calculate norm using only spatial components in yz-plane
                    norm_yz = np.maximum(np.sqrt(sum(spatial_components_yz)), 1e-10)
                else:
                    # If both y and z are time (shouldn't happen after validation), use a default
                    norm_yz = np.ones_like(dy)
                
                # Calculate velocity differences
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate longitudinal and transverse components with time handling
                if time_dims[dims[1]]:  # y is time, z is spatial
                    # Can't calculate proper longitudinal and transverse components
                    # Just use raw velocity differences
                    delta_parallel = dv
                    delta_perp = dw
                elif time_dims[dims[0]]:  # z is time, y is spatial
                    # Can't calculate proper longitudinal and transverse components
                    # Just use raw velocity differences
                    delta_parallel = dw
                    delta_perp = dv
                else:  # Both are spatial
                    # Project velocity difference onto separation direction in yz-plane (longitudinal)
                    delta_parallel = (dv * dy + dw * dz) / norm_yz
                    
                    # Calculate transverse component (perpendicular to separation in yz-plane)
                    delta_perp = (dv * dz - dw * dy) / norm_yz
                
                # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
                sf_val = (delta_parallel ** n) * (delta_perp ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts

def calc_scalar_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D scalar-scalar structure function: (dscalar1^n * dscalar2^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two scalar variables)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar variables, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for scalar-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Unpack order tuple
    n, k = order
    
    # Get the scalar variable names
    scalar1_name, scalar2_name = variables_names
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Get the scalar variables
    scalar1_var = subset[scalar1_name].values
    scalar2_var = subset[scalar2_name].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D scalar-scalar structure function for {scalar1_name} and {scalar2_name}")
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Calculate scalar differences
                dscalar1 = fast_shift_3d(scalar1_var, iz, iy, ix) - scalar1_var
                dscalar2 = fast_shift_3d(scalar2_var, iz, iy, ix) - scalar2_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate scalar-scalar structure function: dscalar1^n * dscalar2^k
                sf_val = (dscalar1 ** n) * (dscalar2 ** k)
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_advective_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate 3D advective structure function: (du*deltaadv_u + dv*deltaadv_v + dw*deltaadv_w)^n
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain velocity and advective components for spatial dimensions)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Count spatial dimensions
    spatial_dims_count = sum(1 for dim in dims if not time_dims.get(dim, False))
    
    # Check that number of variables matches 2 * number of spatial dimensions (vel + adv for each)
    if len(variables_names) != 2 * spatial_dims_count:
        raise ValueError(f"Advective structure function requires {2 * spatial_dims_count} components "
                         f"({spatial_dims_count} velocities and {spatial_dims_count} advective terms) "
                         f"for {spatial_dims_count} spatial dimensions, got {len(variables_names)}")
    
    # We need at least one spatial dimension
    if spatial_dims_count == 0:
        raise ValueError("Advective structure function requires at least one spatial dimension")
    
    # Split variables into velocity and advective components
    vel_vars = variables_names[:spatial_dims_count]
    adv_vars = variables_names[spatial_dims_count:]
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Dictionary mapping spatial dimension indices to velocity and advective components
    vel_by_dim = {}
    adv_by_dim = {}
    var_idx = 0
    
    # Identify which dimensions are spatial and map variables to them
    for i, dim in enumerate(dims):
        if not time_dims[dim]:
            if var_idx < len(vel_vars):
                vel_by_dim[i] = vel_vars[var_idx]
                adv_by_dim[i] = adv_vars[var_idx]
                var_idx += 1
    
    # Get the velocity and advective components
    vel_components = {idx: subset[var].values for idx, var in vel_by_dim.items()}
    adv_components = {idx: subset[var].values for idx, var in adv_by_dim.items()}
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    

    # Create conditioning mask (at origin only)
    if conditioning_var is not None and conditioning_var in subset and conditioning_bins is not None:
        cond_var = subset[conditioning_var].values
        T_lo, T_hi = conditioning_bins[0], conditioning_bins[1]
        cond_mask = (cond_var >= T_lo) & (cond_var < T_hi)
    else:
        cond_mask = None

    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate advective structure function
                advective_term = np.zeros_like(dx)
                
                # Compute sum of velocity * advective differences
                for dim_idx in vel_by_dim.keys():
                    # Get components
                    vel_comp = vel_components[dim_idx]
                    adv_comp = adv_components[dim_idx]
                    
                    # Calculate differences
                    dvel = fast_shift_3d(vel_comp, iz, iy, ix) - vel_comp
                    dadv = fast_shift_3d(adv_comp, iz, iy, ix) - adv_comp
                    
                    # Add to advective term
                    advective_term += dvel * dadv
                
                # Raise to specified order
                sf_val = advective_term ** order
                # Apply conditional averaging (on origin only)
                if cond_mask is not None:
                    sf_val_cond = np.where(cond_mask, sf_val, np.nan)
                    results[idx] = bn.nanmean(sf_val_cond)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val_cond))
                else:
                    results[idx] = bn.nanmean(sf_val)
                    pair_counts[idx] = np.sum(~np.isnan(sf_val))
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts


def calc_pressure_work_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None, conditioning_var=None, conditioning_bins=None):
    """
    Calculate pressure work structure function: (∇_j(δΦ δu_j))^n
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing pressure and velocity components
    variables_names : list
        List of variable names (first is pressure, followed by velocity components for spatial dimensions)
    order : int
        Order of the structure function
    dims : list
        List of dimension names (should be ['z', 'y', 'x'])
    nz, ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Count spatial dimensions
    spatial_dims_count = sum(1 for dim in dims if not time_dims.get(dim, False))
    
    # Check that number of variables matches number of spatial dimensions + 1 (pressure)
    if len(variables_names) != spatial_dims_count + 1:
        raise ValueError(f"Pressure work calculation requires 1 pressure variable plus {spatial_dims_count} velocity components "
                         f"for {spatial_dims_count} spatial dimensions, got {len(variables_names)} total")
    
    # We need at least one spatial dimension
    if spatial_dims_count == 0:
        raise ValueError("Pressure work calculation requires at least one spatial dimension")
    
    if dims != ['z', 'y', 'x']:
        raise ValueError(f"Expected dimensions ['z', 'y', 'x'], got {dims}")
    
    # Extract pressure (first variable)
    pressure_var = variables_names[0]
    
    # Extract velocity variables (remaining variables)
    vel_vars = variables_names[1:]
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)  # Default to 1.0 (no conditioning for this function)
    pair_counts = np.zeros(nz * ny * nx, dtype=np.int64)
    
    # Dictionary mapping spatial dimension indices to velocity components
    vel_by_dim = {}
    var_idx = 0
    
    # Identify which dimensions are spatial and map variables to them
    for i, dim in enumerate(dims):
        if not time_dims[dim]:
            if var_idx < len(vel_vars):
                vel_by_dim[i] = vel_vars[var_idx]
                var_idx += 1
    
    # Get the pressure and velocity components
    pressure_values = subset[pressure_var].values
    vel_components = {idx: subset[var].values for idx, var in vel_by_dim.items()}
    
    # Get coordinate variables as 3D arrays
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    # Convert 1D coordinates to 3D arrays if needed
    if len(x_coord.shape) == 1:
        X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord, indexing='ij')
    else:
        X, Y, Z = x_coord, y_coord, z_coord
    
    # Loop through all points (we still need to loop over shifts)
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Skip zero separation
                if iz == 0 and iy == 0 and ix == 0:
                    idx += 1
                    continue
                # Compute actual physical separation, handling time dimensions properly
                if time_dims[dims[2]]:  # x dimension is time
                    dx = calculate_time_diff_1d(x_coord, ix)
                else:
                    dx = fast_shift_3d(X, iz, iy, ix) - X
                    
                if time_dims[dims[1]]:  # y dimension is time
                    dy = calculate_time_diff_1d(y_coord, iy)
                else:
                    dy = fast_shift_3d(Y, iz, iy, ix) - Y
                    
                if time_dims[dims[0]]:  # z dimension is time
                    dz = calculate_time_diff_1d(z_coord, iz)
                else:
                    dz = fast_shift_3d(Z, iz, iy, ix) - Z
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate pressure difference
                dP = fast_shift_3d(pressure_values, iz, iy, ix) - pressure_values
                
                # Calculate divergence using vectorized operations
                div_flux = np.zeros_like(pressure_values)
                
                # Calculate the product of pressure and velocity increments for each spatial dimension
                for dim_idx, vel_var in vel_by_dim.items():
                    # Get velocity component
                    vel_comp = vel_components[dim_idx]
                    
                    # Calculate velocity difference
                    dvel = fast_shift_3d(vel_comp, iz, iy, ix) - vel_comp
                    
                    # Calculate pressure-velocity flux
                    P_vel_flux = dP * dvel
                    
                    # Calculate gradient only for spatial dimensions
                    if dim_idx == 0:  # z dimension is spatial
                        # For z direction
                        dz_central = np.zeros_like(Z)
                        dz_central[1:-1, :, :] = (Z[2:, :, :] - Z[:-2, :, :])
                        # Use forward/backward differences at boundaries
                        dz_central[0, :, :] = (Z[1, :, :] - Z[0, :, :]) * 2
                        dz_central[-1, :, :] = (Z[-1, :, :] - Z[-2, :, :]) * 2
                        
                        dP_vel_flux_dz = np.zeros_like(P_vel_flux)
                        dP_vel_flux_dz[1:-1, :, :] = (P_vel_flux[2:, :, :] - P_vel_flux[:-2, :, :]) / dz_central[1:-1, :, :]
                        # Use forward/backward differences at boundaries
                        dP_vel_flux_dz[0, :, :] = (P_vel_flux[1, :, :] - P_vel_flux[0, :, :]) / (dz_central[0, :, :] / 2)
                        dP_vel_flux_dz[-1, :, :] = (P_vel_flux[-1, :, :] - P_vel_flux[-2, :, :]) / (dz_central[-1, :, :] / 2)
                        
                        # Add to divergence
                        div_flux += dP_vel_flux_dz
                        
                    elif dim_idx == 1:  # y dimension is spatial
                        # For y direction
                        dy_central = np.zeros_like(Y)
                        dy_central[:, 1:-1, :] = (Y[:, 2:, :] - Y[:, :-2, :])
                        # Use forward/backward differences at boundaries
                        dy_central[:, 0, :] = (Y[:, 1, :] - Y[:, 0, :]) * 2
                        dy_central[:, -1, :] = (Y[:, -1, :] - Y[:, -2, :]) * 2
                        
                        dP_vel_flux_dy = np.zeros_like(P_vel_flux)
                        dP_vel_flux_dy[:, 1:-1, :] = (P_vel_flux[:, 2:, :] - P_vel_flux[:, :-2, :]) / dy_central[:, 1:-1, :]
                        # Use forward/backward differences at boundaries
                        dP_vel_flux_dy[:, 0, :] = (P_vel_flux[:, 1, :] - P_vel_flux[:, 0, :]) / (dy_central[:, 0, :] / 2)
                        dP_vel_flux_dy[:, -1, :] = (P_vel_flux[:, -1, :] - P_vel_flux[:, -2, :]) / (dy_central[:, -1, :] / 2)
                        
                        # Add to divergence
                        div_flux += dP_vel_flux_dy
                        
                    elif dim_idx == 2:  # x dimension is spatial
                        # For x direction
                        dx_central = np.zeros_like(X)
                        dx_central[:, :, 1:-1] = (X[:, :, 2:] - X[:, :, :-2])
                        # Use forward/backward differences at boundaries
                        dx_central[:, :, 0] = (X[:, :, 1] - X[:, :, 0]) * 2
                        dx_central[:, :, -1] = (X[:, :, -1] - X[:, :, -2]) * 2
                        
                        dP_vel_flux_dx = np.zeros_like(P_vel_flux)
                        dP_vel_flux_dx[:, :, 1:-1] = (P_vel_flux[:, :, 2:] - P_vel_flux[:, :, :-2]) / dx_central[:, :, 1:-1]
                        # Use forward/backward differences at boundaries
                        dP_vel_flux_dx[:, :, 0] = (P_vel_flux[:, :, 1] - P_vel_flux[:, :, 0]) / (dx_central[:, :, 0] / 2)
                        dP_vel_flux_dx[:, :, -1] = (P_vel_flux[:, :, -1] - P_vel_flux[:, :, -2]) / (dx_central[:, :, -1] / 2)
                        
                        # Add to divergence
                        div_flux += dP_vel_flux_dx
                
                # Raise to specified order
                sf_val = div_flux ** order
                
                # Compute structure function
                results[idx] = bn.nanmean(sf_val)
                pair_counts[idx] = np.sum(~np.isnan(sf_val))
                # (No conditioning for pressure work)
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals, pair_counts
    
def calculate_structure_function_3d(ds, dims, variables_names, order, fun='longitudinal', 
                                  nbz=0, nby=0, nbx=0, spacing=None, num_bootstrappable=0,
                                  bootstrappable_dims=None, boot_indexes=None, time_dims=None,
                                  conditioning_var=None, conditioning_bins=None):
    """
    Main function to calculate structure functions based on specified type.
    
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
    fun : str, optional
        Type of structure function
    nbz, nby, nbx : int, optional
        Bootstrap indices for z, y, and x dimensions
    spacing : dict or int, optional
        Spacing value to use
    num_bootstrappable : int, optional
        Number of bootstrappable dimensions
    bootstrappable_dims : list, optional
        List of bootstrappable dimensions
    boot_indexes : dict, optional
        Dictionary with spacing values as keys and boot indexes as values
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Start with the full dataset
    subset = ds
    
    # Only subset bootstrappable dimensions
    if num_bootstrappable > 0 and bootstrappable_dims:
        # Get boot indexes for bootstrappable dimensions
        if boot_indexes and spacing is not None:
            if isinstance(spacing, int):
                sp_value = spacing
            else:
                # Get the spacing for a bootstrappable dimension
                for dim in bootstrappable_dims:
                    if dim in spacing:
                        sp_value = spacing[dim]
                        break
                else:
                    sp_value = 1  # Default if no matching dimension found
                
            indexes = boot_indexes.get(sp_value, {}) if sp_value in boot_indexes else {}
        else:
            indexes = {}
        
        # Create subset selection
        subset_dict = {}
        
        if num_bootstrappable == 1:
            # Only one dimension is bootstrappable
            bootstrap_dim = bootstrappable_dims[0]
            # Determine which index (nbz, nby, or nbx) to use based on which dimension is bootstrappable
            nb_index = nbz if bootstrap_dim == dims[0] else (nby if bootstrap_dim == dims[1] else nbx)
            # Add only the bootstrappable dimension to subset dict
            if indexes and bootstrap_dim in indexes and indexes[bootstrap_dim].shape[1] > nb_index:
                subset_dict[bootstrap_dim] = indexes[bootstrap_dim][:, nb_index]
        elif num_bootstrappable == 2:
            # Two dimensions are bootstrappable
            for i, dim in enumerate(dims):
                if dim in bootstrappable_dims:
                    nb_index = nbz if i == 0 else (nby if i == 1 else nbx)
                    if indexes and dim in indexes and indexes[dim].shape[1] > nb_index:
                        subset_dict[dim] = indexes[dim][:, nb_index]
        else:  # num_bootstrappable == 3
            # All three dimensions are bootstrappable
            for i, dim in enumerate(dims):
                nb_index = nbz if i == 0 else (nby if i == 1 else nbx)
                if indexes and dim in indexes and indexes[dim].shape[1] > nb_index:
                    subset_dict[dim] = indexes[dim][:, nb_index]
        
        # Apply subsetting if needed
        if subset_dict:
            subset = ds.isel(subset_dict)
    
    # Check if the required variables exist in the dataset
    for var_name in variables_names:
        if var_name not in subset:
            raise ValueError(f"Variable {var_name} not found in dataset")
    
    # Get dimensions of the first variable to determine array sizes
    var_dims = subset[variables_names[0]].dims
    nz = subset[variables_names[0]].shape[0]
    ny = subset[variables_names[0]].shape[1]
    nx = subset[variables_names[0]].shape[2]
    
    # Create results array for structure function
    results = np.full(nz * ny * nx, np.nan)
    
    # Arrays to store separation distances
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Calculate structure function based on specified type, passing time_dims information
    if fun == 'longitudinal':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_ij':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_ik':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_jk':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'scalar':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'scalar_scalar':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_scalar_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'longitudinal_scalar':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'longitudinal_transverse_ij':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'longitudinal_transverse_ik':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'longitudinal_transverse_jk':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_longitudinal_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_ij_scalar':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ij_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_ik_scalar':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_ik_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'transverse_jk_scalar':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_transverse_jk_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'advective':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_advective_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'pressure_work':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_pressure_work_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    elif fun == 'default_vel':
        results, dx_vals, dy_vals, dz_vals, pair_counts = calc_default_vel_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims, conditioning_var, conditioning_bins)
    else:
        raise ValueError(f"Unsupported function type: {fun}")
            
    return results, dx_vals, dy_vals, dz_vals, pair_counts

##################################################################################################################################################################
