"""Two-dimensional structure function calculations."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import bottleneck as bn
import gc
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime

from .core import (validate_dataset_2d, setup_bootsize_2d, calculate_adaptive_spacings_2d,
                  compute_boot_indexes_2d, get_boot_indexes_2d, is_time_dimension)
from .utils import (fast_shift_2d, check_and_reorder_variables_2d, map_variables_by_pattern_2d,
                  calculate_time_diff_1d)
                   
##################################Structure Functions Types########################################
def calc_longitudinal_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate longitudinal structure function: (du*dx + dv*dy)^n / |r|^n
    or (du*dx + dw*dz)^n / |r|^n or (dv*dy + dw*dz)^n / |r|^n depending on the plane.
    
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
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
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
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            # For mixed time-space cases, we need to be careful about calculating norm
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - use only spatial component for projection
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - use only spatial component for projection
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                # Both spatial or both time (handled as spatial)
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Project velocity difference onto separation direction (longitudinal)
            # For mixed time-space cases, project only onto spatial component
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - project onto spatial component only
                delta_parallel = dcomp1 * (dx/norm)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - project onto spatial component only
                delta_parallel = dcomp2 * (dy/norm)
            else:
                # Both spatial or both time (handled as spatial)
                delta_parallel = dcomp1 * (dx/norm) + dcomp2 * (dy/norm)
            
            # Compute structure function
            sf_val = (delta_parallel) ** order
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals
    

def calc_transverse_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate transverse structure function: (du*dy - dv*dx)^n / |r|^n
    or (du*dz - dw*dx)^n / |r|^n or (dv*dz - dw*dy)^n / |r|^n depending on the plane.
    
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
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
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
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            # For mixed time-space cases, we need to be careful about calculating norm
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - use only spatial component for projection
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - use only spatial component for projection
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                # Both spatial or both time (handled as spatial)
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate transverse component (perpendicular to separation direction)
            # For mixed time-space cases, transverse calculations need special handling
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - transverse is zero for 1D spatial case
                # We're left with only the time derivative contribution
                delta_perp = dcomp2
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - transverse is zero for 1D spatial case
                # We're left with only the time derivative contribution
                delta_perp = dcomp1
            else:
                # Both spatial or both time (handled as spatial)
                delta_perp = dcomp1 * (dy/norm) - dcomp2 * (dx/norm)
            
            # Compute structure function
            sf_val = (delta_perp) ** order
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals


def calc_default_vel_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate default velocity structure function: (du^n + dv^n)
    or (du^n + dw^n) or (dv^n + dw^n) depending on the plane.
    
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
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Default velocity structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Check and reorder variables if needed based on plane
    var1, var2 = check_and_reorder_variables_2d(variables_names, dims, fun='default_vel')
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate default velocity structure function: du^n + dv^n
            # This calculation doesn't depend on separation direction, so no special handling needed for time dimensions
            sf_val = (dcomp1 ** order) + (dcomp2 ** order)
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals

def calc_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate scalar structure function: (dscalar^n)
    
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
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 1:
        raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Get the scalar variable name
    scalar_name = variables_names[0]
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the scalar variable
    scalar_var = subset[scalar_name].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate scalar difference
            dscalar = fast_shift_2d(scalar_var, iy, ix) - scalar_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate scalar structure function: dscalar^n
            sf_val = dscalar ** order
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals
    
def calc_scalar_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate scalar-scalar structure function: (dscalar1^n * dscalar2^k)
    
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
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for scalar-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed based on plane
    var1, var2 = variables_names
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the scalar variable
    scalar_var1 = subset[var1].values
    scalar_var2 = subset[var2].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Calculate scalars difference
            dscalar1 = fast_shift_2d(scalar_var1, iy, ix) - scalar_var1
            dscalar2 = fast_shift_2d(scalar_var2, iy, ix) - scalar_var2
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate scalar-scalar structure function: dscalar^n * dscalar^k
            sf_val = (dscalar1 ** n) * (dscalar2 ** k)
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals
    
def calc_longitudinal_transverse_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate cross longitudinal-transverse structure function: (du_longitudinal^n * du_transverse^k)
    
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
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed based on plane
    var1, var2 = check_and_reorder_variables_2d(variables_names, dims, fun='longitudinal_transverse')
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            # For mixed time-space cases, we need to be careful about calculating norm
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - use only spatial component for projection
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - use only spatial component for projection
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                # Both spatial or both time (handled as spatial)
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1.0e-10)
            
            # Calculate velocity differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Project velocity difference onto separation direction (longitudinal)
            # For mixed time-space cases, project only onto spatial component
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - longitudinal uses only spatial component
                delta_parallel = dcomp1 * (dx/norm)
                # Transverse is the time component
                delta_perp = dcomp2
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - longitudinal uses only spatial component
                delta_parallel = dcomp2 * (dy/norm)
                # Transverse is the time component
                delta_perp = dcomp1
            else:
                # Both spatial or both time (handled as spatial)
                delta_parallel = dcomp1 * (dx/norm) + dcomp2 * (dy/norm)
                delta_perp = dcomp1 * (dy/norm) - dcomp2 * (dx/norm)
            
            # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
            sf_val = (delta_parallel ** n) * (delta_perp ** k)
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals
    
def calc_longitudinal_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate cross longitudinal-scalar structure function: (du_longitudinal^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components and one scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Longitudinal-scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed based on plane
    tmp = check_and_reorder_variables_2d(variables_names, dims, fun='longitudinal_scalar')
    vel_vars, scalar_var = tmp[:2], tmp[-1]
    var1, var2 = vel_vars
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the velocity components and scalar
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
        print(f"Using (y, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
        print(f"Using (z, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
        print(f"Using (z, y) plane with components {var1}, {var2} and scalar {scalar_var}")
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
        print(f"Using {dims} with components {var1}, {var2} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            # For mixed time-space cases, we need to be careful about calculating norm
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - use only spatial component for projection
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - use only spatial component for projection
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                # Both spatial or both time (handled as spatial)
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1.0e-10)
            
            # Calculate velocity and scalar differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            dscalar = fast_shift_2d(scalar_var_values, iy, ix) - scalar_var_values
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Project velocity difference onto separation direction (longitudinal)
            # For mixed time-space cases, project only onto spatial component
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - longitudinal uses only spatial component
                delta_parallel = dcomp1 * (dx/norm)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - longitudinal uses only spatial component
                delta_parallel = dcomp2 * (dy/norm)
            else:
                # Both spatial or both time (handled as spatial)
                delta_parallel = dcomp1 * (dx/norm) + dcomp2 * (dy/norm)
            
            # Calculate longitudinal-scalar structure function: delta_parallel^n * dscalar^k
            sf_val = (delta_parallel ** n) * (dscalar ** k)
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals

def calc_transverse_scalar_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate cross transverse-scalar structure function: (du_transverse^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain two velocity components and one scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse-scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed based on plane
    tmp = check_and_reorder_variables_2d(variables_names, dims, fun='transverse_scalar')
    vel_vars, scalar_var = tmp[:2], tmp[-1]
    var1, var2 = vel_vars
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the velocity components and scalar
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
        print(f"Using (y, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
        print(f"Using (z, x) plane with components {var1}, {var2} and scalar {scalar_var}")
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
        print(f"Using (z, y) plane with components {var1}, {var2} and scalar {scalar_var}")
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
        print(f"Using {dims} with components {var1}, {var2} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(y_coord, iy)
            else:
                dy = fast_shift_2d(y_coord, iy, ix) - y_coord
            
            # Compute norm of separation vector
            # For mixed time-space cases, we need to be careful about calculating norm
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - use only spatial component for projection
                norm = np.maximum(np.abs(dx), 1e-10)
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - use only spatial component for projection
                norm = np.maximum(np.abs(dy), 1e-10)
            else:
                # Both spatial or both time (handled as spatial)
                norm = np.maximum(np.sqrt(dx**2 + dy**2), 1.0e-10)
            
            # Calculate velocity and scalar differences
            dcomp1 = fast_shift_2d(comp1_var, iy, ix) - comp1_var
            dcomp2 = fast_shift_2d(comp2_var, iy, ix) - comp2_var
            dscalar = fast_shift_2d(scalar_var_values, iy, ix) - scalar_var_values
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate transverse component (perpendicular to separation direction)
            # For mixed time-space cases, transverse calculations need special handling
            if time_dims[dims[0]] and not time_dims[dims[1]]:
                # (time, space) case - transverse is the time component
                delta_perp = dcomp2
            elif time_dims[dims[1]] and not time_dims[dims[0]]:
                # (space, time) case - transverse is the time component
                delta_perp = dcomp1
            else:
                # Both spatial or both time (handled as spatial)
                delta_perp = dcomp1 * (dy/norm) - dcomp2 * (dx/norm)
            
            # Calculate transverse-scalar structure function: delta_perp^n * dscalar^k
            sf_val = (delta_perp ** n) * (dscalar ** k)
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals    

def calc_advective_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate advective structure function: (du*deltaadv_u + dv*deltaadv_v)^n
    or (du*deltaadv_u + dw*deltaadv_w)^n or (dv*deltaadv_v + dw*deltaadv_w)^n
    depending on the plane.
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain four velocity components: u, v and adv_u, adv_v)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) != 4:
        raise ValueError(f"Advective structure function requires exactly 4 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # Extract regular and advective velocity components
    # Identify which are regular velocity components and which are advective
    vel_vars = []
    adv_vars = []
    
    for var in variables_names:
        if var.startswith('adv_') or 'adv' in var.lower():
            adv_vars.append(var)
        else:
            vel_vars.append(var)
    
    # Check if we have the right number of components
    if len(vel_vars) != 2 or len(adv_vars) != 2:
        # If automatic detection fails, try a simpler approach - assume first two are regular velocity
        vel_vars = variables_names[:2]
        adv_vars = variables_names[2:]

    
    # Define expected components based on plane
    if dims == ['y', 'x']:
        expected_components = ['u', 'v']
    elif dims == ['z', 'x']:
        expected_components = ['u', 'w']
    elif dims == ['z', 'y']:
        expected_components = ['v', 'w']
    else:
        # For mixed time-space dimensions, we're more flexible with component names
        expected_components = ['comp1', 'comp2']
    
    # Function to map variables to expected components for this plane
    def map_to_components(vars_list, expected):
        if len(vars_list) != len(expected):
            raise ValueError(f"Expected {len(expected)} components, got {len(vars_list)}")
            
        result = [None] * len(expected)
        
        # Try direct matching first
        for i, exp in enumerate(expected):
            for var in vars_list:
                if exp in var.lower():
                    result[i] = var
                    break
        
        # If any component is still None, use order-based matching
        if None in result:
            return vars_list
            
        return result
    
    # Map velocity and advective variables to expected components
    var1, var2 = map_to_components(vel_vars, expected_components)
    advvar1, advvar2 = map_to_components(adv_vars, expected_components)
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get the velocity components
    comp1_var = subset[var1].values
    comp2_var = subset[var2].values
    advcomp1_var = subset[advvar1].values
    advcomp2_var = subset[advvar2].values
    
    # Get coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_coord = subset.x.values
        y_coord = subset.y.values
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_coord = subset.x.values
        y_coord = subset.z.values  # Using y_coord to store z-coordinate for consistency
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_coord = subset.y.values  # Using x_coord to store y-coordinate for consistency
        y_coord = subset.z.values
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_coord = subset[dims[1]].values
        y_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(x_coord, ix)
            else:
                dx = fast_shift_2d(x_coord, iy, ix) - x_coord
                
            if time_dims[dims[0]]:  # y dimension is time
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
            
            # Calculate advective structure function: (du*deltaadv_u + dv*deltaadv_v)^n
            # This calculation doesn't depend on separation direction, so no special handling needed for time dimensions
            advective_term = dcomp1 * dadvcomp1 + dcomp2 * dadvcomp2
            sf_val = advective_term ** order
            results[idx] = bn.nanmean(sf_val)
            
            idx += 1
            
    return results, dx_vals, dy_vals
    
def calc_pressure_work_2d(subset, variables_names, order, dims, ny, nx, time_dims=None):
    """
    Calculate pressure work structure function: (∇_j(δΦ δu_j))^n
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing pressure and velocity components
    variables_names : list
        List of variable names (first is pressure, followed by velocity components)
    order : int
        Order of the structure function
    dims : list
        List of dimension names (should be ['y', 'x'], ['z', 'x'], or ['z', 'y'])
    ny, nx : int
        Array dimensions
    time_dims : dict, optional
        Dictionary indicating which dimensions are time dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values
    """
    if len(variables_names) < 3:  # Need pressure + 2 velocity components
        raise ValueError(f"Pressure work requires pressure and 2 velocity components, got {len(variables_names)}")
    
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # If both dimensions are time, this calculation is not well-defined
    if all(time_dims.values()):
        raise ValueError("Pressure work structure function is not well-defined when both dimensions are time dimensions")
    
    # Check valid planes for purely spatial dimensions
    if not any(time_dims.values()) and dims not in [['y', 'x'], ['z', 'x'], ['z', 'y']]:
        raise ValueError(f"For purely spatial dimensions, must be one of: ['y', 'x'], ['z', 'x'], or ['z', 'y'], got {dims}")
    
    # Extract variables
    pressure_var = subset[variables_names[0]].values
    comp1_var = subset[variables_names[1]].values
    comp2_var = subset[variables_names[2]].values
    
    # Arrays to store results
    results = np.full(ny * nx, np.nan)
    dx_vals = np.full(ny * nx, 0.0)
    dy_vals = np.full(ny * nx, 0.0)
    
    # Get appropriate coordinate variables based on the plane
    if dims == ['y', 'x']:
        # (y, x) plane
        x_name, y_name = 'x', 'y'
        # Components correspond to u and v
        u_var = comp1_var
        v_var = comp2_var
    elif dims == ['z', 'x']:
        # (z, x) plane
        x_name, y_name = 'x', 'z'
        # Components correspond to u and w
        u_var = comp1_var
        v_var = comp2_var
    elif dims == ['z', 'y']:
        # (z, y) plane
        x_name, y_name = 'y', 'z'
        # Components correspond to v and w
        u_var = comp1_var
        v_var = comp2_var
    else:
        # Mixed time-space dimensions - use the actual coordinate names
        x_name, y_name = dims[1], dims[0]
        u_var = comp1_var
        v_var = comp2_var
    
    # Get coordinate variables as 2D arrays
    x_coord = subset[x_name].values
    y_coord = subset[y_name].values
    
    # Convert 1D coordinates to 2D arrays if needed
    if len(x_coord.shape) == 1:
        X, Y = np.meshgrid(x_coord, y_coord)
    else:
        X, Y = x_coord, y_coord
    
    # Loop through all points (we still need to loop over shifts)
    idx = 0
    for iy in range(ny):
        for ix in range(nx):
            # Compute actual physical separation, handling time dimensions properly
            if time_dims[dims[1]]:  # x dimension is time
                dx = calculate_time_diff_1d(X, ix)
            else:
                dx = fast_shift_2d(X, iy, ix) - X
                
            if time_dims[dims[0]]:  # y dimension is time
                dy = calculate_time_diff_1d(Y, iy)
            else:
                dy = fast_shift_2d(Y, iy, ix) - Y
            
            # Store the separation distances
            dx_vals[idx] = bn.nanmean(dx)
            dy_vals[idx] = bn.nanmean(dy)
            
            # Calculate increments at each point
            dP = fast_shift_2d(pressure_var, iy, ix) - pressure_var
            du = fast_shift_2d(u_var, iy, ix) - u_var
            dv = fast_shift_2d(v_var, iy, ix) - v_var
            
            # Calculate the product of pressure and velocity increments at each point
            P_u_flux = dP * du
            P_v_flux = dP * dv
            
            # Calculate divergence using vectorized operations
            div_flux = np.zeros_like(pressure_var)
            
            # Create arrays for coordinate differences (central differences)
            # For x direction - handle time dimension appropriately
            if time_dims[dims[1]]:
                # For time dimension, we need to use time differences
                dx_central = np.ones_like(X)  # Assuming uniform time steps for gradient calculation
            else:
                # For spatial dimension, use central differences
                dx_central = np.zeros_like(X)
                dx_central[:, 1:-1] = (X[:, 2:] - X[:, :-2])
                # Use forward/backward differences at boundaries
                dx_central[:, 0] = (X[:, 1] - X[:, 0]) * 2
                dx_central[:, -1] = (X[:, -1] - X[:, -2]) * 2
            
            # For y direction - handle time dimension appropriately
            if time_dims[dims[0]]:
                # For time dimension, we need to use time differences
                dy_central = np.ones_like(Y)  # Assuming uniform time steps for gradient calculation
            else:
                # For spatial dimension, use central differences
                dy_central = np.zeros_like(Y)
                dy_central[1:-1, :] = (Y[2:, :] - Y[:-2, :])
                # Use forward/backward differences at boundaries
                dy_central[0, :] = (Y[1, :] - Y[0, :]) * 2
                dy_central[-1, :] = (Y[-1, :] - Y[-2, :]) * 2
            
            # Calculate flux derivatives using central differences
            dP_u_flux_dx = np.zeros_like(P_u_flux)
            if not time_dims[dims[1]]:  # Only calculate spatial derivative for spatial dimension
                dP_u_flux_dx[:, 1:-1] = (P_u_flux[:, 2:] - P_u_flux[:, :-2]) / dx_central[:, 1:-1]
                # Use forward/backward differences at boundaries
                dP_u_flux_dx[:, 0] = (P_u_flux[:, 1] - P_u_flux[:, 0]) / (dx_central[:, 0] / 2)
                dP_u_flux_dx[:, -1] = (P_u_flux[:, -1] - P_u_flux[:, -2]) / (dx_central[:, -1] / 2)
            
            dP_v_flux_dy = np.zeros_like(P_v_flux)
            if not time_dims[dims[0]]:  # Only calculate spatial derivative for spatial dimension
                dP_v_flux_dy[1:-1, :] = (P_v_flux[2:, :] - P_v_flux[:-2, :]) / dy_central[1:-1, :]
                # Use forward/backward differences at boundaries
                dP_v_flux_dy[0, :] = (P_v_flux[1, :] - P_v_flux[0, :]) / (dy_central[0, :] / 2)
                dP_v_flux_dy[-1, :] = (P_v_flux[-1, :] - P_v_flux[-2, :]) / (dy_central[-1, :] / 2)
            
            # Sum the derivatives to get the divergence
            div_flux = dP_u_flux_dx + dP_v_flux_dy
            
            # Raise to specified order
            sf_val = div_flux ** order
            
            # Compute structure function
            results[idx] = bn.nanmean(sf_val)
            idx += 1
            
    return results, dx_vals, dy_vals

###################################################################################################

################################Main SF Function###################################################

def calculate_structure_function_2d(ds, dims, variables_names, order, fun='longitudinal', 
                                  nbx=0, nby=0, spacing=None, num_bootstrappable=0, 
                                  bootstrappable_dims=None, boot_indexes=None, time_dims=None):
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
        results, dx_vals, dy_vals = calc_longitudinal_2d(subset, variables_names, order, 
                                                    dims, ny, nx, time_dims)
    elif fun == 'transverse':
        results, dx_vals, dy_vals = calc_transverse_2d(subset, variables_names, order, 
                                                  dims, ny, nx, time_dims)
    elif fun == 'default_vel':
        results, dx_vals, dy_vals = calc_default_vel_2d(subset, variables_names, order, 
                                                   dims, ny, nx, time_dims)
    elif fun == 'scalar':
        results, dx_vals, dy_vals = calc_scalar_2d(subset, variables_names, order, 
                                             dims, ny, nx, time_dims)
    elif fun == 'scalar_scalar':
        results, dx_vals, dy_vals = calc_scalar_scalar_2d(subset, variables_names, order, 
                                                    dims, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse':
        results, dx_vals, dy_vals = calc_longitudinal_transverse_2d(subset, variables_names, order, 
                                                              dims, ny, nx, time_dims)
    elif fun == 'longitudinal_scalar':
        results, dx_vals, dy_vals = calc_longitudinal_scalar_2d(subset, variables_names, order, 
                                                          dims, ny, nx, time_dims)
    elif fun == 'transverse_scalar':
        results, dx_vals, dy_vals = calc_transverse_scalar_2d(subset, variables_names, order, 
                                                        dims, ny, nx, time_dims)
    elif fun == 'advective':
        results, dx_vals, dy_vals = calc_advective_2d(subset, variables_names, order, 
                                                 dims, ny, nx, time_dims)
    elif fun == 'pressure_work':
        results, dx_vals, dy_vals = calc_pressure_work_2d(subset, variables_names, order, 
                                                     dims, ny, nx, time_dims)
    else:
        raise ValueError(f"Unsupported function type: {fun}")
    
    return results, dx_vals, dy_vals

###################################################################################################

#####################################Bootstrapping Monte Carlo#######################################################
def run_bootstrap_sf_2d(args):
    """Standalone bootstrap function for parallel processing in 2D."""
    ds, dims, variables_names, order, fun, nbx, nby, spacing, num_bootstrappable, bootstrappable_dims, boot_indexes, time_dims = args
    return calculate_structure_function_2d(
        ds=ds, dims=dims, variables_names=variables_names, order=order, fun=fun,
        nbx=nbx, nby=nby, spacing=spacing, num_bootstrappable=num_bootstrappable,
        bootstrappable_dims=bootstrappable_dims, boot_indexes=boot_indexes, time_dims=time_dims
    )

def monte_carlo_simulation_2d(ds, dims, variables_names, order, nbootstrap, bootsize, 
                            num_bootstrappable, all_spacings, boot_indexes, bootstrappable_dims,
                            fun='longitudinal', spacing=None, n_jobs=-1, backend='threading', time_dims=None):
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
        
    Returns
    -------
    list, list, list
        Lists of structure function values, DX values, DY values
    """
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, dx_vals, dy_vals = calculate_structure_function_2d(
            ds=ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            time_dims=time_dims  # Pass time_dims to calculate_structure_function_2d
        )
        return [results], [dx_vals], [dy_vals]
    
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
            results, dx_vals, dy_vals = calculate_structure_function_2d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals]
    else:
        # Two bootstrappable dimensions - check both
        valid_y_indices = dims[0] in indexes and indexes[dims[0]].shape[1] > 0
        valid_x_indices = dims[1] in indexes and indexes[dims[1]].shape[1] > 0
        
        if not valid_y_indices or not valid_x_indices:
            print("Warning: Not enough valid indices for bootstrapping with current spacing.")
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals = calculate_structure_function_2d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals]
    
    # Create all argument arrays for parallel processing
    all_args = []
    
    # Prepare parameters based on bootstrappable dimensions
    if num_bootstrappable == 1:
        # One bootstrappable dimension - only randomize that dimension
        bootstrap_dim = bootstrappable_dims[0]
        
        # Generate random indices for the bootstrappable dimension
        random_indices = np.random.choice(indexes[bootstrap_dim].shape[1], size=nbootstrap)
        
        # Create arguments for all bootstrap iterations
        for j in range(nbootstrap):
            if bootstrap_dim == dims[1]:  # x-dimension
                args = (
                    ds, dims, variables_names, order, fun,
                    random_indices[j], 0, sp_value, num_bootstrappable,
                    bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
                )
            else:  # y-dimension
                args = (
                    ds, dims, variables_names, order, fun,
                    0, random_indices[j], sp_value, num_bootstrappable,
                    bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
                )
            all_args.append(args)
            
    else:
        # Two bootstrappable dimensions - randomize both
        # Generate random indices for both dimensions
        nby = np.random.choice(indexes[dims[0]].shape[1], size=nbootstrap) 
        nbx = np.random.choice(indexes[dims[1]].shape[1], size=nbootstrap)
        
        # Create arguments for all bootstrap iterations
        for j in range(nbootstrap):
            args = (
                ds, dims, variables_names, order, fun,
                nbx[j], nby[j], sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
            )
            all_args.append(args)
    
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
    
    batch_size = max(10, nbootstrap//(n_workers*2))
    
    # Run simulations in parallel using module-level function
    results = Parallel(n_jobs=n_jobs, verbose=0, batch_size=batch_size, backend=backend)(
        delayed(run_bootstrap_sf_2d)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    dx_vals = [r[1] for r in results]
    dy_vals = [r[2] for r in results]
    
    return sf_results, dx_vals, dy_vals    
#####################################################################################################################

#################################Main 2D Binning Function############################################################
def _process_bootstrap_batch_2d(sf_results, dx_vals, dy_vals, bins_x, bins_y, 
                               bin_accumulators, target_bins, point_counts=None,
                               spacing_counts=None, sp_value=None, add_to_counts=True):
    """
    Process a batch of bootstrap results for 2D Cartesian binning.
    
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
    
    # Process all bootstrap samples
    for b in range(len(sf_results)):
        sf = sf_results[b]
        dx = dx_vals[b]
        dy = dy_vals[b]
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid):
            continue
            
        sf_valid = sf[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]
        
        # Volume element weights
        weights = np.abs(dx_valid * dy_valid)
        weights = np.maximum(weights, 1e-10)
        
        # Vectorized bin assignment
        x_indices = np.clip(np.digitize(dx_valid, bins_x) - 1, 0, n_bins_x - 1)
        y_indices = np.clip(np.digitize(dy_valid, bins_y) - 1, 0, n_bins_y - 1)
        
        # Create unique bin IDs
        bin_ids = y_indices * n_bins_x + x_indices
        
        # Process each point
        for idx in range(len(sf_valid)):
            bin_id = bin_ids[idx]
            if bin_id not in target_bin_ids:
                continue
                
            j, i = divmod(bin_id, n_bins_x)
            bin_key = (j, i)
            
            # Initialize accumulator if needed
            if bin_key not in bin_accumulators:
                bin_accumulators[bin_key] = {
                    'weighted_sum': 0.0,
                    'total_weight': 0.0,
                    'bootstrap_samples': []
                }
            
            # Accumulate weighted values
            weight = weights[idx]
            bin_accumulators[bin_key]['weighted_sum'] += sf_valid[idx] * weight
            bin_accumulators[bin_key]['total_weight'] += weight
            updated_bins.add(bin_key)
            
            # Update counts for density calculation
            if add_to_counts:
                if point_counts is not None:
                    point_counts[j, i] += 1
                if spacing_counts is not None and sp_value is not None:
                    spacing_counts[sp_value][j, i] += 1
        
        # After each bootstrap, store the contribution
        for bin_key in bin_accumulators:
            j, i = bin_key
            if (j, i) in target_bins:
                # Get current bootstrap contribution
                current_weighted_sum = bin_accumulators[bin_key]['weighted_sum']
                current_total_weight = bin_accumulators[bin_key]['total_weight']
                
                # Find what this bootstrap added
                if len(bin_accumulators[bin_key]['bootstrap_samples']) > 0:
                    prev_sum = sum(s['weighted_sum'] for s in bin_accumulators[bin_key]['bootstrap_samples'])
                    prev_weight = sum(s['total_weight'] for s in bin_accumulators[bin_key]['bootstrap_samples'])
                    
                    bootstrap_sum = current_weighted_sum - prev_sum
                    bootstrap_weight = current_total_weight - prev_weight
                else:
                    bootstrap_sum = current_weighted_sum
                    bootstrap_weight = current_total_weight
                
                if bootstrap_weight > 0:
                    bin_accumulators[bin_key]['bootstrap_samples'].append({
                        'weighted_sum': bootstrap_sum,
                        'total_weight': bootstrap_weight,
                        'mean': bootstrap_sum / bootstrap_weight
                    })
    
    return updated_bins


def _process_bootstrap_batch_polar(sf_results, dx_vals, dy_vals, r_bins, theta_bins,
                                 bin_accumulators, angular_accumulators, target_r_bins,
                                 point_counts=None, spacing_counts=None, sp_value=None,
                                 add_to_counts=True):
    """
    Process a batch of bootstrap results for polar binning.
    
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
        
    Returns
    -------
    updated_r_bins : set
        Set of r bins that were updated
    """
    n_bins_r = len(r_bins) - 1
    n_bins_theta = len(theta_bins) - 1
    updated_r_bins = set()
    
    # Process all bootstrap samples
    for b in range(len(sf_results)):
        sf = sf_results[b]
        dx = dx_vals[b]
        dy = dy_vals[b]
        
        # Create mask for valid values
        valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy)
        if not np.any(valid):
            continue
            
        sf_valid = sf[valid]
        dx_valid = dx[valid]
        dy_valid = dy[valid]
        
        # Convert to polar coordinates
        r_valid = np.sqrt(dx_valid**2 + dy_valid**2)
        theta_valid = np.arctan2(dy_valid, dx_valid)
        
        # Volume element weights (r for polar coordinates)
        weights = r_valid
        weights = np.maximum(weights, 1e-10)
        
        # Create bin indices
        r_indices = np.clip(np.digitize(r_valid, r_bins) - 1, 0, n_bins_r - 1)
        theta_indices = np.clip(np.digitize(theta_valid, theta_bins) - 1, 0, n_bins_theta - 1)
        
        # Process each point
        for idx in range(len(sf_valid)):
            r_idx = r_indices[idx]
            if r_idx not in target_r_bins:
                continue
            
            theta_idx = theta_indices[idx]
            weight = weights[idx]
            value = sf_valid[idx]
            
            # Initialize accumulators if needed
            if r_idx not in bin_accumulators:
                bin_accumulators[r_idx] = {
                    'weighted_sum': 0.0,
                    'total_weight': 0.0,
                    'bootstrap_samples': []
                }
            
            angular_key = (theta_idx, r_idx)
            if angular_key not in angular_accumulators:
                angular_accumulators[angular_key] = {
                    'weighted_sum': 0.0,
                    'total_weight': 0.0
                }
            
            # Accumulate weighted values
            bin_accumulators[r_idx]['weighted_sum'] += value * weight
            bin_accumulators[r_idx]['total_weight'] += weight
            updated_r_bins.add(r_idx)
            
            angular_accumulators[angular_key]['weighted_sum'] += value * weight
            angular_accumulators[angular_key]['total_weight'] += weight
            
            # Update counts
            if add_to_counts:
                if point_counts is not None:
                    point_counts[r_idx] += 1
                if spacing_counts is not None and sp_value is not None:
                    spacing_counts[sp_value][r_idx] += 1
        
        # After each bootstrap, store the contribution
        for r_idx in bin_accumulators:
            if r_idx in target_r_bins:
                # Get current bootstrap contribution
                current_weighted_sum = bin_accumulators[r_idx]['weighted_sum']
                current_total_weight = bin_accumulators[r_idx]['total_weight']
                
                # Find what this bootstrap added
                if len(bin_accumulators[r_idx]['bootstrap_samples']) > 0:
                    prev_sum = sum(s['weighted_sum'] for s in bin_accumulators[r_idx]['bootstrap_samples'])
                    prev_weight = sum(s['total_weight'] for s in bin_accumulators[r_idx]['bootstrap_samples'])
                    
                    bootstrap_sum = current_weighted_sum - prev_sum
                    bootstrap_weight = current_total_weight - prev_weight
                else:
                    bootstrap_sum = current_weighted_sum
                    bootstrap_weight = current_total_weight
                
                if bootstrap_weight > 0:
                    bin_accumulators[r_idx]['bootstrap_samples'].append({
                        'weighted_sum': bootstrap_sum,
                        'total_weight': bootstrap_weight,
                        'mean': bootstrap_sum / bootstrap_weight
                    })
    
    return updated_r_bins


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
            # Overall weighted mean
            sf_means[j, i] = acc['weighted_sum'] / acc['total_weight']
            
            # Bootstrap standard error
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                sf_stds[j, i] = np.std(boot_means, ddof=1)
            else:
                sf_stds[j, i] = np.nan
    
    return sf_means, sf_stds


def _calculate_bootstrap_statistics_polar(bin_accumulators, angular_accumulators, 
                                        n_bins_r, n_bins_theta):
    """
    Calculate statistics for polar binning.
    
    Returns
    -------
    sf_means : array
        Radial means
    sf_stds : array
        Radial standard errors
    sfr : array
        Angular-radial structure function
    sfr_counts : array
        Counts for angular-radial bins
    """
    sf_means = np.full(n_bins_r, np.nan)
    sf_stds = np.full(n_bins_r, np.nan)
    sfr = np.full((n_bins_theta, n_bins_r), np.nan)
    sfr_counts = np.zeros((n_bins_theta, n_bins_r), dtype=np.int_)
    
    # Radial statistics
    for r_idx, acc in bin_accumulators.items():
        if acc['total_weight'] > 0:
            sf_means[r_idx] = acc['weighted_sum'] / acc['total_weight']
            
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                sf_stds[r_idx] = np.std(boot_means, ddof=1)
            else:
                sf_stds[r_idx] = np.nan
    
    # Angular-radial matrix
    for (theta_idx, r_idx), acc in angular_accumulators.items():
        if acc['total_weight'] > 0:
            sfr[theta_idx, r_idx] = acc['weighted_sum'] / acc['total_weight']
            sfr_counts[theta_idx, r_idx] = int(acc['total_weight'])
    
    return sf_means, sf_stds, sfr, sfr_counts


def _update_spacing_effectiveness(bin_spacing_effectiveness, bin_spacing_counts,
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


def _get_spacing_distribution(bin_list, spacing_effectiveness, total_bootstraps,
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
            sp_bootstraps = remaining // len([s for s, e in sorted_spacings if e >= 0])
        
        if sp_bootstraps > 0:
            distribution.append((sp_value, sp_bootstraps))
            remaining -= sp_bootstraps
    
    return distribution


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
    
    # Determine log vs linear bins
    def is_log_spaced(bin_edges):
        if len(bin_edges) < 2:
            return False
        ratios = bin_edges[1:] / bin_edges[:-1]
        ratio_std = np.std(ratios)
        ratio_mean = np.mean(ratios)
        if ratio_std / ratio_mean < 0.01:
            return abs(ratio_mean - 1.0) > 0.01
        return False
    
    log_bins_x = is_log_spaced(bins_x)
    log_bins_y = is_log_spaced(bins_y)
    
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


def _initialize_polar_bins(r_bins, n_theta):
    """
    Initialize polar bin configuration.
    
    Returns
    -------
    config : dict
        Dictionary with polar bin configuration
    """
    # Determine if radial bins are log-spaced
    ratios = r_bins[1:] / r_bins[:-1]
    ratio_std = np.std(ratios)
    ratio_mean = np.mean(ratios)
    
    if ratio_std / ratio_mean < 0.01:
        if np.abs(ratio_mean - 1.0) < 0.01:
            log_bins = False
            r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        else:
            log_bins = True
            r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
    else:
        log_bins = False
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


def _calculate_bin_density_polar(point_counts, r_bins):
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


def _evaluate_convergence(sf_stds, point_counts, bin_bootstraps,
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


def _group_bins_for_iteration(unconverged_indices, bin_density, bootstrap_steps):
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



def _calculate_isotropy_error(sfr, sf_means, window_size_theta):
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


def _calculate_homogeneity_error(sfr, window_size_r):
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


def _calculate_confidence_intervals(means, stds, counts, confidence_level=0.95):
    """Calculate confidence intervals."""
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    ci_upper = np.full_like(means, np.nan)
    ci_lower = np.full_like(means, np.nan)
    
    valid_bins = ~np.isnan(means)
    if np.any(valid_bins):
        # Multiple points
        multiple_points = valid_bins & (counts > 1)
        if np.any(multiple_points):
            std_error = stds[multiple_points] / np.sqrt(counts[multiple_points])
            ci_upper[multiple_points] = means[multiple_points] + z_score * std_error
            ci_lower[multiple_points] = means[multiple_points] - z_score * std_error
        
        # Single point
        single_point = valid_bins & (counts == 1)
        if np.any(single_point):
            ci_upper[single_point] = means[single_point]
            ci_lower[single_point] = means[single_point]
    
    return ci_upper, ci_lower



def _process_no_bootstrap_2d(ds, dims, variables_names, order, fun, bins, time_dims):
    """Handle the special case of no bootstrappable dimensions for 2D."""
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function once
    results, dx_vals, dy_vals = calculate_structure_function_2d(
        ds=ds,
        dims=dims,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        time_dims=time_dims
    )
    
    # Initialize bins
    bins_config = _initialize_2d_bins(bins[dims[1]], bins[dims[0]], dims)
    
    # Bin the results
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    
    # Create 2D binning grid
    x_bins_idx = np.clip(np.digitize(valid_dx, bins_config['bins_x']) - 1, 
                        0, bins_config['n_bins_x'] - 1)
    y_bins_idx = np.clip(np.digitize(valid_dy, bins_config['bins_y']) - 1,
                        0, bins_config['n_bins_y'] - 1)
    
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


def _process_no_bootstrap_polar(ds, dims, variables_names, order, fun, r_bins, n_theta, time_dims):
    """Handle the special case of no bootstrappable dimensions for polar."""
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function
    results, dx_vals, dy_vals = calculate_structure_function_2d(
        ds=ds,
        dims=dims,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        time_dims=time_dims
    )
    
    # Initialize bins
    bins_config = _initialize_polar_bins(r_bins, n_theta)
    
    # Filter and convert to polar
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    
    r_valid = np.sqrt(valid_dx**2 + valid_dy**2)
    theta_valid = np.arctan2(valid_dy, valid_dx)
    
    # Volume element weights
    weights = r_valid
    weights = np.maximum(weights, 1e-10)
    
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
        bin_weights = weights[r_bin_mask]
        bin_theta_indices = theta_indices[r_bin_mask]
        
        point_counts[r_idx] = len(bin_sf)
        
        if len(bin_sf) > 0:
            normalized_weights = bin_weights / np.sum(bin_weights) * len(bin_weights)
            sf_means[r_idx] = np.average(bin_sf, weights=normalized_weights)
            
            if len(bin_sf) > 1:
                weighted_var = np.average((bin_sf - sf_means[r_idx])**2, weights=normalized_weights)
                sf_stds[r_idx] = np.sqrt(weighted_var)
        
        # Process angular bins
        for theta_idx in range(bins_config['n_bins_theta']):
            theta_bin_mask = bin_theta_indices == theta_idx
            if not np.any(theta_bin_mask):
                continue
            
            theta_sf = bin_sf[theta_bin_mask]
            theta_weights = bin_weights[theta_bin_mask]
            
            if len(theta_sf) > 0:
                normalized_theta_weights = theta_weights / np.sum(theta_weights) * len(theta_weights)
                sfr[theta_idx, r_idx] = np.average(theta_sf, weights=normalized_theta_weights)
                sfr_counts[theta_idx, r_idx] = len(theta_sf)
    
    return sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config

def _run_adaptive_bootstrap_loop(valid_ds, dims, variables_names, order, fun,
                               bins_config, initial_nbootstrap, max_nbootstrap,
                               step_nbootstrap, convergence_eps, spacing_values,
                               bootsize_dict, num_bootstrappable, all_spacings,
                               boot_indexes, bootstrappable_dims, n_jobs, backend,
                               time_dims, is_2d=True):
    """
    Generic adaptive bootstrap loop used by both 2D and isotropic functions.
    
    This function now handles both 2D and polar cases internally.
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
        point_counts = np.zeros(result_shape, dtype=np.int_)
        bin_density = np.zeros(result_shape, dtype=np.float32)
        bin_status = np.zeros(result_shape, dtype=bool)
        bin_bootstraps = np.ones(result_shape, dtype=np.int_) * initial_nbootstrap
        bootstrap_steps = np.ones(result_shape, dtype=np.int_) * step_nbootstrap
    else:
        sf_means = np.full(result_shape[0], np.nan)
        sf_stds = np.full(result_shape[0], np.nan)
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
    
    for sp_value in spacing_values:
        print(f"Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals = monte_carlo_simulation_2d(
            ds=valid_ds, dims=dims, variables_names=variables_names,
            order=order, nbootstrap=init_samples_per_spacing,
            bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings, boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims, fun=fun,
            spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims
        )
        
        # Process batch based on type
        if is_2d:
            _process_bootstrap_batch_2d(
                sf_results, dx_vals, dy_vals,
                bins_config['bins_x'], bins_config['bins_y'],
                bin_accumulators, set(all_bins), point_counts,
                bin_spacing_counts, sp_value, True
            )
        else:
            _process_bootstrap_batch_polar(
                sf_results, dx_vals, dy_vals,
                bins_config['r_bins'], bins_config['theta_bins'],
                bin_accumulators, angular_accumulators, set(all_bins),
                point_counts, bin_spacing_counts, sp_value, True
            )
        
        # Update effectiveness
        _update_spacing_effectiveness(
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
        sf_means[:], sf_stds[:], sfr[:], sfr_counts[:] = _calculate_bootstrap_statistics_polar(
            bin_accumulators, angular_accumulators,
            bins_config['n_bins_r'], bins_config['n_bins_theta']
        )
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    if is_2d:
        bin_density = _calculate_bin_density_2d(point_counts, bins_config['bins_x'], 
                                              bins_config['bins_y'])
    else:
        bin_density = _calculate_bin_density_polar(point_counts, bins_config['r_bins'])
    
    print(f"Total points collected: {np.sum(point_counts)}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins_total}")
    
    # Initial convergence check
    bin_status, convergence_reasons = _evaluate_convergence(
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
        # Group bins
#        if is_2d:
#            unconverged_indices = np.where(unconverged)
#        else:
#            unconverged_indices = (np.where(unconverged),)
            
        groups = _group_bins_for_iteration(unconverged_indices, bin_density, bootstrap_steps)
        print(f"Grouped unconverged bins into {len(groups)} groups")
        
        # Process each group
        for (step, density_q), bin_list in sorted(groups.items(),
                                                 key=lambda x: (x[0][1], x[0][0]),
                                                 reverse=True):
            print(f"\nProcessing {len(bin_list)} bins with step size {step} in density quartile {density_q}")
            
            # Get spacing distribution
            distribution = _get_spacing_distribution(
                bin_list, bin_spacing_effectiveness, step, spacing_values
            )
            
            # Process each spacing
            for sp_value, sp_bootstraps in distribution:
                if sp_bootstraps <= 0:
                    continue
                    
                # Run Monte Carlo
                sf_results, dx_vals, dy_vals = monte_carlo_simulation_2d(
                    ds=valid_ds, dims=dims, variables_names=variables_names,
                    order=order, nbootstrap=sp_bootstraps,
                    bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
                    all_spacings=all_spacings, boot_indexes=boot_indexes,
                    bootstrappable_dims=bootstrappable_dims, fun=fun,
                    spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims
                )
                
                # Process batch based on type (no count updates)
                if is_2d:
                    _process_bootstrap_batch_2d(
                        sf_results, dx_vals, dy_vals,
                        bins_config['bins_x'], bins_config['bins_y'],
                        bin_accumulators, set(bin_list), None,
                        bin_spacing_counts, sp_value, False
                    )
                else:
                    _process_bootstrap_batch_polar(
                        sf_results, dx_vals, dy_vals,
                        bins_config['r_bins'], bins_config['theta_bins'],
                        bin_accumulators, angular_accumulators, set(bin_list),
                        None, bin_spacing_counts, sp_value, False
                    )
                
                del sf_results, dx_vals, dy_vals
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
                            sf_means[j, i] = acc['weighted_sum'] / acc['total_weight']
                            if len(acc['bootstrap_samples']) > 1:
                                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                                sf_stds[j, i] = np.std(boot_means, ddof=1)
                        
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
                            sf_means[r_idx] = acc['weighted_sum'] / acc['total_weight']
                            if len(acc['bootstrap_samples']) > 1:
                                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                                sf_stds[r_idx] = np.std(boot_means, ddof=1)
                        
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



def _create_2d_dataset(results, bins_config, dims, order, fun, 
                      bootstrappable_dims, time_dims, convergence_eps,
                      max_nbootstrap, initial_nbootstrap, backend):
    """Create output dataset for 2D binning."""
    ds_binned = xr.Dataset(
        data_vars={
            'sf': ((dims[0], dims[1]), results['sf_means']),
            'sf_std': ((dims[0], dims[1]), results['sf_stds']),
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
            'weighting': 'volume_element',
            'bootstrap_se_method': 'unweighted_std'
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


def _create_isotropic_dataset(results, bins_config, order, fun, window_size_theta,
                            window_size_r, convergence_eps, max_nbootstrap,
                            initial_nbootstrap, bootstrappable_dims, backend,
                            variables_names):
    """Create output dataset for isotropic binning."""
    # Calculate error metrics
    eiso = _calculate_isotropy_error(results['sfr'], results['sf_means'], window_size_theta)
    ehom, r_subset_indices = _calculate_homogeneity_error(results['sfr'], window_size_r)
    
    # Calculate confidence intervals
    ci_upper, ci_lower = _calculate_confidence_intervals(
        results['sf_means'], results['sf_stds'], results['point_counts']
    )
    
    # Prepare data variables
    data_vars = {
        'sf_polar': (('theta', 'r'), results['sfr']),
        'sf': (('r'), results['sf_means']),
        'error_isotropy': (('r'), eiso),
        'std': (('r'), results['sf_stds']),
        'ci_upper': (('r'), ci_upper),
        'ci_lower': (('r'), ci_lower),
        'error_homogeneity': (('r_subset'), ehom),
        'n_bootstrap': (('r'), results['bin_bootstraps']),
        'bin_density': (('r'), results['bin_density']),
        'point_counts': (('r'), results['point_counts']),
        'converged': (('r'), results['bin_status'])
    }
    
    coords = {
        'r': bins_config['r_centers'],
        'r_subset': bins_config['r_centers'][r_subset_indices],
        'theta': bins_config['theta_centers']
    }
    
    
    ds_iso = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            'order': str(order),
            'function_type': fun,
            'window_size_theta': window_size_theta,
            'window_size_r': window_size_r,
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'bin_type': 'logarithmic' if bins_config['log_bins'] else 'linear',
            'variables': variables_names,
            'bootstrappable_dimensions': ','.join(bootstrappable_dims),
            'backend': backend,
        }
    )
    
    # Add bin edges
    ds_iso['r_bins'] = (('r_edge'), bins_config['r_bins'])
    ds_iso['theta_bins'] = (('theta_edge'), bins_config['theta_bins'])
    
    return ds_iso

def bin_sf_2d(ds, variables_names, order, bins, bootsize=None, fun='longitudinal', 
            initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
            convergence_eps=0.1, n_jobs=-1, backend='threading'):
    """
    Bin structure function with proper volume element weighting.
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
    print("="*60 + "\n")
    
    # Validate bins
    if not isinstance(bins, dict) or not all(dim in bins for dim in dims):
        raise ValueError("'bins' must be a dictionary with all dimensions as keys")
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_2d(
            valid_ds, dims, variables_names, order, fun, bins, time_dims
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
        
        return _create_2d_dataset(results, bins_config, dims, order, fun,
                                bootstrappable_dims, time_dims, convergence_eps,
                                max_nbootstrap, initial_nbootstrap, backend)
    
    # Initialize bins
    bins_config = _initialize_2d_bins(bins[dims[1]], bins[dims[0]], dims)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, is_2d=True
    )
    
    results['variables_names'] = variables_names
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_binned = _create_2d_dataset(results, bins_config, dims, order, fun,
                                 bootstrappable_dims, time_dims, convergence_eps,
                                 max_nbootstrap, initial_nbootstrap, backend)
    
    print("2D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned


def get_isotropic_sf_2d(ds, variables_names, order=2.0, bins=None, bootsize=None,
                      initial_nbootstrap=100, max_nbootstrap=1000, 
                      step_nbootstrap=100, fun='longitudinal', 
                      n_bins_theta=36, window_size_theta=None, window_size_r=None,
                      convergence_eps=0.1, n_jobs=-1, backend='threading'):
    """
    Get isotropic (radially binned) structure function with volume element weighting.
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
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_polar(
            valid_ds, dims, variables_names, order, fun, bins['r'], n_bins_theta, time_dims
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
                                       backend, variables_names)
    
    # Initialize bins
    bins_config = _initialize_polar_bins(bins['r'], n_bins_theta)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, is_2d=False
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_iso = _create_isotropic_dataset(
        results, bins_config, order, fun,
        window_size_theta, window_size_r,
        convergence_eps, max_nbootstrap,
        initial_nbootstrap, bootstrappable_dims,
        backend, variables_names
    )
    
    print("ISOTROPIC SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_iso


#####################################################################################################################
