"""Three-dimensional structure function calculations."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import bottleneck as bn
import gc
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view


from .core import (validate_dataset_3d, setup_bootsize_3d, calculate_adaptive_spacings_3d,
                  compute_boot_indexes_3d, get_boot_indexes_3d)
from .utils import (fast_shift_3d, check_and_reorder_variables_3d, map_variables_by_pattern_3d)

##################################Structure Functions Types########################################

def calc_longitudinal_3d(subset, variables_names, order, dims, nz, ny, nx):
    """
    Calculate 3D longitudinal structure function: (du*dx + dv*dy + dw*dz)^n / |r|^n
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain three velocity components)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"3D longitudinal structure function requires exactly 3 velocity components, got {len(variables_names)}")
    
    # Check and reorder variables if needed
    u, v, w = check_and_reorder_variables_3d(variables_names, dims)
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector
                norm = np.maximum(np.sqrt(dx**2 + dy**2 + dz**2), 1e-10)  # to avoid dividing by zero
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Project velocity difference onto separation direction (longitudinal)
                delta_parallel = du * (dx/norm) + dv * (dy/norm) + dw * (dz/norm)
                
                # Compute structure function
                sf_val = (delta_parallel) ** order
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ij(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse_ij structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # Check and reorder variables if needed - ensure we get u and v
    u, v = check_and_reorder_variables_3d(variables_names, dims, fun='transverse_ij')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xy-plane
                norm_xy = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xy-plane)
                delta_perp_ij = du * (dy/norm_xy) - dv * (dx/norm_xy)
                
                # Compute structure function
                sf_val = (delta_perp_ij) ** order
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ik(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse_ik structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # Check and reorder variables if needed - ensure we get u and w
    u, w = check_and_reorder_variables_3d(variables_names, dims, fun='transverse_ik')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xz-plane
                norm_xz = np.maximum(np.sqrt(dx**2 + dz**2), 1e-10)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xz-plane)
                delta_perp_ik = dw * (dx/norm_xz) - du * (dz/norm_xz) 
                
                # Compute structure function
                sf_val = (delta_perp_ik) ** order
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_jk(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Transverse_jk structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    # Check and reorder variables if needed - ensure we get v and w
    v, w = check_and_reorder_variables_3d(variables_names, dims, fun='transverse_jk')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in yz-plane
                norm_yz = np.maximum(np.sqrt(dy**2 + dz**2), 1e-10)
                
                # Calculate velocity differences
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in yz-plane)
                delta_perp_jk = dv * (dz/norm_yz) - dw * (dy/norm_yz)
                
                # Compute structure function
                sf_val = (delta_perp_jk) ** order
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_scalar_3d(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 1:
        raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(variables_names)}")
    
    # Get the scalar variable name
    scalar_name = variables_names[0]
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the scalar variable
    scalar_var = subset[scalar_name].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D scalar structure function for {scalar_name}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Calculate scalar difference
                dscalar = fast_shift_3d(scalar_var, iz, iy, ix) - scalar_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate scalar structure function: dscalar^n
                sf_val = dscalar ** order
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_scalar_3d(subset, variables_names, order, dims, nz, ny, nx):
    """
    Calculate 3D longitudinal-scalar structure function: (du_longitudinal^n * dscalar^k)
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain three velocity components and one scalar)
    order : tuple
        Tuple of orders (n, k) for the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 4:
        raise ValueError(f"3D longitudinal-scalar function requires 4 variables (3 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-scalar structure function, got {order}")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables
    velocity_vars = variables_names[:3]
    scalar_var = variables_names[3]
    u, v, w = check_and_reorder_variables_3d(velocity_vars, dims, fun='longitudinal')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    v_var = subset[v].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D longitudinal-scalar with components {u}, {v}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector
                norm = np.maximum(np.sqrt(dx**2 + dy**2 + dz**2), 1e-10)
                
                # Calculate velocity and scalar differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Project velocity difference onto separation direction (longitudinal)
                delta_parallel = du * (dx/norm) + dv * (dy/norm) + dw * (dz/norm)
                
                # Calculate longitudinal-scalar structure function: delta_parallel^n * dscalar^k
                sf_val = (delta_parallel ** n) * (dscalar ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ij_scalar(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse_ij_scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
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
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    v_var = subset[v].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D transverse_ij_scalar with components {u}, {v} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xy-plane
                norm_xy = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
                
                # Calculate velocity and scalar differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xy-plane)
                delta_perp_ij = du * (dy/norm_xy) - dv * (dx/norm_xy)
                
                # Calculate transverse-scalar structure function: delta_perp_ij^n * dscalar^k
                sf_val = (delta_perp_ij ** n) * (dscalar ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ik_scalar(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse_ik_scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
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
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D transverse_ik_scalar with components {u}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xz-plane
                norm_xz = np.maximum(np.sqrt(dx**2 + dz**2), 1e-10)
                
                # Calculate velocity and scalar differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in xz-plane)
                delta_perp_ik = du * (dz/norm_xz) - dw * (dx/norm_xz)
                
                # Calculate transverse-scalar structure function: delta_perp_ik^n * dscalar^k
                sf_val = (delta_perp_ik ** n) * (dscalar ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_jk_scalar(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 3:
        raise ValueError(f"Transverse_jk_scalar structure function requires 3 variables (2 velocity components and 1 scalar), got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for transverse-scalar structure function, got {order}")
    
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
    
    # Get the velocity components and scalar
    v_var = subset[v].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D transverse_jk_scalar with components {v}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in yz-plane
                norm_yz = np.maximum(np.sqrt(dy**2 + dz**2), 1e-10)
                
                # Calculate velocity and scalar differences
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                dscalar = fast_shift_3d(scalar_var_values, iz, iy, ix) - scalar_var_values
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate transverse component (perpendicular to separation in yz-plane)
                delta_perp_jk = dv * (dz/norm_yz) - dw * (dy/norm_yz)
                
                # Calculate transverse-scalar structure function: delta_perp_jk^n * dscalar^k
                sf_val = (delta_perp_jk ** n) * (dscalar ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_ij(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse_ij structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get u and v
    u, v = check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal_transverse_ij')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D longitudinal-transverse_ij with components {u}, {v}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xy-plane
                norm_xy = np.maximum(np.sqrt(dx**2 + dy**2), 1e-10)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Project velocity difference onto separation direction in xy-plane (longitudinal)
                delta_parallel = (du * dx + dv * dy) / norm_xy
                
                # Calculate transverse component (perpendicular to separation in xy-plane)
                delta_perp = (du * dy - dv * dx) / norm_xy
                
                # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
                sf_val = (delta_parallel ** n) * (delta_perp ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_ik(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse_ik structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get u and w
    u, w = check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal_transverse_ik')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D longitudinal-transverse_ik with components {u}, {w}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in xz-plane
                norm_xz = np.maximum(np.sqrt(dx**2 + dz**2), 1e-10)
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Project velocity difference onto separation direction in xz-plane (longitudinal)
                delta_parallel = (du * dx + dw * dz) / norm_xz
                
                # Calculate transverse component (perpendicular to separation in xz-plane)
                delta_perp = (du * dz - dw * dx) / norm_xz
                
                # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
                sf_val = (delta_parallel ** n) * (delta_perp ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_jk(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Longitudinal-transverse_jk structure function requires exactly 2 velocity components, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for longitudinal-transverse structure function, got {order}")
    
    # Unpack order tuple
    n, k = order
    
    # Check and reorder variables if needed - ensure we get v and w
    v, w = check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal_transverse_jk')
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D longitudinal-transverse_jk with components {v}, {w}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Compute norm of separation vector in yz-plane
                norm_yz = np.maximum(np.sqrt(dy**2 + dz**2), 1e-10)
                
                # Calculate velocity differences
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Project velocity difference onto separation direction in yz-plane (longitudinal)
                delta_parallel = (dv * dy + dw * dz) / norm_yz
                
                # Calculate transverse component (perpendicular to separation in yz-plane)
                delta_perp = (dv * dz - dw * dy) / norm_yz
                
                # Calculate longitudinal-transverse structure function: delta_parallel^n * delta_perp^k
                sf_val = (delta_parallel ** n) * (delta_perp ** k)
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_scalar_scalar_3d(subset, variables_names, order, dims, nz, ny, nx):
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
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 2:
        raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar variables, got {len(variables_names)}")
    
    if not isinstance(order, tuple) or len(order) != 2:
        raise ValueError(f"Order must be a tuple (n, k) for scalar-scalar structure function, got {order}")
    
    # Unpack order tuple
    n, k = order
    
    # Get the scalar variable names
    scalar1_name, scalar2_name = variables_names
    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the scalar variables
    scalar1_var = subset[scalar1_name].values
    scalar2_var = subset[scalar2_name].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    print(f"Using 3D scalar-scalar structure function for {scalar1_name} and {scalar2_name}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_advective_3d(subset, variables_names, order, dims, nz, ny, nx):
    """
    Calculate 3D advective structure function: (du*deltaadv_u + dv*deltaadv_v + dw*deltaadv_w)^n
    
    Parameters
    ----------
    subset : xarray.Dataset
        Subset of the dataset containing required variables
    variables_names : list
        List of variable names (should contain six velocity components: u, v, w and adv_u, adv_v, adv_w)
    order : int
        Order of the structure function
    dims : list
        List of dimension names
    nz, ny, nx : int
        Array dimensions
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    if len(variables_names) != 6:
        raise ValueError(f"3D advective structure function requires exactly 6 velocity components, got {len(variables_names)}")
    
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
    if len(vel_vars) != 3 or len(adv_vars) != 3:
        # If automatic detection fails, try a simpler approach - assume first three are regular velocity
        vel_vars = variables_names[:3]
        adv_vars = variables_names[3:]

    
    # Define expected components for 3D
    expected_components = ['u', 'v', 'w']
    
    # Function to map variables to expected components
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
    u, v, w = map_to_components(vel_vars, expected_components)
    advu, advv, advw = map_to_components(adv_vars, expected_components)
    


    
    # Arrays to store results
    results = np.full(nz * ny * nx, np.nan)
    dx_vals = np.full(nz * ny * nx, 0.0)
    dy_vals = np.full(nz * ny * nx, 0.0)
    dz_vals = np.full(nz * ny * nx, 0.0)
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    w_var = subset[w].values
    advu_var = subset[advu].values
    advv_var = subset[advv].values
    advw_var = subset[advw].values
    
    # Get coordinate variables
    x_coord = subset.x.values
    y_coord = subset.y.values
    z_coord = subset.z.values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # Compute actual physical separation
                dx = fast_shift_3d(x_coord, iz, iy, ix) - x_coord
                dy = fast_shift_3d(y_coord, iz, iy, ix) - y_coord
                dz = fast_shift_3d(z_coord, iz, iy, ix) - z_coord
                
                # Calculate velocity differences
                du = fast_shift_3d(u_var, iz, iy, ix) - u_var
                dv = fast_shift_3d(v_var, iz, iy, ix) - v_var
                dw = fast_shift_3d(w_var, iz, iy, ix) - w_var
                
                # Calculate advective velocity differences
                dadvu = fast_shift_3d(advu_var, iz, iy, ix) - advu_var
                dadvv = fast_shift_3d(advv_var, iz, iy, ix) - advv_var
                dadvw = fast_shift_3d(advw_var, iz, iy, ix) - advw_var
                
                # Store the separation distances
                dx_vals[idx] = bn.nanmean(dx)
                dy_vals[idx] = bn.nanmean(dy)
                dz_vals[idx] = bn.nanmean(dz)
                
                # Calculate advective structure function: (du*deltaadv_u + dv*deltaadv_v + dw*deltaadv_w)^n
                advective_term = du * dadvu + dv * dadvv + dw * dadvw
                sf_val = advective_term ** order
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals
    
##############################################################################################################

################################Main SF Function##############################################################

def calculate_structure_function_3d(ds, dims, variables_names, order, fun='longitudinal', 
                                  nbz=0, nby=0, nbx=0, spacing=None, num_bootstrappable=0,
                                  bootstrappable_dims=None, boot_indexes=None):
    """
    Main function to calculate 3D structure functions based on specified type.
    
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
    spacing : int or dict, optional
        Spacing value to use
    num_bootstrappable : int, optional
        Number of bootstrappable dimensions
    bootstrappable_dims : list, optional
        List of bootstrappable dimensions
    boot_indexes : dict, optional
        Dictionary with spacing values as keys and boot indexes as values
        
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
        Structure function values, DX values, DY values, DZ values
    """
    # Start with the full dataset
    subset = ds
    
    # Only subset bootstrappable dimensions
    if num_bootstrappable > 0 and bootstrappable_dims:
        # Get boot indexes for bootstrappable dimensions
        if boot_indexes is not None and spacing is not None:
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
            if bootstrap_dim == dims[0]:  # z-dimension
                nb_index = nbz
            elif bootstrap_dim == dims[1]:  # y-dimension
                nb_index = nby
            else:  # x-dimension
                nb_index = nbx
                
            # Add only the bootstrappable dimension to subset dict
            if indexes and bootstrap_dim in indexes and indexes[bootstrap_dim].shape[1] > nb_index:
                subset_dict[bootstrap_dim] = indexes[bootstrap_dim][:, nb_index]
        
        elif num_bootstrappable == 2:
            # Exactly two dimensions are bootstrappable, one is not
            # Add the two bootstrappable dimensions to the subset dict
            for dim in bootstrappable_dims:
                if dim == dims[0]:  # z-dimension
                    nb_index = nbz
                elif dim == dims[1]:  # y-dimension
                    nb_index = nby
                else:  # x-dimension
                    nb_index = nbx
                
                if indexes and dim in indexes and indexes[dim].shape[1] > nb_index:
                    subset_dict[dim] = indexes[dim][:, nb_index]
        
        else:  # num_bootstrappable == 3
            # All three dimensions are bootstrappable
            # Add all dimensions to the subset dict
            nb_indexes = {dims[0]: nbz, dims[1]: nby, dims[2]: nbx}
            for dim in bootstrappable_dims:
                if indexes and dim in indexes and indexes[dim].shape[1] > nb_indexes[dim]:
                    subset_dict[dim] = indexes[dim][:, nb_indexes[dim]]
        
        # Apply subsetting if needed
        if subset_dict:
            subset = ds.isel(subset_dict)
    
    # Check if the required variables exist in the dataset
    for var_name in variables_names:
        if var_name not in subset:
            raise ValueError(f"Variable {var_name} not found in dataset")
    
    # Get dimensions of the first variable to determine array sizes
    nz, ny, nx = subset[variables_names[0]].shape
    
    # Calculate structure function based on specified type
    if fun == 'longitudinal':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_3d(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'transverse_ij':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'transverse_ik':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'transverse_jk':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'scalar':
        results, dx_vals, dy_vals, dz_vals = calc_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'scalar_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_scalar_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'longitudinal_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'longitudinal_transverse_ij':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'longitudinal_transverse_ik':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'longitudinal_transverse_jk':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'transverse_ij_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ij_scalar(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'transverse_ik_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ik_scalar(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'transverse_jk_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_jk_scalar(
            subset, variables_names, order, dims, nz, ny, nx)
    elif fun == 'advective':
        results, dx_vals, dy_vals, dz_vals = calc_advective_3d(
            subset, variables_names, order, dims, nz, ny, nx)
    else:
        raise ValueError(f"Unsupported function type: {fun}")
            
    return results, dx_vals, dy_vals, dz_vals
##############################################################################################################

###############################Bootstrap Monte Carlo##########################################################

def monte_carlo_simulation_3d(ds, dims, variables_names, order, nbootstrap, bootsize, 
                            num_bootstrappable, all_spacings, boot_indexes, bootstrappable_dims,
                            fun='longitudinal', spacing=None, n_jobs=-1):
    """
    Run Monte Carlo simulation for structure function calculation with multiple bootstrap samples.
    
    Parameters
    -----------
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
    spacing : dict or int, optional
        Spacing value to use
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is -1 (all cores).
    
    Returns
    --------
    list
        Raw structure function values for all bootstrap samples
    list
        DX values for all bootstrap samples
    list
        DY values for all bootstrap samples
    list
        DZ values for all bootstrap samples
    """
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
            ds=ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable
        )
        return [results], [dx_vals], [dy_vals], [dz_vals]
    
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
    
    # Set the seed for reproducibility
    np.random.seed(10000000)
    
    # Get boot indexes for the specified spacing
    if sp_value in boot_indexes:
        indexes = boot_indexes[sp_value]
    else:
        # Calculate boot indexes on-the-fly
        data_shape = dict(ds.sizes)
        indexes = get_boot_indexes_3d(dims, data_shape, bootsize, all_spacings, boot_indexes, 
                                    bootstrappable_dims, num_bootstrappable, sp_value)
    
    # Prepare the random sampling parameters based on bootstrappable dimensions
    if num_bootstrappable == 1:
        # Only one dimension is bootstrappable
        bootstrap_dim = bootstrappable_dims[0]
        
        if not indexes or bootstrap_dim not in indexes or indexes[bootstrap_dim].shape[1] == 0:
            print(f"Warning: No valid indices for dimension {bootstrap_dim} with spacing {sp_value}.")
            # Fall back to calculating once with full dataset
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for the bootstrappable dimension
        random_indices = np.random.choice(indexes[bootstrap_dim].shape[1], size=nbootstrap)
        
        # Prepare a function to run in parallel
        def simulate_bootstrap(j):
            if bootstrap_dim == dims[0]:  # z-dimension
                return calculate_structure_function_3d(
                    ds=ds,
                    dims=dims,
                    variables_names=variables_names,
                    order=order, 
                    fun=fun, 
                    nbz=random_indices[j],
                    nby=0,  # Fixed value for non-bootstrappable dimension
                    nbx=0,  # Fixed value for non-bootstrappable dimension
                    spacing=sp_value,
                    num_bootstrappable=num_bootstrappable,
                    bootstrappable_dims=bootstrappable_dims,
                    boot_indexes=boot_indexes
                )
            elif bootstrap_dim == dims[1]:  # y-dimension
                return calculate_structure_function_3d(
                    ds=ds,
                    dims=dims,
                    variables_names=variables_names,
                    order=order, 
                    fun=fun, 
                    nbz=0,  # Fixed value for non-bootstrappable dimension
                    nby=random_indices[j],
                    nbx=0,  # Fixed value for non-bootstrappable dimension
                    spacing=sp_value,
                    num_bootstrappable=num_bootstrappable,
                    bootstrappable_dims=bootstrappable_dims,
                    boot_indexes=boot_indexes
                )
            else:  # x-dimension
                return calculate_structure_function_3d(
                    ds=ds,
                    dims=dims,
                    variables_names=variables_names,
                    order=order, 
                    fun=fun, 
                    nbz=0,  # Fixed value for non-bootstrappable dimension
                    nby=0,  # Fixed value for non-bootstrappable dimension
                    nbx=random_indices[j],
                    spacing=sp_value,
                    num_bootstrappable=num_bootstrappable,
                    bootstrappable_dims=bootstrappable_dims,
                    boot_indexes=boot_indexes
                )
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
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for bootstrappable dimensions
        nb_indices = {}
        for dim in bootstrappable_dims:
            nb_indices[dim] = np.random.choice(indexes[dim].shape[1], size=nbootstrap)
        
        # Identify the non-bootstrappable dimension
        non_bootstrap_dim = [dim for dim in dims if dim not in bootstrappable_dims][0]
        
        # Prepare a function to run in parallel
        def simulate_bootstrap(j):
            nbz = nb_indices[dims[0]][j] if dims[0] in bootstrappable_dims else 0
            nby = nb_indices[dims[1]][j] if dims[1] in bootstrappable_dims else 0
            nbx = nb_indices[dims[2]][j] if dims[2] in bootstrappable_dims else 0
            
            return calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order,
                fun=fun,
                nbz=nbz,
                nby=nby,
                nbx=nbx,
                spacing=sp_value,
                num_bootstrappable=num_bootstrappable,
                bootstrappable_dims=bootstrappable_dims,
                boot_indexes=boot_indexes
            )
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
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for all three dimensions
        nbz = np.random.choice(indexes[dims[0]].shape[1], size=nbootstrap) 
        nby = np.random.choice(indexes[dims[1]].shape[1], size=nbootstrap)
        nbx = np.random.choice(indexes[dims[2]].shape[1], size=nbootstrap)
        
        # Prepare a function to run in parallel
        def simulate_bootstrap(j):
            return calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order,
                fun=fun,
                nbz=nbz[j], 
                nby=nby[j], 
                nbx=nbx[j], 
                spacing=sp_value,
                num_bootstrappable=num_bootstrappable,
                bootstrappable_dims=bootstrappable_dims,
                boot_indexes=boot_indexes
            )
    
    # Run simulations in parallel
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(simulate_bootstrap)(j) for j in range(nbootstrap)
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    dx_vals = [r[1] for r in results]
    dy_vals = [r[2] for r in results]
    dz_vals = [r[3] for r in results]
    
    return sf_results, dx_vals, dy_vals, dz_vals
##############################################################################################################

#####################################3D Binning###############################################################

def bin_sf_3d(ds, variables_names, order, bins, bootsize=None, fun='longitudinal', 
            initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
            convergence_eps=0.1, n_jobs=-1):
    """
    Bin 3D structure function with optimized 3D binning performance.
    
    Parameters
    -----------
    ds : xarray.Dataset
        Dataset containing velocity components and/or scalar fields
    variables_names : list
        List of variable names to use
    order : float or tuple
        Order(s) of the structure function
    bins : dict
        Dictionary with dimensions as keys and bin edges as values
    bootsize : dict, optional
        Dictionary with dimensions as keys and bootsize as values
    fun : str, optional
        Type of structure function
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
    # Initialize and validate dataset
    dims, data_shape, valid_ds = validate_dataset_3d(ds)
    
    # Setup bootsize
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
    
    # Calculate spacings
    spacings_info, all_spacings = calculate_adaptive_spacings_3d(dims, data_shape, bootsize_dict, 
                                                              bootstrappable_dims, num_bootstrappable)
    
    # Compute boot indexes
    boot_indexes = compute_boot_indexes_3d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING BIN_SF_3D WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print(f"Bootstrap parameters: initial={initial_nbootstrap}, max={max_nbootstrap}, step={step_nbootstrap}")
    print(f"Convergence threshold: {convergence_eps}")
    print(f"Bootstrappable dimensions: {bootstrappable_dims} (count: {num_bootstrappable})")
    print("="*60 + "\n")
    
    # Quick validation
    if not isinstance(bins, dict) or not all(dim in bins for dim in dims):
        raise ValueError("'bins' must be a dictionary with all dimensions as keys and bin edges as values")
    
    # Get bin properties
    dims_order = dims  # Should be ['z', 'y', 'x']
    bins_x = np.array(bins[dims_order[2]])
    bins_y = np.array(bins[dims_order[1]])
    bins_z = np.array(bins[dims_order[0]])
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1
    n_bins_z = len(bins_z) - 1
    
    print(f"Bin dimensions: {dims_order[0]}={n_bins_z}, {dims_order[1]}={n_bins_y}, {dims_order[2]}={n_bins_x}")
    print(f"Total bins: {n_bins_x * n_bins_y * n_bins_z}")
    
    # Determine log vs linear bins
    log_bins = {}
    for dim, bin_edges in bins.items():
        if len(bin_edges) < 2:
            raise ValueError(f"Bin edges for dimension '{dim}' must have at least 2 values")
        
        # Quick check for log vs linear bins
        ratios = bin_edges[1:] / bin_edges[:-1]
        ratio_std, ratio_mean = np.std(ratios), np.mean(ratios)
        
        if ratio_std / ratio_mean < 0.01:
            log_bins[dim] = abs(ratio_mean - 1.0) > 0.01  # True for log, False for linear
        else:
            log_bins[dim] = False  # Irregular spacing treated as linear
        
        print(f"Bin type for {dim}: {'logarithmic' if log_bins[dim] else 'linear'}")
    
    # Calculate bin centers
    if log_bins.get(dims_order[2], False):
        x_centers = np.sqrt(bins_x[:-1] * bins_x[1:])
    else:
        x_centers = 0.5 * (bins_x[:-1] + bins_x[1:])
        
    if log_bins.get(dims_order[1], False):
        y_centers = np.sqrt(bins_y[:-1] * bins_y[1:])
    else:
        y_centers = 0.5 * (bins_y[:-1] + bins_y[1:])
        
    if log_bins.get(dims_order[0], False):
        z_centers = np.sqrt(bins_z[:-1] * bins_z[1:])
    else:
        z_centers = 0.5 * (bins_z[:-1] + bins_z[1:])
    
    # Special case: no bootstrappable dimensions
    if num_bootstrappable == 0:
        print("\nNo bootstrappable dimensions available. "
             "Calculating structure function once with full dataset.")
        
        # Calculate structure function once with the entire dataset
        results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
            ds=valid_ds,
            dims=dims,
            variables_names=variables_names,
            order=order,
            fun=fun,
            num_bootstrappable=num_bootstrappable
        )
        
        # Bin the results
        valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals) & ~np.isnan(dz_vals)
        valid_results = results[valid_mask]
        valid_dx = dx_vals[valid_mask]
        valid_dy = dy_vals[valid_mask]
        valid_dz = dz_vals[valid_mask]
        
        # Create 3D binning grid
        x_bins_idx = np.clip(np.digitize(valid_dx, bins_x) - 1, 0, n_bins_x - 1)
        y_bins_idx = np.clip(np.digitize(valid_dy, bins_y) - 1, 0, n_bins_y - 1)
        z_bins_idx = np.clip(np.digitize(valid_dz, bins_z) - 1, 0, n_bins_z - 1)
        
        # Initialize result arrays
        sf_means = np.full((n_bins_z, n_bins_y, n_bins_x), np.nan)
        sf_stds = np.full((n_bins_z, n_bins_y, n_bins_x), np.nan)
        point_counts = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32)
        
        # Calculate weights as a volume of separation distances
        weights = np.abs(valid_dx * valid_dy * valid_dz)
        
        # Create a unique bin ID for each point
        bin_ids = z_bins_idx * (n_bins_y * n_bins_x) + y_bins_idx * n_bins_x + x_bins_idx
        
        # Use numpy's unique with counts for fast aggregation
        unique_bins, inverse_indices, counts = np.unique(bin_ids, return_inverse=True, return_counts=True)
        
        # Process each unique bin
        for i, bin_id in enumerate(unique_bins):
            z_idx = bin_id // (n_bins_y * n_bins_x)
            y_idx = (bin_id % (n_bins_y * n_bins_x)) // n_bins_x
            x_idx = bin_id % n_bins_x
            
            # Valid bin check
            if z_idx < 0 or z_idx >= n_bins_z or y_idx < 0 or y_idx >= n_bins_y or x_idx < 0 or x_idx >= n_bins_x:
                continue
            
            # Get mask for this bin
            bin_mask = inverse_indices == i
            bin_count = counts[i]
            
            # Extract values for this bin
            bin_sf = valid_results[bin_mask]
            bin_weights = weights[bin_mask]
            
            # Update counts
            point_counts[z_idx, y_idx, x_idx] = bin_count
            
            # Calculate weighted mean
            weight_sum = np.sum(bin_weights)
            if weight_sum > 0:
                norm_weights = bin_weights / weight_sum
                sf_means[z_idx, y_idx, x_idx] = np.sum(bin_sf * norm_weights)
                sf_stds[z_idx, y_idx, x_idx] = np.sqrt(np.sum(norm_weights * (bin_sf - sf_means[z_idx, y_idx, x_idx])**2))
        
        # Create output dataset
        ds_binned = xr.Dataset(
            data_vars={
                'sf': ((dims_order[0], dims_order[1], dims_order[2]), sf_means),
                'sf_std': ((dims_order[0], dims_order[1], dims_order[2]), sf_stds),
                'point_counts': ((dims_order[0], dims_order[1], dims_order[2]), point_counts)
            },
            coords={
                dims_order[2]: x_centers,
                dims_order[1]: y_centers,
                dims_order[0]: z_centers
            },
            attrs={
                'bin_type_x': 'logarithmic' if log_bins.get(dims_order[2], False) else 'linear',
                'bin_type_y': 'logarithmic' if log_bins.get(dims_order[1], False) else 'linear',
                'bin_type_z': 'logarithmic' if log_bins.get(dims_order[0], False) else 'linear',
                'order': str(order),
                'function_type': fun,
                'variables': variables_names,
                'bootstrappable_dimensions': 'none'
            }
        )
        
        # Add bin edges to the dataset
        ds_binned[f'{dims_order[2]}_bins'] = ((dims_order[2], 'edge'), np.column_stack([bins_x[:-1], bins_x[1:]]))
        ds_binned[f'{dims_order[1]}_bins'] = ((dims_order[1], 'edge'), np.column_stack([bins_y[:-1], bins_y[1:]]))
        ds_binned[f'{dims_order[0]}_bins'] = ((dims_order[0], 'edge'), np.column_stack([bins_z[:-1], bins_z[1:]]))
        
        print("3D SF COMPLETED SUCCESSFULLY (no bootstrapping)!")
        print("="*60)
        
        return ds_binned
    
    # Normal bootstrapping case (1, 2, or 3 bootstrappable dimensions)
    # Get available spacings
    spacing_values = all_spacings
    print(f"Available spacings: {spacing_values}")
    
    # Initialize result arrays
    sf_totals = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.float64)
    sf_sq_totals = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.float64)
    weight_totals = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32) 
    point_counts = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32)
    bin_density = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.float32)
    bin_status = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=bool)
    bin_bootstraps = np.ones((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32) * initial_nbootstrap
    
    # Initialize spacing effectiveness tracking
    bin_spacing_effectiveness = {sp: np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.float32) for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32) for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32) for sp in spacing_values}
    
    # Optimized process function with vectorized 3D binning
    def process_spacing_data(sp_value, bootstraps, add_to_counts=True):
        """Process structure function data for a specific spacing value."""
        if bootstraps <= 0:
            return
            
        print(f"  Processing spacing {sp_value} with {bootstraps} bootstraps")
            
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals, dz_vals = monte_carlo_simulation_3d(
            ds=valid_ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            nbootstrap=bootstraps, 
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims,
            fun=fun, 
            spacing=sp_value,
            n_jobs=n_jobs
        )
        
        # Bin tracking for this spacing
        bin_points_added = np.zeros((n_bins_z, n_bins_y, n_bins_x), dtype=np.int32)
        
        # Process all bootstrap samples
        for b in range(len(sf_results)):
            sf = sf_results[b]
            dx = dx_vals[b]
            dy = dy_vals[b]
            dz = dz_vals[b]
            
            # Create mask for valid values
            valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy) & ~np.isnan(dz)
            if not np.any(valid):
                continue
                
            sf_valid = sf[valid]
            dx_valid = dx[valid]
            dy_valid = dy[valid]
            dz_valid = dz[valid]
            
            # Calculate weights as separation distances volume
            weights = np.abs(dx_valid * dy_valid * dz_valid)
            
            # PERFORMANCE OPTIMIZATION: Use vectorized bin assignment
            # Create 3D bin indices using numpy's digitize
            x_indices = np.clip(np.digitize(dx_valid, bins_x) - 1, 0, n_bins_x - 1)
            y_indices = np.clip(np.digitize(dy_valid, bins_y) - 1, 0, n_bins_y - 1)
            z_indices = np.clip(np.digitize(dz_valid, bins_z) - 1, 0, n_bins_z - 1)
            
            # Create a unique bin ID for each point (much faster than nested loops)
            bin_ids = z_indices * (n_bins_y * n_bins_x) + y_indices * n_bins_x + x_indices
            
            # Use numpy's unique with counts for fast aggregation
            unique_bins, inverse_indices, counts = np.unique(bin_ids, return_inverse=True, return_counts=True)
            
            # Process each unique bin
            for i, bin_id in enumerate(unique_bins):
                z_idx = bin_id // (n_bins_y * n_bins_x)
                y_idx = (bin_id % (n_bins_y * n_bins_x)) // n_bins_x
                x_idx = bin_id % n_bins_x
                
                # Valid bin check
                if z_idx < 0 or z_idx >= n_bins_z or y_idx < 0 or y_idx >= n_bins_y or x_idx < 0 or x_idx >= n_bins_x:
                    continue
                
                # Get mask for this bin
                bin_mask = inverse_indices == i
                bin_count = counts[i]
                
                # Extract values for this bin
                bin_sf = sf_valid[bin_mask]
                bin_weights = weights[bin_mask]
                
                # Update counts
                if add_to_counts:
                    point_counts[z_idx, y_idx, x_idx] += bin_count
                    bin_points_added[z_idx, y_idx, x_idx] += bin_count
                    bin_spacing_counts[sp_value][z_idx, y_idx, x_idx] += bin_count
                
                # Calculate weighted statistics
                weight_sum = np.sum(bin_weights)
                if weight_sum > 0:
                    norm_weights = bin_weights / weight_sum
                    
                    # Update accumulators
                    sf_totals[z_idx, y_idx, x_idx] += np.sum(bin_sf * norm_weights)
                    sf_sq_totals[z_idx, y_idx, x_idx] += np.sum((bin_sf**2) * norm_weights)
                    weight_totals[z_idx, y_idx, x_idx] += 1
        
        # Update spacing effectiveness
        if add_to_counts and bootstraps > 0:
            # Vectorized update of effectiveness
            mask = bin_points_added > 0
            if np.any(mask):
                bin_spacing_effectiveness[sp_value][mask] = bin_points_added[mask] / bootstraps
                bin_spacing_bootstraps[sp_value][mask] += bootstraps
        
        # Clean memory
        del sf_results, dx_vals, dy_vals, dz_vals
        gc.collect()
    
    # Process initial bootstraps
    print("\nINITIAL BOOTSTRAP PHASE")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    for sp_value in spacing_values:
        process_spacing_data(sp_value, init_samples_per_spacing, True)
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    total_points = np.sum(point_counts)
    if total_points > 0:
        # Calculate all bin volumes at once
        x_widths = bins_x[1:] - bins_x[:-1]
        y_widths = bins_y[1:] - bins_y[:-1]
        z_widths = bins_z[1:] - bins_z[:-1]
        
        # Create meshgrid of widths
        Z, Y, X = np.meshgrid(z_widths, y_widths, x_widths, indexing='ij')
        bin_volumes = Z * Y * X
        
        # Vectorized density calculation
        bin_density = np.divide(point_counts, bin_volumes * total_points, 
                              out=np.zeros_like(bin_density, dtype=np.float32), 
                              where=bin_volumes > 0)
    
    # Normalize density
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    if max_density > 0:
        bin_density /= max_density
    
    print(f"Total points collected: {total_points}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins_z * n_bins_y * n_bins_x}")
    print(f"Maximum density bin has {np.max(point_counts)} points")
    
    # Calculate adaptive steps
    bootstrap_steps = np.maximum(
        step_nbootstrap, 
        (step_nbootstrap * (1 + 2 * bin_density)).astype(int)
    )
    
    # Fast calculation of statistics
    def calculate_bin_statistics():
        means = np.full((n_bins_z, n_bins_y, n_bins_x), np.nan)
        stds = np.full((n_bins_z, n_bins_y, n_bins_x), np.nan)
        
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
        z_idxs, y_idxs, x_idxs = np.where(unconverged)
        for k, j, i in zip(z_idxs, y_idxs, x_idxs):
            bin_list.append((k, j, i, bin_density[k, j, i]))
        
        # Sort by density (highest first)
        bin_list.sort(key=lambda x: x[3], reverse=True)
        
        # Track how many bins converged in this iteration
        bins_converged_in_iteration = 0
        max_reached_in_iteration = 0
        
        # Process bins in order of decreasing density
        for bin_idx, (k, j, i, density) in enumerate(bin_list):
            # Skip if already converged
            if bin_status[k, j, i]:
                continue
                
            print(f"\nProcessing bin ({k},{j},{i}) - Density: {density:.4f} - " + 
                 f"Current bootstraps: {bin_bootstraps[k, j, i]} - " + 
                 f"Current std: {sf_stds[k, j, i]:.6f} - " + 
                 f"Points: {point_counts[k, j, i]}")
                 
            # Use exact bootstrap step value
            step = bootstrap_steps[k, j, i]
            print(f"  Adding {step} more bootstraps to bin ({k},{j},{i})")
            
            # Calculate spacing effectiveness for this bin
            spacing_effectiveness = {sp: bin_spacing_effectiveness[sp][k, j, i] for sp in spacing_values}
            
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
                    sp_additional = 0.0
                
                sp_additional = min(sp_additional, remaining_step)
                
                # Process this spacing
                process_spacing_data(sp_value, sp_additional, False)
                
                # Update counters
                total_additional += sp_additional
                remaining_step -= sp_additional
                
                # Stop if we've allocated all bootstraps
                if remaining_step <= 0:
                    break
            
            # Update bootstrap counts
            bin_bootstraps[k, j, i] += total_additional
            
            # Recalculate statistics
            sf_means, sf_stds = calculate_bin_statistics()
            
            # Check for convergence or max bootstraps
            if sf_stds[k, j, i] <= convergence_eps:
                bin_status[k, j, i] = True
                print(f"  Bin ({k},{j},{i}) CONVERGED after additional bootstraps with std {sf_stds[k, j, i]:.6f} <= {convergence_eps}")
                bins_converged_in_iteration += 1
            elif bin_bootstraps[k, j, i] >= max_nbootstrap:
                bin_status[k, j, i] = True
                print(f"  Bin ({k},{j},{i}) reached MAX BOOTSTRAPS {max_nbootstrap}")
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
    
    # Create output dataset
    print("\nCreating output dataset...")
    coord_dims = {
        dims_order[2]: x_centers,
        dims_order[1]: y_centers,
        dims_order[0]: z_centers
    }
    
    ds_binned = xr.Dataset(
        data_vars={
            'sf': ((dims_order[0], dims_order[1], dims_order[2]), sf_means),
            'sf_std': ((dims_order[0], dims_order[1], dims_order[2]), sf_stds),
            'nbootstraps': ((dims_order[0], dims_order[1], dims_order[2]), bin_bootstraps),
            'density': ((dims_order[0], dims_order[1], dims_order[2]), bin_density),
            'point_counts': ((dims_order[0], dims_order[1], dims_order[2]), point_counts),
            'converged': ((dims_order[0], dims_order[1], dims_order[2]), bin_status)
        },
        coords=coord_dims,
        attrs={
            'bin_type_x': 'logarithmic' if log_bins.get(dims_order[2], False) else 'linear',
            'bin_type_y': 'logarithmic' if log_bins.get(dims_order[1], False) else 'linear',
            'bin_type_z': 'logarithmic' if log_bins.get(dims_order[0], False) else 'linear',
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'order': str(order),
            'function_type': fun,
            'spacing_values': list(spacing_values),
            'variables': variables_names,
            'bootstrappable_dimensions': ','.join(bootstrappable_dims)
        }
    )
    
    # Add bin edges to the dataset
    ds_binned[f'{dims_order[2]}_bins'] = ((dims_order[2], 'edge'), np.column_stack([bins_x[:-1], bins_x[1:]]))
    ds_binned[f'{dims_order[1]}_bins'] = ((dims_order[1], 'edge'), np.column_stack([bins_y[:-1], bins_y[1:]]))
    ds_binned[f'{dims_order[0]}_bins'] = ((dims_order[0], 'edge'), np.column_stack([bins_z[:-1], bins_z[1:]]))
    
    print("3D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned
    
##############################################################################################################

#########################################3D Isotropic#########################################################

def get_isotropic_sf_3d(ds, variables_names, order=2.0, bins=None, bootsize=None,
                       initial_nbootstrap=100, max_nbootstrap=1000, 
                       step_nbootstrap=100, fun='longitudinal', 
                       n_bins_theta=36, n_bins_phi=18, 
                       window_size_theta=None, window_size_phi=None, window_size_r=None,
                       convergence_eps=0.1, n_jobs=-1):
    """
    Get isotropic (spherically binned) structure function results for 3D data.
    
    Parameters
    -----------
    ds : xarray.Dataset
        Dataset containing velocity components and/or scalar fields
    variables_names : list
        List of variable names to use
    order : float or tuple
        Order(s) of the structure function
    bins : dict
        Dictionary with 'r' as key and bin edges as values
    bootsize : dict, optional
        Dictionary with dimensions as keys and bootsize as values
    initial_nbootstrap : int, optional
        Initial number of bootstrap samples
    max_nbootstrap : int, optional
        Maximum number of bootstrap samples
    step_nbootstrap : int, optional
        Step size for increasing bootstrap samples
    fun : str, optional
        Type of structure function
    n_bins_theta : int, optional
        Number of azimuthal angular bins (0 to 2π)
    n_bins_phi : int, optional
        Number of polar angular bins (0 to π)
    window_size_theta : int, optional
        Window size for theta bootstrapping
    window_size_phi : int, optional
        Window size for phi bootstrapping
    window_size_r : int, optional
        Window size for radial bootstrapping
    convergence_eps : float, optional
        Convergence threshold for bin standard deviation
    n_jobs : int, optional
        Number of jobs for parallel processing
    
    Returns
    --------
    xarray.Dataset
        Dataset with isotropic structure function results
    """
    # Initialize and validate dataset
    dims, data_shape, valid_ds = validate_dataset_3d(ds)
    
    # Setup bootsize
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
    
    # Calculate spacings
    spacings_info, all_spacings = calculate_adaptive_spacings_3d(dims, data_shape, bootsize_dict, 
                                                              bootstrappable_dims, num_bootstrappable)
    
    # Compute boot indexes
    boot_indexes = compute_boot_indexes_3d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING ISOTROPIC_SF_3D WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print(f"Bootstrap parameters: initial={initial_nbootstrap}, max={max_nbootstrap}, step={step_nbootstrap}")
    print(f"Convergence threshold: {convergence_eps}")
    print(f"Bootstrappable dimensions: {bootstrappable_dims} (count: {num_bootstrappable})")
    print("="*60 + "\n")
    
    # Validate bins
    if bins is None or 'r' not in bins:
        raise ValueError("'bins' must be a dictionary with 'r' as key and bin edges as values")
    
    r_bins = np.array(bins['r'])
    if len(r_bins) < 2:
        raise ValueError("Bin edges for 'r' must have at least 2 values")
    
    # Determine bin type (log or linear)
    ratios = r_bins[1:] / r_bins[:-1]
    ratio_std = np.std(ratios)
    ratio_mean = np.mean(ratios)
    
    if ratio_std / ratio_mean < 0.01:
        if np.abs(ratio_mean - 1.0) < 0.01:
            log_bins = False  # Linear bins
            r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
            print("Detected linear binning for radial dimension")
        else:
            log_bins = True  # Log bins
            r_centers = np.sqrt(r_bins[:-1] * r_bins[1:])
            print("Detected logarithmic binning for radial dimension")
    else:
        log_bins = False  # Default to linear if irregular spacing
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        print("Detected irregular bin spacing for radial dimension, treating as linear")
    
    n_bins_r = len(r_centers)
    
    # Default window sizes if not provided
    if window_size_theta is None:
        window_size_theta = max(n_bins_theta // 3, 1)
    if window_size_phi is None:
        window_size_phi = max(n_bins_phi // 3, 1)
    if window_size_r is None:
        window_size_r = max(n_bins_r // 3, 1)
    
    print(f"Using {n_bins_r} radial bins, {n_bins_theta} azimuthal bins, and {n_bins_phi} polar bins")
    print(f"Using window size {window_size_theta} for theta, {window_size_phi} for phi, and {window_size_r} for r")
    
    # Set up angular bins
    theta_bins = np.linspace(-np.pi, np.pi, n_bins_theta + 1)  # Azimuthal angle (0 to 2π)
    phi_bins = np.linspace(0, np.pi, n_bins_phi + 1)           # Polar angle (0 to π)
    
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    phi_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    
    # Special case: no bootstrappable dimensions
    if num_bootstrappable == 0:
        print("\nNo bootstrappable dimensions available. "
             "Calculating structure function once with full dataset.")
        
        # Calculate structure function once with the entire dataset
        results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
            ds=valid_ds,
            dims=dims,
            variables_names=variables_names,
            order=order,
            fun=fun,
            num_bootstrappable=num_bootstrappable
        )
        
        # Filter out invalid values
        valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals) & ~np.isnan(dz_vals)
        valid_results = results[valid_mask]
        valid_dx = dx_vals[valid_mask]
        valid_dy = dy_vals[valid_mask]
        valid_dz = dz_vals[valid_mask]
        
        if len(valid_results) == 0:
            raise ValueError("No valid results found to bin")
        
        # Convert to spherical coordinates
        r_valid = np.sqrt(valid_dx**2 + valid_dy**2 + valid_dz**2)
        theta_valid = np.arctan2(valid_dy, valid_dx)  # Azimuthal angle (-π to π)
        phi_valid = np.arccos(np.clip(valid_dz / np.maximum(r_valid, 1e-10), -1.0, 1.0))  # Polar angle (0 to π)
        
        # Create bin indices using numpy's digitize
        r_indices = np.clip(np.digitize(r_valid, r_bins) - 1, 0, n_bins_r - 1)
        theta_indices = np.clip(np.digitize(theta_valid, theta_bins) - 1, 0, n_bins_theta - 1)
        phi_indices = np.clip(np.digitize(phi_valid, phi_bins) - 1, 0, n_bins_phi - 1)
        
        # Initialize arrays for binning
        sf_means = np.full(n_bins_r, np.nan)
        sf_stds = np.full(n_bins_r, np.nan)
        point_counts = np.zeros(n_bins_r, dtype=np.int32)
        sfr = np.full((n_bins_phi, n_bins_theta, n_bins_r), np.nan)
        sfr_counts = np.zeros((n_bins_phi, n_bins_theta, n_bins_r), dtype=np.int32)
        
        # Calculate weights (using distance)
        weights = r_valid
        
        # Process radial bins
        for r_idx in range(n_bins_r):
            # Get mask for this radial bin
            r_bin_mask = r_indices == r_idx
            if not np.any(r_bin_mask):
                continue
                
            # Extract values for this bin
            bin_sf = valid_results[r_bin_mask]
            bin_weights = weights[r_bin_mask]
            bin_thetas = theta_valid[r_bin_mask]
            bin_phis = phi_valid[r_bin_mask]
            bin_theta_indices = theta_indices[r_bin_mask]
            bin_phi_indices = phi_indices[r_bin_mask]
                
            # Update counts
            point_counts[r_idx] = np.sum(r_bin_mask)
            
            # Calculate weighted mean
            weight_sum = np.sum(bin_weights)
            if weight_sum > 0:
                norm_weights = bin_weights / weight_sum
                sf_means[r_idx] = np.sum(bin_sf * norm_weights)
                sf_stds[r_idx] = np.sqrt(np.sum(norm_weights * (bin_sf - sf_means[r_idx])**2))
                
            # Process angular bins for this radial bin
            for phi_idx in range(n_bins_phi):
                for theta_idx in range(n_bins_theta):
                    # Get mask for this angular-radial bin
                    angular_mask = (bin_phi_indices == phi_idx) & (bin_theta_indices == theta_idx)
                    if not np.any(angular_mask):
                        continue
                    
                    # Extract values for this angular-radial bin
                    angular_sf = bin_sf[angular_mask]
                    angular_weights = bin_weights[angular_mask]
                    angular_weight_sum = np.sum(angular_weights)
                    
                    if angular_weight_sum > 0:
                        # Calculate weighted average
                        angular_norm_weights = angular_weights / angular_weight_sum
                        sfr[phi_idx, theta_idx, r_idx] = np.sum(angular_sf * angular_norm_weights)
                        sfr_counts[phi_idx, theta_idx, r_idx] = np.sum(angular_mask)

        # Calculate confidence intervals
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_upper = np.full_like(sf_means, np.nan)
        ci_lower = np.full_like(sf_means, np.nan)
        
        # For bins with data
        valid_bins = ~np.isnan(sf_means)
        if np.any(valid_bins):
            # If we have enough points, use standard error
            bins_with_multiple_points = (point_counts[valid_bins] > 1)
            if np.any(bins_with_multiple_points):
                indices = np.where(valid_bins)[0][bins_with_multiple_points]
                ci_upper[indices] = sf_means[indices] + z_score * sf_stds[indices] / np.sqrt(point_counts[indices])
                ci_lower[indices] = sf_means[indices] - z_score * sf_stds[indices] / np.sqrt(point_counts[indices])
            
            # For bins with only one point, just use the mean
            bins_with_one_point = (point_counts[valid_bins] == 1)
            if np.any(bins_with_one_point):
                indices = np.where(valid_bins)[0][bins_with_one_point]
                ci_upper[indices] = sf_means[indices]
                ci_lower[indices] = sf_means[indices]
        
        # Calculate error metrics for final results
        print("\nCalculating error metrics and confidence intervals...")
        
        # Error of isotropy
        eiso = np.zeros(n_bins_r)
        
        # Create sliding windows for theta and phi bootstrapping
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
            
            # Calculate isotropy error by averaging over angular windows
            for i_phi in range(n_samples_phi):
                phi_idx = indices_phi[i_phi]
                for i_theta in range(n_samples_theta):
                    theta_idx = indices_theta[i_theta]
                    mean_sf = bn.nanmean(sfr[np.ix_(phi_idx, theta_idx, range(n_bins_r))])
                    eiso += np.abs(mean_sf - sf_means)
            
            # Normalize by number of samples
            eiso /= max(1, n_samples_phi * n_samples_theta)
        else:
            print("Warning: Window sizes for theta/phi are too large. Skipping isotropy error calculation.")
        
        # Create sliding windows for r bootstrapping
        if n_bins_r > window_size_r:
            indices_r = sliding_window_view(
                np.arange(n_bins_r), 
                (n_bins_r - window_size_r + 1,), 
                writeable=False
            )[::1]
            
            n_samples_r = len(indices_r)
            
            # Use a subset of bins for homogeneity
            r_subset = r_centers[indices_r[0]]
            
            # Calculate mean across all angles
            meanh = np.zeros(len(r_subset))
            ehom = np.zeros(len(r_subset))
            
            for i in range(n_samples_r):
                idx = indices_r[i]
                meanh += bn.nanmean(sfr[:, :, idx])
            
            meanh /= max(1, n_samples_r)  # Avoid division by zero
            
            for i in range(n_samples_r):
                idx = indices_r[i]
                ehom += np.abs(bn.nanmean(sfr[:, :, idx]) - meanh)
            
            ehom /= max(1, n_samples_r)  # Avoid division by zero
        else:
            print("Warning: Window size for r is too large. Using all r bins instead.")
            r_subset = r_centers
            meanh = bn.nanmean(sfr, axis=(0, 1))
            ehom = np.zeros_like(meanh)
        
        # Create output dataset
        ds_iso = xr.Dataset(
            data_vars={
                'sf_spherical': (('phi', 'theta', 'r'), sfr),          # Angular-radial values
                'sf': (('r'), sf_means),                                # Isotropic values
                'error_isotropy': (('r'), eiso),                        # Isotropy error
                'std': (('r'), sf_stds),                                # Standard deviation
                'ci_upper': (('r'), ci_upper),                          # Upper confidence interval
                'ci_lower': (('r'), ci_lower),                          # Lower confidence interval
                'error_homogeneity': (('r_subset'), ehom),              # Homogeneity error
                'point_counts': (('r'), point_counts),                  # Point counts
            },
            coords={
                'r': r_centers,
                'r_subset': r_subset,
                'theta': theta_centers,
                'phi': phi_centers
            },
            attrs={
                'order': str(order),
                'function_type': fun,
                'window_size_theta': window_size_theta,
                'window_size_phi': window_size_phi,
                'window_size_r': window_size_r,
                'bin_type': 'logarithmic' if log_bins else 'linear',
                'variables': variables_names,
                'bootstrappable_dimensions': 'none'
            }
        )
        
        # Add bin edges to the dataset
        ds_iso['r_bins'] = (('r_edge'), r_bins)
        ds_iso['theta_bins'] = (('theta_edge'), theta_bins)
        ds_iso['phi_bins'] = (('phi_edge'), phi_bins)
        
        print("ISOTROPIC SF 3D COMPLETED SUCCESSFULLY (no bootstrapping)!")
        print("="*60)
        
        return ds_iso
    
    # Normal bootstrapping case (1, 2, or 3 bootstrappable dimensions)
    # Get available spacings
    spacing_values = all_spacings
    print(f"Available spacings: {spacing_values}")
    
    # Initialize result arrays
    sf_totals = np.zeros(n_bins_r, dtype=np.float64)       # Sum(sf * weight)
    sf_sq_totals = np.zeros(n_bins_r, dtype=np.float64)    # Sum(sf^2 * weight)
    weight_totals = np.zeros(n_bins_r, dtype=np.int32)     # Sum(weight)
    point_counts = np.zeros(n_bins_r, dtype=np.int32)      # Points per bin
    bin_density = np.zeros(n_bins_r, dtype=np.float32)     # Bin density distribution
    bin_status = np.zeros(n_bins_r, dtype=bool)            # Convergence status
    bin_bootstraps = np.ones(n_bins_r, dtype=np.int32) * initial_nbootstrap  # Bootstraps per bin
    
    # Arrays for angular-radial bins
    sfr = np.full((n_bins_phi, n_bins_theta, n_bins_r), np.nan)        # Angular-radial values
    sfr_counts = np.zeros((n_bins_phi, n_bins_theta, n_bins_r), dtype=np.int32)  # Counts per bin
    
    # Initialize spacing effectiveness tracking for adaptive sampling
    bin_spacing_effectiveness = {sp: np.zeros(n_bins_r, dtype=np.float32) for sp in spacing_values}
    bin_spacing_bootstraps = {sp: np.zeros(n_bins_r, dtype=np.int32) for sp in spacing_values}
    bin_spacing_counts = {sp: np.zeros(n_bins_r, dtype=np.int32) for sp in spacing_values}
    
    # Optimized process function with vectorized binning
    def process_spacing_data(sp_value, bootstraps, add_to_counts=True):
        """
        Process data for a specific spacing value with the specified number of bootstraps
        
        Parameters
        ----------
        sp_value : int
            Spacing value to use
        bootstraps : int
            Number of bootstrap samples to run
        add_to_counts : bool
            Whether to add points to bin counts for density calculation
            
        Returns
        -------
        numpy.ndarray
            Bin points added - number of points added to each bin
        """
        if bootstraps <= 0:
            return np.zeros(n_bins_r, dtype=np.int32)
                
        print(f"  Processing spacing {sp_value} with {bootstraps} bootstraps")
                
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals, dz_vals = monte_carlo_simulation_3d(
            ds=valid_ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            nbootstrap=bootstraps, 
            bootsize=bootsize_dict,
            num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings,
            boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims,
            fun=fun, 
            spacing=sp_value,
            n_jobs=n_jobs
        )
        
        # Bin tracking for this spacing
        bin_points_added = np.zeros(n_bins_r, dtype=np.int32)
        
        # Process all bootstrap samples
        for b in range(len(sf_results)):
            sf = sf_results[b]
            dx = dx_vals[b]
            dy = dy_vals[b]
            dz = dz_vals[b]
            
            # Create mask for valid values
            valid = ~np.isnan(sf) & ~np.isnan(dx) & ~np.isnan(dy) & ~np.isnan(dz)
            if not np.any(valid):
                continue
                
            sf_valid = sf[valid]
            dx_valid = dx[valid]
            dy_valid = dy[valid]
            dz_valid = dz[valid]
            
            # Convert to spherical coordinates
            r_valid = np.sqrt(dx_valid**2 + dy_valid**2 + dz_valid**2)
            theta_valid = np.arctan2(dy_valid, dx_valid)  # Azimuthal angle (-π to π)
            phi_valid = np.arccos(np.clip(dz_valid / np.maximum(r_valid, 1e-10), -1.0, 1.0))  # Polar angle (0 to π)
            
            # Calculate weights as separation distance
            weights = r_valid
            
            # PERFORMANCE OPTIMIZATION: Use vectorized bin assignment
            # Create bin indices using numpy's digitize
            r_indices = np.clip(np.digitize(r_valid, r_bins) - 1, 0, n_bins_r - 1)
            theta_indices = np.clip(np.digitize(theta_valid, theta_bins) - 1, 0, n_bins_theta - 1)
            phi_indices = np.clip(np.digitize(phi_valid, phi_bins) - 1, 0, n_bins_phi - 1)
            
            # Create unique bin IDs for radial bins only
            r_bin_ids = np.unique(r_indices)
            
            # Process each unique radial bin
            for r_bin_id in r_bin_ids:
                # Get mask for this radial bin
                r_bin_mask = r_indices == r_bin_id
                if not np.any(r_bin_mask):
                    continue
                
                # Extract values for this radial bin
                bin_sf = sf_valid[r_bin_mask]
                bin_weights = weights[r_bin_mask]
                bin_theta_indices = theta_indices[r_bin_mask]
                bin_phi_indices = phi_indices[r_bin_mask]
                
                # Update counts if tracking
                if add_to_counts:
                    bin_count = np.sum(r_bin_mask)
                    point_counts[r_bin_id] += bin_count
                    bin_points_added[r_bin_id] += bin_count
                    bin_spacing_counts[sp_value][r_bin_id] += bin_count
                
                # Calculate weighted statistics for radial bin
                weight_sum = np.sum(bin_weights)
                if weight_sum > 0:
                    norm_weights = bin_weights / weight_sum
                    
                    # Update accumulators with vectorized operations
                    sf_totals[r_bin_id] += np.sum(bin_sf * norm_weights)
                    sf_sq_totals[r_bin_id] += np.sum((bin_sf**2) * norm_weights)
                    weight_totals[r_bin_id] += 1
                
                # Process angular bins within this radial bin
                # Get unique angular bin identifiers
                angular_ids = bin_phi_indices * n_bins_theta + bin_theta_indices
                unique_angular_ids = np.unique(angular_ids)
                
                for angular_id in unique_angular_ids:
                    phi_idx = angular_id // n_bins_theta
                    theta_idx = angular_id % n_bins_theta
                    
                    # Get mask for this angular-radial bin
                    angular_mask = (bin_phi_indices == phi_idx) & (bin_theta_indices == theta_idx)
                    if not np.any(angular_mask):
                        continue
                    
                    # Extract values for this angular-radial bin
                    angular_sf = bin_sf[angular_mask]
                    angular_weights = bin_weights[angular_mask]
                    
                    # Calculate weighted statistics
                    angular_weight_sum = np.sum(angular_weights)
                    if angular_weight_sum > 0:
                        # Normalize weights
                        angular_norm_weights = angular_weights / angular_weight_sum
                        
                        # Calculate weighted average
                        angular_weighted_sf = np.sum(angular_sf * angular_norm_weights)
                        
                        # Update angular-radial bin with weighted average
                        if np.isnan(sfr[phi_idx, theta_idx, r_bin_id]):
                            sfr[phi_idx, theta_idx, r_bin_id] = angular_weighted_sf
                        else:
                            # Weighted average with previous value
                            sfr[phi_idx, theta_idx, r_bin_id] = (
                                sfr[phi_idx, theta_idx, r_bin_id] * sfr_counts[phi_idx, theta_idx, r_bin_id] + 
                                angular_weighted_sf
                            ) / (sfr_counts[phi_idx, theta_idx, r_bin_id] + 1)
                        
                        sfr_counts[phi_idx, theta_idx, r_bin_id] += 1
        
        # Update spacing effectiveness after processing all bootstraps
        if add_to_counts and bootstraps > 0:
            # Vectorized update of effectiveness
            mask = bin_points_added > 0
            if np.any(mask):
                bin_spacing_effectiveness[sp_value][mask] = bin_points_added[mask] / bootstraps
                bin_spacing_bootstraps[sp_value][mask] += bootstraps
        
        # Clean memory
        del sf_results, dx_vals, dy_vals, dz_vals
        gc.collect()
        
        return bin_points_added
    
    # Process initial bootstraps
    print("\nINITIAL BOOTSTRAP PHASE")
    init_samples_per_spacing = max(5, initial_nbootstrap // len(spacing_values))
    for sp_value in spacing_values:
        process_spacing_data(sp_value, init_samples_per_spacing, True)
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    total_points = np.sum(point_counts)
    if total_points > 0:
        # Calculate all bin volumes at once (spherical shells)
        bin_volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
        
        # Vectorized density calculation
        bin_density = np.divide(point_counts, bin_volumes * total_points, 
                              out=np.zeros_like(bin_density, dtype=np.float32), 
                              where=bin_volumes > 0)
    
    # Normalize density
    max_density = np.max(bin_density) if np.any(bin_density > 0) else 1.0
    bin_density /= max_density
    
    print(f"Total points collected: {total_points}")
    print(f"Bins with points: {np.count_nonzero(point_counts)}/{n_bins_r}")
    print(f"Maximum density bin has {np.max(point_counts)} points")
    
    # Calculate adaptive steps
    bootstrap_steps = np.maximum(
        step_nbootstrap, 
        (step_nbootstrap * (1 + 2 * bin_density)).astype(int)
    )
    
    # Fast calculation of statistics
    def calculate_bin_statistics():
        """
        Calculate current weighted means and standard deviations for all bins
        
        Returns
        -------
        tuple
            (means, stds) arrays of weighted means and standard deviations
        """
        means = np.full(n_bins_r, np.nan)
        stds = np.full(n_bins_r, np.nan)
        
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
    
    # Calculate initial statistics
    print("\nCALCULATING INITIAL STATISTICS")
    sf_means, sf_stds = calculate_bin_statistics()
    
    # Mark bins with too few points as converged
    low_density_mask = (point_counts <= 10) & ~bin_status
    bin_status |= low_density_mask
    print(f"Marked {np.sum(low_density_mask)} low-density bins (< 10 points) as converged")
    
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
        r_idxs = np.where(unconverged)[0]
        for j in r_idxs:
            bin_list.append((j, bin_density[j]))
        
        # Sort by density (highest first)
        bin_list.sort(key=lambda x: x[1], reverse=True)
        
        # Track how many bins converged in this iteration
        bins_converged_in_iteration = 0
        max_reached_in_iteration = 0
        
        # Process bins in order of decreasing density
        for bin_idx, (j, density) in enumerate(bin_list):
            # Skip if already converged
            if bin_status[j]:
                continue
                
            print(f"\nProcessing bin {j} (r={r_centers[j]:.4f}) - Density: {density:.4f} - " + 
                 f"Current bootstraps: {bin_bootstraps[j]} - " + 
                 f"Current std: {sf_stds[j]:.6f} - " + 
                 f"Points: {point_counts[j]}")
                
            # Use exact bootstrap step value based on density
            step = bootstrap_steps[j]
            
            print(f"  Adding {step} more bootstraps to bin {j}")
            
            # Calculate spacing effectiveness for this bin
            spacing_effectiveness = {sp: bin_spacing_effectiveness[sp][j] for sp in spacing_values}
            
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
                
                # Process this spacing
                process_spacing_data(sp_value, sp_additional, False)
                
                # Update counters
                total_additional += sp_additional
                remaining_step -= sp_additional
                
                # Stop if we've allocated all bootstraps
                if remaining_step <= 0:
                    break
            
            # Update bootstrap counts
            bin_bootstraps[j] += total_additional
            
            # Recalculate statistics
            sf_means, sf_stds = calculate_bin_statistics()
            
            # Check for convergence or max bootstraps
            if sf_stds[j] <= convergence_eps:
                bin_status[j] = True
                print(f"  Bin {j} (r={r_centers[j]:.4f}) CONVERGED with std {sf_stds[j]:.6f} <= {convergence_eps}")
                bins_converged_in_iteration += 1
            elif bin_bootstraps[j] >= max_nbootstrap:
                bin_status[j] = True
                print(f"  Bin {j} (r={r_centers[j]:.4f}) reached MAX BOOTSTRAPS {max_nbootstrap}")
                max_reached_in_iteration += 1
        
        # Next iteration
        iteration += 1
        gc.collect()
    
    # Calculate error metrics for final results
    print("\nCalculating error metrics and confidence intervals...")
    
    # Error of isotropy
    eiso = np.zeros(n_bins_r)
    
    # Create sliding windows for theta bootstrapping
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
        
        # Calculate angular variations
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
    else:
        print("Warning: Window sizes for theta/phi too large. Skipping isotropy error calculation.")
    
    # Create sliding windows for r bootstrapping
    if n_bins_r > window_size_r:
        indices_r = sliding_window_view(
            np.arange(n_bins_r), 
            (n_bins_r - window_size_r + 1,), 
            writeable=False
        )[::1]
        
        n_samples_r = len(indices_r)
        
        # Use a subset of bins for homogeneity
        r_subset = r_centers[indices_r[0]]
        
        # Calculate mean across all angles
        meanh = np.zeros(len(r_subset))
        ehom = np.zeros(len(r_subset))
        
        for i in range(n_samples_r):
            idx = indices_r[i]
            meanh += bn.nanmean(sfr[:, :, idx])
        
        meanh /= max(1, n_samples_r)
        
        for i in range(n_samples_r):
            idx = indices_r[i]
            ehom += np.abs(bn.nanmean(sfr[:, :, idx]) - meanh)
        
        ehom /= max(1, n_samples_r)
    else:
        print("Warning: Window size for r is too large. Using all r bins instead.")
        r_subset = r_centers
        meanh = bn.nanmean(sfr, axis=(0, 1))
        ehom = np.zeros_like(meanh)
    
    # Calculate confidence intervals
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Use weight_totals to determine which bins have data
    has_data = weight_totals > 0
    
    ci_upper = np.full_like(sf_means, np.nan)
    ci_lower = np.full_like(sf_means, np.nan)
    
    # Only calculate CIs for bins with data
    if np.any(has_data):
        ci_upper[has_data] = sf_means[has_data] + z_score * sf_stds[has_data] / np.sqrt(weight_totals[has_data])
        ci_lower[has_data] = sf_means[has_data] - z_score * sf_stds[has_data] / np.sqrt(weight_totals[has_data])
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_iso = xr.Dataset(
        data_vars={
            'sf_spherical': (('phi', 'theta', 'r'), sfr),          # Angular-radial values
            'sf': (('r'), sf_means),                                # Isotropic values
            'error_isotropy': (('r'), eiso),                        # Isotropy error
            'std': (('r'), sf_stds),                                # Standard deviation
            'ci_upper': (('r'), ci_upper),                          # Upper confidence interval
            'ci_lower': (('r'), ci_lower),                          # Lower confidence interval
            'error_homogeneity': (('r_subset'), ehom),              # Homogeneity error
            'n_bootstrap': (('r'), bin_bootstraps),                 # Bootstrap counts
            'bin_density': (('r'), bin_density),                    # Bin densities
            'point_counts': (('r'), point_counts),                  # Point counts
            'converged': (('r'), bin_status)                        # Convergence status
        },
        coords={
            'r': r_centers,
            'r_subset': r_subset,
            'theta': theta_centers,
            'phi': phi_centers
        },
        attrs={
            'order': str(order),
            'function_type': fun,
            'window_size_theta': window_size_theta,
            'window_size_phi': window_size_phi,
            'window_size_r': window_size_r,
            'convergence_eps': convergence_eps,
            'max_nbootstrap': max_nbootstrap,
            'initial_nbootstrap': initial_nbootstrap,
            'bin_type': 'logarithmic' if log_bins else 'linear',
            'variables': variables_names,
            'bootstrappable_dimensions': ','.join(bootstrappable_dims)
        }
    )
    
    # Add bin edges to the dataset
    ds_iso['r_bins'] = (('r_edge'), r_bins)
    ds_iso['theta_bins'] = (('theta_edge'), theta_bins)
    ds_iso['phi_bins'] = (('phi_edge'), phi_bins)
    
    # Add spacing effectiveness information
    for sp in spacing_values:
        # Calculate effectiveness (points per bootstrap)
        eff = np.zeros(n_bins_r)
        for j in range(n_bins_r):
            if bin_spacing_bootstraps[sp][j] > 0:
                eff[j] = bin_spacing_counts[sp][j] / bin_spacing_bootstraps[sp][j]
        
        ds_iso[f'effectiveness_spacing_{sp}'] = (('r'), eff)
        ds_iso[f'bootstraps_spacing_{sp}'] = (('r'), bin_spacing_bootstraps[sp])
    
    print("ISOTROPIC SF 3D COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_iso
##############################################################################################################
