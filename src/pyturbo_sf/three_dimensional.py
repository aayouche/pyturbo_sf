"""Three-dimensional structure function calculations."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import bottleneck as bn
import gc
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view


from .core import (validate_dataset_3d, setup_bootsize_3d, calculate_adaptive_spacings_3d,
                  compute_boot_indexes_3d, get_boot_indexes_3d, is_time_dimension)
from .utils import (fast_shift_3d, check_and_reorder_variables_3d, map_variables_by_pattern_3d,
                  calculate_time_diff_1d)

##################################Structure Functions Types########################################
def calc_default_vel_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_longitudinal_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals
    

def calc_transverse_ij(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ik(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_jk(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the scalar variable
    scalar_var = subset[scalar_name].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_transverse_ij_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    v_var = subset[v].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_ij_scalar with components {u}, {v} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ik_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_ik_scalar with components {u}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_jk_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components and scalar
    v_var = subset[v].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_jk_scalar with components {v}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_longitudinal_transverse_ij(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_ij with components {u}, {v}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_ik(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_ik with components {u}, {w}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_jk(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_jk with components {v}, {w}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_scalar_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the scalar variables
    scalar1_var = subset[scalar1_name].values
    scalar2_var = subset[scalar2_name].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D scalar-scalar structure function for {scalar1_name} and {scalar2_name}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_advective_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_pressure_work_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    dz_vals = np.full(nz * ny * nx, 0.0)
    
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
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals
##############################################################################################################

################################Main SF Function##############################################################
##############################################################################################################
def calculate_structure_function_3d(ds, dims, variables_names, order, fun='longitudinal', 
                                  nbz=0, nby=0, nbx=0, spacing=None, num_bootstrappable=0,
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
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ij':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ik':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_jk':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'scalar':
        results, dx_vals, dy_vals, dz_vals = calc_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'scalar_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_scalar_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse_ij':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse_ik':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse_jk':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ij_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ij_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ik_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ik_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_jk_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_jk_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'advective':
        results, dx_vals, dy_vals, dz_vals = calc_advective_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'pressure_work':
        results, dx_vals, dy_vals, dz_vals = calc_pressure_work_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'default_vel':
        results, dx_vals, dy_vals, dz_vals = calc_default_vel_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    else:
        raise ValueError(f"Unsupported function type: {fun}")
            
    return results, dx_vals, dy_vals, dz_vals
###############################Bootstrap Monte Carlo##########################################################

def run_bootstrap_sf_3d(args):
    """Standalone bootstrap function for parallel processing in 3D."""
    ds, dims, variables_names, order, fun, nbz, nby, nbx, spacing, num_bootstrappable, bootstrappable_dims, boot_indexes, time_dims = args
    return calculate_structure_function_3d(
        ds=ds, dims=dims, variables_names=variables_names, order=order, fun=fun,
        nbz=nbz, nby=nby, nbx=nbx, spacing=spacing, num_bootstrappable=num_bootstrappable, 
        bootstrappable_dims=bootstrappable_dims, boot_indexes=boot_indexes, time_dims=time_dims
    )

def monte_carlo_simulation_3d(ds, dims, variables_names, order, nbootstrap, bootsize, 
                            num_bootstrappable, all_spacings, boot_indexes, bootstrappable_dims,
                            fun='longitudinal', spacing=None, n_jobs=-1, backend='threading',
                            time_dims=None):
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
    list, list, list, list
        Lists of structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
            ds=ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            time_dims=time_dims  # Pass time_dims to calculate_structure_function_3d
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
    
    # Create all argument arrays for parallel processing
    all_args = []
        
    # Prepare parameters based on bootstrappable dimensions
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
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for the bootstrappable dimension
        random_indices = np.random.choice(indexes[bootstrap_dim].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            # Set values based on which dimension is bootstrappable
            nbz = random_indices[j] if bootstrap_dim == dims[0] else 0
            nby = random_indices[j] if bootstrap_dim == dims[1] else 0
            nbx = random_indices[j] if bootstrap_dim == dims[2] else 0
            
            args = (
                ds, dims, variables_names, order, fun, 
                nbz, nby, nbx, sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
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
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for bootstrappable dimensions
        nb_indices = {}
        for dim in bootstrappable_dims:
            nb_indices[dim] = np.random.choice(indexes[dim].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            # Set values based on which dimensions are bootstrappable
            nbz = nb_indices[dims[0]][j] if dims[0] in bootstrappable_dims else 0
            nby = nb_indices[dims[1]][j] if dims[1] in bootstrappable_dims else 0
            nbx = nb_indices[dims[2]][j] if dims[2] in bootstrappable_dims else 0
            
            args = (
                ds, dims, variables_names, order, fun,
                nbz, nby, nbx, sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
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
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for all three dimensions
        nbz = np.random.choice(indexes[dims[0]].shape[1], size=nbootstrap) 
        nby = np.random.choice(indexes[dims[1]].shape[1], size=nbootstrap)
        nbx = np.random.choice(indexes[dims[2]].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            args = (
                ds, dims, variables_names, order, fun,
                nbz[j], nby[j], nbx[j], sp_value, num_bootstrappable,
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
        delayed(run_bootstrap_sf_3d)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    dx_vals = [r[1] for r in results]
    dy_vals = [r[2] for r in results]
    dz_vals = [r[3] for r in results]
    
    return sf_results, dx_vals, dy_vals, dz_vals
##############################################################################################################

#####################################3D Binning###############################################################

"""Three-dimensional structure function calculations - Restructured."""

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
import bottleneck as bn
import gc
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view


from .core import (validate_dataset_3d, setup_bootsize_3d, calculate_adaptive_spacings_3d,
                  compute_boot_indexes_3d, get_boot_indexes_3d, is_time_dimension)
from .utils import (fast_shift_3d, check_and_reorder_variables_3d, map_variables_by_pattern_3d,
                  calculate_time_diff_1d)

##################################Structure Functions Types########################################
def calc_default_vel_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_longitudinal_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals
    

def calc_transverse_ij(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ik(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_jk(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the scalar variable
    scalar_var = subset[scalar_name].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_transverse_ij_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    v_var = subset[v].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_ij_scalar with components {u}, {v} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_ik_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components and scalar
    u_var = subset[u].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_ik_scalar with components {u}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_transverse_jk_scalar(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components and scalar
    v_var = subset[v].values
    w_var = subset[w].values
    scalar_var_values = subset[scalar_var].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D transverse_jk_scalar with components {v}, {w} and scalar {scalar_var}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_longitudinal_transverse_ij(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    v_var = subset[v].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_ij with components {u}, {v}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_ik(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    u_var = subset[u].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_ik with components {u}, {w}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_longitudinal_transverse_jk(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the velocity components
    v_var = subset[v].values
    w_var = subset[w].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D longitudinal-transverse_jk with components {v}, {w}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals

def calc_scalar_scalar_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Get the scalar variables
    scalar1_var = subset[scalar1_name].values
    scalar2_var = subset[scalar2_name].values
    
    # Get coordinate variables
    x_coord = subset[dims[2]].values
    y_coord = subset[dims[1]].values
    z_coord = subset[dims[0]].values
    
    print(f"Using 3D scalar-scalar structure function for {scalar1_name} and {scalar2_name}")
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_advective_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    
    # Loop through all points
    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
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
                results[idx] = bn.nanmean(sf_val)
                
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals


def calc_pressure_work_3d(subset, variables_names, order, dims, nz, ny, nx, time_dims=None):
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
    dz_vals = np.full(nz * ny * nx, 0.0)
    
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
                idx += 1
                
    return results, dx_vals, dy_vals, dz_vals
##############################################################################################################

################################Main SF Function##############################################################
##############################################################################################################
def calculate_structure_function_3d(ds, dims, variables_names, order, fun='longitudinal', 
                                  nbz=0, nby=0, nbx=0, spacing=None, num_bootstrappable=0,
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
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ij':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ik':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_jk':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'scalar':
        results, dx_vals, dy_vals, dz_vals = calc_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'scalar_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_scalar_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_scalar_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse_ij':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_ij(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse_ik':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_ik(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'longitudinal_transverse_jk':
        results, dx_vals, dy_vals, dz_vals = calc_longitudinal_transverse_jk(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ij_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ij_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_ik_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_ik_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'transverse_jk_scalar':
        results, dx_vals, dy_vals, dz_vals = calc_transverse_jk_scalar(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'advective':
        results, dx_vals, dy_vals, dz_vals = calc_advective_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'pressure_work':
        results, dx_vals, dy_vals, dz_vals = calc_pressure_work_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    elif fun == 'default_vel':
        results, dx_vals, dy_vals, dz_vals = calc_default_vel_3d(
            subset, variables_names, order, dims, nz, ny, nx, time_dims)
    else:
        raise ValueError(f"Unsupported function type: {fun}")
            
    return results, dx_vals, dy_vals, dz_vals
###############################Bootstrap Monte Carlo##########################################################

def run_bootstrap_sf_3d(args):
    """Standalone bootstrap function for parallel processing in 3D."""
    ds, dims, variables_names, order, fun, nbz, nby, nbx, spacing, num_bootstrappable, bootstrappable_dims, boot_indexes, time_dims = args
    return calculate_structure_function_3d(
        ds=ds, dims=dims, variables_names=variables_names, order=order, fun=fun,
        nbz=nbz, nby=nby, nbx=nbx, spacing=spacing, num_bootstrappable=num_bootstrappable, 
        bootstrappable_dims=bootstrappable_dims, boot_indexes=boot_indexes, time_dims=time_dims
    )

def monte_carlo_simulation_3d(ds, dims, variables_names, order, nbootstrap, bootsize, 
                            num_bootstrappable, all_spacings, boot_indexes, bootstrappable_dims,
                            fun='longitudinal', spacing=None, n_jobs=-1, backend='threading',
                            time_dims=None):
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
    list, list, list, list
        Lists of structure function values, DX values, DY values, DZ values
    """
    # If time_dims wasn't provided, assume no time dimensions
    if time_dims is None:
        time_dims = {dim: False for dim in dims}
    
    # If no bootstrappable dimensions, just calculate once with the full dataset
    if num_bootstrappable == 0:
        print("No bootstrappable dimensions. Calculating structure function once with full dataset.")
        results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
            ds=ds,
            dims=dims,
            variables_names=variables_names,
            order=order, 
            fun=fun,
            num_bootstrappable=num_bootstrappable,
            time_dims=time_dims  # Pass time_dims to calculate_structure_function_3d
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
    
    # Create all argument arrays for parallel processing
    all_args = []
        
    # Prepare parameters based on bootstrappable dimensions
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
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for the bootstrappable dimension
        random_indices = np.random.choice(indexes[bootstrap_dim].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            # Set values based on which dimension is bootstrappable
            nbz = random_indices[j] if bootstrap_dim == dims[0] else 0
            nby = random_indices[j] if bootstrap_dim == dims[1] else 0
            nbx = random_indices[j] if bootstrap_dim == dims[2] else 0
            
            args = (
                ds, dims, variables_names, order, fun, 
                nbz, nby, nbx, sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
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
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for bootstrappable dimensions
        nb_indices = {}
        for dim in bootstrappable_dims:
            nb_indices[dim] = np.random.choice(indexes[dim].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            # Set values based on which dimensions are bootstrappable
            nbz = nb_indices[dims[0]][j] if dims[0] in bootstrappable_dims else 0
            nby = nb_indices[dims[1]][j] if dims[1] in bootstrappable_dims else 0
            nbx = nb_indices[dims[2]][j] if dims[2] in bootstrappable_dims else 0
            
            args = (
                ds, dims, variables_names, order, fun,
                nbz, nby, nbx, sp_value, num_bootstrappable,
                bootstrappable_dims, boot_indexes, time_dims  # Add time_dims
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
            results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
                ds=ds,
                dims=dims,
                variables_names=variables_names,
                order=order, 
                fun=fun,
                num_bootstrappable=num_bootstrappable,
                time_dims=time_dims  # Pass time_dims
            )
            return [results], [dx_vals], [dy_vals], [dz_vals]
        
        # Generate random indices for all three dimensions
        nbz = np.random.choice(indexes[dims[0]].shape[1], size=nbootstrap) 
        nby = np.random.choice(indexes[dims[1]].shape[1], size=nbootstrap)
        nbx = np.random.choice(indexes[dims[2]].shape[1], size=nbootstrap)
        
        # Create arguments for each bootstrap iteration
        for j in range(nbootstrap):
            args = (
                ds, dims, variables_names, order, fun,
                nbz[j], nby[j], nbx[j], sp_value, num_bootstrappable,
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
        delayed(run_bootstrap_sf_3d)(args) for args in all_args
    )
    
    # Unpack results
    sf_results = [r[0] for r in results]
    dx_vals = [r[1] for r in results]
    dy_vals = [r[2] for r in results]
    dz_vals = [r[3] for r in results]
    
    return sf_results, dx_vals, dy_vals, dz_vals
##############################################################################################################

#####################################3D Binning - Restructured###############################################

# Helper functions for 3D binning
def _process_bootstrap_batch_3d(sf_results, dx_vals, dy_vals, dz_vals, bins_x, bins_y, bins_z,
                               bin_accumulators, target_bins, point_counts=None,
                               spacing_counts=None, sp_value=None, add_to_counts=True):
    """
    Process a batch of bootstrap results for 3D Cartesian binning.
    
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
        
        # Volume element weights
        weights = np.abs(dx_valid * dy_valid * dz_valid)
        weights = np.maximum(weights, 1e-10)
        
        # Vectorized bin assignment
        x_indices = np.clip(np.digitize(dx_valid, bins_x) - 1, 0, n_bins_x - 1)
        y_indices = np.clip(np.digitize(dy_valid, bins_y) - 1, 0, n_bins_y - 1)
        z_indices = np.clip(np.digitize(dz_valid, bins_z) - 1, 0, n_bins_z - 1)
        
        # Create unique bin IDs
        bin_ids = z_indices * n_bins_y * n_bins_x + y_indices * n_bins_x + x_indices
        
        # Process each point
        for idx in range(len(sf_valid)):
            bin_id = bin_ids[idx]
            if bin_id not in target_bin_ids:
                continue
                
            k = bin_id // (n_bins_y * n_bins_x)
            j = (bin_id % (n_bins_y * n_bins_x)) // n_bins_x
            i = bin_id % n_bins_x
            bin_key = (k, j, i)
            
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
                    point_counts[k, j, i] += 1
                if spacing_counts is not None and sp_value is not None:
                    spacing_counts[sp_value][k, j, i] += 1
        
        # After each bootstrap, store the contribution
        for bin_key in bin_accumulators:
            k, j, i = bin_key
            if (k, j, i) in target_bins:
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


def _process_bootstrap_batch_spherical(sf_results, dx_vals, dy_vals, dz_vals, r_bins, theta_bins, phi_bins,
                                     bin_accumulators, angular_accumulators, target_r_bins,
                                     point_counts=None, spacing_counts=None, sp_value=None,
                                     add_to_counts=True):
    """
    Process a batch of bootstrap results for spherical binning.
    
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
        
    Returns
    -------
    updated_r_bins : set
        Set of r bins that were updated
    """
    n_bins_r = len(r_bins) - 1
    n_bins_theta = len(theta_bins) - 1
    n_bins_phi = len(phi_bins) - 1
    updated_r_bins = set()
    
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
        
        # Volume element weights (r² for spherical coordinates)
        weights = r_valid**2
        weights = np.maximum(weights, 1e-10)
        
        # Create bin indices
        r_indices = np.clip(np.digitize(r_valid, r_bins) - 1, 0, n_bins_r - 1)
        theta_indices = np.clip(np.digitize(theta_valid, theta_bins) - 1, 0, n_bins_theta - 1)
        phi_indices = np.clip(np.digitize(phi_valid, phi_bins) - 1, 0, n_bins_phi - 1)
        
        # Process each point
        for idx in range(len(sf_valid)):
            r_idx = r_indices[idx]
            if r_idx not in target_r_bins:
                continue
            
            theta_idx = theta_indices[idx]
            phi_idx = phi_indices[idx]
            weight = weights[idx]
            value = sf_valid[idx]
            
            # Initialize accumulators if needed
            if r_idx not in bin_accumulators:
                bin_accumulators[r_idx] = {
                    'weighted_sum': 0.0,
                    'total_weight': 0.0,
                    'bootstrap_samples': []
                }
            
            angular_key = (phi_idx, theta_idx, r_idx)
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
            # Overall weighted mean
            sf_means[k, j, i] = acc['weighted_sum'] / acc['total_weight']
            
            # Bootstrap standard error
            if len(acc['bootstrap_samples']) > 1:
                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                sf_stds[k, j, i] = np.std(boot_means, ddof=1)
            else:
                sf_stds[k, j, i] = np.nan
    
    return sf_means, sf_stds


def _calculate_bootstrap_statistics_spherical(bin_accumulators, angular_accumulators, 
                                            n_bins_r, n_bins_theta, n_bins_phi):
    """
    Calculate statistics for spherical binning.
    
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
    sfr = np.full((n_bins_phi, n_bins_theta, n_bins_r), np.nan)
    sfr_counts = np.zeros((n_bins_phi, n_bins_theta, n_bins_r), dtype=np.int_)
    
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
    for (phi_idx, theta_idx, r_idx), acc in angular_accumulators.items():
        if acc['total_weight'] > 0:
            sfr[phi_idx, theta_idx, r_idx] = acc['weighted_sum'] / acc['total_weight']
            sfr_counts[phi_idx, theta_idx, r_idx] = int(acc['total_weight'])
    
    return sf_means, sf_stds, sfr, sfr_counts


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
    log_bins_z = is_log_spaced(bins_z)
    
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


def _initialize_spherical_bins(r_bins, n_theta, n_phi):
    """
    Initialize spherical bin configuration.
    
    Returns
    -------
    config : dict
        Dictionary with spherical bin configuration
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


def _calculate_bin_density_spherical(point_counts, r_bins):
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


def _process_no_bootstrap_3d(ds, dims, variables_names, order, fun, bins, time_dims):
    """Handle the special case of no bootstrappable dimensions for 3D."""
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function once
    results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
        ds=ds,
        dims=dims,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        time_dims=time_dims
    )
    
    # Initialize bins
    bins_config = _initialize_3d_bins(bins[dims[2]], bins[dims[1]], bins[dims[0]], dims)
    
    # Bin the results
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals) & ~np.isnan(dz_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    valid_dz = dz_vals[valid_mask]
    
    # Create 3D binning grid
    x_bins_idx = np.clip(np.digitize(valid_dx, bins_config['bins_x']) - 1, 
                        0, bins_config['n_bins_x'] - 1)
    y_bins_idx = np.clip(np.digitize(valid_dy, bins_config['bins_y']) - 1,
                        0, bins_config['n_bins_y'] - 1)
    z_bins_idx = np.clip(np.digitize(valid_dz, bins_config['bins_z']) - 1,
                        0, bins_config['n_bins_z'] - 1)
    
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


def _process_no_bootstrap_spherical(ds, dims, variables_names, order, fun, r_bins, n_theta, n_phi, time_dims):
    """Handle the special case of no bootstrappable dimensions for spherical."""
    print("\nNo bootstrappable dimensions available. "
          "Calculating structure function once with full dataset.")
    
    # Calculate structure function
    results, dx_vals, dy_vals, dz_vals = calculate_structure_function_3d(
        ds=ds,
        dims=dims,
        variables_names=variables_names,
        order=order,
        fun=fun,
        num_bootstrappable=0,
        time_dims=time_dims
    )
    
    # Initialize bins
    bins_config = _initialize_spherical_bins(r_bins, n_theta, n_phi)
    
    # Filter and convert to spherical
    valid_mask = ~np.isnan(results) & ~np.isnan(dx_vals) & ~np.isnan(dy_vals) & ~np.isnan(dz_vals)
    valid_results = results[valid_mask]
    valid_dx = dx_vals[valid_mask]
    valid_dy = dy_vals[valid_mask]
    valid_dz = dz_vals[valid_mask]
    
    r_valid = np.sqrt(valid_dx**2 + valid_dy**2 + valid_dz**2)
    theta_valid = np.arctan2(valid_dy, valid_dx)
    phi_valid = np.arccos(np.clip(valid_dz / np.maximum(r_valid, 1e-10), -1.0, 1.0))
    
    # Volume element weights
    weights = r_valid**2
    weights = np.maximum(weights, 1e-10)
    
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
        bin_weights = weights[r_bin_mask]
        bin_theta_indices = theta_indices[r_bin_mask]
        bin_phi_indices = phi_indices[r_bin_mask]
        
        point_counts[r_idx] = len(bin_sf)
        
        if len(bin_sf) > 0:
            normalized_weights = bin_weights / np.sum(bin_weights) * len(bin_weights)
            sf_means[r_idx] = np.average(bin_sf, weights=normalized_weights)
            
            if len(bin_sf) > 1:
                weighted_var = np.average((bin_sf - sf_means[r_idx])**2, weights=normalized_weights)
                sf_stds[r_idx] = np.sqrt(weighted_var)
        
        # Process angular bins
        for theta_idx in range(bins_config['n_bins_theta']):
            for phi_idx in range(bins_config['n_bins_phi']):
                angular_mask = (bin_theta_indices == theta_idx) & (bin_phi_indices == phi_idx)
                if not np.any(angular_mask):
                    continue
                
                angular_sf = bin_sf[angular_mask]
                angular_weights = bin_weights[angular_mask]
                
                if len(angular_sf) > 0:
                    normalized_angular_weights = angular_weights / np.sum(angular_weights) * len(angular_weights)
                    sfr[phi_idx, theta_idx, r_idx] = np.average(angular_sf, weights=normalized_angular_weights)
                    sfr_counts[phi_idx, theta_idx, r_idx] = len(angular_sf)
    
    return sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config


def _create_3d_dataset(results, bins_config, dims, order, fun, 
                      bootstrappable_dims, time_dims, convergence_eps,
                      max_nbootstrap, initial_nbootstrap, backend, variables_names):
    """Create output dataset for 3D binning."""
    ds_binned = xr.Dataset(
        data_vars={
            'sf': ((dims[0], dims[1], dims[2]), results['sf_means']),
            'sf_std': ((dims[0], dims[1], dims[2]), results['sf_stds']),
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
            'weighting': 'volume_element',
            'bootstrap_se_method': 'unweighted_std'
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


def _create_spherical_dataset(results, bins_config, order, fun, window_size_theta,
                            window_size_phi, window_size_r, convergence_eps, max_nbootstrap,
                            initial_nbootstrap, bootstrappable_dims, backend,
                            variables_names):
    """Create output dataset for spherical binning."""
    # Calculate error metrics
    eiso = _calculate_isotropy_error_3d(results['sfr'], results['sf_means'], 
                                       window_size_theta, window_size_phi)
    ehom, r_subset_indices = _calculate_homogeneity_error_3d(results['sfr'], window_size_r)
    
    # Calculate confidence intervals
    ci_upper, ci_lower = _calculate_confidence_intervals(
        results['sf_means'], results['sf_stds'], results['point_counts']
    )
    
    # Prepare data variables
    data_vars = {
        'sf_spherical': (('phi', 'theta', 'r'), results['sfr']),
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
        'theta': bins_config['theta_centers'],
        'phi': bins_config['phi_centers']
    }
    
    ds_iso = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
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
            'bootstrap_se_method': 'unweighted_std'
        }
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


def _run_adaptive_bootstrap_loop_3d(valid_ds, dims, variables_names, order, fun,
                                  bins_config, initial_nbootstrap, max_nbootstrap,
                                  step_nbootstrap, convergence_eps, spacing_values,
                                  bootsize_dict, num_bootstrappable, all_spacings,
                                  boot_indexes, bootstrappable_dims, n_jobs, backend,
                                  time_dims, is_3d=True):
    """
    Generic adaptive bootstrap loop used by both 3D and spherical functions.
    
    This function handles both 3D Cartesian and spherical cases internally.
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
    
    for sp_value in spacing_values:
        print(f"Processing spacing {sp_value} with {init_samples_per_spacing} bootstraps")
        
        # Run Monte Carlo simulation
        sf_results, dx_vals, dy_vals, dz_vals = monte_carlo_simulation_3d(
            ds=valid_ds, dims=dims, variables_names=variables_names,
            order=order, nbootstrap=init_samples_per_spacing,
            bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
            all_spacings=all_spacings, boot_indexes=boot_indexes,
            bootstrappable_dims=bootstrappable_dims, fun=fun,
            spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims
        )
        
        # Process batch based on type
        if is_3d:
            _process_bootstrap_batch_3d(
                sf_results, dx_vals, dy_vals, dz_vals,
                bins_config['bins_x'], bins_config['bins_y'], bins_config['bins_z'],
                bin_accumulators, set(all_bins), point_counts,
                bin_spacing_counts, sp_value, True
            )
        else:
            _process_bootstrap_batch_spherical(
                sf_results, dx_vals, dy_vals, dz_vals,
                bins_config['r_bins'], bins_config['theta_bins'], bins_config['phi_bins'],
                bin_accumulators, angular_accumulators, set(all_bins),
                point_counts, bin_spacing_counts, sp_value, True
            )
        
        # Update effectiveness
        _update_spacing_effectiveness(
            bin_spacing_effectiveness, bin_spacing_counts,
            bin_spacing_bootstraps, sp_value, all_bins,
            init_samples_per_spacing
        )
        
        del sf_results, dx_vals, dy_vals, dz_vals
        gc.collect()
    
    # Calculate initial statistics based on type
    if is_3d:
        sf_means[:], sf_stds[:] = _calculate_bootstrap_statistics_3d(
            bin_accumulators, result_shape
        )
    else:
        sf_means[:], sf_stds[:], sfr[:], sfr_counts[:] = _calculate_bootstrap_statistics_spherical(
            bin_accumulators, angular_accumulators,
            bins_config['n_bins_r'], bins_config['n_bins_theta'], bins_config['n_bins_phi']
        )
    
    # Calculate bin density
    print("\nCALCULATING BIN DENSITIES")
    if is_3d:
        bin_density = _calculate_bin_density_3d(point_counts, bins_config['bins_x'], 
                                              bins_config['bins_y'], bins_config['bins_z'])
    else:
        bin_density = _calculate_bin_density_spherical(point_counts, bins_config['r_bins'])
    
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
                sf_results, dx_vals, dy_vals, dz_vals = monte_carlo_simulation_3d(
                    ds=valid_ds, dims=dims, variables_names=variables_names,
                    order=order, nbootstrap=sp_bootstraps,
                    bootsize=bootsize_dict, num_bootstrappable=num_bootstrappable,
                    all_spacings=all_spacings, boot_indexes=boot_indexes,
                    bootstrappable_dims=bootstrappable_dims, fun=fun,
                    spacing=sp_value, n_jobs=n_jobs, backend=backend, time_dims=time_dims
                )
                
                # Process batch based on type (no count updates)
                if is_3d:
                    _process_bootstrap_batch_3d(
                        sf_results, dx_vals, dy_vals, dz_vals,
                        bins_config['bins_x'], bins_config['bins_y'], bins_config['bins_z'],
                        bin_accumulators, set(bin_list), None,
                        bin_spacing_counts, sp_value, False
                    )
                else:
                    _process_bootstrap_batch_spherical(
                        sf_results, dx_vals, dy_vals, dz_vals,
                        bins_config['r_bins'], bins_config['theta_bins'], bins_config['phi_bins'],
                        bin_accumulators, angular_accumulators, set(bin_list),
                        None, bin_spacing_counts, sp_value, False
                    )
                
                del sf_results, dx_vals, dy_vals, dz_vals
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
                            sf_means[k, j, i] = acc['weighted_sum'] / acc['total_weight']
                            if len(acc['bootstrap_samples']) > 1:
                                boot_means = np.array([s['mean'] for s in acc['bootstrap_samples']])
                                sf_stds[k, j, i] = np.std(boot_means, ddof=1)
                        
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


# Helper functions adapted for 3D
def _update_spacing_effectiveness(bin_spacing_effectiveness, bin_spacing_counts,
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


def _get_spacing_distribution(bin_list, spacing_effectiveness, total_bootstraps,
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
            sp_bootstraps = remaining // len([s for s, e in sorted_spacings if e >= 0])
        
        if sp_bootstraps > 0:
            distribution.append((sp_value, sp_bootstraps))
            remaining -= sp_bootstraps
    
    return distribution


def _group_bins_for_iteration(unconverged_indices, bin_density, bootstrap_steps):
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


def _evaluate_convergence(sf_stds, point_counts, bin_bootstraps,
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


def _calculate_confidence_intervals(means, stds, counts, confidence_level=0.95):
    """Calculate confidence intervals for 3D."""
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


# Main binning functions
def bin_sf_3d(ds, variables_names, order, bins, bootsize=None, fun='longitudinal', 
            initial_nbootstrap=100, max_nbootstrap=1000, step_nbootstrap=100,
            convergence_eps=0.1, n_jobs=-1, backend='threading'):
    """
    Bin 3D structure function with proper volume element weighting.
    
    Uses the same modular structure as 2D binning with helper functions.
    """
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_3d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
    spacings_info, all_spacings = calculate_adaptive_spacings_3d(dims, data_shape, bootsize_dict, 
                                                               bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_3d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING BIN_SF_3D WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print("="*60 + "\n")
    
    # Validate bins
    if not isinstance(bins, dict) or not all(dim in bins for dim in dims):
        raise ValueError("'bins' must be a dictionary with all dimensions as keys")
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, bins_config = _process_no_bootstrap_3d(
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
        
        return _create_3d_dataset(results, bins_config, dims, order, fun,
                                bootstrappable_dims, time_dims, convergence_eps,
                                max_nbootstrap, initial_nbootstrap, backend, variables_names)
    
    # Initialize bins
    bins_config = _initialize_3d_bins(bins[dims[2]], bins[dims[1]], bins[dims[0]], dims)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_3d(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, is_3d=True
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_binned = _create_3d_dataset(results, bins_config, dims, order, fun,
                                 bootstrappable_dims, time_dims, convergence_eps,
                                 max_nbootstrap, initial_nbootstrap, backend, variables_names)
    
    print("3D SF COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_binned


def get_isotropic_sf_3d(ds, variables_names, order=2.0, bins=None, bootsize=None,
                       initial_nbootstrap=100, max_nbootstrap=1000, 
                       step_nbootstrap=100, fun='longitudinal', 
                       n_bins_theta=36, n_bins_phi=18, 
                       window_size_theta=None, window_size_phi=None, window_size_r=None,
                       convergence_eps=0.1, n_jobs=-1, backend='threading'):
    """
    Get isotropic (spherically binned) structure function with volume element weighting.
    
    Uses the same modular structure as 2D isotropic binning with helper functions.
    """
    # Initialize and validate
    dims, data_shape, valid_ds, time_dims = validate_dataset_3d(ds)
    bootsize_dict, bootstrappable_dims, num_bootstrappable = setup_bootsize_3d(dims, data_shape, bootsize)
    spacings_info, all_spacings = calculate_adaptive_spacings_3d(dims, data_shape, bootsize_dict, 
                                                               bootstrappable_dims, num_bootstrappable)
    boot_indexes = compute_boot_indexes_3d(dims, data_shape, bootsize_dict, all_spacings, bootstrappable_dims)
    
    print("\n" + "="*60)
    print(f"STARTING ISOTROPIC_SF_3D WITH FUNCTION TYPE: {fun}")
    print(f"Variables: {variables_names}, Order: {order}")
    print("="*60 + "\n")
    
    # Validate bins
    if bins is None or 'r' not in bins:
        raise ValueError("'bins' must be a dictionary with 'r' as key")
    
    # Default window sizes
    if window_size_theta is None:
        window_size_theta = max(n_bins_theta // 3, 1)
    if window_size_phi is None:
        window_size_phi = max(n_bins_phi // 3, 1)
    if window_size_r is None:
        window_size_r = max((len(bins['r']) - 1) // 3, 1)
    
    # Special case: no bootstrapping
    if num_bootstrappable == 0:
        sf_means, sf_stds, point_counts, sfr, sfr_counts, bins_config = _process_no_bootstrap_spherical(
            valid_ds, dims, variables_names, order, fun, bins['r'], n_bins_theta, n_bins_phi, time_dims
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
        
        return _create_spherical_dataset(results, bins_config, order, fun,
                                       window_size_theta, window_size_phi, window_size_r,
                                       convergence_eps, max_nbootstrap,
                                       initial_nbootstrap, bootstrappable_dims,
                                       backend, variables_names)
    
    # Initialize bins
    bins_config = _initialize_spherical_bins(bins['r'], n_bins_theta, n_bins_phi)
    
    # Run adaptive bootstrap loop
    results = _run_adaptive_bootstrap_loop_3d(
        valid_ds, dims, variables_names, order, fun,
        bins_config, initial_nbootstrap, max_nbootstrap,
        step_nbootstrap, convergence_eps, all_spacings,
        bootsize_dict, num_bootstrappable, all_spacings,
        boot_indexes, bootstrappable_dims, n_jobs, backend,
        time_dims, is_3d=False
    )
    
    # Create output dataset
    print("\nCreating output dataset...")
    ds_iso = _create_spherical_dataset(
        results, bins_config, order, fun,
        window_size_theta, window_size_phi, window_size_r,
        convergence_eps, max_nbootstrap,
        initial_nbootstrap, bootstrappable_dims,
        backend, variables_names
    )
    
    print("ISOTROPIC SF 3D COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return ds_iso

##############################################################################################################

##############################################################################################################
