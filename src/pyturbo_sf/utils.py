"""Utility functions for structure function calculations."""

import numpy as np
from datetime import datetime
import bottleneck as bn


###################################Shifting Functions####################################################
# 1D

def fast_shift_1d(input_array, shift=0):
    """
    Shift 1D array by the specified integer amount and return the shifted array.
    Works with datetime arrays as well as numeric arrays.

    Parameters
    ----------
        input_array: array_like
            1-dimensional array to be shifted.
        shift: int, optional
            Shift amount.

    Returns
    -------
        shifted_array
            1D array shifted by the specified integer amount
    """
    # First, make a copy to avoid modifying the original array
    if shift == 0:
        return input_array.copy()
    
    # Create an empty array with the same shape and dtype
    shifted_array = np.empty_like(input_array)
    
    # Handle the shifted part
    shifted_array[:-shift] = input_array[shift:]
    
    # Now handle the part that needs to be filled with NA values
    if shift > 0:
        # Determine the appropriate NA value based on the data type
        if np.issubdtype(input_array.dtype, np.datetime64):
            # For numpy datetime64 arrays
            na_value = np.datetime64('NaT')
        elif isinstance(input_array.flat[0], datetime):
            # For Python datetime objects
            na_value = None  # This is placeholder, can't use na_value directly for object arrays
            # Fill with None values and return
            for i in range(-shift, 0):
                shifted_array[i] = None
            return shifted_array
        elif np.issubdtype(input_array.dtype, np.integer):
            # For integer arrays, sometimes using a sentinel value like -9999 is preferable
            # But we'll use 0 by default
            na_value = 0
        elif np.issubdtype(input_array.dtype, np.bool_):
            # For boolean arrays
            na_value = False
        else:
            # For float arrays and others that can handle NaN
            na_value = np.nan
        
        # Fill the remaining elements with the appropriate NA value
        shifted_array[-shift:] = na_value
    
    return shifted_array
    
# 2D

def fast_shift_2d(input_array, y_shift=0, x_shift=0):  # noqa: D417
    """
    Shift 2D array in x and y by the specified integer amounts and returns
    the shifted arrays.

    Parameters
    ----------
        input_array: array_like
            2-dimensional array to be shifted.
        shift_x: int, optional
            Shift amount for x shift.
        shift_y: int, optional
            Shift amount for y shift.

    Returns
    -------
        shifted_xy_array
            2D array shifted in the x-y directions by the specified integer amount
    """
    shifted_xy_array = np.full(np.shape(input_array), np.nan)

    if x_shift == 0 and y_shift == 0:
        shifted_xy_array = input_array
    elif y_shift == 0:
        shifted_xy_array[:, :-x_shift] = input_array[:, x_shift:]
    elif x_shift == 0:
        shifted_xy_array[:-y_shift, :] = input_array[y_shift:, :]
    else:
        shifted_xy_array[:-y_shift, :-x_shift] = input_array[y_shift:, x_shift:]

    return shifted_xy_array
    
# 3D

def fast_shift_3d(input_array, z_shift=0, y_shift=0, x_shift=0):
    """
    Shift 3D array in x, y, and z by the specified integer amounts and returns
    the shifted arrays.

    Parameters
    ----------
        input_array: array_like
            3-dimensional array to be shifted.
        z_shift: int, optional
            Shift amount for z shift.
        y_shift: int, optional
            Shift amount for y shift.
        x_shift: int, optional
            Shift amount for x shift.

    Returns
    -------
        shifted_xyz_array
            3D array shifted in the x-y-z directions by the specified integer amount
    """
    shifted_xyz_array = np.full(np.shape(input_array), np.nan)

    # Handle different shift combinations
    if x_shift == 0 and y_shift == 0 and z_shift == 0:
        shifted_xyz_array = input_array
    elif y_shift == 0 and z_shift == 0:
        shifted_xyz_array[:, :, :-x_shift] = input_array[:, :, x_shift:]
    elif x_shift == 0 and z_shift == 0:
        shifted_xyz_array[:, :-y_shift, :] = input_array[:, y_shift:, :]
    elif x_shift == 0 and y_shift == 0:
        shifted_xyz_array[:-z_shift, :, :] = input_array[z_shift:, :, :]
    elif z_shift == 0:
        shifted_xyz_array[:, :-y_shift, :-x_shift] = input_array[:, y_shift:, x_shift:]
    elif y_shift == 0:
        shifted_xyz_array[:-z_shift, :, :-x_shift] = input_array[z_shift:, :, x_shift:]
    elif x_shift == 0:
        shifted_xyz_array[:-z_shift, :-y_shift, :] = input_array[z_shift:, y_shift:, :]
    else:
        shifted_xyz_array[:-z_shift, :-y_shift, :-x_shift] = input_array[z_shift:, y_shift:, x_shift:]

    return shifted_xyz_array

################################################################################################################

##################################Calculate Time Difference 1D##################################################

def calculate_time_diff_1d(time_array, shift):
    """
    Calculate time difference in seconds when the coordinate is a datetime.
    
    Parameters
    ----------
    time_array : array_like
        Array of datetime values
    shift : int
        Shift amount
        
    Returns
    -------
    array_like
        Array of time differences in seconds
    """
    # Get shifted array
    shifted_array = fast_shift_1d(time_array, shift)
    
    # Initialize result array
    diff = np.full(np.shape(time_array), np.nan, dtype=float)
    
    # If shift is zero, return zeros
    if shift == 0:
        return np.zeros_like(diff)
    
    # Check if time_array contains datetime objects
    if hasattr(time_array, 'dtype') and np.issubdtype(time_array.dtype, np.datetime64):
        # For numpy datetime arrays
        valid_mask = ~np.isnat(shifted_array)
        
        # Calculate differences for valid entries
        if np.any(valid_mask):
            time_delta = shifted_array[valid_mask] - time_array[valid_mask]
            diff[valid_mask] = time_delta.astype('timedelta64[s]').astype(float)
            
    elif len(time_array) > 0 and isinstance(time_array[0], datetime):
        # For Python datetime objects
        # We need to check each element individually
        for i in range(len(time_array)):
            # NaT values for datetime are often represented as NaT
            # But could be None or other special values depending on implementation
            if (i < len(shifted_array) and 
                shifted_array[i] is not None and 
                not (hasattr(shifted_array[i], 'dtype') and np.isnat(shifted_array[i]))):
                diff[i] = (shifted_array[i] - time_array[i]).total_seconds()
    else:
        # For numeric arrays
        # NaN values will propagate naturally in numeric operations
        valid_mask = ~np.isnan(shifted_array)
        diff[valid_mask] = shifted_array[valid_mask] - time_array[valid_mask]
        
    return diff

################################################################################################################

####################################Reorder Variables###########################################################

# 2D

def map_variables_by_pattern_2d(provided, expected, plane_tuple):
    """
    Map provided variables to expected ones using common naming patterns.
    
    Parameters
    ----------
    provided : list
        List of provided variable names
    expected : list
        List of expected variable names
    plane_tuple : tuple
        Tuple of dimension names
    
    Returns
    -------
    tuple or None
        Tuple of mapped variable names or None if mapping fails
    """
    # Common naming patterns for velocity components
    var_patterns = {
        'u': ['u', 'u_vel', 'velocity_x', 'vx', 'vel_x'],
        'v': ['v', 'v_vel', 'velocity_y', 'vy', 'vel_y'],
        'w': ['w', 'w_vel', 'velocity_z', 'vz', 'vel_z']
    }
    
    # Create a mapping of expected variables to provided variables
    mapping = {}
    
    for exp in expected:
        if exp not in var_patterns:
            print(f"Warning: No patterns defined for expected variable {exp}")
            continue
            
        # Get patterns for this expected variable
        patterns = var_patterns[exp]
        found_match = False
        
        # Look for an exact match first
        for prov in provided:
            prov_lower = prov.lower()
            if prov_lower == exp:
                mapping[exp] = prov
                found_match = True
                break
                
        # If no exact match, look for pattern matches
        if not found_match:
            for prov in provided:
                prov_lower = prov.lower()
                for pattern in patterns:
                    # Check for exact match or if the pattern equals the provided variable
                    if prov_lower == pattern:
                        mapping[exp] = prov
                        found_match = True
                        break
                if found_match:
                    break
    
    # Check if we've mapped all expected variables
    mapped_vars = []
    for exp in expected:
        if exp in mapping:
            mapped_vars.append(mapping[exp])
        else:
            # Print failure information for debugging
            print(f"Failed to map expected variable '{exp}' to any of: {provided}")
            return None
    
    if len(mapped_vars) == len(expected):
        print(f"Mapped variables {provided} to {mapped_vars} for {plane_tuple} plane (expected: {expected})")
        return tuple(mapped_vars)
    
    return None


def check_and_reorder_variables_2d(variables_names, dims, fun='longitudinal'):
    """
    Check if the provided variable names match the expected components for the given plane and function type,
    and reorder them if necessary.
    
    Parameters
    ----------
    variables_names : list
        List of variable names provided by the user
    dims : list
        List of dimension names (e.g., ['y', 'x'])
    fun : str
        Type of structure function
    
    Returns
    -------
    tuple
        Tuple of variable names in the correct order for the given plane and function type
    """
    # Expected velocity component mappings for each plane
    velocity_vars = {
        ('y', 'x'): ['u', 'v'],  # (y, x) plane expects u, v components
        ('z', 'x'): ['u', 'w'],  # (z, x) plane expects u, w components
        ('z', 'y'): ['v', 'w']   # (z, y) plane expects v, w components
    }
    
    # Get the expected variables based on function type and plane
    plane_tuple = tuple(dims)
    if plane_tuple not in velocity_vars:
        raise ValueError(f"Unsupported dimension combination: {dims}")
    
    expected_vel = velocity_vars[plane_tuple]
    provided = list(variables_names)
    
    # Handle different function types
    if fun in ['longitudinal', 'transverse', 'default_vel', 'longitudinal_transverse']:
        # These functions need exactly 2 velocity components
        if len(provided) != 2:
            raise ValueError(f"{fun} structure function requires exactly 2 velocity components, got {len(provided)}")
        
        # Check if variables match expected velocity components (in any order)
        if set(provided) == set(expected_vel):
            # Variables match, but might be in wrong order
            if provided != expected_vel:
                print(f"Reordering variables from {provided} to {expected_vel} to match {plane_tuple} plane")
                return tuple(expected_vel)
            return tuple(provided)
        
        # Try to map provided variables to expected ones using pattern matching
        mapped_vars = map_variables_by_pattern_2d(provided, expected_vel, plane_tuple)
        if mapped_vars:
            return mapped_vars
    
    elif fun == 'scalar':
        # Scalar function needs exactly 1 scalar variable
        if len(provided) != 1:
            raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(provided)}")
        
        # No reordering needed for single scalar
        return tuple(provided)
    
    elif fun == 'scalar_scalar':
        # Scalar-scalar function needs exactly 2 scalar variables
        if len(provided) != 2:
            raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar variables, got {len(provided)}")
        
        # No specific ordering required for scalar-scalar
        return tuple(provided)
    
    elif fun in ['longitudinal_scalar', 'transverse_scalar']:
        # These functions need 2 velocity components and 1 scalar
        if len(provided) != 3:
            raise ValueError(f"{fun} structure function requires 2 velocity components and 1 scalar, got {len(provided)}")
        
        # Try to identify which are velocity components and which is the scalar
        vel_candidates = []
        scalar_candidates = []
        
        for var in provided:
            if any(vel_pattern in var.lower() for vel_pattern in ['u', 'v', 'w', 'vel', 'velocity']):
                vel_candidates.append(var)
            else:
                scalar_candidates.append(var)
        
        # If we can't clearly distinguish, assume the first two are velocity and the last is scalar
        if len(vel_candidates) != 2 or len(scalar_candidates) != 1:
            print(f"Warning: Could not clearly distinguish velocity components from scalar in {provided}. "
                 f"Assuming the first two are velocity components and the last is the scalar.")
            vel_candidates = provided[:2]
            scalar_candidates = [provided[2]]
        
        # Check and reorder velocity components
        mapped_vel = map_variables_by_pattern_2d(vel_candidates, expected_vel, plane_tuple)
        if mapped_vel:
            # Return velocity components first, then scalar
            return tuple(list(mapped_vel) + scalar_candidates)
    
    # If we get here, something went wrong with the mapping
    raise ValueError(f"Failed to properly map variables {provided} for {fun} structure function on {plane_tuple} plane.")

# 3D

def check_and_reorder_variables_3d(variables_names, dims, fun='longitudinal'):
    """
    Check if the provided variable names match the expected components for the given function type,
    and reorder them if necessary.
    
    Parameters
    ----------
    variables_names : list
        List of variable names provided by the user
    dims : list
        List of dimension names (e.g., ['z', 'y', 'x'])
    fun : str
        Type of structure function
    
    Returns
    -------
    tuple
        Tuple of variable names in the correct order for the given function type
    """
    # Expected velocity component mappings for 3D
    velocity_vars = ['u', 'v', 'w']  # x, y, z components
    
    # For 3D, we expect dimensions to be ['z', 'y', 'x']
    if dims != ['z', 'y', 'x']:
        raise ValueError(f"Expected dimensions to be ['z', 'y', 'x'], got {dims}")
    
    provided = list(variables_names)
    
    # Handle different function types
    if fun == 'longitudinal':
        # Longitudinal requires all 3 velocity components
        if len(provided) != 3:
            raise ValueError(f"3D longitudinal structure function requires exactly 3 velocity components, got {len(provided)}")
        
        # Check if variables match expected velocity components (in any order)
        if set(provided) == set(velocity_vars):
            # Variables match, but might be in wrong order
            if provided != velocity_vars:
                print(f"Reordering variables from {provided} to {velocity_vars} for 3D longitudinal")
                return tuple(velocity_vars)
            return tuple(provided)
        
        # Try to map provided variables to expected ones using pattern matching
        mapped_vars = map_variables_by_pattern_3d(provided, velocity_vars)
        if mapped_vars:
            return mapped_vars
    
    elif fun.startswith('transverse_'):
        # Transverse requires 2 velocity components for a specific plane
        if len(provided) != 2:
            raise ValueError(f"3D transverse structure function requires exactly 2 velocity components, got {len(provided)}")
        
        # Determine which plane based on the transverse specification
        plane = fun.split('_')[1]  # e.g., 'transverse_ij' -> 'ij'
        
        if plane == 'ij':
            # xy-plane transverse components (u, v)
            expected_vel = [velocity_vars[0], velocity_vars[1]]
        elif plane == 'ik':
            # xz-plane transverse components (u, w)
            expected_vel = [velocity_vars[0], velocity_vars[2]]
        elif plane == 'jk':
            # yz-plane transverse components (v, w)
            expected_vel = [velocity_vars[1], velocity_vars[2]]
        else:
            raise ValueError(f"Unsupported transverse plane: {plane}")
        
        # Check if variables match expected components for this plane
        if set(provided) == set(expected_vel):
            if provided != expected_vel:
                print(f"Reordering variables from {provided} to {expected_vel} for transverse {plane}")
                return tuple(expected_vel)
            return tuple(provided)
        
        # Try to map provided variables
        mapped_vars = map_variables_by_pattern_3d(provided, expected_vel)
        if mapped_vars:
            return mapped_vars
    
    elif fun == 'scalar':
        # Scalar function needs exactly 1 scalar variable
        if len(provided) != 1:
            raise ValueError(f"Scalar structure function requires exactly 1 scalar variable, got {len(provided)}")
        
        # No reordering needed for single scalar
        return tuple(provided)
    
    elif fun == 'scalar_scalar':
        # Scalar-scalar function needs exactly 2 scalar variables
        if len(provided) != 2:
            raise ValueError(f"Scalar-scalar structure function requires exactly 2 scalar variables, got {len(provided)}")
        
        # No specific ordering required for scalar-scalar
        return tuple(provided)
    
    elif fun == 'longitudinal_scalar':
        # This function needs 3 velocity components and 1 scalar
        if len(provided) != 4:
            raise ValueError(f"3D longitudinal-scalar function requires 3 velocity components and 1 scalar, got {len(provided)}")
        
        # Try to identify which are velocity components and which is the scalar
        vel_candidates = []
        scalar_candidates = []
        
        for var in provided:
            if any(vel_pattern in var.lower() for vel_pattern in ['u', 'v', 'w', 'vel', 'velocity']):
                vel_candidates.append(var)
            else:
                scalar_candidates.append(var)
        
        # If we can't clearly distinguish, assume the first three are velocity and the last is scalar
        if len(vel_candidates) != 3 or len(scalar_candidates) != 1:
            print(f"Warning: Could not clearly distinguish velocity components from scalar in {provided}. "
                 f"Assuming the first three are velocity components and the last is the scalar.")
            vel_candidates = provided[:3]
            scalar_candidates = [provided[3]]
        
        # Check and reorder velocity components
        mapped_vel = map_variables_by_pattern_3d(vel_candidates, velocity_vars)
        if mapped_vel:
            # Return velocity components first, then scalar
            return tuple(list(mapped_vel) + scalar_candidates)
    
    elif fun.startswith('transverse_') and fun.endswith('_scalar'):
        # Transverse-scalar requires 2 velocity components for a specific plane and 1 scalar
        if len(provided) != 3:
            raise ValueError(f"3D transverse-scalar function requires 2 velocity components and 1 scalar, got {len(provided)}")
        
        # Extract plane identifier (e.g., 'transverse_ij_scalar' -> 'ij')
        plane = fun.split('_')[1]
        
        # Determine expected velocity components based on plane
        if plane == 'ij':
            expected_vel = [velocity_vars[0], velocity_vars[1]]  # u, v
        elif plane == 'ik':
            expected_vel = [velocity_vars[0], velocity_vars[2]]  # u, w
        elif plane == 'jk':
            expected_vel = [velocity_vars[1], velocity_vars[2]]  # v, w
        else:
            raise ValueError(f"Unsupported transverse plane: {plane}")
        
        # Try to identify velocity components and scalar
        vel_candidates = []
        scalar_candidates = []
        
        for var in provided:
            if any(vel_pattern in var.lower() for vel_pattern in ['u', 'v', 'w', 'vel', 'velocity']):
                vel_candidates.append(var)
            else:
                scalar_candidates.append(var)
        
        # If we can't clearly distinguish
        if len(vel_candidates) != 2 or len(scalar_candidates) != 1:
            print(f"Warning: Could not clearly distinguish velocity components from scalar in {provided}. "
                 f"Assuming the first two are velocity components and the last is the scalar.")
            vel_candidates = provided[:2]
            scalar_candidates = [provided[2]]
        
        # Check and reorder velocity components
        mapped_vel = map_variables_by_pattern_3d(vel_candidates, expected_vel)
        if mapped_vel:
            # Return velocity components first, then scalar
            return tuple(list(mapped_vel) + scalar_candidates)
    
    # Handle longitudinal_transverse functions with explicit plane references
    elif fun.startswith('longitudinal_transverse_'):
        # This function needs 2 velocity components based on the plane
        if len(provided) != 2:
            raise ValueError(f"{fun} structure function requires exactly 2 velocity components, got {len(provided)}")
        
        # Extract plane identifier (e.g., 'longitudinal_transverse_ij' -> 'ij')
        plane = fun.split('_')[2]
        
        # Determine expected velocity components based on plane
        if plane == 'ij':
            expected_vel = [velocity_vars[0], velocity_vars[1]]  # u, v
        elif plane == 'ik':
            expected_vel = [velocity_vars[0], velocity_vars[2]]  # u, w
        elif plane == 'jk':
            expected_vel = [velocity_vars[1], velocity_vars[2]]  # v, w
        else:
            raise ValueError(f"Unsupported plane: {plane}")
        
        # Check if variables match expected components for this plane
        if set(provided) == set(expected_vel):
            if provided != expected_vel:
                print(f"Reordering variables from {provided} to {expected_vel} for {fun}")
                return tuple(expected_vel)
            return tuple(provided)
        
        # Try to map provided variables
        mapped_vars = map_variables_by_pattern_3d(provided, expected_vel)
        if mapped_vars:
            return mapped_vars
    
    # If we get here, something went wrong with the mapping
    raise ValueError(f"Failed to properly map variables {provided} for {fun} structure function in 3D.")


def map_variables_by_pattern_3d(provided, expected):
    """
    Map provided variables to expected ones using common naming patterns.
    
    Parameters
    ----------
    provided : list
        List of provided variable names
    expected : list
        List of expected variable names
    
    Returns
    -------
    tuple or None
        Tuple of mapped variable names or None if mapping fails
    """
    # Common naming patterns for velocity components
    var_patterns = {
        'u': ['u', 'u_vel', 'velocity_x', 'vx', 'vel_x'],
        'v': ['v', 'v_vel', 'velocity_y', 'vy', 'vel_y'],
        'w': ['w', 'w_vel', 'velocity_z', 'vz', 'vel_z']
    }
    
    # Create a mapping of expected variables to provided variables
    mapping = {}
    
    for exp in expected:
        if exp not in var_patterns:
            print(f"Warning: No patterns defined for expected variable {exp}")
            continue
            
        # Get patterns for this expected variable
        patterns = var_patterns[exp]
        found_match = False
        
        # Look for an exact match first
        for prov in provided:
            prov_lower = prov.lower()
            if prov_lower == exp:
                mapping[exp] = prov
                found_match = True
                break
                
        # If no exact match, look for pattern matches
        if not found_match:
            for prov in provided:
                prov_lower = prov.lower()
                for pattern in patterns:
                    # Check for exact match or if the pattern equals the provided variable
                    if prov_lower == pattern:
                        mapping[exp] = prov
                        found_match = True
                        break
                if found_match:
                    break
    
    # Check if we've mapped all expected variables
    mapped_vars = []
    for exp in expected:
        if exp in mapping:
            mapped_vars.append(mapping[exp])
        else:
            # Print failure information for debugging
            print(f"Failed to map expected variable '{exp}' to any of: {provided}")
            return None
    
    if len(mapped_vars) == len(expected):
        print(f"Mapped variables {provided} to {mapped_vars} (for expected: {expected})")
        return tuple(mapped_vars)
    
    return None

#####################################################################################################################
