"""Core functions for structure function calculations."""

import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view
import xarray as xr

#############################Validate Datasets########################################

# 1D

def validate_dataset_1d(ds):
    """
    Validate that the dataset has a single dimension for 1D structure function analysis.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing scalar fields with a single dimension
        
    Returns
    -------
    str
        The name of the single dimension
    dict
        Dictionary of dimension names and sizes
        
    Raises
    ------
    ValueError
        If the dataset doesn't have exactly one dimension
    """
    # Get all dimensions
    dims = list(ds.dims)
    
    # Verify there's only one dimension
    if len(dims) != 1:
        raise ValueError(f"Dataset must have exactly 1 dimension, but got {len(dims)}: {dims}. "
                        f"The structure function analysis requires 1D data.")
    
    # Get the single dimension
    dim = dims[0]
    
    # Get data shape from dataset dimensions
    data_shape = dict(ds.sizes)
    
    # Print the shapes for debugging
    print(f"Data dimensions and shapes: {data_shape}")
    
    return dim, data_shape
    
# 2D

def validate_dataset_2d(ds):
    """
    Validate the dataset has exactly 2 dimensions and ensure they're in the correct order.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing velocity components and/or scalar fields
        
    Returns
    -------
    list
        List of dimension names in correct order
    dict
        Dictionary with dimension names and sizes
    xarray.Dataset
        Potentially transposed dataset to ensure correct dimension order
        
    Raises
    ------
    ValueError
        If dataset doesn't have exactly 2 dimensions or dimensions are incompatible
    """
    # Get dimensions
    dims = list(ds.dims)
    
    # Check for exactly 2 dimensions
    if len(dims) != 2:
        raise ValueError(f"Expected exactly 2 dimensions, but got {len(dims)}: {dims}. "
                        f"The structure function analysis requires 2D data.")
    
    # Expected dimension pairs
    expected_pairs = [('y', 'x'), ('z', 'x'), ('z', 'y')]
    current_pair = tuple(dims)
    
    # If current pair is already one of the expected pairs, no need to transpose
    if current_pair in expected_pairs:
        print(f"Dimensions {current_pair} are already in the expected order")
        return dims, dict(ds.sizes), ds
    
    # Check if the dimensions are in reverse order
    reversed_pair = (dims[1], dims[0])
    if reversed_pair in expected_pairs:
        # Transpose to correct order
        transposed_ds = ds.transpose(dims[1], dims[0])
        transposed_dims = list(transposed_ds.dims)
        print(f"Transposed dimensions from {dims} to {transposed_dims}")
        return transposed_dims, dict(transposed_ds.sizes), transposed_ds
    
    # If we get here, the dimensions don't match any expected pairs even after reversing
    raise ValueError(f"Dimensions {dims} are not compatible with expected dimension pairs: {expected_pairs}. "
                    f"Please provide data with one of the expected dimension pairs.")
                    
# 3D

def validate_dataset_3d(ds):
    """
    Validate that the dataset has exactly 3 dimensions and they are in the correct order (z, y, x).
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing velocity components and/or scalar fields
        
    Returns
    -------
    list
        List of dimension names in correct order
    dict
        Dictionary with dimension names and sizes
    xarray.Dataset
        Dataset with dimensions in the correct order
        
    Raises
    ------
    ValueError
        If dataset doesn't have exactly 3 dimensions or dimensions are incompatible
    """
    # Get dimensions
    dims = list(ds.dims)
    
    # Check for exactly 3 dimensions
    if len(dims) != 3:
        raise ValueError(f"Expected exactly 3 dimensions, but got {len(dims)}: {dims}. "
                        f"The 3D structure function analysis requires 3D data.")
    
    # Expected dimension order
    expected_order = ['z', 'y', 'x']
    
    # If current order is already the expected order, no need to transpose
    if dims == expected_order:
        print(f"Dimensions {dims} are already in the expected order")
        return dims, dict(ds.sizes), ds
    
    # If the dimensions are not in the right order, transpose to correct order
    transposed_ds = ds.transpose(*expected_order)
    transposed_dims = list(transposed_ds.dims)
    print(f"Transposed dimensions to {transposed_dims}")
    
    return transposed_dims, dict(transposed_ds.sizes), transposed_ds
####################################################################################################

####################################Setup Bootsize##################################################

# 1D

def setup_bootsize_1d(dim, data_shape, bootsize=None):
    """
    Set up bootsize parameters for bootstrapping.
    
    Parameters
    ----------
    dim : str
        Name of the dataset dimension
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict or int, optional
        Bootsize for the dimension. Can be a dictionary with dimension name as key,
        or a single integer to be used for the dimension.
        
    Returns
    -------
    dict
        Dictionary with dimension name as key and bootsize as value
    list
        List of bootstrappable dimensions
    int
        Number of bootstrappable dimensions
    """
    # Default bootsize if not provided
    if bootsize is None:
        bootsize_dict = {dim: min(32, data_shape[dim] // 2)}
    elif isinstance(bootsize, dict):
        bootsize_dict = bootsize
    else:
        # If bootsize is a single integer, create a dictionary
        bootsize_dict = {dim: bootsize}
    
    print(f"Using bootsize: {bootsize_dict}")
    
    # Track which dimensions are bootstrappable
    bootstrappable_dims = []
    if bootsize_dict[dim] < data_shape[dim]:
        bootstrappable_dims.append(dim)
    else:
        print(f"Dimension {dim} has bootsize equal to or larger than dimension size. "
              f"No bootstrapping will be done across this dimension.")
    
    # Determine if we have any bootstrappable dimensions
    num_bootstrappable = len(bootstrappable_dims)
    
    return bootsize_dict, bootstrappable_dims, num_bootstrappable
    
# 2D

def setup_bootsize_2d(dims, data_shape, bootsize=None):
    """
    Set up bootsize parameters for bootstrapping.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict, optional
        Dictionary with dimensions as keys and bootsize as values
        
    Returns
    -------
    dict
        Dictionary with dimensions as keys and bootsize as values
    list
        List of bootstrappable dimensions
    int
        Number of bootstrappable dimensions
    """
    # Default bootsize if not provided
    if bootsize is None:
        # Initialize with half of dimension size or 32, whichever is smaller
        bootsize_dict = {dim: min(32, data_shape[dim] // 2) for dim in dims}
    else:
        # Set bootsize directly from input
        bootsize_dict = bootsize
    
    # Track which dimensions are bootstrappable
    bootstrappable_dims = []
    for dim in dims:
        if bootsize_dict[dim] < data_shape[dim]:
            bootstrappable_dims.append(dim)
        else:
            print(f"Dimension {dim} has bootsize equal to or larger than dimension size. "
                 f"No bootstrapping will be done across this dimension.")
    
    print(f"Using bootsize: {bootsize_dict}")
    print(f"Bootstrappable dimensions: {bootstrappable_dims}")
    
    # Determine number of bootstrappable dimensions
    num_bootstrappable = len(bootstrappable_dims)
    
    return bootsize_dict, bootstrappable_dims, num_bootstrappable

# 3D

def setup_bootsize_3d(dims, data_shape, bootsize=None):
    """
    Set up bootsize parameters for bootstrapping.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict, optional
        Dictionary with dimensions as keys and bootsize as values
        
    Returns
    -------
    dict
        Dictionary with dimensions as keys and bootsize as values
    list
        List of bootstrappable dimensions
    int
        Number of bootstrappable dimensions
    """
    # Default bootsize if not provided
    if bootsize is None:
        # Initialize with half of dimension size or 32, whichever is smaller
        bootsize_dict = {dim: min(32, data_shape[dim] // 2) for dim in dims}
    else:
        # Set bootsize directly from input
        bootsize_dict = bootsize
    
    # Track which dimensions are bootstrappable
    bootstrappable_dims = []
    for dim in dims:
        if bootsize_dict[dim] < data_shape[dim]:
            bootstrappable_dims.append(dim)
        else:
            print(f"Dimension {dim} has bootsize equal to or larger than dimension size. "
                 f"No bootstrapping will be done across this dimension.")
    
    print(f"Using bootsize: {bootsize_dict}")
    print(f"Bootstrappable dimensions: {bootstrappable_dims}")
    
    # Count bootstrappable dimensions
    num_bootstrappable = len(bootstrappable_dims)
    
    return bootsize_dict, bootstrappable_dims, num_bootstrappable

####################################################################################################

###############################Adapting sampling Indices############################################

# 1D

def calculate_adaptive_spacings_1d(dim, data_shape, bootsize, num_bootstrappable):
    """
    Calculate adaptive spacings based on dimension size.
    
    Parameters
    ----------
    dim : str
        Name of the dataset dimension
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimension name as key and bootsize as value
    num_bootstrappable : int
        Number of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with spacing information
    list
        List of all spacing values
    """
    # Determine spacings based on bootstrappable dimensions
    if num_bootstrappable == 0:
        # No bootstrappable dimensions - no spacings needed
        print("No bootstrappable dimensions available. Structure functions will be calculated "
              "using the full dataset without bootstrapping or spacings.")
        all_spacings = [1]  # Just a nominal value, won't be used
        spacings_info = {'spacings': all_spacings}
    else:
        # Calculate the ratio (size/bootsize)
        dim_ratio = data_shape[dim] / bootsize[dim]
        
        # Calculate maximum power of 2 that can be used for spacing
        max_power = int(math.floor(math.log2(dim_ratio)))
        
        # Create spacings as powers of 2
        spacings = [1 << i for i in range(max_power + 1)]
        
        print(f"Accessible spacings are: {spacings}")
        
        # Store results
        spacings_info = {'spacings': spacings}
        all_spacings = spacings
    
    return spacings_info, all_spacings
    
# 2D

def calculate_adaptive_spacings_2d(dims, data_shape, bootsize, bootstrappable_dims, num_bootstrappable):
    """
    Calculate adaptive spacings based on dimension sizes.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
    num_bootstrappable : int
        Number of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with spacing information
    list
        List of all spacing values
    """
    if num_bootstrappable == 0:
        # No bootstrappable dimensions - no spacings needed
        print("No bootstrappable dimensions available. Structure functions will be calculated "
             "using the full dataset without bootstrapping or spacings.")
        all_spacings = [1]  # Just a nominal value, won't be used
        spacings_info = {'shared_spacings': all_spacings}
    elif num_bootstrappable == 1:
        # For single bootstrappable dimension, calculate spacings just for that dimension
        bootstrap_dim = bootstrappable_dims[0]
        dim_ratio = data_shape[bootstrap_dim] / bootsize[bootstrap_dim]
        max_power = int(math.floor(math.log2(dim_ratio)))
        all_spacings = [1 << i for i in range(max_power + 1)]
        print(f"Single bootstrappable dimension ({bootstrap_dim}). "
             f"Using spacings: {all_spacings}")
        spacings_info = {'shared_spacings': all_spacings}
    else:
        # For two bootstrappable dimensions, use full spacing calculation
        spacings_info = _get_simplified_adaptive_spacings_2d(data_shape, bootsize)
        all_spacings = spacings_info['shared_spacings']
        print(f"Two bootstrappable dimensions. Available spacings: {all_spacings}")
    
    return spacings_info, all_spacings


def _get_simplified_adaptive_spacings_2d(data_shape, bootsize):
    """
    Calculate adaptive spacings based on dimension sizes.
    Uses shared spacings across all dimensions based on the most limiting dimension.
    
    Parameters
    ----------
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
        
    Returns
    -------
    dict
        Dictionary with 'shared_spacings' for all dimensions
    """
    result = {}
    dim_ratios = {}
    
    # Calculate the ratios (size/bootsize) for each dimension
    for dim in data_shape:
        if dim in bootsize:
            dim_ratios[dim] = data_shape[dim] / bootsize[dim]
    
    if not dim_ratios:
        return {}  # No valid dimensions
    
    # Find the minimum ratio which will limit the maximum spacing
    min_ratio = min(dim_ratios.values())
    min_ratio_dim = min(dim_ratios, key=dim_ratios.get)
    
    # Calculate maximum power of 2 that can be used for spacing
    max_power = int(math.floor(math.log2(min_ratio)))
    
    # Create spacings as powers of 2
    shared_spacings = [1 << i for i in range(max_power + 1)]
    
    # Check if any dimension has a significantly higher ratio and print warning
    max_ratio = max(dim_ratios.values())
    max_ratio_dim = max(dim_ratios, key=dim_ratios.get)
    
    if max_ratio > 2 * min_ratio:
        # Calculate the maximum power of 2 possible for the dimension with highest ratio
        max_possible_power = int(math.floor(math.log2(max_ratio)))
        
        # Calculate what bootsize would be needed to reach this power
        # We want: data_shape[min_ratio_dim] / new_bootsize ≥ 2^max_possible_power
        # Therefore: new_bootsize ≤ data_shape[min_ratio_dim] / 2^max_possible_power
        optimal_bootsize = data_shape[min_ratio_dim] / (1 << max_possible_power)
        
        # Find the nearest power of 2 smaller than or equal to optimal_bootsize
        nearest_power2_bootsize = 1 << int(math.floor(math.log2(optimal_bootsize)))
        
        print(f"WARNING: Dimension '{min_ratio_dim}' with bootsize {bootsize[min_ratio_dim]} "
             f"is limiting the range of accessible scales.")
        print(f"For optimal results, consider adjusting bootsize for '{min_ratio_dim}' to {nearest_power2_bootsize} "
             f"(power of 2) to match the scale range of dimension '{max_ratio_dim}'.")
        print(f"Current accessible spacings are limited to: {shared_spacings}")
    
    # Store results
    result['shared_spacings'] = shared_spacings
    
    return result
    
# 3D

def calculate_adaptive_spacings_3d(dims, data_shape, bootsize, bootstrappable_dims, num_bootstrappable):
    """
    Calculate adaptive spacings based on dimension sizes.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
    num_bootstrappable : int
        Number of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with spacing information
    list
        List of all spacing values
    """
    if num_bootstrappable == 0:
        # No bootstrappable dimensions - no spacings needed
        print("No bootstrappable dimensions available. Structure functions will be calculated "
             "using the full dataset without bootstrapping or spacings.")
        all_spacings = [1]  # Just a nominal value, won't be used
        spacings_info = {'shared_spacings': all_spacings}
    elif num_bootstrappable == 1:
        # For single bootstrappable dimension, calculate spacings just for that dimension
        bootstrap_dim = bootstrappable_dims[0]
        dim_ratio = data_shape[bootstrap_dim] / bootsize[bootstrap_dim]
        max_power = int(math.floor(math.log2(dim_ratio)))
        all_spacings = [1 << i for i in range(max_power + 1)]
        print(f"Single bootstrappable dimension ({bootstrap_dim}). "
             f"Using spacings: {all_spacings}")
        spacings_info = {'shared_spacings': all_spacings}
    else:  # num_bootstrappable >= 2:
        # For multiple bootstrappable dimensions, use full spacing calculation
        spacings_info = _get_simplified_adaptive_spacings_3d(data_shape, bootsize, bootstrappable_dims)
        all_spacings = spacings_info['shared_spacings']
        if num_bootstrappable == 2:
            print(f"Two bootstrappable dimensions {bootstrappable_dims}. "
                 f"Available spacings: {all_spacings}")
        else:  # num_bootstrappable == 3:
            print(f"All three dimensions {bootstrappable_dims} are bootstrappable. "
                 f"Available spacings: {all_spacings}")
    
    return spacings_info, all_spacings


def _get_simplified_adaptive_spacings_3d(data_shape, bootsize, bootstrappable_dims):
    """
    Calculate adaptive spacings based on dimension sizes.
    Uses shared spacings across all dimensions based on the most limiting dimension.
    
    Parameters
    ----------
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with 'shared_spacings' for all dimensions
    """
    result = {}
    dim_ratios = {}
    
    # Calculate the ratios (size/bootsize) for each dimension
    for dim in bootstrappable_dims:
        if dim in bootsize:
            dim_ratios[dim] = data_shape[dim] / bootsize[dim]
    
    if not dim_ratios:
        return {'shared_spacings': [1]}  # No valid dimensions
    
    # Find the minimum ratio which will limit the maximum spacing
    min_ratio = min(dim_ratios.values())
    min_ratio_dim = min(dim_ratios, key=dim_ratios.get)
    
    # Calculate maximum power of 2 that can be used for spacing
    max_power = int(math.floor(math.log2(min_ratio)))
    
    # Create spacings as powers of 2
    shared_spacings = [1 << i for i in range(max_power + 1)]
    
    # Check if any dimension has a significantly higher ratio and print warning
    max_ratio = max(dim_ratios.values())
    max_ratio_dim = max(dim_ratios, key=dim_ratios.get)
    
    if max_ratio > 2 * min_ratio:
        # Calculate the maximum power of 2 possible for the dimension with highest ratio
        max_possible_power = int(math.floor(math.log2(max_ratio)))
        
        # Calculate what bootsize would be needed to reach this power
        optimal_bootsize = data_shape[min_ratio_dim] / (1 << max_possible_power)
        
        # Find the nearest power of 2 smaller than or equal to optimal_bootsize
        nearest_power2_bootsize = 1 << int(math.floor(math.log2(optimal_bootsize)))
        
        print(f"WARNING: Dimension '{min_ratio_dim}' with bootsize {bootsize[min_ratio_dim]} "
             f"is limiting the range of accessible scales.")
        print(f"For optimal results, consider adjusting bootsize for '{min_ratio_dim}' to {nearest_power2_bootsize} "
             f"(power of 2) to match the scale range of dimension '{max_ratio_dim}'.")
        print(f"Current accessible spacings are limited to: {shared_spacings}")
    
    # Store results
    result['shared_spacings'] = shared_spacings
    
    return result
    
####################################################################################################

##################################Bootstrap Indices#################################################

# 1D

def compute_boot_indexes_1d(dim, data_shape, bootsize, all_spacings, num_bootstrappable):
    """
    Pre-compute boot indexes for all possible spacings to improve performance.
    
    Parameters
    ----------
    dim : str
        Name of the dataset dimension
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimension name as key and bootsize as value
    all_spacings : list
        List of all spacing values
    num_bootstrappable : int
        Number of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with spacing values as keys and boot indexes as values
    """
    boot_indexes = {}
    
    # If no bootstrappable dimensions, we don't need to compute indexes
    if num_bootstrappable == 0:
        return boot_indexes
    
    for sp_value in all_spacings:
        # For each spacing value, compute the boot indexes
        
        # Calculate window size based on bootsize and spacing
        window_size = bootsize[dim] * sp_value
        
        # Check if window size is valid
        if window_size > data_shape[dim]:
            print(f"Warning: Window size ({window_size}) exceeds dimension size " 
                  f"({data_shape[dim]}) for dimension {dim} with spacing {sp_value}.")
            continue
            
        # Create sliding windows with proper spacing
        num_windows = data_shape[dim] - window_size + 1
        
        # Store the indexes for this spacing
        boot_indexes[sp_value] = {
            dim: sliding_window_view(
                np.arange(data_shape[dim]), 
                (window_size,), 
                writeable=False
            )[::sp_value]
        }
    
    return boot_indexes


def get_boot_indexes_1d(dim, data_shape, bootsize, all_spacings, boot_indexes, num_bootstrappable, spacing=None):
    """
    Get boot indexes for bootstrappable dimensions.
    
    Parameters
    ----------
    dim : str
        Name of the dataset dimension
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimension name as key and bootsize as value
    all_spacings : list
        List of all spacing values
    boot_indexes : dict
        Dictionary with spacing values as keys and boot indexes as values
    num_bootstrappable : int
        Number of bootstrappable dimensions
    spacing : int or dict, optional
        Spacing value to use
        
    Returns
    -------
    dict
        Dictionary with dimension name as key and boot indexes as values
    """
    # If no bootstrappable dimensions, return empty dict - consistent with SFun2D
    if num_bootstrappable == 0:
        return {}
    
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
    
    # Return pre-computed indexes if available
    if sp_value in boot_indexes:
        return boot_indexes[sp_value]
    
    # If not pre-computed, calculate on-the-fly
    # Calculate window size based on bootsize and spacing
    window_size = bootsize[dim] * sp_value
    
    # Check if window size is valid
    if window_size > data_shape[dim]:
        print(f"Warning: Window size ({window_size}) exceeds dimension size " 
              f"({data_shape[dim]}) for dimension {dim} with spacing {sp_value}.")
        return {}
        
    # Create sliding windows
    indexes = {
        dim: sliding_window_view(
            np.arange(data_shape[dim]), 
            (window_size,), 
            writeable=False
        )[::sp_value]
    }
    
    return indexes
    
# 2D

def compute_boot_indexes_2d(dims, data_shape, bootsize, all_spacings, bootstrappable_dims):
    """
    Pre-compute boot indexes for all possible spacings to improve performance.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    all_spacings : list
        List of all spacing values
    bootstrappable_dims : list
        List of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with spacing values as keys and boot indexes as values
    """
    boot_indexes = {}
    
    for sp_value in all_spacings:
        indexes = {}
        
        for dim in bootstrappable_dims:
            # Calculate window size based on bootsize and spacing
            window_size = bootsize[dim] * sp_value
            
            # Check if window size is valid
            if window_size > data_shape[dim]:
                print(f"Warning: Window size ({window_size}) exceeds dimension size " 
                     f"({data_shape[dim]}) for dimension {dim} with spacing {sp_value}.")
                continue
                
            # Create sliding windows with proper spacing
            num_windows = data_shape[dim] - window_size + 1
            indexes[dim] = sliding_window_view(
                np.arange(data_shape[dim]), 
                (num_windows,), 
                writeable=False
            )[::sp_value]
        
        if indexes:  # Only store if we have valid indexes
            boot_indexes[sp_value] = indexes
    
    return boot_indexes


def get_boot_indexes_2d(dims, data_shape, bootsize, all_spacings, boot_indexes, bootstrappable_dims, num_bootstrappable, spacing=None):
    """
    Get boot indexes for bootstrappable dimensions.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    all_spacings : list
        List of all spacing values
    boot_indexes : dict
        Dictionary with spacing values as keys and boot indexes as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
    num_bootstrappable : int
        Number of bootstrappable dimensions
    spacing : int or dict, optional
        Spacing value to use
        
    Returns
    -------
    dict
        Dictionary with dimension names as keys and boot indexes as values
    """
    # If no bootstrappable dimensions, return empty dict
    if num_bootstrappable == 0:
        return {}
    
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
    
    # Return pre-computed indexes if available
    if sp_value in boot_indexes:
        return boot_indexes[sp_value]
    
    # If not pre-computed, calculate on-the-fly
    indexes = {}
    for dim in bootstrappable_dims:
        window_size = bootsize[dim] * sp_value
        if window_size > data_shape[dim]:
            # Skip if window is too large
            continue
        
        # Create sliding windows
        num_windows = data_shape[dim] - window_size + 1
        indexes[dim] = sliding_window_view(
            np.arange(data_shape[dim]), 
            (num_windows,), 
            writeable=False
        )[::sp_value]
    
    return indexes
    
# 3D

def compute_boot_indexes_3d(dims, data_shape, bootsize, all_spacings, bootstrappable_dims):
    """
    Pre-compute boot indexes for all possible spacings to improve performance.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    all_spacings : list
        List of all spacing values
    bootstrappable_dims : list
        List of bootstrappable dimensions
        
    Returns
    -------
    dict
        Dictionary with spacing values as keys and boot indexes as values
    """
    boot_indexes = {}
    
    for sp_value in all_spacings:
        indexes = {}
        
        for dim in bootstrappable_dims:
            # Calculate window size based on bootsize and spacing
            window_size = bootsize[dim] * sp_value
            
            # Check if window size is valid
            if window_size > data_shape[dim]:
                print(f"Warning: Window size ({window_size}) exceeds dimension size " 
                     f"({data_shape[dim]}) for dimension {dim} with spacing {sp_value}.")
                continue
                
            # Create sliding windows with proper spacing
            num_windows = data_shape[dim] - window_size + 1
            indexes[dim] = sliding_window_view(
                np.arange(data_shape[dim]), 
                (num_windows,), 
                writeable=False
            )[::sp_value]
        
        if indexes:  # Only store if we have valid indexes
            boot_indexes[sp_value] = indexes
    
    return boot_indexes


def get_boot_indexes_3d(dims, data_shape, bootsize, all_spacings, boot_indexes, bootstrappable_dims, num_bootstrappable, spacing=None):
    """
    Get boot indexes for bootstrappable dimensions.
    
    Parameters
    ----------
    dims : list
        List of dimension names
    data_shape : dict
        Dictionary with dimension sizes
    bootsize : dict
        Dictionary with dimensions as keys and bootsize as values
    all_spacings : list
        List of all spacing values
    boot_indexes : dict
        Dictionary with spacing values as keys and boot indexes as values
    bootstrappable_dims : list
        List of bootstrappable dimensions
    num_bootstrappable : int
        Number of bootstrappable dimensions
    spacing : int or dict, optional
        Spacing value to use
        
    Returns
    -------
    dict
        Dictionary with dimension names as keys and boot indexes as values
    """
    # If no bootstrappable dimensions, return empty dict
    if num_bootstrappable == 0:
        return {}
    
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
    
    # Return pre-computed indexes if available
    if sp_value in boot_indexes:
        return boot_indexes[sp_value]
    
    # If not pre-computed, calculate on-the-fly
    indexes = {}
    for dim in bootstrappable_dims:
        window_size = bootsize[dim] * sp_value
        if window_size >= data_shape[dim]:
            # Skip if window is too large
            continue
        
        # Create sliding windows
        num_windows = data_shape[dim] - window_size + 1
        indexes[dim] = sliding_window_view(
            np.arange(data_shape[dim]), 
            (num_windows,), 
            writeable=False
        )[::sp_value]
    
    return indexes
    
######################################################################################################
