"""
Main test file to run all tests for the pyturbo_sf package.

This file provides a convenient entry point to run the complete test suite.
Tests are organized by module, from low-level utilities to high-level API functions.

Test modules included:
- test_utils.py: Utility functions (shift, time diff, confidence intervals, masks)
- test_core.py: Core validation and bootstrapping setup functions
- test_structure_functions.py: All structure function calculations (1D/2D/3D)
- test_binning_tools.py: Binning operations for structure function results
- test_isotropy_tools.py: Isotropization functions (2D polar, 3D spherical)
- test_bootstrapping_tools.py: Bootstrap statistics and convergence evaluation
- test_bessel_tools.py: Bessel decomposition functions (2D/3D)
- test_one_dimensional.py: 1D structure function API (bin_sf_1d)
- test_two_dimensional.py: 2D structure function API (bin_sf_2d, get_isotropic_sf_2d, get_bessel_sf_2d)
- test_three_dimensional.py: 3D structure function API (bin_sf_3d, get_isotropic_sf_3d, get_bessel_sf_3d)

Usage:
    python test_main.py                    # Run all tests
    python test_main.py -v                 # Run with verbose output
    python test_main.py -k "test_name"     # Run specific test by name
    python test_main.py --cov=pyturbo_sf   # Run with coverage report
"""

import pytest
import sys

# List of all test modules in order of dependency
TEST_MODULES = [
    # Low-level utilities and core functions
    "test_utils.py",
    "test_core.py",
    
    # Structure function calculations
    "test_structure_functions.py",
    
    # Binning and processing tools
    "test_binning_tools.py",
    "test_isotropy_tools.py",
    "test_bootstrapping_tools.py",
    "test_bessel_tools.py",
    
    # High-level API functions
    "test_one_dimensional.py",
    "test_two_dimensional.py",
    "test_three_dimensional.py",
]


def run_all_tests(verbose=True, extra_args=None):
    """
    Run all tests in the test suite.
    
    Parameters
    ----------
    verbose : bool
        Whether to run with verbose output. Default is True.
    extra_args : list, optional
        Additional arguments to pass to pytest.
        
    Returns
    -------
    int
        Exit code from pytest (0 for success, non-zero for failures).
    """
    args = []
    
    if verbose:
        args.append("-v")
    
    # Add extra arguments if provided
    if extra_args:
        args.extend(extra_args)
    
    # Add all test modules
    args.extend(TEST_MODULES)
    
    return pytest.main(args)


def run_quick_tests():
    """
    Run a quick subset of tests for rapid feedback during development.
    
    This runs only the core and utility tests, skipping the slower
    integration tests that involve actual structure function calculations.
    """
    quick_modules = [
        "test_utils.py",
        "test_core.py",
    ]
    
    return pytest.main(["-v", "--tb=short"] + quick_modules)


def run_api_tests():
    """
    Run only the high-level API tests (1D, 2D, 3D functions).
    
    This is useful for testing the main user-facing functions.
    """
    api_modules = [
        "test_one_dimensional.py",
        "test_two_dimensional.py",
        "test_three_dimensional.py",
    ]
    
    return pytest.main(["-v"] + api_modules)


def run_internal_tests():
    """
    Run only the internal module tests (binning, isotropy, bootstrap, bessel).
    
    This is useful for testing the internal processing functions.
    """
    internal_modules = [
        "test_structure_functions.py",
        "test_binning_tools.py",
        "test_isotropy_tools.py",
        "test_bootstrapping_tools.py",
        "test_bessel_tools.py",
    ]
    
    return pytest.main(["-v"] + internal_modules)


if __name__ == "__main__":
    # Check for special run modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            sys.exit(run_quick_tests())
        elif sys.argv[1] == "--api":
            sys.exit(run_api_tests())
        elif sys.argv[1] == "--internal":
            sys.exit(run_internal_tests())
        else:
            # Pass all arguments to pytest
            sys.exit(run_all_tests(extra_args=sys.argv[1:]))
    else:
        # Run all tests with verbose output
        sys.exit(run_all_tests())
