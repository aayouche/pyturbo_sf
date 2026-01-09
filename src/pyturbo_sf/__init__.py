# PyTurbo_SF - Python Turbulence Unleashed: Rapid Binning Operator
# Structure function calculations for turbulent flows

"""
PyTurbo_SF
==========

A Python package for efficient structure function calculations in 1D, 2D, and 3D data.

Main Functions
--------------
bin_sf_1d : 1D structure function binning
bin_sf_2d : 2D structure function binning
bin_sf_3d : 3D structure function binning
get_isotropic_sf_2d : 2D isotropic (radial) structure function
get_isotropic_sf_3d : 3D isotropic (spherical) structure function
get_energy_flux_2d : 2D spectral energy flux via Bessel transform

Submodules
----------
one_dimensional : 1D structure function calculations
two_dimensional : 2D structure function calculations
three_dimensional : 3D structure function calculations
core : Core utilities and validation
utils : Helper functions and variable mapping
structure_functions : Low-level SF computation functions
binning_tools : Binning utilities
bootstrapping_tools : Bootstrap resampling for uncertainty quantification
isotropy_tools : Isotropic averaging for 2D/3D data
bessel_tools : Bessel transform for energy flux decomposition
"""

# Import main functions to make them available at package level
from .one_dimensional import bin_sf_1d
from .two_dimensional import bin_sf_2d, get_isotropic_sf_2d, get_energy_flux_2d
from .three_dimensional import bin_sf_3d, get_isotropic_sf_3d

# Explicitly import submodules so autodoc can find them
from . import core
from . import utils
from . import structure_functions
from . import one_dimensional
from . import two_dimensional
from . import three_dimensional
from . import binning_tools
from . import bootstrapping_tools
from . import isotropy_tools
from . import bessel_tools

__version__ = "1.0.7"

__all__ = [
    # Main functions
    'bin_sf_1d',
    'bin_sf_2d',
    'bin_sf_3d',
    'get_isotropic_sf_2d',
    'get_isotropic_sf_3d',
    'get_energy_flux_2d',
    # Submodules
    'core',
    'utils',
    'structure_functions',
    'one_dimensional',
    'two_dimensional',
    'three_dimensional',
    'binning_tools',
    'bootstrapping_tools',
    'isotropy_tools',
    'bessel_tools',
]
