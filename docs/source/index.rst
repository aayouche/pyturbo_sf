PyTurbo_SF Documentation
========================

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://readthedocs.org/projects/pyturbo-sf/badge/?version=latest
   :target: https://pyturbo-sf.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

**PyTurbo_SF** is a Python package for efficient structure function calculations in 1D, 2D, and 3D data. 
The package provides optimized implementations for analyzing turbulent flows and other spatially or temporally varying fields. 
With advanced bootstrapping techniques and adaptive binning, PyTurbo_SF can handle large datasets while maintaining statistical accuracy.

Features
--------

* Fast structure function calculations in 1D, 2D, and 3D
* Optimized memory usage for large datasets  
* Advanced bootstrapping with adaptive sampling indices
* Multiple structure function types: longitudinal, transverse, scalar, and combined
* Isotropic averaging for 2D and 3D data
* Parallel processing for improved performance
* Automatic convergence detection based on a standard error threshold
* Comprehensive statistical analysis

Quick Start
-----------

.. code-block:: python

   import numpy as np
   import xarray as xr
   import pyturbo_sf

   # Create sample data
   n = 1000
   time = np.arange(n) * 0.01
   signal = np.sin(2*np.pi*0.1*time) + 0.2*np.random.randn(n)
   
   ds = xr.Dataset(
       data_vars={"signal": ("time", signal)},
       coords={"time": time}
   )

   # Calculate structure function
   bins = {'time': np.logspace(-2, 1, 20)}
   sf_result = pyturbo_sf.bin_sf_1d(
       ds=ds,
       variables_names=["signal"],
       order=2,
       bins=bins,
       fun='scalar'
   )

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   data_preparation
   structure_functions
   algorithm_details
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

.. toctree::
   :maxdepth: 2
   :caption: Additional Information

   performance
   contributing
   references

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
