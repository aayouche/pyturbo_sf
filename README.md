# PyTurbo
---
[[License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[[Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[[Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://github.com/aayouche/pyturbo_sf)

<p align="center">
<img src="docs/pyturbo_logo.png" alt="PyTurbo Logo" width="400"/>
</p>

# Overview
---
PyTurbo_SF is a Python package for efficient structure function calculations in 1D, 2D, and 3D data. The package provides optimized implementations for analyzing turbulent flows and other spatially or temporally varying fields. With advanced bootstrapping techniques and adaptive binning, PyTurbo_SF can handle large datasets while maintaining statistical accuracy.

# Features
---
- Fast structure function calculations in 1D, 2D, and 3D
- Optimized memory usage for large datasets
- Advanced bootstrapping with adaptive sampling indices
- Multiple structure function types: longitudinal, transverse, scalar, and combined
- Isotropic averaging for 2D and 3D data
- Parallel processing for improved performance
- Automatic convergence detection based on a standard error threshold (in physical units)
- Comprehensive statistical analysis

**For detailed documentation and examples, see the [PyTurbo_SF documentation](https://pyturbo-sf.readthedocs.io).**

# Installation
---
The easiest method to install PyTurbo_SF is with [pip](https://pip.pypa.io/):

```console
$ pip install pyturbo_sf
```

You can also fork/clone this repository to your local machine and install it locally with pip as well:

```console
$ pip install .
```
