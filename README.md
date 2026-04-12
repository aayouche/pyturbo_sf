# PyTurbo_SF
---
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/fluidsf?color=blue)](https://pypi.org/project/pyturbo_sf)
[![Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://pyturbo-sf.readthedocs.io)
[![CI](https://github.com/aayouche/pyturbo_sf/actions/workflows/ci.yml/badge.svg)](https://github.com/aayouche/pyturbo_sf/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/1042380152.svg)](https://doi.org/10.5281/zenodo.19342650)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.09876/status.svg)](https://doi.org/10.21105/joss.09876)

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
# Citing
---
If you use Pyturbo_SF in your research or educational activities, we would be grateful if you credit Pyturbo_SF by name! You can cite our [JOSS publication](https://joss.theoj.org/papers/10.21105/joss.09876), the specific version of Pyturbo_SF with [Zenodo](https://doi.org/10.5281/zenodo.19342650) or use the following citation:

> Ayouche et al., (2026). PyTurbo_SF: An Adaptive Bootstrap Framework for Efficient Structure Function Analysis in Turbulent Flows. Journal of Open Source Software, 11(120), 9876, https://doi.org/10.21105/joss.09876
