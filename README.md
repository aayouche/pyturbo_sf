# PyTurbo
---
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/documentation-latest-blue)](https://github.com/aayouche/pyturbo_sf)

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


**For detailed documentation and examples, see the [PyTurbo_SF documentation](https://github.com/aayouche/PyTurbo).**

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

# Dimension Names and Coordinates
---

PyTurbo_SF processes data stored in xarray Datasets, which should contain both the variables and the coordinate information. Below is a guide for data preparation in different dimensions:

## 1D Data

For 1D data, PyTurbo_SF expects a single dimension which can be either:
- `time`: For time-series data 
- `x`, `y`, or `z`: For spatial data along a single axis

**Coordinate Types and Units:**
- Time coordinates: Can be provided as:
  - Numeric values: seconds, hours, days, or other time units
  - Datetime objects: pandas.DatetimeIndex or numpy.datetime64 objects
  - String timestamps: will be automatically converted to datetime objects by xarray
- Spatial coordinates: Typically in meters, kilometers, or non-dimensional units
- All coordinates should use consistent units within a dataset

**Structure for 1D Dataset:**
```python
import xarray as xr
import numpy as np

# Example 1: 1D spatial data along x-axis
nx = 1024
x = np.linspace(0, 100000, nx)  # Spatial coordinate in meters (100 km domain)
scalar_var = 5 + 2*np.sin(x/10000)  # Example: temperature in °C

ds_1d_spatial = xr.Dataset(
    data_vars={"temperature": ("x", scalar_var)},
    coords={"x": x},
    attrs={
        "description": "Temperature variation along a horizontal transect",
        "units_x": "meters",
        "units_temperature": "celsius"
    }
)

# Example 2: Time series with datetime coordinates
import pandas as pd
from datetime import datetime, timedelta

# Create a time series with datetime objects
start_date = datetime(2023, 1, 1, 0, 0, 0)  # January 1, 2023
n_hours = 720  # 30 days of hourly data
dates = [start_date + timedelta(hours=h) for h in range(n_hours)]
time_index = pd.DatetimeIndex(dates)  # Creates pandas DatetimeIndex

# Generate sample meteorological data
temperature = 15 + 5*np.sin(np.arange(n_hours)*2*np.pi/24) + np.random.normal(0, 1, n_hours)  # Daily cycle + noise
humidity = 70 + 10*np.sin(np.arange(n_hours)*2*np.pi/24 + np.pi) + np.random.normal(0, 3, n_hours)  # Out of phase with temp

ds_datetime = xr.Dataset(
    data_vars={
        "temperature": ("time", temperature),  # Temperature in °C
        "humidity": ("time", humidity)        # Relative humidity in %
    },
    coords={"time": time_index},
    attrs={
        "description": "Hourly meteorological data with datetime coordinates",
        "units_temperature": "celsius",
        "units_humidity": "percent",
        "sampling_frequency": "hourly"
    }
)

# Example 3: Oceanographic vertical profile
nz = 500
z = np.linspace(0, 1000, nz)  # Depth in meters (0-1000m)
temperature = 20 * np.exp(-z/200) + 5  # Temperature decreasing with depth
salinity = 35 + 0.01 * z  # Salinity increasing with depth

ds_profile = xr.Dataset(
    data_vars={
        "temperature": ("z", temperature),  # Temperature in °C
        "salinity": ("z", salinity)        # Salinity in PSU
    },
    coords={"z": z},
    attrs={
        "description": "Oceanographic vertical profile",
        "units_z": "meters",
        "units_temperature": "celsius",
        "units_salinity": "PSU",
        "vertical_resolution": f"{z[1]-z[0]} meters"
    }
)
```

## 2D Data

For 2D data, PyTurbo_SF supports the following dimension combinations:
- `(y, x)`: Common for 2D planar data (horizontal plane)
- `(z, x)`: Vertical slice in x-direction
- `(z, y)`: Vertical slice in y-direction

The order of dimensions is important as it affects how PyTurbo_SF interprets your data and calculates structure functions.

**Coordinate Types and Units:**
- For physical domains, coordinates should typically be in physical units:
  - `x`, `y`: Spatial horizontal coordinates in consistent units (e.g., meters, kilometers)
  - `z`: Vertical coordinate in consistent units (e.g., meters, kilometers)
- For non-dimensional or normalized domains:
  - Coordinates can be normalized with respect to characteristic scales (e.g., L/L₀)

**Structure for 2D Dataset:**
```python
# Example: 2D data in (y, x) plane with physical units
nx, ny = 256, 256
x = np.linspace(0, 10000, nx)  # x-coordinates in meters (0-10 km domain)
y = np.linspace(0, 10000, ny)  # y-coordinates in meters (0-10 km domain)
X, Y = np.meshgrid(x, y)       # 2D coordinate grids

# Create velocity components (e.g., in m/s)
u = 5 * np.sin(X/1000) * np.cos(Y/1000)  # u-velocity in m/s
v = -5 * np.cos(X/1000) * np.sin(Y/1000) # v-velocity in m/s
T = 20 + 2 * np.sin((X+Y)/2000)          # Temperature in °C

# For structured grid (regularly spaced coordinates)
ds_2d_structured = xr.Dataset(
    data_vars={
        "u": (["y", "x"], u),  # Note the order of dimensions 
        "v": (["y", "x"], v),  # Must match the coords definition
        "temperature": (["y", "x"], T),  # Example scalar field
    },
    coords={
        "x": x,  # 1D coordinate array (meters)
        "y": y   # 1D coordinate array (meters)
    },
    attrs={
        "description": "2D flow field in horizontal plane",
        "units_x": "meters",
        "units_y": "meters",
        "units_u": "m/s",
        "units_v": "m/s",
        "units_temperature": "celsius"
    }
)

# For unstructured grid (irregularly spaced coordinates)
# Important: For unstructured grids, x and y must be 2D arrays
# Example: Curvilinear grid or distorted mesh
# This is ESSENTIAL for calculating structure functions on irregular domains
x_irregular = X + 200*np.sin(Y/2000)  # Distorted x-coordinate
y_irregular = Y + 200*np.sin(X/2000)  # Distorted y-coordinate

ds_2d_unstructured = xr.Dataset(
    data_vars={
        "u": (["y", "x"], u),
        "v": (["y", "x"], v),
        "temperature": (["y", "x"], T),
    },
    coords={
        "x": (["y", "x"], x_irregular),  # 2D coordinate array (meters)
        "y": (["y", "x"], y_irregular),  # 2D coordinate array (meters)
    },
    attrs={
        "description": "2D flow field on curvilinear grid",
        "grid_type": "unstructured",
        "units_x": "meters",
        "units_y": "meters"
    }
)

# Example: 2D data in (z, x) plane (vertical section)
nz, nx = 100, 256
x = np.linspace(0, 100000, nx)  # Horizontal distance (100 km)
z = np.linspace(0, 10000, nz)   # Depth/height (0-10 km)
X, Z = np.meshgrid(x, z)

# Create velocity components
u = 5 * np.sin(X/10000) * np.exp(-Z/1000)   # Horizontal velocity (m/s)
w = 0.1 * np.cos(X/10000) * np.exp(-Z/2000) # Vertical velocity (m/s)

ds_zx_plane = xr.Dataset(
    data_vars={
        "u": (["z", "x"], u),  # Note the dimension order here!
        "w": (["z", "x"], w),  # Using u and w for horiz/vert velocities
    },
    coords={
        "x": x,  # Horizontal coordinate (meters)
        "z": z   # Vertical coordinate (meters)
    },
    attrs={
        "description": "Vertical slice in x-direction",
        "units_x": "meters",
        "units_z": "meters",
        "units_u": "m/s",
        "units_w": "m/s"
    }
)
```

## 3D Data

For 3D data, PyTurbo_SF expects dimensions in the order `(z, y, x)`:

**Important Note for Geophysical Flows:** Since in most geophysical fluid flows the vertical length scale is much smaller than the horizontal scale, we advise users to use a stretched coordinate system:

$z^* = \int \frac{N}{f} dz'$

where N is the buoyancy frequency and f is the Coriolis parameter. This transformation helps normalize the structure function calculations across different scales.

**Structure for 3D Dataset:**
```python
# Example: 3D data in (z, y, x) space
nx, ny, nz = 64, 64, 64
x = np.linspace(0, 2*np.pi, nx)
y = np.linspace(0, 2*np.pi, ny)
z = np.linspace(0, 2*np.pi, nz)

# For a regular grid, use meshgrid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Example velocity field (simple vortex)
u = np.sin(X) * np.cos(Y) * np.cos(Z)
v = -np.cos(X) * np.sin(Y) * np.cos(Z)
w = 0.3 * np.sin(Z)

# Create the dataset
ds_3d = xr.Dataset(
    data_vars={
        "u": (["z", "y", "x"], u),
        "v": (["z", "y", "x"], v),
        "w": (["z", "y", "x"], w),
    },
    coords={
        "x": x,
        "y": y,
        "z": z
    }
)

# For unstructured 3D grid, include full coordinate arrays
ds_3d_unstructured = xr.Dataset(
    data_vars={
        "u": (["z", "y", "x"], u),
        "v": (["z", "y", "x"], v),
        "w": (["z", "y", "x"], w),
    },
    coords={
        "x": (["z", "y", "x"], X),
        "y": (["z", "y", "x"], Y),
        "z": (["z", "y", "x"], Z),
    }
)
```

# Structure Functions and Their Mathematical Definitions
---

Structure functions quantify the statistical relationships between points separated by a given distance. PyTurbo_SF implements various structure functions for different dimensions and data types.

## 1D Structure Functions

For 1D data, PyTurbo_SF supports:

### Scalar Structure Function

For a scalar field $f(x)$, the scalar structure function of order $n$ at separation $r$ is:

$$S_n(r) = \langle |f(x + r) - f(x)|^n \rangle_x$$

where $\langle \cdot \rangle_x$ denotes averaging over all positions $x$.

### Scalar-Scalar Structure Function

For two scalar fields $f(x)$ and $g(x)$, the cross-scalar structure function of orders $n$ and $k$ is:

$$S_{n,k}(r) = \langle |f(x + r) - f(x)|^n \cdot |g(x + r) - g(x)|^k \rangle_x$$

## 2D Structure Functions

For 2D fields, PyTurbo_SF provides multiple types of structure functions:

### Longitudinal Structure Function

For velocity components $(u, v)$ in a 2D field, the longitudinal structure function measures the component of velocity difference parallel to the separation vector:

$$S_{\parallel,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \rangle_{\vec{x}}$$

where $\vec{u} = (u, v)$ and $\vec{r}$ is the separation vector.

### Transverse Structure Function

The transverse structure function measures the component of velocity difference perpendicular to the separation vector:

$$S_{\perp,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \times \frac{\vec{r}}{|\vec{r}|})^n \rangle_{\vec{x}}$$

### Default Velocity Structure Function

The default velocity structure function computes the sum of the structure functions for each velocity component's differences:

$S_{n}(r) = \langle |u(\vec{x} + \vec{r}) - u(\vec{x})|^n + |v(\vec{x} + \vec{r}) - v(\vec{x})|^n \rangle_{\vec{x}}$

where $\vec{x}$ represents the position vector and $\vec{r}$ is the separation vector.

### Scalar Structure Function (2D)

Similar to 1D but for 2D scalar field:

$$S_n(r) = \langle |f(\vec{x} + \vec{r}) - f(\vec{x})|^n \rangle_{\vec{x}}$$

### Scalar-Scalar Structure Function (2D)

For two scalar fields in 2D:

$$S_{n,k}(r) = \langle |f(\vec{x} + \vec{r}) - f(\vec{x})|^n \cdot |g(\vec{x} + \vec{r}) - g(\vec{x})|^k \rangle_{\vec{x}}$$

### Longitudinal-Transverse Structure Function

Combines longitudinal and transverse components:

$$S_{\parallel\perp,n,k}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \cdot ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \times \frac{\vec{r}}{|\vec{r}|})^k \rangle_{\vec{x}}$$

### Longitudinal-Scalar Structure Function

Combines longitudinal velocity with scalar field:

$$S_{\parallel S,n,k}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \cdot |f(\vec{x} + \vec{r}) - f(\vec{x})|^k \rangle_{\vec{x}}$$

### Transverse-Scalar Structure Function

Combines transverse velocity with scalar field:

$$S_{\perp S,n,k}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \times \frac{\vec{r}}{|\vec{r}|})^n \cdot |f(\vec{x} + \vec{r}) - f(\vec{x})|^k \rangle_{\vec{x}}$$

### Advective Structure Function

Combines velocity differences with advective velocity differences:

$S_{adv,n}(r) = \langle ( (\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot (\vec{u}_{adv}(\vec{x} + \vec{r}) - \vec{u}_{adv}(\vec{x})) )^n \rangle_{\vec{x}}$

## 3D Structure Functions

PyTurbo_SF extends the structure functions to 3D fields with velocity components $(u, v, w)$:

### Longitudinal Structure Function (3D)

$$S_{\parallel,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \rangle_{\vec{x}}$$

where $\vec{u} = (u, v, w)$ and $\vec{r}$ is the 3D separation vector.

### Transverse Structure Functions in Different Planes

PyTurbo_SF offers transverse structure functions in specific planes:

- **Transverse_ij**: In the xy-plane
- **Transverse_ik**: In the xz-plane 
- **Transverse_jk**: In the yz-plane

### Scalar and Combined Structure Functions

Similar to 2D, PyTurbo_SF offers scalar, scalar-scalar, longitudinal-scalar, and transverse-scalar structure functions in 3D.

### 3D Advective Structure Function

$$S_{adv,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot (\vec{u}_{adv}(\vec{x} + \vec{r}) - \vec{u}_{adv}(\vec{x})))^n \rangle_{\vec{x}}$$

# Quickstart
---
Once PyTurbo_SF is installed, you can perform structure function calculations on your data:

## 1D Example
```python
import numpy as np
import xarray as xr
import pyturbo_sf

# Create sample 1D dataset (time series)
n = 10000
time = np.arange(n) * 0.01  # Time in seconds
signal = np.sin(2*np.pi*0.1*time) + 0.2*np.random.randn(n)  # Noisy sine wave

# Create xarray Dataset
ds = xr.Dataset(
    data_vars={"signal": ("time", signal)},
    coords={"time": time}
)

# Define logarithmic bins
bins = {'time': np.logspace(-2, 1, 20)}

# Calculate 1D structure function
sf_result = pyturbo_sf.bin_sf_1d(
    ds=ds,
    variables_names=["signal"],
    order=2,
    bins=bins,
    fun='scalar',
    bootsize=100,
    initial_nbootstrap=50,
    max_nbootstrap=200,
    convergence_eps=0.1
)
```

## 2D Example
```python
import numpy as np
import xarray as xr
import pyturbo_sf

# Create sample 2D dataset
nx, ny = 256, 256
x = np.linspace(0, 2*np.pi, nx)
y = np.linspace(0, 2*np.pi, ny)
X, Y = np.meshgrid(x, y)

# Create velocity components
u = np.sin(X) * np.cos(Y)
v = -np.cos(X) * np.sin(Y)

# Create xarray Dataset with 2D coordinates
ds = xr.Dataset(
    data_vars={
        "u": (["y", "x"], u),
        "v": (["y", "x"], v),
    },
    coords={
        "x": x,
        "y": y
    }
)

# Define logarithmic bins
bins = {
    'x': np.logspace(-2, 0, 20),
    'y': np.logspace(-2, 0, 20)
}

# Calculate 2D structure function
sf_result = pyturbo_sf.bin_sf_2d(
    ds=ds,
    variables_names=["u", "v"],
    order=2,
    bins=bins,
    fun='longitudinal',
    bootsize=32,
    initial_nbootstrap=50,
    max_nbootstrap=200,
    convergence_eps=0.1
)

# Calculate isotropic structure function
sf_iso = pyturbo_sf.get_isotropic_sf_2d(
    ds=ds,
    variables_names=["u", "v"],
    order=2,
    bins={'r': np.logspace(-2, 0, 20)},
    fun='longitudinal'
)
```

## 3D Example
```python
import numpy as np
import xarray as xr
import pyturbo_sf

# Create sample 3D dataset
nx, ny, nz = 64, 64, 64
x = np.linspace(0, 2*np.pi, nx)
y = np.linspace(0, 2*np.pi, ny)
z = np.linspace(0, 2*np.pi, nz)

# For a regular grid, create 3D arrays
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Example velocity field (simple vortex)
u = np.sin(X) * np.cos(Y) * np.cos(Z)
v = -np.cos(X) * np.sin(Y) * np.cos(Z)
w = 0.3 * np.sin(Z)

# Create the dataset
ds = xr.Dataset(
    data_vars={
        "u": (["z", "y", "x"], u),
        "v": (["z", "y", "x"], v),
        "w": (["z", "y", "x"], w),
    },
    coords={
        "x": x,
        "y": y,
        "z": z
    }
)

# Define logarithmic bins
bins = {
    'x': np.logspace(-2, 0, 15),
    'y': np.logspace(-2, 0, 15),
    'z': np.logspace(-2, 0, 15)
}

# Calculate 3D structure function
sf_result = pyturbo_sf.bin_sf_3d(
    ds=ds,
    variables_names=["u", "v", "w"],
    order=2,
    bins=bins,
    fun='longitudinal',
    bootsize=16,
    initial_nbootstrap=30,
    max_nbootstrap=100,
    convergence_eps=0.15
)

# Calculate isotropic 3D structure function
sf_iso = pyturbo_sf.get_isotropic_sf_3d(
    ds=ds,
    variables_names=["u", "v", "w"],
    order=2,
    bins={'r': np.logspace(-2, 0, 15)},
    fun='longitudinal'
)
```

"Can I use PyTurbo_SF with my data?"
---
PyTurbo_SF is designed to work with various types of data, provided they can be organized into an xarray Dataset with proper dimensional information. It supports:

	- Turbulence simulation data
	- Fluid flow measurements
	- Meteorological and oceanographic data
	- Time series
	- Any structured or unstructured 1D, 2D, or 3D data in cartesian Grid

If you are uncertain about using PyTurbo_SF with your specific dataset, please open an issue or start a discussion in the GitHub repository.

# Performance Benchmarks
---
PyTurbo_SF is optimized for both speed and memory efficiency. Our benchmarks show computational complexity scaling as O(N log N), making it suitable for large datasets.
<p align="center">
<img src="docs/performance_benchmark.png" alt="PyTurbo Performance Benchmarks" width="400"/>
</p>

# Contributing
---
This project welcomes contributions and suggestions. Feel free to open an issue, submit a pull request, or contact the maintainers directly.

# Funding Acknowledgement
---
This software package is based upon work supported by the US Department of Energy grant DE-SC0024572. 

Any opinions, findings, and conclusions or recommendations expressed in this package are those of the authors and do not necessarily reflect the views of the US Department of Energy.

# References
---
- Pearson, B. et al., 2021: _Advective structure functions in anisotropic two-dimensional turbulence._ Journal of Fluid Mechanics.
- Lindborg, E. 2008: _Third-order structure function relations for quasi-geostrophic turbulence._ Journal of Fluid Mechanics.
- Kolmogorov, A.N. 1941: The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers. Proc. R. Soc. Lond. A.
- Frisch, U. 1995: Turbulence: The Legacy of A.N. Kolmogorov. Cambridge University Press.
- Pope, S.B. 2000: Turbulent Flows. Cambridge University Press.
