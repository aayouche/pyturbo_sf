Data Preparation
================

PyTurbo_SF processes data stored in xarray Datasets, which should contain both the variables and the coordinate information. This section provides a comprehensive guide for preparing your data in different dimensions.

General Requirements
--------------------

* Data must be organized in xarray Datasets
* Coordinates should be provided with consistent units
* Variable names should be descriptive and consistent
* Attributes can include metadata about units and descriptions

1D Data Preparation
-------------------

For 1D data, PyTurbo_SF expects a single dimension which can be either:

* **time**: For time-series data
* **x**, **y**, or **z**: For spatial data along a single axis

Coordinate Types and Units
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Time coordinates** can be provided as:

* Numeric values: seconds
* Datetime objects: pandas.DatetimeIndex or numpy.datetime64 objects

**Spatial coordinates** are typically in meters

All coordinates should use consistent units within a dataset.

1D Examples
~~~~~~~~~~~

Spatial Data Along X-Axis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import xarray as xr
   import numpy as np

   # 1D spatial data along x-axis
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

Time Series with Datetime Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import pandas as pd
   from datetime import datetime, timedelta

   # Create a time series with datetime objects
   start_date = datetime(2023, 1, 1, 0, 0, 0)  # January 1, 2023
   n_hours = 720  # 30 days of hourly data
   dates = [start_date + timedelta(hours=h) for h in range(n_hours)]
   time_index = pd.DatetimeIndex(dates)

   # Generate sample meteorological data
   temperature = 15 + 5*np.sin(np.arange(n_hours)*2*np.pi/24) + np.random.normal(0, 1, n_hours)
   humidity = 70 + 10*np.sin(np.arange(n_hours)*2*np.pi/24 + np.pi) + np.random.normal(0, 3, n_hours)

   ds_datetime = xr.Dataset(
       data_vars={
           "temperature": ("time", temperature),  # Temperature in °C
           "humidity": ("time", humidity)  # Relative humidity in %
       },
       coords={"time": time_index},
       attrs={
           "description": "Hourly meteorological data with datetime coordinates",
           "units_temperature": "celsius",
           "units_humidity": "percent",
           "sampling_frequency": "hourly"
       }
   )

Oceanographic Vertical Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   nz = 500
   z = np.linspace(0, 1000, nz)  # Depth in meters (0-1000m)
   temperature = 20 * np.exp(-z/200) + 5  # Temperature decreasing with depth
   salinity = 35 + 0.01 * z  # Salinity increasing with depth

   ds_profile = xr.Dataset(
       data_vars={
           "temperature": ("z", temperature),  # Temperature in °C
           "salinity": ("z", salinity)  # Salinity in PSU
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

2D Data Preparation
-------------------

For 2D data, PyTurbo_SF supports the following dimension combinations:

* **(y, x)**: Common for 2D planar data (horizontal plane)
* **(z, x)**: Vertical slice in x-direction  
* **(z, y)**: Vertical slice in y-direction
* **(space, time)**: space can be x, y or z

.. warning::
   The order of dimensions is important as it affects how PyTurbo_SF interprets your data and calculates structure functions.

Coordinate Types and Units
~~~~~~~~~~~~~~~~~~~~~~~~~~

For physical domains, coordinates should typically be in physical units:

* **x**, **y**: Spatial horizontal coordinates in consistent units (e.g., meters, kilometers)
* **z**: Vertical (stretched) coordinate in consistent units (e.g., meters, kilometers)

2D Examples
~~~~~~~~~~~

2D Data in (y, x) Plane
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 2D data in (y, x) plane with physical units
   nx, ny = 256, 256
   x = np.linspace(0, 10000, nx)  # x-coordinates in meters (0-10 km domain)
   y = np.linspace(0, 10000, ny)  # y-coordinates in meters (0-10 km domain)
   X, Y = np.meshgrid(x, y)  # 2D coordinate grids

   # Create velocity components (e.g., in m/s)
   u = 5 * np.sin(X/1000) * np.cos(Y/1000)  # u-velocity in m/s
   v = -5 * np.cos(X/1000) * np.sin(Y/1000)  # v-velocity in m/s
   T = 20 + 2 * np.sin((X+Y)/2000)  # Temperature in °C

   # For structured grid (regularly spaced coordinates)
   ds_2d_structured = xr.Dataset(
       data_vars={
           "u": (["y", "x"], u),  # Note the order of dimensions
           "v": (["y", "x"], v),  # Must match the coords definition
           "temperature": (["y", "x"], T),  # Example scalar field
       },
       coords={
           "x": (["y", "x"], X),
           "y": (["y", "x"], Y)
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

2D Data in (z, x) Plane (Vertical Section)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   nz, nx = 100, 256
   x = np.linspace(0, 100000, nx)  # Horizontal distance (100 km)
   z = np.linspace(0, 10000, nz)  # Depth/height (0-10 km)
   X, Z = np.meshgrid(x, z)

   # Create velocity components
   u = 5 * np.sin(X/10000) * np.exp(-Z/1000)  # Horizontal velocity (m/s)
   w = 0.1 * np.cos(X/10000) * np.exp(-Z/2000)  # Vertical velocity (m/s)

   ds_zx_plane = xr.Dataset(
       data_vars={
           "u": (["z", "x"], u),  # Note the dimension order here!
           "w": (["z", "x"], w),  # Using u and w for horiz/vert velocities
       },
       coords={
           "x": (["z", "x"], X),  # Horizontal coordinate (meters)
           "z": (["z", "x"], Z)  # Vertical coordinate (meters)
       },
       attrs={
           "description": "Vertical slice in x-direction",
           "units_x": "meters",
           "units_z": "meters",
           "units_u": "m/s",
           "units_w": "m/s"
       }
   )

3D Data Preparation
-------------------

For 3D data, PyTurbo_SF expects dimensions in the order:

* **(z, y, x)**
* **(time, y, x)** 
* **(time, y, z)**
* **(time, z, x)**

Stretched Coordinates for Geophysical Flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::
   For geophysical flows where the vertical length scale is much smaller than the horizontal scale, we recommend using a stretched coordinate system:

   .. math::

      z_{stretched} = z \cdot \frac{N}{f}

   where :math:`N` is the buoyancy frequency and :math:`f` is the Coriolis parameter. This transformation helps normalize the structure function calculations across different scales.

3D Examples
~~~~~~~~~~~

3D Data in (z, y, x) Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 3D data in (z, y, x) space
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

   # For (un)structured 3D grid, include full coordinate arrays
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

Best Practices
--------------

1. **Consistent Units**: Always use consistent units within a dataset
2. **Descriptive Attributes**: Include metadata about units and descriptions in the attrs dictionary
3. **Coordinate Order**: Pay attention to dimension order - it affects structure function calculations
4. **Physical Scales**: For geophysical data, consider using stretched coordinates to normalize different scales
5. **Data Validation**: Check your data structure before analysis:

.. code-block:: python

   # Validate your dataset
   print(ds.dims)  # Check dimensions
   print(ds.coords)  # Check coordinates
   print(ds.attrs)  # Check attributes
   print(ds.data_vars)  # Check data variables

Common Issues and Solutions
---------------------------

**Issue**: Structure functions give unexpected results
   **Solution**: Check dimension order and coordinate definitions

**Issue**: Memory errors with large datasets  
   **Solution**: Use appropriate bootsize parameters and consider data chunking

**Issue**: Inconsistent units
   **Solution**: Ensure all coordinates and variables use the same unit system

**Issue**: Missing coordinate information
   **Solution**: Always provide coordinate arrays, even for regular grids
