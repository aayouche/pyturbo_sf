Quick Start Guide
=================

This guide will get you up and running with PyTurbo_SF in just a few minutes. We'll walk through the basic workflow and show you how to calculate structure functions for different types of data.

Basic Workflow
--------------

The typical PyTurbo_SF workflow consists of four main steps:

1. **Prepare your data** as an xarray Dataset
2. **Define separation bins** for structure function calculation
3. **Choose structure function type** and parameters
4. **Calculate and analyze** the results

Let's walk through each step with examples.

Your First Structure Function
-----------------------------

1D Time Series Example
~~~~~~~~~~~~~~~~~~~~~~

Let's start with a simple 1D time series:

.. code-block:: python

   import numpy as np
   import xarray as xr
   import pyturbo_sf
   import matplotlib.pyplot as plt

   # Step 1: Create sample data
   n = 5000
   dt = 0.01  # time step in seconds
   time = np.arange(n) * dt
   
   # Create a signal with multiple scales
   signal = (2.0 * np.sin(2*np.pi*0.1*time) +      # Low frequency
             1.0 * np.sin(2*np.pi*1.0*time) +      # Medium frequency  
             0.5 * np.sin(2*np.pi*5.0*time) +      # High frequency
             0.3 * np.random.randn(n))              # Noise

   # Create xarray Dataset
   ds = xr.Dataset(
       data_vars={"velocity": ("time", signal)},
       coords={"time": time},
       attrs={"description": "Example velocity time series"}
   )

   # Step 2: Define logarithmic bins
   bins = {'time': np.logspace(-2, 1, 25)}  # From 0.01s to 10s

   # Step 3: Calculate 2nd-order structure function
   sf_result = pyturbo_sf.bin_sf_1d(
       ds=ds,
       variables_names=["velocity"],
       order=2,
       bins=bins,
       fun='scalar',
       bootsize=50,                    # Bootstrap sample size
       initial_nbootstrap=30,          # Initial bootstrap iterations
       max_nbootstrap=100,             # Maximum bootstrap iterations
       convergence_eps=0.05,           # Convergence threshold
       backend='loky'                  # Parallel backend
   )

   # Step 4: Plot results
   plt.figure(figsize=(10, 6))
   
   r = sf_result['sf_mean'].coords['time']
   sf_mean = sf_result['sf_mean'].values
   sf_std = sf_result['sf_std'].values
   
   plt.loglog(r, sf_mean, 'b-', linewidth=2, label='Structure Function')
   plt.fill_between(r, sf_mean - sf_std, sf_mean + sf_std, 
                    alpha=0.3, color='blue', label='±1σ')
   
   # Add theoretical scaling
   plt.loglog(r, 0.1 * r**(2/3), 'r--', label='r^(2/3) scaling')
   
   plt.xlabel('Separation (s)')
   plt.ylabel('Structure Function')
   plt.title('2nd-order Structure Function')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

   print(f"Calculation completed in {sf_result.attrs['wall_time']:.2f} seconds")

2D Spatial Data Example
~~~~~~~~~~~~~~~~~~~~~~~

Now let's try a 2D example with velocity data:

.. code-block:: python

   # Step 1: Create 2D velocity field
   nx, ny = 128, 128
   Lx, Ly = 2*np.pi, 2*np.pi
   
   x = np.linspace(0, Lx, nx)
   y = np.linspace(0, Ly, ny)
   X, Y = np.meshgrid(x, y)

   # Create a simple vortex field
   u = -np.sin(Y) + 0.1 * np.random.randn(ny, nx)
   v = np.sin(X) + 0.1 * np.random.randn(ny, nx)

   # Create xarray Dataset
   ds_2d = xr.Dataset(
       data_vars={
           "u": (["y", "x"], u),
           "v": (["y", "x"], v),
       },
       coords={
           "x": (["y", "x"], X),
           "y": (["y", "x"], Y)
       },
       attrs={"description": "2D velocity field"}
   )

   # Step 2: Define 2D bins
   bins_2d = {
       'x': np.logspace(-1, 0, 20),
       'y': np.logspace(-1, 0, 20)
   }

   # Step 3: Calculate longitudinal structure function
   sf_2d = pyturbo_sf.bin_sf_2d(
       ds=ds_2d,
       variables_names=["u", "v"],
       order=2,
       bins=bins_2d,
       fun='longitudinal',
       bootsize={'x': 16, 'y': 16},
       initial_nbootstrap=20,
       max_nbootstrap=50,
       convergence_eps=0.1,
       backend='threading'
   )

   # Alternative: Calculate isotropic structure function
   sf_iso = pyturbo_sf.get_isotropic_sf_2d(
       ds=ds_2d,
       variables_names=["u", "v"],
       order=2,
       bins={'r': np.logspace(-1, 0, 15)},
       fun='longitudinal',
       backend='threading'
   )

   # Plot isotropic results
   plt.figure(figsize=(8, 6))
   r = sf_iso['sf_mean'].coords['r']
   plt.loglog(r, sf_iso['sf_mean'], 'bo-', label='Longitudinal SF')
   plt.loglog(r, 0.5 * r**(2/3), 'r--', label='r^(2/3)')
   plt.xlabel('Separation r')
   plt.ylabel('Structure Function')
   plt.title('2D Isotropic Structure Function')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Understanding the Output
------------------------

Structure function results are returned as xarray Datasets containing:

Key Variables (2D)
~~~~~~~~~~~~~

* **sf**: Mean structure function values
* **sf_std**: Standard Error from bootstrapping
* **nbootstraps**: Number of bootstraps reached per bin
* **point_counts**: Number of point pairs in each bin
* **converged**: Convergence status for each bin (either converged due to the threshold or max nbootstraps reached)

Key Variables (2D Isotropic)
~~~~~~~~~~~~~

* **sf**: Mean structure function values
* **std**: Standard Error from bootstrapping
* **ci_upper**: Upper 95% Confidence Interval
* **ci_lower**: Lower 95% Confidence Interval
* **error_isotropy**: Isotropy Error (physical units)
* **error_homogeneity**: Homogeneity Error (physical units)
* **n_bootstrap**: Number of bootstraps reached per bin
* **point_counts**: Number of point pairs in each bin
* **converged**: Convergence status for each bin (either converged due to the threshold or max nbootstraps reached)

Key Attributes
~~~~~~~~~~~~~~

* **wall_time**: Total computation time
* **bootstrap_iterations**: Number of bootstrap samples used
* **convergence_rate**: Fraction of bins that converged

.. code-block:: python

   # Examine the results structure
   print("Dataset structure:")
   print(sf_result)
   
   print("\nAttributes:")
   for key, value in sf_result.attrs.items():
       print(f"  {key}: {value}")
   
   print("\nData variables:")
   for var in sf_result.data_vars:
       print(f"  {var}: {sf_result[var].dims}")

Parameter Selection Guide
-------------------------

Choosing Bins
~~~~~~~~~~~~~

**Logarithmic spacing** is usually best for structure functions:

.. code-block:: python

   # Good bin choices
   bins_1d = {'time': np.logspace(-2, 1, 25)}     # 25 bins from 0.01 to 10
   bins_2d = {'x': np.logspace(-1, 0, 20),        # 20 bins in each direction
              'y': np.logspace(-1, 0, 20)}
   bins_3d = {'r': np.logspace(-2, 0, 15)}        # 15 bins for isotropic

**Linear spacing** for specific ranges:

.. code-block:: python

   bins_linear = {'x': np.linspace(0.1, 2.0, 20)}

Bootstrap Parameters
~~~~~~~~~~~~~~~~~~~~

For reliable statistics, choose parameters based on your data size:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Data Size
     - bootsize
     - initial_nbootstrap  
     - max_nbootstrap
     - convergence_eps
   * - Small (< 1000 pts)
     - 10-20
     - 10
     - 50
     - 0.1-0.2
   * - Medium (1K-10K pts)
     - 20-50
     - 20
     - 100
     - 0.05-0.1
   * - Large (> 10K pts)
     - 50-100
     - 30
     - 200
     - 0.02-0.05

Backend Selection
~~~~~~~~~~~~~~~~~

Choose the appropriate parallel backend:

* **'loky'**: Best for CPU-intensive tasks 
* **'threading'**: Good for I/O-bound tasks (default)
* **'multiprocessing'**: Alternative to loky


.. code-block:: python

   # Test different backends
   import time
   
   backends = ['loky', 'threading', 'sequential']
   times = {}
   
   for backend in backends:
       start = time.time()
       result = pyturbo_sf.bin_sf_1d(
           ds=ds, variables_names=["velocity"], order=2,
           bins=bins, fun='scalar', backend=backend,
           bootsize=20, initial_nbootstrap=10, max_nbootstrap=20
       )
       times[backend] = time.time() - start
       print(f"{backend}: {times[backend]:.2f} seconds")

Common Patterns
---------------

Multiple Orders
~~~~~~~~~~~~~~~

Calculate multiple structure function orders:

.. code-block:: python

   orders = [2, 3, 4, 6]
   results = {}
   
   for n in orders:
       results[n] = pyturbo_sf.bin_sf_1d(
           ds=ds, variables_names=["velocity"], order=n,
           bins=bins, fun='scalar'
       )
   
   # Plot scaling
   plt.figure(figsize=(10, 8))
   for n in orders:
       r = results[n]['sf_mean'].coords['time']
       sf = results[n]['sf_mean'].values
       plt.loglog(r, sf, 'o-', label=f'Order {n}')
   
   plt.xlabel('Separation')
   plt.ylabel('Structure Function')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Multiple Variables
~~~~~~~~~~~~~~~~~~

Compare different variables:

.. code-block:: python

   # Create dataset with multiple variables
   ds_multi = xr.Dataset(
       data_vars={
           "velocity": ("time", signal),
           "temperature": ("time", signal + 0.5*np.random.randn(n))
       },
       coords={"time": time}
   )
   
   variables = ["velocity", "temperature"]
   results_multi = {}
   
   for var in variables:
       results_multi[var] = pyturbo_sf.bin_sf_1d(
           ds=ds_multi, variables_names=[var], order=2,
           bins=bins, fun='scalar'
       )

Best Practices
--------------

1. **Start small**: Test with small datasets and few bootstrap iterations
2. **Check convergence**: Monitor convergence flags and rates
3. **Validate results**: Compare with known theoretical predictions
4. **Save results**: Structure function calculations can be expensive

.. code-block:: python

   # Save results to NetCDF
   sf_result.to_netcdf("structure_function_results.nc")
   
   # Load later
   sf_loaded = xr.open_dataset("structure_function_results.nc")

5. **Memory management**: Use appropriate bootsize for your system
6. **Parallel efficiency**: Test different backends for your hardware

Next Steps
----------

Now that you've mastered the basics:

* Explore the detailed :doc:`examples` for specific use cases
* Learn about :doc:`data_preparation` for complex datasets  
* Understand the theory in :doc:`structure_functions`
* Check the :doc:`api_reference` for all available functions
* Review :doc:`performance` tips for large datasets

Troubleshooting
---------------

**Slow calculations**
   - Reduce bootstrap parameters for testing
   - Try different backends
   - Use smaller bootsize

**Memory errors**
   - Reduce bootsize parameters
   - Use fewer bins
   - Process data in chunks

**Poor convergence**
   - Increase max_nbootstrap
   - Adjust convergence_eps
   - Check data quality

**Unexpected results**
   - Verify data preparation
   - Check coordinate definitions
   - Compare with simple test cases
