Performance Guide
================

This section provides comprehensive guidance on optimizing PyTurbo_SF performance for different types of datasets and computational environments.

Computational Complexity
-------------------------

Understanding Algorithm Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTurbo_SF implements optimized algorithms with the following complexity characteristics:

**Time Complexity:**
   - 1D: O(N log N) where N is the number of data points
   - 2D: O(N * M * log N * log M) where N,M are the data size
   - 3D: O(N * M * K log N * log M * log K) where N,M,K are the data size

**Memory Complexity:**
   - Base memory: O(N) for data storage
   - Bootstrap memory: O(B × S) where B is bootsize and S is separation bins
   - Peak memory typically 2-5× base data size

**Scaling Factors:**
   - Bootstrap iterations: Linear scaling with number of iterations
   - Separation bins: Linear scaling with number of bins
   - Structure function order: Minimal impact on computational cost

Benchmark Results
-----------------

2D DYCOMS Turbulence Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive performance tests using 2D DYCOMS (DYnamics and Chemistry of Marine Stratocumulus) data:



**Test Configuration:**
   - Bootstrap size: 16×16
   - Different Convergence thresholds
   - Structure function: 2nd-order longitudinal
   - Backend: loky (8 cores)
   - Hardware: Intel i7 6-core (12 logical cores), 16GB RAM

**Key Findings:**
   - Excellent scalability across grid sizes
   - Memory-efficient processing for large datasets
   - High convergence rates across all scales



Optimization Strategies
-----------------------

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data Size-Based Recommendations:**

.. code-block:: python

   def get_optimal_parameters(data_shape, target_accuracy='standard'):
       """Get optimal parameters based on data size."""
       total_points = np.prod(data_shape)
       
       if total_points < 10000:  # Small datasets
           return {
               'bootsize': 10 if len(data_shape) == 1 else {'x': 8, 'y': 8},
               'initial_nbootstrap': 20,
               'max_nbootstrap': 100,
               'convergence_eps': 0.1,
               'backend': 'threading'
           }
       elif total_points < 100000:  # Medium datasets
           return {
               'bootsize': 50 if len(data_shape) == 1 else {'x': 16, 'y': 16},
               'initial_nbootstrap': 30,
               'max_nbootstrap': 200,
               'convergence_eps': 0.05,
               'backend': 'loky'
           }
       else:  # Large datasets
           return {
               'bootsize': 100 if len(data_shape) == 1 else {'x': 32, 'y': 32},
               'initial_nbootstrap': 50,
               'max_nbootstrap': 300,
               'convergence_eps': 0.02,
               'backend': 'loky'
           }

**Memory-Constrained Environments:**

.. code-block:: python

   def memory_efficient_parameters(available_memory_gb, data_shape):
       """Optimize for limited memory."""
       total_points = np.prod(data_shape)
       estimated_base_memory = total_points * 8 / 1e9  # 8 bytes per float64
       
       if estimated_base_memory > available_memory_gb * 0.3:
           # Reduce bootstrap size for memory efficiency
           if len(data_shape) == 2:
               bootsize = {'x': 8, 'y': 8}
           elif len(data_shape) == 3:
               bootsize = {'x': 4, 'y': 4, 'z': 4}
           else:
               bootsize = 20
           
           return {
               'bootsize': bootsize,
               'max_nbootstrap': 100,
               'backend': 'threading'  # Lower memory overhead
           }

Parallel Processing Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**CPU Core Utilization:**

.. code-block:: python

   import multiprocessing
   import psutil

   def optimize_parallel_backend():
       """Choose optimal backend based on system characteristics."""
       n_cores = multiprocessing.cpu_count()
       available_memory = psutil.virtual_memory().available / 1e9
       
       if n_cores >= 8 and available_memory > 16:
           return 'loky'  # Best for high-core, high-memory systems
       elif n_cores >= 4:
           return 'threading'  # Good balance for medium systems
       else:
           return 'multiprocessing'  # For low-core systems

**Thread vs Process Considerations:**

- **Use loky when:** Large datasets, CPU-intensive, plenty of memory
- **Use threading when:** Moderate datasets, mixed CPU/I-O, memory constrained
- **Use multiprocessing when:** Small datasets, need process isolation


Performance Best Practices
---------------------------

**Summary Checklist:**

1. ✅ Use appropriate bootstrap parameters for your data size
2. ✅ Choose optimal backend for your system (start with 'loky')
3. ✅ Ensure contiguous memory layout
4. ✅ Monitor memory usage for large datasets
5. ✅ Use isotropic functions when appropriate
6. ✅ Start with conservative parameters and optimize iteratively
7. ✅ Profile your specific use case
8. ✅ Consider chunked processing for very large datasets
9. ✅ Validate convergence and adjust parameters accordingly
10. ✅ Use appropriate data types (float64 recommended)

**Performance-Accuracy Trade-offs:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Priority
     - Bootstrap Size
     - Max Iterations
     - Expected Accuracy
   * - Fast exploration
     - Small (8×8)
     - 50
     - 90-95%
   * - Standard analysis
     - Medium (16×16)
     - 100
     - 95-98%
   * - High precision
     - Large (32×32)
     - 200
     - 98-99.5%
   * - Research quality
     - Extra large (64×64)
     - 400
     - >99.5%

Complete Performance Benchmark Notebook
---------------------------------------

The following notebook contains all the code used to generate the benchmark results shown above. Run this notebook to reproduce all performance measurements on your own hardware.

.. toctree::

   benchmark/benchmark_analysis
