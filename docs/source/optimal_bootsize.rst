Optimal Bootstrap Block Size Selection
======================================

Selecting an appropriate bootstrap block size is critical for obtaining reliable uncertainty estimates when computing structure functions. 
The block size controls the spatial extent of each resampled block during the bootstrap procedure, and choosing it incorrectly can lead to biased confidence intervals 
or underestimated uncertainties.

This guide presents two methods for determining the optimal bootstrap block size: one suited for **large datasets** where direct computation is prohibitive, 
and another for **small to medium datasets** where exhaustive testing is feasible.

.. note::
   **Application Dependence**
   
   The optimal bootstrap block size is inherently **application-dependent** and depends on factors such as:
   
   - The spatial correlation structure of your data
   - The specific structure function type being computed (longitudinal, transverse, scalar, advective)
   - The physical scales of interest in your analysis
   - The presence of NaN values or data gaps
   
   The methods presented here provide practical guidance but are **not exhaustive**. Users are encouraged to validate their chosen block size 
   against their specific scientific requirements and consider domain knowledge when making final selections.


Mathematical Background
-----------------------

Bootstrap Block Sizing Theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The block bootstrap method divides a spatial field of :math:`n` total grid points into non-overlapping blocks of size :math:`b`. 
The key assumption is that optimal block size scales with dataset size following a power law:

.. math::

   b_{\text{opt}} = n^{1/k}

where :math:`k` is a scaling exponent that depends on the data's correlation structure. Taking the natural logarithm of both sides:

.. math::

   \ln(b_{\text{opt}}) = \frac{1}{k} \ln(n)

This linear relationship between :math:`\ln(b)` and :math:`\ln(n)` forms the basis for the large dataset extrapolation method.


Error Metrics for Block Size Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For small datasets, the optimal block size is determined by comparing bootstrapped structure functions against a reference (non-bootstrapped) computation. 
Several complementary error metrics are used.

**Symmetric Mean Absolute Percentage Error (SMAPE)**

.. math::

   \text{SMAPE} = \frac{100}{N} \sum_{i=1}^{N} \frac{|S_{\text{boot},i} - S_{\text{ref},i}|}{(|S_{\text{boot},i}| + |S_{\text{ref},i}|)/2}

This metric handles cases where structure functions may pass through zero or have negative values (e.g., third-order structure functions).

**Normalized Root Mean Square Error (NRMSE)**

.. math::

   \text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N}(S_{\text{boot},i} - S_{\text{ref},i})^2}}{\max(S_{\text{ref}}) - \min(S_{\text{ref}})} \times 100

**Coefficient of Determination** (:math:`R^2`)

.. math::

   R^2 = 1 - \frac{\sum_{i=1}^{N}(S_{\text{boot},i} - S_{\text{ref},i})^2}{\sum_{i=1}^{N}(S_{\text{ref},i} - \bar{S}_{\text{ref}})^2}

Values close to 1 indicate excellent agreement between bootstrapped and reference structure functions.


Method 1: Large Dataset Extrapolation
-------------------------------------

For large datasets where testing all possible block sizes is computationally prohibitive, we employ a **subset sampling and extrapolation** approach.

Algorithm Overview
^^^^^^^^^^^^^^^^^^

1. **Subset Sampling**: Randomly extract :math:`M` subsets of varying sizes from the full dataset, maintaining the original aspect ratio.

2. **Optimal Block Size per Subset**: For each subset of size :math:`n_s`, determine the optimal block size :math:`b_s` using convergence testing with a specified error threshold.

3. **Scaling Exponent Estimation**: Compute the scaling exponent for each subset:

   .. math::

      k_s = \frac{\ln(n_s)}{\ln(b_s)}

4. **Regression Analysis**: For multiple subset sizes, perform linear regression on :math:`\ln(n)` vs :math:`\ln(b)`:

   .. math::

      \ln(b) = \alpha \ln(n) + \beta

   where :math:`\alpha = 1/k` (slope) and :math:`\beta` is the intercept.

5. **Extrapolation**: Use the fitted relationship to predict the optimal block size for the full dataset:

   .. math::

      b_{\text{opt,full}} = \exp\left(\alpha \ln(n_{\text{full}}) + \beta\right)

6. **PyTurbo_SF Compatibility**: Adjust the computed block size to satisfy PyTurbo_SF's requirement that the number of blocks be a power of 2.


Configuration Options
^^^^^^^^^^^^^^^^^^^^^

The method supports two operational modes:

- **Multi-subset mode**: Uses multiple subset sizes (e.g., 64×45 to 256×180) with fewer samples per size. Provides robust regression-based estimation.

- **Single-subset mode**: Uses a single subset size with many samples (e.g., 100 subsets of size 128×90). Uses median statistics for estimation.


Method 2: Small Dataset Direct Testing
--------------------------------------

For smaller datasets where computational cost permits, we employ **exhaustive sensitivity analysis** across multiple structure function types.

Algorithm Overview
^^^^^^^^^^^^^^^^^^

1. **Reference Computation**: Compute the "true" structure function using the largest possible block size (no bootstrap subsampling).

2. **Block Size Sweep**: Test a range of block sizes generated by successive halving:

   .. math::

      b_i = \frac{b_{\text{base}}}{2^i}, \quad i = 2, 3, \ldots, i_{\text{max}}

3. **Multi-Configuration Testing**: Evaluate each block size across multiple structure function configurations (longitudinal, transverse, scalar, advective).

4. **Error Computation**: Calculate robust error metrics comparing each bootstrapped result to the reference.

5. **Ranking and Selection**: Rank block sizes by mean error and mean rank across all configurations. Select the optimal block size using consensus methods.


Selection Methods
^^^^^^^^^^^^^^^^^

The analysis provides several methods for selecting the overall optimal block size:

- **Mean Rank Method** (recommended): Selects the block size with the lowest average rank across all configurations.

- **Minimum Error Method**: Selects the block size with the lowest mean error.

- **Consensus Method**: Identifies the block size chosen most frequently as optimal across different configurations.


Jupyter Notebook Tutorials
--------------------------

**Detailed Implementation Examples:**

- :doc:`diagnostics/find_optimal_bootsize_large_dataset` - Subset sampling and extrapolation for large 2D datasets
- :doc:`diagnostics/find_optimal_bootsize_small_dataset` - Exhaustive sensitivity analysis for small to medium datasets

.. toctree::
   :maxdepth: 1
   :hidden:

   diagnostics/find_optimal_bootsize_large_dataset
   diagnostics/find_optimal_bootsize_small_dataset


Practical Recommendations
-------------------------

Choosing Between Methods
^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------------+----------------------------------+----------------------------------+
| Dataset Characteristic    | Recommended Method               | Rationale                        |
+===========================+==================================+==================================+
| Grid points > 1 million   | Large dataset extrapolation      | Direct testing computationally   |
|                           |                                  | prohibitive                      |
+---------------------------+----------------------------------+----------------------------------+
| Grid points < 100,000     | Small dataset direct testing     | Exhaustive testing feasible,     |
|                           |                                  | more accurate results            |
+---------------------------+----------------------------------+----------------------------------+
| 100,000 - 1 million       | Either method                    | Consider computational resources |
|                           |                                  | and accuracy requirements        |
+---------------------------+----------------------------------+----------------------------------+


General Guidelines
^^^^^^^^^^^^^^^^^^

1. **Error Threshold**: A typical error threshold of 5% relative error provides a good balance between accuracy and computational efficiency.

2. **NaN Handling**: Ensure subsets have acceptable NaN fractions (typically < 30%) to avoid biased results.

3. **Aspect Ratio**: When using the large dataset method, maintain the original data's aspect ratio in subset sampling.

4. **Validation**: After selecting a block size, validate by comparing bootstrap confidence intervals against physical expectations or independent estimates.

5. **Reproducibility**: Set random seeds for consistent results across runs.


.. seealso::

   - :doc:`algorithm_details` for information on the bootstrap implementation
   - :doc:`structure_functions` for available structure function types
   - :doc:`examples` for complete analysis workflows
