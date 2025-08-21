Algorithm Details
=================

This section explains the internal workings of PyTurbo_SF, including the adaptive bootstrap methodology, spacing strategies, and convergence mechanisms.

Overview of PyTurbo_SF Algorithm
---------------------------------

PyTurbo_SF uses an innovative **adaptive bootstrap approach** to calculate structure functions with optimal statistical accuracy. The algorithm combines:

1. **Multi-scale sampling** using power-of-2 spacings
2. **Adaptive bootstrapping** with convergence monitoring  
3. **Volume-weighted binning** for accurate statistics
4. **Parallel processing** for computational efficiency

Algorithm Workflow
------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                    INPUT DATASET                            │
   │              (xarray with coordinates)                      │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │              DATASET VALIDATION                             │
   │  • Check dimensions and coordinates                         │
   │  • Identify time vs spatial dimensions                      │
   │  • Determine bootstrappable dimensions                      │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │              SPACING CALCULATION                            │
   │  • Generate power-of-2 spacings: [1, 2, 4, 8, 16, ...]     │
   │  • Create bootstrap indices for each spacing               │
   │  • Optimize for memory and computation efficiency          │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │           INITIAL BOOTSTRAP PHASE                           │
   │  • Sample all spacings with initial bootstrap count        │
   │  • Collect preliminary statistics                          │
   │  • Assess spacing effectiveness per bin                    │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │          ADAPTIVE CONVERGENCE LOOP                          │
   │  • Group unconverged bins by characteristics               │
   │  • Allocate bootstraps based on effectiveness              │
   │  • Continue until convergence or max bootstraps            │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │            FINAL RESULT ASSEMBLY                            │
   │  • Calculate weighted means and bootstrap errors           │
   │  • Generate confidence intervals                           │
   │  • Create output xarray Dataset                            │
   └─────────────────────────────────────────────────────────────┘

Bootstrap Methodology
---------------------

Statistical Foundation
~~~~~~~~~~~~~~~~~~~~~

PyTurbo_SF uses **bootstrap resampling** to estimate uncertainties in structure function calculations. For each separation bin, the algorithm:

1. **Randomly samples** subset of data points (with replacement)
2. **Calculates** structure function for each bootstrap sample
3. **Estimates** mean and standard deviation across bootstrap samples
4. **Provides** confidence intervals based on bootstrap distribution

Mathematical Framework
~~~~~~~~~~~~~~~~~~~~~

For a structure function :math:`S_n(r)`, the bootstrap estimate is:

.. math::

   \hat{S}_n(r) = \frac{1}{B} \sum_{b=1}^{B} S_n^{(b)}(r)

where :math:`B` is the number of bootstrap samples and :math:`S_n^{(b)}(r)` is the structure function calculated from the :math:`b`-th bootstrap sample.

The bootstrap standard error is:

.. math::

   \hat{\sigma}(r) = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left(S_n^{(b)}(r) - \hat{S}_n(r)\right)^2}

Bootstrap Process Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Original Dataset: N points
   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
   │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  ...
   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                           │
                    Bootstrap Sampling
                           │
                           ▼
   Bootstrap Sample 1:     Bootstrap Sample 2:     Bootstrap Sample B:
   ┌─────┬─────┬─────┐     ┌─────┬─────┬─────┐     ┌─────┬─────┬─────┐
   │  2  │  2  │  7  │     │  1  │  5  │  3  │     │  4  │  8  │  1  │
   └─────┴─────┴─────┘     └─────┴─────┴─────┘     └─────┴─────┴─────┘
           │                       │                       │
    Calculate S₁(r)         Calculate S₂(r)        Calculate Sᵦ(r)
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   ▼
                           Final Statistics:
                           Mean: Ŝ(r) = (S₁ + S₂ + ... + Sᵦ)/B
                           Std:  σ̂(r) = std(S₁, S₂, ..., Sᵦ)

Spacing Strategy
---------------

Power-of-2 Spacing Philosophy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTurbo_SF uses **power-of-2 spacings** (1, 2, 4, 8, 16, 32, ...) for several reasons:

1. **Computational efficiency**: Powers of 2 align with memory access patterns
2. **Scale separation**: Each spacing samples different physical scales
3. **Statistical independence**: Different spacings provide complementary information
4. **Adaptive sampling**: Algorithm can focus on most effective spacings

Spacing Generation Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def generate_spacings(data_size, bootsize):
       """Generate optimal power-of-2 spacings."""
       max_spacing = data_size // (2 * bootsize)  # Ensure enough samples
       spacings = []
       
       spacing = 1
       while spacing <= max_spacing:
           spacings.append(spacing)
           spacing *= 2  # Power of 2 progression
           
       return spacings

Spacing Effectiveness Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Data Array (1D example):
   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │
   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

   Spacing = 1 (every point):
   ● ● ● ● ● ● ● ● ● ● ● ● ● ● ● ●    → High resolution, many samples

   Spacing = 2 (every 2nd point):  
   ●   ●   ●   ●   ●   ●   ●   ●      → Medium resolution, good coverage

   Spacing = 4 (every 4th point):
   ●       ●       ●       ●          → Lower resolution, broader scales

   Spacing = 8 (every 8th point):
   ●               ●                  → Large-scale features only

**Benefits of multi-spacing approach:**
   - Small spacings: Capture fine-scale structure, high statistical power
   - Large spacings: Capture large-scale structure, better scale separation
   - Combined: Optimal balance of resolution and computational efficiency

Adaptive Convergence Algorithm
------------------------------

Convergence Criteria
~~~~~~~~~~~~~~~~~~~

A bin is considered **converged** when one of the following occurs:

1. **Statistical convergence**: :math:`\sigma(r) \leq \epsilon_{conv}` (convergence threshold)
2. **Low density**: Number of points in bin :math:`< 10`
3. **Maximum bootstraps**: Bootstrap count reaches ``max_nbootstrap``
4. **NaN standard deviation**: Insufficient valid data

Convergence Process
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Convergence Loop Iteration:
   
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. IDENTIFY UNCONVERGED BINS                                │
   │    • Check σ(r) > εconv                                    │
   │    • Check bootstrap count < max_nbootstrap                │
   │    • Check point count > 10                                │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │ 2. GROUP BINS BY CHARACTERISTICS                            │
   │    • Bootstrap step size                                    │
   │    • Bin density quartile                                  │
   │    • Similar convergence behavior                          │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │ 3. ALLOCATE BOOTSTRAPS INTELLIGENTLY                       │
   │    • High-density bins → more bootstraps                   │
   │    • Effective spacings → more bootstraps                  │
   │    • Poor convergence → larger step sizes                  │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │ 4. RUN TARGETED BOOTSTRAP SIMULATIONS                      │
   │    • Focus on unconverged bins only                        │
   │    • Use optimal spacing distribution                      │
   │    • Parallel processing for efficiency                    │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │ 5. UPDATE STATISTICS AND CHECK CONVERGENCE                 │
   │    • Recalculate means and standard deviations            │
   │    • Mark newly converged bins                             │
   │    • Continue if unconverged bins remain                  │
   └─────────────────────────────────────────────────────────────┘

Spacing Effectiveness Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm tracks how effective each spacing is for each bin:

.. math::

   \text{Effectiveness}_{s,b} = \frac{\text{Points contributed by spacing } s \text{ to bin } b}{\text{Total bootstrap samples for spacing } s}

This allows the algorithm to:
- **Focus** computational resources on most effective spacings
- **Avoid** wasting time on ineffective spacing-bin combinations
- **Adapt** to different data characteristics dynamically

No Bootstrap Case
-----------------

When Bootstrap is Not Possible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, bootstrapping is not feasible:

1. **Single time slice**: Only one time step available
2. **Uniform spatial data**: No variation to bootstrap
3. **Small datasets**: Insufficient data for meaningful bootstrap samples
4. **Memory constraints**: User-specified limits prevent bootstrapping

No Bootstrap Algorithm
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   No Bootstrap Workflow:
   
   ┌─────────────────────────────────────────────────────────────┐
   │              USE FULL DATASET                               │
   │  • Calculate structure function once                       │
   │  • No statistical resampling                               │
   │  • Use all available data points                           │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │           VOLUME-WEIGHTED BINNING                           │
   │  • Weight by separation distance magnitudes                │
   │  • Account for geometric factors                           │
   │  • Normalize by bin areas/volumes                          │
   └─────────────────┬───────────────────────────────────────────┘
                     │
   ┌─────────────────▼───────────────────────────────────────────┐
   │           CALCULATE UNCERTAINTIES                           │
   │  • Use sample standard deviation                           │
   │  • Apply finite-sample corrections                         │
   │  • Set bootstrap count to 1                               │
   └─────────────────────────────────────────────────────────────┘

**Limitations of No Bootstrap:**
   - No uncertainty quantification through resampling
   - Cannot assess sampling variability
   - Less robust to outliers
   - Single estimate without confidence intervals

Memory and Performance Optimizations
------------------------------------

Efficient Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~

PyTurbo_SF implements several optimizations:

.. code-block:: python

   # Memory-efficient accumulator pattern
   bin_accumulators = {
       (bin_j, bin_i): {
           'weighted_sum': 0.0,
           'total_weight': 0.0,
           'bootstrap_samples': []
       }
   }
   
   # Only store data for bins with points
   # Automatic garbage collection between iterations
   # Vectorized operations where possible

Parallel Processing Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Parallel Bootstrap Execution:
   
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │   Worker 1  │  │   Worker 2  │  │   Worker 3  │  │   Worker N  │
   │             │  │             │  │             │  │             │
   │ Bootstrap   │  │ Bootstrap   │  │ Bootstrap   │  │ Bootstrap   │
   │ Samples     │  │ Samples     │  │ Samples     │  │ Samples     │
   │ 1-10        │  │ 11-20       │  │ 21-30       │  │ ...         │
   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
          │                │                │                │
          └────────────────┼────────────────┼────────────────┘
                           │                │
                    ┌──────▼────────────────▼──────┐
                    │     RESULT AGGREGATION       │
                    │  • Combine all samples       │
                    │  • Calculate statistics       │
                    │  • Update convergence status  │
                    └───────────────────────────────┘

Volume-Weighted Binning
----------------------

Physical Weighting Rationale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structure functions require proper weighting to account for:

1. **Geometric factors**: Different separations have different "volumes" in separation space
2. **Sampling density**: Some separations are sampled more frequently than others  
3. **Physical significance**: Larger separations carry more "weight" in 2D/3D analysis

Weighting Schemes
~~~~~~~~~~~~~~~~~

**1D (Time series):**

.. math::

   w = |\Delta t|

**2D (Spatial fields):**

.. math::

   w = |\Delta x \cdot \Delta y|

**3D (Volumetric data):**

.. math::

   w = |\Delta x \cdot \Delta y \cdot \Delta z|

**Polar coordinates (2D isotropic):**

.. math::

   w = r \cdot \Delta r \cdot \Delta \theta

Implementation Details
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Volume element calculation (2D example)
   dx = separation_x_array
   dy = separation_y_array
   weights = np.abs(dx * dy)
   weights = np.maximum(weights, 1e-10)  # Avoid zero weights
   
   # Weighted averaging in bins
   weighted_sum = np.sum(structure_function_values * weights)
   total_weight = np.sum(weights)
   bin_mean = weighted_sum / total_weight

Key Algorithm Advantages
-----------------------

Statistical Robustness
~~~~~~~~~~~~~~~~~~~~~

- **Adaptive sampling**: Focuses computation where it's most needed
- **Bootstrap errors**: Provides rigorous uncertainty estimates  
- **Convergence monitoring**: Ensures statistical reliability
- **Outlier handling**: Volume weighting reduces impact of extreme values

Computational Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~

- **Parallel processing**: Scales with available CPU cores
- **Memory optimization**: Processes large datasets within memory constraints
- **Adaptive convergence**: Avoids unnecessary computation
- **Power-of-2 spacings**: Optimizes memory access patterns

Scientific Accuracy
~~~~~~~~~~~~~~~~~~~

- **Volume weighting**: Accounts for geometric sampling effects
- **Multi-scale analysis**: Captures physics at all relevant scales
- **Proper error propagation**: Maintains statistical validity
- **Flexible function types**: Supports wide range of turbulence analysis

This adaptive bootstrap approach makes PyTurbo_SF both **statistically rigorous** and **computationally efficient**, enabling analysis of large turbulence datasets with quantified uncertainties.
