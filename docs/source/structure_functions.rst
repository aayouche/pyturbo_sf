.. _structure_functions:

=========================================
PyTurbo_SF: Structure Functions Reference
=========================================

This document provides the mathematical foundations and API reference for structure function computations in PyTurbo_SF.

.. contents:: Table of Contents
   :depth: 3
   :local:

----

Mathematical Foundations
========================

Structure functions measure statistical properties of turbulent field increments across separation distances. The general form is:

.. math::

   S_n(r) = \langle |\phi(\vec{x} + \vec{r}) - \phi(\vec{x})|^n \rangle_{\vec{x}}

where :math:`\phi` represents the field variable, :math:`\vec{r}` is the separation vector, and :math:`\langle \cdot \rangle_{\vec{x}}` denotes spatial averaging.


1D Structure Functions
----------------------

For 1D data (time series), PyTurbo_SF supports:

**Scalar Structure Function** (``fun='scalar'``)

.. math::

   S_n(r) = \langle |f(x + r) - f(x)|^n \rangle_{x}

**Scalar-Scalar Structure Function** (``fun='scalar_scalar'``)

.. math::

   S_{n,k}(r) = \langle |f(x + r) - f(x)|^n \cdot |g(x + r) - g(x)|^k \rangle_{x}

.. important::

   **1D only supports** ``scalar`` and ``scalar_scalar`` function types.


2D Structure Functions
----------------------

For 2D velocity fields :math:`\vec{u} = (u, v)`:

**Longitudinal Structure Function** (``fun='longitudinal'``)

Projects velocity increments onto the separation vector:

.. math::

   D_{LL}(\mathbf{r}) = \left\langle \left[ \delta u_L \right]^n \right\rangle

where :math:`\delta u_L = [\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})] \cdot \hat{\mathbf{r}}`

**Transverse Structure Function** (``fun='transverse'``)

Uses the perpendicular velocity component:

.. math::

   D_{TT}(\mathbf{r}) = \left\langle \left[ \delta u_T \right]^n \right\rangle

where :math:`\delta u_T = |[\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})] \times \hat{\mathbf{r}}|`

**Advective Structure Function** (``fun='advective'``)

Third-order longitudinal structure function for energy flux:

.. math::

   D_{LLL}(\mathbf{r}) = \left\langle \left[ \delta u_L \right]^3 \right\rangle

**Scalar Structure Function** (``fun='scalar'``)

.. math::

   D_{\phi\phi}(r) = \left\langle \left[ \phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x}) \right]^n \right\rangle

**Scalar-Scalar Structure Function** (``fun='scalar_scalar'``)

.. math::

   D_{\phi\psi}(r) = \left\langle \delta\phi^n \cdot \delta\psi^k \right\rangle

**Longitudinal-Transverse** (``fun='longitudinal_transverse'``)

.. math::

   D_{LT}(\mathbf{r}) = \left\langle \delta u_L^n \cdot \delta u_T^k \right\rangle

**Longitudinal-Scalar** (``fun='longitudinal_scalar'``)

.. math::

   D_{L\phi}(\mathbf{r}) = \left\langle \delta u_L^n \cdot \delta\phi^k \right\rangle

**Transverse-Scalar** (``fun='transverse_scalar'``)

.. math::

   D_{T\phi}(\mathbf{r}) = \left\langle \delta u_T^n \cdot \delta\phi^k \right\rangle


3D Structure Functions
----------------------

For 3D velocity fields :math:`\vec{u} = (u, v, w)`:

**Longitudinal** (``fun='longitudinal'``)

Requires all three velocity components ``['u', 'v', 'w']``.

**Transverse** (plane-specific)

- ``fun='transverse_ij'`` — xy-plane, requires ``['u', 'v']``
- ``fun='transverse_ik'`` — xz-plane, requires ``['u', 'w']``
- ``fun='transverse_jk'`` — yz-plane, requires ``['v', 'w']``

**Longitudinal-Transverse** (plane-specific)

- ``fun='longitudinal_transverse_ij'``
- ``fun='longitudinal_transverse_ik'``
- ``fun='longitudinal_transverse_jk'``

**Transverse-Scalar** (plane-specific)

- ``fun='transverse_ij_scalar'``
- ``fun='transverse_ik_scalar'``
- ``fun='transverse_jk_scalar'``

**Scalar types**: ``scalar``, ``scalar_scalar``, ``longitudinal_scalar``


----

Bessel Energy Flux Decomposition
================================

.. warning::

   Energy flux decomposition is **only available in 2D** via ``get_energy_flux_2d``.
   
   Only ``fun='advective'`` and ``fun='scalar_scalar'`` are supported.


Core Formula
------------

The spectral energy flux :math:`\Pi(K)` is computed from the angle-averaged advective structure function using a Bessel :math:`J_1` transform:

.. math::

   \boxed{\Pi(K) = -\frac{K}{2} \int_0^{\infty} \tilde{D}_{LLL}(r) \, J_1(Kr) \, dr}

where:

- :math:`\Pi(K)` — Spectral energy flux at wavenumber :math:`K`
- :math:`\tilde{D}_{LLL}(r)` — Angle-averaged advective structure function
- :math:`J_1(Kr)` — Bessel function of the first kind, order 1
- :math:`K` — Wavenumber magnitude


Physical Interpretation
-----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Sign of :math:`\Pi`
     - Physical Meaning
   * - :math:`\Pi(K) > 0`
     - **Forward cascade**: energy flows to smaller scales
   * - :math:`\Pi(K) < 0`
     - **Inverse cascade**: energy flows to larger scales

The magnitude :math:`|\Pi(K)|` represents the rate of energy transfer across wavenumber :math:`K`.


Derivation
----------

The formula derives from the Plancherel theorem relating spectral energy flux to real-space structure functions. For homogeneous isotropic turbulence:

.. math::

   \Pi(K) = -\frac{1}{2} \frac{\partial}{\partial K} \left[ K \int_0^{\infty} \tilde{D}_{LLL}(r) \, J_0(Kr) \, r \, dr \right]

Using the identity :math:`\frac{d}{dz}[z J_0(z)] = z J_1(z)` and integration by parts yields the working formula.


Numerical Implementation
------------------------

The integral is discretized over radial bins:

.. math::

   \Pi(K) \approx -\frac{K}{2} \sum_{i=1}^{N_r} \tilde{D}_{LLL}(r_i) \, J_1(Kr_i) \, \Delta r_i


----

Conditional Structure Functions
===============================

Conditional structure functions compute statistics conditioned on a secondary field, enabling scale-dependent analysis of turbulence-scalar interactions.


Mathematical Definition
-----------------------

.. math::

   \boxed{D(r \,|\, C) = \left\langle \left[ \delta u \right]^n \, \middle| \, C(\mathbf{x}) \in [C_{\text{lo}}, C_{\text{hi}}) \right\rangle}

where:

- :math:`C(\mathbf{x})` — Conditioning variable (e.g., vorticity, temperature, strain rate)
- :math:`[C_{\text{lo}}, C_{\text{hi}})` — Conditioning bin range (half-open interval)


Indicator Function Formulation
------------------------------

Using indicator functions:

.. math::

   D(r \,|\, C) = \frac{\left\langle \left[ \delta u \right]^n \cdot \mathbb{I}_C(\mathbf{x}) \right\rangle}{\left\langle \mathbb{I}_C(\mathbf{x}) \right\rangle}

where:

.. math::

   \mathbb{I}_C(\mathbf{x}) = \begin{cases} 
   1 & \text{if } C(\mathbf{x}) \in [C_{\text{lo}}, C_{\text{hi}}) \\
   0 & \text{otherwise}
   \end{cases}


Implementation
--------------

**Single Conditioning Bin**::

   conditioning_bins = [C_lo, C_hi]

**Multiple Conditioning Bins**::

   conditioning_bins = np.linspace(C_min, C_max, N+1)  # N bins
   # or
   conditioning_bins = np.logspace(log10(C_min), log10(C_max), N+1)

.. note::

   When using multiple bins, the output dataset includes an additional ``cond_bin`` dimension
   with coordinates at bin centers.


Typical Conditioning Variables
------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Conditioning Variable
     - Application
   * - Vorticity magnitude
     - Separate rotational from irrotational regions
   * - Temperature
     - Stratification effects on turbulence
   * - Strain rate
     - Strain-dominated vs. rotation-dominated regions
   * - Scalar gradient
     - Frontal vs. well-mixed regions
   * - Turbulent kinetic energy
     - High-energy vs. low-energy turbulence regions


----

Function Reference
==================

.. important::

   **1D supports only:** ``scalar`` and ``scalar_scalar``
   
   **Energy flux (Bessel) supports only:** ``advective`` and ``scalar_scalar`` (2D only)


1D and 2D Function Types
------------------------

.. list-table::
   :widths: 22 8 22 8 10 10 10
   :header-rows: 1

   * - ``fun``
     - Order
     - ``variables_names``
     - 1D
     - 2D bin
     - 2D iso
     - 2D flux
   * - ``scalar``
     - 2
     - ``['theta']``
     - Yes
     - Yes
     - Yes
     - No
   * - ``scalar_scalar``
     - (1,1)
     - ``['phi', 'psi']``
     - Yes
     - Yes
     - Yes
     - Yes
   * - ``longitudinal``
     - 2
     - ``['u', 'v']``
     - No
     - Yes
     - Yes
     - No
   * - ``transverse``
     - 2
     - ``['u', 'v']``
     - No
     - Yes
     - Yes
     - No
   * - ``advective``
     - 3
     - ``['u', 'v']``
     - No
     - Yes
     - Yes
     - Yes
   * - ``longitudinal_transverse``
     - (1,1)
     - ``['u', 'v']``
     - No
     - Yes
     - Yes
     - No
   * - ``longitudinal_scalar``
     - (1,1)
     - ``['u', 'v', 'theta']``
     - No
     - Yes
     - Yes
     - No
   * - ``transverse_scalar``
     - (1,1)
     - ``['u', 'v', 'theta']``
     - No
     - Yes
     - Yes
     - No


3D Function Types
-----------------

.. list-table::
   :widths: 28 8 28 10 10
   :header-rows: 1

   * - ``fun``
     - Order
     - ``variables_names``
     - 3D bin
     - 3D iso
   * - ``scalar``
     - 2
     - ``['theta']``
     - Yes
     - Yes
   * - ``scalar_scalar``
     - (1,1)
     - ``['phi', 'psi']``
     - Yes
     - Yes
   * - ``longitudinal``
     - 2
     - ``['u', 'v', 'w']``
     - Yes
     - Yes
   * - ``longitudinal_scalar``
     - (1,1)
     - ``['u', 'v', 'w', 'theta']``
     - Yes
     - Yes
   * - ``transverse_ij``
     - 2
     - ``['u', 'v']`` (xy-plane)
     - Yes
     - Yes
   * - ``transverse_ik``
     - 2
     - ``['u', 'w']`` (xz-plane)
     - Yes
     - Yes
   * - ``transverse_jk``
     - 2
     - ``['v', 'w']`` (yz-plane)
     - Yes
     - Yes
   * - ``longitudinal_transverse_ij``
     - (1,1)
     - ``['u', 'v']``
     - Yes
     - Yes
   * - ``longitudinal_transverse_ik``
     - (1,1)
     - ``['u', 'w']``
     - Yes
     - Yes
   * - ``longitudinal_transverse_jk``
     - (1,1)
     - ``['v', 'w']``
     - Yes
     - Yes
   * - ``transverse_ij_scalar``
     - (1,1)
     - ``['u', 'v', 'theta']``
     - Yes
     - Yes
   * - ``transverse_ik_scalar``
     - (1,1)
     - ``['u', 'w', 'theta']``
     - Yes
     - Yes
   * - ``transverse_jk_scalar``
     - (1,1)
     - ``['v', 'w', 'theta']``
     - Yes
     - Yes

.. note::

   **3D Energy Flux** (``get_energy_flux_3d``) is **not implemented**.


Methods Summary
---------------

.. list-table::
   :widths: 25 10 50 15
   :header-rows: 1

   * - Method
     - Dim
     - Description
     - Output Coord
   * - ``bin_sf_1d``
     - 1D
     - Binned SF for time series (scalar only)
     - ``r``
   * - ``bin_sf_2d``
     - 2D
     - Binned SF in Cartesian coordinates
     - ``(dx, dy)``
   * - ``bin_sf_3d``
     - 3D
     - Binned SF in Cartesian coordinates
     - ``(dx, dy, dz)``
   * - ``get_isotropic_sf_2d``
     - 2D
     - Radially-averaged SF with isotropy diagnostics
     - ``r``
   * - ``get_isotropic_sf_3d``
     - 3D
     - Spherically-averaged SF with isotropy diagnostics
     - ``r``
   * - ``get_energy_flux_2d``
     - 2D
     - Bessel J1 energy flux (advective/scalar_scalar)
     - ``K``


----

API Reference
=============

bin_sf_1d
---------

.. code-block:: python

   bin_sf_1d(
       ds,                          # xarray.Dataset
       variables_names,             # ['scalar'] or ['phi', 'psi']
       order,                       # 2 or (1, 1)
       bins,                        # {'time': time_edges}
       bootsize=None,
       fun='scalar',                # 'scalar' or 'scalar_scalar' ONLY
       initial_nbootstrap=100,
       max_nbootstrap=1000,
       step_nbootstrap=100,
       convergence_eps=0.1,
       n_jobs=-1,
       backend='threading',
       conditioning_var=None,
       conditioning_bins=None,
       confidence_interval=0.95,
       ci_method='percentile'
   )


bin_sf_2d
---------

.. code-block:: python

   bin_sf_2d(
       ds,                          # xarray.Dataset with 2D fields
       variables_names,             # See function reference table
       order,                       # 2, 3, or (1, 1)
       bins,                        # {'x': x_edges, 'y': y_edges}
       bootsize=None,
       fun='longitudinal',
       initial_nbootstrap=100,
       max_nbootstrap=1000,
       step_nbootstrap=100,
       convergence_eps=0.1,
       n_jobs=-1,
       backend='threading',
       conditioning_var=None,
       conditioning_bins=None
   )


bin_sf_3d
---------

.. code-block:: python

   bin_sf_3d(
       ds,                          # xarray.Dataset with 3D fields
       variables_names,             # See function reference table
       order,
       bins,                        # {'x': x_edges, 'y': y_edges, 'z': z_edges}
       bootsize=None,
       fun='longitudinal',
       initial_nbootstrap=100,
       max_nbootstrap=1000,
       step_nbootstrap=100,
       convergence_eps=0.1,
       n_jobs=-1,
       backend='threading',
       conditioning_var=None,
       conditioning_bins=None
   )


get_isotropic_sf_2d
-------------------

.. code-block:: python

   get_isotropic_sf_2d(
       ds,
       variables_names,
       order=2.0,
       bins=None,                   # {'r': r_edges}
       bootsize=None,
       initial_nbootstrap=100,
       max_nbootstrap=1000,
       step_nbootstrap=100,
       fun='longitudinal',
       n_bins_theta=36,
       window_size_theta=None,
       window_size_r=None,
       convergence_eps=0.1,
       n_jobs=-1,
       backend='threading',
       conditioning_var=None,
       conditioning_bins=None,
       confidence_interval=0.95,
       ci_method='percentile'
   )


get_isotropic_sf_3d
-------------------

.. code-block:: python

   get_isotropic_sf_3d(
       ds,
       variables_names,
       order=2.0,
       bins=None,                   # {'r': r_edges}
       bootsize=None,
       initial_nbootstrap=100,
       max_nbootstrap=1000,
       step_nbootstrap=100,
       fun='longitudinal',
       n_bins_theta=36,
       n_bins_phi=18,
       window_size_theta=None,
       window_size_phi=None,
       window_size_r=None,
       convergence_eps=0.1,
       n_jobs=-1,
       backend='threading',
       conditioning_var=None,
       conditioning_bins=None,
       confidence_interval=0.95,
       ci_method='percentile'
   )


get_energy_flux_2d
------------------

.. code-block:: python

   get_energy_flux_2d(
       ds,
       variables_names,             # ['u', 'v'] for advective
       order=3.0,
       wavenumbers=None,
       r_bins=None,
       bootsize=None,
       initial_nbootstrap=100,
       max_nbootstrap=1000,
       step_nbootstrap=100,
       fun='advective',             # 'advective' or 'scalar_scalar' ONLY
       n_bins_theta=36,
       n_r_bins=100,
       window_size_theta=None,
       window_size_k=None,
       convergence_eps=0.1,
       n_jobs=-1,
       backend='threading',
       conditioning_var=None,
       conditioning_bins=None,
       confidence_interval=0.95,
       ci_method='percentile'
   )


----

Common Parameters
=================

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``ds``
     - ``xarray.Dataset`` containing velocity/scalar fields
   * - ``variables_names``
     - List of variable names (see function reference tables)
   * - ``order``
     - Structure function order: ``2``, ``3``, or tuple ``(1, 1)`` for mixed
   * - ``bins``
     - Dict of bin edges: ``{'x': [...], 'y': [...]}`` or ``{'r': [...]}``
   * - ``bootsize``
     - Bootstrap block size; should be ``data_size / 2^n`` for optimal results
   * - ``fun``
     - Structure function type (see tables above)
   * - ``initial_nbootstrap``
     - Initial bootstrap iterations (default: 100)
   * - ``max_nbootstrap``
     - Maximum bootstrap iterations (default: 1000)
   * - ``convergence_eps``
     - Convergence threshold for standard error (default: 0.1)
   * - ``n_jobs``
     - Parallel jobs: ``-1`` for all cores
   * - ``backend``
     - ``'threading'``, ``'multiprocessing'``, or ``'loky'``
   * - ``conditioning_var``
     - Name of conditioning variable in dataset
   * - ``conditioning_bins``
     - ``[lo, hi]`` for single bin or array for multiple bins
   * - ``confidence_interval``
     - Confidence level (default: 0.95)
   * - ``ci_method``
     - ``'percentile'`` (bootstrap) or ``'standard'`` (normal approx.)


----

Output Variables
================

All output datasets include:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Description
   * - ``sf`` / ``energy_flux``
     - Mean structure function or energy flux
   * - ``std_error``
     - Standard error from bootstrap
   * - ``ci_upper``, ``ci_lower``
     - Confidence interval bounds
   * - ``point_counts``
     - Number of point pairs per bin
   * - ``converged``
     - Boolean convergence status
   * - ``n_bootstrap``
     - Bootstrap iterations used per bin
   * - ``error_isotropy``
     - Isotropy error (isotropic methods only)
   * - ``error_homogeneity``
     - Homogeneity error (isotropic methods only)
   * - ``mask_quality``
     - Combined quality mask


----

Usage Examples
==============

2D Longitudinal SF (Isotropic)
------------------------------

.. code-block:: python

   import numpy as np
   from pyturbo_sf import get_isotropic_sf_2d
   
   ds_sf = get_isotropic_sf_2d(
       ds, 
       variables_names=['u', 'v'],
       order=2.0,
       bins={'r': np.logspace(2, 4, 30)},
       fun='longitudinal',
       bootsize={'y': 64, 'x': 64}
   )


1D Scalar SF
------------

.. code-block:: python

   from pyturbo_sf import bin_sf_1d
   
   ds_sf = bin_sf_1d(
       ds,
       variables_names=['temperature'],
       order=2,
       bins={'time': np.linspace(0, 3600, 50)},
       fun='scalar'
   )


Energy Flux (2D)
----------------

.. code-block:: python

   from pyturbo_sf import get_energy_flux_2d
   
   ds_flux = get_energy_flux_2d(
       ds,
       variables_names=['u', 'v'],
       order=3.0,
       fun='advective',
       bootsize={'y': 64, 'x': 64}
   )


Conditional SF (Single Bin)
---------------------------

.. code-block:: python

   ds_sf = get_isotropic_sf_2d(
       ds,
       variables_names=['u', 'v'],
       fun='longitudinal',
       conditioning_var='vorticity',
       conditioning_bins=[0.001, 0.01]
   )


Conditional SF (Multiple Bins)
------------------------------

.. code-block:: python

   import numpy as np
   
   ds_sf = get_isotropic_sf_2d(
       ds,
       variables_names=['u', 'v'],
       fun='longitudinal',
       conditioning_var='vorticity',
       conditioning_bins=np.logspace(-4, -1, 6)  # 5 bins
   )
   
   # Access specific conditioning bin
   ds_sf.sel(cond_bin=0.001, method='nearest')


3D Spherically-Averaged SF
--------------------------

.. code-block:: python

   from pyturbo_sf import get_isotropic_sf_3d
   
   ds_sf = get_isotropic_sf_3d(
       ds,
       variables_names=['u', 'v', 'w'],
       order=2.0,
       bins={'r': np.logspace(0, 3, 25)},
       fun='longitudinal',
       bootsize={'z': 32, 'y': 32, 'x': 32}
   )


----

References
==========

1. Kolmogorov, A. N. (1941). The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers. *Dokl. Akad. Nauk SSSR*, 30, 301-305.

2. Lindborg, E. (1999). Can the atmospheric kinetic energy spectrum be explained by two-dimensional turbulence? *J. Fluid Mech.*, 388, 259-288.

3. Cho, J. Y. N., & Lindborg, E. (2001). Horizontal velocity structure functions in the upper troposphere and lower stratosphere. *J. Geophys. Res.*, 106(D10), 10223-10232.
