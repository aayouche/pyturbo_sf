.. _structure_functions:

=========================================
PyTurbo_SF: Structure Functions Reference
=========================================

This document provides the mathematical foundations and API reference for structure function computations in PyTurbo_SF.

.. contents:: Table of Contents
   :depth: 3
   :local:

----

<<<<<<< HEAD
Mathematical Foundations
========================

Standard Structure Functions
-----------------------------

Structure functions measure statistical properties of turbulent field increments across separation distances.
=======
   S_n(r) = \langle |\phi(\vec{x} + \vec{r}) - \phi(\vec{x})|^n \rangle_{\vec{x}}

where :math:`\phi` represents the field variable, :math:`\vec{r}` is the separation vector, and :math:`\langle \cdot \rangle_{\vec{x}}` denotes spatial averaging.

1D Structure Functions
----------------------

For 1D data, PyTurbo_SF supports several types of structure functions:

Scalar Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~

For a scalar field :math:`f(x)`:

.. math::

   S_n(r) = \langle |f(x + r) - f(x)|^n \rangle_{x}

where :math:`r` is the separation distance.

**Usage**: ``fun='scalar'``

Scalar-Scalar Structure Function  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For two scalar fields :math:`f(x)` and :math:`g(x)`:

.. math::

   S_{n,k}(r) = \langle |f(x + r) - f(x)|^n \cdot |g(x + r) - g(x)|^k \rangle_{x}

**Usage**: ``fun='scalar_scalar'``

2D Structure Functions
----------------------

For 2D fields, PyTurbo_SF provides multiple types of structure functions:
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d

Longitudinal Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

<<<<<<< HEAD
The longitudinal structure function projects velocity increments onto the separation vector:

.. math::

   D_{LL}(\mathbf{r}) = \left\langle \left[ \delta u_L(\mathbf{x}, \mathbf{r}) \right]^2 \right\rangle

where the longitudinal velocity increment is:

.. math::

   \delta u_L(\mathbf{x}, \mathbf{r}) = \left[ \mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x}) \right] \cdot \hat{\mathbf{r}}
=======
For velocity components :math:`\vec{u} = (u, v)`:

.. math::

   S_{\parallel,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \rangle_{\vec{x}}

where :math:`\vec{u} = (u, v)` and :math:`\vec{r}` is the separation vector.
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d


Transverse Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transverse structure function uses the perpendicular velocity component:

.. math::

<<<<<<< HEAD
   D_{TT}(\mathbf{r}) = \left\langle \left[ \delta u_T(\mathbf{x}, \mathbf{r}) \right]^2 \right\rangle

where the transverse velocity increment is:

.. math::

   \delta u_T(\mathbf{x}, \mathbf{r}) = \left| \left[ \mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x}) \right] - \delta u_L \hat{\mathbf{r}} \right|
=======
   S_{\perp,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \times \frac{\vec{r}}{|\vec{r}|})^n \rangle_{\vec{x}}

**Usage**: ``fun='transverse'``

Default Velocity Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default velocity structure function computes the sum of the structure functions for each velocity component's differences:

.. math::

   S_{n}(r) = \langle |u(\vec{x} + \vec{r}) - u(\vec{x})|^n + |v(\vec{x} + \vec{r}) - v(\vec{x})|^n \rangle_{\vec{x}}
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d


Advective (Third-Order) Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The advective (third-order longitudinal) structure function is fundamental for energy flux estimation:

.. math::

<<<<<<< HEAD
   D_{LLL}(\mathbf{r}) = \left\langle \left[ \delta u_L(\mathbf{x}, \mathbf{r}) \right]^3 \right\rangle
=======
   S_n(r) = \langle |f(\vec{x} + \vec{r}) - f(\vec{x})|^n \rangle_{\vec{x}}
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d

.. important::

   The advective structure function is **only available in 2D** for energy flux computation via ``get_energy_flux_2d``.


Scalar Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~

For scalar fields (e.g., temperature, concentration):

.. math::

<<<<<<< HEAD
   D_{\phi\phi}(r) = \left\langle \left[ \phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x}) \right]^2 \right\rangle
=======
   S_{n,k}(r) = \langle |f(\vec{x} + \vec{r}) - f(\vec{x})|^n \cdot |g(\vec{x} + \vec{r}) - g(\vec{x})|^k \rangle_{\vec{x}}
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d


<<<<<<< HEAD
Scalar-Scalar (Mixed) Structure Function
=======
Combined Structure Functions (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Longitudinal-Transverse**: Combines longitudinal and transverse components:

.. math::

   S_{\parallel\perp,n,k}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \cdot ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \times \frac{\vec{r}}{|\vec{r}|})^k \rangle_{\vec{x}}

**Usage**: ``fun='longitudinal_transverse'``

**Longitudinal-Scalar**: Combines longitudinal velocity with scalar field:

.. math::

   S_{\parallel S,n,k}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \cdot |f(\vec{x} + \vec{r}) - f(\vec{x})|^k \rangle_{\vec{x}}

**Usage**: ``fun='longitudinal_scalar'``

**Transverse-Scalar**: Combines transverse velocity with scalar field:

.. math::

   S_{\perp S,n,k}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \times \frac{\vec{r}}{|\vec{r}|})^n \cdot |f(\vec{x} + \vec{r}) - f(\vec{x})|^k \rangle_{\vec{x}}

**Usage**: ``fun='transverse_scalar'``

Advanced Structure Functions (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advective Structure Function**: Combines velocity differences with advective velocity differences:

.. math::

   S_{adv,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot (\vec{u}_{adv}(\vec{x} + \vec{r}) - \vec{u}_{adv}(\vec{x})))^n \rangle_{\vec{x}}

**Usage**: ``fun='advective'``

**Pressure Work Structure Function**: Measures the structure function of pressure-velocity work term:

.. math::

   S_{pw,n}(r) = \langle (\nabla \cdot (\delta P \cdot \delta\vec{u}))^n \rangle_{\vec{x}}

where :math:`\delta P = P(\vec{x} + \vec{r}) - P(\vec{x})` is the pressure increment and :math:`\delta\vec{u}` is the velocity increment.

**Usage**: ``fun='pressure_work'``

3D Structure Functions
----------------------

PyTurbo_SF extends the structure functions to 3D fields with velocity components :math:`\vec{u} = (u, v, w)`.

Longitudinal Structure Function (3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   S_{\parallel,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot \frac{\vec{r}}{|\vec{r}|})^n \rangle_{\vec{x}}

where :math:`\vec{u} = (u, v, w)` and :math:`\vec{r}` is the 3D separation vector.

**Usage**: ``fun='longitudinal'``

Default Velocity Structure Function (3D)
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-correlation between two scalar fields:

.. math::

<<<<<<< HEAD
   D_{\phi\psi}(r) = \left\langle \left[ \phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x}) \right] \left[ \psi(\mathbf{x} + \mathbf{r}) - \psi(\mathbf{x}) \right] \right\rangle
=======
   S_{n}(r) = \langle |u(\vec{x} + \vec{r}) - u(\vec{x})|^n + |v(\vec{x} + \vec{r}) - v(\vec{x})|^n + |w(\vec{x} + \vec{r}) - w(\vec{x})|^n \rangle_{\vec{x}}
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d


Longitudinal-Transverse Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mixed longitudinal-transverse correlation:

.. math::

   D_{LT}(\mathbf{r}) = \left\langle \delta u_L(\mathbf{x}, \mathbf{r}) \cdot \delta u_T(\mathbf{x}, \mathbf{r}) \right\rangle


Longitudinal-Scalar Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coupling between velocity and scalar increments:

.. math::

   D_{L\phi}(\mathbf{r}) = \left\langle \delta u_L(\mathbf{x}, \mathbf{r}) \cdot \delta\phi(\mathbf{x}, \mathbf{r}) \right\rangle


<<<<<<< HEAD
----

Bessel Energy Flux Decomposition
================================
=======
* **Advective**: ``fun='advective'``

.. math::

   S_{adv,n}(r) = \langle ((\vec{u}(\vec{x} + \vec{r}) - \vec{u}(\vec{x})) \cdot (\vec{u}_{adv}(\vec{x} + \vec{r}) - \vec{u}_{adv}(\vec{x})))^n \rangle_{\vec{x}}

* **Pressure work**: ``fun='pressure_work'``

.. math::

   S_{pw,n}(r) = \langle (\nabla \cdot (\delta P \cdot \delta\vec{u}))^n \rangle_{\vec{x}}

where :math:`\delta P = P(\vec{x} + \vec{r}) - P(\vec{x})` is the pressure increment and :math:`\delta\vec{u}` is the velocity increment vector.

Function Reference Table
------------------------
>>>>>>> a204517c41958803e48d06e78c5c6b50e8b2929d

.. warning::

   Energy flux decomposition is **only available in 2D** via ``get_energy_flux_2d``.
   Only ``fun='advective'`` and ``fun='scalar_scalar'`` are supported.


Core Formula
------------

The spectral energy flux :math:`\Pi(K)` is computed from the angle-averaged advective structure function using a Bessel J₁ transform:

.. math::

   \boxed{\Pi(K) = -\frac{K}{2} \int_0^{\infty} \tilde{D}_{LLL}(r) \, J_1(Kr) \, dr}

where:

- :math:`\Pi(K)` — Spectral energy flux at wavenumber :math:`K`
- :math:`\tilde{D}_{LLL}(r)` — Angle-averaged advective structure function
- :math:`J_1(Kr)` — Bessel function of the first kind, order 1
- :math:`K` — Wavenumber magnitude


Physical Interpretation
-----------------------

==================== ====================================================
Sign of :math:`\Pi`  Physical Meaning
==================== ====================================================
:math:`\Pi(K) > 0`   **Forward cascade**: energy flows to smaller scales
:math:`\Pi(K) < 0`   **Inverse cascade**: energy flows to larger scales
==================== ====================================================

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

The conditional structure function is defined as:

.. math::

   \boxed{D(r \,|\, C) = \left\langle \left[ \delta u(\mathbf{x}, \mathbf{r}) \right]^n \, \middle| \, C(\mathbf{x}) \in [C_{\text{lo}}, C_{\text{hi}}) \right\rangle}

where:

- :math:`C(\mathbf{x})` — Conditioning variable (e.g., vorticity, temperature, strain rate)
- :math:`[C_{\text{lo}}, C_{\text{hi}})` — Conditioning bin range (half-open interval)


Indicator Function Formulation
------------------------------

Using indicator functions:

.. math::

   D(r \,|\, C) = \frac{\left\langle \left[ \delta u(\mathbf{x}, \mathbf{r}) \right]^n \cdot \mathbb{I}_C(\mathbf{x}) \right\rangle}{\left\langle \mathbb{I}_C(\mathbf{x}) \right\rangle}

where the indicator function is:

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
   with coordinates at bin centers: :math:`C_{\text{center},k} = (C_k + C_{k+1})/2`


Typical Conditioning Variables
------------------------------

================================ =====================================================
Conditioning Variable            Application
================================ =====================================================
Vorticity magnitude              Separate rotational from irrotational regions
Temperature                      Stratification effects on turbulence
Strain rate                      Strain-dominated vs. rotation-dominated regions
Scalar gradient                  Frontal vs. well-mixed regions
Turbulent kinetic energy (TKE)   High-energy vs. low-energy turbulence regions
Sea-Ice Concentration            Separate Open-Water, Marginal Ice Zone and Packed Ice
================================ =====================================================


----

Function Reference
==================

.. important::

   **1D supports only:** ``scalar`` and ``scalar_scalar``
   
   **Energy flux (Bessel) supports only:** ``advective`` and ``scalar_scalar`` (2D only)


Complete Function Type Table
----------------------------

.. list-table:: **1D and 2D Structure Functions**
   :widths: 20 8 28 8 10 10 10
   :header-rows: 1
   :class: longtable

   * - ``fun``
     - Order
     - ``variables_names``
     - ``bin_sf_1d``
     - ``bin_sf_2d``
     - ``get_isotropic_sf_2d``
     - ``get_energy_flux_2d``
   * - ``scalar``
     - 2
     - ``['θ']``
     - ✓
     - ✓
     - ✓
     - —
   * - ``scalar_scalar``
     - (1,1)
     - ``['φ', 'ψ']``
     - ✓
     - ✓
     - ✓
     - ✓
   * - ``longitudinal``
     - 2
     - ``['u', 'v']``
     - —
     - ✓
     - ✓
     - —
   * - ``transverse``
     - 2
     - ``['u', 'v']``
     - —
     - ✓
     - ✓
     - —
   * - ``advective``
     - 1
     - ``['u', 'v','adv_comp1','adv_comp2']`` or ``['u', 'v', 'w', 'adv_comp1','adv_comp2' , 'adv_comp3']``
     - —
     - ✓
     - ✓
     - ✓     
      * - ``default_vel``
     - 2
     - ``['u', 'v']`` or ``['u', 'v', 'w']``
     - —
     - ✓
     - ✓
     - ✓
   * - ``longitudinal_transverse``
     - (1,1)
     - ``['u', 'v']``
     - —
     - ✓
     - ✓
     - —
   * - ``longitudinal_scalar``
     - (1,1)
     - ``['u', 'v', 'θ']``
     - —
     - ✓
     - ✓
     - —
   * - ``transverse_scalar``
     - (1,1)
     - ``['u', 'v', 'θ']``
     - —
     - ✓
     - ✓
     - —


.. list-table:: **3D Structure Functions**
   :widths: 24 8 28 10 10
   :header-rows: 1
   :class: longtable

   * - ``fun``
     - Order
     - ``variables_names``
     - ``bin_sf_3d``
     - ``get_isotropic_sf_3d``
   * - ``scalar``
     - 2
     - ``['θ']``
     - ✓
     - ✓
   * - ``scalar_scalar``
     - (1,1)
     - ``['φ', 'ψ']``
     - ✓
     - ✓
   * - ``longitudinal``
     - 2
     - ``['u', 'v', 'w']``
     - ✓
     - ✓
   * - ``longitudinal_scalar``
     - (1,1)
     - ``['u', 'v', 'w', 'θ']``
     - ✓
     - ✓
   * - ``transverse_ij``
     - 2
     - ``['u', 'v']`` (xy-plane)
     - ✓
     - ✓
   * - ``transverse_ik``
     - 2
     - ``['u', 'w']`` (xz-plane)
     - ✓
     - ✓
   * - ``transverse_jk``
     - 2
     - ``['v', 'w']`` (yz-plane)
     - ✓
     - ✓
   * - ``longitudinal_transverse_ij``
     - (1,1)
     - ``['u', 'v']`` (xy-plane)
     - ✓
     - ✓
   * - ``longitudinal_transverse_ik``
     - (1,1)
     - ``['u', 'w']`` (xz-plane)
     - ✓
     - ✓
   * - ``longitudinal_transverse_jk``
     - (1,1)
     - ``['v', 'w']`` (yz-plane)
     - ✓
     - ✓
   * - ``transverse_ij_scalar``
     - (1,1)
     - ``['u', 'v', 'θ']`` (xy)
     - ✓
     - ✓
   * - ``transverse_ik_scalar``
     - (1,1)
     - ``['u', 'w', 'θ']`` (xz)
     - ✓
     - ✓
   * - ``transverse_jk_scalar``
     - (1,1)
     - ``['v', 'w', 'θ']`` (yz)
     - ✓
     - ✓
   * - ``pressure_work``
     - 1
     - ``['p', 'u', 'v', 'w']`` 
     - ✓
     - ✓
.. note::

   **3D Energy Flux** (``get_energy_flux_3d``) is **not yet implemented**.


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
     - Binned SF for time series (scalar functions only)
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
     - Bessel J₁ energy flux (advective/scalar_scalar only)
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
       bins,                        # {'time': time_edges} or similar
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
       order,                       # 2, 3, or (1, 1) depending on fun
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
       n_bins_theta=36,             # Angular bins for isotropy check
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
       n_bins_theta=36,             # Azimuthal bins
       n_bins_phi=18,               # Polar bins
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
       variables_names,             # ['u', 'v'] for advective, or 2 scalars
       order=3.0,                   # 3 for advective
       wavenumbers=None,            # Auto-generated if None
       r_bins=None,                 # Radial bins for angle-averaging
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
   :widths: 22 78
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
       conditioning_bins=[0.001, 0.01]  # High vorticity only
   )


Conditional SF (Multiple Bins)
------------------------------

.. code-block:: python

   import numpy as np
   
   # Creates output with 'cond_bin' dimension
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
