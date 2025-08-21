Structure Functions
==================

Structure functions quantify the statistical relationships between points separated by a given distance. PyTurbo_SF implements various structure functions for different dimensions and data types.

Mathematical Background
-----------------------

Structure functions are fundamental tools in turbulence analysis that measure the scaling properties of fluctuations at different spatial or temporal separations. The :math:`n`-th order structure function is generally defined as:

.. math::

   S_n(r) = \langle |\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x})|^n \rangle_{\mathbf{x}}

where :math:`\phi` represents the field variable, :math:`\mathbf{r}` is the separation vector, and :math:`\langle \cdot \rangle_{\mathbf{x}}` denotes spatial averaging.

1D Structure Functions
----------------------

For 1D data, PyTurbo_SF supports several types of structure functions:

Scalar Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~

For a scalar field :math:`\phi(x)`:

.. math::

   S_{\phi,n}(r) = \langle |\phi(x + r) - \phi(x)|^n \rangle_{x}

where :math:`r` is the separation distance.

**Usage**: ``fun='scalar'``

Scalar-Scalar Structure Function  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For two scalar fields :math:`\phi_1(x)` and :math:`\phi_2(x)`:

.. math::

   S_{\phi_1\phi_2,n}(r) = \langle |(\phi_1(x + r) - \phi_1(x))(\phi_2(x + r) - \phi_2(x))|^n \rangle_{x}

**Usage**: ``fun='scalar_scalar'``

2D Structure Functions
----------------------

For 2D fields, PyTurbo_SF provides multiple types of structure functions:

Longitudinal Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For velocity components :math:`\mathbf{u} = (u, v)`:

.. math::

   S_{L,n}(r) = \langle |(\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot \hat{\mathbf{r}}|^n \rangle_{\mathbf{x}}

where :math:`\hat{\mathbf{r}} = \mathbf{r}/|\mathbf{r}|` is the unit vector in the direction of separation.

**Usage**: ``fun='longitudinal'``

Transverse Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transverse structure function measures the component of velocity difference perpendicular to the separation vector:

.. math::

   S_{T,n}(r) = \langle |(\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot \hat{\mathbf{n}}|^n \rangle_{\mathbf{x}}

where :math:`\hat{\mathbf{n}}` is perpendicular to :math:`\hat{\mathbf{r}}`.

**Usage**: ``fun='transverse'``

Default Velocity Structure Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default velocity structure function computes the sum of the structure functions for each velocity component's differences:

.. math::

   S_{vel,n}(r) = \langle |(u(\mathbf{x} + \mathbf{r}) - u(\mathbf{x}))|^n + |(v(\mathbf{x} + \mathbf{r}) - v(\mathbf{x}))|^n \rangle_{\mathbf{x}}

where :math:`u` and :math:`v` are the velocity components.

**Usage**: ``fun='default_vel'``

Scalar Structure Function (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to 1D but for 2D scalar field:

.. math::

   S_{\phi,n}(\mathbf{r}) = \langle |\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x})|^n \rangle_{\mathbf{x}}

**Usage**: ``fun='scalar'``

Scalar-Scalar Structure Function (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For two scalar fields in 2D:

.. math::

   S_{\phi_1\phi_2,n}(\mathbf{r}) = \langle |(\phi_1(\mathbf{x} + \mathbf{r}) - \phi_1(\mathbf{x}))(\phi_2(\mathbf{x} + \mathbf{r}) - \phi_2(\mathbf{x}))|^n \rangle_{\mathbf{x}}

**Usage**: ``fun='scalar_scalar'``

Combined Structure Functions (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Longitudinal-Transverse**: Combines longitudinal and transverse components:

.. math::

   S_{LT,n}(r) = S_{L,n}(r) + S_{T,n}(r)

**Usage**: ``fun='longitudinal_transverse'``

**Longitudinal-Scalar**: Combines longitudinal velocity with scalar field:

.. math::

   S_{L\phi,n}(r) = \langle |(\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot \hat{\mathbf{r}} \cdot (\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x}))|^n \rangle_{\mathbf{x}}

**Usage**: ``fun='longitudinal_scalar'``

**Transverse-Scalar**: Combines transverse velocity with scalar field:

.. math::

   S_{T\phi,n}(r) = \langle |(\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot \hat{\mathbf{n}} \cdot (\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x}))|^n \rangle_{\mathbf{x}}

**Usage**: ``fun='transverse_scalar'``

Advanced Structure Functions (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advective Structure Function**: Combines velocity differences with advective velocity differences:

.. math::

   S_{adv,n}(r) = \langle ((\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot (\mathbf{u}_{adv}(\mathbf{x} + \mathbf{r}) - \mathbf{u}_{adv}(\mathbf{x})))^n \rangle_{\mathbf{x}}

**Usage**: ``fun='advective'``

**Pressure Work Structure Function**: Measures the structure function of pressure-velocity work term:

.. math::

   S_{pw,n}(r) = \langle ((\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot (\mathbf{p}(\mathbf{x} + \mathbf{r}) - \mathbf{p}(\mathbf{x})))^n \rangle_{\mathbf{x}}

where :math:`\mathbf{p}` represents the pressure field.

**Usage**: ``fun='pressure_work'``

3D Structure Functions
----------------------

PyTurbo_SF extends the structure functions to 3D fields with velocity components :math:`\mathbf{u} = (u, v, w)`.

Longitudinal Structure Function (3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   S_{L,n}(r) = \langle |(\mathbf{u}(\mathbf{x} + \mathbf{r}) - \mathbf{u}(\mathbf{x})) \cdot \hat{\mathbf{r}}|^n \rangle_{\mathbf{x}}

where :math:`\hat{\mathbf{r}} = \mathbf{r}/|\mathbf{r}|` is the unit vector in the direction of separation.

**Usage**: ``fun='longitudinal'``

Default Velocity Structure Function (3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sums the structure functions for each velocity component:

.. math::

   S_{vel,n}(r) = \langle |(u(\mathbf{x} + \mathbf{r}) - u(\mathbf{x}))|^n + |(v(\mathbf{x} + \mathbf{r}) - v(\mathbf{x}))|^n + |(w(\mathbf{x} + \mathbf{r}) - w(\mathbf{x}))|^n \rangle_{\mathbf{x}}

**Usage**: ``fun='default_vel'``

Transverse Structure Functions (3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTurbo_SF offers transverse structure functions in specific planes:

* **Transverse_ij**: In the xy-plane - ``fun='transverse_ij'``
* **Transverse_ik**: In the xz-plane - ``fun='transverse_ik'``  
* **Transverse_jk**: In the yz-plane - ``fun='transverse_jk'``

Scalar and Combined Functions (3D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to 2D, PyTurbo_SF offers scalar, scalar-scalar, longitudinal-scalar, and transverse-scalar structure functions in 3D:

* **Scalar**: ``fun='scalar'``
* **Scalar-scalar**: ``fun='scalar_scalar'``
* **Longitudinal-scalar**: ``fun='longitudinal_scalar'``
* **Transverse-scalar (xy-plane)**: ``fun='transverse_ij_scalar'``
* **Transverse-scalar (xz-plane)**: ``fun='transverse_ik_scalar'``
* **Transverse-scalar (yz-plane)**: ``fun='transverse_jk_scalar'``

Advanced 3D Functions
~~~~~~~~~~~~~~~~~~~~~

**Longitudinal-Transverse combinations**:

* **xy-plane**: ``fun='longitudinal_transverse_ij'``
* **xz-plane**: ``fun='longitudinal_transverse_ik'``
* **yz-plane**: ``fun='longitudinal_transverse_jk'``

**Advanced structure functions**:

* **Advective**: ``fun='advective'``
* **Pressure work**: ``fun='pressure_work'``

Function Reference Table
------------------------

1D Functions
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function Type
     - Parameter Value
   * - Scalar structure function
     - ``fun='scalar'``
   * - Scalar-scalar structure function  
     - ``fun='scalar_scalar'``

2D Functions
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function Type
     - Parameter Value
   * - Scalar structure function
     - ``fun='scalar'``
   * - Scalar-scalar structure function
     - ``fun='scalar_scalar'``
   * - Longitudinal structure function
     - ``fun='longitudinal'``
   * - Transverse structure function
     - ``fun='transverse'``
   * - Default velocity structure function
     - ``fun='default_vel'``
   * - Longitudinal-transverse structure function
     - ``fun='longitudinal_transverse'``
   * - Longitudinal-scalar structure function
     - ``fun='longitudinal_scalar'``
   * - Transverse-scalar structure function
     - ``fun='transverse_scalar'``
   * - Advective structure function
     - ``fun='advective'``
   * - Pressure work structure function
     - ``fun='pressure_work'``

3D Functions
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function Type
     - Parameter Value
   * - Scalar structure function
     - ``fun='scalar'``
   * - Scalar-scalar structure function
     - ``fun='scalar_scalar'``
   * - Longitudinal structure function
     - ``fun='longitudinal'``
   * - Default velocity structure function
     - ``fun='default_vel'``
   * - Transverse in xy-plane
     - ``fun='transverse_ij'``
   * - Transverse in xz-plane
     - ``fun='transverse_ik'``
   * - Transverse in yz-plane
     - ``fun='transverse_jk'``
   * - Longitudinal-scalar structure function
     - ``fun='longitudinal_scalar'``
   * - Transverse-scalar (xy-plane)
     - ``fun='transverse_ij_scalar'``
   * - Transverse-scalar (xz-plane)
     - ``fun='transverse_ik_scalar'``
   * - Transverse-scalar (yz-plane)
     - ``fun='transverse_jk_scalar'``
   * - Longitudinal-transverse (xy-plane)
     - ``fun='longitudinal_transverse_ij'``
   * - Longitudinal-transverse (xz-plane)
     - ``fun='longitudinal_transverse_ik'``
   * - Longitudinal-transverse (yz-plane)
     - ``fun='longitudinal_transverse_jk'``
   * - Advective structure function
     - ``fun='advective'``
   * - Pressure work structure function
     - ``fun='pressure_work'``

Physical Interpretation
-----------------------

Longitudinal vs Transverse
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Longitudinal structure functions** measure velocity differences parallel to the separation vector. In turbulence, they are related to the energy cascade and follow the famous Kolmogorov scaling :math:`S_L(r) \propto r^{2/3}` in the inertial range.

* **Transverse structure functions** measure velocity differences perpendicular to the separation vector. They provide information about vorticity and enstrophy transfer.

Scalar Structure Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Scalar structure functions** are useful for analyzing passive scalars, temperature, or other scalar quantities in turbulent flows.

* **Scalar-scalar structure functions** help understand correlations between different scalar fields.

Order Dependence
~~~~~~~~~~~~~~~

* **Second-order** (:math:`n=2`) structure functions are most commonly used and relate to energy spectra
* **Third-order** (:math:`n=3`) structure functions are related to energy transfer rates
* **Higher-order** structure functions reveal intermittency and rare events in turbulence

Usage Tips
----------

1. **Choose appropriate function type** based on your physical system and research questions
2. **Consider dimensionality** - use 2D functions for planar data, 3D for volumetric data
3. **Order selection** - start with :math:`n=2` for most applications
4. **Separation range** - ensure adequate statistics at all separation scales
5. **Convergence** - use sufficient bootstrap samples for statistical reliability
