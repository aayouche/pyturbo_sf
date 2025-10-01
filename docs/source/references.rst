References and Background
=========================

This page provides scientific background, references, and theoretical foundations for the structure function calculations implemented in PyTurbo_SF.

Theoretical Background
----------------------

Structure Function Theory
~~~~~~~~~~~~~~~~~~~~~~~~~

Structure functions are fundamental tools for analyzing the statistical properties of turbulent flows and other complex systems. They quantify the scaling behavior of fluctuations as a function of spatial or temporal separation.

**Definition:**

The n-th order structure function is defined as:

.. math::

   S_n(r) = \langle |\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x})|^n \rangle_{\mathbf{x}}

where:
   - :math:`\phi(\mathbf{x})` is the field variable at position :math:`\mathbf{x}`
   - :math:`\mathbf{r}` is the separation vector
   - :math:`\langle \cdot \rangle_{\mathbf{x}}` denotes spatial averaging
   - :math:`n` is the order of the structure function

**Physical Interpretation:**

- **Second-order (n=2):** Related to energy content and variance
- **Third-order (n=3):** Connected to energy transfer rates
- **Higher orders:** Capture intermittency and rare events

Kolmogorov Theory (1941)
~~~~~~~~~~~~~~~~~~~~~~~~

Kolmogorov's seminal work on turbulence provides the theoretical foundation for structure function scaling:

**Key Assumptions:**
   1. Statistical homogeneity and isotropy
   2. Scale separation between energy injection and dissipation
   3. Universal scaling in the inertial range

**Main Results:**

For the inertial subrange, Kolmogorov predicted:

.. math::

   S_2(r) = C_2 (\epsilon r)^{2/3}

   S_3(r) = -\frac{4}{5} \epsilon r

where:
   - :math:`C_2 \approx 2.0` is the Kolmogorov constant
   - :math:`\epsilon` is the energy dissipation rate per unit mass
   - The minus sign in :math:`S_3` reflects the forward energy cascade

**General Scaling:**

For higher-order structure functions:

.. math::

   S_n(r) \propto r^{\zeta_n}

where the scaling exponents follow :math:`\zeta_n = n/3` in K41 theory.

Extended Self-Similarity and Intermittency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real turbulent flows often deviate from Kolmogorov scaling due to intermittency effects.

**Refined Similarity Hypothesis:**

She & Leveque (1994) proposed a log-normal model for intermittency:

.. math::

   \zeta_n = \frac{n}{3} - \frac{\mu}{6}n(n-3)

where :math:`\mu` is the intermittency parameter.

**Multifractal Models:**

More sophisticated models describe intermittency through multifractal scaling:

.. math::

   \zeta_n = \frac{n}{3} + \tau(n/3)

where :math:`\tau(h)` is the multifractal spectrum.

2D Turbulence Theory
~~~~~~~~~~~~~~~~~~~~

Two-dimensional turbulence exhibits different phenomenology from 3D due to the inverse energy cascade.

**Dual Cascade Theory:**

Kraichnan (1967) and Batchelor (1969) predicted:

- **Energy cascade:** :math:`E(k) \propto k^{-5/3}` (to large scales)
- **Enstrophy cascade:** :math:`E(k) \propto k^{-3}` (to small scales)

**Structure Functions in 2D:**

The energy cascade gives similar scaling to 3D for second-order structure functions:

.. math::

   S_2^{2D}(r) \propto r^{2/3}

However, higher-order statistics may differ due to different intermittency properties.

Anisotropic Turbulence
~~~~~~~~~~~~~~~~~~~~~~

For anisotropic flows, structure functions depend on the orientation of the separation vector.

**Longitudinal vs Transverse:**

- **Longitudinal:** :math:`S_L(r) = \langle |u_\parallel(\mathbf{x}+\mathbf{r}) - u_\parallel(\mathbf{x})|^n \rangle`
- **Transverse:** :math:`S_T(r) = \langle |u_\perp(\mathbf{x}+\mathbf{r}) - u_\perp(\mathbf{x})|^n \rangle`

where :math:`u_\parallel` and :math:`u_\perp` are velocity components parallel and perpendicular to :math:`\mathbf{r}`.

**Theoretical Relations:**

For isotropic turbulence, Kolmogorov theory predicts:

.. math::

   S_T(r) = \frac{4}{3} S_L(r)

Key Scientific Papers
---------------------

Foundational Works
~~~~~~~~~~~~~~~~~

.. [Kolmogorov1941] **Kolmogorov, A. N.** (1941). "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers." *Proceedings of the Royal Society of London A*, 434(1890), 9-13.

   *The seminal paper establishing the theoretical foundation for turbulence scaling laws and structure functions.*

.. [Obukhov1941] **Obukhov, A. M.** (1941). "On the distribution of energy in the spectrum of turbulent flow." *Doklady Akademii Nauk SSSR*, 32, 19-21.

   *Independent derivation of similar scaling laws, often cited together with Kolmogorov's work.*

.. [Kolmogorov1962] **Kolmogorov, A. N.** (1962). "A refinement of previous hypotheses concerning the local structure of turbulence in a viscous incompressible fluid at high Reynolds number." *Journal of Fluid Mechanics*, 13(1), 82-85.

   *Kolmogorov's refined theory (K62) introducing intermittency corrections.*

Classical Turbulence Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [Frisch1995] **Frisch, U.** (1995). *Turbulence: The Legacy of A. N. Kolmogorov*. Cambridge University Press.

   *Comprehensive textbook on turbulence theory, including detailed treatment of structure functions and intermittency.*

.. [Pope2000] **Pope, S. B.** (2000). *Turbulent Flows*. Cambridge University Press.

   *Standard reference for turbulence theory and modeling, with extensive coverage of statistical methods.*

.. [Monin1975] **Monin, A. S., & Yaglom, A. M.** (1975). *Statistical Fluid Mechanics, Volume 2: Mechanics of Turbulence*. MIT Press.

   *Classic two-volume work providing rigorous mathematical treatment of turbulence statistics.*

2D Turbulence
~~~~~~~~~~~~~

.. [Kraichnan1967] **Kraichnan, R. H.** (1967). "Inertial ranges in two‐dimensional turbulence." *Physics of Fluids*, 10(7), 1417-1423.

   *Foundational paper on 2D turbulence theory predicting dual cascade phenomenology.*

.. [Batchelor1969] **Batchelor, G. K.** (1969). "Computation of the energy spectrum in homogeneous two‐dimensional turbulence." *Physics of Fluids*, 12(12), II-233.

   *Important work on 2D turbulence spectra and enstrophy cascade.*

.. [Tabeling2002] **Tabeling, P.** (2002). "Two-dimensional turbulence: a physicist approach." *Physics Reports*, 362(1), 1-62.

   *Comprehensive review of 2D turbulence from experimental and theoretical perspectives.*

Intermittency and Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. [SheLeveque1994] **She, Z. S., & Leveque, E.** (1994). "Universal scaling laws in fully developed turbulence." *Physical Review Letters*, 72(3), 336-339.

   *Introduction of the log-normal model for intermittency corrections to Kolmogorov scaling.*

.. [Dubrulle1994] **Dubrulle, B.** (1994). "Intermittency in fully developed turbulence: Log-Poisson statistics and generalized scale covariance." *Physical Review Letters*, 73(7), 959-962.

   *Development of log-Poisson models for turbulence intermittency.*

.. [Benzi1993] **Benzi, R., Ciliberto, S., Tripiccione, R., Baudet, C., Massaioli, F., & Succi, S.** (1993). "Extended self-similarity in turbulent flows." *Physical Review E*, 48(1), R29-R32.

   *Introduction of Extended Self-Similarity (ESS) method for analyzing structure function scaling.*

Structure Function Applications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. [Pearson2021] **Pearson, B., et al.** (2021). "Advective structure functions in anisotropic two-dimensional turbulence." *Journal of Fluid Mechanics*, 915, A8.

   *Modern application of structure functions to analyze anisotropic 2D turbulence with advanced statistical methods.*

.. [Lindborg2008] **Lindborg, E.** (2008). "Third-order structure function relations for quasi-geostrophic turbulence." *Journal of Fluid Mechanics*, 607, 1-11.

   *Application of structure function analysis to geophysical flows and atmospheric turbulence.*

.. [Nastrom1985] **Nastrom, G. D., & Gage, K. S.** (1985). "A climatology of atmospheric wavenumber spectra of wind and temperature observed by commercial aircraft." *Journal of the Atmospheric Sciences*, 42(9), 950-960.

   *Classic observational study using structure function-like analysis for atmospheric turbulence.*

Geophysical Applications
~~~~~~~~~~~~~~~~~~~~~~~

.. [Charney1971] **Charney, J. G.** (1971). "Geostrophic turbulence." *Journal of the Atmospheric Sciences*, 28(6), 1087-1095.

   *Fundamental work on geostrophic turbulence relevant to atmospheric and oceanic flows.*

.. [Vallis2017] **Vallis, G. K.** (2017). *Atmospheric and Oceanic Fluid Dynamics*. Cambridge University Press.

   *Modern textbook covering geophysical fluid dynamics with relevance to structure function analysis.*

.. [Rhines1979] **Rhines, P. B.** (1979). "Geostrophic turbulence." *Annual Review of Fluid Mechanics*, 11(1), 401-441.

   *Comprehensive review of geostrophic turbulence theory and observations.*

Computational Methods
~~~~~~~~~~~~~~~~~~~~~

.. [Yeung2002] **Yeung, P. K., & Zhou, Y.** (2002). "Universality of the Kolmogorov constant in numerical simulations of turbulence." *Physical Review E*, 56(1), 1746-1752.

   *Computational study of structure functions in direct numerical simulations.*

.. [Gotoh2002] **Gotoh, T., Fukayama, D., & Nakano, T.** (2002). "Velocity field statistics in homogeneous steady turbulence obtained using a high-resolution direct numerical simulation." *Physics of Fluids*, 14(3), 1065-1081.

   *High-resolution DNS study providing benchmark results for structure function calculations.*

Experimental Studies
~~~~~~~~~~~~~~~~~~~~

.. [Anselmet1984] **Anselmet, F., Gagne, Y., Hopfinger, E. J., & Antonia, R. A.** (1984). "High-order velocity structure functions in turbulent shear flows." *Journal of Fluid Mechanics*, 140, 63-89.

   *Important experimental study of higher-order structure functions in shear flows.*

.. [Sreenivasan1991] **Sreenivasan, K. R., & Antonia, R. A.** (1997). "The phenomenology of small-scale turbulence." *Annual Review of Fluid Mechanics*, 29(1), 435-472.

   *Comprehensive review of small-scale turbulence phenomenology including structure function observations.*

Mathematical Methods
--------------------

Bootstrap Statistics
~~~~~~~~~~~~~~~~~~~~

The bootstrap method used in PyTurbo_SF for error estimation is based on:

.. [Efron1979] **Efron, B.** (1979). "Bootstrap methods: another look at the jackknife." *The Annals of Statistics*, 7(1), 1-26.

   *Original development of the bootstrap statistical method.*

.. [Efron1993] **Efron, B., & Tibshirani, R. J.** (1993). *An Introduction to the Bootstrap*. Chapman and Hall.

   *Comprehensive treatment of bootstrap methods and applications.*

The adaptive bootstrap algorithm implemented in PyTurbo_SF ensures convergence while minimizing computational cost.

Spectral Analysis
~~~~~~~~~~~~~~~~

The relationship between structure functions and energy spectra:

.. [Batchelor1953] **Batchelor, G. K.** (1953). *The Theory of Homogeneous Turbulence*. Cambridge University Press.

   *Classic work establishing connections between different statistical measures of turbulence.*

For isotropic turbulence, the second-order structure function relates to the energy spectrum via:

.. math::

   S_2(r) = 2 \int_0^\infty E(k) [1 - J_0(kr)] dk

where :math:`J_0` is the zeroth-order Bessel function.

Practical Applications
---------------------

Atmospheric Sciences
~~~~~~~~~~~~~~~~~~~~

.. [Lindborg1999] **Lindborg, E.** (1999). "Can the atmospheric kinetic energy spectrum be explained by two-dimensional turbulence?" *Journal of Fluid Mechanics*, 388, 259-288.

   *Application of 2D turbulence theory to atmospheric mesoscale motions.*

.. [Waite2004] **Waite, M. L., & Snyder, C.** (2004). "The mesoscale kinetic energy spectrum of a baroclinic life cycle." *Journal of the Atmospheric Sciences*, 61(3), 330-352.

   *Structure function analysis of atmospheric baroclinic instability.*

Oceanography
~~~~~~~~~~~~

.. [Callies2013] **Callies, J., & Ferrari, R.** (2013). "Interpreting energy and tracer spectra of upper-ocean turbulence in the submesoscale range (1–200 km)." *Journal of Physical Oceanography*, 43(11), 2456-2474.

   *Modern application of spectral and structure function analysis to oceanic submesoscale dynamics.*

.. [Klein2008] **Klein, P., Hua, B. L., Lapeyre, G., Capet, X., Le Gentil, S., & Sasaki, H.** (2008). "Upper ocean turbulence from high-resolution 3D simulations." *Journal of Physical Oceanography*, 38(8), 1748-1763.

   *High-resolution numerical study using structure function analysis for ocean turbulence.*

Engineering Applications
~~~~~~~~~~~~~~~~~~~~~~~~

.. [George1992] **George, W. K.** (1992). "The decay of homogeneous isotropic turbulence." *Physics of Fluids A*, 4(7), 1492-1509.

   *Engineering application of structure function analysis to turbulence decay.*

.. [Mydlarski2003] **Mydlarski, L.** (2003). "Mixed velocity–passive scalar statistics in high-Reynolds-number turbulence." *Journal of Fluid Mechanics*, 475, 173-203.

   *Study of velocity-scalar structure functions relevant to mixing and combustion applications.*

Astrophysical Applications
~~~~~~~~~~~~~~~~~~~~~~~~~

.. [Goldreich1995] **Goldreich, P., & Sridhar, S.** (1995). "Toward a theory of interstellar turbulence. 2: Strong Alfvenic turbulence." *The Astrophysical Journal*, 438, 763-775.

   *Application of turbulence theory and structure functions to astrophysical plasmas.*

.. [Cho2000] **Cho, J., & Lazarian, A.** (2000). "Compressible magnetohydrodynamic turbulence: mode coupling, scaling relations, anisotropy, viscosity-damped regime and astrophysical implications." *Monthly Notices of the Royal Astronomical Society*, 315(4), 810-830.

   *Structure function analysis in magnetohydrodynamic turbulence.*

Data Analysis Techniques
------------------------

Time Series Analysis
~~~~~~~~~~~~~~~~~~~~

For 1D time series, structure functions provide an alternative to spectral analysis:

.. [Lovejoy1987] **Lovejoy, S., & Schertzer, D.** (1987). "Scale invariance, symmetries, fractals, and stochastic simulations of atmospheric phenomena." *Bulletin of the American Meteorological Society*, 68(1), 21-47.

   *Application of scaling analysis to atmospheric time series.*

Spatial Analysis
~~~~~~~~~~~~~~~

For 2D and 3D spatial data:

.. [Schmitt2007] **Schmitt, F. G.** (2007). "About Boussinesq's turbulent viscosity hypothesis: historical remarks and a direct evaluation of its validity." *Comptes Rendus Mécanique*, 335(9-10), 617-627.

   *Historical perspective and modern validation of turbulence theories using structure function analysis.*




Citation Guidelines
-------------------

If you use PyTurbo_SF in your research, please cite:

**Software Citation:**
   PyTurbo_SF Development Team (2024). PyTurbo_SF: A Python package for efficient structure function calculations. Version X.X.X. https://github.com/aayouche/pyturbo_sf

**Methodology Citations:**

Depending on your application, please also cite relevant theoretical papers:

- For basic structure function theory: Kolmogorov (1941)
- For 2D turbulence applications: Kraichnan (1967), Batchelor (1969)
- For intermittency analysis: She & Leveque (1994)
- For geophysical applications: Lindborg (2008)

**Bibtex Entries:**

.. code-block:: bibtex

   @software{pyturbo_sf,
     title = {PyTurbo_SF: A Python package for efficient structure function calculations},
     author = {PyTurbo_SF Development Team},
     year = {2024},
     url = {https://github.com/aayouche/pyturbo_sf},
     version = {1.0.0}
   }

   @article{kolmogorov1941,
     title = {The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers},
     author = {Kolmogorov, A. N.},
     journal = {Proceedings of the Royal Society of London A},
     volume = {434},
     number = {1890},
     pages = {9--13},
     year = {1941}
   }

Further Reading
---------------

**Textbooks:**
   - Frisch (1995): Comprehensive theoretical treatment
   - Pope (2000): Engineering perspective on turbulence
   - Vallis (2017): Geophysical fluid dynamics applications

**Review Articles:**
   - Sreenivasan & Antonia (1997): Small-scale turbulence phenomenology
   - Tabeling (2002): Two-dimensional turbulence
   - Schmitt (2007): Historical perspective on turbulence theory

**Online Resources:**
   - `Turbulence Online Database <http://turbulence.pha.jhu.edu/>`_

