---
title: 'PyTurbo_SF: An Adaptive Bootstrap Framework for Efficient Structure Function Analysis in Turbulent Flows'
tags:
  - Python
  - turbulence
  - structure functions
  - fluid dynamics
  - bootstrap statistics
  - oceanography
  - atmospheric science
  - energy cascade
  - geophysical flows
authors:
  - name: Adam Ayouche
    orcid: 0009-0001-9075-5911
    affiliation: '1'
    corresponding: true
  - name: Baylor Fox-Kemper
    orcid: 0000-0002-2871-2048
    affiliation: '1'
  - name: Momme Hell
    orcid: 0000-0002-5754-3925
    affiliation: '2'
  - name: Brodie Pearson
    orcid: 0000-0002-0202-0481
    affiliation: '3'
  - name: Cassidy Wagner
    orcid: 0000-0002-1186-2082
    affiliation: '3'
    
affiliations:
 - name: Department of Earth, Environmental, and Planetary Sciences (DEEPS), Brown University, Providence, RI, USA
   index: 1
 - name: Woods Hole Oceanographic Institution, Woods Hole, MA, USA
   index: 2
 - name: College of Earth, Ocean, and Atmospheric Sciences, Oregon State University, Corvallis, OR, USA
   index: 3

date: 30 September 2025
bibliography: paper.bib
---

# Summary

Structure functions (SFs) are fundamental diagnostic tools in turbulence research that quantify spatial correlations between field differences at varying separation distances, revealing energy cascade characteristics and scaling laws [@Frisch1995; @Pope2000]. PyTurbo_SF is a comprehensive Python package providing efficient, statistically rigorous structure function calculations for 1D, 2D, and 3D turbulent datasets through an innovative adaptive bootstrap framework.

The package addresses computational bottlenecks through power-of-2 spacing strategies, adaptive convergence monitoring, and memory-optimized algorithms. PyTurbo_SF delivers robust uncertainty quantification through bootstrap resampling and supports diverse (conditional) structure function types including longitudinal, transverse, scalar, advective, and energy flux functions [@Pearson2021; @Pearson2024].

Applications span oceanographic time series and satellite measurements to high-resolution simulations, enabling consistent methodology across scales from laboratory to planetary systems. Despite its name evoking the broader turbulence field, PyTurbo_SF is specifically designed for structure function analysis — one of the most widely used diagnostic approaches in turbulence research.

# Statement of need

Contemporary turbulence research relies on massive datasets from satellite missions, autonomous platforms, and high-resolution simulations. Traditional structure function calculations face severe limitations: computational intractability for large datasets, absence of uncertainty quantification, manual parameter tuning, and limited function types.

Existing tools address only subsets of these challenges. fastSF provides parallelized implementations but lacks advanced function types and uncertainty quantification [@Sadhukhan2021]. MATLAB toolkits are environmentally limited and lack comprehensive statistical frameworks [@Fuchs2022]. FluidSF [@Wagner2025] is a related Python package supporting core structure function types, which we compare in detail below. Alternative approaches like FlowSieve cannot provide the scale-by-scale information structure functions uniquely deliver [@Storer2023].

There is a growing need for tools to analyze emerging datasets from NASA's SWOT satellite [@Morrow2019] and next-generation atmospheric simulations generating terabyte-scale outputs. Similarly, flux tower networks (e.g., AmeriFlux, FLUXNET) generate extensive eddy-covariance time series of velocity and scalar turbulence that would benefit from standardized structure function diagnostics with uncertainty quantification. Recent advances in structure function theory, particularly advective structure functions [@Pearson2021] and spectral flux estimation [@Pearson2024], require frameworks handling both traditional and novel function types with statistical rigor.

PyTurbo_SF fills this gap by providing the first comprehensive, statistically robust framework that scales from small observational datasets to massive simulation outputs while delivering quantified uncertainties essential for scientific interpretation.

# Software functionality

PyTurbo_SF implements the complete mathematical framework for structure function analysis, supporting functions of the form $S_n(r) = \langle |\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x})|^n \rangle_{\mathbf{x}}$ where $\phi$ represents arbitrary field variables (velocity, scalars, derived quantities), $\mathbf{r}$ is the separation vector, $n$ is the order, and $\langle \cdot \rangle_{\mathbf{x}}$ denotes spatial averaging. For velocity fields, the package computes **longitudinal** SFs using the velocity component parallel to the separation vector ($\delta u_L = \delta \mathbf{u} \cdot \hat{\mathbf{r}}$), **transverse** SFs using the perpendicular component ($\delta u_T = |\delta \mathbf{u} \times \hat{\mathbf{r}}|$), and **scalar** SFs for temperature, salinity, or other tracers. Cross-structure functions correlate different components (e.g., longitudinal-scalar), while **advective** SFs correlate velocity increments with advection increments to directly quantify energy flux [@Pearson2021].

The core algorithmic breakthrough is adaptive bootstrap methodology increasing computational efficiency and statistical reliability. The algorithm employs power-of-2 spacings optimizing memory access patterns while providing optimal scale separation. Adaptive convergence monitoring dynamically allocates computational resources, eliminating manual parameter tuning while guaranteeing robust uncertainty estimates.

Performance optimization enables analysis of previously intractable datasets. Memory-efficient structures maintain peak usage at 2-5× base dataset size. Parallel processing provides near-linear scaling with available cores. Benchmark testing demonstrates O(NM log N log M) complexity for 2D data, enabling analysis of datasets with millions of grid points.

The package provides three main interfaces: `bin_sf_1d()` for time series, `bin_sf_2d()` for surface fields, and `bin_sf_3d()` for volumetric data. All functions automatically optimize computational strategies while supporting both isotropic and directional analysis. Applications extend beyond traditional oceanographic and atmospheric modeling contexts to include eddy-covariance field measurements, where high-frequency sonic anemometer time series of wind components and scalars can be analyzed using `bin_sf_1d()` for temporal structure functions.

Figure \ref{fig:swot} demonstrates PyTurbo_SF applied to NASA SWOT satellite altimetry in the Gulf Stream region. The third-order longitudinal and advective SFs reveal a sign change near 100 km separation, marking the transition from inverse to forward energy cascade characteristic of quasi-geostrophic ocean turbulence.

![SWOT altimetry analysis in the Gulf Stream. **Upper:** zonal geostrophic velocity field ($u_g$, m/s). **Lower:** isotropic $\mathrm{SF}_{LLL}/r$ (blue) and advective SF $\mathrm{ASF}_V$ (red) in m$^2$ s$^{-3}$, with 95% bootstrap confidence intervals (shading). Positive values indicate forward cascade; negative values indicate inverse cascade.\label{fig:swot}](figs/figure1.png)

Figure \ref{fig:croco} applies the same framework to CROCO regional ocean model output [@Shchepetkin2005]. Both third-order energy-flux SFs and traditional second-order SFs are computed isotropically using `get_isotropic_sf_2d()`, with shaded 95% bootstrap confidence intervals on all quantities.

![Isotropic SFs from CROCO simulation. **Left:** absolute vorticity $\omega$ (upper) and zonal velocity $u$ (lower) over a 150 km $\times$ 250 km domain. **Upper right:** third-order SFs ($\mathrm{ASF}_V$, $\mathrm{SF}_{LLL}/r$, $-\mathrm{ASF}_\omega$, $-\mathrm{SF}_{L\omega\omega}/r$). **Lower right:** second-order SFs ($\mathrm{SF}_{LL}$, $\mathrm{SF}_{TT}$, $\mathrm{SF}_{\omega\omega}$). Shaded regions: 95% bootstrap confidence intervals.\label{fig:croco}](figs/figure2.png)

Figure \ref{fig:croco2} shows the full 2D (non-isotropic) SF output from the same CROCO snapshot, computed using `bin_sf_2d()`. The asymmetry about the diagonal confirms anisotropic energy transfer rates in zonal versus meridional directions.

![Non-isotropic SFs from CROCO computed with `bin_sf_2d()`, showing SF amplitude as a function of zonal ($x$) and meridional ($y$) separation distance (log axes, $10^3$–$10^5$ m). **Left:** $\mathrm{ASF}_V$. **Center:** $\mathrm{SF}_{LLL}$. **Right:** $\mathrm{SF}_{LL}$. White regions indicate insufficient data pairs.\label{fig:croco2}](figs/figure3.png)

Figure \ref{fig:dycoms} illustrates the 3D capability using DYCOMS-II large eddy simulation output [@Stevens2005]. The second-order longitudinal SF computed with `bin_sf_3d()` exhibits horizontal isotropy in the $r_x$–$r_y$ plane and distinct vertical anisotropy imposed by stable stratification at the boundary layer top.

![3D second-order longitudinal SF ($\mathrm{SF}_{LL}$, m$^2$ s$^{-2}$) from DYCOMS-II LES, computed with `bin_sf_3d()`. Axes: $r_x$, $r_y$ (horizontal, 0–3500 m) and $r_z$ (vertical, 0–45000 m). Horizontal symmetry indicates isotropic turbulence; vertical anisotropy reflects stable stratification.\label{fig:dycoms}](figs/figure4.png)

# Related work and scientific impact

PyTurbo_SF represents a significant advancement by uniquely combining comprehensive function types, adaptive bootstrap methodology, and optimized algorithms. While fastSF provides basic parallelized calculations [@Sadhukhan2021] and MATLAB toolkits offer specific analyses [@Fuchs2022], no existing software delivers the combination of statistical rigor, efficiency, and breadth required for contemporary turbulence research.

FluidSF [@Wagner2025] is a related Python package for structure function calculations, supporting 1D/2D/3D data and core structure function types including longitudinal, transverse, scalar, and advective functions. At the time of this writing, FluidSF provides second- and third-order structure functions with a simplified isotropic averaging approach and outputs results as NumPy arrays. It offers a velocity-scalar blended cross-term type and a single transverse decomposition in 3D.

PyTurbo_SF is, to our knowledge, the first open-source package to combine arbitrary-order structure function computation, adaptive bootstrap uncertainty quantification with automatic convergence monitoring, and parallel processing in a single framework. It provides extensive cross-term structure function types (longitudinal-transverse, longitudinal-scalar, transverse-scalar, and scalar-scalar), full 3D transverse decomposition across all three coordinate planes (ij, ik, jk), conditional structure functions, exact spherical and polar binning for isotropic averaging, and spectral energy flux estimation via Bessel transform — all accelerated by power-of-2 spacing strategies and joblib-based parallelization. Results are returned as richly annotated xarray Datasets with embedded metadata, convergence diagnostics, and quality masks.

PyTurbo_SF's primary contributions are: (1) rigorous uncertainty quantification through adaptive bootstrap resampling with automatic convergence monitoring, (2) support for arbitrary-order structure functions essential for intermittency analysis, and (3) computational efficiency through parallelization and power-of-2 spacing strategies enabling analysis of large datasets.

The package enables application of recent theoretical developments, particularly advective structure functions providing direct energy flux measurements [@Pearson2021] and spectral flux estimation methodologies [@Pearson2024]. These reveal energy pathways traditional approaches cannot capture, offering insights into cascade mechanisms in ocean and atmospheric turbulence.

Scientific applications demonstrate transformative impact across domains. PyTurbo_SF enables analysis of satellite altimetry data for characterizing surface turbulence and large eddy simulation data for understanding boundary layer dynamics. The consistent methodology enables comparative studies previously impossible due to software limitations.

The adaptive bootstrap framework addresses a fundamental challenge: quantifying uncertainties in structure function estimates. PyTurbo_SF's principled uncertainty quantification enables robust statistical comparisons and hypothesis testing, elevating scientific standards.

# Acknowledgements

This software package, and the contributions of AA and BFK, is based upon work supported by the US Department of Energy grant DE-SC0024572. BP and CW were supported by the National Science Foundation under Grants OCE-2023721 and OCE-2525055.

# References
