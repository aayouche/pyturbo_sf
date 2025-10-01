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
    orcid: 0000-0000-0000-0000
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
 - name: Department of Earth, Environmental, and Planetary Sciences (DEEPS), Brown University, Providence, RI, USA
   index: 3

date: 30 September 2025
bibliography: paper.bib
---

# Summary

Structure functions are fundamental diagnostic tools in turbulence research that reveal energy cascade characteristics, intermittency properties, and scaling laws by quantifying spatial correlations between field differences at varying separation distances [@Frisch1995; @Pope2000]. PyTurbo_SF is a comprehensive Python package that tackles structure function analysis through an innovative adaptive bootstrap framework, enabling efficient and statistically rigorous calculations for 1D, 2D, and 3D turbulent datasets.

The package addresses critical computational bottlenecks in modern turbulence analysis by implementing power-of-2 spacing (sampling indices) strategies, adaptive convergence monitoring, and memory-optimized algorithms that automatically scale with dataset characteristics. PyTurbo_SF provides unprecedented statistical rigor through bootstrap resampling, delivering robust uncertainty quantification that has been missing from existing structure function tools. The software seamlessly processes xarray Datasets and supports diverse structure function types including longitudinal, transverse, scalar, advective, and energy flux functions recently developed for analyzing geophysical energy transfers [@Pearson2021; @Pearson2024].

Applications span from oceanographic mooring time series and satellite altimetry measurements to high-resolution numerical simulations, enabling researchers to apply consistent, statistically sound methodology across scales from laboratory turbulence to planetary circulation systems.

# Statement of need

Contemporary turbulence research increasingly relies on massive, high-resolution datasets from advanced observational platforms (satellite missions, autonomous platforms) and computational simulations (large eddy simulations, direct numerical simulations). Traditional structure function calculations face severe limitations: computational intractability for large 2D/3D datasets, absence of rigorous uncertainty quantification, manual parameter tuning requirements, and limited function types that restrict scientific insights.

Existing software tools address only subsets of these challenges. fastSF provides parallelized C++ implementations but lacks advanced function types and uncertainty quantification [@Sadhukhan2021]. MATLAB-based toolkits offer specific analyses but are environmentally limited and lack comprehensive statistical frameworks [@Fuchs2022]. Alternative approaches like coarse-graining (FlowSieve) address different aspects of turbulence analysis but cannot provide the detailed scale-by-scale information that structure functions uniquely deliver [@Storer2023].

The scientific community urgently needs tools capable of analyzing emerging datasets from missions like NASA's SWOT satellite, which provides unprecedented high-resolution ocean surface measurements, and next-generation atmospheric simulations that generate terabyte-scale 3D datasets. Recent advances in structure function theory, particularly advective structure functions for quantifying energy flux transfers [@Pearson2021] and spectral flux estimation methodologies [@Pearson2024], require computational frameworks that can handle both traditional and novel function types with statistical rigor.

PyTurbo_SF fills this critical gap by providing the first comprehensive, statistically robust framework for structure function analysis that scales from small observational datasets to massive simulation outputs while delivering quantified uncertainties essential for scientific interpretation.

# Software functionality

PyTurbo_SF implements the complete mathematical framework for structure function analysis, supporting functions of the form $S_n(r) = \langle |\phi(\mathbf{x} + \mathbf{r}) - \phi(\mathbf{x})|^n \rangle_{\mathbf{x}}$ where $\phi$ represents arbitrary field variables (velocity, scalars, derived quantities), $\mathbf{r}$ is the separation vector, $n$ is the order, and $\langle \cdot \rangle_{\mathbf{x}}$ denotes spatial averaging. The package supports traditional functions (longitudinal, transverse, scalar), cross functions (longitudinal-transverse, scalar-scalar) and advective structure functions that enable direct energy flux quantification [@Pearson2021] and pressure-work structure functions for analyzing energy transfer mechanisms.

The core algorithmic breakthrough is the adaptive bootstrap methodology that increases computational efficiency and statistical reliability. The algorithm employs power-of-2 spacings (1, 2, 4, 8, 16, ...) that optimize memory access patterns while providing optimal scale separation. Adaptive convergence monitoring dynamically allocates computational resources by grouping unconverged bins by characteristics and prioritizing bootstrap samples based on statistical effectiveness. This approach eliminates manual parameter tuning while guaranteeing robust uncertainty estimates through principled resampling.

Performance optimization enables analysis of previously intractable datasets. Memory-efficient data structures and intelligent garbage collection maintain peak memory usage at 2-5× base dataset size. Parallel processing via multiple backends (multiprocessing, threading, loky) provides near-linear scaling with available cores. Benchmark testing with 2D DYCOMS turbulence data demonstrates time complexity scaling of O(NM log N log M) for 2D data, enabling analysis of datasets with millions of grid points.

The package provides three main interfaces optimized for different dimensionalities: `bin_sf_1d()` for oceanographic and atmospheric time series, `bin_sf_2d()` for satellite and model surface fields, and `bin_sf_3d()` for volumetric simulation data. All functions automatically detect data characteristics and optimize computational strategies accordingly, while supporting both isotropic averaging and directional analysis for anisotropic flows.

![Satellite altimetry analysis showing second-order longitudinal structure functions calculated from SWOT data in the Gulf Stream region. Bootstrap error bars demonstrate statistical rigor while revealing energy cascade signatures in ocean surface turbulence. Results show the characteristic $r^{2/3}$ scaling predicted by geostrophic turbulence theory.\label{fig:swot}](figs/figure1.png)

![Regional ocean model analysis using CROCO simulation data showing energy transfer mechanisms through combined velocity-scalar structure functions. The adaptive bootstrap algorithm efficiently handles the large 2D spatial dataset while providing robust uncertainty quantification for energy flux estimates.\label{fig:croco}](figs/figure2.png)

![Regional ocean model analysis using CROCO simulation data showing energy transfer mechanisms through combined 2D velocity-scalar structure functions.\label{fig:croco2}](figs/figure3.png)

![Atmospheric boundary layer turbulence analysis from DYCOMS large eddy simulation demonstrating 3D structure function capabilities. Volume-weighted binning accurately captures isotropic averaging in the 3D velocity field while bootstrap resampling provides statistical reliability for intermittency characterization.\label{fig:dycoms}](figs/figure4.png)

# Related work and scientific impact

PyTurbo_SF represents a significant advancement over existing structure function software by uniquely combining comprehensive function types, adaptive bootstrap methodology, and optimized computational algorithms. While fastSF provides basic parallelized calculations [@Sadhukhan2021] and MATLAB toolkits offer specific analyses [@Fuchs2022], no existing software delivers the combination of statistical rigor, computational efficiency, and methodological breadth required for contemporary turbulence research.

The package enables application of recent theoretical developments in structure function analysis, particularly advective structure functions that provide direct measurements of energy flux transfers in geophysical flows [@Pearson2021]. These functions reveal energy pathways that traditional approaches cannot capture, offering new insights into energy cascade mechanisms in ocean and atmospheric turbulence. The software also supports spectral flux estimation methodologies that bridge structure function analysis with spectral energy transfer studies [@Pearson2024].

Scientific applications demonstrate the software's transformative impact across multiple domains. In oceanography, PyTurbo_SF enables analysis of satellite altimetry data for characterizing surface turbulence and energy dissipation rates. Atmospheric scientists can analyze large eddy simulation data to understand boundary layer turbulence and energy transfer mechanisms. The consistent methodology across platforms enables comparative studies that were previously impossible due to software limitations and statistical uncertainties.

The adaptive bootstrap framework addresses a fundamental challenge in turbulence analysis: quantifying uncertainties in structure function estimates. Previous studies often reported results without rigorous error bars, limiting scientific interpretation and reproducibility. PyTurbo_SF's principled uncertainty quantification enables robust statistical comparisons and hypothesis testing, elevating the scientific standards of structure function analysis.

# Acknowledgements

This software package is based upon work supported by the US Department of Energy grant DE-SC0024572.

Any opinions, findings, and conclusions or recommendations expressed in this package are those of the authors and do not necessarily reflect the views of the US Department of Energy.

# References
