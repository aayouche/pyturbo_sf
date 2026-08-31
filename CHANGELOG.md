# Changelog

All notable changes to PyTurbo_SF are documented here.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.9] - 2026-08-25

### Fixed

**Correctness fix in the 1D bootstrap path. Users of `bin_sf_1d` with
bootstrapping should re-run affected analyses.**

`compute_boot_indexes_1d` and `get_boot_indexes_1d` passed `(window_size,)`
as the window shape to `numpy.lib.stride_tricks.sliding_window_view`,
producing an array of shape `(n_windows, window_size)`. The 2D and 3D
equivalents pass `(n_windows,)`, producing `(bootsize, n_windows)`, which is
the layout `structure_functions.py` assumes when it selects a bootstrap
sample with `indexes[dim][:, nb]`. The 1D array was therefore transposed
relative to the convention used to read it.

Two consequences followed:

- Each bootstrap sample contained `ceil((N - bootsize*spacing + 1)/spacing)`
  points instead of `bootsize`. At spacing 1 this is close to the full
  record length, so samples spanned almost the entire dataset and separations
  extended far beyond the intended window. The effective window scaled with
  the size of the input, so results were not invariant to record length as
  they should be.
- Only `bootsize * spacing` samples were reachable instead of
  `N - bootsize*spacing + 1`, and those samples were near-duplicates of one
  another. Bootstrap confidence intervals were consequently far too narrow;
  in a representative mooring-length test (N = 10898, bootsize = 170) the
  median relative CI width was roughly sixteen times smaller than it should
  have been.

Point estimates shift modestly. The confidence intervals are the substantive
problem: any published or in-review uncertainty estimate from `bin_sf_1d`
should be regenerated.

The 2D and 3D paths (`bin_sf_2d`, `bin_sf_3d`) are **not affected** — their
index construction was already correct. Calls to `bin_sf_1d` where `bootsize`
equals or exceeds the dimension length are also unaffected, because that case
takes the `num_bootstrappable == 0` branch and uses the full record without
consulting the bootstrap indexes.

Affected released versions: 1.0.4 through 1.0.8.

Reported by Andrey Shcherbina (University of Washington / APL).

### Added

- `tests/test_boot_indexes_1d_layout.py` — regression tests asserting the
  `(bootsize, n_windows)` layout, the contents of each bootstrap column, and
  agreement between the 1D and 2D index builders and between the precomputed
  and on-the-fly code paths.
- `tests/test_version_consistency.py` — asserts that the version string in
  `pyproject.toml`, `src/pyturbo_sf/__init__.py`, and `docs/source/conf.py`
  agree.
- `CHANGELOG.md` (this file).

### Changed

- `docs/source/examples/example_1D_mooring.ipynb` — the introduction
  previously read the timing comparison between bootstrapped and
  non-bootstrapped runs as evidence of package inefficiency. That timing
  anomaly was in fact a symptom of the indexing bug: bootstrapped runs were
  slower than the full-record run, and got faster as `bootsize` grew, both of
  which are backwards. The text now states the expected scaling, and a
  sanity-check cell verifying the index layout has been added.
- `docs/source/conf.py` version strings had drifted to 1.0.7 while the package
  was at 1.0.8; all version strings are now synchronized.

## [1.0.8] - 2026-03-31

Version published alongside the JOSS paper
([10.21105/joss.09876](https://doi.org/10.21105/joss.09876)).
