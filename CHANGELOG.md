# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-12

### Added
- Full-physics NNpHSCC simulation package with 15 coupled mechanisms.
- Chen-Sutherby-Xing combined-parameter crack growth law with frequency saturation at `f_crit = 10^-3 Hz`.
- Newman-Raju 3D semi-elliptical crack geometry (coupled `(a, c)` EDOs).
- BS 7910:2019 Annex Q residual stress SIF for as-welded HAZ.
- BS 7910:2019 Cl. 7.3 coalescence rule for adjacent surface cracks.
- Turnbull (1993) crack-tip pH model with Fe2+ hydrolysis and CO2/HCO3- buffering.
- Non-monotonic CP potential model (minimum CGR at -750 mV CSE).
- Microstructure zones: base metal, HAZ, vintage X52 base, vintage X52 ERW seam weld.
- Crack dormancy criterion (Zhao 2017): Stage I to Stage II transition.
- Latin Hypercube Monte Carlo with 8 sampled parameters, post-ILI POD-corrected `a0`.
- Sun, Zhou and Kang (2021) model structural uncertainty (`COV = 61.2%`) as sampled epistemic variable.
- Bayesian particle-filter posterior update from re-inspection events (Straub 2004).
- API 579-1 Level 2 FAD assessment.
- ASME B31.8 / B31.12 MAOP comparison.
- H2 blending K_IH degradation model (Cui 2024).
- Risk-based inspection interval optimizer.
- SHA-256 hash-chained audit log of all runs.
- 11 publication-quality figures and an animated colony GIF.
- Full LaTeX equations reference (`equations.html`).
- 194 unit and integration tests; benchmarks against Faraday law, Tada flat-plate K_I, Weibull POD.

### Tests
- 194 / 194 passing on a clean Python 3.11+ extract.

[1.0.0]: https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/releases/tag/v1.0.0
