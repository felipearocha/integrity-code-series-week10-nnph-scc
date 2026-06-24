# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 10 Gate-3 hardening tests (audit-chain tamper detection, seeded reproducibility of the model-error sampler, and crack-growth timestep convergence), bringing the suite to **204 tests** (clears the ICS2 QC Gate-3 >200 minimum).

### Fixed
- Corrected the `sun_zhou_kang_2021` BibTeX entry in `paper/paper.bib`: it pointed to a burst-capacity paper (IJPVP) but is cited for the NNpHSCC crack-growth model error (COV = 61.2%); now references the Sun, Zhou & Kang combined-model paper in *Journal of Infrastructure Preservation and Resilience* (vol. 2, art. 6), matching the project's `.zenodo.json`.
- Synced the README "Repository Structure" tree with the real module names (removed stale `_v2`/`_v3` suffixes, a non-existent `linkedin/` folder, and the wrong test/visualization filenames); figure count corrected to 9 panels.
- Removed an unresolved `[SOURCE: ...]` tag rendered as prose in the Unityville incident line and attributed both motivating incidents to their official investigating bodies.

### Changed
- Raised the supported-Python floor to **3.11** across the badge, `pyproject.toml` (`requires-python`, classifiers, ruff `target-version`) and `CONTRIBUTING.md`, matching the CI matrix (3.11, 3.12).

## [1.0.1] - 2026-05-13

### Added
- `.zenodo.json` with full Zenodo deposit metadata (creators, affiliation, keywords, references, related identifiers) so the archived release renders a professional citation page.
- DOI badge placeholder and `## How to Cite` section in `README.md` (citation block, BibTeX snippet, concept-vs-version DOI guidance).
- `identifiers:` section, `affiliation: "Independent Researcher"` and ORCID placeholder in `CITATION.cff`; abstract expanded to a full self-contained summary of the 15 coupled mechanisms.

### Changed
- Package version bumped from `1.0.0` to `1.0.1`.
- Citation file `CITATION.cff` and project metadata aligned with the Zenodo deposit so all three sources (GitHub UI widget, Zenodo deposit page, PyPI long description) agree.

### Why this release
- v1.0.0 predated activation of the Zenodo - GitHub integration, so it was never archived to Zenodo. v1.0.1 is the first release that will trigger Zenodo to mint a concept DOI and a version DOI, establishing a permanent citable record of this work.

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

[1.0.1]: https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/releases/tag/v1.0.1
[1.0.0]: https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/releases/tag/v1.0.0
