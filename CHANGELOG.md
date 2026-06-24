# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed (physics) — found in a rigorous physics/scholarship audit
- **Monte Carlo now integrates.** `run_monte_carlo` previously froze the crack-growth rate at the initial depth `a0` and extrapolated linearly, so PoF was identical for any `n_t` and under-predicted terminal growth. It now steps depth-by-depth (re-evaluating the rate as the crack deepens) via a new `_integrate_sample`, and PoF converges with `n_t`.
- **Failure criterion is now per-zone.** PoF used a single hard-coded 3.3 mm threshold; samples now fail when `K_I` reaches the zone-dependent `K_IH` (HAZ crosses far earlier than base), removing the low bias on the most susceptible population.
- **Stress-intensity factor corrected to the true Newman-Raju surface-crack solution.** The deep-point boundary-correction was a Tada single-edge-crack polynomial mislabelled "Newman-Raju" that over-predicted `K` by up to ~2x for `a/t > 0.5`; replaced with `F = M1 + M2(a/t)^2 + M3(a/t)^4`. `A_CF_BASE` recalibrated (2.4e-14 -> 4.0e-14) to keep da/dt = 0.3 mm/yr at a = 2 mm, and `integrate_full` now terminates at fracture (`K_I >= K_IH`) so post-rupture rates never pollute the trajectory.
- **Dormancy penalty is now applied** (was dead code: dormant and active cracks integrated identically), and the reported velocity `v(t)` now uses the same variable-amplitude rate law that advances the geometry (previously ~4x inconsistent).
- Clarified that the "Xing HEDE" model is cited second-hand via the Sun, Zhou & Kang 2021 review (no separate primary Xing reference is bundled).

### Added
- 11 hardening/regression tests (audit-chain tamper detection, seeded reproducibility, timestep convergence, and a lock against the frozen-rate Monte Carlo bug), bringing the suite to **205 tests** (clears the ICS2 QC Gate-3 >200 minimum).

### Fixed (packaging/docs)
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
