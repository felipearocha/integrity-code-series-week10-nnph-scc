# Integrity Code Series — Week 10 — NNpHSCC Full-Physics Simulation

[![CI](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/actions/workflows/ci.yml)
[![Release](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/actions/workflows/release.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-215%20passed-brightgreen.svg)](#validation)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172241.svg)](https://doi.org/10.5281/zenodo.20172241)

**Author:** Felipe Rocha — Independent Researcher

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| Week 3 | [Integrity-code-series-3](https://github.com/felipearocha/Integrity-code-series-3) | F1 lap simulation (six coupled ODEs) |
| Week 6 | [Integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/Integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| Week 7 | [integrity_code_series_week7_h2_lferw](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw) | LF-ERW H2 conversion (B31.12 + NACE TM0316) |
| Week 8 | [integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater) | Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) |
| Week 9 | [integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui) | CUI thermohygro-electrochemical (3 PDEs, Strang) |
| Week 10 | [integrity-code-series-week-10_nnph_scc](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc) | NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) |
| Bonus | [Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation](https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation) | Vibration-accelerated corrosion (SDOF + Butler-Volmer + Archard) |
| Bonus | [synthetic-integrity-digital-twin-piml](https://github.com/felipearocha/synthetic-integrity-digital-twin-piml) | Physics-informed neural-network surrogate |
| Bonus | [integrity-data-foundation](https://github.com/felipearocha/integrity-data-foundation) | Engineering data validation baseline |

## Quick Start

```bash
# Clone and install
git clone https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc.git
cd integrity-code-series-week-10_nnph_scc
pip install -e .[dev]

# Validate
pytest tests/ -q                          # 215 tests
python -m validation.benchmarks           # textbook constants

# Reproduce all panels + audit-chained outputs
python run_all.py
```

Browse `equations.html` in any modern browser for the full LaTeX equations reference.

## Problem Statement

Near-Neutral pH Stress Corrosion Cracking (NNpHSCC) accounts for a major fraction of
reportable incidents on gas transmission pipelines in North America. Two specific incidents
motivate this package:

- **2018 Prince George, BC rupture** — 914mm gas pipeline; NNpHSCC; downstream of a compressor station (Transportation Safety Board of Canada investigation)
- **2015 Unityville, PA rupture** — 609mm gas pipeline; NNpHSCC (US DOT/PHMSA failure investigation)

PHMSA §192.611(a)(4) (effective March 16, 2026) requires integrity assessment for class location
changes. This package provides the physics foundation for that assessment.

**Key finding from v2:** Even the best available NNpHSCC crack growth model has COV = 61.2% on
validated full-scale test data. This package explicitly incorporates that model structural
uncertainty as a sampled variable — not a footnote.

---

## Governing Equations

### PDE 1 — Laplace (soil electrochemistry)
∇·(σ_soil·∇φ) = 0 in soil; Butler-Volmer BC at coating holiday

### PDE 2 — Oriani-Fick (H diffusion in steel wall)
∂C_L/∂t = D_eff·∂²C_L/∂r²;  D_eff = D_H(T)/(1+K_trap_eff)

### PDE 3 — Chen-Sutherby-Xing crack growth (most accurate available model)
**[SOURCE: Chen & Sutherby 2007; Xing et al. via Sun et al. 2021]**

    da/dN = A_CF × (K_max × ΔK² × f_eff^(-0.1))^n × HE_factor

    HE_factor = (C_H_bulk / C_H_ref)^n_HE        Xing HEDE multiplier
    f_eff = max(f, f_crit = 10^-3 Hz)            frequency saturation
    n = 2.0   [SOURCE: Chen & Sutherby 2007]
    n_HE = 0.88 [SOURCE: Sun et al. 2021, Xing model]

### Variable Amplitude Loading (Type I underload)
**[SOURCE: Chen literature via ScienceDirect Topics Ch.30 — underload VA enhancement]**

    da/dt = f_major × da/dN_major + f_minor × da/dN_minor × F_INT

    F_INT = 10  [ASSUMED: 10x underload-minor interaction — tertiary web
                 source (ScienceDirect Topics Ch.30), not the Chen 2007 journal]

### Crack Dormancy Criterion (Stage I → Stage II)
**[SOURCE: Zhao et al. 2017 — >95% cracks remain dormant below 1mm]**

    Active (Stage II) iff  K_max ≥ K_IH  AND  C_H ≥ C_H_crit ;  dormant otherwise

Because a base-metal flaw only reaches K_IH = 25 MPa√m near ~5 mm depth, the whole
sub-millimetre population is below threshold and dormant — the model reproduces
Zhao's ~95%-dormant result (a sub-mm colony returns ~95–100% dormant).

### Failure Limit State
**[SOURCE: ASME B31G surface-flaw flow-stress criterion; API 579-1 Level 2]**

    Critical if  a ≥ 0.80·t (leak)  OR  net-section collapse (Folias M_T)  OR  K ≥ K_IC

K_IH is the onset threshold for environmental cracking, **not** a rupture
criterion; a coalesced (long) flaw fails by collapse at a shallower depth.

### 3D Semi-Elliptical Crack (two coupled EDOs)
**[SOURCE: Newman & Raju 1981]**

    da/dt = v(K_A, ΔK_A)    (deepest point)
    dc/dt = v(K_C, ΔK_C)    (surface point)

### Crack Tip Acidification
**[SOURCE: Turnbull 1993]**

    pH_tip = pH_bulk − 0.6×log10(c/a) − 0.3×v_diss + buffering_correction
    C_H_corrected = C_H_0 × 10^(0.3 × Δ_pH)

### Residual Stress SIF
**[SOURCE: BS 7910:2019 Annex Q]**

    K_I_res = sqrt(π·a) × (σ_res_m × Y_m + σ_res_b × Y_b)

### ILI POD (Weibull)
**[SOURCE: PHMSA TVC; general EMAT performance]**

    POD(a) = 1 − exp(−(a/a_90)^k)    a_90 = 4mm, k = 2 [ASSUMED]

### Model Structural Uncertainty
**[SOURCE: Sun, Zhou & Kang 2021 JIPR — 39 full-scale CanmetMATERIALS tests]**

    da/dt_real = da/dt_model × ε_model
    ε_model ~ LogNormal(mean=1.06, COV=61.2%)

### BS 7910 Coalescence Rule
**[SOURCE: BS 7910:2019 Clause 7.3]**

    Merge if: gap between cracks s < min(a_i, a_j)

### Bayesian Posterior Update (particle filter)
**[SOURCE: Straub 2004 — inspection-based reliability]**

    p(θ | a_obs) ∝ p(a_obs | θ) × p(θ)

---

## Repository Structure

```
integrity-code-series-week-10_nnph_scc/
├── run_all.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── conftest.py
├── equations.html                LaTeX equations reference
├── src/
│   ├── constants.py              All parameters with [SOURCE] tags
│   ├── hydrogen_diffusion.py     Oriani-Fick PDE + C_H_surface_from_potential
│   ├── pressure_spectrum.py      Chen-Sutherby-Xing + VA loading
│   ├── microstructure.py         HAZ vs base metal zones (incl. vintage ERW)
│   ├── crack_growth.py           3D shape + residual stress + dormancy
│   ├── crack_tip_chemistry.py    Turnbull acidification
│   ├── model_uncertainty.py      Sun 2021 COV=61.2% epistemic error
│   ├── pod_ilicurve.py           ILI POD + post-inspection a0
│   ├── crack_colony.py           Full-physics colony + BS 7910 coalescence
│   ├── bayesian_update.py        Particle filter posterior
│   ├── fad_assessment.py         API 579-1 Level 2 FAD
│   ├── monte_carlo.py            8-param Monte Carlo (post-ILI)
│   ├── surrogate_gbr.py          GBR surrogate (8 features)
│   ├── audit_chain.py            SHA-256 hash-linked audit log
│   ├── cp_optimization.py        Non-monotonic CP curve
│   ├── h2_blending.py            H2 blend K_IH degradation
│   └── inspection_optimizer.py   RBI re-inspection interval
├── validation/benchmarks.py      Analytical benchmarks
├── visualization/
│   ├── plot_all.py               6 core panels + colony GIF generator
│   └── plot_advanced.py          3 extended panels (CP, H2, inspection optimizer)
├── tests/test_week10.py          215 tests
├── assets/figures/               9 panels (300 DPI)
├── assets/animations/            Crack colony GIF
└── assets/audit_chain.json
```

---

## New Mechanisms vs v1

| Mechanism | v1 | v2 | Source |
|-----------|----|----|--------|
| Crack growth model | Paris law | Chen-Sutherby-Xing combined parameter | Chen & Sutherby 2007 |
| Frequency dependence | Linear f | Saturation at f_crit = 10^-3 Hz | Xing et al. |
| VA loading | Constant amplitude | Type I underload with F_INT=10 [ASSUMED] | Chen lit. via ScienceDirect Topics Ch.30 |
| Crack dormancy | None | Stage I/II criterion, 95% dormant | Zhao et al. 2017 |
| Microstructure | Uniform | HAZ vs base, K_IH and da/dt different | Beavers et al. 2001 |
| ILI POD | LogNormal a0 | Post-inspection Weibull rejection sampling | PHMSA TVC |
| Model error | None | LogNormal(1.06, COV=61.2%) per crack | Sun et al. 2021 |
| Crack tip chemistry | None | Turnbull acidification, pH_tip(c/a) | Turnbull 1993 |
| Residual stress | None | BS 7910 Annex Q K_I_res at HAZ | BS 7910:2019 |
| Bayesian update | None | Particle filter posterior from re-ILI | Straub 2004 |
| 3D crack shape | 1D (a only) | 2D (a, c coupled EDOs) | Newman-Raju 1981 |
| Coalescence | Independent cracks | BS 7910:2019 Cl.7.3 merging rule | BS 7910:2019 |

---

## Key [ASSUMED] Parameters

| Parameter | Value | Basis |
|-----------|-------|-------|
| A_CF_BASE | 4.0×10^-14 | Calibrated to CEPA field: 0.3 mm/yr at a=2mm (c/a=4), Newman-Raju SIF |
| K_IH base | 25 MPa√m | CEPA RP 3rd Ed. range 15–35 |
| K_IH HAZ | 17.5 MPa√m | 70% of base [ASSUMED] |
| F_INT | 10 | [ASSUMED] underload interaction — Chen lit. via ScienceDirect Topics Ch.30, not the journal |
| f_crit | 10^-3 Hz | Xing et al. frequency saturation |
| a_90 (POD) | 4 mm | Modern EMAT [ASSUMED] |
| C_H_bulk X65 | 0.02 mol/m³ | Sun et al. 2021 range calibrated |

---

## Escalation Table

| Week | Topic | Key escalation |
|------|-------|---------------|
| 8 | CO2 pipeline | Supercritical EOS |
| 9 | CUI | 3-PDEs, Strang splitting |
| 10 v1 | NNpHSCC | 4-PDEs, crack colony |
| **10 v2** | **NNpHSCC full-physics** | **+12 mechanisms, Chen-Sutherby-Xing, COV=61.2% epistemic** |

---

## Cybersecurity (STRIDE)
SHA-256 hash-chain audit for all runs. Sensor integrity checks. GBR surrogate OOD fallback.

---

## How to Cite

If this software contributes to your work, please cite both the software (this repository) and the underlying methods it implements.

**Software (archived release):**

> Rocha, F. (2026). *Integrity Code Series — Week 10 — NNpHSCC Full-Physics Simulation* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20172241

**BibTeX:**

```bibtex
@software{rocha_2026_nnphscc_fullphysics,
  author       = {Rocha, Felipe},
  title        = {{Integrity Code Series --- Week 10 --- NNpHSCC
                   Full-Physics Simulation}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.20172241},
  url          = {https://doi.org/10.5281/zenodo.20172241}
}
```

The two DOIs Zenodo provides are:

| DOI                                | What it points to                                                  |
|------------------------------------|--------------------------------------------------------------------|
| [`10.5281/zenodo.20172241`](https://doi.org/10.5281/zenodo.20172241) (concept) | Always resolves to the latest version — use this for citation in papers, CV, talks. |
| [`10.5281/zenodo.20172242`](https://doi.org/10.5281/zenodo.20172242) (version) | Pinned to v1.0.1 specifically — use when reproducibility matters. |

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff) — GitHub will display a "Cite this repository" widget at the top right of the repo page that exports BibTeX / APA / RIS automatically.

---

Research tool only. Not for FFS decisions without site-specific calibration and independent review.
