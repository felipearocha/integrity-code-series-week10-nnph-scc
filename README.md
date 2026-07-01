# Integrity Code Series — Week 10 — NNpHSCC Full-Physics Simulation

[![CI](https://github.com/felipearocha/integrity-code-series-week10-nnph-scc/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week10-nnph-scc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 215 passing](https://img.shields.io/badge/tests-215%20passing-brightgreen.svg)](tests)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172241.svg)](https://doi.org/10.5281/zenodo.20172241)

**Author:** Felipe Rocha — Independent Researcher

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| Week 3 | [integrity-code-series-week3-f1-lap-simulation](https://github.com/felipearocha/integrity-code-series-week3-f1-lap-simulation) | F1 lap simulation (six coupled ODEs) |
| Week 6 | [integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| Week 7 | [integrity-code-series-week7-h2-lferw](https://github.com/felipearocha/integrity-code-series-week7-h2-lferw) | LF-ERW H2 conversion (B31.12 + NACE TM0316) |
| Week 8 | [integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater) | Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) |
| Week 9 | [integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui) | CUI thermohygro-electrochemical (3 PDEs, Strang) |
| **Week 10** | **[integrity-code-series-week10-nnph-scc](https://github.com/felipearocha/integrity-code-series-week10-nnph-scc)** | **NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) — this repo** |
| Week 11 | [integrity-code-series-week11-erosion-corrosion-multiphase](https://github.com/felipearocha/integrity-code-series-week11-erosion-corrosion-multiphase) | Erosion-corrosion multiphase (NORSOK M-506 + DNV-RP-O501 + G119 + API 579) |
| Bonus | [Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation](https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation) | Vibration-accelerated corrosion (SDOF + Butler-Volmer + Archard) |
| Bonus | [synthetic-integrity-digital-twin-piml](https://github.com/felipearocha/synthetic-integrity-digital-twin-piml) | Physics-informed neural-network surrogate |
| Bonus | [integrity-data-foundation](https://github.com/felipearocha/integrity-data-foundation) | Engineering data validation baseline |

## Quick Start

```bash
# Clone and install
git clone https://github.com/felipearocha/integrity-code-series-week10-nnph-scc.git
cd integrity-code-series-week10-nnph-scc
pip install -e .[dev]

# Validate
pytest tests/ -q                          # 215 tests
python -m validation.benchmarks           # textbook constants

# Reproduce all panels + audit-chained outputs
python run_all.py
```

Browse [`docs/equations.html`](docs/equations.html) in any modern browser for the full LaTeX equations reference.

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

[**view the full rendered reference**](https://htmlpreview.github.io/?https://github.com/felipearocha/integrity-code-series-week10-nnph-scc/blob/main/docs/equations.html)

Every constant is tagged to its source standard or paper. The headline equations below
render natively on GitHub; the complete derivation set (12 sections) lives in
**[docs/equations.html](docs/equations.html)**.

### Chen-Sutherby-Xing crack growth (most accurate available model)
**[SOURCE: Chen & Sutherby 2007; Xing et al. via Sun et al. 2021]**

$$ \frac{da}{dN} \;=\; A_{\text{CF}}\,\bigl(K_{\max}\,\Delta K^{\,2}\,f_{\text{eff}}^{-0.1}\bigr)^{n}\;\Phi_{\text{HE}}(C_H) $$

HEDE (Xing) multiplier, with $n_{\text{HE}} = 0.88$ and frequency saturation $f_{\text{eff}} = \max(f, f_{\text{crit}})$:

$$ \Phi_{\text{HE}}(C_H) \;=\; \Bigl(\tfrac{C_H^{\,\text{bulk}}}{C_H^{\,\text{ref}}}\Bigr)^{n_{\text{HE}}}, \qquad n_{\text{HE}} = 0.88 $$

### Oriani-Fick — hydrogen diffusion in the steel wall (with trapping)
**[SOURCE: Kiuchi & McLellan 1983; Turnbull 1996; San Marchi 2012]**

$$ \frac{\partial C_L}{\partial t} \;=\; D_{\text{eff}}(T,\sigma_h)\,\frac{\partial^{2} C_L}{\partial r^{2}} + \frac{D_{\text{eff}}\,V_H}{R\,T}\,\frac{\partial}{\partial r}\!\Bigl(C_L\,\frac{\partial \sigma_h}{\partial r}\Bigr) $$

### Stress Intensity Factor — Newman-Raju surface crack
**[SOURCE: Newman & Raju 1981]**

$$ K_I^{\,(A)} \;=\; \sigma_h\,\sqrt{\frac{\pi\,a}{Q}}\;F\!\Bigl(\tfrac{a}{t},\tfrac{a}{c},\,\tfrac{\pi}{2}\Bigr) $$

### Crack-tip acidification (Turnbull)
**[SOURCE: Turnbull 1993]**

$$ \text{pH}_{\text{tip}} \;=\; \text{pH}_{\text{bulk}} \;-\; 0.6\,\log_{10}\!\Bigl(\tfrac{c}{a}\Bigr) \;-\; 0.3\,v_{\text{diss}} \;+\; \text{buffer correction} $$

### Model structural uncertainty (headline v2 finding)
**[SOURCE: Sun, Zhou & Kang 2021 JIPR — 39 full-scale CanmetMATERIALS tests]**

$$ \left(\tfrac{da}{dt}\right)_{\!\text{real}} \;=\; \left(\tfrac{da}{dt}\right)_{\!\text{model}}\!\!\cdot\,\varepsilon_{\text{model}},\qquad \varepsilon_{\text{model}} \sim \operatorname{LogNormal}(\mu=1.06,\;\mathrm{COV}=61.2\%) $$

### API 579-1 Level 2 FAD curve
**[SOURCE: API 579-1/ASME FFS-1 Level 2]**

$$ K_r(L_r) \;=\; \bigl(1 + \tfrac{1}{2} L_r^{2}\bigr)^{-1/2}\, \bigl(\,0.3 + 0.7\,e^{-0.65\,L_r^{6}}\bigr) $$

### H₂ blending — K_IH degradation
**[SOURCE: Cui et al. 2024]**

$$ K_{IH}(x_{\text{H}_2},x_{\text{CO}_2}) \;=\; K_{IH,0}\,\bigl(1 - \alpha_{\text{H}}\,x_{\text{H}_2}^{\,m}\bigr)\,\bigl(1 + \alpha_{\text{CO}_2}\,x_{\text{CO}_2}\bigr)^{-1} $$

---

## Repository Structure

```
integrity-code-series-week10-nnph-scc/
├── run_all.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── conftest.py
├── docs/equations.html           LaTeX equations reference (MathJax)
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
├── tests/                        215 tests (test_week10.py + test_physics_correctness.py)
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

## Anti-Hallucination Note

Every equation and constant carries an explicit `[SOURCE: ...]` or `[ASSUMED]` tag,
graded by tier:

- **T1 (standard / peer-reviewed paper)** — read directly from a controlled source:
  Chen & Sutherby 2007 (combined-parameter crack growth, n = 2.0), Sun, Zhou & Kang
  2021 JIPR (model error LogNormal mean 1.06, COV = 61.2% from 39 CanmetMATERIALS
  tests; n_HE = 0.88), Zhao et al. 2017 (>95% sub-mm dormancy), Turnbull 1993
  (crack-tip acidification), Newman & Raju 1981 (semi-elliptical SIF), BS 7910:2019
  (Annex Q residual-stress SIF, Cl. 7.3 coalescence), ASME B31G / API 579-1 Level 2
  (failure limit state), Straub 2004 (Bayesian inspection update).
- **T2 (derived)** — quantities computed from T1 inputs (e.g. `D_eff` from `D_H(T)`
  and trap density, `pH_tip` propagated into `C_H_corrected`, post-inspection `a0`
  from the Weibull POD by rejection sampling).
- **T3 (practitioner / assumed)** — modelling choices that no standard fixes, each
  flagged `[ASSUMED]` in the README and source: the underload interaction factor
  `F_INT = 10` (tertiary web source, not the Chen 2007 journal), the ILI POD
  parameters `a_90 = 4 mm`, `k = 2`, the HAZ threshold `K_IH = 17.5 MPa√m` (70% of
  base), and the base growth calibration `A_CF_BASE = 4.0×10^-14` (fitted to CEPA
  field rate 0.3 mm/yr at a = 2 mm).

The tiers are applied honestly: where a coefficient is an assumption rather than a
codified value it is tagged `[ASSUMED]` (T3) and listed in the "Key [ASSUMED]
Parameters" table above, not presented as a standard constant.

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

## Disclaimer

Research tool only. Not for design, fitness-for-service, or safety-critical decisions without site-specific calibration and independent PE review.

## License

MIT — Felipe Rocha. See [LICENSE](LICENSE).
