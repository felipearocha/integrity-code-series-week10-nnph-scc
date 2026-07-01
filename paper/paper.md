---
title: 'NNpHSCC Full-Physics: A Python framework for full-physics simulation of near-neutral pH stress corrosion cracking in gas transmission pipelines with explicit model-structural uncertainty'
tags:
  - Python
  - pipeline integrity
  - stress corrosion cracking
  - hydrogen embrittlement
  - cathodic protection
  - fracture mechanics
  - Bayesian inference
  - Monte Carlo
  - fitness-for-service
  - PHMSA
authors:
  - name: Felipe Rocha
    # orcid: 0000-0000-0000-0000   # TODO: insert real ORCID before JOSS submission
    affiliation: 1
affiliations:
  - name: Independent Researcher, Calgary, Canada
    index: 1
date: 13 May 2026
bibliography: paper.bib
---

# Summary

Near-Neutral pH Stress Corrosion Cracking (NNpHSCC) accounts for a large fraction of
reportable incidents on gas transmission pipelines in North America, yet existing
open-source tools either reduce the problem to a single-equation Paris-law surrogate
or are wrapped inside proprietary fitness-for-service software. `NNpHSCC Full-Physics`
is a Python framework that resolves the problem end-to-end as a coupled physical
system: a Laplace equation for soil electrochemistry with Butler–Volmer boundary
conditions at coating holidays, an Oriani–Fick hydrogen-diffusion equation with
trap-corrected effective diffusivity through the steel wall, and the
Chen–Sutherby–Xing combined-parameter crack-growth law with frequency saturation
and Type-I underload variable-amplitude interaction
[@chen_sutherby_2007; @sun_zhou_kang_2021]. Crack geometry evolves as a
Newman–Raju semi-elliptical surface flaw integrating two coupled ODEs for the
deepest and surface points [@newman_raju_1981]. Residual stress at the heat-affected
zone follows BS 7910:2019 Annex Q, and adjacent flaws are merged using the
BS 7910:2019 Cl. 7.3 coalescence rule [@bs7910_2019]. Crack-tip pH is corrected
using the @turnbull_1993 acidification model, and the @zhao_2017 Stage-I dormancy
criterion separates dormant from propagating colonies. Latin Hypercube Monte Carlo
sampling propagates eight uncertain parameters and — critically — samples the
@sun_zhou_kang_2021 model structural error (lognormal, mean = 1.06,
COV = 61.2 %) as an explicit epistemic variable rather than a footnote.
A Straub-style particle filter [@straub_2004] performs Bayesian posterior updates
from in-line inspection events, and a SHA-256 hash-chained audit log records every
run for traceability. The package ships with 215 automated tests, a CI workflow,
analytical benchmarks against textbook constants, and a Zenodo-archived release
[@rocha_zenodo_2026].

# Statement of Need

PHMSA §192.611(a)(4), effective March 16, 2026, requires US gas-transmission
operators to perform integrity assessments when class location changes occur, and
NNpHSCC is the dominant time-dependent threat that those assessments must address
in legacy pre-1970s pipelines [@phmsa_2024_192611]. The 2018 Prince George (BC)
914 mm rupture and the 2015 Unityville (PA) 609 mm rupture were both attributed
to NNpHSCC, and both occurred downstream of compressor stations where pressure
spectra produce exactly the variable-amplitude underload conditions captured by
the Chen–Sutherby model [@sun_zhou_kang_2021]. Operators preparing §192.611
assessments need a transparent physics-based tool that (i) implements the most
predictive crack-growth law currently available rather than a calibrated Paris
fit, (ii) propagates the substantial epistemic error of that law to the integrity
decision, and (iii) supports Bayesian re-inspection planning under the resulting
posterior distribution.

To the authors' knowledge, no openly licensed framework currently combines all of
these capabilities. Commercial fitness-for-service suites such as those used for
API 579-1 Level 2 assessments [@api579_2021] treat NNpHSCC growth as a constant
input rate, not as an emergent quantity from coupled electrochemistry, hydrogen
transport and fracture mechanics. Research codes published alongside individual
papers typically isolate one mechanism — Laplace solvers without crack growth,
or crack-growth laws without crack-tip chemistry. `NNpHSCC Full-Physics` closes
that gap with a single audit-chained pipeline that exposes every parameter with
a `[SOURCE]` citation tag and every assumption with an `[ASSUMED]` flag, allowing
reviewers, regulators and operators to trace each numerical result back to either
a peer-reviewed source or an explicitly declared engineering judgment.

The framework is intended for three audiences. Pipeline-integrity engineers can
use the Monte Carlo and Bayesian update modules to scope re-inspection intervals
under the COV = 61.2 % epistemic uncertainty band rather than under deterministic
point estimates. Researchers can use the modular `src/` package to substitute
alternative crack-growth laws, hydrogen-diffusion models or POD curves and
compare the integrated outcome on identical Monte Carlo seeds. Graduate students
in corrosion and fracture mechanics can use the LaTeX equation reference in
`docs/equations.html`, the analytical benchmarks in `validation/benchmarks.py`, and
the 215 unit tests as a worked example of how the standards (BS 7910:2019,
API 579-1, API 571, NACE TM0316) compose into an end-to-end integrity model.

# Acknowledgements

This work was carried out without external funding, as part of the ongoing
Integrity Code Series of open physics-first integrity simulators. The author
acknowledges the public availability of the CanmetMATERIALS full-scale NNpHSCC
test dataset that anchors the model-uncertainty calibration of
@sun_zhou_kang_2021, without which the explicit COV = 61.2 % epistemic term
in this framework would not be possible.

# References
