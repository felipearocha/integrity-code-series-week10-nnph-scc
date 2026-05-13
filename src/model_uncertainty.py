"""
Model structural uncertainty for NNpHSCC crack growth predictions.

[SOURCE: Sun, Zhou & Kang 2021 JIPR — validated against 39 full-scale CanmetMATERIALS tests]
Best available model (Xing HEDE): mean ratio = 1.06, COV = 61.2%
=> ε_model ~ LogNormal(μ_ln, σ_ln) with:
   μ_ln = ln(1.06) - 0.5 × σ_ln²
   σ_ln = sqrt(ln(1 + 0.612²)) = 0.575

Separate epistemic uncertainty (model form error) from aleatory (parameter variability):
  da/dt_real = da/dt_model(θ) × ε_model(epistemic) × ε_param(aleatory via MC)

This quantification tells the operator: even if all parameters are exactly known,
the model prediction still has ±61% uncertainty from its functional form.
"""
import numpy as np
from src.constants import MODEL_ERROR_COV, MODEL_ERROR_MEAN


def lognormal_params_from_moments(mean: float, cov: float) -> tuple:
    """Convert mean and COV to LogNormal (mu_ln, sigma_ln) parameters."""
    sigma_ln = np.sqrt(np.log(1 + cov ** 2))
    mu_ln    = np.log(mean) - 0.5 * sigma_ln ** 2
    return mu_ln, sigma_ln


def sample_model_error(n: int, seed: int = 42,
                         mean: float = MODEL_ERROR_MEAN,
                         cov:  float = MODEL_ERROR_COV) -> np.ndarray:
    """
    Sample n realisations of ε_model ~ LogNormal(mean=1.06, COV=61.2%).
    [SOURCE: Sun, Zhou & Kang 2021]

    ε_model = 1.0 means model is exact.
    ε_model > 1.0 means model underpredicts (unsafe).
    ε_model < 1.0 means model overpredicts (conservative).
    """
    rng = np.random.default_rng(seed)
    mu_ln, sigma_ln = lognormal_params_from_moments(mean, cov)
    return rng.lognormal(mu_ln, sigma_ln, n)


def uncertainty_quantiles(mean: float = MODEL_ERROR_MEAN,
                            cov:  float = MODEL_ERROR_COV) -> dict:
    """Percentile bounds of ε_model."""
    mu_ln, sigma_ln = lognormal_params_from_moments(mean, cov)
    from scipy.stats import lognorm as _ln
    rv = _ln(s=sigma_ln, scale=np.exp(mu_ln))
    return {
        'P05': float(rv.ppf(0.05)), 'P25': float(rv.ppf(0.25)),
        'P50': float(rv.ppf(0.50)), 'P75': float(rv.ppf(0.75)),
        'P95': float(rv.ppf(0.95)),
        'mean': mean, 'cov': cov, 'source': 'Sun et al. 2021 JIPR',
    }


def separate_uncertainty_intervals(da_dt_model: float,
                                     confidence: float = 0.90) -> dict:
    """
    Given a model prediction da/dt, return the uncertainty band.
    [lower, upper] at given confidence level from ε_model distribution.
    """
    q = uncertainty_quantiles()
    lo_frac = (1.0 - confidence) / 2.0
    hi_frac = 1.0 - lo_frac
    from scipy.stats import lognorm as _ln
    mu_ln, sigma_ln = lognormal_params_from_moments(MODEL_ERROR_MEAN, MODEL_ERROR_COV)
    rv = _ln(s=sigma_ln, scale=np.exp(mu_ln))
    eps_lo = rv.ppf(lo_frac)
    eps_hi = rv.ppf(hi_frac)
    return {
        'da_dt_model': da_dt_model,
        'da_dt_lower': da_dt_model * eps_lo,
        'da_dt_upper': da_dt_model * eps_hi,
        'confidence': confidence,
        'eps_P05': q['P05'], 'eps_P95': q['P95'],
        'interpretation': (f"Model structural uncertainty (Sun 2021): at {confidence:.0%} confidence, "
                           f"true da/dt is within [{da_dt_model*eps_lo*1e3*365.25*24*3600:.4f}, "
                           f"{da_dt_model*eps_hi*1e3*365.25*24*3600:.4f}] mm/yr"),
    }
