"""
Monte Carlo v2 — full-physics NNpHSCC with ALL gap closures.

8 sampled parameters (expanded from 6):
  E_pipe      : V vs CSE           Normal(-0.85, 0.05)
  C_CO2       : mol/L              LogNormal(ln 0.05, 0.5)
  a0_mm       : mm                 post-ILI POD-corrected LogNormal
  zone_frac   : fraction in HAZ    Uniform(0, 0.5)  → assigns HAZ vs base
  spectrum    : VA type            Categorical (Type_I: 0.6, II: 0.3, III: 0.1)
  eps_model   : model error        LogNormal(1.06, COV=61.2%)  [SOURCE: Sun 2021]
  C_H_mult    : H entry multiplier Normal(1.0, 0.3)
  f_int       : underload factor   Uniform(5, 15)  [SOURCE: Chen 2007]

Separates aleatory (parameter) from epistemic (model form) uncertainty.
"""
import numpy as np
from scipy.stats import norm, lognorm, uniform as sp_uni, spearmanr
from src.constants import (E_CP_V, C_CO2_MOL, A0_MEAN, PIPE_WT, DESIGN_LIFE,
                            P_OP_BAR, SOIL_PH, K_IH_BASE_MPa, K_IH_HAZ_MPa)
from src.crack_growth import K_I_deeppoint, delta_K, crack_is_dormant
from src.pressure_spectrum import da_dt_variable_amplitude, PressureSpectrum, da_dN_Chen_Xing
from src.model_uncertainty import sample_model_error, MODEL_ERROR_MEAN, MODEL_ERROR_COV
from src.pod_ilicurve import sample_a0_post_inspection
from src.microstructure import get_zone_properties
from src.crack_tip_chemistry import C_H_entry_corrected, crack_tip_pH
from src.hydrogen_diffusion import C_H_surface_from_potential


SEC_PER_YR = 365.25 * 24 * 3600
PARAM_NAMES = ["E_pipe", "C_CO2", "a0_mm", "zone_HAZ_frac",
               "spectrum_idx", "eps_model", "C_H_mult", "f_int"]


def latin_hypercube_sample(n, d, seed=42):
    rng = np.random.default_rng(seed)
    u = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n); u[:,j] = (perm + rng.uniform(size=n)) / n
    return u


def _u_to_params(u, a0_post_ILI=None):
    N = u.shape[0]
    E_pipe      = norm.ppf(u[:,0], loc=E_CP_V, scale=0.05)
    C_CO2       = np.clip(lognorm.ppf(u[:,1], s=0.5, scale=C_CO2_MOL), 0.001, 0.5)
    if a0_post_ILI is not None:
        # Use pre-sampled POD-corrected a0
        idx = (u[:,2] * len(a0_post_ILI)).astype(int).clip(0, len(a0_post_ILI)-1)
        a0_mm = a0_post_ILI[idx] * 1000
    else:
        a0_mm = np.clip(lognorm.ppf(u[:,2], s=0.5, scale=A0_MEAN*1000), 0.05, PIPE_WT*1000*0.5)
    zone_frac   = sp_uni.ppf(u[:,3], loc=0, scale=0.5)
    spectrum_idx= (u[:,4] * 3).astype(int).clip(0, 2)  # 0=TypeI, 1=TypeII, 2=TypeIII
    eps_model   = _sample_eps(N, u[:,5])
    C_H_mult    = np.clip(norm.ppf(u[:,6], loc=1.0, scale=0.3), 0.1, 3.0)
    f_int       = sp_uni.ppf(u[:,7], loc=5, scale=10)  # underload factor [5, 15]
    return {"E_pipe": E_pipe, "C_CO2": C_CO2, "a0_mm": a0_mm,
            "zone_HAZ_frac": zone_frac, "spectrum_idx": spectrum_idx,
            "eps_model": eps_model, "C_H_mult": C_H_mult, "f_int": f_int}


def _sample_eps(N, u_col):
    """Sample model error from LogNormal via inverse CDF."""
    import numpy as np
    from scipy.stats import lognorm as _ln
    from src.model_uncertainty import lognormal_params_from_moments
    mu_ln, sig_ln = lognormal_params_from_moments(MODEL_ERROR_MEAN, MODEL_ERROR_COV)
    return _ln.ppf(u_col, s=sig_ln, scale=np.exp(mu_ln))


def _da_dt_fast(E_pipe_k, C_CO2_k, a0_mm_k, zone_frac_k, spec_idx_k,
                 eps_k, C_H_mult_k, f_int_k, t_yr):
    """Fast crack growth for MC inner loop — fixed v at initial conditions."""
    zone = 'haz' if zone_frac_k > 0.30 else 'base'
    props = get_zone_properties(zone)
    K_IH = props['K_IH']

    a0 = a0_mm_k * 1e-3; c0 = max(a0 * 3, 2e-3)
    C_H_0 = C_H_surface_from_potential(E_pipe_k, SOIL_PH) * C_H_mult_k
    pH_tip = crack_tip_pH(a0, c0, SOIL_PH)
    C_H_tip = C_H_entry_corrected(C_H_0, pH_tip, SOIL_PH)

    # Check dormancy
    if crack_is_dormant(a0, c0, C_H_tip, P_OP_BAR, K_IH, zone):
        # Stage I: very slow dissolution only (10% of Stage II rate)
        dormancy_factor = 0.10
    else:
        dormancy_factor = 1.0

    spec_types = ["Type_I", "Type_II", "Type_III"]
    sp_type = spec_types[int(spec_idx_k)]
    spectrum = PressureSpectrum(spectrum_type=sp_type)

    K_max = K_I_deeppoint(a0, c0, P_OP_BAR)
    dK_val = delta_K(a0, c0, P_OP_BAR, 0.55)

    from src.pressure_spectrum import da_dN_Chen_Xing as _da, f_eff, N_MINOR_PER_MAJOR
    da_dN_major = _da(a0, K_max, dK_val, spectrum.f_major, props['C_H_bulk'],
                       microstructure_factor=props['da_dt_factor'], model_error=eps_k)
    da_dN_minor = _da(a0, K_max, dK_val*0.1, spectrum.f_minor, props['C_H_bulk'],
                       microstructure_factor=props['da_dt_factor'], model_error=eps_k)

    if sp_type == "Type_I":
        da_dt = (spectrum.f_major * da_dN_major +
                 spectrum.f_minor * da_dN_minor * f_int_k)
    elif sp_type == "Type_II":
        da_dt = spectrum.f_major * da_dN_major
    else:
        da_dt = (spectrum.f_major * da_dN_major +
                 spectrum.f_minor * da_dN_minor * f_int_k * 0.5)

    da_dt *= dormancy_factor
    a_final = a0 + da_dt * t_yr * SEC_PER_YR
    return float(min(a_final * 1000, PIPE_WT * 1000))   # mm


def run_monte_carlo(n_samples=10_000, t_assess_yr=None, n_t=30,
                        seed=42, post_ILI=True):
    if t_assess_yr is None: t_assess_yr = DESIGN_LIFE

    # Pre-sample POD-corrected a0 if requested
    if post_ILI:
        a0_pool = sample_a0_post_inspection(n_samples * 3, seed=seed)
    else:
        a0_pool = None

    u = latin_hypercube_sample(n_samples, 8, seed)
    params = _u_to_params(u, a0_post_ILI=a0_pool)
    t_years = np.linspace(0, t_assess_yr, n_t)

    # Failure criterion: a >= 3.3mm (K_IH crossing)
    a_limit_mm = 3.3
    wall_loss_final = np.zeros(n_samples)
    censored = np.zeros(n_samples, dtype=bool)
    PoF_t = np.zeros(n_t)

    for k in range(n_samples):
        wl_traj = np.array([
            _da_dt_fast(params["E_pipe"][k], params["C_CO2"][k], params["a0_mm"][k],
                         params["zone_HAZ_frac"][k], params["spectrum_idx"][k],
                         params["eps_model"][k], params["C_H_mult"][k],
                         params["f_int"][k], t)
            for t in t_years
        ])
        wall_loss_final[k] = wl_traj[-1]
        if wl_traj[-1] >= PIPE_WT * 1000:
            censored[k] = True
        PoF_t += (wl_traj >= a_limit_mm).astype(float)

    PoF_t /= n_samples
    return {"params": params, "wall_loss": wall_loss_final,
            "censored": censored, "PoF_t": PoF_t, "t_years": t_years,
            "pof_final": float(PoF_t[-1]), "n_samples": n_samples,
            "post_ILI": post_ILI}


def spearman_sensitivity(mc):
    wl = mc["wall_loss"]
    rho = {}
    for name in PARAM_NAMES:
        arr = mc["params"].get(name)
        if arr is not None:
            rho[name] = float(spearmanr(arr, wl)[0])
    return dict(sorted(rho.items(), key=lambda x: abs(x[1]), reverse=True))
