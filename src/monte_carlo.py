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
from src.crack_growth import (K_I_deeppoint, K_I_surfacepoint, delta_K,
                              delta_K_surface, crack_is_dormant, flaw_is_critical,
                              K_I_residual)
from src.cp_optimization import CGR_factor_from_potential
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


def _integrate_sample(E_pipe_k, C_CO2_k, a0_mm_k, zone_frac_k, spec_idx_k,
                       eps_k, C_H_mult_k, f_int_k, t_years):
    """
    Proper per-sample crack-growth integration (replaces the old frozen-rate
    linear extrapolation). The depth rate is re-evaluated at the current depth
    on every step, dormancy is applied, and the sample FAILS via the unified
    limit state flaw_is_critical (leak at 80% wall / net-section collapse / K_IC)
    — NOT at K_IH, which is only the environmental-cracking onset threshold.
    Potential and CO2 both modulate the rate (see below).

    Returns (a_traj_mm[len(t_years)], failed, t_fail_yr).
    """
    zone = 'haz' if zone_frac_k > 0.30 else 'base'
    props = get_zone_properties(zone)
    K_IH = props['K_IH']
    f_micro = props['da_dt_factor']
    spec_types = ["Type_I", "Type_II", "Type_III"]
    spectrum = PressureSpectrum(spectrum_type=spec_types[int(spec_idx_k)])
    C_H_0 = C_H_surface_from_potential(E_pipe_k, SOIL_PH) * C_H_mult_k

    # Potential and CO2 modulate the growth rate, anchored so that at the NACE
    # reference (E_CP = -0.85 V, reference CO2) both factors are 1.0 and the
    # base-metal calibration is preserved. CP uses the non-monotonic NNpHSCC CGR
    # curve (minimum near -0.75 V), so PoF genuinely responds to potential and
    # -0.75 V minimises it; higher CO2 weakens the crack-tip buffer.
    pot_factor = CGR_factor_from_potential(E_pipe_k) / CGR_factor_from_potential(E_CP_V)
    pH_co2 = crack_tip_pH(1e-3, 3e-3, SOIL_PH, c_co2=C_CO2_k)
    pH_ref = crack_tip_pH(1e-3, 3e-3, SOIL_PH, c_co2=C_CO2_MOL)
    co2_factor = 10.0 ** ((pH_ref - pH_co2) * 0.3)
    C_H_for_rate = props['C_H_bulk'] * max(C_H_mult_k, 0.0) * pot_factor * co2_factor
    in_haz = (zone == 'haz')

    a = a0_mm_k * 1e-3
    c = max(a * 3, 2e-3)
    n = len(t_years)
    a_traj = np.empty(n)
    failed = False
    t_fail = np.inf

    for i in range(n):
        a_traj[i] = a * 1000.0
        if flaw_is_critical(a, c, P_OP_BAR, in_HAZ=in_haz):
            failed = True
            t_fail = t_years[i]
            a_traj[i:] = a * 1000.0   # frozen at failure
            break
        if i < n - 1:
            dt_s = (t_years[i+1] - t_years[i]) * SEC_PER_YR
            pH_tip = crack_tip_pH(a, c, SOIL_PH, c_co2=C_CO2_k)
            C_H_tip = C_H_entry_corrected(C_H_0, pH_tip, SOIL_PH)
            df = 0.10 if crack_is_dormant(a, c, C_H_tip, P_OP_BAR, K_IH, zone) else 1.0
            da_dt = da_dt_variable_amplitude(
                a, spectrum, C_H_for_rate,
                K_I_func=lambda am, P: K_I_deeppoint(am, c, P)
                          + (K_I_residual(am) if in_haz else 0.0),
                delta_K_func=lambda am, P, R: delta_K(am, c, P, R),
                microstructure_factor=f_micro, model_error=eps_k,
                interaction_factor=f_int_k, spectrum_type=spectrum.type)
            dc_dt = da_dt_variable_amplitude(
                a, spectrum, C_H_for_rate,
                K_I_func=lambda am, P: K_I_surfacepoint(am, c, P),
                delta_K_func=lambda am, P, R: delta_K_surface(am, c, P, R),
                microstructure_factor=f_micro, model_error=eps_k,
                interaction_factor=f_int_k, spectrum_type=spectrum.type)
            a = min(a + df * da_dt * dt_s, PIPE_WT * 0.98)
            c = c + df * dc_dt * dt_s
    return a_traj, failed, t_fail


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

    wall_loss_final = np.zeros(n_samples)
    censored = np.zeros(n_samples, dtype=bool)   # reached the wall thickness
    failed_by_t = np.zeros((n_samples, n_t), dtype=bool)

    for k in range(n_samples):
        a_traj, failed, t_fail = _integrate_sample(
            params["E_pipe"][k], params["C_CO2"][k], params["a0_mm"][k],
            params["zone_HAZ_frac"][k], params["spectrum_idx"][k],
            params["eps_model"][k], params["C_H_mult"][k], params["f_int"][k],
            t_years)
        wall_loss_final[k] = a_traj[-1]
        if a_traj[-1] >= PIPE_WT * 1000 * 0.98:
            censored[k] = True
        if failed:
            # cumulative failure: counts towards PoF at every t >= t_fail
            failed_by_t[k] = t_years >= t_fail

    # PoF(t) = fraction of samples that have reached K_IH (rupture) by time t
    PoF_t = failed_by_t.mean(axis=0)
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
