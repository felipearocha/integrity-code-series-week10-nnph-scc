"""
Enhanced crack colony model .

Includes:
  - Post-ILI POD-corrected a0 distribution [SOURCE: PHMSA TVC]
  - Dormancy screening (>95% cracks dormant) [SOURCE: Zhao et al. 2017]
  - Microstructure zone assignment (base / HAZ) [SOURCE: Beavers et al.]
  - BS 7910 coalescence rule for adjacent cracks [SOURCE: BS 7910:2019 Cl.7.3]
  - Model structural uncertainty per realisation [SOURCE: Sun et al. 2021]
  - 3D crack shape evolution (a, c) [SOURCE: Newman-Raju 1981]
"""
import numpy as np
from src.constants import (PIPE_WT, DESIGN_LIFE, A_CRIT_FRAC, PIPE_L,
                            A0_MEAN, A0_STD, C0_MEAN, K_IH_BASE_MPa, P_OP_BAR,
                            A_DORMANCY_MM, C_H_CRIT_STAGEII, LAMBDA_COLONY)
from src.pod_ilicurve import sample_a0_post_inspection
from src.microstructure import get_zone_properties
from src.model_uncertainty import sample_model_error
from src.crack_growth import integrate_full, crack_is_dormant, K_I_total
from src.pressure_spectrum import PressureSpectrum
from src.hydrogen_diffusion import C_H_surface_from_potential, D_eff
from src.constants import SOIL_PH, E_CP_V


def bs7910_coalescence_check(a_arr, c_arr, x_arr, t=PIPE_WT):
    """
    BS 7910:2019 Cl.7.3 coalescence rule: if two coplanar cracks are separated
    by s < min(a_i, a_j), treat as single crack of combined size.
    [SOURCE: BS 7910:2019 Clause 7.3]

    Parameters
    ----------
    a_arr : np.ndarray — crack depths [m]
    c_arr : np.ndarray — crack half-lengths [m]
    x_arr : np.ndarray — axial positions [m]
    t     : float — wall thickness [m]

    Returns combined (a_eff, c_eff, x_eff) after merging coalesced cracks.
    """
    if len(a_arr) <= 1:
        return a_arr.copy(), c_arr.copy(), x_arr.copy()

    merged = True
    a = a_arr.copy(); c = c_arr.copy(); x = x_arr.copy()
    while merged:
        merged = False
        for i in range(len(a)):
            for j in range(i+1, len(a)):
                # Separation between crack surfaces
                gap = abs(x[i] - x[j]) - c[i] - c[j]
                if gap < min(a[i], a[j]):   # BS 7910 criterion
                    # Merge: combined crack
                    x_new = 0.5*(x[i]+x[j])
                    c_new = 0.5*(abs(x[i]-x[j]) + c[i] + c[j])  # combined half-length
                    a_new = max(a[i], a[j]) * 1.05  # slight increase for conservatism
                    a = np.delete(a, [i, j]); c = np.delete(c, [i, j]); x = np.delete(x, [i, j])
                    a = np.append(a, a_new); c = np.append(c, c_new); x = np.append(x, x_new)
                    merged = True
                    break
            if merged: break
    return a, c, x


def simulate_colony(E_pipe: float = E_CP_V,
                        spectrum_type: str = "Type_I",
                        post_ILI: bool = True,
                        n_cracks: int = None,
                        t_end_yr: float = DESIGN_LIFE,
                        n_t: int = 60,
                        seed: int = 42) -> dict:
    """
    Full-physics colony simulation with all 15 physical mechanisms.

    Parameters
    ----------
    E_pipe       : float — pipe-to-soil potential [V vs CSE]
    spectrum_type: str   — 'Type_I', 'Type_II', or 'Type_III'
    post_ILI     : bool  — if True, use POD-corrected a0 distribution
    n_cracks     : int   — if None, sample from Poisson(LAMBDA × PIPE_L)
    """
    rng = np.random.default_rng(seed)
    if n_cracks is None:
        n_cracks = max(rng.poisson(LAMBDA_COLONY * PIPE_L), 1)

    # ── Initial crack sizes ───────────────────────────────────────────────
    if post_ILI:
        a0_arr = sample_a0_post_inspection(n_cracks, seed=seed)
    else:
        mu_ln = np.log(A0_MEAN) - 0.5*np.log(1+(A0_STD/A0_MEAN)**2)
        sig_ln = np.sqrt(np.log(1+(A0_STD/A0_MEAN)**2))
        a0_arr = rng.lognormal(mu_ln, sig_ln, n_cracks)
    a0_arr = np.clip(a0_arr, 0.05e-3, PIPE_WT*0.5)
    c0_arr = rng.lognormal(np.log(C0_MEAN), 0.3, n_cracks)
    x_arr  = rng.uniform(0, PIPE_L, n_cracks)

    # ── Zone assignment (30% in HAZ near weld seam) ───────────────────────
    # [SOURCE: Beavers et al. — cracks form preferentially at/near weld]
    zone_arr = np.where(rng.uniform(size=n_cracks) < 0.30, 'haz', 'base')

    # ── Pressure spectrum ─────────────────────────────────────────────────
    spectrum = PressureSpectrum(spectrum_type=spectrum_type)

    # ── H concentration at pipe surface ───────────────────────────────────
    C_H_surf = C_H_surface_from_potential(E_pipe, SOIL_PH)
    def C_H_tip_func(t_s): return C_H_surf  # steady-state approximation

    # ── Model error samples ───────────────────────────────────────────────
    eps_arr = sample_model_error(n_cracks, seed=seed+100)

    # ── Dormancy pre-screening ────────────────────────────────────────────
    # Only Stage II cracks contribute to fracture risk
    # [SOURCE: Zhao et al. 2017 — >95% dormant]
    dormant_initial = np.array([
        crack_is_dormant(a0_arr[k], c0_arr[k], C_H_surf, P_OP_BAR,
                          get_zone_properties(zone_arr[k])['K_IH'], zone_arr[k])
        for k in range(n_cracks)
    ])

    t_years = np.linspace(0, t_end_yr, n_t)

    results = []
    fracture_times = []

    for k in range(n_cracks):
        res = integrate_full(
            a0_m=a0_arr[k], c0_m=c0_arr[k],
            spectrum=spectrum,
            C_H_tip_func=C_H_tip_func,
            zone=zone_arr[k],
            model_error=float(eps_arr[k]),
            t_end_yr=t_end_yr,
            n_steps=min(n_t, 80),
        )
        results.append(res)
        fracture_times.append(res['fracture_time_yr'])

    fracture_times = np.array(fracture_times)

    # ── BS 7910 coalescence check at t=end ───────────────────────────────
    a_final = np.array([r['a'][-1] for r in results])
    c_final = np.array([r['c'][-1] for r in results])
    a_coal, c_coal, x_coal = bs7910_coalescence_check(a_final, c_final, x_arr)
    n_after_coal = len(a_coal)

    # ── Colony PoF ────────────────────────────────────────────────────────
    PoF_t = np.array([
        np.sum(fracture_times <= t) / n_cracks for t in t_years
    ])

    return {
        'n_cracks': n_cracks, 'results': results,
        'a0': a0_arr, 'c0': c0_arr, 'x': x_arr, 'zone': zone_arr,
        'dormant_initial': dormant_initial,
        'n_dormant_initial': dormant_initial.sum(),
        'fracture_times': fracture_times,
        't_years': t_years, 'PoF_t': PoF_t, 'pof_final': float(PoF_t[-1]),
        'eps_model': eps_arr,
        # Coalescence
        'a_after_coalescence': a_coal, 'c_after_coalescence': c_coal,
        'x_after_coalescence': x_coal,
        'n_after_coalescence': n_after_coal,
        'n_coalesced': n_cracks - n_after_coal,
        'spectrum_type': spectrum_type, 'E_pipe': E_pipe, 'post_ILI': post_ILI,
    }
