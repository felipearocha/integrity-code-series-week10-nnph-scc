"""
Full-physics NNpHSCC crack growth — Week 10.

Implements:
1. Chen-Sutherby combined parameter [SOURCE: Chen & Sutherby 2007]
2. Xing HEDE multiplier [SOURCE: Xing et al. via Sun et al. 2021]
3. Variable amplitude loading with underload interaction [SOURCE: Chen 2007]
4. Frequency saturation at f_crit = 10^-3 Hz [SOURCE: Xing et al.]
5. Crack dormancy criterion Stage I → Stage II [SOURCE: Zhao et al. 2017]
6. 3D semi-elliptical crack shape (a, c coupled EDOs) [SOURCE: Newman-Raju 1981]
7. Residual stress K_I contribution [SOURCE: BS 7910 Annex Q]
8. Microstructure-dependent properties [SOURCE: Beavers et al. 2001]
9. Model structural uncertainty term [SOURCE: Sun, Zhou & Kang 2021 COV=61.2%]
"""
import numpy as np
from src.constants import (
    PIPE_OD, PIPE_WT, P_OP_BAR, STEEL_SMYS,
    K_TH_STAGE2, K_IH_BASE_MPa,
    A_DORMANCY_MM, C_H_CRIT_STAGEII,
    SIGMA_RES_M_FRAC, SIGMA_RES_B_FRAC,
)
from src.pressure_spectrum import (
    da_dN_Chen_Xing, da_dt_variable_amplitude, PressureSpectrum
)
from src.microstructure import get_zone_properties

SEC_PER_YR = 365.25 * 24 * 3600


# ── SIF functions ─────────────────────────────────────────────────────────

def K_I_deeppoint(a_m: float, c_m: float,
                   P_bar: float = P_OP_BAR,
                   D: float = PIPE_OD, t: float = PIPE_WT) -> float:
    """
    K_I at deepest point of semi-elliptical surface crack (Mode I).
    Newman-Raju (1981) solution for surface crack in flat plate / pipe wall.
    [SOURCE: Newman & Raju 1981 Engineering Fracture Mechanics 15(1-2)]

    K_I = sigma_h × sqrt(pi × a / Q) × F(a/t, a/c, phi=pi/2)
    Q = shape factor for semi-ellipse: Q = 1 + 1.464*(a/c)^1.65 for a/c <= 1

    Returns K_I [MPa sqrt(m)], capped at fracture.
    """
    P_MPa = P_bar * 0.1
    sigma_h = P_MPa * D / (2.0 * t)  # hoop stress [MPa]
    x = min(a_m / t, 0.97)
    ac = min(a_m / max(c_m, 1e-6), 1.0)

    # Shape factor Q (Newman-Raju)
    if ac <= 1.0:
        Q = 1.0 + 1.464 * (ac ** 1.65)
    else:
        Q = 1.0 + 1.464 * ((1.0/ac) ** 1.65)

    # Boundary correction F at phi=pi/2 (deepest point)
    # Simplified Tada flat-plate for surface crack (conservative)
    F = (1.12 - 0.231*x + 10.55*x**2 - 21.72*x**3 + 30.39*x**4)

    KI = sigma_h * np.sqrt(np.pi * a_m / Q) * F
    return float(np.clip(KI, 0.0, 1e4))


def K_I_surfacepoint(a_m: float, c_m: float,
                      P_bar: float = P_OP_BAR,
                      D: float = PIPE_OD, t: float = PIPE_WT) -> float:
    """
    K_I at surface end of semi-elliptical crack (phi=0).
    Controls surface propagation rate dc/dt.
    [SOURCE: Newman & Raju 1981]
    K at phi=0 is typically lower than at phi=pi/2 for a/c < 1.
    """
    P_MPa = P_bar * 0.1
    sigma_h = P_MPa * D / (2.0 * t)
    ac = min(a_m / max(c_m, 1e-6), 1.0)

    if ac <= 1.0:
        Q = 1.0 + 1.464 * (ac ** 1.65)
    else:
        Q = 1.0 + 1.464 * ((1.0/ac) ** 1.65)

    # At phi=0 (surface): Mx factor reduces K_I
    Mx = ac ** 0.5  # simplified Newman-Raju Mx for phi=0
    x = min(a_m / t, 0.97)
    F_surf = 0.627 + 2.084*x - 1.703*x**2  # [ASSUMED simplified]

    KI_surf = sigma_h * np.sqrt(np.pi * a_m / Q) * F_surf * Mx
    return float(np.clip(KI_surf, 0.0, 1e4))


def delta_K(a_m: float, c_m: float, P_bar: float, R: float,
             dP_frac: float = 0.15) -> float:
    """
    ΔK [MPa sqrt(m)] at deepest point for given R ratio and ΔP amplitude.
    ΔK = K_I(P_max) - K_I(P_min)
    """
    K_max = K_I_deeppoint(a_m, c_m, P_bar)
    K_min = K_I_deeppoint(a_m, c_m, P_bar * R)
    return max(K_max - K_min, 0.0)


def delta_K_surface(a_m: float, c_m: float, P_bar: float, R: float) -> float:
    K_max = K_I_surfacepoint(a_m, c_m, P_bar)
    K_min = K_I_surfacepoint(a_m, c_m, P_bar * R)
    return max(K_max - K_min, 0.0)


def K_I_residual(a_m: float, t: float = PIPE_WT,
                  sigma_res_m_frac: float = SIGMA_RES_M_FRAC,
                  sigma_res_b_frac: float = SIGMA_RES_B_FRAC) -> float:
    """
    Residual stress SIF contribution at weld toe [SOURCE: BS 7910:2019 Annex Q].
    Membrane + bending residual stress, as-welded condition.

    K_I_res = sqrt(pi × a) × (sigma_res_m × Y_m + sigma_res_b × Y_b)
    Y_m ≈ 1.12 (membrane weight function)
    Y_b ≈ 0.56 (bending weight function for surface crack)
    [ASSUMED simplified weight functions]
    """
    sigma_res_m = sigma_res_m_frac * STEEL_SMYS * 1e-6  # MPa
    sigma_res_b = sigma_res_b_frac * STEEL_SMYS * 1e-6  # MPa
    x = min(a_m / t, 0.95)
    # Membrane and bending weight functions (Shen & Glinka simplified)
    Y_m = 1.12 * (1.0 - 0.31 * x + 0.09 * x**2)
    Y_b = 0.56 * (1.0 - 0.30 * x)
    K_res = np.sqrt(np.pi * a_m) * (sigma_res_m * Y_m + sigma_res_b * Y_b)
    return float(np.clip(K_res, 0.0, 100.0))


def K_I_total(a_m: float, c_m: float, P_bar: float = P_OP_BAR,
               in_HAZ: bool = False) -> float:
    """Total K_I including residual stress (only in HAZ / weld region)."""
    K_applied = K_I_deeppoint(a_m, c_m, P_bar)
    K_res = K_I_residual(a_m) if in_HAZ else 0.0
    return float(np.clip(K_applied + K_res, 0.0, 1e4))


# ── Crack dormancy model ──────────────────────────────────────────────────

def crack_is_dormant(a_m: float, c_m: float, C_H_tip: float,
                      P_bar: float = P_OP_BAR,
                      K_IH: float = K_IH_BASE_MPa,
                      zone: str = 'base') -> bool:
    """
    Stage I → Stage II transition criterion.
    [SOURCE: Zhao et al. 2017 — dormancy at < 1mm; Stage II requires both:
       (1) ΔK > ΔK_th_Stage2  AND
       (2) C_H_bulk > C_H_crit_StageII]
    [SOURCE: Shirazi et al. 2024 — growth ceases ~1mm due to dissolution rate reduction]

    Returns True if crack is dormant (not growing in Stage II).
    """
    a_mm = a_m * 1000
    # Stage I dormancy depth criterion
    if a_mm < A_DORMANCY_MM * 0.1:          # < 0.1mm: certainly dormant
        return True

    K_max = K_I_total(a_m, c_m, P_bar, in_HAZ=(zone == 'haz'))
    dK_val = delta_K(a_m, c_m, P_bar, 0.55)

    # Both conditions must be met to escape dormancy
    dk_active = dK_val >= K_TH_STAGE2
    ch_active = C_H_tip >= C_H_CRIT_STAGEII

    return not (dk_active and ch_active)


# ── 3D crack shape evolution ──────────────────────────────────────────────

def crack_shape_evolution(a0_m: float, c0_m: float,
                           spectrum: PressureSpectrum,
                           C_H_bulk: float,
                           K_I_func, delta_K_func,
                           microstructure_factor: float = 1.0,
                           model_error: float = 1.0,
                           dt_s: float = 30*86400) -> tuple:
    """
    Advance (a, c) by dt_s seconds using coupled EDOs.

    da/dt = da_dt(K_A, ΔK_A)   deepest point
    dc/dt = da_dt(K_C, ΔK_C)   surface point

    Returns: (a_new [m], c_new [m])
    """
    K_max_A = K_I_deeppoint(a0_m, c0_m, spectrum.P_max)
    dK_A    = delta_K(a0_m, c0_m, spectrum.P_max, spectrum.R_major())
    K_max_C = K_I_surfacepoint(a0_m, c0_m, spectrum.P_max)
    dK_C    = delta_K_surface(a0_m, c0_m, spectrum.P_max, spectrum.R_major())

    da_dN_A = da_dN_Chen_Xing(a0_m, K_max_A, dK_A, spectrum.f_major,
                                C_H_bulk, microstructure_factor=microstructure_factor,
                                model_error=model_error)
    da_dN_C = da_dN_Chen_Xing(a0_m, K_max_C, dK_C, spectrum.f_minor,
                                C_H_bulk, microstructure_factor=microstructure_factor,
                                model_error=model_error)

    # Type I underload: apply interaction to depth direction too
    da_dt_A = spectrum.f_major * da_dN_A + spectrum.f_minor * da_dN_A * 0.5
    dc_dt_C = spectrum.f_major * da_dN_C + spectrum.f_minor * da_dN_C * 0.5

    a_new = min(a0_m + da_dt_A * dt_s, PIPE_WT * 0.98)
    c_new = c0_m + dc_dt_C * dt_s
    return float(max(a_new, a0_m)), float(max(c_new, c0_m))


# ── Full crack growth integrator ──────────────────────────────────────────

def integrate_full(a0_m: float, c0_m: float,
                    spectrum: PressureSpectrum,
                    C_H_tip_func,     # callable(t_s) → C_H [mol/m^3]
                    zone: str = 'base',
                    model_error: float = 1.0,
                    t_end_yr: float = 20.0,
                    n_steps: int = 500,
                    P_bar: float = P_OP_BAR) -> dict:
    """
    Full NNpHSCC growth integration: 3D shape + VA loading + dormancy + residual stress.
    """
    props = get_zone_properties(zone)
    K_IH  = props['K_IH']
    f_micro = props['da_dt_factor']
    C_H_bulk = props['C_H_bulk']

    dt_s = t_end_yr * SEC_PER_YR / n_steps
    t_arr   = np.zeros(n_steps + 1)
    a_arr   = np.zeros(n_steps + 1)
    c_arr   = np.zeros(n_steps + 1)
    KI_arr  = np.zeros(n_steps + 1)
    ac_arr  = np.zeros(n_steps + 1)  # a/c aspect ratio
    dormant_arr = np.zeros(n_steps + 1, dtype=bool)
    v_arr   = np.zeros(n_steps + 1)

    a, c = float(a0_m), float(c0_m)
    fracture_time_yr = np.inf

    for k in range(n_steps + 1):
        t_s = k * dt_s
        t_arr[k] = t_s / SEC_PER_YR
        a_arr[k], c_arr[k] = a, c
        C_H_tip = C_H_tip_func(t_s)
        KI_arr[k] = K_I_total(a, c, P_bar, in_HAZ=(zone == 'haz'))
        ac_arr[k]  = a / max(c, 1e-6)
        dormant_arr[k] = crack_is_dormant(a, c, C_H_tip, P_bar, K_IH, zone)

        # VA crack growth
        da_dt_val = da_dt_variable_amplitude(
            a, spectrum, C_H_bulk,
            K_I_func=lambda am, P: K_I_deeppoint(am, c, P),
            delta_K_func=lambda am, P, R: delta_K(am, c, P, R),
            microstructure_factor=f_micro,
            model_error=model_error,
            spectrum_type=spectrum.type
        )
        v_arr[k] = da_dt_val

        # Check fracture
        if KI_arr[k] >= K_IH and fracture_time_yr == np.inf:
            fracture_time_yr = t_s / SEC_PER_YR

        if k < n_steps:
            if dormant_arr[k]:
                # Stage I slow dissolution only — 10% of Stage II rate [ASSUMED]
                da_dt_s1 = da_dt_val * 0.10
            else:
                da_dt_s1 = da_dt_val

            a, c = crack_shape_evolution(a, c, spectrum, C_H_bulk,
                                          K_I_func=lambda am, P: K_I_deeppoint(am, c, P),
                                          delta_K_func=lambda am, P, R: delta_K(am, c, P, R),
                                          microstructure_factor=f_micro,
                                          model_error=model_error,
                                          dt_s=dt_s)

    return {
        't_yr': t_arr, 'a': a_arr, 'c': c_arr, 'KI': KI_arr,
        'ac_ratio': ac_arr, 'dormant': dormant_arr, 'v': v_arr,
        'fracture_time_yr': fracture_time_yr,
        'a0': a0_m, 'c0': c0_m, 'zone': zone,
        'pipe_wt': PIPE_WT, 'K_IH': K_IH,
    }
