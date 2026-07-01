"""
Full-physics NNpHSCC crack growth — Week 10.

Implements:
1. Chen-Sutherby combined parameter [SOURCE: Chen & Sutherby 2007]
2. Xing HEDE multiplier [SOURCE: Xing et al. via Sun et al. 2021]
3. Variable amplitude loading with underload interaction [SOURCE: Chen 2007]
4. Frequency saturation at f_crit = 10^-3 Hz [SOURCE: Xing model via Sun, Zhou & Kang 2021]
5. Crack dormancy criterion Stage I → Stage II [SOURCE: Zhao et al. 2017]
6. 3D semi-elliptical crack shape (a, c coupled EDOs) [SOURCE: Newman-Raju 1981]
7. Residual stress K_I contribution [SOURCE: BS 7910 Annex Q]
8. Microstructure-dependent properties [SOURCE: Beavers et al. 2001]
9. Model structural uncertainty term [SOURCE: Sun, Zhou & Kang 2021 COV=61.2%]
"""
import numpy as np
from src.constants import (
    PIPE_OD, PIPE_WT, P_OP_BAR, STEEL_SMYS, STEEL_UTS,
    K_IH_BASE_MPa, K_IC_AIR, A_CRIT_FRAC,
    C_H_CRIT_STAGEII,
    SIGMA_RES_M_FRAC, SIGMA_RES_B_FRAC,
)
from src.pressure_spectrum import (
    da_dN_Chen_Xing, da_dt_variable_amplitude, PressureSpectrum
)
from src.microstructure import get_zone_properties

SEC_PER_YR = 365.25 * 24 * 3600


# ── SIF functions ─────────────────────────────────────────────────────────

def _newman_raju_M(ac: float) -> tuple:
    """Newman-Raju (1981) geometry coefficients M1, M2, M3 for a/c <= 1."""
    M1 = 1.13 - 0.09 * ac
    M2 = -0.54 + 0.89 / (0.2 + ac)
    M3 = 0.5 - 1.0 / (0.65 + ac) + 14.0 * (1.0 - ac) ** 24
    return M1, M2, M3


def K_I_deeppoint(a_m: float, c_m: float,
                   P_bar: float = P_OP_BAR,
                   D: float = PIPE_OD, t: float = PIPE_WT) -> float:
    """
    K_I at the deepest point (phi = pi/2) of a semi-elliptical surface crack (Mode I).
    Newman & Raju (1981) boundary-correction factor for a surface crack in a
    finite-thickness plate under membrane (hoop) loading.
    [SOURCE: Newman & Raju 1981, Eng. Fract. Mech. 15(1-2):185-192]

    K_I = sigma_h * sqrt(pi * a / Q) * F(a/t, a/c, phi=pi/2)
    Q   = 1 + 1.464 (a/c)^1.65                              (a/c <= 1)
    F   = M1 + M2 (a/t)^2 + M3 (a/t)^4    with g = f_phi = f_w = 1 at the deep point.

    (The previous release used a Tada single-edge-crack front-face polynomial here,
    which over-predicted K by up to ~2x for a/t > 0.5 and was mislabelled
    Newman-Raju; this is the true surface-crack solution. f_w ~ 1 for a pipe wall
    where the circumference >> 2c.)

    Returns K_I [MPa sqrt(m)].
    """
    P_MPa = P_bar * 0.1
    sigma_h = P_MPa * D / (2.0 * t)  # hoop stress [MPa]
    x = min(a_m / t, 0.95)
    ac = min(a_m / max(c_m, 1e-6), 1.0)

    Q = 1.0 + 1.464 * (ac ** 1.65)
    M1, M2, M3 = _newman_raju_M(ac)
    F = M1 + M2 * x**2 + M3 * x**4        # phi = pi/2: g = f_phi = f_w = 1

    KI = sigma_h * np.sqrt(np.pi * a_m / Q) * F
    return float(np.clip(KI, 0.0, 1e4))


def K_I_surfacepoint(a_m: float, c_m: float,
                      P_bar: float = P_OP_BAR,
                      D: float = PIPE_OD, t: float = PIPE_WT) -> float:
    """
    K_I at the surface end (phi = 0) of the semi-elliptical crack; controls dc/dt.
    [SOURCE: Newman & Raju 1981]

    F_s = [M1 + M2 (a/t)^2 + M3 (a/t)^4] * f_phi * g
    f_phi(phi=0) = sqrt(a/c),   g(phi=0) = 1.1 + 0.35 (a/t)^2

    For the elongated NNpHSCC flaws of interest (a/c <~ 0.75, i.e. c/a >~ 1.3)
    the (f_phi * g) factor is below 1, so K at the surface point is below K at
    the deep point; near a/c = 1 (semicircular) the two converge and can cross.
    """
    P_MPa = P_bar * 0.1
    sigma_h = P_MPa * D / (2.0 * t)
    x = min(a_m / t, 0.95)
    ac = min(a_m / max(c_m, 1e-6), 1.0)

    Q = 1.0 + 1.464 * (ac ** 1.65)
    M1, M2, M3 = _newman_raju_M(ac)
    f_phi = np.sqrt(ac)
    g = 1.1 + 0.35 * x**2
    F_s = (M1 + M2 * x**2 + M3 * x**4) * f_phi * g

    KI_surf = sigma_h * np.sqrt(np.pi * a_m / Q) * F_s
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


# ── Limit state (leak / net-section collapse / brittle) ───────────────────

def folias_factor(c_m: float, D: float = PIPE_OD, t: float = PIPE_WT) -> float:
    """
    Folias bulging factor M_T for an axial through-length 2c in a pressurised
    shell [SOURCE: ASME B31G-2012 / modified log-secant]. A longer flaw has a
    larger M_T, i.e. a higher local (bulging) stress, so it collapses at a
    shallower depth — the mechanism that makes crack-colony coalescence matter.
    """
    L = 2.0 * c_m
    lam = L * L / (D * t)
    if lam <= 50.0:
        return float(np.sqrt(max(1.0 + 0.6275 * lam - 0.003375 * lam * lam, 1.0)))
    return float(0.032 * lam + 3.3)


def flaw_is_critical(a_m: float, c_m: float,
                      P_bar: float = P_OP_BAR,
                      D: float = PIPE_OD, t: float = PIPE_WT,
                      in_HAZ: bool = False) -> bool:
    """
    Limit-state check for an axial semi-elliptical surface flaw. Critical if ANY:
      (1) depth reaches A_CRIT_FRAC of the wall — leak / ligament exhaustion;
      (2) net-section plastic collapse via the Folias factor — a long *coalesced*
          flaw ruptures at a shallower depth than a short one;
      (3) K_I reaches the material fracture toughness K_IC — brittle backstop.
    K_IH is deliberately NOT used here: it is the onset threshold for
    environmental cracking, not a rupture criterion.
    [SOURCE: ASME B31G surface-flaw flow-stress criterion; API 579-1 Level 2.]
    """
    if a_m >= A_CRIT_FRAC * t:
        return True
    if K_I_total(a_m, c_m, P_bar, in_HAZ=in_HAZ) >= K_IC_AIR:
        return True
    M_T = folias_factor(c_m, D, t)
    sigma_flow = 0.5 * (STEEL_SMYS + STEEL_UTS) * 1e-6      # flow stress [MPa]
    sigma_h = (P_bar * 0.1) * D / (2.0 * t)                 # hoop stress [MPa]
    a_t = min(a_m / t, 0.999)
    sigma_fail = sigma_flow * (1.0 - a_t) / (1.0 - a_t / M_T)
    return sigma_h >= sigma_fail


# ── Crack dormancy model ──────────────────────────────────────────────────

def crack_is_dormant(a_m: float, c_m: float, C_H_tip: float,
                      P_bar: float = P_OP_BAR,
                      K_IH: float = K_IH_BASE_MPa,
                      zone: str = 'base') -> bool:
    """
    Stage I → Stage II transition criterion.

    A crack is in active Stage-II growth only when BOTH:
       (1) K_max >= K_IH   — the environmental-cracking stress-intensity threshold
       (2) C_H_tip >= C_H_crit_StageII — enough hydrogen to sustain HEDE
    otherwise it is dormant (Stage I, slow dissolution-controlled creep).

    [SOURCE: Zhao et al. 2017 — >95% of NNpHSCC cracks stay sub-mm and never reach
       Stage II. That emerges here from K_max < K_IH: at operating hoop stress a
       base-metal flaw only reaches K_IH=25 MPa√m near ~5 mm depth, so the whole
       sub-mm population is below threshold and dormant.]
    [SOURCE: Shirazi et al. 2024 — growth ceases while K stays below the EAC threshold.]

    NOTE: the previous release gated on ΔK >= K_TH_STAGE2 (crossed at ~0.28 mm),
    which flagged nearly every sub-mm crack as active and produced 4% dormant
    instead of Zhao's ~95%.

    Returns True if crack is dormant (not growing in Stage II).
    """
    K_max = K_I_total(a_m, c_m, P_bar, in_HAZ=(zone == 'haz'))

    # Both conditions must be met to escape dormancy into Stage II.
    k_active  = K_max >= K_IH
    ch_active = C_H_tip >= C_H_CRIT_STAGEII

    return not (k_active and ch_active)


# ── 3D crack shape evolution ──────────────────────────────────────────────

def crack_shape_evolution(a0_m: float, c0_m: float,
                           spectrum: PressureSpectrum,
                           C_H_bulk: float,
                           K_I_func=None, delta_K_func=None,
                           microstructure_factor: float = 1.0,
                           model_error: float = 1.0,
                           dormancy_factor: float = 1.0,
                           in_haz: bool = False,
                           dt_s: float = 30*86400) -> tuple:
    """
    Advance (a, c) by dt_s seconds using coupled EDOs, with the SAME
    variable-amplitude rate law that integrate_full reports as v(t).

    da/dt = dormancy_factor * da_dt_VA(deepest point)
    dc/dt = dormancy_factor * da_dt_VA(surface point)

    dormancy_factor < 1 applies the Stage-I slow-dissolution penalty for a
    dormant crack (so the reported velocity and the advance use one rate law).
    K_I_func / delta_K_func are accepted for backward compatibility but the
    deep- and surface-point SIFs are built internally so (a, c) stay consistent.

    Returns: (a_new [m], c_new [m])
    """
    da_dt_A = da_dt_variable_amplitude(
        a0_m, spectrum, C_H_bulk,
        K_I_func=lambda am, P: K_I_deeppoint(am, c0_m, P)
                  + (K_I_residual(am) if in_haz else 0.0),
        delta_K_func=lambda am, P, R: delta_K(am, c0_m, P, R),
        microstructure_factor=microstructure_factor, model_error=model_error,
        spectrum_type=spectrum.type)
    dc_dt_C = da_dt_variable_amplitude(
        a0_m, spectrum, C_H_bulk,
        K_I_func=lambda am, P: K_I_surfacepoint(am, c0_m, P),
        delta_K_func=lambda am, P, R: delta_K_surface(am, c0_m, P, R),
        microstructure_factor=microstructure_factor, model_error=model_error,
        spectrum_type=spectrum.type)

    a_new = min(a0_m + dormancy_factor * da_dt_A * dt_s, PIPE_WT * 0.98)
    c_new = c0_m + dormancy_factor * dc_dt_C * dt_s
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
    fractured = False

    for k in range(n_steps + 1):
        t_s = k * dt_s
        t_arr[k] = t_s / SEC_PER_YR
        a_arr[k], c_arr[k] = a, c
        C_H_tip = C_H_tip_func(t_s)
        KI_arr[k] = K_I_total(a, c, P_bar, in_HAZ=(zone == 'haz'))
        ac_arr[k]  = a / max(c, 1e-6)
        dormant_arr[k] = crack_is_dormant(a, c, C_H_tip, P_bar, K_IH, zone)

        # Failure limit state: leak (80% wall), net-section collapse (Folias, so
        # a long coalesced flaw fails shallower), or brittle K_IC. K_IH is the
        # *onset* threshold for environmental cracking — NOT a rupture criterion —
        # so it must not freeze growth here (using it inverted the regime and
        # froze active flaws, e.g. the HAZ baseline, at t=0).
        if flaw_is_critical(a, c, P_bar, in_HAZ=(zone == 'haz')):
            if not fractured:
                fracture_time_yr = t_s / SEC_PER_YR
                fractured = True
            v_arr[k] = 0.0
        else:
            in_haz = (zone == 'haz')
            v_arr[k] = da_dt_variable_amplitude(
                a, spectrum, C_H_bulk,
                # Residual stress raises the mean SIF (K_max), so in the HAZ it
                # accelerates growth — consistent with how it enters K_I_total.
                # ΔK stays residual-free (the weld residual field is static).
                K_I_func=lambda am, P: K_I_deeppoint(am, c, P)
                          + (K_I_residual(am) if in_haz else 0.0),
                delta_K_func=lambda am, P, R: delta_K(am, c, P, R),
                microstructure_factor=f_micro,
                model_error=model_error,
                spectrum_type=spectrum.type
            )

        if k < n_steps and not fractured:
            # Stage-I dormant cracks advance at 10% of the Stage-II rate [ASSUMED];
            # the same VA rate law (v_arr above) advances the geometry.
            dormancy_factor = 0.10 if dormant_arr[k] else 1.0
            a, c = crack_shape_evolution(a, c, spectrum, C_H_bulk,
                                          microstructure_factor=f_micro,
                                          model_error=model_error,
                                          dormancy_factor=dormancy_factor,
                                          in_haz=(zone == 'haz'),
                                          dt_s=dt_s)

    return {
        't_yr': t_arr, 'a': a_arr, 'c': c_arr, 'KI': KI_arr,
        'ac_ratio': ac_arr, 'dormant': dormant_arr, 'v': v_arr,
        'fracture_time_yr': fracture_time_yr,
        'a0': a0_m, 'c0': c0_m, 'zone': zone,
        'pipe_wt': PIPE_WT, 'K_IH': K_IH,
    }
