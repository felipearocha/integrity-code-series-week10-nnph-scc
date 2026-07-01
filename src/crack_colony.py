"""
Time-stepped full-physics NNpHSCC crack colony.

The cracks in one colony are clustered over a short axial patch (spacing
~COLONY_SPACING_M), grown together step by step, and merged by the BS 7910
Cl.7.3 coalescence rule *during* growth. A coalesced (longer) flaw carries a
larger Folias bulging factor, so it can rupture by net-section collapse at a
shallower depth than an isolated flaw — the mechanism that drives real NNpHSCC
colony failure. Failure of any flaw uses the unified limit state
flaw_is_critical (leak / collapse / K_IC), never K_IH.

Includes:
  - Post-ILI POD-corrected a0 distribution [SOURCE: PHMSA TVC]
  - Dormancy screen K_max < K_IH [SOURCE: Zhao et al. 2017]
  - Microstructure zone assignment (base / HAZ) [SOURCE: Beavers et al.]
  - BS 7910 coalescence during growth [SOURCE: BS 7910:2019 Cl.7.3]
  - Model structural uncertainty per crack [SOURCE: Sun et al. 2021]
  - 3D crack shape evolution (a, c) [SOURCE: Newman-Raju 1981]
"""
import numpy as np
from src.constants import (PIPE_WT, DESIGN_LIFE, C0_MEAN, A0_MEAN, A0_STD,
                           P_OP_BAR, COLONY_SPACING_M)
from src.pod_ilicurve import sample_a0_post_inspection
from src.microstructure import get_zone_properties
from src.model_uncertainty import sample_model_error
from src.crack_growth import (crack_is_dormant, crack_shape_evolution,
                              flaw_is_critical)
from src.pressure_spectrum import PressureSpectrum
from src.hydrogen_diffusion import C_H_surface_from_potential
from src.constants import SOIL_PH, E_CP_V

SEC_PER_YR = 365.25 * 24 * 3600


def bs7910_coalescence_check(a_arr, c_arr, x_arr, t=PIPE_WT):
    """
    BS 7910:2019 Cl.7.3 coalescence rule: if two coplanar cracks are separated
    by s < min(a_i, a_j), treat as a single crack of combined size.
    [SOURCE: BS 7910:2019 Clause 7.3]

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
                gap = abs(x[i] - x[j]) - c[i] - c[j]
                if gap < min(a[i], a[j]):   # BS 7910 criterion
                    x_new = 0.5*(x[i]+x[j])
                    c_new = 0.5*(abs(x[i]-x[j]) + c[i] + c[j])
                    a_new = max(a[i], a[j]) * 1.05  # slight increase for conservatism
                    a = np.delete(a, [i, j]); c = np.delete(c, [i, j]); x = np.delete(x, [i, j])
                    a = np.append(a, a_new); c = np.append(c, c_new); x = np.append(x, x_new)
                    merged = True
                    break
            if merged:
                break
    return a, c, x


def _coalesce_step(a, c, x, zone, eps, members):
    """
    One BS 7910 Cl.7.3 coalescence pass over the current flaw set. Merges any
    pair whose surface gap < min(a_i, a_j) and repeats until stable. Tracks the
    original-crack membership so per-crack trajectories and PoF stay well defined
    after merges. All list arguments are mutated in place and returned.
    """
    n_merged = 0
    merged = True
    while merged and len(a) > 1:
        merged = False
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                gap = abs(x[i] - x[j]) - c[i] - c[j]
                if gap < min(a[i], a[j]):
                    x_new = 0.5 * (x[i] + x[j])
                    c_new = 0.5 * (abs(x[i] - x[j]) + c[i] + c[j])
                    a_new = max(a[i], a[j])
                    z_new = 'haz' if (zone[i] == 'haz' or zone[j] == 'haz') else 'base'
                    e_new = max(eps[i], eps[j])
                    m_new = members[i] + members[j]
                    for idx in (j, i):        # remove the higher index first
                        del a[idx]; del c[idx]; del x[idx]
                        del zone[idx]; del eps[idx]; del members[idx]
                    a.append(a_new); c.append(c_new); x.append(x_new)
                    zone.append(z_new); eps.append(e_new); members.append(m_new)
                    n_merged += 1
                    merged = True
                    break
            if merged:
                break
    return a, c, x, zone, eps, members, n_merged


def simulate_colony(E_pipe: float = E_CP_V,
                        spectrum_type: str = "Type_I",
                        post_ILI: bool = True,
                        n_cracks: int = None,
                        t_end_yr: float = DESIGN_LIFE,
                        n_t: int = 60,
                        seed: int = 42) -> dict:
    """
    Full-physics colony simulation: clustered cracks grown together with
    coalescence during growth and a unified leak/collapse/K_IC limit state.

    Parameters
    ----------
    E_pipe       : float — pipe-to-soil potential [V vs CSE]
    spectrum_type: str   — 'Type_I', 'Type_II', or 'Type_III'
    post_ILI     : bool  — if True, use the POD-corrected a0 distribution
    n_cracks     : int   — number of cracks in the colony (default 30)
    """
    rng = np.random.default_rng(seed)
    if n_cracks is None:
        n_cracks = 30

    # ── Initial crack sizes (aged colony) ─────────────────────────────────
    if post_ILI:
        a0_arr = sample_a0_post_inspection(n_cracks, seed=seed)
    else:
        mu_ln = np.log(A0_MEAN) - 0.5*np.log(1+(A0_STD/A0_MEAN)**2)
        sig_ln = np.sqrt(np.log(1+(A0_STD/A0_MEAN)**2))
        a0_arr = rng.lognormal(mu_ln, sig_ln, n_cracks)
    a0_arr = np.clip(a0_arr, 0.05e-3, PIPE_WT*0.5)
    c0_arr = rng.lognormal(np.log(C0_MEAN), 0.3, n_cracks)
    # Clustered colony patch: neighbours are within coalescence range as they lengthen.
    x_arr  = rng.uniform(0, max(n_cracks, 1) * COLONY_SPACING_M, n_cracks)

    # ── Zone assignment (30% in HAZ near weld seam) [SOURCE: Beavers et al.] ─
    zone_arr = np.where(rng.uniform(size=n_cracks) < 0.30, 'haz', 'base')
    eps_arr = sample_model_error(n_cracks, seed=seed+100)

    spectrum = PressureSpectrum(spectrum_type=spectrum_type)
    C_H_surf = C_H_surface_from_potential(E_pipe, SOIL_PH)
    props = {'base': get_zone_properties('base'), 'haz': get_zone_properties('haz')}

    # ── Initial dormancy screen (K_max < K_IH) ────────────────────────────
    dormant_initial = np.array([
        crack_is_dormant(a0_arr[k], c0_arr[k], C_H_surf, P_OP_BAR,
                          props[zone_arr[k]]['K_IH'], zone_arr[k])
        for k in range(n_cracks)
    ])

    # ── Time-stepped growth + coalescence ─────────────────────────────────
    t_years = np.linspace(0, t_end_yr, n_t)
    dt_s = (t_end_yr * SEC_PER_YR / (n_t - 1)) if n_t > 1 else t_end_yr * SEC_PER_YR

    a = list(a0_arr); c = list(c0_arr); x = list(x_arr)
    zone = list(zone_arr); eps = [float(e) for e in eps_arr]
    members = [[k] for k in range(n_cracks)]

    a_hist = np.zeros((n_t, n_cracks))
    c_hist = np.zeros((n_t, n_cracks))
    failed_time = np.full(n_cracks, np.inf)
    n_coalesced_total = 0

    for step in range(n_t):
        t = t_years[step]
        # record per-original-crack state and failures at this time
        for j in range(len(a)):
            crit = flaw_is_critical(a[j], c[j], P_OP_BAR, in_HAZ=(zone[j] == 'haz'))
            for m in members[j]:
                a_hist[step, m] = a[j]; c_hist[step, m] = c[j]
                if crit and not np.isfinite(failed_time[m]):
                    failed_time[m] = t
        if step == n_t - 1:
            break
        # grow the non-critical flaws by one step
        for j in range(len(a)):
            if flaw_is_critical(a[j], c[j], P_OP_BAR, in_HAZ=(zone[j] == 'haz')):
                continue
            pr = props[zone[j]]
            dormant = crack_is_dormant(a[j], c[j], C_H_surf, P_OP_BAR, pr['K_IH'], zone[j])
            dfac = 0.10 if dormant else 1.0
            a[j], c[j] = crack_shape_evolution(
                a[j], c[j], spectrum, pr['C_H_bulk'],
                microstructure_factor=pr['da_dt_factor'], model_error=eps[j],
                dormancy_factor=dfac, in_haz=(zone[j] == 'haz'), dt_s=dt_s)
        # coalesce flaws that have grown into each other
        a, c, x, zone, eps, members, nm = _coalesce_step(a, c, x, zone, eps, members)
        n_coalesced_total += nm

    # ── Per-original-crack trajectories (for plotting) ────────────────────
    results = [{'a': a_hist[:, m], 'c': c_hist[:, m], 't_yr': t_years,
                'zone': zone_arr[m], 'a0': a0_arr[m]} for m in range(n_cracks)]

    # ── Colony PoF(t) = fraction of original cracks failed by t ────────────
    PoF_t = np.array([np.mean(failed_time <= tt) for tt in t_years])

    a_final = np.array(a); c_final = np.array(c); x_final = np.array(x)

    return {
        'n_cracks': n_cracks, 'results': results,
        'a0': a0_arr, 'c0': c0_arr, 'x': x_arr, 'zone': zone_arr,
        'dormant_initial': dormant_initial,
        'n_dormant_initial': int(dormant_initial.sum()),
        'fracture_times': failed_time,
        't_years': t_years, 'PoF_t': PoF_t, 'pof_final': float(PoF_t[-1]),
        'eps_model': eps_arr,
        'a_after_coalescence': a_final, 'c_after_coalescence': c_final,
        'x_after_coalescence': x_final,
        'n_after_coalescence': len(a_final),
        'n_coalesced': n_coalesced_total,
        'spectrum_type': spectrum_type, 'E_pipe': E_pipe, 'post_ILI': post_ILI,
    }
