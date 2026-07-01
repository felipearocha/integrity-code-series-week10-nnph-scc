"""
Risk-based inspection interval optimization for NNpHSCC.

Determines the optimal re-inspection interval T* that minimises the
total expected cost rate (inspection cost + risk).

Method: PoF threshold approach
[SOURCE: ScienceDirect — "optimal re-assessment intervals using PoF threshold as
decision variable"; Li et al. 2018; Abubakirov et al. 2020 Dynamic Bayesian Network]

Decision model:
  Total expected cost rate = C_insp / T + PoF(T) × CoF

  C_insp : inspection cost [$/km]
  CoF    : consequence of failure [$/km] (leak, rupture, environmental)
  PoF(T) : probability of failure at time T

Optimal T* minimises total cost rate. Also finds T_PoF_limit where
PoF(T) reaches the regulatory limit (typically 10^-4 to 10^-2).

For NNpHSCC: PHMSA requires maximum 5yr re-assessment for class 3 segments.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable, Optional


# Default cost parameters [ASSUMED order-of-magnitude estimates]
C_INSP_DEFAULT   = 150_000.0   # $/km  (ILI EMAT crack tool) [ASSUMED]
COF_LEAK_DEFAULT = 1_000_000.0 # $/km  (leak: repair + environmental) [ASSUMED]
COF_RUPT_DEFAULT = 10_000_000.0# $/km  (rupture: fatality risk + cleanup) [ASSUMED]

# Regulatory PoF limit for NNpHSCC (PHMSA IMP guidance)
POF_LIMIT_REGULATORY = 0.01    # 1% per inspection interval [ASSUMED per CEPA guidance]


def total_expected_cost_rate(T_yr: float,
                               PoF_func: Callable,
                               C_insp: float = C_INSP_DEFAULT,
                               CoF: float = COF_RUPT_DEFAULT) -> float:
    """
    Total expected cost rate [$/km/yr] at inspection interval T.
    = C_insp / T + PoF(T) × CoF / T

    Parameters
    ----------
    T_yr    : float — inspection interval [yr]
    PoF_func: callable(T) → PoF at time T
    C_insp  : float — inspection cost per km [$/km]
    CoF     : float — consequence of failure per km [$/km]
    """
    if T_yr <= 0:
        return 1e20
    pof = float(np.clip(PoF_func(T_yr), 0, 1))
    return (C_insp + pof * CoF) / T_yr


def optimal_inspection_interval(PoF_func: Callable,
                                  T_min_yr: float = 0.5,
                                  T_max_yr: float = 20.0,
                                  C_insp: float = C_INSP_DEFAULT,
                                  CoF: float = COF_RUPT_DEFAULT) -> dict:
    """
    Find T* that minimises total expected cost rate.
    Also computes T_PoF_limit (when PoF reaches regulatory threshold).

    Parameters
    ----------
    PoF_func  : callable(T_yr) → PoF [0, 1]
    T_min_yr  : float — minimum inspection interval [yr]
    T_max_yr  : float — maximum inspection interval considered [yr]
    C_insp    : float — ILI cost per km [$]
    CoF       : float — consequence of failure per km [$]

    Returns
    -------
    dict with T_opt, total_cost_rate_opt, T_pof_limit, cost_curve
    """
    def cost_fn(T):
        return total_expected_cost_rate(T, PoF_func, C_insp, CoF)

    result = minimize_scalar(cost_fn, bounds=(T_min_yr, T_max_yr), method='bounded')
    T_opt_cost = float(result.x)

    # Cost curve for visualization
    T_arr = np.linspace(T_min_yr, T_max_yr, 100)
    cost_arr = np.array([cost_fn(t) for t in T_arr])
    PoF_arr  = np.array([float(np.clip(PoF_func(t), 0, 1)) for t in T_arr])

    # Find T where PoF first exceeds regulatory limit
    T_pof_limit = T_max_yr
    for i, (t, p) in enumerate(zip(T_arr, PoF_arr)):
        if p >= POF_LIMIT_REGULATORY:
            T_pof_limit = float(t)
            break

    # Risk constraint: the interval must not exceed T_pof_limit — you have to
    # re-inspect before PoF reaches the regulatory limit. Without this cap a
    # fast-rising PoF lets the C_insp/T term dominate and pushes the cost optimum
    # to T_max, which is backwards for an aggressive (short-life) segment.
    T_opt = float(min(T_opt_cost, T_pof_limit))
    cost_opt = float(cost_fn(T_opt))

    # PHMSA 5yr limit check
    phmsa_compliant = T_opt <= 5.0

    return {
        'T_opt_yr': T_opt,
        'total_cost_opt': cost_opt,
        'T_pof_limit_yr': T_pof_limit,
        'T_arr': T_arr,
        'cost_arr': cost_arr,
        'PoF_arr': PoF_arr,
        'phmsa_5yr_compliant': phmsa_compliant,
        'C_insp': C_insp, 'CoF': CoF,
        'regulatory_PoF_limit': POF_LIMIT_REGULATORY,
        'note': ('Optimal interval minimises C_insp/T + PoF(T)×CoF/T. '
                 '[ASSUMED] cost parameters. PHMSA class 3 limit: 5 yr. '
                 '[SOURCE: Li et al. 2018; Abubakirov et al. 2020 DBN approach]'),
    }


def PoF_from_mc_trajectory(mc_result: dict) -> Callable:
    """
    Create a PoF(T) interpolating function from MC time-series results.
    """
    from scipy.interpolate import interp1d
    t = mc_result['t_years']
    pof = mc_result['PoF_t']
    return interp1d(t, pof, kind='linear', bounds_error=False,
                    fill_value=(pof[0], pof[-1]))
