"""
PDE 2: Oriani hydrogen diffusion in pipeline steel wall.

Governing equation (modified Fick with trapping):
  (1 + K_trap * N_trap / C_L) * dC_L/dt
       = D_eff * d²C_L/dr²  +  D_eff * V_H/(RT) * d/dr(C_L * sigma_h * dr)

where:
  C_L   : lattice H concentration [mol m^-3]
  C_T   : trapped H concentration = K_trap * N_trap * C_L / (1 + K_trap * C_L) [ASSUMED Oriani]
  D_eff = D_H / (1 + K_trap * N_trap / C_L^2)   effective diffusivity [ASSUMED]
  sigma_h: hydrostatic stress [Pa]

Boundary conditions:
  C_L(r_inner) = C_H_entry(E_pipe, pH)   electrochemical entry (Langmuir)
  dC_L/dr|_{r_outer} = 0                  zero flux at outer surface (Devanathan)

Simplification applied here: 1D radial through-wall diffusion, 
hydrostatic stress included via Oriani fugacity correction [ASSUMED constant sigma_h].
"""

import numpy as np
from scipy.linalg import solve_banded
from src.constants import (
    R_GAS, FARADAY, D_H_LATTICE, E_A_DIFF, T_REF_H,
    V_H, N_TRAP, K_TRAP_EQ, C_H_SURF_REF,
    PIPE_OD, PIPE_WT, T_OP_K,
)


# ── Diffusivity ────────────────────────────────────────────────────────────

def D_H(T: float) -> float:
    """
    Hydrogen lattice diffusivity in alpha-Fe [m^2 s^-1].
    D_H(T) = D_H0 * exp(-E_a / (R*T))
    Reference: Kiuchi & McLellan (1983) Acta Metall.
    """
    return D_H_LATTICE * np.exp(-E_A_DIFF / (R_GAS * T))


def D_eff(T: float, C_L: float = None) -> float:
    """
    Effective H diffusivity in steel with reversible trapping [m^2 s^-1].

    Simplified Oriani model: D_eff = D_H(T) / (1 + K_trap_eff)
    where K_trap_eff is a dimensionless retardation factor representing
    the equilibrium trapping at grain boundaries and dislocations.

    [ASSUMED] K_trap_eff = K_TRAP_EQ (dimensionless) = 10, calibrated to
    D_eff ~ 1e-9 m^2/s for X65 pipeline steel per San Marchi & Somerday (2012).
    This gives diffusion time across 12.7mm wall: ~2 days (fast equilibration).
    """
    return D_H(T) / (1.0 + K_TRAP_EQ)


# ── Electrochemical surface concentration ─────────────────────────────────

def C_H_surface_from_potential(E_pipe: float, pH: float,
                 T: float = T_OP_K,
                 beta_H: float = 0.5) -> float:
    """
    Subsurface H concentration [mol m^-3] via Langmuir-McLean adsorption
    combined with Volmer step kinetics.

    C_H = C_H_ref * exp(F * (E_free - E_pipe) / (beta_H * R * T))

    More cathodic potential -> higher H entry.
    [ASSUMED] linear Tafel region for H recombination kinetics.

    Parameters
    ----------
    E_pipe : float, pipe-to-soil potential [V vs CSE]
    pH     : float, near-crack-tip pH
    beta_H : float, symmetry factor for H entry [ASSUMED 0.5]
    """
    from src.constants import E_FREE_V
    delta_E = E_FREE_V - E_pipe   # positive for cathodic protection
    C_H = C_H_SURF_REF * np.exp(FARADAY * delta_E / (beta_H * R_GAS * T))
    # pH correction: higher pH -> lower H entry (less H+ available)
    pH_factor = 10 ** (-(pH - 7.0) * 0.3)   # [ASSUMED] empirical factor
    # Cap: physical maximum in alpha-Fe at NNpHSCC conditions ~0.1 mol/m^3 [ASSUMED per literature]
    return float(np.clip(C_H * pH_factor, 0.0, 0.1))


# ── Stress-assisted diffusion correction (Oriani) ─────────────────────────

def C_H_stress_corrected(C_H_0: float, sigma_h: float,
                           T: float = T_OP_K) -> float:
    """
    H concentration at crack tip enhanced by hydrostatic stress.
    C_H(sigma_h) = C_H_0 * exp(sigma_h * V_H / (R * T))
    Reference: Oriani (1970) Acta Metall., Li et al. (1966).
    """
    return C_H_0 * np.exp(sigma_h * V_H / (R_GAS * T))


# ── 1D radial FDM solver ──────────────────────────────────────────────────

def build_H_mesh(n_r: int = 20) -> dict:
    """
    1D radial mesh through pipe wall.
    r in [r_i, r_o] where r_i = inner radius, r_o = outer radius.
    """
    r_i = (PIPE_OD - 2 * PIPE_WT) / 2  # inner radius
    r_o = PIPE_OD / 2
    r = np.linspace(r_i, r_o, n_r)
    return {"r": r, "r_i": r_i, "r_o": r_o, "n_r": n_r}


def hydrogen_step_CN(mesh: dict, C_L: np.ndarray,
                      C_H_entry: float, sigma_h: float,
                      dt: float, T: float = T_OP_K) -> np.ndarray:
    """
    Advance H concentration field by dt [s] using Crank-Nicolson.
    
    BC: C_L[0] = C_H_entry * exp(sigma_h * V_H / (R*T))  (stress-enhanced entry)
    BC: dC_L/dr|outer = 0  (zero flux, or Devanathan permeation side)
    """
    r = mesh["r"]
    n = mesh["n_r"]
    C_entry = C_H_stress_corrected(C_H_entry, sigma_h, T)

    # Build tridiagonal system (Crank-Nicolson)
    diag  = np.zeros(n)
    upper = np.zeros(n - 1)
    lower = np.zeros(n - 1)
    rhs   = np.zeros(n)

    # Inner BC (Dirichlet)
    diag[0] = 1.0
    rhs[0]  = C_entry

    dr = np.diff(r)

    for i in range(1, n - 1):
        drm = dr[i - 1]; drp = dr[i]
        rc  = r[i]
        Deff_m = 0.5 * (D_eff(T, C_L[i-1]) + D_eff(T, C_L[i]))
        Deff_p = 0.5 * (D_eff(T, C_L[i])   + D_eff(T, C_L[i+1]))
        r_m = 0.5 * (r[i-1] + r[i])
        r_p = 0.5 * (r[i]   + r[i+1])
        cm = Deff_m * r_m / (drm * rc)
        cp = Deff_p * r_p / (drp * rc)
        rc_dt = 1.0 / dt
        lower[i-1]  = -0.5 * cm
        diag[i]     = rc_dt + 0.5 * (cm + cp)
        upper[i]    = -0.5 * cp
        rhs[i] = (rc_dt - 0.5*(cm+cp))*C_L[i] + 0.5*cm*C_L[i-1] + 0.5*cp*C_L[i+1]

    # Outer BC (zero flux = Neumann, reflecting)
    diag[-1]  =  1.0
    lower[-1] = -1.0
    rhs[-1]   =  0.0

    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1, :]  = diag
    ab[2, :-1]= lower

    C_new = solve_banded((1, 1), ab, rhs)
    return np.clip(C_new, 0.0, 200.0)


def run_H_diffusion(C_H_entry: float, sigma_h: float,
                     t_total_s: float, n_r: int = 20,
                     n_t: int = 100, T: float = T_OP_K) -> dict:
    """
    Run H diffusion PDE from dry initial condition.
    Returns time histories and crack-tip H concentration.
    """
    mesh = build_H_mesh(n_r)
    C_L = np.zeros(n_r)
    dt = t_total_s / n_t
    t_arr = np.arange(n_t + 1) * dt

    C_tip_hist = np.zeros(n_t + 1)
    C_tip_hist[0] = 0.0

    for k in range(n_t):
        C_L = hydrogen_step_CN(mesh, C_L, C_H_entry, sigma_h, dt, T)
        C_tip_hist[k + 1] = C_L[0]  # crack tip is at inner surface

    return {
        "t_s": t_arr,
        "C_tip": C_tip_hist,
        "C_final": C_L,
        "r": mesh["r"],
        "C_H_entry": C_H_entry,
        "sigma_h": sigma_h,
    }


def steady_state_H_concentration(C_H_entry: float, sigma_h: float,
                                   T: float = T_OP_K, n_r: int = 20) -> np.ndarray:
    """
    Steady-state H concentration profile across wall.
    Linear profile for constant D_eff (approximate).
    """
    mesh = build_H_mesh(n_r)
    C_entry = C_H_stress_corrected(C_H_entry, sigma_h, T)
    # With zero-flux outer BC: C_L = C_entry everywhere (steady state)
    return np.full(n_r, C_entry)
