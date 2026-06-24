"""
Non-monotonic CP potential effect on NNpHSCC crack growth rate.

CRITICAL FINDING: The relationship between pipe-to-soil potential and
NNpHSCC crack growth rate is NOT monotone. There is an optimal potential
that minimises CGR.

[SOURCE: ResearchGate / ScienceDirect — "As cathodic potential negatively
decreases, the crack growth rates of four microstructures first decrease
and then increase with the minimum CGR at -750mV, which depends on the
variation of hydrogen concentration."]

Physical explanation:
  - At free corrosion potential (E ~ -0.68 V CSE): anodic dissolution dominates
  - At mild cathodic protection (E ~ -0.75 V CSE): dissolution suppressed,
    C_H low — MINIMUM CGR window
  - At standard CP (-0.85 V CSE) and beyond: C_H increases rapidly
    (overprotection H generation) — CGR INCREASES again
  - At overprotection (E < -1.1 V CSE): very high H generation, coating
    disbondment risk, maximum CGR

Model: CGR_factor(E) = alpha * exp(beta_1 * (E - E_opt)^2)
Parameters calibrated to represent the parabolic minimum.

Operational implication: The industry standard NACE criterion of -850 mV CSE
is in the INCREASING branch of the CGR curve for NNpHSCC.
Optimal CP for NNpHSCC management: -750 mV CSE [ASSUMED per literature].
"""
import numpy as np
from src.constants import E_CP_V, E_FREE_V, T_OP_K, R_GAS, FARADAY


# Optimal potential for minimum NNpHSCC CGR
E_OPT_NNpH = -0.75    # V vs CSE  [SOURCE: literature on minimum CGR]
E_FREE = E_FREE_V      # -0.68 V vs CSE

# Parabolic CGR factor curve parameters [ASSUMED, calibrated to qualitative shape]
# CGR_factor(E_OPT=-0.75) = 1.0   (normalized to the minimum)
# CGR_factor(E=-0.85 NACE)   ~ 2.0   (standard CP, slightly worse than optimal)
# CGR_factor(E=-0.68 free)   ~ 5.0   (free corrosion: anodic dissolution dominates)
# CGR_factor(E=-1.10 overpr) ~ 13    (excess H generation -> highest CGR, matching
#                                     the module narrative that overprotection is worst)
CP_PARABOLA_A  = 816.0   # anodic branch, calibrated: CGR_factor(E=-0.68)=5.0 [ASSUMED]
CP_PARABOLA_B  = 100.0   # cathodic branch, calibrated: CGR_factor(E=-0.85)=2.0 [ASSUMED]
CP_FREE_FACTOR = 8.0     # sets the clip headroom (max factor = 16) [ASSUMED]


def CGR_factor_from_potential(E_pipe: float) -> float:
    """
    Multiplicative CGR factor as function of pipe-to-soil potential.
    Normalized to 1.0 at E_opt = -750 mV CSE.

    Non-monotonic: minimum at E_opt, increasing both toward free corrosion
    and toward strong overprotection.

    [SOURCE: Literature — minimum CGR at -750 mV CSE for NNpHSCC]
    [ASSUMED] Parabolic model with asymmetric branches.

    Parameters
    ----------
    E_pipe : float — pipe-to-soil potential [V vs CSE]

    Returns
    -------
    float — CGR multiplier (>= 1.0)
    """
    dE = E_pipe - E_OPT_NNpH

    if E_pipe >= E_OPT_NNpH:
        # Anodic side (less cathodic than optimal): dissolution increases CGR
        factor = 1.0 + CP_PARABOLA_A * dE ** 2
    else:
        # Cathodic side (more cathodic than optimal): H generation increases CGR
        factor = 1.0 + CP_PARABOLA_B * dE ** 2

    return float(np.clip(factor, 1.0, CP_FREE_FACTOR * 2))


def optimal_CP_potential() -> float:
    """Return the optimal CP potential for NNpHSCC management."""
    return E_OPT_NNpH


def CGR_factor_vs_potential_curve(E_min: float = -1.2,
                                    E_max: float = -0.60,
                                    n: int = 200) -> dict:
    """
    Generate the CGR factor vs potential curve for visualization.
    Also returns NACE criterion marker and free corrosion marker.
    """
    E_vals = np.linspace(E_min, E_max, n)
    factors = np.array([CGR_factor_from_potential(E) for E in E_vals])
    return {
        'E': E_vals,
        'CGR_factor': factors,
        'E_opt': E_OPT_NNpH,
        'E_NACE': E_CP_V,         # -0.85 V (standard CP criterion)
        'E_free': E_FREE_V,       # -0.68 V (free corrosion)
        'CGR_at_NACE': CGR_factor_from_potential(E_CP_V),
        'CGR_at_free': CGR_factor_from_potential(E_FREE_V),
        'CGR_at_opt':  1.0,
        'note': ('Minimum CGR at -750 mV CSE [SOURCE: literature]. '
                 'NACE -850 mV criterion gives higher CGR than optimal for NNpHSCC. '
                 '[ASSUMED] parabolic model — qualitative shape only.'),
    }


def C_H_vs_potential(E_pipe: float, C_H_ref: float = 5e-6,
                      T: float = T_OP_K) -> float:
    """
    Subsurface H concentration in steel as a function of pipe potential.

    MONOTONIC in potential: H entry increases as the potential becomes more
    cathodic (delegated to hydrogen_diffusion.C_H_surface_from_potential).
    The non-monotonic CGR minimum at E_opt = -0.75 V is NOT produced by C_H
    being non-monotonic; it arises from the competition between anodic
    dissolution (anodic side) and cathodic hydrogen embrittlement (cathodic
    side), captured separately in CGR_factor_from_potential.
    """
    from src.hydrogen_diffusion import C_H_surface_from_potential
    from src.constants import SOIL_PH
    return C_H_surface_from_potential(E_pipe, SOIL_PH)
