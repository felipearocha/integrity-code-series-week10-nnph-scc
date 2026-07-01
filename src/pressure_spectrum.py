"""
Variable Amplitude Pressure Loading — NNpHSCC context.

Implements Chen & Sutherby (2007) combined parameter with underload interaction.

Pressure spectrum types [SOURCE: Chen literature, ScienceDirect Topics Ch.30]:
  Type I  — underload (downstream compressor, within 30 km)
  Type II — mean-load fluctuation (mid-pipeline)
  Type III — mixed random

Key result: underload cycles accelerate minor cycle growth by factor F_INT = 10
[SOURCE: Chen 2007 — "crack growth enhancement by factor of 10 under underload-type VA"]

Combined parameter [SOURCE: Chen & Sutherby 2007]:
  C = K_max × ΔK² × f^(-0.1)   where f_eff = max(f, f_crit)

Xing HEDE multiplier [SOURCE: Xing et al. via Sun et al. 2021]:
  da/dN = A_CF × C^n × (C_H_bulk / C_H_ref)^n_HE
"""
import numpy as np
from src.constants import (
    P_OP_BAR, P_CYCLE_FRAC_MAJOR, P_CYCLE_FRAC_MINOR,
    FREQ_MAJOR_HZ, FREQ_MINOR_HZ, F_CRIT_HZ, F_INT_UNDERLOAD,
    R_RATIO_MAJOR, R_RATIO_MINOR, N_MINOR_PER_MAJOR,
    A_CF_BASE, N_CF, N_HE_XING, K_TH_STAGE2, K_TH_AIR,
    C_H_BULK_X65, C_H_SURF_REF,
)
SEC_PER_YR = 365.25 * 24 * 3600


class PressureSpectrum:
    """Operational pressure spectrum for NNpHSCC crack growth calculation."""

    def __init__(self, spectrum_type="Type_I",
                 P_max_bar=P_OP_BAR,
                 frac_major=P_CYCLE_FRAC_MAJOR,
                 frac_minor=P_CYCLE_FRAC_MINOR,
                 f_major_Hz=FREQ_MAJOR_HZ,
                 f_minor_Hz=FREQ_MINOR_HZ,
                 n_minor_per_major=N_MINOR_PER_MAJOR):
        """
        Parameters
        ----------
        spectrum_type : str — 'Type_I' (underload), 'Type_II' (CA), 'Type_III' (mixed)
        P_max_bar     : float — maximum operating pressure [bar]
        frac_major    : float — ΔP/P_max for major underload cycle [ASSUMED]
        frac_minor    : float — ΔP/P_max for minor ripple cycles [ASSUMED]
        f_major_Hz    : float — frequency of major cycles [Hz]
        f_minor_Hz    : float — frequency of minor cycles [Hz]
        n_minor_per_major : int — number of minor cycles between underloads
        """
        self.type = spectrum_type
        self.P_max = P_max_bar
        self.frac_major = frac_major
        self.frac_minor = frac_minor
        self.f_major = f_major_Hz
        self.f_minor = f_minor_Hz
        self.n_minor = n_minor_per_major

    def dP_major(self):
        return self.P_max * self.frac_major

    def dP_minor(self):
        return self.P_max * self.frac_minor

    def R_major(self):
        return R_RATIO_MAJOR

    def R_minor(self):
        return R_RATIO_MINOR

    def describe(self):
        return (f"{self.type}: ΔP_major={self.dP_major():.1f} bar "
                f"@ {self.f_major*86400:.1f} cycles/day | "
                f"ΔP_minor={self.dP_minor():.1f} bar × {self.n_minor} ripples/major")


def f_eff(f_Hz):
    """Saturate frequency at f_crit [SOURCE: Xing model via Sun, Zhou & Kang 2021]
    — below f_crit the environment controls, not fatigue frequency."""
    return max(f_Hz, F_CRIT_HZ)


def combined_parameter(K_max_MPa, delta_K_MPa, f_Hz):
    """
    Chen-Sutherby combined parameter [SOURCE: Chen & Sutherby 2007].
    C = K_max × ΔK² × f_eff^(-0.1)
    Units: MPa sqrt(m) × (MPa sqrt(m))² × Hz^(-0.1) = MPa³ m^1.5 Hz^(-0.1)
    """
    fe = f_eff(f_Hz)
    return K_max_MPa * (delta_K_MPa ** 2) * (fe ** (-0.1))


def xing_HE_factor(C_H_bulk_mol_m3, C_H_ref=0.001,  # air baseline ~0 mol/m^3 [ASSUMED]
                    n_HE=N_HE_XING):
    """
    Xing HEDE hydrogen enhancement multiplier [SOURCE: Xing et al. via Sun 2021].
    HE_factor = (C_H_bulk / C_H_ref)^n_HE
    n_HE = 0.88 for X52; use 0.88 conservatively for X65 [ASSUMED same order]
    """
    ratio = max(C_H_bulk_mol_m3, 1e-10) / max(C_H_ref, 1e-10)
    return max(ratio ** n_HE, 0.1)   # floor at 0.1 (air baseline)


def da_dN_Chen_Xing(a_m, K_max_MPa, delta_K_MPa, f_Hz, C_H_bulk,
                     A_CF=A_CF_BASE, n=N_CF,
                     microstructure_factor=1.0,
                     model_error=1.0):
    """
    da/dN [m/cycle] using Chen-Sutherby combined parameter + Xing HEDE multiplier.

    da/dN = A_CF × (K_max × ΔK² × f_eff^(-0.1))^n × HE_factor × f_micro × ε_model

    [ASSUMED] A_CF calibrated so that at nominal conditions:
    da/dN ≈ 3e-8 mm/cycle × (C_mech)^2, yielding da/dt ≈ 0.1 mm/yr at f=1e-5 Hz.

    Parameters
    ----------
    model_error : float — multiplicative model structural uncertainty ε_m
    """
    if delta_K_MPa <= 0 or K_max_MPa <= K_TH_AIR:
        return 0.0
    C_mech = combined_parameter(K_max_MPa, delta_K_MPa, f_Hz)
    HE_fac = xing_HE_factor(C_H_bulk)
    da_dN = A_CF * (C_mech ** n) * HE_fac * microstructure_factor * model_error
    return float(np.clip(da_dN, 0.0, 1e-3))  # max 1mm/cycle physical cap


def da_dt_variable_amplitude(a_m, spectrum: PressureSpectrum, C_H_bulk,
                               K_I_func,            # callable(a, P) → K_I [MPa sqrt(m)]
                               delta_K_func,         # callable(a, P, R) → ΔK
                               microstructure_factor=1.0,
                               model_error=1.0,
                               interaction_factor=F_INT_UNDERLOAD,
                               spectrum_type="Type_I"):
    """
    Effective da/dt [m/s] for variable amplitude loading.

    For Type I (underload spectrum) [SOURCE: Chen 2007]:
      da/dt = f_major × da/dN_major + f_minor × da/dN_minor × F_INT

    where F_INT = 10 accounts for the underload cycle activating minor cycles that
    would otherwise be non-propagating (ΔK < ΔK_th in constant amplitude).

    For Type II (constant amplitude, conservative):
      da/dt = f_major × da/dN_major

    For Type III (mixed):
      da/dt = 0.5 × Type_I + 0.5 × Type_II
    """
    P_max = spectrum.P_max

    # Major cycle SIFs
    K_max_maj = K_I_func(a_m, P_max)
    dK_maj    = delta_K_func(a_m, P_max, spectrum.R_major())

    # Minor ripple cycles ride at the peak (same K_max) with a small stress
    # swing governed by R_minor. NOTE: the minor-cycle ΔK is set by R_minor, not
    # by P_CYCLE_FRAC_MINOR — the previous expression algebraically collapsed to
    # P_max, so frac_minor never entered ΔK. Kept behaviour-identical (R_minor).
    K_max_min = K_max_maj
    dK_min    = delta_K_func(a_m, P_max, spectrum.R_minor())

    da_dN_maj = da_dN_Chen_Xing(a_m, K_max_maj, dK_maj, spectrum.f_major,
                                  C_H_bulk, microstructure_factor=microstructure_factor,
                                  model_error=model_error)
    da_dN_min = da_dN_Chen_Xing(a_m, K_max_min, dK_min, spectrum.f_minor,
                                  C_H_bulk, microstructure_factor=microstructure_factor,
                                  model_error=model_error)

    if spectrum_type == "Type_I":
        # Underload interaction: minor cycles activated by preceding underload
        da_dt = (spectrum.f_major * da_dN_maj +
                 spectrum.f_minor * da_dN_min * interaction_factor)
    elif spectrum_type == "Type_II":
        da_dt = spectrum.f_major * da_dN_maj
    else:  # Type_III mixed
        da_dt_I  = (spectrum.f_major * da_dN_maj +
                    spectrum.f_minor * da_dN_min * interaction_factor)
        da_dt_II = spectrum.f_major * da_dN_maj
        da_dt = 0.5 * da_dt_I + 0.5 * da_dt_II

    return float(np.clip(da_dt, 0.0, 1e-7))  # max ~3 m/yr physical cap
