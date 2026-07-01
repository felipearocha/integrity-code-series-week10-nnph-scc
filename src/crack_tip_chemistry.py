"""
Crack tip chemistry model — acidification and enhanced H entry.

[SOURCE: Turnbull 1993 — crack tip pH drops due to Fe²⁺ hydrolysis]
[SOURCE: Chen et al. 1999 Acta Mater — CO₂ and bicarbonate buffering in crack]

Inside the crack the electrolyte is trapped: exchange with bulk soil is
diffusion-limited through the crack mouth. The Fe²⁺ produced by anodic
dissolution hydrolyses:
    Fe²⁺ + 2H₂O → Fe(OH)₂ + 2H⁺
    K_hyd = [Fe(OH)₂][H⁺]² / [Fe²⁺]  at equilibrium

This lowers crack-tip pH, which increases C_H entry (more H⁺ available for
Volmer-Heyrovsky step), further accelerating HE.

Simplified model [ASSUMED]:
  pH_tip = pH_bulk - delta_pH(a/c, v_diss)
  delta_pH ≈ 0.5 × log10(a/(c × D_eff_electrolyte))  [geometric dilution factor]
  C_H_tip_corrected = C_H_entry × 10^(pH_bulk - pH_tip)^0.3   [ASSUMED pH sensitivity]
"""
import numpy as np
from src.constants import SOIL_PH, C_CO2_MOL, C_HCO3_MOL, T_OP_K


def crack_tip_pH(a_m: float, c_m: float,
                  pH_bulk: float = SOIL_PH,
                  v_diss_mm_yr: float = 0.001,
                  c_co2: float = C_CO2_MOL) -> float:
    """
    Estimate crack-tip pH accounting for Fe²⁺ hydrolysis and CO₂ buffering.
    [SOURCE: Turnbull 1993 simplified; Shipilov 2002]

    Crack aspect ratio a/c determines electrolyte exchange rate.
    Deeper/narrower cracks have lower pH at tip.

    pH_tip = pH_bulk + delta_pH_geo + delta_pH_dissolution
    delta_pH_geo ≈ -0.6 × log10(a/c)  [ASSUMED geometric effect]
    delta_pH_diss ≈ -0.3 × v_diss    [ASSUMED: 0.3 pH units per mm/yr dissolution]

    c_co2 : dissolved-CO₂ concentration [mol/L]. Higher CO₂ weakens the
            HCO₃⁻/CO₂ buffer, so the crack tip is more acidic (more H entry) —
            this is the path by which the sampled C_CO2 modulates da/dt.
    """
    ca_ratio = max(c_m, 1e-6) / max(a_m, 1e-9)   # c/a: elongated=larger
    # Longer/shallower crack -> more stagnant tip -> more acidic [Turnbull 1993]
    delta_pH_geo  = -0.6 * np.log10(max(ca_ratio, 1.0))
    delta_pH_diss = -0.3 * min(v_diss_mm_yr, 1.0)

    # CO₂/HCO₃ buffering reduces pH drop (partial buffer)
    buffer_factor = min(C_HCO3_MOL / max(c_co2, 1e-6), 2.0)
    buffering = 0.3 * np.log10(buffer_factor + 1)

    pH_tip = pH_bulk + delta_pH_geo + delta_pH_diss + buffering
    return float(np.clip(pH_tip, 4.5, pH_bulk))  # floor at 4.5


def C_H_entry_corrected(C_H_0: float, pH_tip: float, pH_bulk: float = SOIL_PH,
                          T: float = T_OP_K) -> float:
    """
    H entry concentration corrected for crack-tip pH drop.
    [SOURCE: Turnbull 1993 — C_H ∝ [H⁺]^0.3 for Volmer step at NNpH]
    C_H_corrected = C_H_0 × 10^((pH_bulk - pH_tip) × 0.3)
    """
    dpH = pH_bulk - pH_tip   # positive if tip is more acidic
    pH_factor = 10 ** (dpH * 0.3)
    return float(np.clip(C_H_0 * pH_factor, C_H_0, C_H_0 * 10))


def crack_tip_chemistry_summary(a_m: float, c_m: float,
                                  C_H_base: float,
                                  pH_bulk: float = SOIL_PH,
                                  v_diss_mm_yr: float = 0.001) -> dict:
    """Full crack tip chemistry calculation."""
    pH_tip = crack_tip_pH(a_m, c_m, pH_bulk, v_diss_mm_yr)
    C_H_corr = C_H_entry_corrected(C_H_base, pH_tip, pH_bulk)
    return {
        'pH_bulk': pH_bulk, 'pH_tip': pH_tip,
        'delta_pH': pH_bulk - pH_tip,
        'C_H_base': C_H_base, 'C_H_corrected': C_H_corr,
        'enhancement_factor': C_H_corr / max(C_H_base, 1e-15),
    }
