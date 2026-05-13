"""
Microstructure zones and material property gradients.

[SOURCE: Beavers et al. 2001 — HAZ growth 30% faster than base metal]
[SOURCE: Shirazi et al. 2024 — K_IH reduced in HAZ; ferrite-pearlite more susceptible]
[SOURCE: Sun et al. 2021 review — C_H_bulk 0.01–10 mol/m³ grade-dependent]

Zones modelled:
  'base'  — bainitic acicular ferrite (X65, X70, X80)
  'haz'   — ferrite-pearlite at HAZ boundary (within ~3mm of weld toe)
  'weld'  — weld metal (assumed similar to HAZ for conservatism) [ASSUMED]
"""
import numpy as np
from src.constants import (
    STEEL_SMYS, K_IH_BASE_MPa, K_IH_HAZ_MPa, F_MICRO_HAZ,
    C_H_BULK_X65, C_H_BULK_HAZ, C_H_BULK_X52,
    A_CF_BASE,
)


ZONE_PROPERTIES = {
    'base': {
        'K_IH':           K_IH_BASE_MPa,   # MPa sqrt(m)
        'da_dt_factor':   1.0,
        'C_H_bulk':       C_H_BULK_X65,    # mol/m^3
        'A_CF':           A_CF_BASE,
        'description':    'Bainitic-acicular ferrite (X65 base metal)',
    },
    'haz': {
        'K_IH':           K_IH_HAZ_MPa,    # [SOURCE: 30% lower toughness assumption]
        'da_dt_factor':   F_MICRO_HAZ,     # 1.30 [SOURCE: Beavers et al.]
        'C_H_bulk':       C_H_BULK_HAZ,    # 4× higher H [SOURCE: Sun 2021]
        'A_CF':           A_CF_BASE * F_MICRO_HAZ,
        'description':    'Ferrite-pearlite HAZ (within 3 mm of weld toe)',
    },
    'vintage_erw_seam': {
        'K_IH':           12.0,            # MPa sqrt(m) — ERW seam weld, oriented pearlite [SOURCE: Sun 2024 inferred]
        'da_dt_factor':   2.00,            # [SOURCE: Sun 2024 — seam weld significantly faster]
        'C_H_bulk':       C_H_BULK_HAZ * 1.5,  # pearlite bands increase H susceptibility [ASSUMED]
        'A_CF':           A_CF_BASE * 2.0,
        'description':    'Vintage X52 ERW seam weld (oriented pearlite) [SOURCE: Sun 2024 Fatigue Fract]',
    },
    'vintage_x52': {
        'K_IH':           18.0,            # MPa sqrt(m) [ASSUMED lower for vintage]
        'da_dt_factor':   1.50,            # [ASSUMED: vintage X52 more susceptible]
        'C_H_bulk':       C_H_BULK_X52,
        'A_CF':           A_CF_BASE * 1.5,
        'description':    'Ferrite-pearlite vintage X52 (pre-1980 pipeline)',
    },
}


def get_zone_properties(zone: str = 'base') -> dict:
    """Return material properties for given microstructural zone."""
    if zone not in ZONE_PROPERTIES:
        raise ValueError(f"Unknown zone '{zone}'. Valid: {list(ZONE_PROPERTIES.keys())}")
    return dict(ZONE_PROPERTIES[zone])


def zone_from_position(x_from_weld_mm: float, weld_toe_width_mm: float = 3.0) -> str:
    """
    Determine microstructural zone based on distance from weld centreline.
    x_from_weld_mm : float — distance from weld centreline [mm]
    Returns zone identifier string.
    """
    if abs(x_from_weld_mm) <= weld_toe_width_mm:
        return 'haz'
    return 'base'


def K_IH_from_zone(zone: str) -> float:
    """K_IH [MPa sqrt(m)] for given zone."""
    return ZONE_PROPERTIES[zone]['K_IH']


def C_H_bulk_from_zone(zone: str) -> float:
    """C_H_bulk [mol/m^3] for given zone — controls HEDE multiplier."""
    return ZONE_PROPERTIES[zone]['C_H_bulk']
