"""
Phase-1 physics-correctness tests (Week 10 remediation).

These encode the CORRECT NNpHSCC crack-growth regime, which the audit found the
code had inverted:

    dormant (Stage I)      when  K_max < K_IH      (below environmental threshold)
    active  (Stage II)     when  K_IH <= K_max < failure
    failure                when  a >= A_CRIT_FRAC*wall  (or K_max >= K_IC)

The previous code used K >= K_IH as a *rupture* criterion (freezing cracks the
moment they crossed the growth-onset threshold) and gated dormancy on ΔK >= 2.2
MPa√m (crossed at ~0.28 mm). That produced two false headlines: the HAZ baseline
frozen flat at t=0, and a sub-millimetre colony reporting 4% dormant instead of
the ~95% of Zhao et al. 2017. See the audit for details.

Written test-first: with the pre-remediation code these FAIL; they define the
target behaviour for the fix.
"""
import numpy as np
import pytest

from src.crack_growth import (
    integrate_full, crack_is_dormant, K_I_total, K_I_deeppoint, delta_K,
)
from src.pressure_spectrum import PressureSpectrum, da_dt_variable_amplitude
from src.hydrogen_diffusion import C_H_surface_from_potential
from src.crack_colony import simulate_colony
from src.constants import (
    SOIL_PH, E_CP_V, K_IH_BASE_MPa, K_IH_HAZ_MPa, K_IC_AIR,
    PIPE_WT, A_CRIT_FRAC, C_H_BULK_X65,
)

SEC_PER_YR = 365.25 * 24 * 3600


def _C_H_at_cp():
    return C_H_surface_from_potential(E_CP_V, SOIL_PH)


class TestFailureCriterion:
    """Failure is wall penetration / K_IC — never K_IH (the growth-onset threshold)."""

    def test_haz_baseline_is_active_not_frozen_at_t0(self):
        # run_all.py's HAZ baseline: K_total ~= 20 MPa√m > K_IH_HAZ=17.5 but far
        # below 80% wall (10.16 mm) and K_IC (100). It must grow as an active
        # Stage-II crack, not freeze at t=0 and print a benign flat 1.500 mm.
        sp = PressureSpectrum("Type_I")
        r = integrate_full(1.5e-3, 6e-3, sp, lambda t: _C_H_at_cp(),
                           zone='haz', model_error=1.3, t_end_yr=20, n_steps=80)
        assert r['a'][-1] > r['a'][0] * 1.05, (
            f"active HAZ crack must grow >5% over 20 yr; got "
            f"{r['a'][0]*1000:.3f} -> {r['a'][-1]*1000:.3f} mm")
        assert r['fracture_time_yr'] != 0.0, "must not report rupture at t=0"

    def test_crack_above_KIH_but_below_wall_keeps_growing(self):
        # A base crack at ~5 mm has K_max ~ K_IH (25) but is far below 80% wall
        # and K_IC. Under the old rupture-at-K_IH rule it froze immediately.
        sp = PressureSpectrum("Type_I")
        r = integrate_full(5e-3, 20e-3, sp, lambda t: _C_H_at_cp(),
                           zone='base', t_end_yr=5, n_steps=40)
        assert r['a'][-1] > r['a'][0], (
            "crack above K_IH but below the real limit state must keep growing")

    def test_failure_is_wall_penetration_not_KIC(self):
        # A deep, fast HAZ crack must eventually fail by reaching ~80% wall,
        # with the K at failure well below K_IC (i.e. the limit state is depth).
        sp = PressureSpectrum("Type_I")
        r = integrate_full(8e-3, 32e-3, sp, lambda t: _C_H_at_cp(),
                           zone='haz', model_error=1.5, t_end_yr=20, n_steps=80)
        assert np.isfinite(r['fracture_time_yr']), "deep fast crack must fail in 20 yr"
        assert r['a'][-1] >= 0.78 * PIPE_WT, (
            f"failure should be wall penetration (~80%); reached "
            f"{r['a'][-1]/PIPE_WT:.0%} of wall")
        K_at_fail = K_I_total(r['a'][-1], r['c'][-1], in_HAZ=True)
        assert K_at_fail < K_IC_AIR, "wall-penetration failure, not brittle K_IC"


class TestDormancyRegime:
    """Dormancy is gated by K_max < K_IH (with a C_H floor), reproducing Zhao 2017."""

    def test_dormancy_gate_is_KIH_not_small_deltaK(self):
        C_H = _C_H_at_cp()  # 0.023 >> C_H_crit, so C_H never gates here
        # 2 mm base: K_max ~ 14 < K_IH(25) -> dormant even with ample hydrogen
        assert crack_is_dormant(2e-3, 8e-3, C_H, zone='base') is True
        # 6 mm base: K_max ~ 30 > K_IH(25) -> active
        assert crack_is_dormant(6e-3, 24e-3, C_H, zone='base') is False

    def test_submillimetre_cracks_are_dormant(self):
        # Zhao et al. 2017: sub-millimetre NNpHSCC cracks stay dormant because
        # their K_max is far below K_IH. The code previously flagged them active
        # (ΔK gate crossed at ~0.28 mm), reporting 4% dormant instead of ~95%.
        C_H = _C_H_at_cp()
        rng = np.random.default_rng(0)
        a = rng.uniform(0.1e-3, 0.9e-3, 300)   # all sub-millimetre
        c = a * 4.0
        dormant = [crack_is_dormant(a[i], c[i], C_H, zone='base') for i in range(len(a))]
        assert np.mean(dormant) >= 0.99, (
            f"sub-mm cracks (K<<K_IH) must be dormant; got {np.mean(dormant):.0%}")


class TestCalibrationPreserved:
    """The Phase-1a limit-state/dormancy fix must not disturb the rate-law calibration."""

    def test_base_metal_rate_stays_in_cepa_band(self):
        # A_CF is calibrated so base metal grows ~0.3 mm/yr at a=2 mm, c/a=4
        # (CEPA field band). The limit-state change must not move this.
        sp = PressureSpectrum("Type_I")
        a, c = 2e-3, 8e-3
        v = da_dt_variable_amplitude(
            a, sp, C_H_BULK_X65,
            K_I_func=lambda am, P: K_I_deeppoint(am, c, P),
            delta_K_func=lambda am, P, R: delta_K(am, c, P, R),
            spectrum_type="Type_I")
        v_mm_yr = v * SEC_PER_YR * 1000
        assert 0.2 <= v_mm_yr <= 0.4, f"base calibration drifted to {v_mm_yr:.3f} mm/yr"


class TestCollapseCriterion:
    """A coalesced (long) axial flaw fails by net-section collapse (Folias) at a
    shallower depth than a short flaw — which is why colony coalescence matters."""

    def test_short_deep_flaw_fails_by_wall_leak(self):
        from src.crack_growth import flaw_is_critical
        assert flaw_is_critical(0.82 * PIPE_WT, 2e-3) is True   # deep -> leak
        assert flaw_is_critical(0.50 * PIPE_WT, 2e-3) is False  # half wall, short -> safe

    def test_long_flaw_lowers_critical_depth_vs_short(self):
        from src.crack_growth import flaw_is_critical
        a = 0.78 * PIPE_WT                       # just under the 80% depth floor
        assert flaw_is_critical(a, 0.30) is True     # very long (2c=0.6 m) -> collapse
        assert flaw_is_critical(a, 2e-3) is False    # same depth, short -> still safe


class TestAgedColony:
    """An aged NNpHSCC colony (mean ~1.2 mm, deep tail) has a small active
    fraction that deepens and coalesces to a non-zero, honest PoF."""

    def test_aged_colony_nonzero_damage_and_coalescence_fires(self):
        # The colony's pof_final is the fraction of cracks that end up in a
        # critical (coalesced) flaw — a mechanism illustration, non-degenerate
        # (neither 0 nor everything). The headline PoF probability is the MC.
        from src.crack_colony import simulate_colony
        r = simulate_colony(spectrum_type='Type_I', post_ILI=False,
                            n_cracks=120, t_end_yr=20, n_t=40, seed=42)
        assert 0.02 < r['pof_final'] < 0.99, (
            f"colony damage fraction should be non-degenerate; got {r['pof_final']:.3f}")
        assert r['n_coalesced'] > 0, "coalescence must fire in a clustered colony"

    def test_aged_colony_still_mostly_dormant(self):
        from src.crack_colony import simulate_colony
        r = simulate_colony(post_ILI=False, n_cracks=120, t_end_yr=20, n_t=40, seed=42)
        frac_dormant = r['n_dormant_initial'] / r['n_cracks']
        assert frac_dormant >= 0.80, (
            f"even an aged colony is mostly dormant (Zhao); got {frac_dormant:.0%}")
