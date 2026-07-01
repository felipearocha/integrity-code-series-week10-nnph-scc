"""Integrity Code Series — Week 10 — Full-physics NNpHSCC with all 15 gap mechanisms.

Author: Felipe Rocha
Run: python run_all.py
"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:                                    # emit UTF-8 so the script does not crash
    sys.stdout.reconfigure(encoding="utf-8")   # on a legacy Windows cp1252 console
except Exception:
    pass
os.makedirs("assets/figures", exist_ok=True); os.makedirs("assets/animations", exist_ok=True)

print("="*72)
print("ICS2 WEEK 10 — NNpHSCC FULL PHYSICS")
print("Chen-Sutherby-Xing | VA Spectrum | Dormancy | POD | Model Error | Bayes")
print("Regulatory hook: PHMSA §192.611(a)(4) eff. March 16, 2026")
print("="*72); t_wall = time.time()

# [1] Benchmarks
print("\n[1/10] Validation benchmarks...")
import validation.benchmarks as bm
bm.benchmark_chen_sutherby_combined(); bm.benchmark_K_I_flat_plate()
bm.benchmark_faraday(); bm.benchmark_pod_weibull()
bm.benchmark_model_error_moments(); bm.benchmark_hoop_stress()

# [2] Baseline crack growth — both zones
print("\n[2/10] Baseline crack growth — base metal + HAZ...")
from src.pressure_spectrum import PressureSpectrum
from src.crack_growth import integrate_full
from src.hydrogen_diffusion import C_H_surface_from_potential
from src.constants import SOIL_PH, E_CP_V, K_IH_BASE_MPa, K_IH_HAZ_MPa
C_H = C_H_surface_from_potential(E_CP_V, SOIL_PH)
sp = PressureSpectrum("Type_I")
bl_base = integrate_full(0.5e-3, 3e-3, sp, lambda t: C_H, zone='base', t_end_yr=20, n_steps=80)
bl_haz  = integrate_full(1.5e-3, 6e-3, sp, lambda t: C_H, zone='haz', model_error=1.3, t_end_yr=20, n_steps=80)
print(f"  Base: a0=0.5mm -> a(20yr)={bl_base['a'][-1]*1000:.3f}mm  (dormant, no failure; K_IH={K_IH_BASE_MPa})")
_haz_ft = bl_haz['fracture_time_yr']
_haz_state = (f"active -> wall penetration at t={_haz_ft:.2f}yr"
              if np.isfinite(_haz_ft) else f"a(20yr)={bl_haz['a'][-1]*1000:.3f}mm (no failure)")
print(f"  HAZ:  a0=1.5mm {_haz_state}  (K_IH={K_IH_HAZ_MPa})")
assert bl_base['a'][0] == 0.5e-3, "Initial crack must equal a0"
assert np.all(np.diff(bl_base['a']) >= -1e-15), "Crack must be monotone"
assert np.all(np.diff(bl_haz['a']) >= -1e-15), "HAZ crack must be monotone"

# [3] MC N=10,000
print("\n[3/10] Monte Carlo — N=10,000 (8 params, post-ILI)...")
t0=time.time()
from src.monte_carlo import run_monte_carlo, spearman_sensitivity
mc = run_monte_carlo(n_samples=10_000, n_t=20, seed=42, post_ILI=True)
print(f"  Done in {time.time()-t0:.0f}s  PoF={mc['pof_final']:.4f}")
print(f"  P50={np.percentile(mc['wall_loss'],50):.3f}mm  P95={np.percentile(mc['wall_loss'],95):.3f}mm")
rho = spearman_sensitivity(mc)
print("  Spearman:", {k:round(v,3) for k,v in list(rho.items())[:4]})

# [4] GBR surrogate
print("\n[4/10] GBR surrogate (8 features)...")
from src.surrogate_gbr import train_surrogate
surr = train_surrogate(mc["params"], mc["wall_loss"])
print(f"  R²_test={surr['r2_test']:.4f}  MAE={surr['mae_test']:.4f}mm")

# [5] Crack colony
print("\n[5/10] Colony simulation — post-ILI, Type I, N=25 cracks...")
from src.crack_colony import simulate_colony
colony = simulate_colony(E_pipe=E_CP_V, spectrum_type="Type_I",
                              post_ILI=True, n_cracks=25, t_end_yr=20, n_t=40, seed=42)
print(f"  Dormant initially: {colony['n_dormant_initial']}/{colony['n_cracks']} ({colony['n_dormant_initial']/colony['n_cracks']:.0%})")
print(f"  Coalescences (BS 7910): {colony['n_coalesced']}  colony damage fraction(20yr): {colony['pof_final']:.3f}")
print(f"  (headline failure PROBABILITY is the Monte Carlo PoF above; the colony is one realisation illustrating the coalescence mechanism)")

# [6] Bayesian update
print("\n[6/10] Bayesian posterior update (particle filter)...")
from src.bayesian_update import multi_inspection_update
from src.pod_ilicurve import sample_a0_post_inspection
particles = sample_a0_post_inspection(1000, seed=42) * 1000
# Two re-ILI detections in the tail of the prior, so the posterior genuinely
# shifts toward the larger, faster-growing flaws (informative update).
inspection_log = [{'t_yr':5.0,'a_obs_mm':2.2,'detected':True},
                   {'t_yr':10.0,'a_obs_mm':3.2,'detected':True}]
bayes = multi_inspection_update(particles, inspection_log, da_dt_mm_yr=0.15)
print(f"  Prior P50={np.percentile(particles,50):.3f}mm -> Posterior P50={bayes['posterior']['P50']:.3f}mm")
print(f"  ESS history: {[round(e,0) for e in bayes['ess_history']]}")

# [7] Model uncertainty report
print("\n[7/10] Model structural uncertainty (Sun et al. 2021)...")
from src.model_uncertainty import uncertainty_quantiles, separate_uncertainty_intervals
q = uncertainty_quantiles()
da_nominal = 0.3e-3 / (365.25*24*3600)  # 0.3 mm/yr in m/s
iv = separate_uncertainty_intervals(da_nominal)
print(f"  ε_model: P05={q['P05']:.3f}  P50={q['P50']:.3f}  P95={q['P95']:.3f}")
print(f"  At da/dt=0.3mm/yr: 90% CI = [{iv['da_dt_lower']*1e3*365.25*24*3600:.3f}, "
      f"{iv['da_dt_upper']*1e3*365.25*24*3600:.3f}] mm/yr")

# [8] Audit chain
print("\n[8/10] Audit chain...")
from src.audit_chain import log_run, get_chain
log_run("baseline", {"a0_m":0.5e-3,"zone":"base","t_yr":20},
        {"a_final_mm":float(bl_base['a'][-1]*1000),"frac_t":float(bl_base['fracture_time_yr'])})
log_run("mc_10k", {"n":10000,"n_t":20,"post_ILI":True},
        {"pof":mc['pof_final'],"p50":float(np.percentile(mc['wall_loss'],50))})
chain = get_chain()
open("assets/audit_chain.json","w").write(chain.to_json())
print(f"  Chain valid: {chain.verify_chain()}  entries: {len(chain)}")

# [9] Visualization
print("\n[9/10] Generating all panels + colony GIF...")
from visualization.plot_all import (plot_crack_3d_evolution, plot_va_loading,
                                         plot_mc_full, plot_pod_and_chemistry,
                                         plot_colony_and_coalescence, plot_bayesian_and_fad,
                                         animate_crack_colony)
plot_crack_3d_evolution(bl_base, bl_haz)
plot_va_loading()
plot_mc_full(mc)
plot_pod_and_chemistry()
plot_colony_and_coalescence(colony)
plot_bayesian_and_fad(bayes)
animate_crack_colony(colony)

# [10] Summary
print(f"\n{'='*72}")
print(f"Complete in {time.time()-t_wall:.0f}s")
n_figs = len(os.listdir("assets/figures"))
print(f"Panels: {n_figs} | Tests: run `pytest tests/ -q` (215 tests)")
print(f"Key mechanisms: Chen-Sutherby-Xing | VA Type I | Dormancy | POD | ε_model | Bayes")
print(f"{'='*72}")

# ── Extended physics blocks (CP curve, H2 blend, RBI optimizer) ─────────────────────────────────────────────────
print("\n[Extended physics]")

# CP optimization (non-monotonic NNpHSCC CGR vs E_pipe)
print("[ext-1] Non-monotonic CP potential curve...")
from src.cp_optimization import CGR_factor_vs_potential_curve, CGR_factor_from_potential
curve = CGR_factor_vs_potential_curve()
print(f"  CGR at E_opt(-750mV)=1.0x  NACE(-850mV)={curve['CGR_at_NACE']:.1f}x  free_corr={curve['CGR_at_free']:.1f}x")

# H2 blending K_IH degradation
print("[ext-2] H2 blending K_IH degradation...")
from src.h2_blending import K_IH_blend, MAOP_blend
for x in [0.0, 0.10, 0.30, 1.0]:
    b = MAOP_blend(x)
    print(f"  x_H2={x:.0%}: K_IH={K_IH_blend(x):.1f} MPa√m  MAOP={b['MAOP_blend_bar']:.0f} bar (-{b['reduction_pct']:.0f}%)")

# Inspection optimizer (RBI re-inspection interval)
print("[ext-3] Risk-based inspection interval...")
from src.crack_colony import simulate_colony
from src.inspection_optimizer import optimal_inspection_interval, PoF_from_mc_trajectory
from scipy.interpolate import interp1d
colony_ext = simulate_colony(E_pipe=E_CP_V, spectrum_type='Type_I', post_ILI=False,
                                 n_cracks=30, t_end_yr=20, n_t=30, seed=42)
PoF_f = interp1d(colony_ext['t_years'], colony_ext['PoF_t'], kind='linear',
                  bounds_error=False, fill_value=(0,colony_ext['PoF_t'][-1]))
insp_result = optimal_inspection_interval(PoF_f, T_max_yr=20, CoF=1e7)
print(f"  T*={insp_result['T_opt_yr']:.1f}yr  T_PoF_limit={insp_result['T_pof_limit_yr']:.1f}yr  PHMSA_5yr={insp_result['phmsa_5yr_compliant']}")

# Extended visualizations
print("[ext-4] Generating extended panels...")
from visualization.plot_advanced import plot_cp_optimization, plot_h2_blending, plot_inspection_optimizer
plot_cp_optimization()
plot_h2_blending()
mc_synth = {'t_years': colony_ext['t_years'], 'PoF_t': colony_ext['PoF_t']}
plot_inspection_optimizer(insp_result, mc_synth)
print("  Extended panels complete")
