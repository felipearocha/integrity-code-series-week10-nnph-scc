"""New visualization panels for Week 10 — additional physics."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.constants import (COLOR_NAVY, COLOR_STEEL, COLOR_RED, COLOR_TEAL,
                            COLOR_CHARCOAL, P_OP_BAR)

def _ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_linewidth(0.7)
    ax.tick_params(direction="out",length=4,width=0.7)
    ax.grid(True,linewidth=0.35,color="#cccccc",alpha=0.7); ax.set_axisbelow(True)


def plot_cp_optimization(out="assets/figures/panel_i_cp_optimization.png"):
    """Panel i: Non-monotonic CP potential vs CGR."""
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.cp_optimization import CGR_factor_vs_potential_curve, E_OPT_NNpH
    from src.constants import E_CP_V, E_FREE_V

    fig, axes = plt.subplots(1, 2, figsize=(13,5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    # Left: CGR factor vs potential
    ax = axes[0]
    curve = CGR_factor_vs_potential_curve(-1.2, -0.55)
    ax.plot(curve['E'], curve['CGR_factor'], color=COLOR_NAVY, linewidth=2.0,
            label='CGR factor (normalised to E_opt)')
    ax.axvline(E_OPT_NNpH, color='#2a8c2a', linewidth=1.2, linestyle='--',
               label=f'E_opt = {E_OPT_NNpH} V CSE\n(minimum CGR for NNpHSCC)')
    ax.axvline(E_CP_V, color=COLOR_RED, linewidth=1.0, linestyle=':',
               label=f'NACE -850 mV criterion\nCGR = {curve["CGR_at_NACE"]:.1f}× optimal')
    ax.axvline(E_FREE_V, color=COLOR_STEEL, linewidth=0.9, linestyle='-.',
               label=f'Free corrosion E = {E_FREE_V} V\nCGR = {curve["CGR_at_free"]:.1f}× optimal')

    ax.fill_betweenx([1, 20], E_OPT_NNpH-0.03, E_OPT_NNpH+0.03,
                      alpha=0.15, color='#2a8c2a', label='Optimal CP window')
    ax.fill_betweenx([1, 20], -1.1, -0.85, alpha=0.08, color=COLOR_RED,
                      label='Overprotection zone')
    ax.fill_betweenx([1, 20], -0.68, -0.55, alpha=0.08, color=COLOR_STEEL,
                      label='Under-protection zone')

    ax.text(-0.90, 8, 'OVERPROTECTION\n(excess H₂ gen)', fontsize=8,
            color=COLOR_RED, ha='center', alpha=0.8)
    ax.text(-0.61, 8, 'UNDER-\nPROTECTION\n(dissolution)', fontsize=8,
            color=COLOR_STEEL, ha='center', alpha=0.8)

    ax.set_xlabel('Pipe-to-soil potential E [V vs CSE]', fontsize=10)
    ax.set_ylabel('CGR factor [—]', fontsize=10)
    ax.set_title('(i)  Non-Monotonic CP Potential vs NNpHSCC CGR\n'
                 '[SOURCE: ScienceDirect — minimum CGR at −750 mV CSE]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    ax.set_ylim(0.5, 20); ax.set_yscale('log')

    # Right: Operational implication — CP voltage vs crack life
    ax2 = axes[1]
    E_vals = np.linspace(-1.1, -0.60, 50)
    from src.cp_optimization import CGR_factor_from_potential
    # Base lifetime at E_opt = 20yr
    T_base = 20.0
    T_vals = T_base / np.array([CGR_factor_from_potential(E) for E in E_vals])

    ax2.plot(E_vals, T_vals, color=COLOR_NAVY, linewidth=1.8, label='Service life estimate')
    ax2.axvline(E_OPT_NNpH, color='#2a8c2a', linewidth=1.2, linestyle='--',
                label=f'E_opt = {E_OPT_NNpH} V (max life)')
    ax2.axvline(E_CP_V, color=COLOR_RED, linewidth=1.0, linestyle=':',
                label=f'NACE -850 mV: {T_base/CGR_factor_from_potential(E_CP_V):.0f} yr')
    ax2.axhline(T_base, color=COLOR_CHARCOAL, linewidth=0.7, alpha=0.5, linestyle='--')
    ax2.set_xlabel('Pipe-to-soil potential E [V vs CSE]', fontsize=10)
    ax2.set_ylabel('Estimated service life [yr]', fontsize=10)
    ax2.set_title("(i')  Service Life vs CP Potential\n"
                  "Optimal CP extends NNpHSCC service life",
                  fontsize=10, fontweight='bold', loc='left')
    ax2.legend(fontsize=8, frameon=False)

    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f'Saved: {out}')


def plot_h2_blending(out="assets/figures/panel_j_h2_blending.png"):
    """Panel j: K_IH degradation with H2 blend + MAOP cascade."""
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.h2_blending import K_IH_blend, HE_index_blend, MAOP_blend
    from src.fad_assessment import maop_comparison

    fig, axes = plt.subplots(1, 3, figsize=(18,5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    x_H2_arr = np.linspace(0, 1, 100)

    # Panel 1: K_IH vs H2 fraction for different CO2 levels
    ax = axes[0]
    for x_CO2, col, lbl in [(0.0, COLOR_NAVY, 'x_CO2 = 0 (pure blend)'),
                              (0.10, COLOR_TEAL, 'x_CO2 = 10%'),
                              (0.40, COLOR_RED, 'x_CO2 = 40% [SOURCE: Cui 2024]')]:
        K_IH_arr = [K_IH_blend(x, x_CO2) for x in x_H2_arr]
        ax.plot(x_H2_arr*100, K_IH_arr, color=col, linewidth=1.4, label=lbl)

    ax.axhline(2.2, color=COLOR_CHARCOAL, linewidth=0.8, linestyle='--', alpha=0.6,
               label='ΔK_th Stage II = 2.2 MPa√m')
    ax.axvline(10, color='#888', linewidth=0.7, linestyle=':', alpha=0.5)
    ax.axvline(30, color='#888', linewidth=0.7, linestyle=':', alpha=0.5)
    ax.set_xlabel('H₂ blend fraction [%]', fontsize=10)
    ax.set_ylabel('K_IH [MPa√m]', fontsize=10)
    ax.set_title('(j)  K_IH Degradation with H₂ Blend\n[SOURCE: DOAJ 2024; Cui et al. 2024]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=8, frameon=False)

    # Panel 2: MAOP cascade
    ax2 = axes[1]
    comp = maop_comparison()
    x_H2_pts = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
    MAOP_pts = [MAOP_blend(x)['MAOP_blend_bar'] for x in x_H2_pts]
    ax2.plot([x*100 for x in x_H2_pts], MAOP_pts, color=COLOR_NAVY,
             linewidth=1.5, marker='o', markersize=5, label='MAOP blend [bar]')
    ax2.axhline(P_OP_BAR, color=COLOR_RED, linewidth=0.9, linestyle='--',
                label=f'Current MAOP = {P_OP_BAR} bar')
    ax2.axhline(comp['B31_8_Class3_bar'], color=COLOR_TEAL, linewidth=0.9,
                linestyle=':', label=f'B31.8 Class 3 = {comp["B31_8_Class3_bar"]:.0f} bar')
    ax2.set_xlabel('H₂ blend fraction [%]', fontsize=10)
    ax2.set_ylabel('MAOP [bar]', fontsize=10)
    ax2.set_title("(j')  MAOP Degradation with H₂ Blend\n"
                  "B31.12 + K_IH reduction from literature",
                  fontsize=10, fontweight='bold', loc='left')
    ax2.legend(fontsize=8, frameon=False)

    # Panel 3: Microstructure zone comparison K_IH
    ax3 = axes[2]
    from src.microstructure import ZONE_PROPERTIES
    zones = list(ZONE_PROPERTIES.keys())
    K_IHs = [ZONE_PROPERTIES[z]['K_IH'] for z in zones]
    da_factors = [ZONE_PROPERTIES[z]['da_dt_factor'] for z in zones]
    zone_labels = ['Base metal\n(bainitic X65)', 'HAZ\n(ferrite-pearlite)',
                   'Vintage X52\n(base metal)', 'Vintage X52\nERW seam weld']

    colors_bar = [COLOR_TEAL, COLOR_NAVY, COLOR_STEEL, COLOR_RED]
    x_pos = np.arange(len(zones))
    bars = ax3.bar(x_pos, K_IHs, color=colors_bar, edgecolor='none', width=0.6)
    for bar, v, f in zip(bars, K_IHs, da_factors):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.3,
                 f'K_IH={v}\n×{f} da/dt', ha='center', fontsize=8,
                 color=COLOR_CHARCOAL)
    ax3.axhline(2.2, color=COLOR_CHARCOAL, linewidth=0.8, linestyle='--', alpha=0.6)
    ax3.set_xticks(x_pos); ax3.set_xticklabels(zone_labels, fontsize=8)
    ax3.set_ylabel('K_IH [MPa√m]', fontsize=10)
    ax3.set_title("(j'')  Zone K_IH Comparison\n[SOURCE: Beavers 2001; Sun 2024]",
                  fontsize=10, fontweight='bold', loc='left')

    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f'Saved: {out}')


def plot_inspection_optimizer(insp_result, mc_result, out="assets/figures/panel_k_inspection_optimizer.png"):
    """Panel k: Risk-based inspection interval optimization."""
    os.makedirs(os.path.dirname(out),exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18,5), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    T_arr = insp_result['T_arr']
    cost_arr = insp_result['cost_arr']
    PoF_arr = insp_result['PoF_arr']

    # Panel 1: Cost rate decomposition
    ax = axes[0]
    C_insp = insp_result['C_insp']
    cost_insp = C_insp / T_arr
    cost_risk  = PoF_arr * insp_result['CoF'] / T_arr
    ax.semilogy(T_arr, cost_arr, color=COLOR_NAVY, linewidth=2.0, label='Total cost rate')
    ax.semilogy(T_arr, cost_insp, color=COLOR_TEAL, linewidth=1.2, linestyle='--',
                label='Inspection cost / T')
    ax.semilogy(T_arr, cost_risk+1, color=COLOR_RED, linewidth=1.2, linestyle=':',
                label='Risk cost (PoF×CoF) / T')
    T_opt = insp_result['T_opt_yr']
    ax.axvline(T_opt, color=COLOR_NAVY, linewidth=1.2, linestyle='--',
               label=f'T* = {T_opt:.1f} yr (optimal)')
    ax.axvline(5.0, color=COLOR_RED, linewidth=0.9, linestyle=':',
               label='PHMSA Class 3 limit: 5 yr')
    ax.set_xlabel('Re-inspection interval T [yr]', fontsize=10)
    ax.set_ylabel('Expected cost rate [$/km/yr]', fontsize=10)
    ax.set_title('(k)  Risk-Based Inspection Interval Optimization\n'
                 '[SOURCE: Li et al. 2018; Abubakirov et al. 2020]',
                 fontsize=10, fontweight='bold', loc='left')
    ax.legend(fontsize=8, frameon=False)

    # Panel 2: PoF trajectory with decision thresholds
    ax2 = axes[1]
    t_mc = mc_result['t_years']
    pof_mc = mc_result['PoF_t']*100
    ax2.plot(t_mc, pof_mc, color=COLOR_NAVY, linewidth=1.5, label='PoF(t) from MC')
    ax2.axhline(insp_result['regulatory_PoF_limit']*100, color=COLOR_RED,
                linewidth=1.0, linestyle='--',
                label=f'Regulatory limit: {insp_result["regulatory_PoF_limit"]*100:.1f}%')
    ax2.axvline(insp_result['T_pof_limit_yr'], color=COLOR_RED, linewidth=0.9,
                linestyle=':', label=f'T_PoF_limit = {insp_result["T_pof_limit_yr"]:.1f} yr')
    ax2.axvline(T_opt, color=COLOR_NAVY, linewidth=1.2, linestyle='--',
                label=f'T* = {T_opt:.1f} yr')
    ax2.set_xlabel('Time [yr]', fontsize=10)
    ax2.set_ylabel('Colony PoF [%]', fontsize=10)
    ax2.set_title("(k')  PoF Trajectory vs Inspection Decision Points",
                  fontsize=10, fontweight='bold', loc='left')
    ax2.legend(fontsize=8, frameon=False)

    # Panel 3: Cost-benefit analysis (different CoF scenarios)
    ax3 = axes[2]
    from src.inspection_optimizer import optimal_inspection_interval, PoF_from_mc_trajectory
    PoF_func = PoF_from_mc_trajectory(mc_result)
    CoF_scenarios = [1e6, 5e6, 1e7, 5e7]
    CoF_labels = ['$1M (leak)', '$5M (major leak)', '$10M (rupture)', '$50M (major rupture)']
    CoF_colors = [COLOR_STEEL, COLOR_TEAL, COLOR_NAVY, COLOR_RED]

    T_opts = [optimal_inspection_interval(PoF_func, T_max_yr=20, CoF=c)['T_opt_yr']
              for c in CoF_scenarios]
    bars3 = ax3.bar(CoF_labels, T_opts, color=CoF_colors, edgecolor='none', width=0.6)
    for bar, v in zip(bars3, T_opts):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.1,
                 f'{v:.1f} yr', ha='center', fontsize=9, color=COLOR_CHARCOAL)
    ax3.axhline(5.0, color=COLOR_RED, linewidth=0.9, linestyle='--',
                label='PHMSA 5-yr limit')
    ax3.set_ylabel('Optimal inspection interval T* [yr]', fontsize=10)
    ax3.set_title("(k'')  T* vs Consequence of Failure Scenario\n"
                  "Higher CoF → shorter optimal interval",
                  fontsize=10, fontweight='bold', loc='left')
    ax3.legend(fontsize=8.5, frameon=False)

    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig); print(f'Saved: {out}')
