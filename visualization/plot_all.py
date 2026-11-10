"""All visualization panels for Week 10 — full physics."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from src.constants import (COLOR_NAVY, COLOR_STEEL, COLOR_RED, COLOR_TEAL,
                            COLOR_CHARCOAL,
                            PIPE_WT, K_IH_BASE_MPa, K_IH_HAZ_MPa,
                            K_TH_STAGE2)

# ICS2-palette sequential map for continuous fields (e.g. model error):
# teal (conservative, eps<1) -> steel (~exact) -> dark red (unsafe, eps>1).
ICS2_SEQ = LinearSegmentedColormap.from_list(
    "ics2_seq", [COLOR_TEAL, COLOR_STEEL, COLOR_RED])

def a_KIH_mm():
    """Depth [mm] where K_I reaches K_IH (base metal, c/a = 4), from the SIF solution."""
    from scipy.optimize import brentq
    from src.constants import K_IH_BASE_MPa, PIPE_WT
    from src.crack_growth import K_I_deeppoint
    return 1e3 * brentq(lambda a: K_I_deeppoint(a, 4 * a) - K_IH_BASE_MPa, 1e-4, 0.95 * PIPE_WT)


def a_KIH_haz_mm():
    """Depth [mm] where K_I (with weld residual stress) reaches K_IH in the HAZ, c/a = 4."""
    from scipy.optimize import brentq
    from src.crack_growth import K_I_deeppoint, K_I_residual
    return 1e3 * brentq(lambda a: K_I_deeppoint(a, 4 * a) + K_I_residual(a) - K_IH_HAZ_MPa,
                        1e-5, 0.95 * PIPE_WT)


# Leak criterion of the limit state (failure_mode): growth stops at 80 % of the wall
from src.constants import A_CRIT_FRAC
A_LEAK_MM = PIPE_WT * A_CRIT_FRAC * 1000


def _colony_span_m(colony_result):
    """Axial extent [m] of the simulated colony, with a 5 % margin."""
    x_all=np.concatenate([np.asarray(colony_result['x']),
                          np.asarray(colony_result['x_after_coalescence'])])
    return 1.05*float(x_all.max())


def _ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_linewidth(0.7)
    ax.tick_params(direction="out",length=4,width=0.7)
    ax.grid(True,linewidth=0.35,color="#cccccc",alpha=0.7); ax.set_axisbelow(True)

def plot_crack_3d_evolution(res_base, res_haz, out="assets/figures/panel_ab_crack_3d.png"):
    os.makedirs(os.path.dirname(out),exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)
    t=res_base['t_yr']; a=res_base['a']*1000; c=res_base['c']*1000
    t_h=res_haz['t_yr']; a_h=res_haz['a']*1000; c_h=res_haz['c']*1000
    ft=res_haz['fracture_time_yr']
    t_ev,a_ev=t_h,a_h
    if np.isfinite(ft):   # add the failure event, which falls between output steps
        k=int(np.searchsorted(t_h,ft))
        t_ev=np.insert(t_h,k,ft); a_ev=np.insert(a_h,k,res_haz['a_fail']*1000)
    # Panel a: a(t) for base and HAZ
    ax=axes[0]
    ax.plot(t,a,color=COLOR_NAVY,linewidth=1.5,label='Base metal (bainitic X65)')
    ax.plot(t_ev,a_ev,color=COLOR_RED,linewidth=1.5,linestyle='--',label='HAZ (ferrite-pearlite)')
    ax.axhline(a_KIH_mm(),color=COLOR_TEAL,linewidth=0.8,linestyle=':',label=f'K$_{{IH}}$ base crossing {a_KIH_mm():.1f} mm')
    ax.axhline(a_KIH_haz_mm(),color=COLOR_RED,linewidth=0.7,linestyle=':',alpha=0.6,
               label=f'K$_{{IH}}$ HAZ crossing {a_KIH_haz_mm():.1f} mm (with residual stress)')
    ax.axhline(A_LEAK_MM,color=COLOR_CHARCOAL,linewidth=0.8,linestyle='-.',alpha=0.7,
               label=f'Leak criterion 0.8·t = {A_LEAK_MM:.2f} mm (failure)')
    if np.isfinite(ft):
        ax.annotate(f'HAZ fails by {res_haz["failure_mode"]} at t = {ft:.2f} yr,\n'
                    f'a = {res_haz["a_fail"]*1000:.2f} mm; curve frozen after',
                    (ft, res_haz['a_fail']*1000), xytext=(ft+1.5, A_LEAK_MM*0.25), fontsize=8,
                    color=COLOR_RED, arrowprops=dict(arrowstyle='->', color=COLOR_RED, lw=0.7))
    ax.set_ylim(0, A_LEAK_MM*1.15)
    ax.set_xlabel('Time [yr]',fontsize=10); ax.set_ylabel('Crack depth a [mm]',fontsize=10)
    ax.set_title('(a)  3D Crack Depth a(t)\nBase vs HAZ zone [SOURCE: Beavers et al. 2001]',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8,frameon=False,loc='center right',bbox_to_anchor=(1.0,0.62))  # clear of the threshold lines
    # Panel b: aspect ratio c/a
    ax=axes[1]
    ax.plot(t, c/a,color=COLOR_NAVY,linewidth=1.5,label='Base (a/c coupled ODEs)')
    ax.plot(t_h,c_h/a_h,color=COLOR_RED,linewidth=1.5,linestyle='--',label='HAZ')
    ax.set_xlabel('Time [yr]',fontsize=10); ax.set_ylabel('Crack aspect ratio c/a [—]',fontsize=10)
    ax.set_title('(b)  Crack Shape Evolution c/a\n[SOURCE: Newman-Raju 1981]',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8,frameon=False)
    # Panel c: K_I(t) vs thresholds — visualises Stage I/II transition and fracture margin
    ax=axes[2]
    KI_b = res_base['KI']
    KI_h = res_haz['KI']
    ax.plot(t,KI_b,color=COLOR_NAVY,linewidth=1.5,label='K$_I$(t) base')
    ax.plot(t_h,KI_h,color=COLOR_RED,linewidth=1.5,linestyle='--',label='K$_I$(t) HAZ')
    ax.axhline(K_TH_STAGE2,color=COLOR_CHARCOAL,linewidth=0.8,linestyle=':',
               label=f'ΔK$_{{th}}$ Stage II = {K_TH_STAGE2} MPa√m')
    ax.axhline(K_IH_BASE_MPa,color=COLOR_TEAL,linewidth=0.9,linestyle='--',
               label=f'K$_{{IH}}$ base = {K_IH_BASE_MPa} MPa√m')
    ax.axhline(K_IH_HAZ_MPa,color=COLOR_RED,linewidth=0.9,linestyle='--',alpha=0.6,
               label=f'K$_{{IH}}$ HAZ = {K_IH_HAZ_MPa} MPa√m')
    dormant_b = res_base['dormant']
    dormant_h = res_haz['dormant']
    ax.fill_between(t,0,KI_b,where=dormant_b,alpha=0.10,color=COLOR_NAVY,
                     step='mid',label='Stage I (dormant) base')
    ax.fill_between(t_h,0,KI_h,where=dormant_h,alpha=0.15,color=COLOR_RED,
                     step='mid',label='Stage I (dormant) HAZ')
    ax.set_xlabel('Time [yr]',fontsize=10); ax.set_ylabel('K$_I$ [MPa√m]',fontsize=10)
    ax.set_title('(c)  K$_I$(t) vs Dormancy & Fracture Thresholds\n'
                 '[SOURCE: Zhao et al. 2017; BS 7910:2019]',
                 fontsize=10,fontweight='bold',loc='left')
    # Show the whole HAZ curve; it is flat after the flaw goes critical (growth stopped)
    ax.set_ylim(0,max(K_IH_BASE_MPa*1.1, float(np.max(KI_b)*1.05), float(np.max(KI_h)*1.08)))
    ax.legend(fontsize=7.5,frameon=False,loc='center right',bbox_to_anchor=(1.0,0.72))
    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')

def plot_va_loading(out="assets/figures/panel_d_va_loading.png"):
    """Panel d: variable amplitude loading spectrum and da/dt comparison."""
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.pressure_spectrum import PressureSpectrum, da_dN_Chen_Xing, da_dt_variable_amplitude
    from src.crack_growth import K_I_deeppoint, delta_K
    from src.constants import C_H_BULK_X65, P_OP_BAR

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    # Left: da/dt vs crack depth for 3 spectrum types
    ax=axes[0]
    # Same rate law the integrator uses (da_dt_variable_amplitude), base metal, c/a = 4
    a_vals = np.linspace(0.5e-3, 6e-3, 60)
    curves = {}
    for sp_type, col, ls, lbl in [("Type_I",COLOR_RED,'-',"Type I (underload, downstream compressor)"),
                                    ("Type_II",COLOR_NAVY,'--',"Type II (constant amplitude, mid-pipe)"),
                                    ("Type_III",COLOR_TEAL,':',"Type III (mixed VA)")]:
        sp = PressureSpectrum(sp_type)
        da_dt_arr = np.array([da_dt_variable_amplitude(
            a, sp, C_H_BULK_X65,
            K_I_func=lambda am, P, c=4*a: K_I_deeppoint(am, c, P),
            delta_K_func=lambda am, P, R, c=4*a: delta_K(am, c, P, R),
            spectrum_type=sp_type) for a in a_vals]) * 1e3 * 365.25*24*3600
        curves[sp_type] = da_dt_arr
        ax.semilogy(a_vals*1000, da_dt_arr, color=col, linewidth=1.4, linestyle=ls, label=lbl)
    ratio = curves["Type_I"] / np.where(curves["Type_II"] > 0, curves["Type_II"], np.nan)
    r_lo, r_hi = np.nanmin(ratio), np.nanmax(ratio)
    ax.text(0.98, 0.30, f'Type I / Type II = {r_lo:.2f}'
            + ('' if r_hi - r_lo < 0.005 else f'–{r_hi:.2f}') + ' over this depth range',
            transform=ax.transAxes, ha='right', fontsize=8, color=COLOR_CHARCOAL)
    ax.axhline(0.3, color='#888', linewidth=0.8, linestyle='--', alpha=0.7, label='CEPA field obs.: 0.3 mm/yr')
    ax.axhline(0.63, color='#888', linewidth=0.8, linestyle=':', alpha=0.7, label='CEPA field obs.: 0.63 mm/yr')
    ax.axvline(1.0, color=COLOR_CHARCOAL, linewidth=0.7, linestyle='-.',alpha=0.5, label='Stage I/II boundary ~1mm')
    ax.set_xlabel('Crack depth a [mm]',fontsize=10); ax.set_ylabel('da/dt [mm yr$^{-1}$]',fontsize=10)
    ax.set_title('(d)  VA Loading Spectrum Types — da/dt\nType I (underload) vs II (CA) vs III (mixed)',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=7.5,frameon=False,loc='upper left')

    # Right: frequency effect on da/dt at fixed a=2mm
    ax2=axes[1]
    freqs = np.logspace(-6, 0, 60)
    sp_ref = PressureSpectrum('Type_I')
    da_vals = []
    for f in freqs:
        sp_ref.f_major = f
        a = 2e-3; c = 8e-3
        K_max = K_I_deeppoint(a, c, P_OP_BAR)
        dK = delta_K(a, c, P_OP_BAR, 0.55)
        da_dN = da_dN_Chen_Xing(a, K_max, dK, f, C_H_BULK_X65)
        da_dt = f * da_dN
        da_vals.append(da_dt * 1e3 * 365.25*24*3600)
    ax2.loglog(freqs, da_vals, color=COLOR_NAVY, linewidth=1.5)
    ax2.axvline(1e-3, color=COLOR_RED, linewidth=1.0, linestyle='--',
                label='f_crit = 10$^{-3}$ Hz [SOURCE: Xing et al.]')
    ax2.axvline(1.16e-5, color=COLOR_TEAL, linewidth=0.9, linestyle=':',
                label='f_pipeline = 1.16×10$^{-5}$ Hz\n(1 cycle/day)')
    ax2.set_xlabel('Loading frequency f [Hz]',fontsize=10)
    ax2.set_ylabel('da/dt [mm yr$^{-1}$]',fontsize=10)
    ax2.set_title("(d')  Frequency Saturation Effect at a = 2 mm\nBelow f_crit: environment-dominated SCC regime",fontsize=10,fontweight='bold',loc='left')
    ax2.legend(fontsize=8.5,frameon=False)
    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')

def plot_mc_full(mc_result, out="assets/figures/panel_e_mc_full.png"):
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.monte_carlo import spearman_sensitivity
    wl=mc_result['wall_loss']; limit=a_KIH_mm()
    rho=spearman_sensitivity(mc_result)
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)
    # CDF
    ax=axes[0]; s=np.sort(wl); cdf=np.arange(1,len(s)+1)/len(s)
    ax.step(s,cdf,color=COLOR_NAVY,linewidth=1.2,where='post')
    xmax=float(s.max())
    for pct,lbl,y in [(50,'P50',0.40),(90,'P90',0.62),(95,'P95',0.70)]:
        v=np.percentile(wl,pct)
        ax.axvline(v,color=COLOR_STEEL,linewidth=0.6,alpha=0.5)
        right=v>0.8*xmax   # keep labels near the right edge inside the axes
        ax.text(v-0.1 if right else v+0.1,y,f'{lbl} = {v:.2f} mm',fontsize=7.5,color=COLOR_STEEL,
                va='top',ha='right' if right else 'left')
    ax.set_xlabel('Crack depth at t* = 20 yr [mm]',fontsize=10)
    ax.set_ylabel('CDF [—]',fontsize=10)
    ax.set_title('(e)  Crack Depth CDF — N = 10,000\nPost-ILI POD-corrected a$_0$ + model error ε',fontsize=10,fontweight='bold',loc='left')
    ax.axvline(limit,color=COLOR_RED,linewidth=0.9,linestyle=':',label=f'K$_{{IH}}$ crossing {limit:.1f} mm')
    n_leak=int(np.sum(wl>=A_LEAK_MM-1e-6))
    ax.axvline(A_LEAK_MM,color=COLOR_CHARCOAL,linewidth=0.8,linestyle='-.',alpha=0.7,
               label=f'Leak criterion 0.8·t = {A_LEAK_MM:.2f} mm ({n_leak/len(wl):.1%} of samples failed there)')
    ax.set_ylim(0,1.05); ax.legend(fontsize=8,frameon=False,loc='upper left')
    # Tornado
    ax=axes[1]
    names=list(rho.keys()); rhos=list(rho.values())
    order=sorted(range(len(rhos)),key=lambda i:abs(rhos[i]))
    names=[names[i] for i in order]; rhos=[rhos[i] for i in order]
    colors=[COLOR_RED if r>0 else COLOR_STEEL for r in rhos]
    bars=ax.barh(names,rhos,color=colors,edgecolor='none',height=0.6)
    ax.axvline(0,color='black',linewidth=0.7)
    # room for the value labels so negative ones do not run into the tick labels
    ax.set_xlim(min(min(rhos),0)-0.1, max(max(rhos),0)+0.1)
    for bar,r in zip(bars,rhos):
        ax.text(r+0.008*np.sign(r),bar.get_y()+bar.get_height()/2,
                f'{r:.3f}',va='center',ha='left' if r>=0 else 'right',fontsize=8)
    ax.set_xlabel('Spearman ρ [—]',fontsize=10)
    ax.set_title("(e')  Sensitivity Tornado\n8 parameters: aleatory + epistemic",fontsize=10,fontweight='bold',loc='left')
    # Model error distribution
    ax=axes[2]
    eps=mc_result['params']['eps_model']
    ax.hist(eps,bins=40,color=COLOR_STEEL,alpha=0.7,density=True,edgecolor='none')
    from scipy.stats import lognorm as _ln
    from src.model_uncertainty import MODEL_ERROR_MEAN, MODEL_ERROR_COV, lognormal_params_from_moments
    mu_ln,sig_ln=lognormal_params_from_moments(MODEL_ERROR_MEAN,MODEL_ERROR_COV)
    x_pdf=np.linspace(0.01,5,200)
    pdf=_ln.pdf(x_pdf,s=sig_ln,scale=np.exp(mu_ln))
    ax.plot(x_pdf,pdf,color=COLOR_RED,linewidth=1.5,label=f'LogNormal(μ={MODEL_ERROR_MEAN}, COV=61.2%)')
    ax.axvline(1.0,color=COLOR_NAVY,linewidth=0.9,linestyle='--',label='ε=1.0 (exact model)')
    ax.axvline(MODEL_ERROR_MEAN,color=COLOR_RED,linewidth=0.8,linestyle=':',label=f'Mean={MODEL_ERROR_MEAN}')
    ax.set_xlabel('Model structural uncertainty ε [—]',fontsize=10)
    ax.set_ylabel('Density [—]',fontsize=10)
    ax.set_title("(e'')  Epistemic Model Error Distribution\n[SOURCE: Sun, Zhou & Kang 2021 — COV=61.2%]",fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8,frameon=False); ax.set_xlim(0,6)
    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')

def plot_pod_and_chemistry(out="assets/figures/panel_f_pod_chemistry.png"):
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.pod_ilicurve import pod
    from src.crack_tip_chemistry import crack_tip_pH, C_H_entry_corrected
    from src.constants import SOIL_PH

    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    # POD curve
    ax=axes[0]
    a_vals=np.linspace(0,15,300)
    pod_vals=np.array([pod(a) for a in a_vals])
    ax.plot(a_vals,pod_vals*100,color=COLOR_NAVY,linewidth=1.5,label='EMAT/TFI POD(a)')
    ax.fill_between(a_vals,0,pod_vals*100,alpha=0.15,color=COLOR_NAVY)
    ax.axvline(4.0,color=COLOR_RED,linewidth=0.8,linestyle='--',label='a$_{90}$ = 4 mm (POD=90%)')
    ax.axvline(2.0,color=COLOR_TEAL,linewidth=0.8,linestyle=':',label='a$_{50}$ ≈ 2 mm')
    ax.set_xlabel('Crack depth a [mm]',fontsize=10); ax.set_ylabel('POD [%]',fontsize=10)
    ax.set_title('(f)  ILI Tool POD Curve (EMAT)\n[ASSUMED: Weibull, a₉₀=4mm]',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8.5,frameon=False)

    # Post-ILI a0 distribution vs prior
    ax2=axes[1]
    from scipy.stats import lognorm as _ln
    from src.constants import A0_MEAN, A0_STD
    mu_ln=np.log(A0_MEAN)-0.5*np.log(1+(A0_STD/A0_MEAN)**2)
    sig_ln=np.sqrt(np.log(1+(A0_STD/A0_MEAN)**2))
    a_mm=np.linspace(0,8,300)
    prior_pdf=_ln.pdf(a_mm/1000,s=sig_ln,scale=np.exp(mu_ln))/1000
    miss_pdf=prior_pdf*np.array([1-pod(a) for a in a_mm])
    miss_pdf/=(miss_pdf.max()+1e-20); miss_pdf*=prior_pdf.max()
    ax2.plot(a_mm,prior_pdf,color=COLOR_NAVY,linewidth=1.3,label='Prior field population')
    ax2.plot(a_mm,miss_pdf,color=COLOR_RED,linewidth=1.3,linestyle='--',label='Post-ILI (cracks ILI missed)')
    ax2.set_xlabel('Initial crack depth a₀ [mm]',fontsize=10); ax2.set_ylabel('Density [mm$^{-1}$]',fontsize=10)
    ax2.set_title("(f')  Post-ILI a₀ Distribution\nOnly undetected cracks remain in risk pool",fontsize=10,fontweight='bold',loc='left')
    ax2.legend(fontsize=8.5,frameon=False)

    # Crack tip chemistry — Δ_pH and C_H enhancement vs c/a (twin axis for visible magnitude)
    ax3=axes[2]
    ca_vals = np.logspace(0, 1.7, 60)
    delta_pH = np.zeros_like(ca_vals)
    enhance = np.zeros_like(ca_vals)
    a_ref = 2e-3
    C_H_ref = 1.0
    for k, ca in enumerate(ca_vals):
        c_m = a_ref * ca
        pH_tip = crack_tip_pH(a_ref, c_m, SOIL_PH)
        delta_pH[k] = SOIL_PH - pH_tip
        C_H_tip = C_H_entry_corrected(C_H_ref, pH_tip, SOIL_PH)
        enhance[k] = C_H_tip / C_H_ref
    l1, = ax3.plot(ca_vals, delta_pH, color=COLOR_NAVY, linewidth=1.6,
                    label='ΔpH = pH$_{bulk}$ − pH$_{tip}$')
    ax3.set_xscale('log')
    ax3.set_xlabel('Crack aspect ratio c/a [—]', fontsize=10)
    ax3.set_ylabel('ΔpH (acidification at tip) [—]', fontsize=10, color=COLOR_NAVY)
    ax3.tick_params(axis='y', labelcolor=COLOR_NAVY)
    ax3_twin = ax3.twinx()
    l2, = ax3_twin.plot(ca_vals, enhance, color=COLOR_RED, linewidth=1.6, linestyle='--',
                        label='C$_H$ enhancement (Turnbull)')
    ax3_twin.set_ylabel('C$_{H,tip}$ / C$_{H,bulk}$ [—]', fontsize=10, color=COLOR_RED)
    ax3_twin.tick_params(axis='y', labelcolor=COLOR_RED)
    ax3_twin.spines['top'].set_visible(False)
    ax3.axvspan(5, 10, alpha=0.08, color=COLOR_TEAL)
    ax3.text(7.07, max(delta_pH)*0.92, 'Typical\nNNpHSCC\nc/a', ha='center', fontsize=8,
             color=COLOR_TEAL, alpha=0.9)
    ax3.legend(handles=[l1, l2], fontsize=8, frameon=False, loc='upper left')
    ax3.set_title("(f'')  Crack Tip Acidification & H Enhancement\n"
                  "[SOURCE: Turnbull 1993; pH drops by Fe²⁺ hydrolysis]",
                  fontsize=10, fontweight='bold', loc='left')

    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')

def plot_colony_and_coalescence(colony_result, out="assets/figures/panel_g_colony.png"):
    os.makedirs(os.path.dirname(out),exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)
    t=colony_result['t_years']; PoF=colony_result['PoF_t']
    ax=axes[0]
    ax.plot(t,PoF*100,color=COLOR_RED,linewidth=1.5,label='Colony damage fraction')
    ax.set_xlabel('Time [yr]',fontsize=10); ax.set_ylabel('Colony damage fraction [%]',fontsize=10)
    n_dorm=colony_result['n_dormant_initial']
    n_tot=colony_result['n_cracks']
    ax.set_title(f'(g)  Crack Colony Damage Fraction\n{n_dorm}/{n_tot} cracks initially dormant ({n_dorm/n_tot:.0%})',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8.5,frameon=False)

    # Crack positions before/after coalescence — use real axial coordinates
    ax2=axes[1]
    x=colony_result['x']; a=colony_result['a0']*1000
    zone_col=[COLOR_RED if z=='haz' else COLOR_NAVY for z in colony_result['zone']]
    ax2.scatter(x,a,c=zone_col,s=30,alpha=0.8,zorder=4,label='Initial a₀ by zone')
    # Coalesced positions: actual x_after_coalescence (fixed: was range(len(a_c)))
    x_coal=colony_result['x_after_coalescence']
    a_coal=colony_result['a_after_coalescence']*1000
    ax2.scatter(x_coal,a_coal,marker='x',s=80,color=COLOR_TEAL,zorder=5,
                linewidth=1.5,
                label=f'After BS 7910 coalescence ({colony_result["n_coalesced"]} merges)')
    p_haz=mpatches.Patch(color=COLOR_RED,label='Initial a$_0$, HAZ')
    p_base=mpatches.Patch(color=COLOR_NAVY,label='Initial a$_0$, base metal')
    ax2.legend(handles=[p_haz,p_base,
                         plt.Line2D([0],[0],marker='x',color=COLOR_TEAL,linestyle='None',
                                    markersize=8,markeredgewidth=1.5,
                                    label=f'Flaws after coalescence, a at 20 yr ({colony_result["n_coalesced"]} merges)')],
                fontsize=8,frameon=False,loc='center right')
    # The colony is one cluster (COLONY_SPACING_M apart), not spread over PIPE_L
    span=_colony_span_m(colony_result)
    ax2.set_xlim(0, span)
    ax2.set_xlabel('Axial position [m]',fontsize=10); ax2.set_ylabel('Crack depth a [mm]',fontsize=10)
    ax2.set_title(f"(g')  Colony Spatial Map (one cluster, {span:.2f} m axial span)\n"
                  "BS 7910:2019 Coalescence Applied",
                  fontsize=10,fontweight='bold',loc='left')

    # Model error spread
    ax3=axes[2]
    eps=colony_result['eps_model']
    a0=colony_result['a0']*1000
    frac=[r['a'][-1]*1000 for r in colony_result['results']]
    sc3=ax3.scatter(a0,frac,c=eps,cmap=ICS2_SEQ,s=20,alpha=0.8,vmin=0.3,vmax=3.0)
    fig.colorbar(sc3,ax=ax3,fraction=0.046,pad=0.04).set_label('ε_model [—]',fontsize=9)
    ax3.axhline(A_LEAK_MM,color=COLOR_CHARCOAL,linewidth=0.8,linestyle='-.',alpha=0.7,
                label=f'Leak criterion 0.8·t = {A_LEAK_MM:.2f} mm')
    ax3.legend(fontsize=8,frameon=False,loc='center right')
    ax3.set_xlabel('Initial a₀ [mm]',fontsize=10); ax3.set_ylabel('Final a(t=20yr) [mm]',fontsize=10)
    ax3.set_title("(g'')  Model Error Spread per Crack\nEpistemic uncertainty visible in scatter",fontsize=10,fontweight='bold',loc='left')

    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')

def plot_bayesian_and_fad(bayesian_result, out="assets/figures/panel_h_bayes_fad.png"):
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.fad_assessment import fad_curve, Lr_max_value, maop_comparison, fad_trajectory
    from src.constants import P_OP_BAR

    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    # Bayesian update — weighted-histogram posterior densities (a proper density,
    # not a line over unsorted particles). particle_history/weight_history are
    # kept aligned across resampling by multi_inspection_update.
    ax=axes[0]
    wh_all = bayesian_result['weight_history']
    ph = bayesian_result.get('particle_history',
                             [bayesian_result['particles']] * len(wh_all))
    all_p = np.concatenate([np.asarray(p) for p in ph])
    bins = np.linspace(float(all_p.min()), float(all_p.max()), 28)
    centers = 0.5*(bins[:-1]+bins[1:])
    cols = [COLOR_STEEL, COLOR_TEAL, COLOR_RED]
    for i,(pp,ww) in enumerate(zip(ph, wh_all)):
        dens,_ = np.histogram(np.asarray(pp), bins=bins, weights=np.asarray(ww), density=True)
        lbl = 'Prior' if i==0 else f'After inspection {i}'
        ax.plot(centers, dens, color=cols[min(i,2)], alpha=0.8, linewidth=1.5, label=lbl)
    ax.set_xlabel('Initial crack depth a₀ [mm]',fontsize=10)
    ax.set_ylabel('Posterior density [mm⁻¹]',fontsize=10)
    ax.set_title('(h)  Bayesian Posterior Update\n[SOURCE: Straub 2004 — particle filter ILI]',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8.5,frameon=False)

    # FAD
    ax2=axes[1]; _ax(ax2)
    Lr_max=Lr_max_value()
    Lr_line=np.linspace(0,Lr_max*1.08,400)
    Kr_line=fad_curve(Lr_line)
    ax2.plot(Lr_line,Kr_line,color=COLOR_NAVY,linewidth=1.5,label='API 579-1 Level 2')
    ax2.axvline(Lr_max,color=COLOR_NAVY,linewidth=0.8,linestyle='--')
    ax2.fill_between(Lr_line,Kr_line,1.05,alpha=0.07,color=COLOR_RED,label='Unacceptable region')
    wl_arr=np.linspace(0,5e-3,100)
    traj_b=fad_trajectory(wl_arr)
    traj_h=fad_trajectory(wl_arr, Kmat=K_IH_HAZ_MPa)
    ax2.plot(traj_b['Lr'],traj_b['Kr'],color=COLOR_TEAL,linewidth=1.3,label='Base metal (K$_{mat}$ = 70 MPa√m), a = 0–5 mm')
    ax2.plot(traj_h['Lr'],traj_h['Kr'],color=COLOR_RED,linewidth=1.3,linestyle='--',
             label=f'HAZ screening (K$_{{mat}}$ = K$_{{IH}}$ = {K_IH_HAZ_MPa} MPa√m) [ASSUMED]')
    # The HAZ path leaves the plotted range; mark where it crosses the FAD
    bad=np.flatnonzero(traj_h['status']=='unacceptable')
    if bad.size:
        k=bad[0]
        ax2.plot(traj_h['Lr'][k],traj_h['Kr'][k],'o',color=COLOR_RED,ms=5)
        ax2.annotate(f'HAZ path crosses\nthe FAD at a = {wl_arr[k]*1e3:.2f} mm;\n'
                     f'K$_r$ = {traj_h["Kr"].max():.1f} at a = 5 mm\n(off scale)',
                     (traj_h['Lr'][k],traj_h['Kr'][k]),xytext=(0.03,0.50),textcoords='axes fraction',
                     fontsize=8,color=COLOR_RED,arrowprops=dict(arrowstyle='->',color=COLOR_RED,lw=0.7))
    ax2.set_xlim(0,Lr_max*1.12); ax2.set_ylim(0,1.10)
    ax2.set_xlabel('L$_r$ = σ$_{ref}$/σ$_Y$ [—]',fontsize=10); ax2.set_ylabel('K$_r$ = K$_I$/K$_{mat}$ [—]',fontsize=10)
    ax2.set_title("(h')  FAD — Base vs HAZ\nHAZ screened with K$_{mat}$ = K$_{IH}$ [ASSUMED]",fontsize=10,fontweight='bold',loc='left')
    ax2.legend(fontsize=8,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.14),ncol=2)

    # MAOP comparison
    ax3=axes[2]
    comp=maop_comparison()
    labels=['B31.8\nClass 2','B31.8\nClass 3','B31.12\nH2']
    vals=[comp['B31_8_Class2_bar'],comp['B31_8_Class3_bar'],comp['B31_12_H2_bar']]
    colors_bar=[COLOR_TEAL,COLOR_NAVY,COLOR_RED]
    bars=ax3.bar(labels,vals,color=colors_bar,edgecolor='none',width=0.5)
    for bar,v in zip(bars,vals):
        ax3.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v:.0f} bar',
                 ha='center',fontsize=11,fontweight='bold',color=COLOR_CHARCOAL)
    ax3.axhline(P_OP_BAR,color='#888',linewidth=0.9,linestyle='--',label=f'Current MAOP = {P_OP_BAR} bar')
    ax3.set_ylabel('MAOP [bar]',fontsize=10)
    ax3.set_title("(h'')  MAOP — B31.8 vs B31.12\nPHMSA §192.611(a)(4) context",fontsize=10,fontweight='bold',loc='left')
    ax3.legend(fontsize=8.5,frameon=False)

    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')


def animate_crack_colony(colony_result, out="assets/animations/nnph_scc_crack_colony.gif",
                          fps: int = 7, n_frames: int = 60):
    """
    Animated GIF: colony of NNpHSCC cracks growing along the colony's axial span
    plus the simultaneous Colony damage fraction trajectory.

    Meets the ICS2 QC Gate-2 GIF standard: >=50 interpolated frames, DPI 150, a
    progress bar, a per-frame time stamp in the suptitle, start/end holds, and an
    ICS2 brand watermark. Top axes: scatter of (axial position x, crack depth a),
    markers grow and turn larger as a crack crosses K_IH, coloured by zone
    (base/HAZ). Bottom axes: running Colony damage fraction.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t_years = np.asarray(colony_result['t_years'], dtype=float)
    PoF = np.asarray(colony_result['PoF_t'], dtype=float)
    x = colony_result['x']
    zones = colony_result['zone']
    a_traj = np.array([r['a']*1000 for r in colony_result['results']])  # [n_cracks, n_steps]
    res_t = np.asarray(colony_result['results'][0]['t_yr'], dtype=float)

    # Decouple frame count from simulation resolution: interpolate every crack
    # trajectory and the PoF curve onto a fine, uniform time grid so the GIF is
    # smooth and always clears the 50-frame minimum.
    n_real = max(int(n_frames), 50)
    t_fine = np.linspace(res_t[0], res_t[-1], n_real)
    a_fine = np.vstack([np.interp(t_fine, res_t, a_traj[i]) for i in range(a_traj.shape[0])])
    pof_fine = np.interp(t_fine, t_years, PoF) * 100.0
    a_kih = a_KIH_mm()

    # Start/end holds for readability: repeat first/last frame so the eye rests
    # (~430 ms start, ~715 ms end at fps=7) while middle frames stay ~143 ms.
    hold_start, hold_end = 3, 5
    order = ([0]*hold_start) + list(range(n_real)) + ([n_real-1]*hold_end)

    zone_col = [COLOR_RED if z == 'haz' else COLOR_NAVY for z in zones]

    fig, (ax_top, ax_bot, ax_prog) = plt.subplots(
        3, 1, figsize=(10, 7.4), constrained_layout=True,
        gridspec_kw={'height_ratios': [2.2, 1, 0.10]})
    fig.patch.set_facecolor('white')
    _ax(ax_top); _ax(ax_bot)

    a_max = float(a_fine.max())
    ax_top.set_xlim(0, _colony_span_m(colony_result))  # one cluster, not PIPE_L
    ax_top.set_ylim(0, max(PIPE_WT*1000*0.6, a_max*1.05))
    ax_top.axhline(a_KIH_mm(), color=COLOR_RED, linewidth=0.8, linestyle=':')
    ax_top.axhline(A_LEAK_MM, color=COLOR_CHARCOAL, linewidth=0.8, linestyle='-.', alpha=0.7)
    ax_top.set_xlabel('Axial position [m]')
    ax_top.set_ylabel('Crack depth a [mm]')
    ax_top.set_title('NNpHSCC Crack Colony — Spatial Growth',
                      fontsize=11, fontweight='bold', loc='left')

    scat = ax_top.scatter(x, a_fine[:, 0], c=zone_col, s=40,
                           alpha=0.85, edgecolor='white', linewidth=0.4)
    p_base = mpatches.Patch(color=COLOR_NAVY, label='Base metal')
    p_haz = mpatches.Patch(color=COLOR_RED, label='HAZ')
    ax_top.legend(handles=[p_base, p_haz,
                            plt.Line2D([0],[0],color=COLOR_RED,linestyle=':',
                                       label=f'K$_{{IH}}$ base crossing {a_KIH_mm():.1f} mm'),
                            plt.Line2D([0],[0],color=COLOR_CHARCOAL,linestyle='-.',
                                       label=f'Leak criterion 0.8·t = {A_LEAK_MM:.2f} mm')],
                   fontsize=8, frameon=False, loc='center right')

    ax_bot.set_xlim(0, t_fine[-1])
    ax_bot.set_ylim(0, max(pof_fine.max()*1.1, 5.0))
    ax_bot.set_xlabel('Time [yr]')
    ax_bot.set_ylabel('Colony damage fraction [%]')
    ax_bot.set_title('Colony damage fraction (share of cracks past the failure limit state)',
                      fontsize=10, fontweight='bold', loc='left')
    pof_line, = ax_bot.plot([], [], color=COLOR_RED, linewidth=1.8)
    pof_dot, = ax_bot.plot([], [], 'o', color=COLOR_RED, markersize=6)

    # Progress bar (thin bottom strip), short enough to clear the watermark
    PROG_W = 0.62
    ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1); ax_prog.axis('off')
    ax_prog.add_patch(mpatches.Rectangle((0, 0.25), PROG_W, 0.5,
                       facecolor='#e6e6e6', edgecolor='none'))
    prog_fg = mpatches.Rectangle((0, 0.25), 0.0, 0.5,
                                  facecolor=COLOR_STEEL, edgecolor='none')
    ax_prog.add_patch(prog_fg)

    # ICS2 brand watermark, bottom-right
    fig.text(0.99, 0.012, 'Infinity Growth · Integrity Code Series · Week 10',
             ha='right', va='bottom', fontsize=7, color=COLOR_CHARCOAL, alpha=0.55)

    def update(display_idx):
        k = order[display_idx]
        a_now = a_fine[:, k]
        scat.set_offsets(np.column_stack([x, a_now]))
        scat.set_sizes(np.where(a_now >= a_kih, 110, 40))
        pof_line.set_data(t_fine[:k+1], pof_fine[:k+1])
        pof_dot.set_data([t_fine[k]], [pof_fine[k]])
        prog_fg.set_width(PROG_W*k/(n_real-1))
        fig.suptitle(f'NNpHSCC Crack Colony Evolution   |   t = {t_fine[k]:.1f} yr',
                     fontsize=12, fontweight='bold', color=COLOR_CHARCOAL)
        return scat, pof_line, pof_dot, prog_fg

    anim = FuncAnimation(fig, update, frames=len(order), interval=1000/fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')
