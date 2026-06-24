"""All visualization panels for Week 10 — full physics."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from src.constants import (COLOR_NAVY, COLOR_STEEL, COLOR_RED, COLOR_TEAL,
                            COLOR_CHARCOAL,
                            PIPE_WT, PIPE_L, K_IH_BASE_MPa, K_IH_HAZ_MPa,
                            K_TH_STAGE2)

# ICS2-palette sequential map for continuous fields (e.g. model error):
# teal (conservative, eps<1) -> steel (~exact) -> dark red (unsafe, eps>1).
ICS2_SEQ = LinearSegmentedColormap.from_list(
    "ics2_seq", [COLOR_TEAL, COLOR_STEEL, COLOR_RED])

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
    # Panel a: a(t) for base and HAZ
    ax=axes[0]
    ax.plot(t,a,color=COLOR_NAVY,linewidth=1.5,label='Base metal (bainitic X65)')
    ax.plot(t_h,a_h,color=COLOR_RED,linewidth=1.5,linestyle='--',label='HAZ (ferrite-pearlite)')
    ax.axhline(3.3,color=COLOR_TEAL,linewidth=0.8,linestyle=':',label='K_IH base crossing ~3.3 mm')
    ax.axhline(1.5,color=COLOR_RED,linewidth=0.7,linestyle=':',alpha=0.6,label='K_IH HAZ crossing ~1.5 mm')
    ax.set_xlabel('Time [yr]',fontsize=10); ax.set_ylabel('Crack depth a [mm]',fontsize=10)
    ax.set_title('(a)  3D Crack Depth a(t)\nBase vs HAZ zone [SOURCE: Beavers et al. 2001]',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8,frameon=False)
    # Panel b: aspect ratio c/a
    ax=axes[1]
    ax.plot(t, c/a,color=COLOR_NAVY,linewidth=1.5,label='Base (a/c coupled EDOs)')
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
    ax.set_ylim(0,max(K_IH_BASE_MPa*1.1, float(np.max(KI_b)*1.05)))
    ax.legend(fontsize=7.5,frameon=False,loc='upper left')
    plt.savefig(out,dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig); print(f'Saved: {out}')

def plot_va_loading(out="assets/figures/panel_d_va_loading.png"):
    """Panel d: variable amplitude loading spectrum and da/dt comparison."""
    os.makedirs(os.path.dirname(out),exist_ok=True)
    from src.pressure_spectrum import PressureSpectrum, da_dN_Chen_Xing
    from src.crack_growth import K_I_deeppoint, delta_K
    from src.constants import C_H_BULK_X65, P_OP_BAR, F_INT_UNDERLOAD

    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)

    # Left: da/dt vs crack depth for 3 spectrum types
    ax=axes[0]
    a_vals = np.linspace(0.5e-3, 6e-3, 60)
    for sp_type, col, lbl in [("Type_I",COLOR_RED,"Type I (underload, downstream compressor)"),
                                ("Type_II",COLOR_NAVY,"Type II (constant amplitude, mid-pipe)"),
                                ("Type_III",COLOR_TEAL,"Type III (mixed VA)")]:
        sp = PressureSpectrum(sp_type)
        da_dt_arr = []
        for a in a_vals:
            c = a * 4
            K_max = K_I_deeppoint(a, c, P_OP_BAR)
            dK = delta_K(a, c, P_OP_BAR, 0.55)
            da_dN = da_dN_Chen_Xing(a, K_max, dK, sp.f_major, C_H_BULK_X65)
            da_dN_min = da_dN_Chen_Xing(a, K_max, dK*0.1, sp.f_minor, C_H_BULK_X65)
            if sp_type == "Type_I":
                da_dt = sp.f_major * da_dN + sp.f_minor * da_dN_min * F_INT_UNDERLOAD
            elif sp_type == "Type_II":
                da_dt = sp.f_major * da_dN
            else:
                da_dt = 0.5*(sp.f_major*da_dN + sp.f_minor*da_dN_min*F_INT_UNDERLOAD) + 0.5*sp.f_major*da_dN
            da_dt_arr.append(da_dt * 1e3 * 365.25*24*3600)
        ax.semilogy(a_vals*1000, da_dt_arr, color=col, linewidth=1.4, label=lbl)
    ax.axhline(0.3, color='#888', linewidth=0.8, linestyle='--', alpha=0.7, label='CEPA field obs.: 0.3 mm/yr')
    ax.axhline(0.63, color='#888', linewidth=0.8, linestyle=':', alpha=0.7, label='CEPA field obs.: 0.63 mm/yr')
    ax.axvline(1.0, color=COLOR_CHARCOAL, linewidth=0.7, linestyle='-.',alpha=0.5, label='Stage I/II boundary ~1mm')
    ax.set_xlabel('Crack depth a [mm]',fontsize=10); ax.set_ylabel('da/dt [mm yr$^{-1}$]',fontsize=10)
    ax.set_title('(d)  VA Loading Spectrum Types — da/dt\nType I (underload) vs II (CA) vs III (mixed)',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=7.5,frameon=False)
    ax.set_ylim(1e-5, 100)

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
    wl=mc_result['wall_loss']; limit=3.3
    rho=spearman_sensitivity(mc_result)
    fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes: _ax(ax)
    # CDF
    ax=axes[0]; s=np.sort(wl); cdf=np.arange(1,len(s)+1)/len(s)
    ax.step(s,cdf,color=COLOR_NAVY,linewidth=1.2,where='post')
    ax.axvline(limit,color=COLOR_RED,linewidth=0.9,linestyle=':',label='K_IH crossing 3.3 mm')
    for pct,lbl in [(50,'P50'),(90,'P90'),(95,'P95')]:
        v=np.percentile(wl,pct)
        ax.axvline(v,color=COLOR_STEEL,linewidth=0.6,alpha=0.5)
        ax.text(v+0.01,pct/100-0.07,f'{lbl}\n{v:.2f}',fontsize=7.5,color=COLOR_STEEL,va='top')
    ax.set_xlabel('Crack depth at t* = 20 yr [mm]',fontsize=10)
    ax.set_ylabel('CDF [—]',fontsize=10)
    ax.set_title('(e)  Crack Depth CDF — N = 10,000\nPost-ILI POD-corrected a0 + model error ε',fontsize=10,fontweight='bold',loc='left')
    ax.set_ylim(0,1.05); ax.legend(fontsize=8,frameon=False)
    # Tornado
    ax=axes[1]
    names=list(rho.keys()); rhos=list(rho.values())
    order=sorted(range(len(rhos)),key=lambda i:abs(rhos[i]))
    names=[names[i] for i in order]; rhos=[rhos[i] for i in order]
    colors=[COLOR_RED if r>0 else COLOR_STEEL for r in rhos]
    bars=ax.barh(names,rhos,color=colors,edgecolor='none',height=0.6)
    ax.axvline(0,color='black',linewidth=0.7)
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
                    label='Δ_pH = pH$_{bulk}$ − pH$_{tip}$')
    ax3.set_xscale('log')
    ax3.set_xlabel('Crack aspect ratio c/a [—]', fontsize=10)
    ax3.set_ylabel('Δ_pH (acidification at tip) [—]', fontsize=10, color=COLOR_NAVY)
    ax3.tick_params(axis='y', labelcolor=COLOR_NAVY)
    ax3_twin = ax3.twinx()
    l2, = ax3_twin.plot(ca_vals, enhance, color=COLOR_RED, linewidth=1.6, linestyle='--',
                        label='C_H enhancement × (Turnbull)')
    ax3_twin.set_ylabel('C_H_tip / C_H_bulk [—]', fontsize=10, color=COLOR_RED)
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
    ax.plot(t,PoF*100,color=COLOR_RED,linewidth=1.5,label='Colony PoF(t)')
    ax.set_xlabel('Time [yr]',fontsize=10); ax.set_ylabel('Colony PoF [%]',fontsize=10)
    n_dorm=colony_result['n_dormant_initial']
    n_tot=colony_result['n_cracks']
    ax.set_title(f'(g)  Crack Colony PoF\n{n_dorm}/{n_tot} cracks initially dormant ({n_dorm/n_tot:.0%})',fontsize=10,fontweight='bold',loc='left')
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
    p_haz=mpatches.Patch(color=COLOR_RED,label='HAZ zone')
    p_base=mpatches.Patch(color=COLOR_NAVY,label='Base metal')
    ax2.legend(handles=[p_haz,p_base,
                         plt.Line2D([0],[0],marker='x',color=COLOR_TEAL,linestyle='None',
                                    markersize=8,markeredgewidth=1.5,
                                    label=f'Coalesced ({colony_result["n_coalesced"]} merges)')],
                fontsize=8,frameon=False,loc='upper right')
    ax2.set_xlim(0, PIPE_L)
    ax2.set_xlabel('Axial position [m]',fontsize=10); ax2.set_ylabel('Initial crack depth a₀ [mm]',fontsize=10)
    ax2.set_title("(g')  Colony Spatial Map (axial distribution along PIPE_L = 1 km)\n"
                  "BS 7910:2019 Cl.7.3 Coalescence Applied",
                  fontsize=10,fontweight='bold',loc='left')

    # Model error spread
    ax3=axes[2]
    eps=colony_result['eps_model']
    a0=colony_result['a0']*1000
    frac=[r['a'][-1]*1000 for r in colony_result['results']]
    sc3=ax3.scatter(a0,frac,c=eps,cmap=ICS2_SEQ,s=20,alpha=0.8,vmin=0.3,vmax=3.0)
    fig.colorbar(sc3,ax=ax3,fraction=0.046,pad=0.04).set_label('ε_model [—]',fontsize=9)
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

    # Bayesian update
    ax=axes[0]
    ax.plot(bayesian_result['particles'], bayesian_result['weight_history'][0]*len(bayesian_result['particles']),
            color=COLOR_STEEL,alpha=0.6,linewidth=1,label='Prior (uniform)')
    for i,wh in enumerate(bayesian_result['weight_history'][1:],1):
        col=[COLOR_TEAL,COLOR_RED][min(i-1,1)]
        ax.plot(bayesian_result['particles'],wh*len(bayesian_result['particles']),
                color=col,alpha=0.7,linewidth=1,label=f'After inspection {i}')
    ax.set_xlabel('Initial crack depth a₀ [mm]',fontsize=10)
    ax.set_ylabel('Normalised weight [—]',fontsize=10)
    ax.set_title('(h)  Bayesian Posterior Update\n[SOURCE: Straub 2004 — particle filter ILI]',fontsize=10,fontweight='bold',loc='left')
    ax.legend(fontsize=8.5,frameon=False)

    # FAD
    ax2=axes[1]; _ax(ax2)
    Lr_max=Lr_max_value()
    Lr_line=np.linspace(0,Lr_max*1.08,400)
    Kr_line=fad_curve(Lr_line)
    ax2.plot(Lr_line,Kr_line,color=COLOR_NAVY,linewidth=1.5,label='API 579-1 Level 2')
    ax2.axvline(Lr_max,color=COLOR_NAVY,linewidth=0.8,linestyle='--')
    ax2.fill_between(Lr_line,Kr_line,1.05,alpha=0.07,color=COLOR_RED)
    wl_arr=np.linspace(0,5e-3,100)
    traj_b=fad_trajectory(wl_arr)
    traj_h=fad_trajectory(wl_arr, Kmat=17.5)
    ax2.plot(traj_b['Lr'],traj_b['Kr'],color=COLOR_TEAL,linewidth=1.3,label='Base metal (K_IH=25)')
    ax2.plot(traj_h['Lr'],traj_h['Kr'],color=COLOR_RED,linewidth=1.3,linestyle='--',label='HAZ (K_IH=17.5)')
    ax2.set_xlim(0,Lr_max*1.12); ax2.set_ylim(0,1.10)
    ax2.set_xlabel('Lr = σ_ref/σ_Y [—]',fontsize=10); ax2.set_ylabel('Kr = KI/Kmat [—]',fontsize=10)
    ax2.set_title("(h')  FAD — Base vs HAZ\nHAZ crosses boundary earlier (lower K_IH)",fontsize=10,fontweight='bold',loc='left')
    ax2.legend(fontsize=8.5,frameon=False)

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
    Animated GIF: colony of NNpHSCC cracks growing along the PIPE_L axial range
    plus the simultaneous Colony PoF(t) trajectory.

    Meets the ICS2 QC Gate-2 GIF standard: >=50 interpolated frames, DPI 150, a
    progress bar, a per-frame time stamp in the suptitle, start/end holds, and an
    ICS2 brand watermark. Top axes: scatter of (axial position x, crack depth a),
    markers grow and turn larger as a crack crosses K_IH, coloured by zone
    (base/HAZ). Bottom axes: running Colony PoF(t).
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
    ax_top.set_xlim(0, PIPE_L)
    ax_top.set_ylim(0, max(PIPE_WT*1000*0.6, a_max*1.05))
    ax_top.axhline(3.3, color=COLOR_RED, linewidth=0.8, linestyle=':')
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
                                       label='K$_{IH}$ ~3.3 mm')],
                   fontsize=8, frameon=False, loc='upper right')

    ax_bot.set_xlim(0, t_fine[-1])
    ax_bot.set_ylim(0, max(pof_fine.max()*1.1, 5.0))
    ax_bot.set_xlabel('Time [yr]')
    ax_bot.set_ylabel('Colony PoF [%]')
    ax_bot.set_title('Colony Probability of Fracture (any crack > K$_{IH}$)',
                      fontsize=10, fontweight='bold', loc='left')
    pof_line, = ax_bot.plot([], [], color=COLOR_RED, linewidth=1.8)
    pof_dot, = ax_bot.plot([], [], 'o', color=COLOR_RED, markersize=6)

    # Progress bar (thin bottom strip)
    ax_prog.set_xlim(0, 1); ax_prog.set_ylim(0, 1); ax_prog.axis('off')
    ax_prog.add_patch(mpatches.Rectangle((0, 0.25), 1.0, 0.5,
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
        scat.set_sizes(np.where(a_now >= 3.3, 110, 40))
        pof_line.set_data(t_fine[:k+1], pof_fine[:k+1])
        pof_dot.set_data([t_fine[k]], [pof_fine[k]])
        prog_fg.set_width(k/(n_real-1))
        fig.suptitle(f'NNpHSCC Crack Colony Evolution   |   t = {t_fine[k]:.1f} yr',
                     fontsize=12, fontweight='bold', color=COLOR_CHARCOAL)
        return scat, pof_line, pof_dot, prog_fg

    anim = FuncAnimation(fig, update, frames=len(order), interval=1000/fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out, writer=writer, dpi=150)
    plt.close(fig)
    print(f'Saved: {out}')
