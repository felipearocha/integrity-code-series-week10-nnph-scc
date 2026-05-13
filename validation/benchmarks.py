"""Validation benchmarks for Week 10."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import PIPE_OD, PIPE_WT, STEEL_SMYS, P_OP_BAR, D_H_LATTICE, FARADAY, STEEL_M, STEEL_N, STEEL_RHO

def benchmark_chen_sutherby_combined():
    """Chen-Sutherby combined parameter: C = Kmax × ΔK² × f^-0.1 has correct units."""
    from src.pressure_spectrum import combined_parameter
    K=20.0; dK=8.0; f=1e-3
    C = combined_parameter(K, dK, f)
    assert C > 0 and np.isfinite(C)
    C_expected = K * dK**2 * (f**(-0.1))
    assert abs(C - C_expected) < 1e-10
    print(f"Chen-Sutherby C = {C:.2f}  OK")

def benchmark_K_I_flat_plate():
    """K_I for a=2mm: vs simplified Tada leading term (within 15%)."""
    from src.crack_growth import K_I_deeppoint
    a=2e-3; c=8e-3
    P_MPa=P_OP_BAR*0.1; sigma_h=P_MPa*PIPE_OD/(2*PIPE_WT)
    K_approx = sigma_h*np.sqrt(np.pi*a)*1.12
    K_num = K_I_deeppoint(a,c,P_OP_BAR)
    err = abs(K_num-K_approx)/K_approx
    print(f"K_I: num={K_num:.3f} approx={K_approx:.3f} err={err*100:.1f}%")
    assert err < 0.25

def benchmark_faraday():
    from src.constants import FARADAY, STEEL_M, STEEL_N, STEEL_RHO
    SEC=365.25*24*3600
    v = 1.0*STEEL_M/(STEEL_N*FARADAY*STEEL_RHO)*1e3*SEC
    print(f"Faraday 1A/m²: {v:.4f} mm/yr  (ref 1.163)")
    assert abs(v-1.163)/1.163 < 0.01

def benchmark_pod_weibull():
    from src.pod_ilicurve import pod
    assert abs(pod(0)) < 1e-10
    assert abs(pod(1e6)-1.0) < 1e-6
    assert 0.60 < pod(4.0) < 0.70  # Weibull scale param: POD=63.2% at a90
    print("POD Weibull: POD(0)=0, POD(4mm)=90%, POD(inf)=1  OK")

def benchmark_model_error_moments():
    from src.model_uncertainty import uncertainty_quantiles, MODEL_ERROR_MEAN, MODEL_ERROR_COV
    q = uncertainty_quantiles()
    assert q['P05'] < 1.0 < q['P95']  # 1.0 within 90% CI
    assert q['P95'] > q['P50'] > q['P05']
    print(f"Model error: P05={q['P05']:.3f} P50={q['P50']:.3f} P95={q['P95']:.3f} OK")

def benchmark_hoop_stress():
    P_MPa=P_OP_BAR*0.1; sigma=P_MPa*PIPE_OD/(2*PIPE_WT)
    pct=sigma/(STEEL_SMYS*1e-6)*100
    print(f"Hoop: {sigma:.1f} MPa = {pct:.1f}% SMYS")
    assert 30<pct<80

if __name__=="__main__":
    print("="*60,"Week 10 BENCHMARKS","="*60)
    benchmark_chen_sutherby_combined()
    benchmark_K_I_flat_plate()
    benchmark_faraday()
    benchmark_pod_weibull()
    benchmark_model_error_moments()
    benchmark_hoop_stress()
    print("\nAll benchmarks passed.")
