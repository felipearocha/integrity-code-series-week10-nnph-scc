"""API 579-1 Level 2 FAD + B31.12 vs B31.8 code comparison."""
import numpy as np
from src.constants import (PIPE_OD, PIPE_WT, STEEL_SMYS, STEEL_UTS, KMAT,
                            P_OP_BAR, F_B31_8_CL2, F_B31_8_CL3, HDF_B31_12,
                            K_IC_AIR, K_IH_BASE_MPa)
K_IH = K_IH_BASE_MPa
from src.crack_growth import K_I_deeppoint as K_I

def Lr_max_value():
    return 0.5*(1.0+STEEL_UTS/STEEL_SMYS)

def fad_curve(Lr, Lr_max=None):
    if Lr_max is None: Lr_max = Lr_max_value()
    Lr = np.asarray(Lr, dtype=float); Kr = np.zeros_like(Lr)
    v = Lr <= Lr_max
    Kr[v] = ((1+0.5*Lr[v]**2)**(-0.5) * (0.3+0.7*np.exp(-0.65*Lr[v]**6)))
    return Kr

def assessment_point(a_mm: float, P_bar: float = P_OP_BAR,
                      Kmat: float = KMAT):
    a_m = a_mm * 1e-3
    P_MPa = P_bar * 0.1
    t_rem = max(PIPE_WT - a_m, 1e-4)
    sigma_h = P_MPa * PIPE_OD / (2.0 * t_rem)
    KI_val = K_I(a_m, max(a_m*3,2e-3), P_bar)
    Lr = sigma_h / (STEEL_SMYS * 1e-6)
    Kr = KI_val / Kmat
    return float(Lr), float(Kr)

def fad_trajectory(a_arr: np.ndarray, P_bar: float = P_OP_BAR, Kmat: float = KMAT):
    Lr_max = Lr_max_value()
    Lr_arr = np.zeros(len(a_arr)); Kr_arr = np.zeros(len(a_arr))
    for k, a in enumerate(a_arr):
        Lr_arr[k], Kr_arr[k] = assessment_point(a*1000, P_bar, Kmat=Kmat)
    Kr_bound = fad_curve(Lr_arr, Lr_max)
    return {"Lr":Lr_arr,"Kr":Kr_arr,"Kr_boundary":Kr_bound,"Lr_max":Lr_max,
            "status":np.where(Kr_arr<=Kr_bound,"acceptable","unacceptable")}

def maop_comparison(pipe_wt: float = PIPE_WT, D: float = PIPE_OD) -> dict:
    """
    Compare MAOP under B31.8 Class 2, B31.8 Class 3, and B31.12 H2 service.
    MAOP = 2*SMYS*t/D * F * E * T_factor  (E=1, T=1 for API 5L seamless)
    """
    def maop_bar(F):
        return 2.0*STEEL_SMYS*pipe_wt/D*F * 1e-5  # Pa -> bar
    m_cl2  = maop_bar(F_B31_8_CL2)
    m_cl3  = maop_bar(F_B31_8_CL3)
    m_h2   = maop_bar(F_B31_8_CL2 * HDF_B31_12)
    return {"B31_8_Class2_bar":m_cl2, "B31_8_Class3_bar":m_cl3,
            "B31_12_H2_bar":m_h2,
            "MAOP_reduction_cl2_to_cl3_pct":(m_cl2-m_cl3)/m_cl2*100,
            "MAOP_reduction_cl2_to_H2_pct":(m_cl2-m_h2)/m_cl2*100}
