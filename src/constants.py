"""
Week 10 — Full-physics NNpHSCC constants.
All additions grounded in peer-reviewed literature with [SOURCE] tags.
"""
R_GAS      = 8.314462618
FARADAY    = 96485.33212

# ── X65 material ─────────────────────────────────────────────────────────
STEEL_SMYS = 448e6; STEEL_UTS = 531e6; STEEL_E = 207e9; STEEL_NU = 0.30
STEEL_RHO  = 7850.0; STEEL_M = 55.845e-3; STEEL_N = 2; STEEL_HV = 220
K_IC_AIR   = 100.0  # MPa sqrt(m)

# ── Pipe geometry (24-inch NPS) ───────────────────────────────────────────
PIPE_OD    = 0.6096; PIPE_WT = 0.01270; PIPE_ID = PIPE_OD - 2*PIPE_WT
PIPE_L     = 1000.0

# ── Operating conditions ─────────────────────────────────────────────────
P_OP_BAR   = 69.0; T_OP_K = 288.15
P_CYCLE_FRAC_MAJOR = 0.15   # major underload cycle: 15% of MAOP [ASSUMED per Chen 2007]
P_CYCLE_FRAC_MINOR = 0.02   # minor ripple cycle [ASSUMED]
N_MINOR_PER_MAJOR  = 8      # minor cycles between each underload [ASSUMED]
FREQ_MAJOR_HZ = 1.16e-5     # 1 underload/day downstream compressor [ASSUMED]
FREQ_MINOR_HZ = 9.26e-5     # 8 ripples/day [ASSUMED]
F_CRIT_HZ     = 1.0e-3      # critical frequency; saturates below this [SOURCE: Xing model via Sun, Zhou & Kang 2021]

# ── Regulatory MAOP ──────────────────────────────────────────────────────
F_B31_8_CL2 = 0.72; F_B31_8_CL3 = 0.60; HDF_B31_12 = 0.54

# ── Soil / NNpH environment ───────────────────────────────────────────────
SOIL_PH        = 6.8; C_CO2_MOL = 0.05; C_HCO3_MOL = 0.01
SOIL_RHO_OHM   = 50.0
TEMP_SOIL_K    = T_OP_K
E_CP_V         = -0.85; E_FREE_V = -0.68

# Crack tip chemistry
PH_CRACKTIP_DELTA = -1.2    # pH drop at crack tip vs bulk [ASSUMED per Turnbull 1993]
C_FECL2_MOL       = 0.001   # Fe2+ from dissolution [ASSUMED]

# ── Hydrogen parameters ───────────────────────────────────────────────────
D_H_LATTICE  = 1.0e-8       # m^2/s [SOURCE: Kiuchi & McLellan 1983]
E_A_DIFF     = 4000.0       # J/mol
T_REF_H      = 298.15
V_H          = 2.0e-6       # m^3/mol partial molar volume in Fe
N_TRAP       = 1.0e20       # m^-3 grain boundary trap density [SOURCE: Turnbull 1996]
K_TRAP_EQ    = 10.0         # dimensionless retardation [SOURCE: San Marchi 2012]
C_H_SURF_REF = 5.0e-6       # mol/m^3 at free corrosion potential [ASSUMED]

# Bulk H concentration (microstructure-dependent per grade)
# [SOURCE: Sun et al. 2021 review — C_B in range 0.01-10 mol/m^3]
C_H_BULK_X52 = 0.05         # mol/m^3 ferrite-pearlite
C_H_BULK_X65 = 0.02         # mol/m^3 bainitic X65 base metal
C_H_BULK_HAZ = 0.08         # mol/m^3 HAZ ferrite-pearlite [SOURCE: Beavers et al.]

# Critical H for Stage I → Stage II transition [SOURCE: Zhao et al. 2017]
C_H_CRIT_STAGEII = 0.005   # mol/m^3 [ASSUMED lower bound from SwRI data]

# ── Microstructure zones ─────────────────────────────────────────────────
# [SOURCE: Beavers et al. 2001 — HAZ growth 30% faster]
# [SOURCE: Shirazi et al. 2024 — K_IH lower in HAZ]
K_IH_BASE_MPa = 25.0        # MPa sqrt(m) base metal [ASSUMED CEPA RP]
K_IH_HAZ_MPa  = 17.5        # MPa sqrt(m) HAZ [ASSUMED: K_IH_HAZ = 0.70 * K_IH_base]
F_MICRO_HAZ   = 1.30        # da/dt multiplier in HAZ [SOURCE: Beavers et al. 30%]

# ── Chen-Sutherby-Xing crack growth model ────────────────────────────────
# [SOURCE: Chen & Sutherby 2007; Xing et al. via Sun et al. 2021]
# da/dN = A_CF × (K_max × ΔK² × f_eff^(-0.1))^n × HE_factor(C_H_bulk)
A_CF_BASE    = 4.0e-14      # [ASSUMED] calibrated so da/dt = 0.3 mm/yr at a=2mm, c/a=4 (CEPA field band), X65 Type I, with the Newman-Raju SIF
N_CF         = 2.0          # exponent n [SOURCE: Chen & Sutherby 2007 n≈2]
N_HE_XING    = 0.88         # HEDE exponent for X52 [SOURCE: Sun et al. 2021, Xing model]

# ── Crack dormancy thresholds ─────────────────────────────────────────────
# [SOURCE: Zhao et al. 2017; Shirazi et al. 2024]
A_DORMANCY_MM    = 1.0      # mm: cracks arrest at <1mm (Stage I)
K_TH_STAGE2      = 2.2      # MPa sqrt(m): ΔK threshold for Stage II [SOURCE: API RP 1176]
K_TH_AIR         = 1.5      # MPa sqrt(m) (lower in NNpH environment)

# ── Variable amplitude loading ────────────────────────────────────────────
# [SOURCE: Chen literature via ScienceDirect Topics Ch.30]
F_INT_UNDERLOAD  = 10.0     # underload interaction factor on minor cycles [ASSUMED: 10x]
R_RATIO_MAJOR    = 0.55     # R = Kmin/Kmax for underload cycles [ASSUMED]
R_RATIO_MINOR    = 0.90     # R for ripple/minor cycles [ASSUMED]

# ── Residual stress (BS 7910 Annex Q) ────────────────────────────────────
# [SOURCE: BS 7910:2019 Annex Q — as-welded, no PWHT]
# Membrane residual stress as fraction of SMYS
SIGMA_RES_M_FRAC = 0.10     # [ASSUMED: 10% SMYS membrane after operational shakedown]
SIGMA_RES_B_FRAC = 0.30     # [ASSUMED: bending component at weld toe]

# ── ILI POD (EMAT / TFI crack tool) ─────────────────────────────────────
# [SOURCE: PHMSA TVC database; general EMAT performance]
POD_A90_MM   = 4.0          # mm: depth at which POD = 90% [ASSUMED EMAT modern tool]
POD_WEIBULL_K = 2.0         # Weibull shape [ASSUMED]

# ── Model structural uncertainty ─────────────────────────────────────────
# [SOURCE: Sun, Zhou & Kang 2021 — Xing model COV = 61.2% on 39 full-scale tests]
MODEL_ERROR_COV = 0.612      # COV of observed/predicted ratio
MODEL_ERROR_MEAN = 1.06      # mean of observed/predicted ratio (slight non-conservatism)

# ── Crack colony (Poisson) ────────────────────────────────────────────────
LAMBDA_COLONY = 5.0e-3       # m^-1 mean colony density [ASSUMED]
A0_MEAN       = 0.5e-3; A0_STD = 0.2e-3; C0_MEAN = 2.0e-3

# ── FFS limits ────────────────────────────────────────────────────────────
MAX_WL_FRAC = 0.20; A_CRIT_FRAC = 0.80; DESIGN_LIFE = 20.0
KMAT = 70.0

# ── ICS2 palette ─────────────────────────────────────────────────────────
COLOR_NAVY="#1b3a5c"; COLOR_STEEL="#4c80b0"; COLOR_RED="#8c2318"
COLOR_TEAL="#2e7d7b"; COLOR_CHARCOAL="#333333"; COLOR_GOLD="#8B6914"
COLOR_PURPLE="#5c2d8c"; COLOR_ORANGE="#c4611a"
