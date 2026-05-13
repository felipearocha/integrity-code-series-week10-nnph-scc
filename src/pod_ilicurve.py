"""
ILI Probability of Detection (POD) and post-inspection a0 distribution.

[SOURCE: General EMAT/TFI performance data; PHMSA technology verification]
POD(a) = 1 - exp(-(a/a90)^k)   Weibull form
a90 = 4 mm (depth at which POD = 90%) [ASSUMED modern EMAT tool]
k   = 2 (Weibull shape) [ASSUMED]

Post-inspection distribution of surviving cracks:
  f_post(a) ∝ f_prior(a) × (1 - POD(a))
  (i.e., cracks NOT detected follow this distribution)

This truncation of a0 is critical — the dangerous cracks are those the ILI missed,
whose depth distribution is NOT the same as the original field population.
"""
import numpy as np
from src.constants import POD_A90_MM, POD_WEIBULL_K, A0_MEAN, A0_STD, PIPE_WT


def pod(a_mm: float,
        a90: float = POD_A90_MM,
        k: float = POD_WEIBULL_K) -> float:
    """
    Probability of Detection for crack of depth a_mm [mm].
    POD(a) = 1 - exp(-(a/a90)^k)
    [ASSUMED] Weibull model per PHMSA TVC guidance.
    """
    return float(1.0 - np.exp(-((a_mm / a90) ** k)))


def prob_missed(a_mm: float, a90: float = POD_A90_MM, k_weibull: float = POD_WEIBULL_K) -> float:
    """P(missed by ILI) = 1 - POD(a)."""
    return 1.0 - pod(a_mm, a90=a90, k=k_weibull)


def sample_a0_post_inspection(n_samples: int, seed: int = 42,
                                a90: float = POD_A90_MM,
                                k_weibull: float = POD_WEIBULL_K,
                                a0_mean_m: float = A0_MEAN,
                                a0_std_m: float = A0_STD,
                                max_a0_mm: float = PIPE_WT * 500) -> np.ndarray:
    """
    Sample initial crack depths CONDITIONED on not being detected by ILI.

    Uses rejection sampling:
      1. Sample a0 from prior LogNormal
      2. Accept with probability (1 - POD(a0))  — i.e., only keep missed cracks

    This correctly represents the distribution of cracks that survived the inspection.
    Returns array of a0 [m].
    """
    rng = np.random.default_rng(seed)

    # LogNormal prior parameters
    mu_ln  = np.log(a0_mean_m) - 0.5 * np.log(1 + (a0_std_m / a0_mean_m) ** 2)
    sig_ln = np.sqrt(np.log(1 + (a0_std_m / a0_mean_m) ** 2))

    samples = []
    n_tries = 0
    max_tries = n_samples * 1000

    while len(samples) < n_samples and n_tries < max_tries:
        a_m = rng.lognormal(mu_ln, sig_ln)
        a_mm = a_m * 1000
        p_miss = prob_missed(a_mm, a90=a90, k_weibull=k_weibull)
        if rng.uniform() < p_miss:          # accept if ILI missed it
            samples.append(np.clip(a_m, 1e-5, max_a0_mm / 1000))
        n_tries += 1

    if len(samples) < n_samples:
        # Pad with smallest detectable cracks if rejection sampling exhausted
        samples.extend([A0_MEAN * 0.1] * (n_samples - len(samples)))

    return np.array(samples[:n_samples])


def pod_curve_info(a90: float = POD_A90_MM, k: float = POD_WEIBULL_K) -> dict:
    """Summary statistics for the POD curve."""
    a_vals = np.linspace(0.1, 15, 300)
    pod_vals = np.array([pod(a, a90, k) for a in a_vals])
    a50 = a_vals[np.argmin(np.abs(pod_vals - 0.50))]
    a90v = a_vals[np.argmin(np.abs(pod_vals - 0.90))]
    return {'a50_mm': float(a50), 'a90_mm': float(a90v),
            'a_vals_mm': a_vals, 'pod_vals': pod_vals}
