"""
Bayesian updating of crack depth distribution from ILI observations.

When a re-inspection ILI observes crack depth a_obs (or does not detect it),
we update the prior distribution of model parameters using Bayes' theorem:

  p(θ | a_obs) ∝ p(a_obs | θ) × p(θ)

Implemented using Sequential Importance Sampling (particle filter approach)
for computational tractability.

[SOURCE: Straub (2004) — Bayesian updating for inspection-based reliability;
         Luque & Straub (2019) — Bayesian network for pipeline integrity]
"""
import numpy as np
from scipy.stats import norm


def likelihood_observation(a_obs_mm: float, a_pred_mm: float,
                             sigma_ILI_mm: float = 0.5) -> float:
    """
    Likelihood p(a_obs | a_predicted) assuming Gaussian ILI sizing error.
    sigma_ILI: ILI tool sizing uncertainty [mm] [ASSUMED: ±0.5mm at 80% confidence]
    [SOURCE: PHMSA TVC — typical EMAT sizing error ~0.5-1.0mm std]
    """
    return float(norm.pdf(a_obs_mm, loc=a_pred_mm, scale=sigma_ILI_mm))


def likelihood_not_detected(a_pred_mm: float,
                              a90_mm: float = 4.0,
                              k_weibull: float = 2.0) -> float:
    """
    Likelihood p(not detected | a_predicted) = 1 - POD(a_predicted).
    Used when ILI did NOT find a crack at a known location.
    """
    pod = 1.0 - np.exp(-((a_pred_mm / a90_mm) ** k_weibull))
    return float(1.0 - pod)


def bayesian_update_particles(particles_a0_mm: np.ndarray,
                                particles_weight: np.ndarray,
                                a_obs_mm: float,
                                detected: bool = True,
                                sigma_ILI_mm: float = 0.5,
                                t_obs_yr: float = 5.0,
                                da_dt_mm_yr: float = 0.05) -> tuple:
    """
    Update particle weights given one ILI observation.

    Parameters
    ----------
    particles_a0_mm : array of initial crack depth particles [mm]
    particles_weight: current importance weights (normalised)
    a_obs_mm        : observed crack depth at t_obs_yr [mm] (used if detected=True)
    detected        : bool — True if ILI found the crack, False if not found
    da_dt_mm_yr     : assumed average growth rate to project a0 → a(t_obs)

    Returns (updated_weights, effective_sample_size)
    """
    a_at_obs = particles_a0_mm + da_dt_mm_yr * t_obs_yr  # propagate particles

    if detected:
        likelihoods = np.array([likelihood_observation(a_obs_mm, ap, sigma_ILI_mm)
                                  for ap in a_at_obs])
    else:
        likelihoods = np.array([likelihood_not_detected(ap) for ap in a_at_obs])

    # Update weights
    new_weights = particles_weight * likelihoods
    total = new_weights.sum()
    if total < 1e-300:
        return particles_weight, len(particles_weight)  # degenerate — keep prior
    new_weights /= total

    # Effective sample size (measure of degeneracy)
    ess = 1.0 / np.sum(new_weights ** 2)

    return new_weights, float(ess)


def posterior_summary(particles_a0_mm: np.ndarray,
                        weights: np.ndarray) -> dict:
    """Compute posterior statistics from weighted particles."""
    w = weights / weights.sum()
    mean  = float(np.dot(w, particles_a0_mm))
    var   = float(np.dot(w, (particles_a0_mm - mean)**2))
    std   = float(np.sqrt(var))
    # Weighted quantiles
    sort_idx = np.argsort(particles_a0_mm)
    p_sorted = particles_a0_mm[sort_idx]
    w_sorted = weights[sort_idx]
    cumw = np.cumsum(w_sorted)
    p10 = float(p_sorted[np.searchsorted(cumw, 0.10)])
    p50 = float(p_sorted[np.searchsorted(cumw, 0.50)])
    p90 = float(p_sorted[np.searchsorted(cumw, 0.90)])
    return {'mean': mean, 'std': std, 'cov': std/max(mean,1e-10),
            'P10': p10, 'P50': p50, 'P90': p90}


def multi_inspection_update(a0_prior_mm: np.ndarray,
                              inspection_log: list,
                              da_dt_mm_yr: float = 0.05) -> dict:
    """
    Update a0 distribution through multiple ILI inspections.

    inspection_log: list of dicts {'t_yr', 'a_obs_mm', 'detected'}
    Returns posterior distribution summary and weight history.
    """
    N = len(a0_prior_mm)
    weights = np.ones(N) / N  # uniform prior weights
    weight_history = [weights.copy()]
    ess_history = [float(N)]

    for insp in inspection_log:
        weights, ess = bayesian_update_particles(
            a0_prior_mm, weights,
            a_obs_mm=insp.get('a_obs_mm', 0.0),
            detected=insp['detected'],
            t_obs_yr=insp['t_yr'],
            da_dt_mm_yr=da_dt_mm_yr,
        )
        weight_history.append(weights.copy())
        ess_history.append(ess)

        # Resample if ESS < N/2 (avoid weight collapse)
        if ess < N / 2:
            rng = np.random.default_rng(42)
            resample_idx = rng.choice(N, size=N, p=weights, replace=True)
            a0_prior_mm = a0_prior_mm[resample_idx]
            weights = np.ones(N) / N

    posterior = posterior_summary(a0_prior_mm, weights)
    return {
        'posterior': posterior,
        'particles': a0_prior_mm,
        'weights': weights,
        'ess_history': ess_history,
        'weight_history': weight_history,
        'n_inspections': len(inspection_log),
    }
