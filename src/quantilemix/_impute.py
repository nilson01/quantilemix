# SPDX-License-Identifier: GPL-2.0-or-later
"""Left-censored lognormal imputation for a Python missing-data workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, special

from ._core import _matrix, _real_array


def _boolean_mask(values: Any, name: str, n: int) -> np.ndarray:
    raw = np.asanyarray(values)
    if raw.ndim != 1 or len(raw) != n or raw.dtype != np.dtype(bool):
        raise ValueError(f"{name} must contain one Boolean value per row")
    return np.asarray(raw, dtype=bool)


def impute_leftcenslognorm(
    y: Any, ry: Any, x: Any = None, wy: Any = None, *,
    lod: Any = None, seed: int | None = None,
) -> np.ndarray:
    """Draw positive values below a limit of detection (LOD).

    ``y`` is one measured variable. ``ry`` marks observed values; selected
    left-censored rows have ``ry=False`` and are marked by ``wy`` (by default,
    all unobserved rows). Observed values must be finite and positive. ``x``
    contains complete numeric predictors without an intercept; an intercept
    is added. ``lod`` is a positive scalar or one positive value per row. If
    omitted, the smallest observed value is used for every censored row.

    A lognormal regression uses both observed densities and censored lower-
    tail probabilities. One parameter vector is drawn from the fitted
    normal approximation, then each selected value is sampled from its
    lognormal distribution truncated above at its LOD. The return contains
    imputed values in selected-row order, as in R's MICE helper. Assign them
    to those rows in an external imputation workflow; this package does not
    silently pool multiply imputed analyses.

    This is a Python interface to the R qgcomp 2.19.6 left-censored lognormal
    method, not an R MICE callback. Its random stream and optimizer differ
    from R's ``survreg`` implementation. The normal parameter draw is an
    asymptotic approximation and can be unreliable in small or weakly
    identified samples. ``impute_tobit`` is an alias for this same algorithm.
    """
    raw_y = np.asanyarray(y)
    if raw_y.ndim != 1 or len(raw_y) == 0:
        raise ValueError("y must be a nonempty one-dimensional vector")
    n = len(raw_y)
    observed = _boolean_mask(ry, "ry", n)
    selected = ~observed if wy is None else _boolean_mask(wy, "wy", n)
    if np.any(selected & observed):
        raise ValueError("wy may select only left-censored rows")
    if np.any(~observed & ~selected):
        raise ValueError("every unobserved row must be selected for imputation")
    try:
        observed_y = np.asarray(raw_y[observed], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("observed y values must be positive finite numbers") from exc
    if (not np.all(np.isfinite(observed_y)) or np.any(observed_y <= 0)):
        raise ValueError("observed y values must be positive finite numbers")
    if not np.any(selected):
        return np.empty(0, dtype=float)
    if lod is None:
        if len(observed_y) == 0:
            raise ValueError("lod cannot be inferred without observed y values")
        limits = np.full(n, float(np.min(observed_y)))
    else:
        limits = _real_array(lod, "lod")
        if limits.ndim == 0:
            limits = np.full(n, float(limits))
        if limits.ndim != 1 or len(limits) != n or np.any(limits <= 0):
            raise ValueError("lod must be a positive scalar or one positive value per row")
    if x is None:
        design = np.ones((n, 1))
    else:
        matrix, _ = _matrix(x, "x")
        if len(matrix) != n:
            raise ValueError("x must have one complete row per y value")
        design = np.column_stack((np.ones(n), matrix))
    p = design.shape[1]
    if len(observed_y) < p+2:
        raise ValueError("too few observed values to fit the lognormal regression")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                             or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    log_y = np.log(observed_y)
    log_limit = np.log(limits[selected])
    x_obs = design[observed]
    x_cens = design[selected]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:p]
        sigma = np.exp(theta[p])
        z = (log_y-x_obs @ beta)/sigma
        a = (log_limit-x_cens @ beta)/sigma
        log_cdf = special.log_ndtr(a)
        log_lambda = -0.5*a*a-0.5*np.log(2*np.pi)-log_cdf
        ratio = np.exp(log_lambda)
        log_likelihood = (-np.log(observed_y)-np.log(sigma)
                          -0.5*z*z-0.5*np.log(2*np.pi)).sum()+log_cdf.sum()
        score_beta = (x_obs.T @ z-x_cens.T @ ratio)/sigma
        score_scale = np.sum(z*z-1)-np.sum(a*ratio)
        return -float(log_likelihood), -np.r_[score_beta, score_scale]

    initial_beta = np.linalg.lstsq(x_obs, log_y, rcond=None)[0]
    initial_sigma = max(float(np.std(log_y-x_obs @ initial_beta)), 0.25)
    initial = np.r_[initial_beta, np.log(initial_sigma)]
    fitted = optimize.minimize(objective, initial, jac=True, method="BFGS",
                               options={"gtol": 1e-8, "maxiter": 1000})
    if (not np.all(np.isfinite(fitted.x)) or not np.isfinite(fitted.fun)
            or np.max(np.abs(fitted.jac)) > 1e-5):
        raise ValueError("left-censored lognormal fit did not converge")
    theta = fitted.x
    hessian = np.empty((p+1, p+1))
    for j in range(p+1):
        step = 1e-5*(1+abs(theta[j]))
        direction = np.zeros(p+1)
        direction[j] = step
        hessian[:, j] = (objective(theta+direction)[1]
                         -objective(theta-direction)[1])/(2*step)
    hessian = (hessian+hessian.T)/2
    if not np.all(np.isfinite(hessian)) or np.linalg.eigvalsh(hessian)[0] <= 0:
        raise ValueError("left-censored lognormal information is not positive definite")
    covariance = np.linalg.solve(hessian, np.eye(p+1))
    rng = np.random.default_rng(seed)
    drawn = rng.multivariate_normal(theta, (covariance+covariance.T)/2)
    sigma = float(np.exp(drawn[p]))
    mu = x_cens @ drawn[:p]
    log_upper = special.log_ndtr((log_limit-mu)/sigma)
    log_u = log_upper+np.log(rng.random(len(log_upper)))
    values = np.exp(mu+sigma*special.ndtri_exp(log_u))
    if (not np.all(np.isfinite(values)) or np.any(values <= 0)
            or np.any(values > limits[selected]*(1+1e-12))):
        raise ValueError("left-censored lognormal draw is nonfinite or exceeds its LOD")
    return values


impute_tobit = impute_leftcenslognorm
