# SPDX-License-Identifier: GPL-2.0-or-later
"""Marginal bootstrap pointwise comparisons following R qgcomp 2.19.6."""

from __future__ import annotations

from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from scipy import special

from ._bootstrap import GaussianBootstrapResult
from ._core import _readonly
from ._pointwise import PointwiseBoundResult
from ._poisson_bootstrap import PoissonBootstrapResult


def pointwisebound_boot(
    result: GaussianBootstrapResult | PoissonBootstrapResult,
    *, pointwiseref: int = 1, alpha: float = 0.05,
) -> PointwiseBoundResult:
    """Compare marginal fitted means at each score against one reference score.

    ``pointwiseref`` is one-based, matching R. In R qgcomp 2.19.6, the
    Poisson bootstrap covariance is calculated from marginal-model predictions
    on the log scale. The displayed fitted means are calculated on the response
    scale, then logged for the linear-predictor column.
    """
    if not isinstance(result, (GaussianBootstrapResult, PoissonBootstrapResult)):
        raise TypeError("result must be a supported Gaussian or Poisson bootstrap fit")
    q = result.point_fit.q
    if q is None or not isinstance(q, Integral) or q < 2:
        raise ValueError("pointwise bootstrap comparisons require a non-null q of at least two")
    if len(result.intervention_means) != q or result.bootstrap_means.shape != (result.B, q):
        raise ValueError("bootstrap profiles do not match the fitted quantile count")
    if isinstance(pointwiseref, (bool, np.bool_)) or not isinstance(pointwiseref, Integral):
        raise ValueError("pointwiseref must be an integer from 1 to q")
    reference = int(pointwiseref)
    if not 1 <= reference <= q:
        raise ValueError("pointwiseref must be an integer from 1 to q")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")

    scores = np.arange(q, dtype=float)
    midpoint = (scores + 0.5) / q
    means = result.intervention_means
    profiles = (result.bootstrap_means if isinstance(result, GaussianBootstrapResult)
                else np.log(result.bootstrap_means))
    differences = profiles - profiles[:, [reference-1]]
    se_difference = np.std(differences, axis=0, ddof=1)
    se_difference[reference-1] = 0.0
    critical = float(-special.ndtri_exp(np.log(float(alpha)) - np.log(2.0)))
    if isinstance(result, GaussianBootstrapResult):
        link, measure = "identity", "marginal mean difference"
        linpred = means
        contrast = means - means[reference-1]
        lower, upper = contrast-critical*se_difference, contrast+critical*se_difference
        columns = ("quantile", "quantile.midpoint", "linpred", "mean.diff",
                   "se.diff", "ll.diff", "ul.diff", "ll.linpred", "ul.linpred")
        values = (scores, midpoint, linpred, contrast, se_difference, lower, upper,
                  linpred-critical*se_difference, linpred+critical*se_difference)
    else:
        link, measure = "log", "marginal mean ratio"
        if np.any(means <= 0):
            raise ValueError("Poisson marginal means must be positive")
        linpred = np.log(means)
        contrast = linpred-linpred[reference-1]
        lower, upper = np.exp(contrast-critical*se_difference), np.exp(contrast+critical*se_difference)
        columns = ("quantile", "quantile.midpoint", "linpred", "rr",
                   "se.lnrr", "ll.rr", "ul.rr", "ll.linpred", "ul.linpred")
        values = (scores, midpoint, linpred, np.exp(contrast), se_difference,
                  lower, upper, linpred-critical*se_difference, linpred+critical*se_difference)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("pointwise bootstrap comparison exceeds the finite numeric range")
    table = MappingProxyType(dict(zip(columns, (_readonly(value) for value in values))))
    return PointwiseBoundResult(columns, table, reference, float(alpha), result.family, link, measure)
