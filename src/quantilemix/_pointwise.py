# SPDX-License-Identifier: GPL-2.0-or-later
"""Conditional pointwise comparisons for additive non-bootstrap fits.

The column names and reference-level convention follow R qgcomp 2.19.6
``pointwisebound.noboot``. These bounds concern contrasts, not individual
predictions or population-average outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import special

from ._core import QGCompResult, _readonly
from ._binomial import BinomialResult
from ._poisson import PoissonResult


@dataclass(frozen=True, eq=False)
class PointwiseBoundResult:
    """Read-only R-named columns for one chosen joint-score reference level.

    ``pointwiseref`` is one-based, as in R. ``quantile`` values in ``table``
    are zero-based joint exposure scores. ``linpred`` follows the convention
    of its producing function: conditional for non-bootstrap fits and
    marginal for bootstrap fits.
    """

    columns: tuple[str, ...]
    table: Mapping[str, np.ndarray]
    pointwiseref: int
    alpha: float
    family: str
    link: str
    measure: str

    def __getitem__(self, name: str) -> np.ndarray:
        """Return a read-only column, using the corresponding R column name."""
        return self.table[name]


def pointwisebound_noboot(
    result: QGCompResult | BinomialResult | PoissonResult,
    *, pointwiseref: int = 1, alpha: float = 0.05,
) -> PointwiseBoundResult:
    """Compare each joint exposure score with a selected score level.

    Parameters
    ----------
    result
        An additive Gaussian, binomial-logit or Poisson-log fit made with a
        non-null quantile count. Poisson fits with observation-time offsets
        are excluded because this conditional reference profile omits time.
    pointwiseref
        One-based reference level from 1 to ``result.q``. For example, 1
        compares every score with joint score zero.
    alpha
        Tail area of each normal confidence interval, strictly between zero
        and one.

    Returns
    -------
    PointwiseBoundResult
        Read-only columns named as in R qgcomp 2.19.6. ``linpred`` is on the
        model link scale. Gaussian ``mean.diff`` is a conditional mean
        difference. Binary ``or`` is a conditional odds ratio; Poisson ``rr``
        is a conditional mean ratio for equal observation periods. The
        standard error is on the link scale. These are not marginal contrasts
        or uncertainty intervals for individual predictions.

    Raises
    ------
    TypeError
        If the fit type is outside the supported non-bootstrap families.
    ValueError
        If quantile levels, reference level, alpha or computed values are
        invalid, or if a Poisson time offset is present.
    """
    if not isinstance(result, (QGCompResult, BinomialResult, PoissonResult)):
        raise TypeError("result must be an additive Gaussian, binomial or Poisson non-bootstrap fit")
    if result.q is None:
        raise ValueError("pointwise comparisons require a non-null quantile count q")
    q = result.q
    if not isinstance(q, Integral) or q < 2:
        raise ValueError("the fitted quantile count must be at least two")
    if isinstance(pointwiseref, (bool, np.bool_)) or not isinstance(pointwiseref, Integral):
        raise ValueError("pointwiseref must be an integer from 1 to q")
    reference = int(pointwiseref)
    if not 1 <= reference <= q:
        raise ValueError("pointwiseref must be an integer from 1 to q")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite number strictly between zero and one")
    if isinstance(result, PoissonResult) and result.exposure_time is not None:
        raise ValueError("pointwise comparisons do not support a Poisson observation-time offset")

    scores = np.arange(q, dtype=float)
    midpoint = (scores + 0.5) / q
    linpred = float(result.coefficients[0]) + scores * float(result.psi)
    difference = linpred - linpred[reference - 1]
    se_difference = np.abs(scores - (reference - 1)) * float(result.se)
    critical = float(-special.ndtri_exp(np.log(float(alpha)) - np.log(2.0)))
    lower = difference - critical * se_difference
    upper = difference + critical * se_difference

    if isinstance(result, QGCompResult):
        family, link, measure = "gaussian", "identity", "conditional mean difference"
        columns = ("quantile", "quantile.midpoint", "linpred", "mean.diff",
                   "se.diff", "ll.diff", "ul.diff")
        values = (scores, midpoint, linpred, difference, se_difference, lower, upper)
    elif isinstance(result, BinomialResult):
        family, link, measure = "binomial", "logit", "conditional odds ratio"
        columns = ("quantile", "quantile.midpoint", "linpred", "or",
                   "se.lnor", "ll.or", "ul.or")
        values = (scores, midpoint, linpred, np.exp(difference), se_difference,
                  np.exp(lower), np.exp(upper))
    else:
        family, link, measure = "poisson", "log", "conditional mean ratio"
        columns = ("quantile", "quantile.midpoint", "linpred", "rr",
                   "se.lnrr", "ll.rr", "ul.rr")
        values = (scores, midpoint, linpred, np.exp(difference), se_difference,
                  np.exp(lower), np.exp(upper))
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("pointwise comparison values exceed the finite numeric range")
    table = MappingProxyType(dict(zip(columns, (_readonly(value) for value in values))))
    return PointwiseBoundResult(columns, table, reference, float(alpha), family, link, measure)
