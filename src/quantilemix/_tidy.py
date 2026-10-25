# SPDX-License-Identifier: GPL-2.0-or-later
"""Compact joint-effect tables for supported conditional fits."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import special, stats

from ._binomial import BinomialResult
from ._core import QGCompResult
from ._cox import CoxResult
from ._poisson import PoissonResult


@dataclass(frozen=True, eq=False)
class TidyResult:
    """Read-only columns for the intercept and joint exposure contrast."""

    columns: tuple[str, ...]
    table: Mapping[str, np.ndarray]
    family: str
    scale: str
    conf_level: float | None

    def __getitem__(self, column: str) -> np.ndarray:
        return self.table[column]


def _freeze(values) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def tidy(
    result: QGCompResult | BinomialResult | PoissonResult | CoxResult, *,
    conf_level: float | None = None, exponentiate: bool = False,
    quick: bool = False,
) -> TidyResult:
    """Report the fitted intercept and joint effect in a small named table.

    Only additive non-bootstrap Gaussian, binomial, Poisson and conditional
    Cox fits are supported. Cox has no intercept row. By default the
    confidence level is the fit's ``1-alpha``. ``quick=True`` returns only
    term names and estimates. Exponentiation is available for log or logit
    links; standard errors and test statistics remain on the link scale.

    Unlike R qgcomp 2.19.6's ``tidy`` method, the upper exponentiated limit
    uses the upper limit, and an intercept has its own p-value instead of a
    recycled joint-effect p-value. These are explicit correctness repairs.
    The Gaussian joint p-value retains R qgcomp's reference degrees of
    freedom; its intercept uses the fitted model's residual degrees of
    freedom. Normal confidence limits are used for both.
    """
    if not isinstance(result, (QGCompResult, BinomialResult, PoissonResult, CoxResult)):
        raise TypeError("tidy requires a supported additive conditional fit")
    if not isinstance(exponentiate, bool) or not isinstance(quick, bool):
        raise ValueError("exponentiate and quick must be Boolean")
    if exponentiate and isinstance(result, QGCompResult):
        raise ValueError("Gaussian mean differences have no log-scale exponentiation")
    if conf_level is None:
        conf_level = 1-result.alpha
    if (isinstance(conf_level, bool) or not isinstance(conf_level, Real)
            or not np.isfinite(conf_level) or not 0 < conf_level < 1):
        raise ValueError("conf_level must be finite and strictly between zero and one")
    if isinstance(result, CoxResult):
        terms = ("psi1",)
        estimates = np.array([result.psi])
        se = np.array([result.se])
        statistic = np.array([result.test_statistic])
        p_value = np.array([result.p_value])
        family = "cox"
    else:
        terms = ("(Intercept)", "psi1")
        estimates = np.array([result.coefficients[0], result.psi])
        se = np.array([np.sqrt(result.covariance[0, 0]), result.se])
        statistic = np.divide(estimates, se, out=np.full(2, np.nan), where=se > 0)
        statistic[1] = result.test_statistic
        if isinstance(result, QGCompResult):
            intercept_p = 2*stats.t.sf(abs(statistic[0]), result.residual_df)
            family = "gaussian"
        else:
            intercept_p = 2*special.ndtr(-abs(statistic[0]))
            family = result.family
        p_value = np.array([intercept_p, result.p_value])
    critical = float(special.ndtri((1+conf_level)/2))
    lower = estimates-critical*se
    upper = estimates+critical*se
    table = {"term": _freeze(terms),
             "estimate": _freeze(np.exp(estimates) if exponentiate else estimates)}
    if not quick:
        table.update({"std_error": _freeze(se),
                      "lower_ci": _freeze(np.exp(lower) if exponentiate else lower),
                      "upper_ci": _freeze(np.exp(upper) if exponentiate else upper),
                      "statistic": _freeze(statistic),
                      "p_value": _freeze(p_value)})
    return TidyResult(tuple(table), MappingProxyType(table), family,
                      "response_ratio" if exponentiate else "link", conf_level)
