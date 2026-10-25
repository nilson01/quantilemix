# SPDX-License-Identifier: GPL-2.0-or-later
"""One-row fit summaries for supported conditional models."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import stats

from ._binomial import BinomialResult
from ._core import QGCompResult
from ._poisson import PoissonResult


@dataclass(frozen=True, eq=False)
class GlanceResult:
    """Read-only, one-row summary of a fitted conditional model.

    Missing numerical measures have value ``None``. The row contains only
    model information; it does not repeat the fitting call or input names.
    """

    columns: tuple[str, ...]
    table: Mapping[str, float | int | None]
    family: str

    def __getitem__(self, column: str) -> float | int | None:
        return self.table[column]

    def to_dict(self) -> dict[str, float | int | None]:
        return dict(self.table)


def glance(result: QGCompResult | BinomialResult | PoissonResult) -> GlanceResult:
    """Give one row of likelihood and fit measures for a conditional model.

    Supports additive Gaussian, Bernoulli-logit and Poisson-log fits. ``nobs``
    counts rows, ``df_model`` counts slope coefficients, and ``df_residual``
    is the number of rows minus all fitted coefficients. ``deviance`` and
    ``null_deviance`` are weighted when case weights were used; Gaussian
    null deviance is the weighted total sum of squares. The null
    Poisson model retains each row's observation time. AIC and BIC use the
    maximized full likelihood for Gaussian and the model likelihood for the
    two generalized linear models. Gaussian AIC/BIC count the error variance
    as one additional parameter; BIC uses the number of rows.

    Zero residual sum of squares makes a Gaussian maximum-likelihood variance
    degenerate. In that case likelihood-based measures are ``None`` instead
    of infinite. An undefined R-squared or F test is also ``None``.
    Bootstrap, estimating-equation, Cox and other result classes are not yet
    supported; they must not inherit likelihood claims from a different fit.
    """
    if not isinstance(result, (QGCompResult, BinomialResult, PoissonResult)):
        raise TypeError("glance requires an additive conditional Gaussian, binomial or Poisson fit")
    n = len(result.fitted_values)
    k = len(result.coefficients)
    common: dict[str, float | int | None] = {
        "nobs": n, "df_model": k-1, "df_residual": result.residual_df,
    }
    if isinstance(result, QGCompResult):
        weights = np.ones(n) if result.case_weights is None else result.case_weights
        y = result.fitted_values + result.residuals
        mean_y = float(np.average(y, weights=weights))
        sse = float(np.sum(weights * result.residuals**2))
        tss = float(np.sum(weights * (y-mean_y)**2))
        degenerate = bool(result.diagnostics.get("inference_degenerate", False))
        sigma = float(np.sqrt(result.residual_variance))
        r_squared = None if tss <= 0 else float(1-sse/tss)
        adjusted = (None if r_squared is None else
                    float(1-(1-r_squared)*(n-1)/result.residual_df))
        if not degenerate and sse > 0 and tss > 0 and k > 1:
            statistic = float(((tss-sse)/(k-1))/(sse/result.residual_df))
            f_p_value = float(stats.f.sf(statistic, k-1, result.residual_df))
        else:
            statistic = f_p_value = None
        if not degenerate and sse > 0:
            log_likelihood = float(0.5 * (
                np.sum(np.log(weights))-n*(np.log(2*np.pi)+1-np.log(n)+np.log(sse))))
            aic = float(-2*log_likelihood+2*(k+1))
            bic = float(-2*log_likelihood+np.log(n)*(k+1))
        else:
            log_likelihood = aic = bic = None
        common.update(r_squared=r_squared, adjusted_r_squared=adjusted,
                      sigma=sigma, f_statistic=statistic, f_p_value=f_p_value,
                      log_likelihood=log_likelihood, aic=aic, bic=bic,
                      deviance=sse, null_deviance=tss, df_null=n-1)
        family = "gaussian"
    else:
        log_likelihood = result.log_likelihood
        common.update(log_likelihood=log_likelihood,
                      aic=float(-2*log_likelihood+2*k),
                      bic=float(-2*log_likelihood+np.log(n)*k),
                      deviance=result.deviance,
                      null_deviance=result.null_deviance,
                      df_null=n-1)
        family = result.family
    return GlanceResult(tuple(common), MappingProxyType(common), family)
