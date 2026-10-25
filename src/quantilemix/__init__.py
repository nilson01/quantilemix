# SPDX-License-Identifier: GPL-2.0-or-later
"""Quantile g-computation transforms exposures into ordinal quantile scores and estimates the joint effect of increasing all exposures simultaneously by one quantile. The package supports selected models for continuous, binary, count, and survival outcomes, with more limited support for censored, multicategory, and zero-inflated outcomes. Each API entry documents the effect scale, required inputs, and methodological limitations.

Quantile cutpoints estimated during model fitting are reused for prediction. Sign-specific weights summarize fitted exposure coefficients rather than causal importance. Successful model fitting does not establish the validity of model assumptions or identify a causal effect without additional causal identification assumptions.

Method: Keil et al. (2020), doi:10.1289/EHP5838. Selected numerical
conventions were checked against Alexander Keil's R qgcomp 2.19.6.
"""

from ._core import QGCompResult, QGCompWarning, QuantizationResult, fit, quantize
from ._binomial import BinomialResult, fit_binomial, fit_binomial_formula
from ._bootstrap import GaussianBootstrapResult, fit_bootstrap
from ._ee import GaussianEEResult, fit_gaussian_ee
from ._poisson_ee import PoissonEEResult, fit_poisson_ee
from ._binomial_ee import BinomialEEResult, fit_binomial_ee
from ._formula import fit_bootstrap_formula, fit_formula
from ._poisson import PoissonResult, fit_poisson, fit_poisson_formula
from ._poisson_bootstrap import PoissonBootstrapResult, fit_poisson_bootstrap
from ._risk_ratio import BinomialRiskRatioResult, fit_binomial_risk_ratio
from ._odds_ratio import BinomialOddsRatioResult, fit_binomial_odds_ratio
from ._quadratic import QuadraticGaussianBootstrapResult, fit_quadratic_bootstrap
from ._quadratic_ee import QuadraticGaussianEEResult, fit_quadratic_gaussian_ee
from ._cox import CoxResult, fit_cox, fit_cox_formula
from ._cox_bootstrap import CoxBootstrapResult, fit_cox_bootstrap
from ._cox_survcurve import CoxSurvivalCurves, qgcomp_survcurve_boot
from ._simdata import QuantizedSimulation, simdata_quantized
from ._impute import impute_leftcenslognorm, impute_tobit
from ._tidy import TidyResult, tidy
from ._glance import GlanceResult, glance
from ._tobit import TobitResult, fit_tobit
from ._multinomial import MultinomialResult, fit_multinomial
from ._multinomial_bootstrap import MultinomialBootstrapResult, fit_multinomial_bootstrap
from ._zero_inflated import ZeroInflatedPoissonResult, fit_zero_inflated_poisson
from ._zero_inflated_negbin import ZeroInflatedNegativeBinomialResult, fit_zero_inflated_negative_binomial
from ._zero_inflated_negbin_bootstrap import ZeroInflatedNegativeBinomialBootstrapResult, fit_zero_inflated_negative_binomial_bootstrap
from ._zero_inflated_bootstrap import ZeroInflatedPoissonBootstrapResult, fit_zero_inflated_poisson_bootstrap
from ._hurdle import HurdlePoissonResult, fit_hurdle_poisson
from ._hurdle_negbin import HurdleNegativeBinomialResult, fit_hurdle_negative_binomial
from ._hurdle_negbin_bootstrap import HurdleNegativeBinomialBootstrapResult, fit_hurdle_negative_binomial_bootstrap
from ._hurdle_bootstrap import HurdlePoissonBootstrapResult, fit_hurdle_poisson_bootstrap
from ._contrasts import se_comb, vc_comb
from ._partials import DataSplit, GaussianPartialResult, GLMPartialResult, split_data, qgcomp_partials
from ._pointwise import PointwiseBoundResult, pointwisebound_noboot
from ._pointwise_boot import pointwisebound_boot
from ._modelbound_boot import ModelBoundResult, modelbound_boot
from ._modelbound_ee import modelbound_ee
from ._msm_predict import msm_predict
from ._dispatch import qgcomp
from ._plots import plot_weights, plot_response

__all__ = [
    "fit", "fit_formula", "fit_bootstrap_formula", "fit_binomial", "fit_binomial_formula",
    "fit_poisson", "fit_poisson_formula", "fit_poisson_bootstrap", "fit_binomial_risk_ratio", "fit_binomial_odds_ratio", "fit_bootstrap",
    "fit_quadratic_bootstrap", "fit_quadratic_gaussian_ee", "fit_gaussian_ee", "fit_poisson_ee", "fit_binomial_ee", "fit_cox", "fit_cox_formula", "fit_cox_bootstrap", "CoxBootstrapResult", "qgcomp_survcurve_boot", "CoxSurvivalCurves", "fit_tobit", "fit_multinomial", "simdata_quantized", "QuantizedSimulation", "impute_leftcenslognorm", "impute_tobit", "tidy", "TidyResult", "glance", "GlanceResult",
    "fit_multinomial_bootstrap", "MultinomialBootstrapResult", "fit_zero_inflated_poisson", "fit_zero_inflated_negative_binomial", "ZeroInflatedNegativeBinomialResult", "fit_zero_inflated_poisson_bootstrap", "ZeroInflatedPoissonBootstrapResult", "fit_zero_inflated_negative_binomial_bootstrap", "ZeroInflatedNegativeBinomialBootstrapResult", "fit_hurdle_poisson", "fit_hurdle_negative_binomial", "HurdleNegativeBinomialResult", "fit_hurdle_poisson_bootstrap", "HurdlePoissonBootstrapResult", "fit_hurdle_negative_binomial_bootstrap", "HurdleNegativeBinomialBootstrapResult", "quantize", "se_comb", "vc_comb",
    "QGCompResult", "BinomialResult", "BinomialRiskRatioResult", "BinomialOddsRatioResult", "PoissonResult", "PoissonBootstrapResult", "GaussianBootstrapResult",
    "QuadraticGaussianBootstrapResult", "QuadraticGaussianEEResult", "GaussianEEResult", "PoissonEEResult", "BinomialEEResult", "CoxResult", "TobitResult", "MultinomialResult",
    "ZeroInflatedPoissonResult", "HurdlePoissonResult", "QuantizationResult",
    "QGCompWarning",
    "split_data", "qgcomp_partials", "DataSplit", "GaussianPartialResult", "GLMPartialResult",
    "pointwisebound_noboot", "pointwisebound_boot", "modelbound_boot", "modelbound_ee", "msm_predict", "qgcomp",
    "PointwiseBoundResult", "ModelBoundResult", "plot_weights", "plot_response",
]
