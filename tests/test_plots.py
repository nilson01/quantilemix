"""Plot labels and values must reflect the saved statistical results."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from quantilemix import (
    fit, fit_binomial, fit_binomial_odds_ratio, fit_binomial_risk_ratio,
    fit_bootstrap, fit_poisson, fit_poisson_bootstrap,
    plot_response, plot_weights,
)


class RecordingAxes:
    """Capture the values supplied to the plotting backend."""

    def __init__(self):
        self.bars = []
        self.lines = []
        self.labels = {}

    def barh(self, y, values, **kwargs):
        self.bars.append((np.asarray(y), np.asarray(values), kwargs))

    def plot(self, x, y, **kwargs):
        self.lines.append((np.asarray(x), np.asarray(y), kwargs))

    def set_yticks(self, positions, labels):
        self.labels["yticks"] = (np.asarray(positions), labels)

    def set_xticks(self, values):
        self.labels["xticks"] = np.asarray(values)

    def set_xlim(self, left, right):
        self.labels["xlim"] = (left, right)

    def set_xlabel(self, label):
        self.labels["xlabel"] = label

    def set_ylabel(self, label):
        self.labels["ylabel"] = label

    def set_title(self, label):
        self.labels["title"] = label

    def legend(self):
        self.labels["legend"] = True


def _scores():
    i = np.arange(160)
    x = np.column_stack((i % 2, (i // 2) % 2)).astype(float)
    z = ((i*7) % 13 / 12)[:, None]
    binary = (((i*19+3) % 29) / 29 < .22 + .1*x[:, 0] + .09*x[:, 1]
              + .11*z[:, 0]).astype(float)
    return x, z, binary


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_weights_show_separate_sign_groups(family):
    x, z, binary = _scores()
    if family == "gaussian":
        result = fit(2 + 2*x[:, 0] - x[:, 1] + .1*z[:, 0]
                     + .1*x[:, 0]*x[:, 1], x, z, q=None)
    elif family == "binomial":
        result = fit_binomial(binary, x, z, q=None)
    else:
        count = (x[:, 0] + 2*x[:, 1] + 1).astype(float)
        result = fit_poisson(count, x, z, q=None)
    ax = RecordingAxes()
    assert plot_weights(result, ax=ax) is ax
    assert ax.labels["yticks"][1] == list(result.exposure_names)
    assert ax.labels["xlim"] == (0, 1)
    assert "within coefficient-sign group" in ax.labels["xlabel"]
    assert_allclose(ax.bars[0][1], [result.pos_weights.get(n, 0)
                                    for n in result.exposure_names])
    assert_allclose(ax.bars[1][1], [result.neg_weights.get(n, 0)
                                    for n in result.exposure_names])
    assert ax.bars[0][2]["label"] == "Positive"
    assert ax.bars[1][2]["label"] == "Negative"


@pytest.mark.parametrize("family,expected_label", [
    ("gaussian", "Standardized mean outcome"),
    ("poisson", "Standardized expected count"),
    ("risk_ratio", "Standardized event risk"),
    ("odds_ratio", "Standardized event risk"),
])
def test_response_uses_saved_level_values_and_scale(family, expected_label):
    x, z, binary = _scores()
    if family == "gaussian":
        result = fit_bootstrap(2 + x[:, 0] + 2*x[:, 1] + .1*z[:, 0]
                               + .1*x[:, 0]*x[:, 1],
                               x, z, q=None, B=4, seed=5)
        values = result.intervention_means
    elif family == "poisson":
        result = fit_poisson_bootstrap((x[:, 0] + 2*x[:, 1] + 1).astype(float),
                                       x, z, q=None, B=4, seed=5)
        values = result.intervention_means
    elif family == "risk_ratio":
        result = fit_binomial_risk_ratio(binary, x, z, q=None, B=4, seed=5)
        values = result.intervention_risks
    else:
        result = fit_binomial_odds_ratio(binary, x, z, q=None, B=4, seed=5)
        values = result.intervention_risks
    ax = RecordingAxes()
    assert plot_response(result, ax=ax) is ax
    assert_allclose(ax.lines[0][0], result.intervention_levels)
    assert_allclose(ax.lines[0][1], values)
    assert_allclose(ax.labels["xticks"], result.intervention_levels)
    assert ax.labels["ylabel"] == expected_label
    assert "each exposure" in ax.labels["xlabel"]


def test_wrong_result_fails_before_drawing():
    ax = RecordingAxes()
    with pytest.raises(TypeError, match="plot_weights needs"):
        plot_weights(object(), ax=ax)
    with pytest.raises(TypeError, match="plot_response needs"):
        plot_response(object(), ax=ax)
    assert not ax.bars and not ax.lines
