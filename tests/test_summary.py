"""Display checks with pinned R values and independent algebraic targets."""

import ast
from dataclasses import fields, is_dataclass
from itertools import product
import json
import math
from pathlib import Path
from collections.abc import Mapping
import warnings

import numpy as np
import pytest
from scipy.stats import norm, t

import quantilemix
from quantilemix import QGCompWarning, fit
from quantilemix import _core


CASES = json.loads(
    (Path(__file__).parent / "fixtures/r_gaussian_2_19_6.json").read_text()
)["cases"]


def joint_row(text):
    """Read the one displayed mixture row, without calling private formatters."""
    rows = [line.split() for line in text.splitlines() if line.startswith("psi ")]
    assert len(rows) == 1
    assert len(rows[0]) == 7
    return rows[0][1:]


def named_rows(text, heading, following):
    block = text.split(heading, 1)[1].split(following, 1)[0].strip().splitlines()
    if block == ["None (no coefficients in this sign group)."]:
        return []
    return [(ast.literal_eval(line.rsplit(maxsplit=1)[0]),
             line.rsplit(maxsplit=1)[1]) for line in block[1:] if line]


def rational_data():
    # The residual (a-3/2)(b-3/2) is orthogonal to the intercept, a, a+b,
    # and both crossed covariates. SSE=100, residual df=59, psi=1,
    # Var(psi)=5/236. These targets are derived in test_core.py.
    rows = np.asarray(list(product(range(4), range(4), (-1, 1), (-1, 1))), dtype=float)
    a, b = rows[:, 0], rows[:, 1]
    x, z = np.column_stack((a, a + b)), rows[:, 2:]
    residual = (a - 1.5) * (b - 1.5)
    y = 3 + 2 * x[:, 0] - x[:, 1] + z @ np.array([.5, -.25]) + residual
    return y, x, z, residual


@pytest.fixture
def rational_fit():
    y, x, z, _ = rational_data()
    return fit(y, x, z, q=None)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_summary_displays_pinned_r_estimates_uncertainty_and_weights(case):
    inputs, expected = case["inputs"], case["expected"]
    result = fit(inputs["y"], inputs["exposures"], inputs["covariates"],
                 q=inputs["q"], breaks=inputs["breaks"], alpha=inputs["alpha"])
    text = result.summary()
    # R computes 2*(1-pt(abs(t), df)); subtraction can round a small p-value
    # to zero. Verify Python's preserved tail against an independent SciPy
    # distribution call, using R's stored statistic and degrees of freedom.
    expected_tail = 2 * t.sf(abs(expected["statistic"]), expected["df_pvalue"])
    target = [expected["psi"], expected["se"], *expected["ci"],
              expected["statistic"], expected_tail]
    assert joint_row(text) == [format(value, ".6g") for value in target]
    assert f"Residual degrees of freedom: {expected['df_residual']}" in text
    assert f"degrees of freedom = {expected['df_pvalue']}" in text
    assert f"alpha = {inputs['alpha']!r}" in text
    assert f"Observations: {len(inputs['y'])}" in text
    assert "R qgcomp 2.19.6" in text
    residual = np.asarray(inputs["y"]) - np.asarray(expected["fitted"])
    residual_variance = residual @ residual / expected["df_residual"]
    assert f"Residual variance: {residual_variance:.6g}" in text
    for sign in ("Positive", "Negative"):
        prefix = sign.lower()
        assert f"{sign} coefficient sum: {expected[prefix + '_psi']:.6g}" in text
        following = "Negative weights:" if sign == "Positive" else "Exactly zero"
        displayed = named_rows(text, f"{sign} weights:", following)
        target_weights = [(name, format(value, ".6g"))
                          for name, value in expected[prefix + "_weights"].items()]
        assert displayed == target_weights
    coefficient_rows = named_rows(text, "Model coefficients (estimates only; fitted column order)",
                                  "Numerical and grouping diagnostics")
    n_exposures = len(inputs["exposures"][0])
    n_covariates = 0 if inputs["covariates"] is None else len(inputs["covariates"][0])
    names = ["(Intercept)"] + [f"x{i + 1}" for i in range(n_exposures)]
    names += [f"z{i + 1}" for i in range(n_covariates)]
    assert coefficient_rows == list(zip(names, [format(v, ".6g") for v in expected["coefficients"]]))


def test_summary_independent_rational_targets_and_distinct_inference_rules(rational_fit):
    text = rational_fit.summary()
    se = math.sqrt(5 / 236)
    expected = [1, se, 1 - norm.ppf(.975) * se, 1 + norm.ppf(.975) * se,
                1 / se, 2 * t.sf(1 / se, 61)]
    assert joint_row(text) == [format(value, ".6g") for value in expected]
    assert "Residual variance: 1.69492" in text
    assert "Residual degrees of freedom: 59" in text
    assert "degrees of freedom = 61" in text
    assert "Normal confidence interval; alpha = 0.05" in text
    assert "normal interval and t test need not give the same threshold decision" in text
    assert "Weights do not measure selection probability or causal importance" in text
    assert "do not test model assumptions or establish causality" in text


@pytest.mark.parametrize("alpha", [.1, .2])
def test_summary_reports_the_fitted_alpha_and_corresponding_limits(alpha):
    y, x, z, _ = rational_data()
    result = fit(y, x, z, q=None, alpha=alpha)
    row = joint_row(result.summary())
    se = math.sqrt(5 / 236)
    expected_limits = [1 - norm.ppf(1 - alpha / 2) * se,
                       1 + norm.ppf(1 - alpha / 2) * se]
    assert row[2:4] == [format(value, ".6g") for value in expected_limits]
    assert row[-1] == format(2 * t.sf(1 / se, 61), ".6g")
    assert f"alpha = {alpha}" in result.summary()


@pytest.mark.parametrize("digits", range(1, 18))
def test_summary_significant_digits_preserve_small_nonzero_values(rational_fit, digits):
    text = rational_fit.summary(digits=digits)
    displayed = joint_row(text)
    se = math.sqrt(5 / 236)
    expected = [1, se, 1 - norm.ppf(.975) * se, 1 + norm.ppf(.975) * se,
                1 / se, 2 * t.sf(1 / se, 61)]
    for actual, target in zip(displayed, expected):
        # At high display precision, allow numerical OLS roundoff; at low
        # precision, allow half a unit in the last significant decimal place.
        rounding = .5000001 * 10 ** (math.floor(math.log10(abs(target))) - digits + 1)
        assert float(actual) == pytest.approx(target, abs=rounding + 1e-12 * abs(target))
    assert "e-" in displayed[-1]
    assert float(displayed[-1]) > 0
    assert f"Display precision: {digits} significant digits" in text
    assert "result attributes retain full precision" in text


def test_summary_accepts_numpy_integer_precision(rational_fit):
    assert rational_fit.summary(digits=np.int64(6)) == rational_fit.summary()


@pytest.mark.parametrize("digits", [0, 18, -1, 10**30, True, False, np.bool_(True),
                                    6.0, 6.5, "6", None, np.nan, np.inf, [], np.array(6)])
def test_summary_invalid_precision_is_explicit(rational_fit, digits):
    with pytest.raises(ValueError, match="digits must be an integer from 1 through 17"):
        rational_fit.summary(digits=digits)


def test_summary_precision_is_keyword_only(rational_fit):
    with pytest.raises(TypeError):
        rational_fit.summary(6)


@pytest.mark.parametrize("sign", [-1, 1])
def test_summary_absent_sign_group_and_weight_order(sign):
    _, x, z, residual = rational_data()
    result = fit(sign * (2 * x[:, 0] + x[:, 1]) + residual, x, z, q=None)
    text = result.summary()
    present, absent = ("Positive", "Negative") if sign == 1 else ("Negative", "Positive")
    following = "Negative weights:" if sign == 1 else "Exactly zero"
    assert named_rows(text, f"{present} weights:", following) == [("x1", "0.666667"), ("x2", "0.333333")]
    assert f"{absent} weights:\n  None (no coefficients in this sign group)." in text
    assert "Exactly zero exposure coefficients: none" in text


def test_summary_exact_zeros_and_degenerate_inference_are_not_significance():
    x = np.asarray(list(product((0, 1), repeat=2)), dtype=float)
    with pytest.warns(QGCompWarning, match="at numerical zero"):
        result = fit(np.zeros(4), x, q=None)
    text = result.summary()
    assert joint_row(text) == ["0", "0", "0", "0", "NA", "NA"]
    assert "Exactly zero exposure coefficients: 'x1', 'x2'" in text
    assert text.count("None (no coefficients in this sign group).") == 2
    assert "does not establish certainty" in text
    assert "t and p-value are unavailable (NA)" in text


def test_summary_explains_recorded_ill_conditioning():
    x = np.repeat([0., 1.], 4)
    z = np.tile(np.repeat([0., 1.], 2), 2)
    y = 1 + 2 * x + 3 * z + np.tile([-.125, .125], 4)
    with pytest.warns(QGCompWarning, match="condition number"):
        result = fit(y, x, 1e-13 * z, q=None)
    text = result.summary()
    assert "The design is ill-conditioned" in text
    assert "coefficients can be sensitive to small changes" in text
    assert "below the numerical warning threshold" not in text
    assert "Design rank: 3" in text


def test_summary_explains_tied_quantiles_without_claiming_four_groups():
    x = np.repeat([0., 0., 0., 1.], 4)
    y = 1 + 2 * x + np.tile([-.1, .1], 8)
    text = fit(y, x).summary()
    assert "Quantization: quantile; q: 4" in text
    assert "Tied cutpoints reduced the number of categories for: 'x1'" in text
    assert "Defined bins per exposure: 'x1'=2" in text


def test_summary_explains_empty_explicit_bins():
    x = np.repeat([.1, .9], 4)
    y = 1 + 2 * x + np.tile([-.1, .1], 4)
    text = fit(y, x, breaks=[0, .2, .4, .6, 1]).summary()
    assert "Quantization: explicit; q: not used" in text
    assert "Defined bins per exposure: 'x1'=4" in text
    assert "Empty fitted bins (zero-based scores): 'x1': (1, 2)" in text


def test_summary_escapes_control_characters_and_preserves_unicode_names():
    class LabeledArray:
        def __init__(self, values, columns):
            self.values, self.columns = values, columns

        def __array__(self, dtype=None, copy=None):
            array = np.asarray(self.values, dtype=dtype)
            return array.copy() if copy else array

    y, x, z, _ = rational_data()
    names = ("dose β\nnext\tline\x1b[31m", "'quoted' exposure\u202e")
    covariate_names = ("âge", "second\x00covariate")
    text = fit(y, LabeledArray(x, names), LabeledArray(z, covariate_names), q=None).summary()
    for name in (*names, *covariate_names):
        assert repr(name) in text
    for character in ("\t", "\x1b", "\u202e", "\x00"):
        assert character not in text
    coefficients = named_rows(text, "Model coefficients (estimates only; fitted column order)",
                              "Numerical and grouping diagnostics")
    assert [name for name, _ in coefficients] == ["(Intercept)", *names, *covariate_names]


def state(value):
    """Capture values, array ownership flags and nested records before display."""
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes(), value.flags.writeable
    if is_dataclass(value):
        return tuple((field.name, state(getattr(value, field.name))) for field in fields(value))
    if isinstance(value, Mapping):
        return tuple((key, state(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(state(item) for item in value)
    return repr(value)


def test_summary_is_quiet_repeatable_and_reads_the_saved_fit(rational_fit, monkeypatch, capsys):
    before = state(rational_fit)

    def forbidden_calculation(*args, **kwargs):
        raise AssertionError("summary must use the saved fit")

    for module, names in ((quantilemix, ("fit", "quantize")),
                          (_core, ("fit", "quantize")),
                          (np.linalg, ("svd", "solve", "lstsq")),
                          (_core.special, ("ndtri", "stdtr"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden_calculation)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        text = rational_fit.summary()
        assert isinstance(text, str)
        assert rational_fit.summary() == text
    assert not recorded
    assert capsys.readouterr() == ("", "")
    assert state(rational_fit) == before
