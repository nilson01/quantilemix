"""Independent split and held-out partial-effect checks."""

import numpy as np
import pytest

from quantilemix import qgcomp_partials, split_data


def test_split_is_disjoint_reproducible_and_preserves_rows():
    data = {"x": np.arange(20), "label": np.array(["a", "b"] * 10)}
    one = split_data(data, seed=7)
    two = split_data(data, seed=7)
    assert one.train_indices == two.train_indices
    assert len(one.train_indices) == 8
    assert set(one.train_indices).isdisjoint(one.validation_indices)
    assert sorted(one.train_indices + one.validation_indices) == list(range(20))
    np.testing.assert_array_equal(one.train_data["x"], data["x"][list(one.train_indices)])


def test_cluster_split_assigns_whole_groups():
    data = {"x": np.arange(24), "id": np.repeat(np.arange(6), 4)}
    split = split_data(data, cluster="id", seed=31)
    assert len(split.train_indices) == 8
    assert set(split.train_data["id"]).isdisjoint(set(split.validation_data["id"]))


@pytest.mark.parametrize("bad", [0, 1, float("nan"), True])
def test_split_rejects_invalid_fraction(bad):
    with pytest.raises(ValueError, match="prop_train"):
        split_data({"x": np.arange(10)}, prop_train=bad)


def test_split_rejects_missing_cluster_and_empty_side():
    with pytest.raises(ValueError, match="cluster IDs"):
        split_data({"id": [0, 0, float("nan"), 1]}, cluster="id")
    with pytest.raises(ValueError, match="at least one"):
        split_data({"x": [1, 2]}, prop_train=.1)


def test_partials_use_training_signs_and_validation_coefficients():
    grid = np.array([(a, b, r) for r in range(1, 9) for a in range(4)
                     for b in range(4)], dtype=float)
    x = grid[:, :2]
    z = (grid[:, 2] % 3).reshape(-1, 1)
    y = 1 + .8*x[:, 0] - .45*x[:, 1] + .2*z[:, 0] + .1*np.sin(np.arange(1, len(x)+1))
    train = grid[:, 2] <= 4
    res = qgcomp_partials(y[train], x[train], y[~train], x[~train],
                          train_covariates=z[train], validation_covariates=z[~train], q=None)
    assert res.positive_exposures == ("x1",)
    assert res.negative_exposures == ("x2",)
    assert res.positive_fit is not None and res.negative_fit is not None
    # Independent direct least-squares contrast and full covariance, with
    # the other exposure kept as a held-out adjustment column.
    for selected, fit in ((0, res.positive_fit), (1, res.negative_fit)):
        design = np.column_stack((np.ones((~train).sum()), x[~train, selected],
                                  x[~train, 1-selected], z[~train]))
        beta = np.linalg.lstsq(design, y[~train], rcond=None)[0]
        residual = y[~train] - design @ beta
        variance = residual @ residual / (len(residual) - design.shape[1])
        se = np.sqrt(variance * np.linalg.inv(design.T @ design)[1, 1])
        assert fit.psi == pytest.approx(beta[1], abs=1e-12)
        assert fit.se == pytest.approx(se, abs=1e-12)
    assert "held-out" in res.summary()


def test_all_positive_signs_produce_no_negative_inference():
    x = np.array([(a, b, r) for r in range(8) for a in range(3)
                  for b in range(3)], dtype=float)[:, :2]
    y = 1 + .6*x[:, 0] + .4*x[:, 1] + .01*np.sin(np.arange(len(x)))
    res = qgcomp_partials(y[:36], x[:36], y[36:], x[36:], q=None)
    assert res.negative_exposures == ()
    assert res.negative_fit is None
    assert "unavailable" in res.summary()


def test_partials_reject_misaligned_inputs():
    x = np.arange(20).reshape(10, 2)
    with pytest.raises(ValueError, match="both be present"):
        qgcomp_partials(np.arange(10), x, np.arange(10), x,
                        train_covariates=np.ones(10), q=None)


def test_raw_exposure_cuts_are_reused_when_requested():
    grid = np.array([(a, b, r) for r in range(8) for a in range(4)
                     for b in range(4)], dtype=float)
    x = grid[:, :2]
    y = 1 + .7*x[:, 0] - .4*x[:, 1] + .03*np.sin(np.arange(len(x)))
    train = grid[:, 2] < 4
    result = qgcomp_partials(y[train], x[train], y[~train], x[~train], q=4)
    assert result.positive_exposures == ("x1",)
    assert result.negative_exposures == ("x2",)
    np.testing.assert_array_equal(result.positive_fit.breaks[0], result.training_fit.breaks[0])
    np.testing.assert_array_equal(result.negative_fit.breaks[0], result.training_fit.breaks[1])


def test_named_exposures_keep_training_group_names():
    class NamedArray:
        columns = ("lead", "arsenic")

        def __init__(self, values):
            self.values = values

        def __array__(self, dtype=None, copy=None):
            return np.array(self.values, dtype=dtype, copy=copy)

    x = np.array([(a, b, r) for r in range(8) for a in range(4)
                  for b in range(4)], dtype=float)[:, :2]
    y = 1 + .8*x[:, 0] - .45*x[:, 1] + .02*np.sin(np.arange(len(x)))
    result = qgcomp_partials(y[:64], NamedArray(x[:64]), y[64:], NamedArray(x[64:]), q=None)
    assert result.positive_exposures == ("lead",)
    assert result.negative_exposures == ("arsenic",)
    assert result.positive_fit.exposure_names == ("lead",)
    assert result.negative_fit.exposure_names == ("arsenic",)
    assert result.positive_fit.covariate_names == ("arsenic",)
    assert result.negative_fit.covariate_names == ("lead",)
