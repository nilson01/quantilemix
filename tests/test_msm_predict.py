"""Independent prediction identities and public-input checks."""

import numpy as np
import pytest

from quantilemix import fit, fit_bootstrap, fit_poisson_bootstrap, msm_predict


def _rows():
    x = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (12, 1))
    row = np.arange(len(x))
    return x, row


def _gaussian():
    x, row = _rows()
    z = (row % 5).astype(float)
    y = 2 + 1.2*x[:, 0] - 0.4*x[:, 1] + 0.7*z + 0.08*(row % 7)
    return fit_bootstrap(y, x, z, q=2, B=12, seed=17)


def _poisson():
    x, row = _rows()
    time = 1 + row % 3
    y = (1 + x[:, 0] + 2*x[:, 1])*time + row % 2
    return fit_poisson_bootstrap(y, x, q=2, exposure_time=time, B=20, seed=9)


def test_gaussian_default_block_order_and_marginal_scale():
    model = _gaussian()
    n = len(model.point_fit.scores)
    predictions = msm_predict(model)
    np.testing.assert_allclose(predictions[:n], model.intervention_means[0])
    np.testing.assert_allclose(predictions[n:], model.intervention_means[1])
    assert predictions.shape == (2*n,)
    assert not predictions.flags.writeable
    assert abs(model.intervention_means[0]-model.point_fit.intercept) > 0.5


def test_gaussian_supplied_scores_follow_known_linear_identity():
    model = _gaussian()
    scores = np.array([-0.5, 0, 0.5, 1, 1.5])
    expected = model.intervention_means[0] + scores*model.psi
    np.testing.assert_allclose(msm_predict(model, {"psi": scores}), expected)
    np.testing.assert_allclose(msm_predict(model, scores), expected)
    np.testing.assert_allclose(np.diff(expected), 0.5*model.psi)


def test_poisson_supplied_scores_follow_log_link_with_time():
    model = _poisson()
    scores = np.array([-0.5, 0, 0.5, 1, 1.5])
    predictions = msm_predict(model, {"psi": scores})
    np.testing.assert_allclose(np.diff(np.log(predictions)), 0.5*model.psi)
    np.testing.assert_allclose(predictions[1], model.intervention_means[0])
    np.testing.assert_allclose(predictions[3], model.intervention_means[1])
    n = len(model.point_fit.scores)
    np.testing.assert_allclose(msm_predict(model), np.repeat(model.intervention_means, n))


def test_frame_like_input_uses_psi_column():
    class Frame:
        columns = ("psi", "irrelevant")

        def __getitem__(self, key):
            return {"psi": [0.0, 1.0], "irrelevant": [3, 4]}[key]

        def __contains__(self, key):
            return key in self.columns

    model = _gaussian()
    np.testing.assert_allclose(msm_predict(model, Frame()), model.intervention_means)


@pytest.mark.parametrize("bad", [[], [[0, 1]], [np.nan], [np.inf], ["one"]])
def test_bad_score_columns_rejected(bad):
    with pytest.raises(ValueError, match="newdata psi"):
        msm_predict(_gaussian(), {"psi": bad})


def test_missing_column_and_wrong_fit_rejected():
    model = _gaussian()
    with pytest.raises(ValueError, match="psi score column"):
        msm_predict(model, {"dose": [0, 1]})
    x, row = _rows()
    with pytest.raises(TypeError, match="bootstrap"):
        msm_predict(fit(2 + x[:, 0] + row % 3, x, q=2))


def test_count_extrapolation_overflow_rejected():
    with pytest.raises(ValueError, match="finite numeric range"):
        msm_predict(_poisson(), {"psi": [1e308]})
