"""Independent arithmetic and contract checks for bootstrap pointwise tables."""

import numpy as np
import pytest

from quantilemix import fit_bootstrap, fit_poisson_bootstrap, pointwisebound_boot


def _data():
    cells = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (12, 1))
    row = np.arange(len(cells))
    return cells, row


def test_gaussian_profile_contrast_and_reference():
    x, row = _data()
    y = 2 + 1.4*x[:, 0] - 0.4*x[:, 1] + 0.13*(row % 5)
    model = fit_bootstrap(y, x, q=2, B=20, seed=5)
    table = pointwisebound_boot(model, pointwiseref=2)
    expected = model.bootstrap_means[:, 0]-model.bootstrap_means[:, 1]
    assert table.columns[-2:] == ("ll.linpred", "ul.linpred")
    np.testing.assert_allclose(table["mean.diff"], [-model.psi, 0], atol=1e-12)
    np.testing.assert_allclose(table["se.diff"], [np.std(expected, ddof=1), 0])
    np.testing.assert_allclose(table["linpred"], model.intervention_means)
    assert not table["linpred"].flags.writeable
    assert not model.bootstrap_means.flags.writeable


def test_poisson_log_profile_contrast_with_time():
    x, row = _data()
    time = 1 + row % 3
    y = (1+x[:, 0]+2*x[:, 1])*time + row % 2
    model = fit_poisson_bootstrap(y, x, q=2, exposure_time=time, B=20, seed=9)
    table = pointwisebound_boot(model)
    expected = np.log(model.bootstrap_means[:, 1]/model.bootstrap_means[:, 0])
    np.testing.assert_allclose(table["se.lnrr"][1], np.std(expected, ddof=1))
    np.testing.assert_allclose(table["rr"][1],
                               model.intervention_means[1]/model.intervention_means[0])
    np.testing.assert_allclose(table["se.lnrr"][0], 0)


@pytest.mark.parametrize("reference", [0, 3, 1.5, True])
def test_bad_reference_rejected(reference):
    x, row = _data()
    y = 2+x[:, 0]+row % 3
    model = fit_bootstrap(y, x, q=2, B=12, seed=3)
    with pytest.raises(ValueError, match="pointwiseref"):
        pointwisebound_boot(model, pointwiseref=reference)


@pytest.mark.parametrize("alpha", [0, 1, np.nan, True])
def test_bad_alpha_rejected(alpha):
    x, row = _data()
    y = 2+x[:, 0]+row % 3
    model = fit_bootstrap(y, x, q=2, B=12, seed=3)
    with pytest.raises(ValueError, match="alpha"):
        pointwisebound_boot(model, alpha=alpha)


def test_nonbootstrap_fit_rejected():
    with pytest.raises(TypeError, match="bootstrap"):
        pointwisebound_boot(object())


def test_null_q_rejected():
    x, row = _data()
    y = 2+x[:, 0]+row % 3
    model = fit_bootstrap(y, x, q=None, B=12, seed=3)
    with pytest.raises(ValueError, match="non-null q"):
        pointwisebound_boot(model)
