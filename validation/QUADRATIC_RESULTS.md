# Restricted quadratic Gaussian joint response

`fit_quadratic_bootstrap` fits every exposure score
and its square in a conditional Gaussian model. It reports a degree-two
marginal joint-response curve, not one constant joint effect. Its saved
linear and quadratic coefficients determine the fitted mean change from
joint score `a` to `a+1` as `psi[0] + (2*a+1)*psi[1]`.

The two coefficients use the full covariance of `B` independent-row bootstrap
fits. Exposure cutpoints are fixed before resampling. The normal intervals
and Student t p-values follow the restricted R qgcomp 2.19.6 degree-two
convention. The method requires complete numeric rows, at least three
intervention levels, a full-rank conditional design and positive reference
degrees of freedom. These two cases omit interactions, clusters and case weights.

## Fixed reference evidence

Two cases from an isolated installed R qgcomp **2.19.6** check prequantized
scores and automatic quartile cuts. The saved
[fixture](../tests/fixtures/r_quadratic_2_19_6.json) includes exact R
bootstrap row schedules. Python reuses those schedules, then compares full
conditional coefficients and covariance, all fitted and intervention means,
the marginal polynomial, every bootstrap coefficient pair, their covariance,
intervals, test statistics and p-values. Scores and intervention levels agree
exactly. The largest observed absolute Python/R field gap is
`2.7711166694643907e-13` (one test statistic); all numeric fields pass
`1e-10` absolute tolerance. The
[report](quadratic-reference-results.json) saves field-level differences.
The fixture SHA-256 is
`4a22a63abcc07d05223818b6b9614be09bbfffd34a05c1c71550cc0bced647ea`;
the exporter SHA-256 is
`a41b95caf702cc263c4dcf01ec6a697eacae832d2c67173052aae264790c8934`.

Re-export and compare with the optional installed-package checker:

```sh
python validation/check_quadratic_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker verifies its saved source identities, the loaded installed R
namespace, case inputs and shared row schedules. Independent Python tests
solve the polynomial design directly, check the complete bootstrap covariance,
verify changes between consecutive intervention levels and reject invalid
inputs. R saves synthetic cutpoint endpoints for prequantized bootstrap
scores; Python keeps its established `breaks=None` contract. The scores and
inference agree.

## Limits

This is fixed software agreement, not repeated-sample interval calibration.
The earlier 24,000-dataset Gaussian bootstrap study tested only the additive
model and does not transfer to this curved model. Causal interpretation still
requires appropriate study assumptions. Separate reports cover [interactions](INTERACTIONS_RESULTS.md) and
[clustered quadratic fits](QUADRATIC_GAUSSIAN_EE_RESULTS.md).
