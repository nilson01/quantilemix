# Quadratic Gaussian estimating equations

`fit_quadratic_gaussian_ee` fits a conditional
Gaussian model with each exposure score and its square. Optional terms add
pairwise exposure products and exposure-by-covariate products. It then fits
the population-average curve
`mean(a) = theta[0] + theta[1]*a + theta[2]*a*a`, where `a` sets every
exposure score to the same level. `psi = (theta[1], theta[2])`; the mean change
from `a` to `a+1` is `theta[1] + (2*a+1)*theta[2]`. Positive case weights are
allowed. Cluster scores, rather than individual row scores, form the robust
covariance. Exposure cutpoints remain unweighted.

Three fixed 128-row, 32-cluster cases use an **isolated installed** R qgcomp
2.19.6 library: squares alone, squares with one exposure pair, and both
exposure-pair and exposure-by-covariate products with case weights. The
[saved fixture](../tests/fixtures/r_quadratic_gaussian_ee_2_19_6.json)
has SHA-256 `dacf79833b4432584af5b7ca8335f47f6f0cfbd1a422e0f7694ac1d5ba53e8dd`.
The [replay record](quadratic-gaussian-ee-reference-results.json) verifies
that R's and Python's conditional coefficients, three marginal curve
coefficients, two `psi` coefficients, intervention means, scores and cutpoints
agree. The largest `psi` gap is `2.39e-13`. The derivative bread agrees within
`8.21e-7` because R uses a numerical derivative and Python uses the exact
one.

**Their robust intervals do not always agree.** R's cluster-meat helper
receives a `model.frame` containing already calculated `I(x1^2)` and similar
formula columns. Within each cluster it changes `x1` and `x2` to an
intervention level, but those saved formula columns retain their original
values. This can change the cluster scores used for its variance. A separate
R calculation rebuilt the full design at every intervention, then computed
each cluster score. Python's meat agrees with this fresh R calculation within
`3.39e-10`, and its full covariance agrees within `8.74e-12`. The installed
method's meat differs materially in all three cases. For example, with
squares alone, R gives a quadratic-term SE of `0.05243`; the fresh-design
calculation gives `0.03823`. Their 95% intervals are approximately
`[0.10025, 0.30578]` and `[0.12809, 0.27794]`, respectively. This is an
explicit correction to R 2.19.6's cluster-meat behavior, not a claim that
either interval has validated repeated-dataset coverage.

Independent Python tests derive the two polynomial contrasts from a
conditional least-squares fit and its cluster sandwich. Another test repeats
integer-weight rows within their original clusters and verifies that the
weighted and repeated-row coefficients, marginal curve and full robust
covariance agree. Input and checker failure cases are covered. The optional
installed-R replay requires Rscript plus an isolated library with qgcomp
2.19.6 and jsonlite:

```sh
PYTHONPATH=src python validation/check_quadratic_gaussian_ee_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

R is not needed at Python runtime. The three fixed cases do not establish
small-cluster performance, repeated-dataset interval coverage, other
nonlinear bases, binary or count nonlinear effects, missing-data behavior,
or a causal interpretation.
