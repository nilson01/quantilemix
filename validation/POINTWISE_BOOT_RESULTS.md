# Marginal bootstrap pointwise comparison check

The supported subset is additive Gaussian and Poisson row-bootstrap fits with a
specified quantile count. Poisson fits may include an observation-time offset.
The result compares each joint exposure score with a selected score, using
one-based `pointwiseref` as in R. It returns separate normal confidence limits
for each comparison, not a simultaneous confidence band.

The fixed reference uses an isolated installation of R `qgcomp` 2.19.6.
`export_pointwise_boot_r.R` verifies the loaded namespace is in the requested
library, fits three datasets, and saves six pointwise tables. Each dataset has
24 specified bootstrap row draws. Gaussian, Poisson and offset-Poisson fits
each use the first and last score as references. The output is pinned at
`tests/fixtures/r_pointwise_boot_2_19_6.json`, SHA-256
`ce38439ebda62c27eefc4d56c2185e9ba39710bf1d8270347f2a69783fd1c710`.

Run the optional comparison from the repository root with an installed R
`qgcomp` 2.19.6 library:

```sh
PYTHONPATH=src python validation/check_pointwise_boot_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library
```

The check regenerates each R table and compares its inputs, marginal means,
bootstrap covariance and every displayed column with the pinned fixture. It
then fits Python using the same bootstrap rows. The six Python tables agree
with installed R within `2e-8`; the largest observed absolute difference in
this environment was `1.23e-10`. The paired covariance and mean checks also
pass within `2e-8`. Independent tests check the contrast arithmetic, reference
row, log-scale count uncertainty, read-only arrays and invalid inputs.

One implementation detail matters for interpretation. R stores the fitted
marginal outcome mean on the response scale, but its `cov.yhat` for Poisson
bootstrap fits comes from `predict(msmfit)` on the default **log scale**. The
pointwise standard error therefore measures variation of log-scale
differences. The source documentation describes these predictions less
precisely; this check follows the installed computation. Gaussian identity-link
predictions have the same scale as the means.

These fixed-case agreements check software behavior only. The separate
Gaussian 24,000-dataset study assessed the overall joint-effect interval, not
this new set of pointwise bootstrap intervals. Binary, nonlinear, clustered,
zero-inflated and other bootstrap families, repeated-data coverage and
simultaneous bounds remain open. Python imports and fits without R installed;
the R checker is optional.
