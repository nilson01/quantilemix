# Weighted Gaussian row bootstrap: fixed checks

The additive Gaussian bootstrap now accepts one strictly positive case weight
per independent row. Exposure cuts use the original unweighted exposure values.
A bootstrap draw keeps the row's outcome, exposure scores, adjustments and case
weight together. The fitted mean at joint score `a` averages predicted values
with these weights. The reported interval remains a normal interval based on
the standard deviation of the bootstrap effects.

Two deterministic cases use installed **R qgcomp 2.19.6** in an isolated R 4.5.1
library: already scored exposures and newly quantized exposures. Each has 54
rows and 12 shared bootstrap row schedules. R's resampled effects also agree
with separate weighted regressions on those same rows. Python and installed R
agree on scores, point effect, standard error, normal interval, all 24
bootstrap effects and weighted intervention means. The largest absolute
Python–R difference is `7.11e-15`. The [saved result](weighted-bootstrap-reference-results.json),
[fixture](../tests/fixtures/r_weighted_bootstrap_2_19_6.json),
[R exporter](export_weighted_bootstrap_r.R) and
[optional checker](check_weighted_bootstrap_r.py) record the comparison.

To repeat it with a compatible R 4.5.1 executable and an isolated library
containing qgcomp 2.19.6 and jsonlite, run from the package root:

```sh
python validation/check_weighted_bootstrap_r.py \
  --rscript /path/to/Rscript \
  --r-library /path/to/isolated-R-library \
  --output /new/path/weighted-bootstrap-check.json
```

The normal Python package and test suite do not require R. Independent Python
weighted normal-equation checks, fixed-R fixture checks and formula-row checks
cover the new path. The earlier 24,000-dataset bootstrap study used no case
weights, so it does **not** establish weighted interval coverage. These weights
are not a complex survey design, cluster resampling or an imputation method.
Zero and negative weights are rejected. Other bootstrap families remain
unweighted unless their own documentation says otherwise.
