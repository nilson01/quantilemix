# Clustered Gaussian marginal-curve bounds

`modelbound_ee` gives pointwise limits and the
R-method simultaneous bounds for an additive clustered Gaussian estimating-
equation fit. Both methods use 5,000 draws of the marginal intercept and slope
from their fitted normal approximation. The two implementations use different
random generators, so the same integer seed does not produce the same table.

The optional exporter fits one 96-row, 32-cluster dataset with four joint
quantile levels using an isolated installation of R qgcomp 2.19.6. It saves
R's actual 5,000 paired coefficient draws and runs `modelbound.ee` with and
without simultaneous bounds. Python fits the same data and calculates both
tables from those saved draws. The largest table difference is
`7.99e-15`; marginal coefficients and covariance differ by at most
`1.39e-15` and `1.16e-14`. The saved fixture SHA-256 is
`7f8141e241ed84a7ac97dd5d0d598cb48d1c67a607688a28d3105897575934fc`.
The [machine-readable comparison](modelbound-ee-r-results.json) records each
column's difference.

Reproduce the installed-R comparison with an Rscript compatible with the
isolated library:

```sh
PYTHONPATH=src python validation/check_modelbound_ee_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

R and `jsonlite` are optional validation tools. The Python package does not
need either at runtime. The check verifies that the loaded R qgcomp comes
from the supplied library. The Python-only tests verify seed reproducibility,
pointwise scale, missing simultaneous columns, read-only output and clear
input failures. Installed R `modelbound.ee` errors for `q=NULL` at
`seq_len(x$q)`; this comparison therefore uses `q=4`. Python accepts a
prequantized Gaussian EE result if it has consecutive intervention levels.
Fixed-data agreement does not establish confidence-interval coverage across
repeated datasets.
