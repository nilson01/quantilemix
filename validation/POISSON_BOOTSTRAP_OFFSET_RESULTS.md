# Marginal Poisson bootstrap with unequal observation times

Two fixed cases were exported from an isolated installed
R 4.5.1 library containing qgcomp 2.19.6. The exporter checks that the loaded
qgcomp namespace comes from that library. One case has two already-scored
levels and an exact joint rate ratio of six. The other has estimated quartile
cuts and one numeric adjustment covariate. Each case contains positive,
unequal observation times and 24 R-generated row-resampling schedules.

R fits `y ~ x1 + x2 + offset(log(time))`, adding `z` in the second case. It
predicts expected counts at each joint score while retaining each row's
observation time. Its marginal model uses those predicted counts, so the saved
intervention means are **counts over the observed time distribution**, not
one-unit-time rates. Because the same time distribution is used at both score
levels, the additive log-link model makes the ratio of these means equal to
the conditional one-step rate ratio. During bootstrap resampling, the time
value stays with its original count, scores and covariate value.

The optional live checker compared the saved fixture with a new export from
the installed package. It then compared Python's joint log ratio, bootstrap
standard error, confidence interval, p-value, intervention means, scores,
cutpoints and all 48 replicate effects. A separate SciPy score-equation solver
checked the point and every replicate without calling the package fitter.
All comparisons passed at an absolute tolerance of `2e-9` for effects and
`1e-10` for cutpoints. The largest observed absolute difference was
`5.83e-10` in one prequantized replicate. R stores synthetic boundaries for
already-scored inputs; Python retains `breaks=None` because it did not group
the exposures.

The fixed R fixture is
`tests/fixtures/r_poisson_bootstrap_offset_2_19_6.json` (SHA-256
`63c22e14b45ac709e8cf68654c79a51d23d74c117f3e2789197b2c0370988108`).
The exporter is `validation/export_poisson_bootstrap_offset_r.R` (SHA-256
`89b856960b420de2cc6fee9dc2ea9583fd7c2281bf5b06b487994578e7078a84`).
The saved live-check report is
`validation/poisson-bootstrap-offset-reference-results.json` (SHA-256
`c5818d7cbf132008391cd74d3feda14f5891e01c8a9b57d77c3d571dae3c0cf9`).
To repeat the optional comparison, run:

```sh
PYTHONPATH=src python validation/check_poisson_bootstrap_offset_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

This establishes fixed-case software agreement and the stated same-time
interpretation. The earlier 4,000-dataset marginal bootstrap study used equal
periods. It does **not** establish confidence-interval coverage with unequal
times, interactions, overdispersed counts, clustered rows, case weights or
other R bootstrap options. R is needed only for this optional development
check, not to use the Python package.
