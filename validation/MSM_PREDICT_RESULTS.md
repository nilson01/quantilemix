# Marginal-model prediction check

`msm_predict` covers additive Gaussian and Poisson row-bootstrap fits, including
Poisson fits with unequal observation times. It returns predicted outcomes on
the response scale from the fitted marginal model. The ordinary conditional
prediction method answers a different question. No uncertainty interval is
returned.

The fixed reference uses the exported `msm.predict` function in an isolated
installation of R `qgcomp` 2.19.6. `export_msm_predict_r.R` verifies the loaded
namespace, fits five datasets and saves both default predictions and
predictions for supplied joint scores. The cases cover prequantized and
quantized Gaussian and Poisson inputs, adjustment covariates, and an
observation-time offset. The exporter also records fitted levels, marginal
coefficients, fitted means and default score order. Its 24 bootstrap row draws
per case are saved so Python can fit the same data under the same resampling
schedule. The pinned fixture is
`tests/fixtures/r_msm_predict_2_19_6.json`, SHA-256
`0759a9c0d79a239501bcd5fc85ab0f29a1693c69554d8286eb84ffb3cbdcb66a`.

Run the optional comparison from the repository root with R qgcomp 2.19.6
installed in an isolated library:

```sh
PYTHONPATH=src python validation/check_msm_predict_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library
```

The checker regenerates the R output, checks the pinned inputs and results,
and compares both public Python prediction forms with installed R. All five
cases pass. The largest absolute Python/R difference is `3.17e-08`, at an
extrapolated Poisson prediction above 2,000; the relative difference there is
about `1.6e-11`. R's Poisson marginal fit has a tiny optimizer-scale difference
in its saved coefficients, which grows on exponentiation and extrapolation.
The fitted-level means agree much more closely. The comparison uses a mixed
absolute (`2e-8`) and relative (`3e-11`) tolerance so that the high prediction
is judged at its scale. The checker separately verifies the R score-block
order and marginal coefficients.

Independent tests check the Gaussian linear and Poisson log-link identities,
adjusted marginal versus conditional means, row-block output shape, read-only
results, supplied arrays and frame-like inputs, invalid scores, overflow,
changed fixtures and a missing R executable. Python users do not need R at
runtime. This is fixed-case software agreement, not a repeated-data assessment
of prediction error or evidence for unsupported model families.
