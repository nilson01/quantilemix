# Four-level marginal binary risk ratio

R qgcomp 2.19.6 fits a degree-one log-binomial marginal structural model
after predicting event risk under joint exposure scores 0, 1, 2 and 3. Its
slope is the log risk ratio for a one-category increase **according to that
model**. With four levels, it need not equal the log of the standardized risk
at score 1 divided by the risk at score 0. The two-level model is saturated,
so those values coincide there.

The Python `fit_binomial_risk_ratio` now accepts four levels; the public
`qgcomp(..., family="binomial")` route uses them by default with `q=4`.
The existing `fit_binomial_risk_ratio` default remains `q=2` for callers of
that direct function. The fit standardizes the conditional logit probabilities
over the observed adjustment rows, fits the degree-one log-risk model to
those probabilities, and repeats both fits in every fixed-cut row-bootstrap
draw. The result saves standardized risks, the marginal-model coefficients
and its fitted risks, the slope, its exponentiated ratio, bootstrap standard
error and normal interval. A failed draw raises an error.

Two deterministic 320-row cases, one with numeric adjustment and one without,
call R's exported `qgcomp` through an isolated installed R qgcomp 2.19.6.
The saved fixture is `tests/fixtures/r_risk_ratio_multilevel_2_19_6.json`
(SHA-256 `053728944a11ca6d91c180ae0f79772b9efa27389a047a5b5e8ee6e06dff99ed`).
The optional check regenerates both cases and compares the same 24 R
resampled-row schedules. The largest absolute Python/R difference is
`4.70e-08` in one saved bootstrap slope; point slopes differ by at most
`3.44e-09`. A separate test solves the fractional binomial score equations
independently and checks the marginal slope and all four fitted risks. The
public dispatcher and installed-package example check default `q=4`.
The dated machine-readable check is
`validation/risk-ratio-multilevel-reference-results.json`.

From the repository root, after installing R qgcomp 2.19.6 in an isolated
library, run the optional live check:

```sh
PYTHONPATH=src python validation/check_risk_ratio_multilevel_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

Python has no R runtime dependency. The R and Python seeded generators do
not have the same random stream; matching bootstrap uncertainty in this
check uses the exported R row schedules. These fixed cases establish
software agreement, not interval coverage. The earlier 4,000-dataset binary
risk-ratio study used **two** intervention levels and cannot be applied to
four-level inference. Case weights, missing-row removal, nonlinear
conditional terms, higher-degree marginal models, cluster resampling and
separated or boundary fits remain outside this route. R may return a finite
stopping value for some separated draws; Python reports a failed draw
instead, as documented for the earlier two-level case.
