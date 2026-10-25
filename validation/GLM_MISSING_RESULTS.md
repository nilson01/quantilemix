# Binary and count complete-case formulas: fixed reference check

The comparison covers
`fit_binomial_formula` and `fit_poisson_formula` with an explicit
`missing="drop"` option. Their default still rejects missing values. Array
fits still require complete inputs. The formula result records original row
count and retained zero-based positions. Exposure cutpoints use all observed
exposure values **before** incomplete model rows are omitted, matching the
installed R qgcomp 2.19.6 sequence.

Four deterministic cases were exported from an isolated installed R library:
one estimated-quartile and one prequantized case for each family. Each contains
missing outcome, exposure, numeric adjustment and category values. The
quantized cases retain 105 of 128 rows; the prequantized cases retain 124.
In the quantized cases, estimating the first exposure's first cut after row
removal instead would change it from 0.26171875 to 0.390625. It would change
the joint log odds estimate by about -0.0192 and the joint log mean estimate
by about +0.0334. The implemented order agrees with R.

The pinned [fixture](../tests/fixtures/r_glm_missing_2_19_6.json),
[exporter](export_glm_missing_r.R) and optional
[live checker](check_glm_missing_r.py) establish the comparison. The checker
verified the loaded library/version, saved and newly exported R values, and
Python retained rows, scores, coefficients, full covariance, joint estimate,
confidence limits, p-values, fitted means and new-data predictions. All four
cases passed. The largest absolute coefficient difference was below
`2e-13`; the largest full-covariance difference was `6.16e-8` because R's
default GLM stops before the tighter Python scoring criterion. The checker
uses exact row/score comparisons, `1e-10` tolerances for coefficients and
predictions, and `1e-7` for uncertainty fields. Case-level differences are in
the [machine record](glm-missing-reference-results.json).

The export used R 4.5.1 with qgcomp 2.19.6 in an isolated library. Use an
R executable compatible with its compiled dependencies.

These cases do not establish unbiased complete-case estimation, imputation
behavior or interval coverage under missingness.
