# Non-bootstrap pointwise comparisons

`pointwisebound_noboot` implements a bounded counterpart of R qgcomp 2.19.6
`pointwisebound.noboot` for additive Gaussian, binomial-logit and
Poisson-log fits. It accepts a non-null quantile count and a one-based
reference level. For score $s$ and reference score $r$, the conditional
linear predictor is $h(s)=\widehat\beta_0+s\widehat\psi$ and its contrast is
$(s-r)\widehat\psi$. Here $\widehat\beta_0$ is the fitted intercept and
$\widehat\psi$ sums the exposure coefficients. The contrast standard error
is $|s-r|\operatorname{SE}(\widehat\psi)$, including covariance between
exposure coefficients. Normal limits are exponentiated for binary odds and
Poisson mean ratios. All adjustment variables are set to their encoded
reference value. These intervals concern score contrasts, not the linear
predictor itself or individual predictions.

## Installed R comparison

`export_pointwise_noboot_r.R` calls the exported fit and pointwise functions
from an isolated installed qgcomp 2.19.6 library, verifies the loaded
namespace path, and saves six fixed cases: Gaussian, binomial and Poisson,
each with reference levels 1 and 3. All cases use four exposure groups and
128 rows. Reference level 3 and alpha 0.10 separately exercise nondefault
arguments. The saved file is `tests/fixtures/r_pointwise_noboot_2_19_6.json`,
SHA-256 `b5f3090b9e60df8eca29e2b49494874b4029bcbf675130b73b0fc87346d1084e`.
The optional checker re-exports from installed R, checks all inputs and R
outputs against that file, then compares every Python output column. Run:

```sh
PYTHONPATH=src python validation/check_pointwise_noboot_r.py \
  --rscript /Library/Frameworks/R.framework/Resources/bin/Rscript \
  --library ../.r-library-qgcomp-2.19.6
```

All six cases passed. The largest absolute Python/R
column difference was **1.10e-6**, within the existing `2e-6` additive GLM
reference tolerance. The largest difference is in a binary confidence
limit; Python and R agree more closely on the fitted effect and odds ratio.
This fixed-input comparison does not measure repeated-data interval coverage.

## Independent arithmetic and limits

A separate Gaussian example computes the full least-squares covariance and
contracts it with the two-exposure gradient, including their cross term.
It checks every contrast and interval at a nondefault reference. Exact binary
cell odds give a joint odds ratio of nine; exact Poisson cell means give a
joint mean ratio of four. Failure tests reject invalid reference levels,
alpha values, prequantized fits without `q`, unsupported fit types and a
Poisson observation-time offset. Checker tests detect changed inputs,
changed R outputs and a missing R executable.

The returned columns keep R's names in read-only NumPy arrays. Their
`linpred` is conditional on zero/reference adjustment variables. A binary
`or` is not a risk ratio, and a Poisson `rr` assumes equal observation
periods. The helper does not provide marginal profiles, bootstrap or
simultaneous bounds, other outcome families, or a causal interpretation.
Those require separate methods and evidence.
