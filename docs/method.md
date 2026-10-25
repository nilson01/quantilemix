# Additive models: methods and numerical conventions

This guide explains selected model calculations. For function inputs, use the
[reference manual](manual/README.md); for other methods, use the
[coverage map](r-api-parity.md).

| Topic | Start here |
| --- | --- |
| Gaussian effect, inputs and uncertainty | [Gaussian model](#gaussian-model) |
| Binary and count effects | [Binary and count models](#binary-and-count-models) |
| Bootstrap and clustered estimates | [Resampling and clustered models](#resampling-and-clustered-models) |

## Gaussian model

### Fit and limits

The Gaussian fit follows the additive, continuous-outcome case of Keil et al. (2020),
[DOI: 10.1289/EHP5838](https://doi.org/10.1289/EHP5838), Methods, “Quantile
g-Computation.” The numerical reference is CRAN **qgcomp 2.19.6**,
`qgcomp.glm.noboot(..., family=gaussian(), bayes=FALSE)` with an intercept
and complete observations. Positive case weights are supported, and omitting
them gives each row equal influence. Exact source locations
are in [reference record](reference.md).

The `fit` interface supports several time-fixed exposures, numeric
adjustment covariates, mixed coefficient signs, and model-based uncertainty
for one joint exposure contrast. Its named-data counterpart also accepts
categorical adjustment variables. This interface does not fit nonlinear terms,
interactions, clustered observations, survival outcomes, complex-survey
designs or imputation. Separate package functions cover some of these models;
see the [coverage map](r-api-parity.md). Unsupported options raise errors.

### Model and estimand

For observation \(i\), let \(Y_i\) be a continuous outcome, \(X_{ji}\) exposure
\(j\), and \(Z_i\) a vector of \(c\) adjustment variables. There are \(n\)
observations and \(d\) exposures. Following Keil et al. (2020), write
\(X_{ji}^{q}\) for the quantized score of exposure \(j\). With four
categories, its possible scores are 0, 1, 2 and 3. Ties can leave fewer
categories. The paper's additive equation omits adjustment variables; the
\(Z_i\) term below makes the package's adjusted fit explicit.

The working conditional mean is

\[
m(\mathbf x^{q},z)=E(Y_i\mid\mathbf X_i^{q}=\mathbf x^{q},Z_i=z)
=\beta_0+\sum_{j=1}^{d}\beta_j x_j^{q}+\gamma^\top z.
\]

Here \(\mathbf X_i^{q}\) holds observation \(i\)'s exposure scores,
\(\mathbf x^{q}\) is a fixed score profile, \(\beta_0\) is the intercept,
\(\beta_j\) is the coefficient of exposure score \(j\), and \(\gamma\)
contains the adjustment coefficients. Let \(\mathbf1\) be a vector of
\(d\) ones. Where both score profiles are meaningful,

\[
m(\mathbf x^{q}+\mathbf1,z)-m(\mathbf x^{q},z)
=\sum_j\beta_j(x_j^{q}+1-x_j^{q})=\sum_j\beta_j=\psi.
\]

Thus \(\psi\) is the adjusted change in the expected outcome per simultaneous
one-category increase in every exposure. It is not a one-unit increase in
each raw exposure. Averaging the displayed contrast over the same covariate
distribution leaves \(\psi\) unchanged because the covariate terms cancel.
For a common score level \(a\), the analogous contrast between all exposures
at \(a+1\) and all at \(a\) is also \(\psi\), when these levels exist.

This does not define a population policy that raises people already in the
top category. Clipping each increase at its top category produces a different
contrast: some components then contribute zero. No clipping-based policy
effect or unsupported joint profile should be labeled as \(\psi\).

The statistical interpretation requires a correctly specified additive mean
on the **score** scale. A linear mean on the raw exposure scale is a different
model. A causal interpretation additionally needs a well-defined intervention
within exposure categories, consistency, no interference, adequate joint
exposure support given covariates, and no unmeasured confounding. Quantization
and software diagnostics cannot establish these conditions. Without them,
\(\psi\) is an adjusted joint score contrast, not a health-intervention effect.

### Inputs and results

The [reference manual](manual/README.md) gives current function signatures.
The rules below describe the additive array and named-data fits covered here.

Inputs are numeric arrays with shapes `(n,)`, `(n, d)`, and optionally
`(n, c)`, in documented column order. A fit needs at least one exposure,
finite values and matching rows; it adds one intercept. Duplicate column
identifiers and masked values are rejected. Masks inside nested sequences or
returned by array conversion count as missing inputs. In the
array interface, categorical covariates must be encoded in advance with an
identified reference level.
Neither exposures nor adjustment columns are automatically selected.

The named-data interface accepts additive terms only, such as
`y ~ x1 + x2 + z + C(group)`. The exposure names are passed separately
and must appear as numeric main effects. `C(group)` converts a complete text
category into indicator columns. Sorted observed levels determine the reference
level; the first level receives no indicator. The fitted result saves those
levels and reuses them for `predict_data`. A new prediction level is an error.
The formula interface accepts mappings and table objects with named columns;
it does not require pandas.

Interactions, transformations, offsets, no-intercept
formulas and categorical exposures are not implemented. These terms are
rejected rather than treated as ordinary column names. The default
`missing="raise"` rejects incomplete rows. With `missing="drop"`, rows
missing an outcome, exposure, adjustment variable, category or case weight
are removed from fitting. The result stores original zero-based positions in
`training_row_indices` and the input size in `original_row_count`.
Automatic exposure quantiles use all observed exposure values before removing
incomplete model rows, following R qgcomp 2.19.6.

A row with a missing outcome
may therefore change a cutpoint even though it is not fitted. There is no
imputation or inverse-probability correction; complete-case analysis can be
biased when missingness depends on unobserved values.

`q` is an integer at least 2; booleans are invalid. Explicit `breaks` are one
strictly increasing finite vector per exposure, including outer endpoints.
As in R, explicit breaks take priority. `q=None, breaks=None` accepts
already quantized nonnegative integer scores and bypass quantization; this
restricted input mode is useful for exact fixtures. R also accepts general
raw numeric inputs with `q=NULL`; that wider behavior is outside this version.
The interval error level must satisfy `0 < alpha < 1`.

`weights` is an optional positive, finite case-weight vector of shape `(n,)`.
It changes how strongly each training row contributes to the Gaussian fit.
Zero, negative, missing, infinite and masked weights are rejected. R's GLM
allows some weight inputs outside this narrower contract; the Python package
does not silently drop rows or interpret these as survey-design weights.

Results contain coefficients, intercept, their covariance matrix, fitted means,
residual degrees of freedom, fitted breakpoints, actual category counts,
\(\hat\psi\), its variance and standard error, confidence limits, test
statistic, reference p-value and its degrees of freedom, positive/negative
partial sums and descriptive weights, and rank/conditioning diagnostics.
Results also store input column order, quantization mode, confidence level,
reference version and numerical-library versions.

### Case-weighted Gaussian fit

Let \(D\) be the design matrix with an intercept, exposure scores and numeric
adjustment variables. Let \(w_i>0\) be the weight for row \(i\), \(n\) the
number of rows, and \(k\) the number of fitted coefficients. The fit minimizes

\[
\sum_{i=1}^{n}w_i(Y_i-D_i\theta)^2.
\]

It uses a singular-value decomposition of the weighted design, rather than
forming a matrix inverse. If \(S\) is the weighted residual sum of squares,
the model-based dispersion is \(s^2=S/(n-k)\). The coefficient covariance is

\[
\widehat V=s^2(D^\top W D)^{-1},
\qquad W=\operatorname{diag}(w_1,\ldots,w_n).
\]

The joint estimate remains the sum of the exposure-score coefficients. Its
variance uses the full covariance: \(a^\top\widehat V a\), where \(a\) is one
at each exposure position and zero elsewhere. Normal confidence limits and
the R-reference t-test rule remain as in the unweighted fit. Quantile cuts
are computed from exposures without case weights, matching the R call order.
Multiplying every case weight by the same positive constant leaves fitted
coefficients and their model-based covariance unchanged; dispersion scales
with that constant. The result saves the case weights as a read-only array.

Case weights are different from the *sign weights* in the result. Sign weights
describe the relative sizes of positive or negative exposure coefficients.
Neither type of weight alone establishes a causal effect, a complex-survey
variance, or appropriate inference for clustered observations. The weighted
option has fixed numerical comparisons with installed R qgcomp 2.19.6, but
no repeated-dataset calibration study yet.

### Quantization contract

The reference calls R `quantile` with its default type 7, removes duplicate
values from the **full** quantile sequence, replaces the first and last
values by `-1e64` and `1e64`, and applies right-closed `cut` intervals with
`include.lowest=TRUE`, subtracting one from the resulting labels.
See [pinned source](https://github.com/alexpkeil1/qgcomp/blob/579e5093c894babaa51c9986704c8ffb6bf2825b/R/base_utility.R#L203).

For sorted values \(x_{(1)},\ldots,x_{(n)}\), type 7 at probability \(p\)
interpolates at \(h=1+(n-1)p\). If \(k=\lfloor h\rfloor\) and
\(g=h-k\), the value is \((1-g)x_{(k)}+gx_{(k+1)}\), with the endpoint
values used at \(p=0,1\). Use probabilities \(r/q\), \(r=0,\ldots,q\).
The implementation preserves R's probability grid, one-based arithmetic order
and shortcut for equal adjacent values. Floating-point differences at exact boundaries can
otherwise change a score, even when two interpolation formulas are algebraically
equivalent.

An interior cut belongs to the lower category. For example, with observations
`[0, 0, 0, 1]` and `q=4`, the full quantiles are `[0, 0, 0, .25, 1]`.
After deduplication and endpoint replacement the breaks are
`[-1e64, .25, 1e64]`, so scores are `[0, 0, 0, 1]`. Deduplicating only the
interior quantiles instead gives a different result for this example.

Automatic breaks require finite exposure magnitudes strictly below `1e64`.
Constant exposures and degenerate break vectors raise an error. Results report
reduced categories and empty bins; ties are not broken at random.
Explicit-break inputs outside the supplied range raise an error.
Automatic endpoints allow values beyond the
training minimum/maximum to enter the extreme bins, but prediction must flag
such extrapolation. Prediction always reuses stored training breaks.

### Estimation and numerical conventions

Let \(D=[\mathbf1,X^{q},Z]\) be the design matrix with \(k=1+d+c\) columns and
\(\theta=(\beta_0,\beta_1,\ldots,\beta_d,\gamma^\top)^\top\). Fit ordinary
least squares by a stable QR or singular-value decomposition. The formulas
below define the result; explicitly inverting normal equations is unnecessary.

\[
\hat\theta=(D^\top D)^{-1}D^\top Y,\qquad
\hat\sigma^2=\frac{\|Y-D\hat\theta\|^2}{n-k},\qquad
\widehat V=\hat\sigma^2(D^\top D)^{-1}.
\]

The fit requires full column rank and \(n>k\). The rank tolerance is
`max(n,k) * machine_epsilon * largest_singular_value`. Results report the
rank tolerance and design condition number. A condition number above `1e8`
raises a numerical warning; this is not a universal statistical cutoff.
Rank failure raises an error; the fit does not switch to a minimum-norm or
regularized solution. Rank-deficient but estimable contrasts remain open.

Let \(a\) be zero for the intercept and adjustment coefficients and one for
each exposure coefficient. Then

\[
\hat\psi=a^\top\hat\theta,\qquad
\widehat{\operatorname{Var}}(\hat\psi)=a^\top\widehat V a
=\sum_j\widehat V_{jj}+2\sum_{j<\ell}\widehat V_{j\ell},
\qquad \operatorname{SE}(\hat\psi)=\sqrt{a^\top\widehat V a},
\]

where the last sums run only over exposure coefficients. The off-diagonal
covariances are essential. Standard errors cannot be added or combined using
only their squares when exposures are correlated.

For reference agreement, the interval is
\(\hat\psi\pm z_{1-\alpha/2}\operatorname{SE}(\hat\psi)\), where
\(z_u\) is the standard-normal quantile. In the pinned R source, the Gaussian
p-value uses a Student t distribution with `df.null - d`, which here is
\(n-1-d\), **not** the residual degrees of freedom \(n-1-d-c\).
The two-sided tail is calculated as
\(2F_{t_\nu}(-|\hat\psi/\operatorname{SE}|)\).
The result labels this `qgcomp_2.19.6` inference. The interval and test need
not give the same decision at a given alpha. The normal interval is not an
exact finite-sample Gaussian interval. A conventional t interval with
\(n-k\) degrees of freedom would be a different comparator.

The fit uses double precision. A materially negative variance or nonfinite fit fails.
A negative variance within an explicitly recorded floating-point roundoff
bound may be set to zero with a diagnostic. A zero residual variance permits
point prediction but marks inference degenerate; do not turn division by zero
into a reassuring significance claim. Independent observations and constant
conditional residual variance underlie the model-based covariance. Robust,
clustered, and survey uncertainty need separate support.

The solve first subtracts the outcome mean, then restores it to the intercept.
Residuals are computed before this restoration, so a large outcome offset does
not hide small, representable residual differences. The numerical-zero check
uses the centered outcome and the absolute sizes of individual fitted terms.
This retains a check for cancellation while avoiding a tolerance that changes
solely because a covariate uses smaller units. Centering cannot recover
differences already lost when the input was stored in double precision.

### Direction-specific weights and prediction

Define \(P=\{j:\hat\beta_j>0\}\), \(N=\{j:\hat\beta_j<0\}\),
\(\hat\psi_+=\sum_{P}\hat\beta_j\), and
\(\hat\psi_-=\sum_{N}\hat\beta_j\). Within a nonempty group,

\[
w_j^+=\hat\beta_j/\hat\psi_+,\qquad
w_j^-=|\hat\beta_j|/|\hat\psi_-|.
\]

Each group's weights sum to one; \(\hat\psi=\hat\psi_++\hat\psi_-\).
An absent sign group returns an empty mapping, not a division by zero. Exact
zero coefficients are listed separately. R puts zero coefficients in its
nonpositive group; this deliberate presentation difference must be tested
and documented. Near-zero coefficients are not rounded into another group
before calculation. Weights are descriptive fitted proportions, with no
selection probability, causal-importance ranking, or confidence interval.
Dividing all coefficients by \(\hat\psi\) is unsuitable for canceling effects.

`predict` returns \(\hat\beta_0+X_{new}^{q}\hat\beta+Z_{new}\hat\gamma\)
using the same columns and stored cuts. It predicts a conditional mean, not
an individual-outcome interval or a causal policy value. `transform` returns
scores and diagnostics without fitting or updating cuts.
In prequantized mode both methods accept scores directly and do not invent
new breaks. Reference checks compare these predictions with the underlying
R GLM because the R wrapper raises an error for the documented prequantized
new-data case.

### Input errors and interpretation limits

| Case | Package behavior or limit |
| --- | --- |
| Missing outcome, exposure, covariate or weight | Rejected by default. `fit_formula(..., missing="drop")` removes incomplete training rows and records their positions; it does not impute values. |
| Infinite outcome, exposure, covariate or weight | Rejected; infinity is not a missing value. |
| Constant exposure or exact collinearity after quantization | The fit raises a rank or grouping error. It does not silently drop terms or add a penalty. |
| Highly correlated exposures | The result reports the condition number and full covariance. Correlation alone does not make every joint contrast unstable. |
| Too few residual degrees of freedom | Model-based inference is unavailable. |
| Mixed signs or \(\hat\psi\) near zero | The joint estimate and sign-specific summaries remain available; sign weights are not divided by \(\hat\psi\). |
| New prediction order, missing covariate or wrong shape | Rejected rather than aligned by guesswork. |
| Nonlinear mean, unequal residual variance, repeated subjects or unmeasured confounding | The model-based interval or causal interpretation may fail. Diagnostics do not establish these assumptions. |

Source agreement, statistical calibration, and software usefulness are separate
requirements in [validation record](../validation/RESULTS.md).

### Numerical and input limits

Automatic q is limited to 100,000 to reject oversized grids before allocation.
Explicit breaks take priority. Joint variance is calculated from the fitted
SVD factors to avoid cancellation when contracting a large covariance matrix;
the literal covariance contraction is retained as a numerical diagnostic.
These implementation choices and input restrictions are tested and are not
claims of full R API parity.

## Binary and count models

### Additive binary-outcome subset

`fit_binomial` accepts a Bernoulli outcome \(Y_i\in\{0,1\}\), where 1 is the
recorded event. It uses the same exposure scoring rules as `quantize`. With
exposure scores \(X_{ji}^{q}\) and adjustment variables \(Z_i\), the working model is

\[
\operatorname{logit}\Pr(Y_i=1\mid\mathbf X_i^{q},Z_i)
=\beta_0+\sum_{j=1}^{d}\beta_jX_{ji}^{q}+\gamma^\top Z_i.
\]

The joint estimate is \(\hat\psi=\sum_j\hat\beta_j\). It is the **conditional
log odds ratio** for increasing every exposure score by one while holding
adjustment values fixed. `result.odds_ratio` is \(\exp(\hat\psi)\). Neither
quantity is a risk difference, and an odds ratio is generally not a risk ratio.
The one-category contrast still needs a supported joint exposure profile for
a causal interpretation.

For positive case weights \(w_i\), the fit maximizes the weighted Bernoulli
log likelihood \(\sum_i w_i\{y_i\log p_i+(1-y_i)\log(1-p_i)\}\), where
\(p_i\) is the fitted event probability. The model-based information is
\(D^\top\operatorname{diag}\{w_i p_i(1-p_i)\}D\), where \(D\) is the design
matrix. Integer weights have the same coefficient and information calculations
as repeating observations. These are GLM prior weights, not survey or cluster
variance corrections. Exposure cutpoints remain unweighted.

The fit maximizes the Bernoulli log likelihood with Fisher scoring and a
backtracking step when needed. It rejects constant outcomes, rank-deficient
designs, nonconvergence and numerical separation. It saves the inverse final
Fisher-information matrix as the model-based coefficient covariance. For
contrast vector \(a=(0,1,\ldots,1,0,\ldots,0)^\top\), the joint variance is
\(a^\top\widehat\Sigma a\). The interval is
\(\hat\psi\pm z_{1-\alpha/2}\widehat{\mathrm{SE}}\), and the two-sided
p-value uses a normal z statistic, matching the binomial branch of R qgcomp
2.19.6. This differs from the Gaussian branch's Student t p-value. Predicted
values are conditional event probabilities, calculated with the saved exposure
cuts. The separate `BinomialResult.summary()` labels each scale.

`fit_binomial_formula` accepts the restricted additive named-data syntax and
`C(name)` text adjustment categories. Its explicit `missing="drop"` option
removes incomplete training rows after exposure cutpoints are estimated from
all observed exposure values; it records the retained original positions.
The array entry point requires complete inputs. Both accept only individual
0/1 outcomes and an intercept; grouped trials, offsets,
interactions, alternative links and marginal risk contrasts remain outside
this subset. Neither imputes values. These limits are checked rather than
silently ignored.

An independent four-cell construction has event proportions 0.25, 0.5, 0.5
and 0.75. Its cell odds are 1/3, 1, 1 and 3. Thus each exposure log-odds
coefficient is \(\log 3\), the joint estimate is \(\log 9\), and the joint
odds ratio is 9. The implementation checks these values and independently
computes \((X^\top W X)^{-1}\) from the fitted cell probabilities.

Two fixed software cases also agree with isolated installed R qgcomp 2.19.6.
R's default `glm.fit` stops when its deviance change is small, before its
reported information matrix fully stabilizes. Python iterates to a tighter
coefficient and score criterion. In the fixed cases, coefficients and
predictions differ by less than `5e-10`, while the largest joint-variance
difference is about `9.2e-7`. The optional checker allows `2e-6` absolute
and relative tolerance for uncertainty fields and records each measured
difference. This is a numerical convention limit, not evidence that one
method has better repeated-sample interval coverage.

### Additive count-outcome subset

`fit_poisson` accepts nonnegative integer counts. Equal observation periods
need no offset; for unequal periods, supply a positive observation time
\(t_i\) for each row. With the exposure scores and adjustment variables defined
above, its conditional mean model is

\[
\log E(Y_i\mid\mathbf X_i^{q},Z_i,t_i)
=\log t_i+\beta_0+\sum_{j=1}^{d}\beta_jX_{ji}^{q}+\gamma^\top Z_i.
\]

When no observation time is supplied, \(t_i=1\) for every row.

The joint estimate is \(\hat\psi=\sum_j\hat\beta_j\). Holding adjustment
variables and observation time fixed, \(\exp(\hat\psi)\) is the ratio of fitted
mean counts when every exposure score increases by one. With a positive
`exposure_time` offset for unequal observation periods, it is also the fitted
rate ratio. This is a conditional contrast, not a marginal effect or an
absolute count difference.

Positive case weights \(w_i\) multiply each Poisson log-likelihood term.
The model-based information is
\(D^\top\operatorname{diag}(w_i\mu_i)D\), with fitted mean \(\mu_i\).
Integer weights agree with repeating observations; exposure cutpoints
remain unweighted. These weights do not correct survey sampling, clustering or
overdispersion. Use `exposure_time` to account for unequal observation periods.

The fit maximizes the Poisson likelihood with Fisher scoring and a
backtracking step when needed. It uses the inverse final Fisher information
for model-based coefficient covariance. The joint variance is
\(a^\top\widehat\Sigma a\), where \(a\) selects and sums all exposure
coefficients. The confidence interval and p-value use normal critical values
and the normal z statistic, following the Poisson branch of R qgcomp 2.19.6.
Predictions are conditional expected counts using the saved exposure cuts.
`fit_poisson_formula` accepts the restricted additive named-data syntax and
text categorical adjustment. Its explicit `missing="drop"` option removes
incomplete training rows after exposure cutpoints are estimated from all
observed exposure values; it records the retained original positions. The
array entry point requires complete data. Both require full rank, a positive
total count and a finite interior fit. They accept a positive observation-time
offset but reject other offsets, interactions and alternative links. Neither
imputes values.

An independent four-cell construction has fitted means 1, 2, 3 and 6. The
exposure coefficients are \(\log 2\) and \(\log 3\), so the joint mean
ratio is exactly 6. A separate calculation of
\((X^\top\operatorname{diag}(\hat\mu)X)^{-1}\) checks the covariance,
where \(X\) is the model matrix and \(\hat\mu\) its fitted means.
Two fixed cases agree with isolated installed R qgcomp 2.19.6. The largest
observed joint-variance difference is about \(1.4\times10^{-7}\), caused
by the default R GLM stopping sooner than the tighter Python solver.
These fixed cases establish software agreement within the stated tolerance;
they do not establish interval calibration across repeated count datasets.

## Resampling and clustered models

### Fixed-cut Gaussian row bootstrap

`fit_bootstrap` estimates an additive Gaussian contrast. It accepts
a complete continuous outcome, exposure columns and optional numeric adjustment
columns, with independent rows and an intercept. Positive case weights remain
attached to sampled rows. It does not accept clusters, nonlinear conditional
terms, interactions or a polynomial joint response. Separate functions cover
some of these settings.

For intervention level \(a\), set every exposure score to \(a\), predict
each row's outcome with its observed adjustment values, and average. Let
\(w_i\) be a positive case weight, or 1 when no weights are supplied:

\[
\widehat\mu(a)=\frac{\sum_{i=1}^{n}w_i
\widehat E(Y_i\mid X_{1i}^{q}=\cdots=X_{di}^{q}=a,Z_i)}
{\sum_{i=1}^{n}w_i}.
\]

Here \(n\) is the number of observations, \(d\) is the number of exposures,
\(X^{q}\) contains their ordered scores, and \(Z_i\) contains row \(i\)'s
adjustment values. A linear marginal model of \(\widehat\mu(a)\) against
\(a\) gives a joint mean-difference slope. For the supported additive
Gaussian conditional model, that slope equals the sum of exposure
coefficients exactly in the model equation. The result saves the intervention
levels and corresponding standardized means so the distinction is visible.

The package determines exposure cutpoints once using the full training data.
For each of \(B\) replicates it samples \(n\) rows with replacement, reuses
those cutpoints, refits the conditional Gaussian model, and records the joint
slope. A failed or rank-deficient replicate raises an error that names its
position; it is not silently replaced. The standard error is the sample
standard deviation of these \(B\) effects, with divisor \(B-1\). The
interval is a normal Wald interval, not a percentile interval. Following
R qgcomp 2.19.6, the two-sided p-value uses a Student t reference with
\(n-d-c-2\) degrees of freedom, where \(c\) is the number of numeric
adjustment columns. An exactly zero bootstrap standard error makes the
test statistic and p-value undefined in Python.

`seed` creates a local NumPy random generator and does not change global
random state. It repeats draws within the same numerical environment; NumPy
and R do not turn the same integer seed into the same row schedule. The
optional R comparison therefore exports the exact rows drawn by R and runs
Python with that saved schedule. Two fixed cases agree at floating-point
precision on individual bootstrap slopes, standard errors, intervals and
intervention means. This checks software agreement, not interval coverage
across repeated datasets. Prequantized input is supported only when the pooled
scores have 2 through 9 observed levels; supplied break vectors must have
equal category counts across exposures. These restrictions keep the joint
intervention grid well defined.

### Restricted quadratic Gaussian joint response

`fit_quadratic_bootstrap` is a separate method for complete, independent
Gaussian rows. Let \(X_{ji}^{q}\) be exposure score \(j\) for row \(i\), and let
\(Z_i\) be its numeric adjustment variables. Its conditional model is

\[
E(Y_i\mid\mathbf X_i^{q},Z_i)=\beta_0+\sum_j\beta_j X_{ji}^{q}
                  +\sum_j\delta_j (X_{ji}^{q})^2
                  +\sum_{j<k}\eta_{jk}X_{ji}^{q}X_{ki}^{q}+\gamma^\top Z_i.
\]

The pairwise product terms are included only when
`pairwise_interactions=True`. With the default `False`, every \(\eta_{jk}\)
term is absent.

At joint intervention level \(a\), set every exposure score to \(a\), keep
each row's adjustment values, predict its mean, and average over rows. The
resulting degree-two marginal curve is
\(\mu(a)=\theta_0+\theta_1a+\theta_2a^2\). In this restricted model,
\(\theta_1=\sum_j\beta_j\) and
\(\theta_2=\sum_j\delta_j+\sum_{j<k}\eta_{jk}\). The mean
change from joint level \(a\) to \(a+1\) is
\(\theta_1+(2a+1)\theta_2\); `result.effect_at(a)` calculates it.
The intercept also contains the average adjustment contribution. This
identity is specific to these squared and optional pairwise terms.

Exposure cuts are estimated once, then held fixed. Each of `B` bootstrap
replicates resamples independent rows and refits the conditional model.
The sample covariance of the two saved bootstrap coefficients estimates
uncertainty. Each coefficient gets a normal Wald interval and a two-sided
Student t p-value with \(n-2d-h-c-3\) degrees of freedom, matching the
restricted R qgcomp 2.19.6 degree-two branch; \(n\) is the number of rows,
\(d\) the number of exposures, and \(c\) the number of numeric adjustment
columns. Here \(h=d(d-1)/2\) when pairwise products are included, and zero
otherwise. The result saves the full two-by-two bootstrap covariance, so a
level-specific contrast can use its cross term. `result.summary()` labels
both coefficients and the changing contrast.

The input needs at least three consecutive intervention levels starting at
zero, a full-rank conditional design and positive reference degrees of
freedom. Pairwise products require at least two exposures. The method accepts
no arbitrary formula, case weights or clusters. The equations above describe
the square-only and pairwise models. Separate
[exposure-covariate interaction checks](../validation/COVARIATE_INTERACTIONS_RESULTS.md)
cover the `covariate_interactions=True` option.
Two square-only and two pairwise cases agree with isolated installed R qgcomp
2.19.6 on conditional fit, marginal curve,
shared bootstrap coefficients and inference within `1e-10`. R stores
synthetic cutpoint boundaries for prequantized bootstrap input; Python keeps
its established `breaks=None` contract. These cases establish software
agreement, not repeated-sample interval coverage or causal identification.

### Clustered Gaussian estimating equations

`fit_gaussian_ee` fits an additive Gaussian conditional model, then estimates
the marginal mean at each common joint exposure score. Let \(X_i\) be the
conditional design row, \(\beta\) its coefficients, \(X_i(a)\) the row after
all exposure scores are set to \(a\), and \(m(a)=(1,a)^\top\). The conditional
and marginal estimating equations are

\[
\sum_i X_i\{Y_i-X_i^\top\beta\}=0,\qquad
\sum_i\sum_a m(a)\{X_i(a)^\top\beta-m(a)^\top\theta\}=0.
\]

Here \(\theta=(\theta_0,\theta_1)^\top\). Under the stated additive model,
\(\theta_1=\sum_j\beta_j\) is the fitted mean difference when all exposure
scores rise by one category. The marginal intercept averages the observed
adjustment contribution. To account for repeated observations, first sum
both estimating-function components within each cluster. The robust
covariance is \(A^{-1}BA^{-\top}\), where \(A\) is the derivative of the
stacked equations and \(B\) sums the outer products of cluster scores.
There is no small-cluster correction. R qgcomp 2.19.6 computes \(A\)
numerically; Python uses exact derivatives. Normal Wald limits and a
two-sided Student t test with \(n-d-c-2\) reference degrees of freedom follow
the pinned R convention, where \(n\) counts rows, \(d\) exposures and \(c\)
adjustment columns. This row-based reference rule is not a claim that the
Student t approximation is well calibrated with few clusters.

The Python method requires complete numeric rows, exact integer cluster IDs,
at least two clusters, consecutive joint levels starting at zero, full rank
and positive reference degrees of freedom. It accepts positive case weights.
Categorical adjustment, nonlinear terms and time structure are outside this
function; a separate degree-two Gaussian function covers selected nonlinear
terms.
Two fixed 32-cluster cases agree with isolated installed R on the conditional,
marginal and full robust covariance and effect inference. An independent
conditional cluster-sandwich calculation checks the additive contrast. Fixed
agreement does not establish repeated-sample coverage or causal validity.
