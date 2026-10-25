# Numerical reference and reproducibility

This guide explains how selected Python results were compared with installed
R qgcomp 2.19.6. Each check uses fixed inputs and a saved report. These checks
measure numerical agreement for those inputs; the [validation reports](../validation/RESULTS.md)
give separate evidence about interval behavior across repeated datasets.
Only these comparison checks need R; Python fitting and prediction do not.

The method is described by [Keil et al. (2020)](https://doi.org/10.1289/EHP5838).
The numerical reference is Alexander Keil's GPL-licensed R qgcomp 2.19.6 at
[source commit `579e509`](https://github.com/alexpkeil1/qgcomp/tree/579e5093c894babaa51c9986704c8ffb6bf2825b).

| Topic | Checks |
| --- | --- |
| Gaussian | [Fixtures](#gaussian-source-fixtures), [formulas](#gaussian-formula-and-category-check), [installed-R fit](#installed-r-gaussian-check), [weights](#weighted-gaussian-check), [missing rows](#complete-case-formula-check) |
| Binary and count | [Logit](#bernoulli-logit-check), [Poisson](#poisson-log-check), [offsets](#poisson-exposure-time-offset-check), [missing rows](#binarycount-complete-case-check), [weights](#weighted-binarycount-check) |
| Resampling and clusters | [Gaussian bootstrap](#fixed-cut-gaussian-bootstrap-check), [quadratic fit](#quadratic-gaussian-bootstrap-check), [interactions](#gaussian-pairwise-interaction-check), [clustered fit](#clustered-gaussian-estimating-equation-check), [marginal odds](#marginal-binary-odds-ratio-check) |

## Gaussian checks

### Gaussian source fixtures

The fixture generator executes unmodified `R/base.R` and `R/base_utility.R`
from that source, with their hashes checked before sourcing. SHA-256:

| File | SHA-256 |
| --- | --- |
| R/base.R | `14fcbbeb3192310645069ad2c85326cd3f1c1f4f9c74ec5593b7f026b7a8529d` |
| R/base_utility.R | `6592524e48fdcb45740091abbe5109de9e901cd87fa33dbed23ea31aeb984581` |

The generator checks the corresponding MD5 values using base R, avoiding an
extra hashing dependency. SHA-256 values above were checked independently.
This is source-function agreement. It does not establish full installed-package
compatibility, all R dependencies, or generic-method dispatch behavior.

To regenerate fixtures, obtain the pinned upstream source in a separate local
directory and provide that package root as the first argument. R and `jsonlite`
are needed only for regeneration, never for native Python use:

```sh
Rscript --vanilla validation/generate_r_fixtures.R /path/to/pinned/qgcomp tests/fixtures/r_gaussian_2_19_6.json
python -m pytest tests/test_r_reference.py
```

The nine cases use fixed literal/arithmetic inputs, not a Monte Carlo study.
The JSON records R and jsonlite versions, source identities, input arrays and
numeric expected output. Tests check its identity and complete case set.
For predictions, the oracle is the fitted R GLM applied to inputs transformed
with the fitted cuts. The boundary case includes new observations and exact cuts.
Review fixture differences before committing; do not regenerate expected output
merely to make a failing test pass.

The `quantile_rounding_boundaries` case uses values 0 through 49 with 49 groups.
It checks R's exact category assignments when floating-point rounding places
cuts on, or just below, observed values. The implementation follows R 4.5.0's
type-7 one-based interpolation order and skips interpolation for equal adjacent
values. An algebraically equivalent interpolation order can change a category.
Numeric cutpoints in JSON are compared within tolerance; integer scores are
compared exactly. The original eight cases were unchanged when this case was added.

DeepMix is acknowledged as prior partial Python implementation. It is not our
numerical oracle, and no DeepMix code was copied. See [NOTICE.md](../NOTICE.md).

### Gaussian formula and category check

The restricted Python formula/categorical interface has two additional fixed
comparisons from isolated installed R qgcomp 2.19.6. The exporter
`validation/export_formula_r.R` fits one automatically quantized case and
one prequantized weighted case with a three-level adjustment factor. It stores
the inputs and R numerical values in
`tests/fixtures/r_formula_gaussian_2_19_6.json`. Python encodes
`C(g)` using the sorted first text level as reference. R's model matrix
calls the indicator columns `gB` and `gC`; Python calls them
`g[T.B]` and `g[T.C]`. Their column order and numerical values agree.
`validation/check_formula_r.py` reruns both cases against the
installed package and checks the saved fixture and Python result. The Python
runtime does not call R.

Run the check with an isolated R library prepared below:

```sh
python validation/check_formula_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The Python formula grammar is narrower than R's. Both cases have complete
rows; they do not test missing rows, other factor contrasts, interactions or
transformed terms. Fixed-case agreement does not establish interval coverage
across repeated datasets with categorical adjustment.

### Installed-R Gaussian check

`validation/check_installed_r.py` compares those same nine fixed inputs with an
installed qgcomp **2.19.6**. Its R exporter calls `qgcomp::qgcomp.glm.noboot`,
`qgcomp::quantize` and the registered `predict.qgcompfit` method. It does not
source R implementation files or regenerate the stored fixtures. Neither this
check nor its installer dependencies are required by Python package users.

### Prepare a separate R library

Use an existing R installation and a new library directory outside this checkout.
Obtain `qgcomp_2.19.6.tar.gz` from [CRAN's source directory](https://cran.r-project.org/src/contrib/)
or its [qgcomp archive](https://cran.r-project.org/src/contrib/Archive/qgcomp/).
Verify the archive SHA-256 before installation:

```text
4cd9cd8f90aac9220bfae8e8dc4fbebef852515d07902fc28832ffc58026dfd4
```

For example, `shasum -a 256 qgcomp_2.19.6.tar.gz` on macOS or
`sha256sum qgcomp_2.19.6.tar.gz` on Linux prints the digest. Stop if it differs.
The pinned DESCRIPTION requires R >= 3.5.0. Its mandatory imports are AER, arm,
future, future.apply, generics, ggplot2 >= 3.3.0, grDevices, grid, gridExtra,
nnet, numDeriv, pscl, rootSolve, stats, survival and tibble. The exporter also
requires jsonlite. Dependencies of those packages may require a newer R.

In a fresh `R --vanilla` session, adapt these two paths:

```r
lib <- "/path/to/new/r-library-qgcomp-2.19.6"
archive <- "/path/to/qgcomp_2.19.6.tar.gz"
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
lib <- normalizePath(lib)
.libPaths(c(lib, .Library), include.site = FALSE)
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 120)
install.packages(c("AER", "arm", "future", "future.apply", "generics",
                   "ggplot2", "gridExtra", "numDeriv", "pscl", "rootSolve",
                   "tibble", "jsonlite"), lib = lib, dependencies = NA)
install.packages(archive, lib = lib, repos = NULL, type = "source")
stopifnot(as.character(packageVersion("qgcomp", lib.loc = lib)) == "2.19.6")
```

Base and recommended packages, including nnet and survival, can come from the
chosen R installation. Every other loaded package must be in `lib`. If R finds
a third-party dependency in its system library, install that dependency into
`lib` explicitly; the exporter rejects such fallback. `dependencies = NA`
requests mandatory dependencies, not the full Suggested set. Some Suggested
packages may still be required through another package's mandatory dependencies.

Use binary dependencies where available for the selected R version and
architecture. A source-only platform may need C/C++/Fortran build tools for
dependencies; qgcomp itself has `NeedsCompilation: no`. See the official
[R installation manual](https://cran.r-project.org/doc/manuals/r-release/R-admin.html)
and [macOS binaries](https://cran.r-project.org/bin/macosx/). An installation
failure is a blocker with the package name, version and build error, not a
passed comparison. Do not modify the system or ordinary user R library.

### Run the comparison

With the Python development environment installed, run from this checkout:

```sh
python validation/check_installed_r.py \
  --rscript /path/to/Rscript \
  --r-library /path/to/new/r-library-qgcomp-2.19.6 \
  --output /path/to/new/installed-r-comparison.json
```

Choose a new output filename on each run. The checker refuses existing files,
checks the exact fixture digest, runs R with `--vanilla`, excludes site/user
library fallback, and audits loaded namespace locations. It never installs
packages. Exit 0 means the stated comparisons passed; nonzero means failed
comparison, invalid evidence, unavailable R, timeout or report-write failure.
The default timeout is 60 seconds and can be changed with `--timeout`.

The checker also calls the registered `summary.qgcompfit` method separately
from the underlying GLM summary. It retains the returned matrix, row and
column names, printed header, warnings and errors. The common numeric columns
are compared with both the fixed reference and the current native fit.

In the saved comparison, installed qgcomp 2.19.6 on R 4.5.1 returned five columns
for every one of the nine Gaussian cases: estimate, standard error, lower and
upper confidence limits, and p-value. The t-statistic column was missing.
The source explains this: `summary.qgcompfit` reads `object$zstat`, while the
Gaussian fit stores `tstat`; `print.qgcompfit` explicitly copies that field.
All returned numeric values agreed within the existing tolerances. The checker
records the omitted column as a known reference limitation. Any other column
change, warning, error or numeric disagreement fails the comparison.

Python's `result.summary()` displays the already-verified `test_statistic`.
It returns a string with sign weights and diagnostic explanations, rather
than R's list containing a coefficient matrix. Its full-model coefficient
table shows estimates only. R's intercept inference and summary structure
are not claimed as implemented. Python also preserves stable small p-values:
two fixed R cases round to zero, and another loses tail precision through
subtraction. Display tests check those tails separately against a Student t
survival calculation using the pinned R statistic and degrees of freedom.

Every installed fit is compared separately with the stored fixture and the
current native Python result. Checks cover scores, cuts, coefficients, full
covariance, joint effect and uncertainty, degrees of freedom, sign summaries,
fitted means and saved-cut predictions. Scores and degrees of freedom must
match exactly. Cut tolerances are absolute/relative `1e-12`; other numerical
tolerances are absolute `1e-10` and relative `1e-8`. Array shapes and named
weights must match. Ordered coefficient and exposure names must agree with the
fixed inputs and Python's fitted result. Malformed or nonfinite evidence fails.

### Executed result and differences

The [saved report](../validation/installed-r-results.json) records the executed
comparison, file hashes, tolerances and all dependency versions. It records
MD5 fingerprints of the installed qgcomp DESCRIPTION, NAMESPACE and lazy-load
code files, plus the selected R runtime file. These identify local artifacts;
they do not authenticate the installation. The separately verified source
archive SHA-256 above records the installation source. All nine
fitted models and underlying GLM predictions agreed with both the stored
fixtures and Python. Registered R prediction also agreed for eight cases.
For `prequantized`, where `q=NULL` and no cuts are stored, R's wrapper raised
`'by' must be of length 1`. Only that exact case and error are recorded as a
known reference limitation. Other prediction errors fail the check. Python
continues to predict from the supplied scores; it does not reproduce this R
wrapper defect. This limitation is separate from the successful underlying
GLM comparison for the same case.

The installed run used CRAN framework R 4.5.1 on Apple silicon; the original
source-function fixtures used R 4.5.0. The isolated library held 75 packages,
including qgcomp and jsonlite. Of 60 loaded namespaces, 42 came from that
library and 18 from R's base/recommended library; none came from elsewhere.
Official R 4.5 ARM binaries supplied third-party dependencies, with compatible
existing packages copied into the isolated library; the pinned qgcomp archive
was installed from source. The default Homebrew R was not used for this run.
Dependency versions are recorded rather than fully locked, so a new installation
may resolve newer dependencies and must produce its own report.

This establishes agreement for these fixed Gaussian inputs and options.
It does not establish every R method, platform, package version, causal
interpretation or repeated-sample uncertainty calibration. Ordinary Python
CI tests the checker with synthetic exporter responses and does not install R.

### Weighted Gaussian check

The positive-case-weight check uses a separate pair of fixed inputs. They
exercise raw measurements with numeric adjustment and correlated exposure
scores that are already grouped. R qgcomp 2.19.6 passes `weights` to the
underlying Gaussian GLM. It computes exposure cutpoints before fitting, so
case weights do not change those cuts. The [fixed weighted fixture](../tests/fixtures/r_weighted_gaussian_2_19_6.json)
records the installed-R results, exporter hash and source identity. It does not
replace or edit the nine unweighted cases.

With the isolated R library prepared above, run:

```sh
python validation/check_weighted_r.py \
  --rscript /path/to/Rscript \
  --r-library /path/to/new/r-library-qgcomp-2.19.6 \
  --output /path/to/new/weighted-comparison.json
```

The output path must be new. The checker confirms the fixture and exporter
identities, calls the installed R package, and compares both R's current
values with the saved fixture and the Python fit. It checks coefficients,
full covariance, cutpoints, scores, joint inference, fitted means, predictions
and sign summaries. Exact scores and degrees of freedom must match; other
numbers use absolute and relative tolerances of `5e-12`. The Python package
itself needs no R installation. Missing R, changed inputs, malformed output or
a failed comparison produces a nonzero exit without a pass report.

Both fixed cases passed using isolated R 4.5.1 and
qgcomp 2.19.6. The [saved comparison](../validation/weighted-reference-results.json)
records the checked fixture and exporter hashes. The positive-only Python
contract is narrower than every weight value that R's GLM may accept. These
fixed comparisons show numerical agreement for the stated cases; they do not
validate survey designs, clustered data, causal weighting, or repeated-data
interval coverage.

### Complete-case formula check

The explicit `fit_formula(..., missing="drop")` option has two separate fixed
comparisons with installed R qgcomp 2.19.6. The first omits one row each for a
missing outcome, numeric adjustment value and exposure. The second omits a
missing text category and a missing positive case weight. R computes exposure
quantiles before its Gaussian GLM removes incomplete rows. The Python option
uses that order. The returned `training_row_indices` are zero-based positions
in the original data, and fitted values contain retained rows only.

The [fixed fixture](../tests/fixtures/r_missing_gaussian_2_19_6.json) records
the inputs, row positions and installed-R values. The [saved report](../validation/missing-reference-results.json)
records the completed comparison and pinned fixture and exporter hashes.
To rerun it against an isolated installation, use:

```sh
python validation/check_missing_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker refuses an existing output, checks fixture and exporter identities,
and compares current installed R, saved R and Python values. It checks retained
rows, exposure scores and cuts, coefficients and full covariance, joint effect
and uncertainty, fitted means and new-data predictions. Both cases passed with
isolated R 4.5.1/qgcomp 2.19.6. Ordinary package use and
tests have no R dependency. This is complete-case fitting, not imputation or a
claim that complete-case estimates are unbiased under every missingness process.

## Binary and count checks

### Bernoulli-logit check

The binary check uses `fit_binomial` and `fit_binomial_formula` for
complete 0/1 outcomes. The joint coefficient is on the conditional log-odds
scale; predictions are event probabilities. Two fixed cases come from isolated
installed R qgcomp 2.19.6. One has four cells with exact odds ratio 9 and
already grouped exposure scores. The other uses four quantile groups and a
numeric adjustment variable. The [pinned fixture](../tests/fixtures/r_binomial_2_19_6.json)
records R's full-model coefficients and covariance, scores, cuts, joint
inference, fitted values and underlying-GLM predictions.

Run the installed-package check with the isolated R library:

```sh
python validation/check_binomial_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker confirms fixture and exporter identities, refuses an existing
output, verifies the installed qgcomp namespace, and compares current R with
both the saved fixture and Python. It uses exact scores and residual degrees
of freedom. Python coefficients and probabilities use `5e-10` absolute and
relative tolerances; uncertainty fields use `2e-6`. The
[saved report](../validation/binomial-reference-results.json) records both
passing cases and the observed maximum absolute difference for each field.

R qgcomp 2.19.6 stores a **normal z p-value** for binomial fits. Its Gaussian
branch instead stores a Student t p-value. The binary Python result follows
the binomial rule. In the fixed exact-cell case, R's default GLM stops after
four scoring iterations; the Python solver takes five. The coefficients and
predictions agree within `5e-10`, but R's reported covariance is based on its
earlier stopping point. The largest joint-variance difference is
`9.18751461354006e-7`; the p-value differs by `2.237894946048244e-8`.
Python uses the final Fisher information. The independent exact-cell test
checks that calculation against the inverse of the weighted design cross-product.

These comparisons establish the stated additive logit software behavior. They
do not establish marginal risk effects, behavior under separation,
repeated-sample interval coverage, or full R-package support.

### Separated risk-ratio draw check

The two fixed marginal risk-ratio cases in
[RISK_RATIO_RESULTS.md](../validation/RISK_RATIO_RESULTS.md) use nonseparated
data. A later 4,000-dataset study retained one bootstrap draw separated by
its binary adjustment variable. The [exact-draw report](../validation/RISK_RATIO_SEPARATION_R_RESULTS.md)
compares that one input with installed R qgcomp 2.19.6.

Use the R 4.5.1 executable that matches the isolated library's build version:

```sh
python validation/check_risk_ratio_separation_r.py \
  --rscript /path/to/R-4.5.1/Rscript \
  --library /path/to/isolated-r-library \
  --output /path/to/new-result.json
```

The checker regenerates the exact draw from the saved study seed. It verifies
the original input hash, separation counts, current Python error, isolated R
package origin and R result. The installed package's internal bootstrap-fit
step and an exported four-resample call both return a finite stopping value.
The exported call uses new R resamples and does not replay the study's full
200-draw schedule. The separated conditional model has no finite maximum-
likelihood coefficient, so Python retains its explicit no-interval error.

### Poisson-log check

The count check uses `fit_poisson` and `fit_poisson_formula` for
complete nonnegative integer counts observed over equal periods. The joint
coefficient is a conditional log mean ratio; predictions are expected counts.
Two fixed cases come from isolated installed R qgcomp 2.19.6. One has exact
cell means 1, 2, 3 and 6, giving a joint mean ratio of 6. The other uses
four quantile groups and a numeric adjustment variable. The
[pinned fixture](../tests/fixtures/r_poisson_2_19_6.json) records the fitted
R values, and the [saved report](../validation/poisson-reference-results.json)
records field-level differences.

```sh
python validation/check_poisson_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker verifies the fixture, exporter and loaded R namespace,
then compares saved R, current R and Python. Scores and residual degrees of
freedom match exactly. Coefficients and predictions use `5e-10` absolute and
relative tolerances; uncertainty fields use `2e-6`. The largest observed
joint-variance difference is `1.3648731733706398e-7`. R's default GLM stops
before the tighter Python scoring criterion, so the reported information
matrix differs slightly. The separate exact-cell calculation checks Python's
final Fisher covariance at tighter precision.

These cases do not validate repeated-sample coverage, unequal observation
times, overdispersed counts or full R-package behavior.

### Poisson exposure-time offset check

`fit_poisson` accepts a positive observation-time vector. It adds its logarithm
to the count model with coefficient fixed at one, so the joint coefficient
compares rates at equal observation time. Named-data fits take the time column
through `exposure_time="time"`; the restricted formula grammar does not parse
R's inline `offset(log(time))` expression. New expected-count predictions
require new times. Missing times can be omitted only with `missing="drop"`.

Two fixed cases were exported from isolated installed R qgcomp 2.19.6. The
first has exact rates 1, 2, 3 and 6 over periods 1 through 5. The second
uses estimated quartile cuts and a numeric adjustment. The [fixture](../tests/fixtures/r_poisson_offset_2_19_6.json)
and [saved comparison](../validation/POISSON_OFFSET_RESULTS.md) contain the
inputs and results. Recheck the installed package with:

```sh
python validation/check_poisson_offset_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker verifies the fixture and exporter identities, installed package
version and namespace location, then compares current R and Python values.
The fixed cases check implementation agreement, not interval coverage across
repeated datasets or under overdispersion. A separate
[marginal offset check](../validation/POISSON_BOOTSTRAP_OFFSET_RESULTS.md)
covers the bootstrap method.

### Binary/count complete-case check

The named-data `fit_binomial_formula` and `fit_poisson_formula` functions now
accept `missing="drop"`. The four fixed [R cases](../tests/fixtures/r_glm_missing_2_19_6.json)
cover each family with estimated quartile cuts and with already assigned
scores. Missing outcomes, an exposure, a numeric adjustment and a text
category are represented. The quantized cases deliberately remove early
outcomes: using only retained rows to estimate the first exposure's first
cut would move it from 0.26171875 to 0.390625 and change the joint estimate.
The Python result follows R's pre-omission cutpoint order and records the
retained original row positions.

```sh
python validation/check_glm_missing_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker pins its R exporter and fixture, verifies the loaded
qgcomp 2.19.6 library, compares saved and live R values, then checks Python
rows, scores, coefficients, covariance, joint inference, fitted means and
predictions. All four cases passed against isolated R 4.5.1/qgcomp 2.19.6.
The largest full-covariance difference was `6.16e-8`; the checker permits
`1e-7` absolute and relative tolerance for GLM uncertainty fields, and
`1e-10` for coefficients and predictions. Exact row and score checks remain.
The [saved report](../validation/glm-missing-reference-results.json) gives
case-level differences. This verifies fixed software behavior, not unbiased
complete-case estimation, imputation, or interval coverage across datasets.

### Weighted binary/count check

Positive case weights in the two additive GLMs are R-style prior weights.
They multiply each row's log-likelihood contribution and Fisher information;
they do not change exposure quantile cuts. Four fixed cases from an isolated
installation of R qgcomp 2.19.6 cover exact and quantized exposure scores in
both outcome families. The [fixture](../tests/fixtures/r_glm_weights_2_19_6.json)
records coefficients, full covariance, scores, cuts, joint inference,
fitted values and predictions. Independent integer-frequency tests compare
each weighted fit with repeated rows. The largest fixed Python/R field gap
is below `5e-7`; R and Python use different GLM stopping tolerances.

Run the live re-export with the matching R installation:

```sh
python validation/check_glm_weights_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library
```

This check verifies the saved fixture hash, the loaded R namespace, inputs
and R outputs. These cases do not establish weighted interval coverage or
survey and cluster standard errors.

## Resampling and cluster checks

### Fixed-cut Gaussian bootstrap check

`fit_bootstrap` estimates the original additive Gaussian joint effect and its
row-bootstrap standard error. Exposure cuts are determined once from the
original data. Each replicate resamples rows, refits the conditional model,
and recalculates the joint slope. The standard deviation of the replicate
slopes uses `B-1` in its denominator. The reported interval is a normal
Wald interval; the test uses the R Gaussian bootstrap Student t degrees of
freedom. These are reference conventions, not a claim of calibrated coverage.

Two fixed cases check prequantized scores and automatic quantile cuts with
numeric adjustment. R and NumPy do not use the same random-number generator.
The [fixture](../tests/fixtures/r_bootstrap_2_19_6.json) therefore records the
exact R row schedule, which the Python comparison reuses. It compares every
bootstrap slope, intervention mean, point effect and inference field, as well
as the saved R values. The [report](../validation/bootstrap-reference-results.json)
records the installed-package run and numerical differences.

```sh
python validation/check_bootstrap_r.py --rscript /path/to/Rscript \
  --r-library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker requires an isolated installation of R qgcomp 2.19.6 and refuses
an existing output. It checks the fixture, exporter, installed namespace,
shapes and finite values; numeric agreement uses `5e-10` absolute and
relative tolerance. Python's `breaks=None` for prequantized inputs differs
from R's fabricated display breaks. The Python package itself needs only
NumPy and SciPy. These fixed comparisons do not test repeated-dataset
interval coverage, nonlinear models, interactions, cluster resampling or
other outcome families.

### Quadratic Gaussian bootstrap check

`fit_quadratic_bootstrap` fits linear and squared terms for every exposure
score. Under a joint intervention setting all scores to level \(a\), its
marginal mean is a degree-two polynomial. Two fixed cases from isolated
installed R qgcomp 2.19.6 check a prequantized and an automatically
quantized fit. The [fixture](../tests/fixtures/r_quadratic_2_19_6.json)
records conditional coefficients and covariance, intervention means, the
marginal polynomial, exact R bootstrap row schedules, bootstrap coefficient
pairs and joint inference. The [report](../validation/quadratic-reference-results.json)
records field-level Python/R differences. All compared numeric fields agree
within `1e-10`; independent polynomial design and bootstrap calculations
check the Python path separately.

```sh
python validation/check_quadratic_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker pins its exporter and fixture, verifies the loaded
qgcomp namespace, rejects an existing report path and compares saved R,
current R and Python values. R synthesizes break vectors when input scores
are already assigned; Python retains `breaks=None` for that mode. This is
a documented interface difference, not a difference in fitted scores or
inference. The fixed cases do not establish interval coverage under repeated
datasets, clustered variance or full R-package parity.

### Gaussian pairwise-interaction check

With `pairwise_interactions=True`, `fit_quadratic_bootstrap` adds every distinct
exposure-score product to the conditional model. When all scores are set to
level (a), each product contributes to the coefficient of (a^2). The
[fixture](../tests/fixtures/r_interactions_2_19_6.json) saves two installed-R
qgcomp 2.19.6 cases, exact row schedules and full conditional, marginal and
bootstrap outputs. The [report](../validation/interactions-reference-results.json)
gives field-level differences. All compared fields agree within `1e-10`.
An independent three-exposure least-squares calculation checks all three
distinct products and the full bootstrap covariance.

```sh
python validation/check_interactions_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker pins the fixture and R exporter, rejects changed inputs or row
schedules, and compares live installed-R values with both saved R and Python.
R's synthetic prequantized cutpoints remain a documented interface difference.
These fixed cases do not establish repeated-sample interval coverage. Separate
reports assess [exposure-covariate interactions](../validation/COVARIATE_INTERACTIONS_RESULTS.md)
and [clustered quadratic fits](../validation/QUADRATIC_GAUSSIAN_EE_RESULTS.md).

### Clustered Gaussian estimating-equation check

`fit_gaussian_ee` uses additive conditional and marginal estimating equations,
then sums their scores by cluster for robust covariance. Two fixed cases from
isolated installed R qgcomp 2.19.6 each contain 96 observations in 32 clusters.
The [fixture](../tests/fixtures/r_gaussian_ee_2_19_6.json) saves the full
conditional, marginal and stacked robust covariance, cluster-score meat,
derivative bread, effect inference, scores, cuts and intervention means. The
[report](../validation/gaussian-ee-reference-results.json) records field-level
differences. All fields except the derivative bread agree within `2e-10`.
Python uses an exact derivative; R's numerical Richardson derivative differs
by at most `1.4460056263487786e-08`, within a separate `2e-8` tolerance.
An independent conditional cluster-sandwich calculation checks the joint
effect and variance.

```sh
python validation/check_gaussian_ee_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The checker pins the R exporter and fixture, rejects changed inputs or
reference outputs, and compares live installed R with both saved R and Python.
R saves synthetic boundaries for assigned scores; Python keeps `breaks=None`.
These cases do not establish coverage over repeated datasets, suitable
small-cluster inference or causal interpretation.

### Marginal binary odds-ratio check

The [odds-ratio fixture](../tests/fixtures/r_odds_ratio_2_19_6.json) contains two
fixed four-level binary cases, one unadjusted and one adjusted, with 24 saved
bootstrap row sets each. The checker calls installed R qgcomp
2.19.6's exported `qgcomp.glm.boot(..., rr=FALSE)`, checks its loaded namespace
and version, compares the fresh R output with the pinned fixture, and then
compares Python using the same row sets.

Use an R executable compatible with the isolated library's compiled packages.
The library used for the saved result was prepared under R 4.5.1; selecting a
different Homebrew R 4.5.0 executable caused a status-139 load failure.

```sh
python validation/check_odds_ratio_r.py --rscript /path/to/Rscript \
  --library /path/to/isolated-r-library --output /path/to/new-report.json
```

The [saved result](../validation/odds-ratio-reference-results.json) reports
differences for the effect, uncertainty, standardized risks, marginal-model
coefficients and all bootstrap slopes. It is a fixed-case software comparison,
not a repeated-data interval study.

## Other fixed comparisons

The [conditional result-table comparison](../validation/RESULT_TABLES_RESULTS.md)
uses seven existing fixed inputs and an isolated installed-R checker.
The marginal Poisson bootstrap also has an
[unequal-observation-time comparison](../validation/POISSON_BOOTSTRAP_OFFSET_RESULTS.md).
Its checker uses two installed-R cases with identical 24-draw row
schedules, compares every replicate and uses an independent Poisson
score-equation calculation. The count means average over each case's observed
time distribution; the one-step mean ratio equals the rate ratio under the
additive log-link model. This fixed comparison does not measure interval
coverage across repeated datasets.
