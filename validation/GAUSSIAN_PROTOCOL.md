# Gaussian uncertainty validation

Protocol: `gaussian-known-scores-v1`.
The separate pilot and full study are complete. See the [study report](GAUSSIAN_RESULTS.md) and
[pilot record](calibration-pilot-results.json). The prospective text below is
preserved from the pre-study protocol at revision `8cb4619`; statements about
future execution and uninspected outcomes describe that preparation stage.
The machine-readable design is
[calibration-config.json](calibration-config.json). This is an internal dated
protocol, not a registered study or evidence that calibration has passed.

## Question and scope

We will check how the implemented additive Gaussian model estimates a known
joint effect and reports its uncertainty. The study separates two questions:
whether Python agrees with installed R qgcomp 2.19.6 on identical data, and
whether each method's intervals and tests have the expected repeated-sample
behavior. Agreement alone cannot answer the second question.

The model and joint coefficient sum follow [Keil et al. (2020)](https://doi.org/10.1289/EHP5838).
That paper examined bias, coverage and rejection rates using qgcomp 1.3.
This protocol evaluates the present Python implementation against qgcomp
2.19.6; its selected settings are not a reproduction of the paper's full study.
We retain the current Python method and its documented R conventions.

The target is a one-category joint change in four **known exposure scores**,
conditional on two adjustment variables. All rows are independent. Outcomes
have additive means, independent normal errors and constant error variance.
The study does not evaluate empirical quantile estimation, causal identification,
missing data, weights, nonlinear effects, other families or dependence between
rows. Those questions need separate designs. Earlier plans for empirical-cut
and assumption-violation examples remain deferred, not completed or discarded
because of observed results. No study outcomes have been inspected.

## Data-generating process

For each row, draw independent standard normal variables
\(Z_1,Z_2,U,\eta_1,\ldots,\eta_4,\epsilon\). Here \(Z_1,Z_2\) are
adjustment variables, \(U\) is shared by the four exposures, and \(\eta_j\)
is exposure-specific noise. Define

\[
 L_j=0.5Z_1+\sqrt{\rho}\,U+\sqrt{1-\rho}\,\eta_j,
 \qquad j=1,\ldots,4.
\]

Each latent exposure has variance \(0.25+\rho+(1-\rho)=1.25\).
Its marginal normal quartiles are therefore fixed and known. Let
\(c_k=\sqrt{1.25}\,\Phi^{-1}(k/4)\), for \(k=1,2,3\), where
\(\Phi\) is the standard normal distribution function. Assign score
\(Q_j=\sum_{k=1}^3 1\{L_j>c_k\}\). This gives scores 0, 1, 2 or 3;
a value exactly at a cut stays in the lower category. We pass these scores
directly to each method, with `q=None` in Python and `q=NULL` in R.

The parameter \(\rho\) describes the shared residual component before
quantization. It is not the Pearson correlation of the scores. Even at
\(\rho=0\), the latent exposures share \(Z_1\). Their unconditional
correlation is \((0.25+\rho)/1.25\); their conditional correlation given
\(Z_1\) is \(\rho\).

Generate the outcome as

\[
 Y=1+\sum_{j=1}^4\beta_j Q_j+0.5Z_1-0.25Z_2+\epsilon.
\]

The true joint effect is \(\psi=\sum_j\beta_j\). We fit an intercept,
all four scores and both adjustment variables. Thus a full-rank design has
seven coefficients and residual degrees of freedom \(\nu=n-7\).

| Design component | Values and purpose |
| --- | --- |
| Sample size \(n\) | 100 and 500: smaller and larger samples |
| Residual exposure correlation \(\rho\) | 0 and 0.75: lower and higher dependence |
| Null coefficients | \((0,0,0,0)\), joint effect 0 |
| Cancellation coefficients | \((0.25,-0.25,0,0)\), joint effect 0 despite opposing component effects |
| Positive coefficients | \((0.10,0.10,0.05,0)\), joint effect 0.25 |

The Cartesian product contains 12 cells. Each cell has 2,000 independent
replicates, for 24,000 planned datasets. Seeds are assigned by mode, cell and
replicate identity, not by completion order. The configuration fixes the master
seed, separate pilot/study namespaces and PCG64 random-number generation.
The environment and input hashes are recorded because equal seeds alone do
not guarantee identical data across software versions or languages. R receives
the exact generated score, covariate and outcome arrays; it does not generate
another dataset from the seed.

## Comparators and numerical agreement

Each dataset is evaluated by three methods:

1. Native Python `quantilemix.fit`, with prequantized scores and the current
   `qgcomp_2.19.6` inference convention.
2. Installed R `qgcomp::qgcomp.glm.noboot`, Gaussian identity link, explicit
   exposure names, both covariates, unit weights, `bayes=FALSE`, `q=NULL`
   and `alpha=0.05`. The isolated library must contain qgcomp 2.19.6.
3. An independent QR least-squares calculation of the same coefficient sum
   and covariance, with a conventional Student t interval and p-value using
   \(n-7\) degrees of freedom. It does not call Python qgcomp to construct
   its expected coefficients or variance.

For full-column-rank design matrix \(X=QR\), the independent calculation uses
\(\widehat b=R^{-1}Q^TY\) and residual variance
\(s^2=\|Y-X\widehat b\|^2/(n-7)\). For contrast
\(a=(0,1,1,1,1,0,0)^T\), it computes
\(\widehat\psi=a^T\widehat b\) and
\(\widehat{\operatorname{Var}}(\widehat\psi)
=s^2\|R^{-T}a\|^2\). Solves replace an explicit matrix inverse.
The comparator supplies an independent implementation, not an independent
numerical library: NumPy/SciPy are still shared dependencies.

Compare Python and R estimates, variances, standard errors and interval
endpoints using `atol=1e-10, rtol=1e-8`; compare ordinary p-values with absolute
tolerance `1e-10` and degrees of freedom exactly. Compare the independent
estimate, variance and standard error against Python with the same numerical
tolerances. The independent t interval and p-value intentionally differ from
the reference convention and are not numerical-agreement targets. Keep every
mismatch with its input and reported values. Do not change tolerances after
seeing a failed result without a dated protocol amendment and new run identity.

Only fitting is used. The documented prequantized R prediction-wrapper defect
does not affect this comparison. R is an optional validation dependency;
Python package users still need only NumPy and SciPy.

## Expected interval and test behavior

Conditional on any full-rank design in this correctly specified normal model,
\((\widehat\psi-\psi)/\widehat{\operatorname{SE}}\) has a Student t
distribution with \(\nu=n-7\) degrees of freedom. This gives an analytic
benchmark in addition to repeated samples.

Python and the R reference use normal critical values for intervals, but
\(n-5\) degrees of freedom for p-values. If \(F_\nu\) is the Student t
distribution function, their 95% interval coverage under this design is
\(2F_\nu(z_{0.975})-1\). Under either joint-null coefficient vector, their
rejection probability is
\(2[1-F_\nu(t_{n-5,0.975})]\). Here \(z_p\) and \(t_{d,p}\) are the
normal and Student t quantiles. These formulas follow by applying the actual
critical values to the same pivot; they do not assume the interval and test
use the same rule.

| Sample size | Analytic normal-interval coverage | Analytic reference null rejection |
| --- | --- | --- |
| 100 | 0.94700655 | 0.05006228 |
| 500 | 0.94943710 | 0.05000227 |

These are deterministic distribution calculations, not simulation results.
The conventional t comparator has coverage 0.95 and null rejection 0.05
under the stated model and full-rank condition. Consequently, a small shortfall
from 95% for the reference normal interval is expected and must not trigger
an undocumented change to the estimator. Rank or fitting failures remain
visible and limit any unconditional performance claim.

## Performance measures and denominators

For every cell and method, report planned, attempted, successful, failed and
missing replicate counts. Report paired comparison counts separately. A
successful method result must contain finite valid estimates and uncertainty;
a method failure does not make another method's result disappear.

Among the \(m\) successful replicates, let \(e_r=\widehat\psi_r-\psi\).
Report mean error (bias) \(\bar e\), empirical standard deviation of
\(\widehat\psi\), and Monte Carlo standard error of mean error
\(s_e/\sqrt m\). Monte Carlo standard error describes uncertainty due to
the finite number of replicates; it differs from the empirical standard
deviation itself. Also report the square root of the mean reported variance,
mean interval width, 95% interval coverage and its Wilson 95% binomial
interval, and the rejection fraction with a Wilson interval. Label rejection
as false-positive rate for either \(\psi=0\) case and power for \(\psi=0.25\).

Coverage and rejection use successful-result denominators and are explicitly
conditional on success. Also report covered/planned and rejected/planned
fractions as operational counts, without calling missing results noncoverage
or nonrejection. Give a worst/best bound on all-planned coverage by assigning
unavailable results first to uncovered and then to covered. Failure rates and
missing fractions accompany every performance table. With fewer than two
successful replicates, empirical SD and bias MCSE are unavailable. With none,
all performance measures are unavailable; an empty run cannot pass.
Empirical SD, the square root of mean reported variance, and mean interval
width are descriptive summaries in this design. No uncertainty interval or
formal variance-calibration test is planned for these three summaries.

At coverage 0.95, 2,000 replicates give approximate Monte Carlo standard error
\(\sqrt{0.95(1-0.95)/2000}=0.00487\). Report the actual intervals and counts
rather than treating this planned precision as observed accuracy.
For power near 0.5, the corresponding Monte Carlo standard error can be about
0.0112; the coverage precision does not apply to every metric. We follow
the simulation-design distinctions in [Morris, White and Crowther (2019)](https://doi.org/10.1002/sim.8086):
aims, data generation, targets, methods and performance measures are specified
before results are inspected.

## Failure handling and interpretation

Any numerical disagreement, rank failure, malformed response, nonfinite
result, missing reference dependency or changed environment must be reported.
There is no automatic retry of a statistical failure with a fresh seed.
Missing or invalid R output cannot silently turn into a Python-only study.
Infrastructure failure stops the current execution; valid earlier atomic
checkpoints are retained. Resume must verify the same configuration, source,
environment and R identity, and reject corrupt, duplicate or stale records.
Incomplete batches may be repeated deterministically but cannot be counted
twice.

Existing completed failures are not replaced by successful retries.
Finite unexpected R degrees of freedom are retained as numerical mismatches,
with both returned values and valid sibling results. The batch is saved before
execution stops. Each valid R result is checked against the planned degrees
of freedom even if Python fails; reference-convention counts remain separate
from paired-method availability. Malformed responses remain infrastructure failures. Stored
failure codes distinguish missing dependencies, timeouts, invalid responses,
rank failures and numerical failures without copying arbitrary exception text.
Warning categories recorded before a Python or QR failure are retained.

Completion means all planned replicate identities are accounted for. It does
not mean the statistical results are favorable. The report must distinguish
incomplete execution, completed execution with failures/mismatches, and
complete numerical agreement. Calibration remains an interpretation of the
reported estimates and uncertainty, not a boolean returned by the program.

For investigation, flag absolute bias exceeding three MCSEs and a coverage
Wilson interval wholly outside [0.93, 0.97]. Compare coverage and joint-null
rejection with their analytic benchmarks as well. These are descriptive
diagnostics across multiple cells, not multiplicity-adjusted hypothesis tests
or automatic release criteria. An unusual result requires checking its
Monte Carlo uncertainty, failures and implementation before interpretation.
Never add replicates until a result appears acceptable. Changes after the
pilot or study need a dated amendment; retain prior outputs.

## Pilot, resources and execution boundary

The timing pilot contains 20 replicates from the largest sample-size,
highest-correlation positive-effect cell. It uses a separate seed namespace
and output directory and is excluded from study aggregates. It checks the
end-to-end process, including R startup, serialization, fitting, checkpointing
and aggregation. It cannot establish calibration or identify all possible
runtime bottlenecks across cells. Record elapsed time, output bytes and
external peak-memory measurements for Python and its R child; memory that
was not measured stays unknown.

Use one foreground worker and one numerical-library thread, with batches of
at most 20 datasets and a bounded R subprocess timeout. Start no worker pool.
The full study retains inputs and results, with a 3 GB output ceiling plus
separate temporary/environment headroom. The program's ceiling is a stop
condition, not a guaranteed size estimate or a disk-space reservation. Before
execution, check free space and the workspace-wide resource queue. Full-study
timing and machine class remain unassessed until the pilot. Extrapolate using
the pilot's measured rate with an explicit conservative contingency; do not
reduce repetitions or controls to fit a desired budget. A server-class study
requires a separately prepared and authorized job.

## Planned outputs and closure

The runner produces an identity manifest, atomic batch records containing
inputs and per-method results, and a cell-by-method aggregate. The aggregate
records counts, accuracy measures, analytic targets and paired disagreements.
Local run directories remain outside version control. Publish only reviewed,
path-free summaries when explicitly authorized; retain detailed failures locally.

Before a timing pilot, require: strict config/CLI checks; hand-calculated
aggregate examples; independent-contrast checks; injected missing/malformed R
responses; interrupted/resumed and stale-evidence tests; an actual installed-R
adapter check on existing fixed inputs; and package tests on supported delivery
environments. Record the reviewed source identity and unresolved conditions.
