# Gaussian bootstrap uncertainty validation

Protocol ID: `gaussian-bootstrap-known-scores-v1`.
The design was revised after the separate 20-case pilot and fixed before the
full study. The pilot's inspected outcomes are excluded from the study.

## Question and target

We ask whether the normal intervals returned by `fit_bootstrap` cover the
known joint mean difference in repeated independent datasets. Agreement with
installed R qgcomp 2.19.6 on two fixed cases establishes software behavior
for those cases, not interval coverage. The target and data-generating process
are the same as the earlier [known-score Gaussian protocol](GAUSSIAN_PROTOCOL.md):
four exposure scores formed from fixed population quartiles, two adjustment
variables, an additive Gaussian outcome with independent unit-variance errors,
and a true joint effect equal to the sum of four exposure coefficients.
Specifically, for any score vector whose four entries are 0, 1 or 2, compare
its conditional mean with the mean at a vector one category higher in every
entry, holding the adjustment variables fixed.

Both vectors stay within the
0–3 score range. In this additive model that difference is the same at each
such starting vector and equals the coefficient sum. We do not define an
all-row intervention that moves an observed top-category score from 3 to 4.
No causal conclusion follows from this simulation.

The earlier 24,000 datasets and their outcomes have already been inspected.
They are not reused for this prospective method check. A new master seed and
disjoint pilot/study namespaces generate fresh datasets. The established
`calibration.py` generator and its fixed design are reused unchanged; its
configuration and source hashes are included in run identity. The analysis
protocol is new even though the data-generating design is shared.

## Conditions and methods

The 12 cells are `n` in {100, 500}, residual latent exposure correlation
`rho` in {0, 0.75}, and coefficient setting `null`, `cancellation`, or
`positive`. The true joint effects are 0, 0, and 0.25. `rho` refers to the
shared residual component before scoring, not the observed score correlation.
Each full-study cell plans 2,000 independent datasets; this gives about
0.0049 Monte Carlo standard error for coverage near 95% in a cell. The timing
pilot has 20 separate datasets from the largest/highest-correlation positive
cell. Pilot data never enter study summaries.

On every dataset, fit `fit_bootstrap(y, scores, z, q=None, alpha=0.05,
B=200, seed=bootstrap_seed)`. Two hundred row resamples match the public
default; changing `B` after inspecting results requires a dated amendment and
new run identity. Each replicate draws `n` whole rows with replacement. The
original score cuts are fixed. Bootstrap seeds are derived from stable case
identities in a namespace separate from data-generation seeds. The runner
records both, plus input and source hashes. A failed bootstrap resample makes
that dataset's bootstrap method fail; it is not replaced by another draw.

On the same arrays, compute the model-based Gaussian `fit` interval and an
independent QR full-model Student-t contrast. These comparators help locate
implementation or convention differences; they do not turn bootstrap
intervals into exact intervals. The true coefficient sum is a target from the
generator, not an estimate inferred from the fitted model. The two existing
installed-R bootstrap fixtures are the reference check for method convention.
R is not run on every simulated dataset, and matching integer seeds across R
and Python is not presented as matching bootstrap samples. The Python package
has no R runtime dependency.

## Measures and decision rules

For every cell and method, retain planned, attempted, successful, failed and
missing counts. Among successful fits, report mean error (bias), its Monte
Carlo standard error, empirical standard deviation of point estimates, root
mean reported variance, mean interval width, 95% interval coverage, and a
Wilson 95% interval for coverage. Report the fraction whose two-sided p-value
is below 0.05. Call it a false-positive rate for the **zero joint effect** in the two joint-null settings
and power in the positive setting. Also report the fraction whose interval
excludes zero: the bootstrap's normal interval and Student-t test need not
make identical threshold decisions. For paired successful cases, report
bootstrap-minus-model-based differences in interval width and coverage, with
the paired count and both directional coverage-discordance counts; do not
compare rates with different hidden denominators. The cancellation setting
has nonzero, opposing component coefficients despite its zero joint effect.

Coverage and rejection rates use the successful-fit denominator. Separately
show covered/planned and the worst/best all-planned coverage bounds obtained
by treating unavailable results first as uncovered and then as covered. Never
silently remove a failure. Fewer than two successes make empirical standard
deviation and bias Monte Carlo standard error unavailable. The pilot reports
these counts only as execution diagnostics, not statistical calibration.

Before seeing the new study, flag for investigation an absolute bias above
three bias Monte Carlo standard errors and a coverage Wilson interval wholly
outside [0.93, 0.97]. These descriptive flags are not release criteria or
multiplicity-adjusted tests. Compare with the earlier correctly specified
Gaussian results, but do not treat their observed values as targets. A shortfall
requires checking the implementation, finite-B variability, failures and
Monte Carlo precision before interpretation. Do not add replicates or change
settings until a favorable result appears.
With 2,000 datasets, a true 95% coverage rate has about 0.49 percentage
points of Monte Carlo standard error. A cell without a descriptive flag does
not thereby prove equivalence to 95% coverage. The tested method uses its
finite default of 200 resamples, not an infinite-resample idealization.

## Execution, integrity and resources

Plan mode is read-only. Explicit pilot/study modes require a fresh output
directory; resume must match configuration, source and environment identity.
Before each batch, save generated inputs and seeds atomically. Completed
batches, including failed fits, are terminal and are never refit. A partially
written batch can be replayed only with its saved inputs. Keep case-level
failure codes, warning categories and the original rows needed to audit a
failure. Do not persist arbitrary exception text or local credentials.
Aggregate only verified checkpoints; a read-only checker independently
regenerates fixed inputs and recomputes interval/test conventions, counts and
all claim-bearing summary statistics. Its integrity status must separately
state whether all planned cases are terminal. Record CPU, memory, elapsed time,
output bytes, software versions and hash identities. Use one numerical thread
and one foreground worker for the local pilot. No full study runs on the Mac.

The 20-case pilot at `n=500` requires 4,000 bootstrap fits, plus paired
comparators. Its provisional storage ceiling is 100 MiB and its wall-time
ceiling is 10 minutes. Before it runs, recheck free storage, current workspace
worker reservations and available memory; reserve the workers and stop if
limits cannot be met. Record process-family peak memory when externally
measurable and distinguish unmeasured peaks. Use measured pilot throughput,
storage and failures to decide whether the full 24,000-case, 4.8-million-fit
study is feasible on the configured CPU server. A server job needs a separate
specification; do not shrink cells, repetitions or resamples to
fit the laptop. Keep the four earlier Gaussian estimators/fixtures and study
records unchanged.

A complete study means every planned case has a verified terminal record and
all metrics, failures and uncertainty are reported, regardless of whether the
method performs well. It would establish behavior only for this known-score,
correctly specified, independent-row design. Empirical cuts, nonlinear terms,
interactions, weights, missingness, clusters, other outcomes and reproduction
of the original paper's simulation results remain separate work.
