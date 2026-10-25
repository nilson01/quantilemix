# Marginal Cox bootstrap interval study protocol

These settings were fixed before the full study. The separate
[20-case pilot](COX_BOOTSTRAP_PILOT_RESULTS.md) measured feasibility and is
excluded from the study. The [results](COX_BOOTSTRAP_STUDY_RESULTS.md)
record the completed run and its failure-record deviation.

## Question and estimand

What are the bias and coverage of the normal 95% interval from `fit_cox_bootstrap` in four correctly specified, homogeneous-hazard proportional-hazards settings? The primary result is an estimated coverage rate with a 95% Wilson interval; there is **no pass/fail equivalence claim**. Four independent exposure scores (Q_j) are each uniform on ({0,1,2,3}). For person (i), the event hazard is

`lambda_i = 0.08 * exp{b * (sum_j Q_ij - 6)/4}`.

The estimator fits a conditional Cox model and **sets all four scores to the same level `a` for every simulated person**. There are no other covariates. Under this data-generating model, each fixed-level regime has hazard `0.08 exp{b(a-1.5)}` for every person. Changing the common level from `a` to `a+1` therefore changes the log hazard by exactly `b`; the fixed-regime marginal hazards are proportional and have the same slope. This set-to-level argument, rather than a shift of each person's observed scores, supplies the analytic target. The finite-sample fitted baseline and simulated event times can still introduce bias, which the study measures. The amended analysis uses fixed breaks `[-0.5, 0.5, 1.5, 2.5, 3.5]` for each exposure and `q=4`. These map the generated integer scores exactly to levels 0, 1, 2, and 3. Save the actual intervention levels for each fit.

Independent exponential censoring has rate `c`, and administrative follow-up ends at 12. The analysis uses fixed four-category breaks, degree one, `B=200`, `MCsize=10000` per intervention level, and `alpha=0.05`. This design is a narrow baseline for the supported numeric, independent-row method. After the fixed intervention, every person has the same hazard. Therefore its results cannot establish performance for heterogeneous or correlated mixtures, covariate-dependent hazards, nonproportional marginal hazards, empirical cutpoint uncertainty, confounding, informative censoring, degree two, clusters, or causal identification from observational data. Broader marginal-method coverage needs a separately reviewed design with an independently justified target.

| Factor | Values |
| --- | --- |
| True log hazard ratio `b` | 0, 0.4 |
| Independent censoring rate `c` | 0.10, 0.30 |
| Rows per dataset | 300 |
| Pilot | 5 fresh datasets in each cell; 20 total |
| Study | 2,000 fresh datasets in each cell; 8,000 total |

Cells are ordered by censoring rate, then effect. The null cells assess false positives; nonzero-cell rejection rates describe power. A separate smaller-sample stress study can follow if failure rates or coverage warrant it. It cannot be folded into these four cells after seeing their results.

## Streams, outputs, and checks

Use NumPy PCG64 with master seed 20261024. For data, construct `SeedSequence([20261024, 301 or 302, cell, repetition])`, where 301 is pilot and 302 is study. Draw the score matrix, exponential event times, then independent exponential censoring times, in that order. For the fit seed, take the first unsigned 64-bit word from `SeedSequence([20261024, 303, 1 or 2, cell, repetition])`. Pilot and study data are disjoint. Hash input arrays as little-endian 64-bit scores, durations, then event indicators. Exact replay requires the recorded NumPy version.

The fixed pilot cell-0/repetition-0 input hash is `4a4d3b2108723168a14459cdf798581c930f5a519b00c136b4186bdc90b6d50b`, with 129 observed events. The hash agrees under the locally checked NumPy 2.0.0 and 2.5.4 environments. This is a drift guard, not a coverage result.

Run disjoint batches identified by mode, cell, starting repetition and count. Save source/environment identity, exact invocation, thread settings, settings, every generated input hash, event count, elapsed time, actual intervention levels, all bootstrap effects, the estimate, standard error, confidence limits, p-value, warnings and every failure. Hash the estimator's Cox, bootstrap, and core modules, its package initializer, protocol, runner and checker. Keep the committed source or private retrievable bundle with the returned results. Write and flush a journal header, a start marker **before** each case, and its row after fitting; finish with a complete batch file, then remove its journal.

Do not overwrite either. On a catchable interruption, retain a terminal marker and the partial journal; a hard kill can lack that marker, so retain the scheduler exit log. The checker audits partial journals, labels started-but-unreported cases interrupted and the rest never started, and rejects duplicates across partial and complete batches. A failed or interrupted fit is never replaced under the same case ID. Nonfinite fit output is recorded as a failed fit. A complete-study audit rejects partial journals or missing cases.

The checker must reject duplicate, out-of-range, inconsistent or nonfinite completed records. It regenerates the case input and independently recomputes bootstrap standard deviation, normal limits and p-value from saved effects. Shared generator code means this is not a wholly independent data-generating-process audit; analytic truth and fixed seeded examples supply separate checks.

Per cell, report planned, attempted, successful, failed, interrupted, never-started, warning and missing counts; observed event fraction, total censored fraction, and its administrative and independent-censoring components; mean estimate minus truth with Monte Carlo standard error; empirical estimate standard deviation; root mean reported variance; mean interval width; 95% coverage with Wilson interval; rejection **among successful fits** with Wilson interval; and all-planned lower/upper bounds for both coverage and rejection by treating every unsuccessful or missing case in the two extreme ways. **Every estimate-based summary is conditional on successful fits** and must be read beside the failure counts; it is not an all-planned bias estimate if fits fail.

At 2,000 successful fits, Monte Carlo standard error near 95% coverage is about 0.0049. Null rejection is the false-positive rate among successful fits. Flag any warning, failure, nonfinite value, checker discrepancy, absolute bias above three Monte Carlo standard errors, or coverage Wilson interval wholly outside [0.93, 0.97] for investigation. The broad flag band is an investigation trigger, not an acceptance range. Report observed coverage, its denominator and difference from 0.95 even when unflagged. An unflagged cell does not establish adequate coverage.

Report all outcomes even if unfavorable.

The installed-R [fixed-case checks](COX_RERUN_20261024.md) assess software stages separately. R qgcomp 2.19.6 and Python use different Monte Carlo event generators, so an end-to-end output match is not a valid criterion for this study. Do not present fixed-case agreement as interval coverage.

## Compute and stopping boundary

The [four-level pilot](COX_BOOTSTRAP_PILOT_RESULTS.md) measured a mean
596.13 fit seconds per case, a 761.56-second maximum and peak task memory
of 66.13 MB. The [server job specification](server_cox_bootstrap/README.md)
records CPU, memory, time and output limits, stop rules and the terminal audit.
Preserve every attempted result and failure; do not reduce draws or
repetitions to fit resources.
