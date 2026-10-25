# Scenario 5: correlated Gaussian exposure scores

Revised after independent protocol review, before full execution. Protocol ID
`keil-2020-scenario5-qgcomp-v1`. This is a separate study from the completed
[Table 3 comparison](PAPER_REPRODUCTION_RESULTS.md). It tests whether the
existing additive Gaussian estimator's bias and interval coverage change when
two causal exposure scores are correlated. No scientifically approved tolerance
for "low bias" is available, so this study makes no low-bias pass/fail claim.
It does not test WQS or settle the earlier cancellation
variance difference.

## Sources and design choice

[Keil et al. (2020)](https://doi.org/10.1289/EHP5838) give the scenario-5
coefficients and correlation settings in Table 1 and show the qgcomp bias
distributions in Figure 1. The official [supplementary code](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7228100/supplementaryFiles)
has nested code ZIP SHA-256
`ba2a8a858a31f24593d38d94dd204db2a18a214b4aa61645cf7748f988f54ccd`.
It provides `dgm_quantized` and `analysis_1_5`. The latter fits additive
Gaussian `qgcomp.noboot` with `q=NULL`. The released script's scenario-5
illustration uses four exposures and five iterations, and explicitly says it
does not reproduce the original 1,000-iteration results. Figure 1 instead
specifies nine exposures, n=500 and 1,000 datasets per setting. Its boxplots
do not supply a complete numeric result table.

The generator argument `ncor=2` in the released illustration selects the
first **three** exposure columns because its condition is `k <= ncor+1`.
The article describes correlation between the two causal exposures. We will
therefore keep the executable interpretation (`ncor=2`) as the **primary**
translation and treat `ncor=1`, which routes just those two columns through
the shared-master mechanism, as a separately labeled sensitivity study.
Neither is asserted to be the paper's unavailable production stream.
The article's scenario-5 prose lists total effects 0.05, 0.15 and 0.25,
although Table 1's coefficients and Figure 1's caption imply 0.05, 0.15
and **0.20**. The coefficients govern this protocol; the highest effect is
0.25 − 0.05 = 0.20. The paper's 0.25 prose value remains an unresolved
source inconsistency.

## Data and estimand

Each dataset has 500 rows and nine integer exposure-score columns. The
coefficient for X1 is 0.25; X2 has one of −0.20, −0.10 or −0.05; X3–X9 have
zero coefficients. The true joint one-score effect is ψ = 0.25 + β2, namely
0.05, 0.15 or 0.20. The shared-master generator uses correlation argument
ρ = 0, 0.4 or 0.75. A balanced master vector contains 125 copies of each
score 0–3. For every exposure column, a Bernoulli mask has success probability
sqrt(ρ). Masked rows copy their master's score; unmasked master scores are
permuted among those rows. Columns selected by `ncor` use this construction;
other columns independently permute the balanced master vector. Each outcome
is the coefficient-weighted score sum plus one independent standard-normal
error. No covariates, missingness, quantization, weights or interactions are
introduced. The Python fit includes an intercept, all nine already scored
exposures and `q=None`.

Both generator variants cross the same nine coefficient/correlation cells.
The pilot has five datasets per cell and variant (90 fits); the planned full
study has 1,000 per cell and variant (18,000 fits). The primary nine cells
match Figure 1's sample size, exposure count and coefficient/correlation
grid while translating the released generator. The sensitivity nine cells
test the source ambiguity. Pilot and full-study streams use disjoint NumPy
PCG64 `SeedSequence` namespaces with master seed 20261023, variant, cell and
replicate recorded. They do not attempt R's random stream. Each generated
input has a row-order SHA-256 digest. All planned cells, including failed fits,
remain in a flushed JSONL journal.

## Checks and result table

Before the full run, compare a fixed subset of pilot inputs with isolated,
installed R qgcomp 1.3.0 and 2.19.6. Save package/dependency identities,
input hashes, effects and variances. This tests same-input software agreement,
not repeated-data performance. Independently recompute every successful fit
with QR least squares and the nine-score contrast, including variance,
interval and p-value. Verify all balanced-column counts and report realized
Pearson correlations for X1–X2, X1–X3 and X2–X3. The latter two expose the
generator-variant difference; the realized correlations are observations,
not assumptions of exact finite-sample equality to ρ.

For every variant and cell, report planned/successful/failed fits, bias,
empirical standard deviation, root mean reported variance, median and
interquartile range of estimate-minus-truth, 95% interval coverage, fraction
with p<0.05, bias Monte Carlo standard error, Wilson 95% intervals for rates,
and mean and standard deviation of realized correlations. Report rates among successful fits and
worst/best all-planned bounds when failures occur. With 1,000 successful
datasets, bias Monte Carlo error is empirical SD divided by sqrt(1000).
Report a descriptive bias interval of the estimate plus/minus 1.96 Monte Carlo
standard errors.

Compare zero bias with a descriptive three-MCSE flag, and coverage with
0.95 using three binomial standard errors. These flags trigger inspection;
their absence is not equivalence, proof of nominal coverage, or a causal
guarantee. Report all nine primary cells with no pooled verdict. Because
every true effect is nonzero, p<0.05 is empirical rejection (power), not a
false-positive rate. Intervals use a normal critical value while p-values
use a Student t tail; their threshold events need not coincide.

Compare
generator variants cell by cell, with uncertainty for their independent-stream
bias difference. At zero requested correlation, inspect agreement in
distribution rather than seed-paired equality. Figure 1 permits a
qualitative comparison of bias distributions only, not exact numeric
agreement or a claim to have reconstructed its original random draws.

If a fit fails, report its variant, cell, exception, realized correlations,
design rank and condition number. Successful-fit bias has no finite
all-planned bound without assumptions on missing estimates. Withhold any
unconditional bias interpretation until failures and their pattern have been
reviewed. Retain warning counts by cell and inspect unusual correlations.

The independent checker must reject changed input hashes, duplicate/missing
identities, nonfinite outcomes, altered fit values, omitted failures, stale
summary denominators and journal mismatch. Fault-injection tests should
include valid controls. A stopped run retains completed journal rows and
uses a new run ID for any rerun. Do not silently resume or overwrite.

## Compute and interpretation boundary

The 90-fit pilot is a bounded implementation and resource check only. Record
wall time, process memory, output size, fit warnings and independent-check
time. Extrapolate conservatively to 18,000 fits. The full study is
`SERVER_CPU` and needs a reviewed protocol, current server admission, source
archive verification, explicit CPU/memory/time/storage limits and a clean
revision before launch. The runner and return checker must verify clean Git
status, the exact commit, and that the runner and package core match it.
Verify the server archive hash before execution and on return. Do not shrink
its cells or repetitions to fit the
Mac. Review unexpected correlation patterns or rank failures before
interpreting bias. The qgcomp arm alone cannot reproduce the paper's WQS
comparison, and the source inconsistencies limit any full Figure 1 claim.
Call every result a nine-exposure qgcomp-only translation of the released
generator and article settings, not an exact Figure 1 reconstruction.
