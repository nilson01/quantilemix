# Gaussian fixed-cut bootstrap: repeated-dataset validation

Protocol: `gaussian-bootstrap-known-scores-v1`.
This is a completed internal simulation for the additive Gaussian subset of
`fit_bootstrap`. It does not validate the other R qgcomp methods or establish
paper reproduction. The reviewed [protocol](BOOTSTRAP_PROTOCOL.md), frozen
[configuration](bootstrap-calibration-config.json), and
[exact saved summary](bootstrap-calibration-study-results.json) define this
result. The raw case inputs, 200 bootstrap effects per case, server logs and
independent checker reports are retained privately outside the source archive.

## Design and execution

The study used 12 conditions: 100 or 500 rows; latent residual exposure
correlation 0 or 0.75; and a zero joint effect, a zero joint effect with
opposing component coefficients (cancellation), or a positive joint effect of
0.25. Each condition had 2,000 newly generated datasets. Scores used fixed
population quartiles. Every dataset received 200 whole-row bootstrap
resamples, a model-based Gaussian comparison and an independent QR/Student-t
comparison. The target is an adjacent one-category increase in all four
scores, within the 0–3 score range. This is a model-based mean contrast, not a
causal intervention on observed people.

All 24,000 cases and all 72,000 method fits completed. There were zero saved
method failures or missing outcomes. A separate read-only census of every
saved case found zero warning entries across all three methods. A separate checker
reconstructed the fixed plan, inputs and seeds, recalculated the intervals,
p-values and every summary field, and passed 24,000/24,000 on the server. All
1,200 returned batch files and the summary matched those checker hashes.
After allowing at most `2e-15` absolute cross-platform rounding in the
independently regenerated floating inputs, the local checker passed all
24,000/24,000 too.

Saved file hashes, score categories, case identity and
seeds remain exact. Across 24,000 cases, only 21 had any Mac/Linux input
rounding difference; the largest absolute difference was `8.88e-16`, with
no score-category difference. The local checker's original exact-float
comparison failure and its repair are retained. A private replication inventory
identifies the source, environment, transfer and audit receipts. The checker
tests internal saved-record identity and calculations; it is not an independent
replication of the fitting engine.

## Results

Coverage is the percentage of successful intervals containing the known joint
effect. Each denominator is 2,000; because no fit failed, all-planned
coverage is the same. Wilson intervals show pointwise Monte Carlo uncertainty
for each coverage estimate. The last column is bootstrap minus model-based
coverage in percentage points on the same datasets. `p<.05` is a false-positive
rate for the zero **joint** effect in null and cancellation conditions, and
power for the positive condition. Bias MCSE is the Monte Carlo standard error
of estimated bias. `rho` is latent residual exposure correlation, not the
observed score correlation.

| Condition (n, rho, effect) | Bootstrap coverage, % (Wilson 95% interval) | Model-based, % | QR, % | Bootstrap p<.05, % | Bias (MCSE) | Paired coverage difference, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 / 0 / null | 94.05 (92.93–95.00) | 94.95 | 95.25 | 5.70 | -0.0022 (0.0046) | -0.90 |
| 100 / 0 / cancellation | 93.10 (91.90–94.13) | 94.00 | 94.40 | 6.45 | +0.0050 (0.0046) | -0.90 |
| 100 / 0 / positive | 94.50 (93.41–95.42) | 95.30 | 95.55 | 23.00 | -0.0027 (0.0045) | -0.80 |
| 100 / 0.75 / null | 94.70 (93.63–95.60) | 95.00 | 95.20 | 5.05 | +0.0008 (0.0026) | -0.30 |
| 100 / 0.75 / cancellation | 95.35 (94.34–96.19) | 95.25 | 95.55 | 4.20 | -0.0021 (0.0026) | +0.10 |
| 100 / 0.75 / positive | 93.70 (92.55–94.68) | 94.45 | 94.75 | 56.00 | -0.0013 (0.0026) | -0.75 |
| 500 / 0 / null | 95.55 (94.56–96.37) | 95.55 | 95.55 | 4.45 | +0.0012 (0.0019) | +0.00 |
| 500 / 0 / cancellation | 94.70 (93.63–95.60) | 95.30 | 95.30 | 5.25 | -0.0018 (0.0020) | -0.60 |
| 500 / 0 / positive | 93.65 (92.50–94.64) | 94.15 | 94.15 | 80.60 | +0.0015 (0.0020) | -0.50 |
| 500 / 0.75 / null | 94.65 (93.58–95.55) | 94.90 | 94.95 | 5.30 | +0.0010 (0.0011) | -0.25 |
| 500 / 0.75 / cancellation | 94.40 (93.30–95.33) | 94.55 | 94.60 | 5.55 | -0.0011 (0.0011) | -0.15 |
| 500 / 0.75 / positive | 94.40 (93.30–95.33) | 94.60 | 94.60 | 99.90 | -0.0011 (0.0012) | -0.20 |

The bootstrap coverage estimates range from 93.10% to 95.55%. Its largest
absolute estimated bias is 0.0050; in every condition, absolute bias is less
than three bias MCSEs. Bootstrap coverage is lower than model-based coverage
in ten conditions, equal in one and higher in one. The largest paired gap is
0.90 percentage points. These paired comparisons use 2,000 jointly successful
cases in every condition. In the two largest-gap cells, `n=100, rho=0, null`
has 16 cases covered only by bootstrap and 34 only by the model-based method;
`n=100, rho=0, cancellation` has 18 and 36. Both yield -0.90 percentage
points, with approximate paired Monte Carlo standard errors of 0.35 and 0.37
percentage points. The 12 paired comparisons have no multiplicity adjustment.
Other directional counts and width differences are in the saved JSON.

Three bootstrap pointwise Wilson intervals lie wholly below the nominal 95%
coverage value: `n=100, rho=0, cancellation` (93.10%; 91.90–94.13%),
`n=100, rho=0.75, positive` (93.70%; 92.55–94.68%), and
`n=500, rho=0, positive` (93.65%; 92.50–94.64%). Model-based and QR coverage
in these same three cells is respectively 94.00% and 94.40%, 94.45% and
94.75%, and 94.15% and 94.15%. The first cell also has a 6.45% false-positive
rate for the zero joint effect (Wilson interval 5.45–7.61%). That rate counts
129/2,000 Student-t p-values below .05; the normal bootstrap interval excludes
zero in 138/2,000 (6.90%), so p-value rejection is not simply one minus
interval coverage.

These observations warrant investigation; their intervals are
pointwise across 12 prespecified conditions and do not adjust for multiple
comparisons. The protocol's broader descriptive flag asks whether a coverage
Wilson interval is wholly outside 93–97%; no condition meets that flag.
Absence of that flag does not establish equivalence to 95% or justify a broad
calibration claim. The finite 200-resample normal interval is the tested
method; these results do not describe an infinite-resample idealization.

## Resources, identity and limits

Arseven Slurm job `2899480` completed 0:0 in 54 minutes 29 seconds with one
numerical CPU thread, 16 GiB requested memory and no GPU. The kernel cgroup
peak was 3,275,436,032 bytes (3.05 GiB), below the 14 GiB guard. The run
saved 629,574,181 bytes, below the fixed 3 GB output ceiling. The independent
server check was job `2899546`, completed 0:0 in 1 minute 39 seconds. The
source commit was private `418f871`. The summary SHA-256 is
`d8bab1967fe409f9eefd3dbcfb2b9cf1c215e8958c08dcabf36b07f2537d78d1`;
the identical server/local checker-report SHA-256 is
`defd71391839149b421c615f56279cb1eb4a623fb93cef271ba6200d92023f22`.
The post-run portability checker repair changes no generated input, fit,
interval, seed or saved summary. Exact versions and source hashes are in the
private manifest and server logs. The frozen summary JSON still contains a
generic pre-interpretation `interpretation` label saying a separate study
interpretation is needed. It describes the state when that record was written;
this report and the dated interpretation review supply that later assessment
without changing the saved numerical record.

The setup record retains an earlier server pilot job that finished its 20
fits but exited 2:0 after a checker invocation omitted a required path. A
separate read-only job passed that pilot without refitting. The full study
then used an independently measured 24,000-record memory rehearsal; copied
pilot records in that rehearsal were resource templates and were never used
as statistical data.

These findings apply to correctly specified additive Gaussian outcomes,
independent rows, four fixed-cut scores, two adjustment variables and the
specified coefficient/correlation grid. They do not test empirical score
cuts, weighted fits, missing data, binary/count outcomes, nonlinear effects,
interactions, clusters or causal validity. R qgcomp 2.19.6 agreement remains
a separate fixed-case check; this repeated-data study does not run R on every
dataset. The separate synthetic interpretation review is recorded privately;
its findings and responses accompany this report. The three lower-coverage
cells are descriptive signals, not evidence that finite bootstrap resampling
caused the difference. Broader paper and R-package coverage remains open.
