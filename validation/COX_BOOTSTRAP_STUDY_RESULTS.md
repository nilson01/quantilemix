# Marginal Cox bootstrap: four-condition known-score interval study

This report follows the [advance protocol](COX_BOOTSTRAP_PROTOCOL.md) and the separate [20-case feasibility pilot](COX_BOOTSTRAP_PILOT_RESULTS.md). We assessed the 95% normal interval from 200 row-bootstrap draws and 10,000 simulated profiles per intervention level. The target is the log hazard ratio when every exposure score is set to one common level rather than the preceding level. Under this study's homogeneous proportional-hazards generator, that target equals the generating value `b`. It is not a general marginal Cox guarantee.

Each condition has 2,000 planned independent datasets of 300 people. The two censoring rates `c` are 0.10 and 0.30; the true log hazard ratios `b` are 0 and 0.40. All results based on estimates use **successful fits only**. A Wilson 95% interval beside a coverage or rejection rate describes uncertainty from using a finite number of generated datasets. It is not a confidence interval for a fitted person's hazard ratio.

| `c` | `b` | Returned / planned | Bias (MCSE) | Coverage [Wilson 95%] | Coverage − 0.95 | Rejection [Wilson 95%] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0 | 2,000 / 2,000 | -0.00764 (0.00376) | 0.9605 [0.9510, 0.9682] | +0.0105 | 0.0395 [0.0318, 0.0490] |
| 0.10 | 0.40 | 2,000 / 2,000 | 0.00204 (0.00372) | 0.9585 [0.9488, 0.9664] | +0.0085 | 0.6385 [0.6172, 0.6593] |
| 0.30 | 0 | 1,995 / 2,000 | -0.00412 (0.00529) | 0.9469 [0.9361, 0.9559] | -0.0031 | 0.0531 [0.0441, 0.0639] |
| 0.30 | 0.40 | 1,999 / 2,000 | 0.01216 (0.00520) | 0.9605 [0.9510, 0.9682] | +0.0105 | 0.3857 [0.3646, 0.4072] |

All 8,000 planned cases were attempted. No case was interrupted, missing, or replaced, and no warning was recorded. Six cases returned no interval: five at `c=0.30, b=0` (repetitions 778, 1005, 1164, 1378, 1473) and one at `c=0.30, b=0.40` (repetition 679). Their bootstrap replicates 32, 62, 84, 90, 175 and 46, respectively, failed because the Cox score did not converge. The six cases had 50–63 observed events. The records give the failing draw and fit error, but do not provide a usable estimate or interval for those cases. **Protocol deviation:** although the advance plan required saving every completed bootstrap effect, the current failure path discarded those earlier effects and the original point estimate.

They cannot be examined from the frozen record. We keep each case failed and do not reconstruct its missing diagnostics by a new fit. Failure itself triggered the protocol's investigation rule. These results do not show whether a different fitting algorithm would resolve the six draws. A future study needs structured partial-error records before execution.

The table below reports the other prespecified diagnostics. Empirical SD is the standard deviation of returned log hazard-ratio estimates across datasets. Root mean variance is the square root of the mean reported bootstrap variance. The event and censoring fractions use all 2,000 attempted datasets in each cell. Administrative censoring is caused by the fixed follow-up end at time 12; independent censoring comes from the generated censoring time. The two components sum to the total censored fraction.

| `c`, `b` | Event fraction | Total censored (admin.; independent) | Empirical SD | Root mean variance | Mean interval width |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.10, 0 | 0.39187 | 0.60813 (0.11545; 0.49268) | 0.16794 | 0.17510 | 0.68309 |
| 0.10, 0.40 | 0.39525 | 0.60475 (0.11460; 0.49015) | 0.16627 | 0.17522 | 0.68375 |
| 0.30, 0 | 0.20848 | 0.79152 (0.01062; 0.78090) | 0.23639 | 0.24540 | 0.95513 |
| 0.30, 0.40 | 0.21082 | 0.78918 (0.01032; 0.77886) | 0.23269 | 0.24671 | 0.95993 |

For each cell, the following lower and upper bounds divide by all 2,000 planned cases and treat each failure first as neither covered nor rejected, then as covered or rejected. These are sensitivity bounds for an incomplete procedure, not intervals returned by failed fits.

| `c`, `b` | All-planned coverage bounds | All-planned rejection bounds |
| --- | ---: | ---: |
| 0.10, 0 | [0.9605, 0.9605] | [0.0395, 0.0395] |
| 0.10, 0.40 | [0.9585, 0.9585] | [0.6385, 0.6385] |
| 0.30, 0 | [0.9445, 0.9470] | [0.0530, 0.0555] |
| 0.30, 0.40 | [0.9600, 0.9605] | [0.3855, 0.3860] |

Observed coverage was about 1.05 percentage points above 95% in two cells, and their **cellwise, unadjusted** Wilson intervals narrowly exclude 0.95 at their lower endpoints. The four intervals are not simultaneous, so this is not a study-wide overcoverage finding. In the other two cells, the cellwise intervals contain 0.95. No absolute bias exceeded three Monte Carlo standard errors. The protocol's broad 0.93–0.97 coverage flag did not trigger; that band was only an investigation trigger, not a calibration acceptance rule. The null rejection rates are false-positive rates among successful fits. Under the same two-sided normal test, null coverage and rejection are complementary rather than independent checks. Nonnull rejection rates describe power in these conditions. Higher censoring coincided with wider intervals and lower nonnull rejection, but these four cells do not isolate every reason for that difference.

## Execution and audit

The frozen study used clean commit `8b080aaaf6992e3956dfee6ddfdda105bbb0e129`, an isolated Python 3.11.5 wheel with NumPy 2.0.0 and SciPy 1.14.0, and no R runtime. Arseven Slurm array `2927143` had 800 tasks of ten cases each; every task finished `COMPLETED 0:0`. The longest task took 9,212 seconds. Across tasks, elapsed times summed to 4,794,302 seconds; this is parallel task time, not wall time for the full array. Peak recorded task resident memory was 101,490,688 bytes. The frozen terminal auditor completed as separate job `2931276` with zero issues and accepted all 8,000 case IDs, 800 batch mappings, task output hashes, source/wheel identity, scheduler exits, resource bounds, input hashes, bootstrap standard errors, normal limits and p-values. A first audit wrapper attempt `2931274` failed before reading cases because it omitted the server Python module; its log was retained. The corrected wrapper loaded that module and did not rerun the study.

Raw case records are retained outside this repository. Their returned archive SHA-256 is `23ed29a5f609238edfa762c86b31acf38f55abb766368921e2f1e630f7a627ad`; the acceptance JSON SHA-256 is `e146204f0828f9dc5fcbbe16fac4124d03b22219e7bcfefa1469ae2477d3cb99`. The 31,165,061-byte archive digest matched server and local copies. Local verification matched the source bundle, wheel and every listed batch-output digest. A separate local full-case replay regenerated inputs and checked the saved inference for all 8,000 cases under Python 3.11, NumPy 2.0.0 and SciPy 1.14.0, with zero summary differences. Its macOS numerical-library build differs from the Linux server build, so this is a cross-platform check rather than an identical environment. A separate standard-library aggregation of the saved records also matched key counts, bias, variation, coverage and rejection in all four cells. These checks support the saved calculation and accounting; they do not prove interval calibration beyond the specified generator.

The fixed installed-R comparisons in [the Cox software report](COX_RERUN_20261024.md) test selected stages on shared outcomes. R qgcomp 2.19.6 and Python use different Monte Carlo event generators, so this study does not claim end-to-end numerical identity with R. The synthetic setting has independent uniform known exposure scores, homogeneous hazards after joint intervention, independent censoring, no adjustment covariates and fixed breaks. It does not assess heterogeneous or correlated mixtures, estimated exposure cuts, informative censoring, nonproportional marginal hazards, degree-two curves, clusters, confounding, causal identification or other outcome families.
