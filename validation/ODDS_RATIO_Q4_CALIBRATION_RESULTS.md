# Four-level marginal binary odds-ratio interval study

This report follows the [advance protocol](ODDS_RATIO_Q4_CALIBRATION_PROTOCOL.md) and the separate [20-case timing pilot](ODDS_RATIO_Q4_CALIBRATION_PILOT.md). The question is whether the package's 95% normal interval from 200 fixed-cut row resamples covers its population target across repeated datasets with four known exposure-score levels. The target is the slope of an equally weighted logistic fit to the four population intervention risks. It is not the conditional log odds contrast used to generate outcomes or the odds ratio for any one adjacent-level change.

Each of four conditions has 1,000 planned independent datasets of 500 rows. The table reports only returned intervals. `r0` is the population risk at all-zero exposure scores, averaged over the binary adjustment variable; `b` is the conditional log odds contrast from all-zero to all-three scores. The true and bias columns use the marginal **log** odds-ratio scale. MCSE is the Monte Carlo standard error of bias, reflecting variation across generated datasets. The bracketed limits are cellwise 95% Wilson intervals for the reported coverage proportions.

| `r0` | `b` | True log OR | Successful / planned | Bias (MCSE) | 95% interval coverage (Wilson limits) | Rejection rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0 | 0 | 999 / 1,000 | -0.0011 (0.0088) | 0.954 [0.939, 0.965] | 0.046 |
| 0.10 | 0.80 | 0.26465 | 1,000 / 1,000 | 0.0115 (0.0073) | 0.955 [0.940, 0.966] | 0.203 |
| 0.40 | 0 | 0 | 1,000 / 1,000 | -0.0036 (0.0052) | 0.957 [0.943, 0.968] | 0.043 |
| 0.40 | 0.80 | 0.26259 | 1,000 / 1,000 | 0.0018 (0.0050) | 0.957 [0.943, 0.968] | 0.370 |

The following planned diagnostics use the same condition order and successful-fit denominators. The event fraction uses all 1,000 attempted datasets in each condition. Empirical SD is the standard deviation of the returned log odds-ratio estimates across datasets; root mean variance is the square root of their mean reported bootstrap variance. Rejection Wilson limits describe simulation uncertainty in the rejection rate. The final two columns are sensitivity bounds among **all 1,000 planned** datasets: their lower endpoint treats a failed or missing result as neither covered nor rejected, and their upper endpoint treats it as covered or rejected. They do not assign an interval or test to a failed fit.

| `r0`, `b` | Event fraction | Empirical SD | Root mean variance | Rejection Wilson 95% limits | All-planned coverage bounds | All-planned rejection bounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10, 0 | 0.099886 | 0.27674 | 0.28245 | [0.0347, 0.0609] | [0.953, 0.954] | [0.046, 0.047] |
| 0.10, 0.80 | 0.143420 | 0.23204 | 0.23896 | [0.1792, 0.2290] | [0.955, 0.955] | [0.203, 0.203] |
| 0.40, 0 | 0.399878 | 0.16531 | 0.16641 | [0.0321, 0.0574] | [0.957, 0.957] | [0.043, 0.043] |
| 0.40, 0.80 | 0.497194 | 0.15788 | 0.16427 | [0.3406, 0.4004] | [0.957, 0.957] | [0.370, 0.370] |

All 4,000 planned datasets were attempted. There were no recorded warnings and no missing cases. The one failure occurred in the first condition, at repetition 936 with 34 events: bootstrap draw 170 had logistic separation or diverged binomial coefficients. The result retains the original fit and 169 completed bootstrap effects. That case was not replaced or counted as a returned interval. In the two null conditions, rejection is the false-positive rate; in the two nonnull conditions, it is descriptive power. Null coverage and rejection for this same two-sided normal procedure are complementary, not independent confirmations.

The predeclared descriptive review flags did not trigger: every absolute bias was below three MCSEs, each coverage Wilson interval included 0.95, and each observed coverage lay within 0.93–0.97. These checks do not prove exact 95% coverage or equivalence. They support this restricted known-score design only. Estimated cutpoints, case weights, missing rows, clustering, nonlinear outcome models and other outcome families were outside this study.

## Reproducibility and audit

The protected run used clean source commit `47589f09e56b1b66b7eceb0e617e231a6bf12c71`, an isolated Python 3.11.5 installation with NumPy 2.0.0 and SciPy 1.14.0, and no R runtime. Its fixed run ID is `e8455a2eb116202cf1179d9a853b5ca372744a9392fb5b3a5f9506488f30d0dc`. Slurm job `2930968` completed with exit `0:0` in 34 minutes 21 seconds. The runner used 1,494.660 seconds; the independent checker used 557.034 seconds. The scheduler reported 234,168 KiB maximum batch resident memory. Output occupied about 27,462 KiB.

Raw output and server logs are retained outside this repository. The saved `result.json` SHA-256 is `0fdb7fff36f2bd9f7bf52a813068bc7661c6503b5e41a96ecd8423a4c5224120`; `check.json` is `e64da593a4b7864a3be3a3e738aaba346040f8158edd279687784451d2ce5d74`; and the server `aggregate.json` is `d5e9b8f063b168e1184856c90f10353f7b6b0f28a0be6a4ec9ede767d060d637`. The manifest and all 12 listed files passed server and local SHA-256 verification. The result, append-only journal, completion receipt and persistent identity marker agree. The independent checker paired all 4,000 datasets, found the same single failed draw, and reported zero issues across 3,999 successful fits. Its maximum absolute numerical difference was `5.72e-11`. A separate standard-library aggregate matched the server summary with zero differences. These audits verify the saved calculation and case accounting; they are separate from the statistical coverage question.

The fixed installed-R comparisons in [the odds-ratio software report](ODDS_RATIO_RESULTS.md) check selected calculations under matched resamples. They do not replace this repeated-dataset study.
