# Four-level marginal binary risk-ratio interval study

This [reviewed study](RISK_RATIO_Q4_CALIBRATION_PROTOCOL.md) tests the 200-resample normal interval for the fitted common log risk ratio across four known joint exposure levels. The separate [fixed-data comparisons](RISK_RATIO_MULTILEVEL_RESULTS.md) show agreement with installed R qgcomp 2.19.6; agreement does not establish repeated-data coverage.

Each of four conditions used 1,000 independently generated datasets of 500 rows. Four known four-level exposure scores, one binary adjustment covariate, and a correctly specified logistic outcome model were used. The procedure standardized event risks at joint scores 0, 1, 2, and 3, then fitted a degree-one log-risk model with equal level weights. Its exponentiated slope is a fitted *common per-level ratio*, not an exact ratio for each adjacent level. The true target was computed from the population risks under this same fitting rule. `r0` is the all-zero risk; `b` is the generating model's joint conditional log-odds contrast. Neither is itself the fitted marginal log risk ratio.

All 4,000 planned datasets were recorded. One interval was unavailable in the low-risk null condition because bootstrap draw 49 separated; the package and independent checker failed on the same draw. No case or draw was replaced. The table reports coverage among successful intervals and keeps the failure visible. Bias is the average estimate minus the known target; MCSE is its Monte Carlo standard error. Brackets are separate, cellwise 95% Wilson intervals, not a simultaneous statement across the four cells. Rejection is the two-sided normal test: a false-positive rate in null cells and descriptive power otherwise.

| `r0` | `b` | True log ratio | Success / fail / missing | Bias (MCSE) | Coverage [Wilson 95%] | Rejection [Wilson 95%] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| .10 | 0 | 0 | 999 / 1 / 0 | +.00066 (.00777) | .93794 [.92123, .95129] | .06206 [.04871, .07877] |
| .10 | .8 | .22565 | 1000 / 0 / 0 | +.00152 (.00630) | .942 [.92575, .95487] | .213 [.18875, .23945] |
| .40 | 0 | 0 | 1000 / 0 / 0 | +.00221 (.00311) | .937 [.92021, .95045] | .063 [.04955, .07979] |
| .40 | .8 | .13026 | 1000 / 0 / 0 | +.00151 (.00250) | .937 [.92021, .95045] | .419 [.38879, .44983] |

All four observed coverages are below .95. Each cellwise Wilson interval still includes .95, three only narrowly. Thus these data permit a modest common shortfall; they do not establish exact 95% coverage or a clear departure in any one cell. The two null-cell rejection rates are complements of the corresponding coverage counts for this same test, not independent confirmation. For the failed cell, all-planned coverage is bounded by .937–.938 and the all-planned false-positive fraction by .062–.063, depending on its unknown interval result. No bias exceeds three MCSEs. No coverage Wilson interval lies wholly outside the protocol's descriptive .93–.97 review band, which is not an equivalence margin. The single separation failure warrants follow-up in sparse-event settings.

The empirical standard deviations were .24574, .19923, .09833, and .07918 in table order. Square roots of mean reported variances were .25097, .19881, .09777, and .07742. These describe the studied design, without comparing another method.

## Audit and provenance

The reviewed 20-case local and clean-source server pilots passed first. The full clean-source Slurm job `2917185` completed `0:0` in 27 minutes 52 seconds from source commit `2c080ac3b6e83fabf0a42462c3bc9bc45ccfc8eb`. It used Python 3.11.5, NumPy 2.0.0, SciPy 1.14.0, one CPU, 4 GiB requested memory, and one numerical thread. The runner took 1,217.976 seconds; the independent all-case checker took 446.243 seconds. The checker verified 4,000 case identities, 3,999 paired fits, the matching separation failure, and zero unexplained issues. Its maximum absolute numerical difference was `6.625e-11`. A separate standard-library aggregation reproduced the summaries with zero differences. All returned server files passed local SHA-256 checks.

The compact [machine-readable summary](risk-ratio-q4-calibration-study-summary.json) gives full-precision measures, denominators, versions, and digests. The full raw result SHA-256 is `a76119d3539dbafe7747186b9fa4a244ac745dff191c7f24ed658d10be9305bd`; the independent audit is `1d16142a19011d950c243c5ea9127911138f6900f0b11ce43d388cfa9ceeb68d`. Exact case-level files remain in private project validation storage, outside the PR. A reader can rerun the committed protocol and code, but cannot inspect this exact raw run from the PR alone.

## Limits

The result covers known scores 0–3, independent rows, one binary adjustment covariate, four correctly specified logistic conditions, and this 200-resample normal interval. It does not assess estimated cutpoints, weights, missing data, clusters, other interval types, or other outcome families. No causal effect or general R-package equivalence is claimed. The earlier [two-level study](RISK_RATIO_CALIBRATION_RESULTS.md) estimates a different two-level target and must be read separately.
