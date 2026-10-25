# Four-level marginal binary risk-ratio validation protocol

Protocol `binary-risk-ratio-four-level-known-scores-v1`, prepared before the study. This is a private advance plan, not a public registration. It asks whether the implemented 200-resample normal interval covers its population target when every exposure has four known score levels. Agreement on fixed datasets with installed R qgcomp 2.19.6 is already recorded separately in [RISK_RATIO_MULTILEVEL_RESULTS.md](RISK_RATIO_MULTILEVEL_RESULTS.md). Matching R does not establish interval coverage.

## Target and design

Generate four independent exposure scores `Q1` through `Q4`, each uniformly distributed on `{0,1,2,3}`, and an independent binary adjustment variable `Z` with probability 0.5. Each independent dataset has 500 rows. The outcome is Bernoulli with probability

    expit(a + b * (Q1 + Q2 + Q3 + Q4)/12 + 0.5*Z),

where `expit(t)=1/(1+exp(-t))`. Choose `a` so that the population risk when all scores are zero, averaged over `Z`, equals `r0`. The joint change from all-zero scores to all-three scores has conditional log odds contrast `b`. It is generally different from the reported marginal effect.

For joint intervention level `s` in `{0,1,2,3}`, set every exposure score to `s`. The population standardized risk is `R(s)=[expit(a+b*s/3)+expit(a+b*s/3+0.5)]/2`. The true effect is the slope of the degree-one log-binomial marginal structural model fitted to the four fractional risks `R(s)` with equal weight for each level. It is **not** `log[R(3)/R(0)]/3` in general. The checker solves the population score equations independently of the runner. Fitted risks standardize over the observed `Z` rows, and row resampling includes those rows. This evaluates a population target under random `Z` sampling. No causal claim is made.

| Factor | Fixed values |
| --- | --- |
| All-zero intervention risk `r0` | 0.10, 0.40 |
| Joint conditional log odds contrast `b` | 0, 0.80 |
| Independent datasets per cell | 5 pilot; 1,000 protected study |
| Bootstrap | 200 fixed-cut independent-row resamples per dataset |
| Total protected study | 4,000 datasets; 800,000 bootstrap draws |
| Random streams | NumPy PCG64, separate `SeedSequence` namespaces for data and bootstrap |
| Test and interval | Two-sided normal test, 95% normal interval for the marginal log ratio |

The four cells are ordered by `r0`, then `b`. Run the public `fit_binomial_risk_ratio(y, Q, Z, q=None, B=200, seed=...)`. All inputs are already integer scores; no cutpoints are estimated. Do not change the design or replace a failed dataset or bootstrap draw after observing results.

## Audit and reporting

The runner records each input hash, bootstrap seed, event count, original result or failure, warnings, and every bootstrap effect. Its append-only journal records case starts and terminals. A unique run identity binds the plan, committed revision and source hashes. It refuses an existing output or active run identity, publishes without replacement, and writes a completion receipt. Partial journals remain inspectable but do not count as completed studies; an interrupted run is never silently resumed under the same identity.

The checker regenerates every dataset and bootstrap schedule. It solves the conditional logistic score equations independently, then solves the four-level log-binomial score equations for each original fit and every resample. It recomputes standard errors, intervals, p-values and cell summaries, checks the receipt and journal, and checks all four population truths by a separate baseline-risk algebra calculation. Runner and checker share the case generator and summary code, so the audit is not a fully independent implementation of the design. Exact random-stream replay requires the recorded NumPy version. A separated conditional fit is a failure, including when a numerical root has a tiny score residual but no finite maximum-likelihood solution.

Report each cell's planned, attempted, successful, failed and missing cases; observed event fraction; bias and its Monte Carlo standard error (MCSE); empirical estimate standard deviation; root mean reported bootstrap variance; successful-fit interval coverage and rejection rates with cellwise Wilson 95% intervals; and all-planned worst/best coverage bounds. At 1,000 successful fits, nominal 95% coverage has approximate MCSE 0.0069. Null rejection is the false-positive rate; nonzero rejection is descriptive power. In null cells, coverage and rejection are complements of the same two-sided test, not independent findings.

Investigate warnings, failures, missing cases, numerical mismatches, bias above three MCSEs, a coverage Wilson interval excluding 0.95, or one wholly outside the descriptive 0.93–0.97 band. These flags are not pass criteria or an equivalence test. If a Wilson interval includes 0.95, report only that no departure was detected in that cell; do not call the method calibrated in general. Preserve adverse findings and any post-pilot amendment explicitly.

## Compute and scope

A 20-case timing pilot may run locally with one numerical thread after a fresh resource check, below 2 GiB memory, ten minutes and 100 MiB new output. The protected study is `SERVER_CPU`; it needs a protocol review, successful pilot, clean committed source, fresh quota and queue checks, a bounded Slurm specification, and an inactive duplicate identity. Keep raw case records and server logs in private project storage. The package has no R runtime dependency. This study does not validate raw-exposure quantization, weights, missing values, clusters, nonlinear exposure response, other intervals or other outcome families. The earlier two-level study remains separate.
