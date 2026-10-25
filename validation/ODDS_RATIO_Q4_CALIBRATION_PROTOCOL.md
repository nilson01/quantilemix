# Four-level marginal binary odds-ratio validation protocol

Protocol `binary-odds-ratio-four-level-known-scores-v1`, prepared before the study. This is a private advance plan, not a public registration. It asks whether the implemented 200-resample normal interval covers its population target when each exposure has four known score levels. Two fixed datasets already agree with installed R qgcomp 2.19.6 under matched bootstrap schedules. That software agreement does not establish interval coverage across datasets.

## Target and design

Generate four independent exposure scores `Q1` through `Q4`, each uniform on `{0,1,2,3}`, and an independent binary adjustment variable `Z` with probability 0.5. Each dataset has 500 independent rows. Generate a binary outcome with probability

    expit(a + b * (Q1 + Q2 + Q3 + Q4)/12 + 0.5*Z),

where `expit(t)=1/(1+exp(-t))`. Choose `a` so the population risk at all-zero scores, averaged over `Z`, is `r0`. The conditional log odds contrast between all-zero and all-three scores is `b`. It is generally different from the marginal slope being assessed.

Under joint intervention level `s` in `{0,1,2,3}`, set all four scores to `s`. The population standardized risk is `R(s)=[expit(a+b*s/3)+expit(a+b*s/3+0.5)]/2`. The true effect is the slope of a degree-one logistic marginal structural model fitted with equal weight to the four fractional risks `R(s)`. Its two score equations are `sum_s [R(s)-expit(c+d*s)]=0` and `sum_s s*[R(s)-expit(c+d*s)]=0`; `d` is the target log odds ratio. Thus `exp(d)` is a fitted summary across four joint levels. It need not equal the marginal odds ratio for any particular adjacent-level change. It is generally not the conditional contrast `b/3` or the endpoint log-odds contrast divided by three. The checker derives `a` independently and solves the population equations separately from the runner.

| Factor | Fixed values |
| --- | --- |
| All-zero intervention risk `r0` | 0.10, 0.40 |
| Joint conditional log odds contrast `b` | 0, 0.80 |
| Independent datasets per cell | 5 timing-pilot cases; 1,000 protected-study cases |
| Bootstrap | 200 fixed-cut independent-row resamples per dataset |
| Protected total | 4,000 datasets; 800,000 bootstrap draws |
| Random streams | NumPy PCG64, separate `SeedSequence` namespaces for data and bootstrap; master seed 2026102402 |
| Test and interval | Two-sided normal test and 95% normal interval on the marginal log-odds scale |

The four cells are ordered by `r0`, then `b`. Call `fit_binomial_odds_ratio(y, Q, Z, q=None, B=200, seed=...)`. Scores are already integers; no cutpoints are estimated. Do not replace failed cases or draws. The pilot and protected study have different random streams. The pilot is for timing, resource use and software audit, not for choosing a favorable study condition or making a coverage claim.

## Audit and reporting

The runner records an input hash, bootstrap seed, event count, fit or named failure, warnings and every completed bootstrap effect. If a draw fails, it retains the original estimate, original standardized risks, completed effects and failed-draw number; the case remains failed. An append-only journal records case starts and terminals. A run identity binds the plan and full package-source hashes. The protected study requires one authoritative identity-registry directory shared by all output locations. Its active marker remains after interruption; its completed marker remains after completion. A new attempt with that identity is refused even in another output directory. Recovery requires an explicit audit and a documented new plan or source identity. A complete result has a matching journal and receipt. An interrupted journal remains inspectable but is not a completed study.

The checker regenerates all cases and row schedules. A separate reference generator checks the exposure, covariate and outcome arrays and bootstrap seeds for every case; fixed first/last study-case vectors guard against a shared change to both generators. The checker independently solves the conditional logistic equations, standardizes risks, solves the logistic marginal equations, and recomputes standard errors, intervals and p-values for each original fit and resample. It verifies population truths by separate baseline-risk algebra, all identities, hashes, warnings, partial failed-draw records and summaries. The two generators still use the same NumPy random-stream specification, so exact replay requires the recorded NumPy version. A separated logistic fit is a failure, even if a numerical root has a small score residual.

Report each cell's planned, attempted, successful, failed and missing cases; event fraction; bias and its Monte Carlo standard error (MCSE); empirical estimate standard deviation; root mean reported bootstrap variance; coverage and rejection rates conditional on a returned interval/test with cellwise Wilson 95% intervals; and all-planned sensitivity bounds for both rates. The lower coverage bound is the fraction of all planned cases that both returned an interval and covered truth. The upper bound treats every failed or missing case as covered; it is not the coverage probability of a procedure that returned no interval. Apply the same sensitivity logic to rejection. At 1,000 successful cases, nominal 95% coverage has approximate MCSE 0.0069. Null rejection is the false-positive rate; nonzero rejection is descriptive power. Null coverage and rejection for this same two-sided normal test are complementary, not independent findings.

Investigate any warning, missing or failed case, numerical mismatch, bias beyond three MCSEs, coverage Wilson interval excluding 0.95, or one wholly outside the descriptive 0.93–0.97 band. These are review flags, not pass criteria or an equivalence test. Preserve adverse findings and any amendment after the pilot. No causal interpretation is claimed.

## Compute and scope

A 20-case timing pilot may run locally with one numerical thread after a fresh resource check, below 2 GiB memory, ten minutes and 100 MiB new output. The protected study is `SERVER_CPU`; it needs protocol review, a successful pilot, clean committed source, current quota and queue checks, a bounded Slurm job specification, and confirmation that the run identity is not already active or complete in the shared registry. Install the exact commit in an isolated environment with `python -m pip install -e .`; record the installed version, loaded package origin, complete Python-source hashes, NumPy and SciPy versions, and `pyproject.toml` hash.

From that same checkout, `python validation/odds_ratio_q4_calibration.py --mode pilot --output <new-path>` creates a small replay; `python validation/check_odds_ratio_q4_calibration.py <new-path> --output <new-audit-path>` audits it. The protected run also requires `--identity-registry <shared-existing-directory>`. Raw cases and server logs remain in private project storage. The Python package has no R runtime dependency. This narrow study uses independent, exchangeable scores with an additive conditional logit; it does not validate estimated exposure cuts, weights, missing rows, clusters, nonlinear response models or other outcome families.
