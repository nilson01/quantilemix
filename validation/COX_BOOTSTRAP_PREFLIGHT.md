# Marginal Cox bootstrap calibration preflight

This is a bounded software and timing check. It is **not** an interval-coverage study.

The fixed generator creates 300 independent people with four exposure scores, each uniformly drawn from 0–3. The event hazard is `0.08 exp{0.4 (sum(scores)-6)/4}`. Independent exponential censoring has rate 0.10, and follow-up ends at time 12. Thus, increasing all four scores by one gives a true conditional log hazard ratio of 0.4. With no adjustment variables or other effect heterogeneity, the corresponding marginal Cox model has the same target. Seed 20261024 and the exact input hash are saved in the JSON results.

| Check | Settings | Result | Fit time | Peak process memory |
| --- | --- | --- | ---: | ---: |
| [Small smoke](cox-bootstrap-preflight-results.json) | `B=2`, `MCsize=300` | Passed; no warnings | 0.17 s | 99 MB measured by external process monitor |
| [Medium timing](cox-bootstrap-timing-medium.json) | `B=20`, `MCsize=1000` | Passed; no warnings | 2.01 s | 101 MB |
| [Initial default-size timing](cox-bootstrap-timing-default-mc.json) | `B=20`, `MCsize=10000` | Failed at replicate 6: Cox score did not converge | 6.82 s to failure | 116 MB |
| [Repaired default-size timing](cox-bootstrap-timing-default-mc-repaired.json) | Same settings and seed | Passed all 20 replicates; no warnings | 17.11 s | 121 MB |

The initial failure was numerical. At marginal replicate 6, the candidate Cox score fell to about `3.8e-9`, but the summed log likelihood of about `-149969` rounded down by roughly `1e-9`. The strict line search rejected the accurate Newton step. The repair permits a likelihood difference within a floating-point roundoff bound **only when the score improves**. The final score test remains unchanged. A fixed-seed regression test now reaches this replicate at the default Monte Carlo size.

Each check uses one observed dataset and cannot estimate bias, coverage or
false-positive rates. See the separate [study report](COX_BOOTSTRAP_STUDY_RESULTS.md).

The earlier [installed-R stage comparison](COX_BOOTSTRAP_RESULTS.md) remains the reference evidence for two fixed cases. Five [fresh fixed-case R checks](COX_RERUN_20261024.md) then passed for conditional Cox fits, Efron ties, conditional curves, marginal bootstrap stages and marginal survival-curve tables. These checks retain the earlier fixed-case scope; they do not estimate interval coverage.

To reproduce the bounded checks from an installed Python development environment, first run the scripts with no arguments to inspect their fixed plans. Then supply a **new** output path, for example `python validation/cox_bootstrap_preflight.py --output /tmp/cox-preflight.json` and `python validation/cox_bootstrap_timing.py --profile default_mc --output /tmp/cox-timing.json`. Neither script starts the full study.
