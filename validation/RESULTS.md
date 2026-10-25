# Validation record

Each report gives its design, results, failures and limits. For supported
functions and options, see the [coverage map](../docs/r-api-parity.md).

| Method | Repeated-dataset result | Main limit |
| --- | --- | --- |
| Gaussian | [24,000-dataset study](GAUSSIAN_RESULTS.md) | Known exposure scores |
| Gaussian row bootstrap | [24,000-dataset study](BOOTSTRAP_RESULTS.md) | Additive fit and fixed cutpoints |
| Binary logit and marginal ratios | [Conditional logit](BINOMIAL_CALIBRATION_RESULTS.md), [risk ratio](RISK_RATIO_Q4_CALIBRATION_RESULTS.md), [odds ratio](ODDS_RATIO_Q4_CALIBRATION_RESULTS.md) | Separate targets and study settings |
| Poisson models | [Conditional](POISSON_CALIBRATION_RESULTS.md), [unequal-time conditional](POISSON_OFFSET_CALIBRATION_RESULTS.md), [marginal bootstrap](POISSON_BOOTSTRAP_CALIBRATION_RESULTS.md), [unequal-time bootstrap](POISSON_BOOTSTRAP_OFFSET_CALIBRATION_RESULTS.md) | Some sparse-count settings may under-cover |
| Marginal Cox | [8,000-dataset study](COX_BOOTSTRAP_STUDY_RESULTS.md) | Six intervals unavailable |
| Paper scenarios | [Gaussian scenarios 1–4](PAPER_REPRODUCTION_RESULTS.md), [scenario-5 translation](PAPER_SCENARIO5_RESULTS.md) | Scenario 5 is not an exact Figure 1 replay |

Fixed-input comparisons are in the [R reference guide](../docs/reference.md).

Software delivery checks are available in [GitHub Actions](https://github.com/nilson01/quantilemix/actions).
Earlier implementation records remain in [repository history](https://github.com/nilson01/quantilemix/blob/ab1b5cbf4d071e661a2816d0ffc28f4245b51489/validation/RESULTS.md).
