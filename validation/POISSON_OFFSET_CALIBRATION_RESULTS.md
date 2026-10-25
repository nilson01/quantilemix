# Conditional Poisson offset known-score calibration

We assessed the implemented conditional Poisson-log
coefficient-sum interval when observation times differ. The advance design is
in `POISSON_OFFSET_PROTOCOL.md`; the compact machine-readable result is
`poisson-offset-calibration-study-summary.json`. This is a four-condition
software study, not a reproduction of the full source paper.

Each of 4,000 independent datasets has 500 rows, four independent exposure
scores from {0, 1, 2, 3}, and observation time drawn independently from
{0.25, 0.5, 1, 2} in an arbitrary time unit. Let `S` be the score sum, `t`
the observation time, `m` the rate at the hypothetical center score vector
(1.5, 1.5, 1.5, 1.5), and `b` the true sum of four equal score coefficients.
We generated counts with mean `t * exp(log(m) + b*(S-6)/4)`.

We then fit all
four saved scores (`q=None`) with `log(t)` as a fixed-coefficient offset.
Increasing each score by one within the model changes the conditional log
rate by `b`; at equal observation time, the conditional rate ratio is
`exp(b)`. The joint shift at score 3 extends beyond the generated 0–3 score
range, so the contrast is model-defined at that boundary. The null assigns
all four component effects zero; the nonzero condition assigns each effect
0.10.

This design does not test cancellation between positive and negative
component effects.

| Center rate `m` | True `b` | Mean count | Zero fraction | Mean time | Bias (MCSE) | Estimate SD / root mean variance | 95% interval coverage [Wilson 95%] | Rejection [Wilson 95%] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0 | 0.1879 | 0.8360 | 0.9380 | -0.0095 (0.0060) | 0.1893 / 0.1875 | 0.947 [0.931, 0.959] | 0.053 [0.041, 0.069] |
| 0.20 | 0.40 | 0.1928 | 0.8334 | 0.9381 | +0.0070 (0.0058) | 0.1822 / 0.1865 | 0.962 [0.948, 0.972] | 0.587 [0.556, 0.617] |
| 2.00 | 0 | 1.8843 | 0.2803 | 0.9400 | -0.0016 (0.0019) | 0.0594 / 0.0588 | 0.941 [0.925, 0.954] | 0.059 [0.046, 0.075] |
| 2.00 | 0.40 | 1.9237 | 0.2845 | 0.9377 | +0.0005 (0.0018) | 0.0585 / 0.0585 | 0.948 [0.932, 0.960] | 1.000 [0.996, 1.000] |

Every row has 1,000 planned, attempted and successful fits, with zero failed
or missing cases and zero captured warnings. Coverage and rejection therefore
use 1,000 successful fits per row, and the best/worst all-planned coverage
bounds equal the displayed fractions. Bias is the average estimated `b` minus
true `b`; MCSE is its Monte Carlo standard error across datasets. The Wilson
interval describes uncertainty in a coverage or rejection fraction from
1,000 repeated datasets. These fractions average over newly drawn scores,
times and counts; they are not coverage guarantees for every fixed score/time
configuration. The center rate `m` is not the observed mean count because
observation times and scores vary.

All four coverage Wilson intervals include 0.95. None lies wholly outside
the prespecified [0.93, 0.97] investigation band. This means no screening
flag was triggered; it does not establish equivalence to 95% or acceptable
coverage under other conditions. The lowest observed coverage was 0.941 in
the higher-rate null cell; its false-positive rejection fraction was 0.059.
The lower-rate nonzero cell had 0.962 coverage and 0.587 rejection, while the
higher-rate nonzero cell rejected in all 1,000 observed datasets. The latter
fraction has a Wilson lower bound of 0.996, so it is not a claim of certain
future rejection. At the null, rejection and noncoverage are complements for
this Wald test, not independent confirmations. None of the four observed
biases exceeded three of its MCSEs. Zero failures in 1,000 draws per cell
still allows a failure probability of about 0.003 at the one-sided 95% upper
binomial bound under this generator.

## Software checks and provenance

The independent checker reconstructed every input from the written PCG64
seed schedule, including observation time and counts, and solved the Poisson
score equations separately. It paired all 4,000 package fits with zero
mismatches or issues. The largest absolute difference across estimates,
variances, interval endpoints and p-values was 2.47e-11. A separate
read-only check of returned case rows reproduced every compact cell metric.
Two fixed cases agree with isolated installed R qgcomp 2.19.6, but those
cases are not the 4,000 simulated datasets and do not establish interval
behavior.

The clean source commit was `3770cb0039f93d95872fe748285fa28d53536f82`.
The server used Python 3.11.5, NumPy 2.0.0 and SciPy 1.14.0. Slurm pilot
2911176 and study 2911177 each finished 0:0; the study requested one CPU,
2 GiB of memory, ten minutes and no GPU. Runner and checker elapsed times
inside the study were 8.83 and 3.91 seconds. The retained server job script
records exact commands, thread controls, CPU and memory allocation, module
and dependency inventory, start/end times and exit code.

All nine returned
output hashes matched their server manifest. Private case-level evidence is
under `../validation-local/poisson-offset-calibration-20261023/study-001/`:
`result.json` has SHA-256
`4d031dc720fb084a1cbd9d82c8ed2501a607ede6d9585d34158fb55f8c7df021`;
`check.json` has SHA-256
`30faa254342fe0e06dec8667cecb58f9678cf807060e8af8ecd928a005472f4a`.
The complete raw records stay private and are not in the package distribution.
A separate read-only replay from a clean local checkout of that same commit
also passed 4,000/4,000 cases under Python 3.14.3, NumPy 2.5.4 and SciPy
1.18.1. The private `local-replay.sh` script (SHA-256
`0e271d51b4365c1863e189d2355e64cf2f63b1c192bacf5a6c39e68a39b3e099`)
and its exit-zero receipt (SHA-256
`f027f0592a166438714f34b5e2b9fef39c386af1e88bc96ae9ace8269bad3464`)
record the exact command, source, environment and output.

This is a second
execution of the same checker, not a new statistical method or dataset.

The earlier equal-observation-period Poisson study is separate. Its low-rate
null coverage was 0.934, compared with 0.947 here, but the two studies use
different datasets and different time/count distributions. They cannot show
that adding an offset improved coverage or resolved the earlier sparse-null
concern. This study covers independent known scores and times, equal positive
component effects, a correctly specified additive Poisson rate and nominal
normal intervals. It does not assess estimated cutpoints, correlated or
mixed-sign exposures, confounding, overdispersion, time associated with
exposure, missingness, model error, marginal effects or causal identification.
It does not establish broad paper or R-package coverage.
