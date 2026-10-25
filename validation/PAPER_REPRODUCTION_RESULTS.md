# Gaussian comparison with Keil et al. (2020), Table 3

We completed protocol
`keil-2020-gaussian-scenarios-1-4-v2` on the Arseven server. This is the
qgcomp arm of **four Gaussian scenarios only**, each with 500 observations,
four already scored exposures, and 1,000 independent datasets. The
[protocol](PAPER_REPRODUCTION_PROTOCOL.md) was fixed and reviewed before the
full study. The original [paper and Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7228100/)
provide the published comparison; its [supplement](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7228100/supplementaryFiles)
defines the balanced exposure-score generator.

We report bias as mean estimated joint effect minus the true effect.
Empirical SD is the standard deviation of the 1,000 estimates; root mean
variance is the square root of their mean reported variance. Coverage is the
fraction of 95% intervals containing the truth. Rejection is the fraction
with a two-sided p-value below 0.05. Under the null scenarios, rejection is
the observed false-positive rate; under the two effect scenarios, it is
observed power. Every scenario had **1,000 successful fits and zero failures**.

| Scenario | Bias (paper) | Empirical SD (paper) | Root mean variance (paper) | Coverage (paper) | Rejection (paper) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Null | 0.0025 (0.00) | 0.0790 (0.08) | 0.0803 (0.08) | 0.946 (0.94) | 0.053 (0.06) |
| Cancellation | -0.0030 (0.00) | 0.0818 (0.09) | 0.0802 (0.09) | 0.941 (0.95) | 0.059 (0.05) |
| Single active exposure | 0.0036 (0.00) | 0.0812 (0.08) | 0.0803 (0.08) | 0.945 (0.94) | 0.891 (0.88) |
| Distributed effect | 0.0036 (0.00) | 0.0812 (0.08) | 0.0804 (0.08) | 0.941 (0.95) | 0.879 (0.86) |

The largest absolute difference in coverage is 0.009; in rejection it is
0.019. The independent checker found no descriptive divergence flag under
the protocol's rule: three combined Monte Carlo standard errors, including
both 1,000-run studies, plus 0.005 for the paper's two-decimal rounding.
For example, the distributed-effect rejection difference is 0.019, while
its combined Monte Carlo standard error is 0.0151 before rounding. Wilson
95% intervals for our coverage rates range from 0.925–0.954 to
0.930–0.958 across the four scenarios.

Those intervals describe our
repeated-dataset uncertainty; they do not establish equivalence with the
paper. The paper's “MCSE” column is the across-dataset empirical SD, not the
standard error of its reported bias.
For cancellation, our empirical SD and root mean variance are about 0.082
and 0.080, while both published rounded entries are 0.09. In particular,
the root mean variance is below even the paper value's rounding interval
(0.085–0.095). The new study's Monte Carlo standard error for that measure
is about 0.0001, so sampling variation in our study alone does not explain
the difference.

The source of this disagreement remains unresolved. The
protocol's divergence rule was defined for bias, coverage and rejection;
this dispersion measure has **not** been reproduced.
As a limited source check, the released `qgcomp_analysis.R` prints mean
`var.qgc = 0.006380424` for a later **five-run** cancellation illustration
using qgcomp 2.1.3. Its square root is about 0.0799, close to our 0.0802.
Five runs do not identify the cause of the published 1,000-run value.
The original qgcomp 1.3 calculation and production settings need a separate
provenance check before resolving the difference.

Bias, coverage and rejection are close in these cells under the stated
descriptive rule. The cancellation dispersion difference limits the
reproduction claim.
The paper used R qgcomp 1.3 and a different random stream, while Python
follows tested R qgcomp 2.19.6 Gaussian conventions. An optional, separate
same-input pilot compared 20/20 Python fits with **installed** R qgcomp
2.19.6; the largest effect-estimate or variance difference was 2.99e-15.
A [post-study check](PAPER_R13_PROVENANCE.md) then installed qgcomp 1.3.0
in a separate R library. On the same 20 fixed inputs, its estimates and
variances equal 2.19.6's values exactly at saved precision. This narrows a
version explanation for these inputs, but it does not resolve the published
cancellation dispersion. These checks establish local software agreement,
not coverage. We have not
reproduced the paper's WQS arm, its other sample sizes or exposure counts,
or its nonlinear and non-Gaussian scenarios. These results do not establish
full paper or R-package coverage.

## Execution and integrity record

The full study ran as Arseven Slurm job `2901441`:
one CPU, 4 GiB requested memory, 30-minute limit, one numerical thread,
12 seconds elapsed, exit `0:0`. The job verified the transferred Git archive
SHA-256 `af6779f7e82d2d1e9d719be7f7f42ebcfc72fc5a8db6be2d92adedc2f0a21ee0`,
all 238 extracted files, and the imported Gaussian core path and hash before
fitting. Its source revision was
`41f11f006be8b94761c3a516abc46907070b6c4a`. Scratch quota was
185/200 GiB used; the job required at least 10 GiB headroom. The output
directory used 1,305 KiB in server accounting. The full case journal was
flushed after each fit.

The read-only checker regenerated all 4,000 input hashes, reconciled the
journal and summaries, and independently recomputed **all 4,000** estimates,
variances, intervals and p-values by QR least squares. Its largest absolute
numerical difference was `3.91e-14`. It found zero failures. The final
result SHA-256 is
`f129f8a36089251ab1756c0a60a132a593b4ccdc4e381bee6c4e7d7f68d6739c`;
the journal SHA-256 is
`0f212496a9f6580dadd9a967abf71b450672dc273c8888549388b5bb3ea6e83e`.

To inspect a saved result, run the read-only checker from the matching source
revision and installed Python dependencies:

```sh
python validation/check_paper_results.py /path/to/result.json
```

The generator, study runner and checker are Python-only. R is an optional
reference check and is not a package runtime dependency.
