# Paper reproduction pilot: Gaussian scenarios 1–4

Protocol v2 replaced v1 after a formal protocol review,
before any full-study results existed. This pilot checks the source translation, runtime and
same-input software agreement for the [prospective protocol](PAPER_REPRODUCTION_PROTOCOL.md).
It is **not** a reproduction of the paper's findings. Five datasets per
scenario give very wide uncertainty intervals for coverage and rejection.

The source is the [Keil et al. (2020) supplement](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7228100/supplementaryFiles).
The outer and inner archive hashes are fixed in the protocol and runner.
The supplement's independent, balanced exposure scores differ from the
normal-cut scores in our earlier 24,000-dataset Gaussian study.

| Scenario | Successful / planned | Bias | Coverage | Rejection |
| --- | ---: | ---: | ---: | ---: |
| Null | 5 / 5 | -0.0141 | 5 / 5 | 0 / 5 |
| Cancellation | 5 / 5 | -0.0082 | 5 / 5 | 0 / 5 |
| Single exposure | 5 / 5 | -0.0657 | 4 / 5 | 3 / 5 |
| Distributed effect | 5 / 5 | -0.0001 | 5 / 5 | 4 / 5 |

All 20 v2 fits succeeded. Fitting and summary construction took 0.020 seconds
on the local M3 Pro using one numerical thread. The saved result is
11,038 bytes and its flushed case journal is 5,783 bytes. The independent
checker regenerated and verified all 20 input hashes, case identities,
denominators and summaries. It also recomputed all 20 fits through independent
QR least squares; the largest numerical difference was 1.11e-14. For a 5/5
coverage count, the Wilson 95% interval is about 0.57–1.00.

These rates
cannot support a statistical performance claim.
The v2 result SHA-256 is
`305308bb62ed80cebd42958b486028312180b681769d2d97ca0d508bd21e72ef`;
its journal SHA-256 is
`c440f0ff166039509622a7ae301b60a138e11e4153b505ee24f2d6fa7a3918dd`.
An additional 20-case resource pass after the review repairs took 0.65 seconds
end to end (including Python startup and output), with 98,779,136 bytes
maximum resident memory measured by macOS `/usr/bin/time -l`. Its fit loop
took 0.026 seconds, output sizes were again 11,038 and 5,783 bytes, and
the independent checker passed all 20 cases. This bounds a small local pilot;
these local timings do not measure server throughput.

The optional comparison passed all 20 **identical datasets** against isolated,
installed R qgcomp 2.19.6 on R 4.5.1. The largest absolute Python/R
difference across effect estimates and their variances was 2.99e-15; the
predeclared check uses `atol=1e-10, rtol=1e-8`. This is a software-agreement
check, separate from the paper's qgcomp 1.3 result and from interval coverage.
The revised R report retains all 20 case IDs, input hashes, paired values,
differences and installed-package artifact MD5 values. Its SHA-256 is
`fc3c3c84b9090c65a386db7b7af63904aaee7c0583ce949f0dd40f55f3ddadff`.
The saved v2 pilot, earlier v1 pilot and R check are in the project's private
`validation-local/paper-reproduction-20261023/` directory; the repository
contains the deterministic generator and independent checker.

To repeat the optional installed-R check:

```sh
python validation/check_paper_r.py \
  --rscript /path/to/Rscript --r-library /path/to/isolated-r-library \
  --output /path/to/fresh/report.json
```

Python package users do not need R. A later server pilot and full study completed; see the
[study results](PAPER_REPRODUCTION_RESULTS.md). The 20-fit local pilot is
still an implementation check and is excluded from those results.
