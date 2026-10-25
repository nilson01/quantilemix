# Bernoulli-logit known-score server job

Protocol `binomial-known-scores-v1`. This study checks
conditional joint log-odds intervals for four independent known scores. It does
not assess all binary models. The [reviewed protocol](../BINOMIAL_PROTOCOL.md)
gives the design.

Use Arseven, account `stat`, partition `short`, one CPU, no GPU, one numerical
thread, 2 GiB RAM and ten minutes for 4,000 fits and their independent audit.
A separate 20-case server pilot requests 1 GiB and five minutes. Keep each
run's new output under 50 MiB and home quota headroom above 5 GiB. No sample size, cell count, seed or comparison tolerance may be changed to
meet this bound.

Transfer a Git bundle of the exact clean branch to `source.bundle`, record its
SHA-256 and the full commit, and clone it to `code/`. The batch scripts check
bundle hash, checkout commit and cleanliness before any fit. They use the
existing verified Python 3.11.5 environment with NumPy 2.0.0 and SciPy
1.14.0, record module/import versions, host, UTC time, quota, stdout and
Slurm logs, run the optional independent checker, and save output hashes.
`pilot-001` must pass before `study-001` is submitted. A checker concern
produces nonzero exit and a saved case audit. Do not resume a partial run in
place; any retry needs a new run ID and a documented reason. Verify all
returned SHA-256 values locally and interpret every cell, warning and failure.
R is not required on the server or by package users; the same-input installed
R 2.19.6 check already passed separately on the Mac.
