# Conditional Poisson exposure-time offset study job

Protocol `poisson-offset-known-scores-v1`.
The question is whether model-based intervals for the conditional joint log
rate ratio behave as intended when observation times vary. The design is in
`../POISSON_OFFSET_PROTOCOL.md`; the local timing evidence and seed repair
are in `../POISSON_OFFSET_PILOT_RESULTS.md`. This is a bounded software check,
not a full reproduction of the source paper.

Use the configured private Arseven Slurm account `stat`, partition `short`,
one CPU, no GPU, one numerical thread and at most 50 MiB output per attempt.
The server pilot requests 1 GiB and five minutes; the 4,000-case study requests
2 GiB and ten minutes. The inputs are synthetic and generated on the server
from the fixed protocol seed. The clean source Git bundle, result, audit and
logs remain in a new private home directory. Confirm at least 5 GiB home
quota headroom, a working short partition, the existing Python environment,
and no duplicate qgcomp job immediately before each submission.

Stage the exact clean commit as `source.bundle`, save its SHA-256 in
`CODE_BUNDLE_SHA256.txt` and its full revision in `SOURCE_REVISION.txt`, then
clone the bundle to `code/`. The scripts check these identities and a clean
checkout before fitting. They record CPU and memory allocation, module and
Python/dependency inventories, host and UTC times, exact command through the
saved script, case results, independent audit, output hashes and exit code.
The server pilot must exit zero with all 20 cases paired before the study is
submitted. Check quota and duplicate jobs again between pilot and study.
Never resume a partial attempt in place; audit its journal, retain the
interruption reason, and use a new attempt ID for any justified retry.
After result return, verify all hashes, all planned case identities, warnings,
failures and cell metrics before interpreting interval behavior. The earlier
two-case installed-R comparison is separate numerical compatibility evidence.
Python users do not need R.
