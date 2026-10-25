# Four-level odds-ratio validation jobs

These Slurm jobs implement protocol `binary-odds-ratio-four-level-known-scores-v1`. They are private research checks. Python users do not need Slurm or R. The pilot checks the clean server installation and 20-case execution path; the protected study estimates interval behavior across 4,000 datasets only after that pilot and the resource gate pass.

| Job | Request | Planned work | Output guard |
| --- | --- | --- | --- |
| Pilot | `short`, one CPU, 1 GiB, five minutes | 20 datasets, 4,000 bootstrap draws, independent numerical audit and separate summary calculation | under 50 MiB |
| Study | `short`, one CPU, 4 GiB, one hour | 4,000 datasets, 800,000 draws, the same audits | under 100 MiB, at least 5 GiB home-quota headroom |

The private submission root must be named `qgcomp-binary-odds-ratio-four-level-known-scores-v1`. Before either job, create its empty `identity-registry/` and `logs/` directories. Stage `source.bundle`, its SHA-256 file, the exact revision file, a clean `code/` checkout and an isolated `python/` environment containing this package, NumPy and SciPy. Verify the bundle and installed wheel fingerprints before submission. The scripts check the revision, clean checkout, existing registry and installed import. The runner creates a persistent marker for each run identity. A failed or interrupted marker is retained for an explicit audit, never silently reset. The jobs save case journals, completion receipts, identity markers, independent audits, aggregates, logs and output hashes. Do not submit a duplicate job or overwrite an earlier attempt.

Before the protected submission, check the current queue, partition, quota, clean source, empty output directory and absence of an active or completed odds-ratio study identity. Other running cluster work is a capacity constraint. After return, verify every file digest, the completion receipt, all case IDs, the independent checker, the separate aggregate and any failure/warning before interpreting coverage. Report coverage and rejection conditional on returned intervals/tests together with failure counts and all-planned sensitivity bounds. The two fixed installed-R cases test software agreement, not repeated-data interval coverage. No R runtime is needed for either job.
