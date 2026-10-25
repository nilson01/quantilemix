# Marginal Cox bootstrap: fixed-break server pilot

The reviewed [four-cell protocol](COX_BOOTSTRAP_PROTOCOL.md) ran its **20-case feasibility pilot**, with five datasets in each cell. Each 300-row fit used four fixed score levels, 200 row-bootstrap draws and 10,000 simulated profiles per level. This pilot checks completion and resource use. Five datasets per cell cannot establish confidence-interval coverage or a false-positive rate.

| Check | Observed result |
| --- | ---: |
| Planned, completed, successful fits | 20, 20, 20 |
| Fit failures, warning fits, interrupted or missing cases | 0, 0, 0 |
| Bootstrap fits attempted | 4,000 |
| Fit time per case: minimum, median, mean, maximum | 395.52, 596.51, 596.13, 761.56 seconds |
| Total fit time across cases | 3.312 hours of summed one-thread fit time |
| Slurm task time: minimum, median, mean, maximum | 401, 602.5, 601.7, 769 seconds |
| Peak task memory across all tasks | 66,134,016 bytes (about 63.1 MiB) |
| Task outputs and scheduler logs | 445,832 bytes |

All 20 Slurm array tasks in job `2926651` ended `COMPLETED 0:0`. The terminal auditor accepted all 20 case IDs with no issues. It checked source and wheel digests, committed source, fitted levels, input hashes, bootstrap standard errors, normal intervals and p-values, task checksums, scheduler exits, memory, wall time and output bounds. The returned archive SHA-256 is `8fae26fcbcfb1d8981719a3f5e5904b0def66da436275f7add072045fb9c7760`. The frozen fitting source commit is `01bb2f14bb00feecd5ccd1a2f65137a20243bbe3`; the source bundle and wheel digests are recorded in the private compute receipt. A separate local replay under NumPy 2.5.4 regenerated all 20 inputs and checked every saved case. Its summary numbers differ from the exact-version server summary by at most `1.11e-16`, only in Wilson interval rounding.

**Audit correction.** The frozen terminal auditor requested Slurm `JobIDRaw`, which this server reports as flattened job numbers. The array-indexed `JobID` field is needed to match case IDs. Commit `e6655bff21e3d795a799fc9df228ac0ec4781f45` changes only that field and adds a guard assertion. The corrected auditor ran outside the frozen checkout and imported the original exact-version case checker. The fitting code, datasets, seeds and 20 results were not changed or rerun. The private compute record retains both versions, raw `sacct` output, task files and acceptance receipt.

The [completed study](COX_BOOTSTRAP_STUDY_RESULTS.md) reports interval
coverage and failures; this pilot measures feasibility only.
