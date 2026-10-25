# Cox bootstrap calibration server jobs

This is the server job specification for the completed
[8,000-case study](../COX_BOOTSTRAP_STUDY_RESULTS.md). The
[protocol](../COX_BOOTSTRAP_PROTOCOL.md), [20-case pilot](../COX_BOOTSTRAP_PILOT_RESULTS.md)
record the design and resource checks. The instructions below describe the submitted workflow;
the study report gives the observed results and deviations.

| Stage | Matrix | Resource limits | Result |
| --- | --- | --- | --- |
| Pilot | Twenty array tasks, one case per task; five repetitions in each of four cells | One CPU and one numerical thread per task; 2 GiB; 45 minutes per task; four concurrent tasks at most; under 2 MiB per task and 50 MiB aggregate task/log output | Complete; 20/20 cases and terminal audit passed |
| Full study | Four cells × 2,000 repetitions; 800 tasks with ten sequential cases each | One CPU and numerical thread; 2 GiB; four hours per task; initially 16 concurrent tasks; under 2 MiB per task and 500 MiB aggregate task/log output | Completed; see the study report for actual outcomes |

The four-level pilot took 395.52–761.56 fit seconds per case; its mean gives a rough 1,325 CPU-hour full-study estimate before scheduling or contingency. The pilot's 45-minute per-case limit was not reached. A future study allocation must allow for slower cases rather than trimming `B` or `MCsize`.

Stage a private `source.bundle`, `CODE_BUNDLE_SHA256.txt`, `SOURCE_REVISION.txt`, a clean `code` checkout, the built wheel and `WHEEL_SHA256.txt`, a fresh wheel installation in `python`, and `logs/`. The installation must include NumPy 2.x, SciPy 1.14 or newer, and no R runtime. Record exact versions, package distribution RECORD digest, imported module paths and hashes, Python executable, BLAS build, thread settings, command, bundle and wheel digests, node, Slurm job ID, timestamps, peak RSS, exit code and quota snapshot. The pilot runner checks that every imported Python package module comes from the installed wheel and matches the committed source before generating a case. The source bundle and built wheel must be retained privately with the results.

Each pilot array task writes a new `pilot-case-<id>/result.json`; any interruption leaves `.partial.jsonl`. Array index `i` maps to cell `i // 5` and repetition `i % 5`. Keep Slurm stdout/stderr, the runner exit code and start/end epochs. The runner records each fit's elapsed time; obtain peak RSS from terminal Slurm accounting (`sacct`), since the server image does not provide `/usr/bin/time`. A failed or warning fit, failed arithmetic check, or wrapper error exits nonzero and calls `scancel` on the whole pilot array. Other tasks may already be running; retain their interrupted journals and terminal Slurm records. Do not replace any failed case under the same pilot ID.

After all tasks reach terminal state, use the exact staged Python/wheel environment to run `"$ROOT/python/bin/python" "$ROOT/code/validation/server_cox_bootstrap/audit_pilot.py" --root "$ROOT" --job "$JOB_ID" --output "$ROOT/pilot-acceptance.json"` from outside `code`. The auditor checks exactly 20 files and case IDs, independent bootstrap arithmetic, success and warning counts, every task checksum, source/wheel digests, clean revision, terminal Slurm state and exit, 2 GiB maximum RSS, 45-minute wall limit and 50 MiB aggregate task/log output. It writes an explicit accepted/failed receipt and preserves raw `sacct` text.

If it cannot prove a bound, it fails. For an incomplete attempt, use the normal checker on all existing complete files and partial journals for a diagnostic report; never label that report an accepted pilot. Report elapsed time, peak RSS and output bytes per case and cell. On return, compare the recorded commit and bundle/wheel hashes with the staged artifacts and rerun the exact-version checker before interpreting anything. Do not interpret interval coverage from the pilot.

The completed pilot used frozen source `01bb2f1` and a separately corrected `JobID` scheduler auditor; [the results note](../COX_BOOTSTRAP_PILOT_RESULTS.md) records that one-field audit correction and the original exact-version case check.

The pilot Slurm script is [pilot.sbatch](pilot.sbatch); the terminal acceptance command is [audit_pilot.py](audit_pilot.py). The full-study job is [study.sbatch](study.sbatch), with [audit_study.py](audit_study.py) for terminal acceptance.
