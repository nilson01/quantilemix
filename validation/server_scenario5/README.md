# Scenario-5 Gaussian translated-study server job

Protocol `keil-2020-scenario5-qgcomp-v1` is reviewed for a descriptive, nine-exposure qgcomp-only translation of Keil et al. (2020) scenario 5. The full design has 18,000 independent fits across two generator interpretations; it is not an exact Figure 1 reproduction. The exact full source revision is recorded at launch. No full-study result is claimed in this job specification.

Use Arseven `short`, account `stat`, one CPU, no GPU. The bounded server pilot requests 1 GiB and five minutes for 90 fits. If it passes source verification, all case checks, zero unexpected warnings/failures and capacity checks, the full run requests 4 GiB and 30 minutes. The local pilot used about 100 MB peak memory and under one second end to end; the server limits provide headroom. Keep output under 100 MiB; do not reduce cell count or repetitions.

Transfer a Git bundle of the exact branch and record its SHA-256 in `CODE_BUNDLE_SHA256.txt` plus full commit in `SOURCE_REVISION.txt`. Clone the bundle to `code/` on the server and require a clean checkout at that revision. The batch scripts recheck bundle hash, commit and cleanliness before any fit. The study runner and checker compare the runner and core bytes against that commit. Retain the submitted script, preflight receipt, module/Python/dependency inventory, Slurm logs, case journal, checker report, output SHA-256 and exit code. Never resume a partial run in place; new attempts use new IDs.

The pilot later completed as job 2904090, and the full study as job 2904103
from the clean source commit recorded in
[the result report](../PAPER_SCENARIO5_RESULTS.md). Both jobs completed with
exit code zero. The result report contains the complete case counts,
independent checks, hashes and scientific limits. The post-run
`export_cross_platform_inputs.py` is an exact copy of the private five-case
export script used to audit ARM/x86 replay rounding; it was not part of the
original study bundle.
