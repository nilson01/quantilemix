# Gaussian bootstrap validation: timing pilot

This is the 20-dataset timing and integrity pilot for the
[separate bootstrap protocol](BOOTSTRAP_PROTOCOL.md). It is **not** the
24,000-dataset calibration study. The [machine-readable pilot record](bootstrap-calibration-pilot-results.json)
preserves the run identity, source hashes, environment and observed resources.
Raw case records and process observations remain in the private development
directory `validation-local/bootstrap-calibration-20261023` outside this
source distribution. No prior Gaussian study dataset or outcome was reused.

The pilot used 20 newly generated datasets, each with 500 rows, residual
latent exposure correlation 0.75, and a true joint effect of 0.25. It used
200 row bootstrap resamples per dataset: 4,000 bootstrap fits in all. The
model-based Gaussian and independent QR comparisons ran on the same data.
All 20 cases reached a terminal record; all 60 method results succeeded.
The read-only [saved-record checker](check_bootstrap_calibration.py) verified
all 20 cases, the saved inputs and batch hash, and the reported counts and
key statistics. No R process or R package is needed for this runner. The two
fixed installed-R bootstrap comparisons remain the separate reference check.

| Pilot measure | Observed value |
| --- | ---: |
| Complete external invocation | 1.971 seconds |
| Runner through summary writing | 1.284 seconds |
| Sampled process-family peak resident memory | 108,396,544 bytes (103.4 MiB) |
| Operating-system self maximum resident memory | 108,871,680 bytes (103.8 MiB) |
| Output directory size | 804,377 bytes (0.77 MiB) |
| Failed methods; missing cases; swaps | 0; 0; 0 |

The observer took 23 memory samples, with a maximum gap of 0.091 seconds.
Sampling can miss brief peaks. The result stayed below the pilot's 10-minute,
2-GiB resident-memory and 100-MiB storage ceilings. The bootstrap intervals
covered the known effect in all 20 cases; the model-based and QR intervals
covered it in 19 and 20. These small counts describe pipeline behavior only.
They do **not** estimate interval coverage, bias or power reliably.

Scaling the observed runner time by 1,200 gives about 25.7 minutes for
24,000 cases; a threefold allowance gives about 77 minutes. Linear scaling
of output gives about 965 MB. These are planning estimates. These pilot estimates do not measure full-run throughput or calibration.
The separate [study report](BOOTSTRAP_RESULTS.md) gives observed results.

The reproducible commands below require the optional development dependencies
and run from the repository root. `plan` is read-only; `pilot` writes to a new
directory. A second call with `--resume` checks identity and completed
checkpoints before continuing. Set numerical thread counts to one as in the
recorded run. The independent checker reads the run and writes a separate
report; it does not draw data or refit models.

```sh
python validation/bootstrap_calibration.py plan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python validation/bootstrap_calibration.py pilot --output /new/private/pilot-run
python validation/check_bootstrap_calibration.py \
  /new/private/pilot-run /new/private/pilot-check.json
```

The runner freezes the configuration in
[bootstrap-calibration-config.json](bootstrap-calibration-config.json). The
normal interval uses the sample standard deviation of the 200 bootstrap
effects; its two-sided p-value follows the documented reference convention.
The Python package itself still depends only on NumPy and SciPy and does not
import this optional validation tooling at runtime.

## Post-review integrity recheck

After three independent protocol perspectives identified gaps in the
saved-record audit, the runner gained a signed paired coverage difference and
the checker gained independent reconstruction of the fixed cells, input rows
and seeds. It now recalculates interval and p-value conventions, every
claim-bearing cell statistic, failures, and paired results. A checked partial
run is explicitly marked incomplete. Fault tests change a saved effect's
interval, p-value, truth, seed, input, bias uncertainty and failure status;
the checker rejects corruptions and correctly counts a recorded failure.

The changed runner completed a new bounded run of the same separate 20 pilot
cases with 200 resamples each. Its external invocation took 1.944
seconds; the sampled process-family peak was 106,840,064 bytes (101.9 MiB),
and the completed run directory occupied 807,404 bytes (0.77 MiB). All 60
method fits succeeded and the stronger independent checker verified
20/20 terminal cases, their inputs, seeds, inference and full summaries.
The [new compact record](bootstrap-calibration-review-pilot-results.json)
preserves the revised source identity and checks. The earlier record remains
historical evidence for the earlier exact runner and checker; its observed
timing is not silently reassigned to changed code.

This repair does not change the frozen full-study dataset counts, `B=200`,
data generator or study seeds. The revised protocol clarifies the joint
effect as an adjacent within-range score contrast. Formal review closure and
a measured CPU-server resource bound remain required before the full study.
