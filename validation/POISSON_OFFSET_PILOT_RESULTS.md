# Poisson offset calibration timing pilots

Two local 20-case pilots were run for
`poisson-offset-known-scores-v1`. The first used mode integer 81 because of
a runner error; the written protocol specified 91. Its numerical results
remain timing evidence for that older stream, but it is **not** a pilot of
the written stream. Three independent synthetic protocol-review perspectives
found this mismatch before any full study. The runner was corrected to use
91 for the pilot and 92 for the future study; fixed-hash checks now pin
both streams. No full offset study has run.

| Check | Original pilot (81) | Corrected pilot (91) |
| --- | ---: | ---: |
| Package fits attempted/successful | 20/20 | 20/20 |
| Independent score-equation fits paired | 20/20 | 20/20 |
| Package failures, independent failures, warnings, audit issues | 0 each | 0 each |
| Largest absolute paired numerical difference | 1.14e-12 | 2.58e-12 |
| Runner command wall time | 0.54 s | 0.52 s |
| Checker command wall time | 0.45 s | 0.47 s on a repeated read-only timing pass |
| Runner maximum resident memory | 98,861,056 bytes | 98,435,072 bytes |
| Checker maximum resident memory | 98,385,920 bytes | 99,500,032 bytes on the timing pass |

The corrected runner and checker used one numerical thread. Their exact
invocations, from the package root, were:

```
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src /usr/bin/time -l ../.venv/bin/python validation/poisson_offset_calibration.py --mode pilot --output ../validation-local/poisson-offset-calibration-20261023/pilot-002/result.json
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src /usr/bin/time -l ../.venv/bin/python validation/check_poisson_offset_calibration.py ../validation-local/poisson-offset-calibration-20261023/pilot-002/result.json --output ../validation-local/poisson-offset-calibration-20261023/pilot-002/check.json
```

Both commands exited zero. Timing used macOS `/usr/bin/time -l` on an Apple
M3 Pro with 19,327,352,832 bytes of memory. The Python environment had
Python 3.14.3, qgcomp-python 0.1.0.dev0, NumPy 2.5.4 and SciPy 1.18.1.
Fresh local free space was 94,251,972 KiB before the corrected pilot. One
local worker slot was reserved and released. The corrected raw files are
retained privately under `../validation-local/poisson-offset-calibration-20261023/pilot-002/`:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `result.json` | 14,012 | `12e8382952483556a6a45907ccde2e1215ff4749336adff1d89730c21680adf4` |
| `check.json` | 15,125 | `0a50019576161b3d63bcabc4e7f21cb569d46cd03c7f2a1f8c7a1325aa0dcf58` |

The original files remain under `pilot-001/`; neither set was overwritten.
The corrected pilot measures feasibility for a one-CPU server study. Twenty datasets do
not estimate interval coverage, false-positive rate or bias with useful
precision. The completed equal-period study is a separate result.
