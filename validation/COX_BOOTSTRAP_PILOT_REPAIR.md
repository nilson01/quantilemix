# Cox pilot numerical failure and bounded repair

The [calibration protocol](COX_BOOTSTRAP_PROTOCOL.md) calls for a 20-dataset server pilot before the full repeated-dataset study. The first Slurm array stopped before data generation because its wrapper called `/usr/bin/time`, which the server image lacks. Its outputs remain preserved. A second array used the corrected wrapper and was canceled under the protocol stop rule when two first-case fits failed. Neither failed case was replaced within that array.

| Pilot cell | Censoring rate | True log hazard ratio | Failed bootstrap draw | Recorded error |
| --- | ---: | ---: | ---: | --- |
| 2 | 0.30 | 0 | 72 | Cox score did not converge |
| 3 | 0.30 | 0.4 | 43 | Cox score did not converge |

The exact-version checker audited the preserved partial journals after cancellation: 20 cases were planned, six started, two produced explicit failed records, four were interrupted mid-fit, and 14 never started. No case succeeded. This confirms that the stop was recorded rather than treated as missing successful data.

The synthetic input arrays and random-number streams were reconstructed from the frozen pilot seeds. The exact failing second-stage Cox fits each have 30,000 simulated rows. An independent scalar score-root search found coefficients `-0.132083323493977` and `0.077220119349638`. The previous solver evaluated candidates with nearly zero score but rejected their final Newton step: summing a log likelihood near `-175,586` or `-206,055` changed by about `10^-8` from floating-point roundoff. The rejected step left a score above the existing `10^-7` convergence threshold.

The repair increases only the likelihood roundoff allowance from `128 ε |L|` to `512 ε |L|`, where `ε` is double-precision machine epsilon and `L` is the log likelihood. A step must still improve the score when it uses that allowance. The final score threshold, coefficient-divergence guard, information-matrix checks and separation errors remain. Both exact failed draws now converge with coefficients `-0.132083323493995` and `0.077220119349683`.

As an independent software check, R 4.5.1 `survival::coxph(..., ties="efron")` from survival 3.8.3 fit the identical saved simulated outcomes. It gave `-0.132083323393032` and `0.077220119312000`; the largest Python–R coefficient gap is about `1.01 × 10^-10`. Two deterministic tests replay the original failing draws and check those installed-R values. The raw, synthetic replay files, both Slurm arrays, partial journals and R output remain in the private project compute record.

This establishes a numerical repair for two observed software failures. It does **not** establish bootstrap interval coverage or guarantee that no other pilot case can fail.

The third isolated array used the repaired wheel. Its exact-version partial audit found 20 planned cases: two successful, one explicitly failed, four interrupted during cancellation, and 13 never started. The completed earlier-path cases took 387.48 and 403.48 seconds. Cell 1, repetition 1 failed because empirical pooled quartiles were `[1, 2, 2]`; the bootstrap requires distinct intervention levels. The pooled score counts were `[288, 311, 316, 285]`. This is a design problem in the protocol's `q=None` path, which follows the R package's pooled-quantile convention, rather than evidence of a Cox solver regression. The array was canceled under the stop rule. Its results and scheduler logs remain in `compute/cox-bootstrap-pilot-20261024/attempt-003/interrupted-2926644/`.

The reviewed amendment sets `q=4` and fixed breaks `[-0.5, 0.5, 1.5, 2.5, 3.5]` separately for all four exposures. A bounded replay of the exact failing input confirmed that the transformed scores equal the generated scores, all four categories remain present, and a two-draw Cox bootstrap fit completed without warnings. The separate [v2 20-case pilot](COX_BOOTSTRAP_PILOT_RESULTS.md) has now completed and passed its terminal audit. The separate [study report](COX_BOOTSTRAP_STUDY_RESULTS.md) gives interval coverage and failures.
