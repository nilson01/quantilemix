# Nine-exposure Gaussian scenario-5 translation: saved results

This is a **qgcomp-only translation**, not an exact reproduction of [Keil et al. (2020) Figure 1](https://doi.org/10.1289/EHP5838). The [source audit](PAPER_SCENARIO5_SOURCE_AUDIT.md) records what the public paper and supplement do and do not provide. The article gives nine-exposure settings, but its released executable illustration uses four exposures and five runs. We used its shared-master score generator in two named interpretations. `released_ncor2` follows the illustration's `ncor=2` argument, which also correlates noncausal X3. `two_causal_ncor1` routes only the two outcome-relevant scores through that mechanism. The inspected public release does not include the original 1,000-run Figure 1 code or numeric case results. The article's prose gives 0.25 as the largest joint effect, while its coefficients and caption give 0.20; this study uses the coefficient sum 0.25 − 0.05 = 0.20.

Each cell has 1,000 independent datasets of 500 observations and nine balanced scores. The true joint one-score effects are 0.05, 0.15 and 0.20. The requested X1–X2 correlation is 0, 0.4 or 0.75. The outcome is additive with independent standard-normal error, and all nine already scored exposures enter an intercept-containing Gaussian fit. All 18,000 planned fits completed. There were **zero failed fits and zero captured warnings**. The server checker independently recomputed all 18,000 fits by QR least squares; its largest absolute numerical difference was `3.83e-14`. The server batch job completed `0:0` in 75 seconds with 104,476 KiB maximum resident memory and 6,591 KiB of saved output. These checks support this restricted computation, not practical low bias or general causal validity.

Bias is the mean estimate minus the true joint effect. MCSE is its Monte Carlo standard error. Coverage is the fraction of nominal 95% intervals containing the truth; the parenthesized bounds are 95% Wilson intervals for that *observed rate*. Rejection is the fraction with p<0.05. Since every true effect is nonzero, rejection is empirical power, not a false-positive rate. The estimator uses a normal critical value for its interval and a Student t tail for its p-value, so the two threshold events need not coincide.

| Generator | True effect | Requested correlation | Bias (MCSE) | Coverage (Wilson 95% interval) | Rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| `released_ncor2` | 0.05 | 0.00 | +0.0025 (0.0038) | 0.945 (0.929, 0.958) | 0.081 |
| `released_ncor2` | 0.05 | 0.40 | +0.0001 (0.0036) | 0.956 (0.941, 0.967) | 0.076 |
| `released_ncor2` | 0.05 | 0.75 | -0.0027 (0.0034) | 0.958 (0.944, 0.969) | 0.066 |
| `released_ncor2` | 0.15 | 0.00 | +0.0027 (0.0040) | 0.943 (0.927, 0.956) | 0.252 |
| `released_ncor2` | 0.15 | 0.40 | -0.0007 (0.0035) | 0.955 (0.940, 0.966) | 0.256 |
| `released_ncor2` | 0.15 | 0.75 | +0.0014 (0.0035) | 0.948 (0.932, 0.960) | 0.298 |
| `released_ncor2` | 0.20 | 0.00 | +0.0011 (0.0039) | 0.943 (0.927, 0.956) | 0.385 |
| `released_ncor2` | 0.20 | 0.40 | +0.0017 (0.0037) | 0.931 (0.914, 0.945) | 0.447 |
| `released_ncor2` | 0.20 | 0.75 | -0.0018 (0.0034) | 0.957 (0.943, 0.968) | 0.438 |
| `two_causal_ncor1` | 0.05 | 0.00 | -0.0013 (0.0037) | 0.959 (0.945, 0.970) | 0.060 |
| `two_causal_ncor1` | 0.05 | 0.40 | +0.0017 (0.0037) | 0.947 (0.931, 0.959) | 0.068 |
| `two_causal_ncor1` | 0.05 | 0.75 | +0.0031 (0.0036) | 0.954 (0.939, 0.965) | 0.072 |
| `two_causal_ncor1` | 0.15 | 0.00 | +0.0036 (0.0037) | 0.953 (0.938, 0.964) | 0.229 |
| `two_causal_ncor1` | 0.15 | 0.40 | +0.0004 (0.0037) | 0.948 (0.932, 0.960) | 0.262 |
| `two_causal_ncor1` | 0.15 | 0.75 | -0.0028 (0.0036) | 0.947 (0.931, 0.959) | 0.249 |
| `two_causal_ncor1` | 0.20 | 0.00 | +0.0009 (0.0038) | 0.949 (0.934, 0.961) | 0.384 |
| `two_causal_ncor1` | 0.20 | 0.40 | +0.0009 (0.0037) | 0.939 (0.922, 0.952) | 0.397 |
| `two_causal_ncor1` | 0.20 | 0.75 | -0.0016 (0.0037) | 0.943 (0.927, 0.956) | 0.401 |

The remaining prespecified distribution and correlation summaries are below. The three correlation values are mean realized Pearson correlations in X1–X2, X1–X3 and X2–X3 order. The saved machine-readable result also contains their standard deviations, every case-level estimate, variance, interval, p-value, input hash and identity.

| Generator | True effect | Requested correlation | Empirical SD | Root mean reported variance | Bias median [Q1, Q3] | Mean realized correlations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `released_ncor2` | 0.05 | 0.00 | 0.1212 | 0.1210 | +0.0058 [-0.0766, +0.0819] | -0.000, +0.000, -0.002 |
| `released_ncor2` | 0.05 | 0.40 | 0.1130 | 0.1119 | -0.0026 [-0.0737, +0.0785] | +0.403, +0.401, +0.401 |
| `released_ncor2` | 0.05 | 0.75 | 0.1062 | 0.1079 | -0.0063 [-0.0724, +0.0709] | +0.753, +0.752, +0.753 |
| `released_ncor2` | 0.15 | 0.00 | 0.1269 | 0.1212 | +0.0013 [-0.0906, +0.0900] | +0.000, -0.002, +0.001 |
| `released_ncor2` | 0.15 | 0.40 | 0.1093 | 0.1117 | +0.0008 [-0.0730, +0.0709] | +0.404, +0.401, +0.401 |
| `released_ncor2` | 0.15 | 0.75 | 0.1114 | 0.1082 | -0.0002 [-0.0779, +0.0791] | +0.753, +0.753, +0.754 |
| `released_ncor2` | 0.20 | 0.00 | 0.1232 | 0.1209 | +0.0032 [-0.0805, +0.0799] | -0.003, +0.001, +0.001 |
| `released_ncor2` | 0.20 | 0.40 | 0.1170 | 0.1117 | +0.0044 [-0.0761, +0.0765] | +0.403, +0.403, +0.404 |
| `released_ncor2` | 0.20 | 0.75 | 0.1085 | 0.1082 | -0.0064 [-0.0758, +0.0758] | +0.753, +0.753, +0.753 |
| `two_causal_ncor1` | 0.05 | 0.00 | 0.1157 | 0.1212 | -0.0014 [-0.0743, +0.0782] | -0.001, -0.001, -0.006 |
| `two_causal_ncor1` | 0.05 | 0.40 | 0.1164 | 0.1170 | +0.0036 [-0.0778, +0.0799] | +0.403, +0.001, +0.001 |
| `two_causal_ncor1` | 0.05 | 0.75 | 0.1123 | 0.1149 | +0.0025 [-0.0718, +0.0825] | +0.752, -0.001, +0.000 |
| `two_causal_ncor1` | 0.15 | 0.00 | 0.1169 | 0.1209 | +0.0045 [-0.0724, +0.0781] | -0.001, +0.001, -0.002 |
| `two_causal_ncor1` | 0.15 | 0.40 | 0.1179 | 0.1173 | -0.0010 [-0.0820, +0.0839] | +0.401, -0.002, -0.002 |
| `two_causal_ncor1` | 0.15 | 0.75 | 0.1143 | 0.1149 | -0.0067 [-0.0828, +0.0775] | +0.751, +0.001, +0.000 |
| `two_causal_ncor1` | 0.20 | 0.00 | 0.1187 | 0.1212 | -0.0002 [-0.0804, +0.0805] | -0.000, -0.000, +0.000 |
| `two_causal_ncor1` | 0.20 | 0.40 | 0.1173 | 0.1171 | +0.0007 [-0.0734, +0.0776] | +0.402, -0.001, -0.000 |
| `two_causal_ncor1` | 0.20 | 0.75 | 0.1166 | 0.1150 | -0.0040 [-0.0809, +0.0753] | +0.752, +0.000, -0.001 |

The nine primary mean biases range from −0.00268 to 0.00274, with MCSE 0.00336–0.00401. These observed values are near zero at this simulation precision. **No scientifically justified practical-bias margin was set**, so this is not a low-bias pass/fail result. All nine primary-minus-sensitivity bias-difference 95% Monte Carlo intervals include zero; this does not establish equivalence between generators. The two arms intentionally differ in X3 dependence at nonzero requested correlation. Their zero-correlation cells agree in distribution within the observed Monte Carlo variation. The saved checker gives each difference and its interval.

The lowest primary coverage was **0.931** (931/1,000) at true effect 0.20 and requested correlation 0.4; its Wilson interval is approximately [0.914, 0.945]. Its bias was +0.00169 (MCSE 0.00370), illustrating that near-zero bias alone does not ensure nominal interval coverage. This cell did not cross the protocol's three-binomial-MCSE screening rule: its departure is about 2.76 nominal standard errors. That rule is descriptive, not an all-cells validity test. This is the lowest of nine primary cells; an ordinary single-cell 95% interval does not establish a family-wide or persistent coverage defect. The use of a normal rather than t critical value also affects expected coverage slightly. The result warrants caution and independent confirmation before a general calibration claim. No simulation settings were changed after viewing this cell.

The broad near-zero bias pattern is qualitatively consistent with the paper's Figure 1 discussion. We cannot compare case values or numeric boxplot summaries because its nine-exposure production stream is unavailable. We did not run WQS. The result says nothing about continuous-exposure quantization, missingness, confounding, interactions, non-Gaussian outcomes, or whether a real mixture can be shifted one score level for every component.

## Exact-input R and reproducibility evidence

Before this study, one fixed pilot input in each of the 18 generator/cell combinations was fit with isolated installed R qgcomp 1.3.0 and 2.19.6. All 36 effects and variances agreed with Python; the largest difference was `3.93e-15`. This is same-input software agreement, separate from the 18,000-dataset performance study. R is optional validation software and is not a Python runtime dependency. The package and dependency identities are saved in `validation-local/scenario5-20261023/r13-pilot-v2.json` and `r219-pilot-v2.json`. The comparisons used R 4.5.1 with compatible compiled libraries.

The full study ran on Arseven from private Git commit `e1349fa53057b8a5611629a4d7427ccffcbcbf3a` in a clean checkout. The transferred Git bundle SHA-256 is `fcbb4c0194ce7fe519ba2d981f1df244398ee721b7713c62dc96add6e1916bfd`. Job `2904103` used Python 3.11.5, NumPy 2.0.0 and SciPy 1.14.0, one numerical thread, one requested CPU and 4 GiB memory. The result (`9916e5254ce8e00119c1ca1ccd1ee1b373842743b24560356bbd11e0c3a7575f`), complete case journal (`f33073f69a4d81ba241b5607a47265573a888ccb7a23a28a78a1a9f5ec00b2b7`) and server QR receipt (`86d9e7d62d2ca40b60730e3219535c6c5a3cc454238926267592790cf7695f21`) matched the returned file manifest. The 90-case server pilot had already passed from the same commit and bundle. No case was dropped or rerun.

A read-only Mac replay matched 17,995 exact input hashes. Five cases had the same exposure bytes but one outcome value differed at the last bits by at most `8.88e-16`; the ARM/x86 numerical path is a plausible explanation, not a separately established mechanism. The server re-exported those five exact inputs, each linked by its saved input hash. The supplemental input file SHA-256 is `344da9d2fdb3e233de331e3c9817f228bc496f141788c875f6e35f1e13835764`. The local checker used those five saved outcomes and otherwise regenerated all inputs; it then independently checked all 18,000 fits. The five identities are `released_ncor2` cells 2/704, 5/335, 7/825 and `two_causal_ncor1` cells 3/443, 6/978 (cell/repetition). This qualification did not change any server fit. The original server checker remains the primary check of the original inputs.

The server export script was run from private file SHA-256 `a104f68d6348e1cfc788990186e55c42ff8f0c62f9bd815f137aa32adfbfe7ad`, now retained as `validation/server_scenario5/export_cross_platform_inputs.py`. The post-run local checker source SHA-256 is `56e2f6cde3c3b738f1a6a4fd102cdadbdef55e3bca3314009ff7fe0ee078d1dc`. The returned result, journal, receipts, logs, source bundle and five-case supplement remain in the private `01-qgcomp/validation-local/scenario5-20261023/` directory. The raw result, journal and five-case supplement are retained separately;
this repository is not a complete raw-study archive.
