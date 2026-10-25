# Marginal survival bootstrap: fixed-stage evidence and limits

`fit_cox_bootstrap` fits a conditional Cox model,
simulates potential event times at each joint exposure score, fits a marginal
Cox curve to those outcomes, and uses a fixed-cut row bootstrap for normal
confidence intervals. Degree one reports one log hazard-ratio coefficient;
degree two reports linear and squared score terms. A one-level change from
score `a` is `psi[0] + (2a+1)*psi[1]` at degree two. The method accepts complete
numeric independent rows, Efron event ties and optional numeric adjustment
columns. It does not support case weights, clusters, offsets or time-varying
exposures. Failed bootstrap replicates raise an error and are not replaced.

The [optional installed-R check](cox-bootstrap-reference-results.json) uses
two fixed 128-row inputs. Python saves the exact simulated event times,
indicators, intervention scores and three bootstrap effects in the
[stage-input fixture](../tests/fixtures/r_cox_bootstrap_stage_inputs_2_19_6.json).
Isolated installed R qgcomp 2.19.6 and `survival::coxph` then fit the **same
synthetic outcomes**, at degree one and degree two, and calculate the same
bootstrap covariance, normal limits and p-values. The maximum absolute
point-estimate difference is `1.18e-9`; the maximum p-value difference is
`8.62e-9`. The [Python input exporter](export_cox_bootstrap_python.py),
[R stage exporter](export_cox_bootstrap_stage_r.R) and
[checker](check_cox_bootstrap_r.py) retain their exact inputs and hashes.
An independent inverse-cumulative-hazard example checks event times and
administrative censoring. Fixed-seed tests check saved draws, bootstrap sample
SD, covariance, intervals, result shapes and failure inputs. R is optional for
validation; Python runtime uses NumPy and SciPy only.

**The full simulation engines are intentionally different.** Installed R
`qgcomp.cox.boot` compares each fitted cumulative-hazard *increment* directly
with a uniform draw, then uses a shortened follow-up-time vector. The
[bounded R audit](cox-bootstrap-r-behavior.json), reproduced by
[audit_cox_bootstrap_r.R](audit_cox_bootstrap_r.R), found 47 missing simulated
follow-up times among 512 potential outcomes in one fixed q=4, B=2, MCsize=128
run. The R Cox fit omits those missing outcomes. Python instead draws an
exponential event threshold and finds its first crossing of the fitted
cumulative hazard; people without a crossing are censored at the last observed
time. This preserves every simulated person and follows the Cox survival
function `S(t|x)=exp(-H_0(t) exp(x beta))`. The observed R behavior is a
software fact, not a claim about R's interval performance. Direct numerical
agreement between the two *end-to-end* bootstrap pipelines is therefore not
claimed.

Replay the stage check with an isolated installed R qgcomp 2.19.6 library,
`survival` and `jsonlite`:

```sh
PYTHONPATH=src python validation/check_cox_bootstrap_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

These fixed cases do not establish proportional hazards, independent
censoring or causal identification. Degree-two curve uncertainty uses the
full bootstrap covariance. The separate [degree-one study](COX_BOOTSTRAP_STUDY_RESULTS.md)
reports repeated-dataset coverage in four known-score conditions; it does
not validate degree-two intervals or survival-curve bands.
