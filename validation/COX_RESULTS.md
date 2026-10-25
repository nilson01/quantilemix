# Restricted conditional Cox comparison

The optional check compares `fit_cox` with **installed R qgcomp 2.19.6** in
an isolated R library. It uses two fixed sets of 128 right-censored rows with
91 observed events each. Follow-up times are distinct. The first case supplies
integer exposure scores; the second supplies continuous measurements grouped
into four categories. Their event-time order and fitted joint effects differ.

Run the check from the package root after installing the Python package:

```sh
python validation/check_cox_r.py \
  --r-library ../.r-library-qgcomp-2.19.6 \
  --rscript /Library/Frameworks/R.framework/Resources/bin/Rscript \
  --output /tmp/qgcomp-cox-new-check.json
```

The output path must not exist. The checker verifies the saved fixture and R
exporter digests, reruns the exporter against the installed R package, and
compares the new R export with the pinned fixture before comparing Python.
R, `qgcomp`, `survival` and `jsonlite` are **optional validation tools**; Python
fitting needs only NumPy and SciPy. See
[`cox-reference-results.json`](cox-reference-results.json) for installed
versions and per-field differences. The largest numerical difference across
coefficients, covariance, joint effect, variance, confidence limits and
p-value is about **3.1 × 10⁻¹⁵**. Scores and event counts agree exactly.

An independent three-person example checks risk sets, partial log likelihood,
score and information in `tests/test_cox.py`. These untied coefficient cases
do not measure interval coverage or test proportional hazards. Separate
reports cover [ties](COX_TIES_RESULTS.md), [curves](COX_CURVES_RESULTS.md),
[weights](COX_WEIGHTS_RESULTS.md), [offsets](COX_OFFSET_RESULTS.md) and
[clusters](COX_CLUSTER_RESULTS.md).
