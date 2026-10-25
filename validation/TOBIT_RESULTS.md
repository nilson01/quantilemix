# Restricted Gaussian Tobit comparison

The optional check compares `fit_tobit` with **installed R qgcomp 2.19.6**
using its Gaussian `qgcomp.tobit.noboot` path. Two fixed sets of 160 rows use
the same latent construction but different censoring and exposure inputs:
52 left-censored rows with prequantized exposures, then 52 left-censored and
14 right-censored rows with four exposure categories. The fitted joint effects
differ. The estimand is the change in the **latent Gaussian mean** when each
exposure score rises by one; it is not the change in the observed censored
outcome's mean.

From an installed Python checkout, the optional check can be rerun with:

```sh
python validation/check_tobit_r.py \
  --r-library ../.r-library-qgcomp-2.19.6 \
  --rscript /Library/Frameworks/R.framework/Resources/bin/Rscript \
  --output /tmp/qgcomp-tobit-new-check.json
```

The output path must be new. The checker pins the fixture and exporter digests,
reruns the exporter against the isolated installed R package, compares that
export with the fixture, then compares Python. R, `qgcomp`, `AER`, `survival`
and `jsonlite` are **optional validation tools**. Python fitting needs only
NumPy and SciPy. The [machine-readable report](tobit-reference-results.json)
records versions and per-field gaps. The maximum absolute numerical gap is
about **5.0 × 10⁻¹³** for the two cases. Scores and censor counts agree
exactly. R computes a small p-value by subtracting from two and loses some
tail precision; Python evaluates the normal tail directly.

The independent check in `tests/test_tobit.py` constructs the censored-normal
likelihood from normal density and tail probabilities, then compares its
finite-difference score and information with the analytic implementation.
This tests the likelihood separately from the R fixture. Fixed-case agreement
does **not** establish interval coverage or validate the Gaussian latent
distribution. The Python subset requires already-clipped complete numeric
outcomes and known limits. Weights, clusters, formulas, non-Gaussian latent
distributions, observed-mean g-computation and repeated-data calibration
remain open.
