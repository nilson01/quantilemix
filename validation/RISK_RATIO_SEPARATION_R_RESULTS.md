# Installed-R check of the retained separated risk-ratio draw

The 4,000-dataset [risk-ratio study](RISK_RATIO_CALIBRATION_RESULTS.md) retained one failed Python fit at cell 0, repetition 190, bootstrap draw 83. This check asks a narrower question: how does installed R qgcomp 2.19.6 handle **that exact 500-row resample**? It does not rerun the study or estimate interval coverage.

The original dataset and row schedule were regenerated from the committed study seed. Their input SHA-256 is `c214ef57d695c54e68fb1b7d9f6791d5acff1eda71ad846404873b333aa32f44`; the saved bootstrap seed is `7377651011041772520`. The sorted resample has 313 distinct original row IDs, 255 rows with adjustment value `z=0` and no events in those rows, and 32 events among 245 rows with `z=1`. Its reproducible CSV SHA-256 is `a9f4a079d92a7ed65153ffdd8806161ab0758db680fe1a22a25a5ccb8fe70512`. The case-level study files remain private; the optional checker regenerates this one input without shipping those files.

The [saved result](risk-ratio-separation-r-results.json) records R 4.5.1, qgcomp 2.19.6, jsonlite 2.0.0 and the R-driver hash.

On the exact draw, the installed package's internal `msm_fit` routine—the routine called for each `qgcomp.glm.boot` resample—returned a conditional logistic fit after 19 iterations and a marginal log risk ratio of `-0.0888064053`, without a model warning. A public `qgcomp.glm.boot` call on that same separated input, with four **new** R bootstrap resamples, also returned `-0.0888064053` and a finite interval. The four new resamples are only an exported-function behavior check; they are not the Python study's original 200-draw schedule. The public Python fit on the original dataset still raises `bootstrap replicate 83 failed: binomial coefficients diverged; check separation`, and a direct Python fit on the separated draw raises the same underlying error.

This difference is deliberate for now. With no events at `z=0`, the conditional logistic likelihood increases as the intercept tends to negative infinity while the `z` coefficient tends to positive infinity with their sum fixed. Thus those two coefficients have **no finite maximum-likelihood estimate**. R's approximately `-20.79` intercept and `+18.67` `z` coefficient are finite stopping values, not evidence that the separated model has a finite maximum. Accepting them in Python solely to match R would remove a diagnostic without validating the resulting uncertainty. The Python estimator and the 4,000-dataset study therefore remain unchanged.

Run the optional check only with an isolated R 4.5.1 library containing installed qgcomp 2.19.6 and jsonlite:

```sh
python validation/check_risk_ratio_separation_r.py \
  --rscript /path/to/R-4.5.1/Rscript \
  --library /path/to/isolated-r-library \
  --output /path/to/new-result.json
```

The output path must be new. The checker verifies the saved input identity, exact separated-row counts, current Python failure, isolated R package origin, R fit values, and public-call outcome. Ordinary package installation and use do not need R. The saved JSON SHA-256 is `bdae91a79a2ed7e8d59b0a849849c7d7520aea236190b77ffe14ed6440ae6340`. This single diagnostic does not establish how a complete installed-R 200-draw bootstrap run would behave with an identical supplied row schedule, nor does it change the successful-fit coverage estimates or broad R-parity status.
