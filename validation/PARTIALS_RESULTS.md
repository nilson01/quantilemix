# Sample-split Gaussian partial effects

The new `split_data` and `qgcomp_partials` functions cover one bounded part of
R qgcomp 2.19.6's public sample-splitting interface. The result is exploratory:
group membership is learned from training rows, while each reported partial
effect and its model-based interval come from disjoint validation rows. The
positive held-out fit adjusts for exposures outside the positive group, and the
negative fit does the same for exposures outside the negative group. A held-out
coefficient may change sign; it is not removed or clipped. An empty training
group has no held-out estimate. The held-out results retain the selected
exposure and adjustment names.

## Fixed reference and independent checks

`export_partials_r.R` calls **installed** qgcomp 2.19.6 from the isolated
`../.r-library-qgcomp-2.19.6` library. It uses 128 prequantized rows from a
fixed 4-by-4 score grid, four training replicates and four validation
replicates. The training and validation indices are specified, avoiding
cross-language random-number differences. Its output is saved in
`tests/fixtures/r_partials_2_19_6.json` with SHA-256
`63024bdcb5d4932ac79725f2e3e79ddbb1684cc380bb055e54839e518de01149`.
The optional checker first compares a live R export with that saved input and
output, then compares Python training and held-out estimates, standard errors
and confidence limits. Run it with:

```sh
PYTHONPATH=src python validation/check_partials_r.py \
  --rscript /Library/Frameworks/R.framework/Resources/bin/Rscript \
  --library ../.r-library-qgcomp-2.19.6
```

The installed-R check passed. The training mixture effect
was 0.3512749111; the positive and negative held-out effects were
0.7983422015 and -0.4500591385. Both held-out standard errors were
0.008174938958. An independent NumPy least-squares calculation in
`tests/test_partials.py` verifies both held-out coefficients and their full
covariance-based standard errors. The fixed comparison checks numerical
agreement, not repeated-sample coverage or causal identification.

## Limits and deliberate difference

`qgcomp_partials` currently accepts complete numeric arrays for additive
Gaussian fits, optional numeric covariates, and either prequantized scores or
raw exposures with training cutpoints reused by default. It does not provide
R's Cox or zero-inflated variants, global cutpoints, weights, formulas,
bootstrap, repeated splitting or cross-fitting. A user must decide the split
without inspecting validation results and must assess the usual regression
and causal assumptions separately. The model-based intervals assume
independent observations; a group-preserving split does not make them
cluster-robust. Python excludes an exactly zero training coefficient from
both sign groups. R's nonpositive group can include zero, but its normalized
sign weights are undefined when that group sums to zero.

`split_data` uses a Python seed, so its selected rows do not match an R seed.
For clustered input it assigns each whole cluster to one side. In R
`base_samplesplits.R`, `.split.cluster.data` samples rows **inside each
cluster**, despite the public description promising separation by cluster.
The installed-R fixture confirms this on 12 rows: all three cluster IDs
occur on both sides of its seeded 50/50 split.
Python follows the stated purpose and records this as an intentional
behavior difference. Its zero-based indices also differ from R's one-based
indices.
