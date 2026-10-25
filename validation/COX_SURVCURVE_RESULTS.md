# Marginal Cox survival curves: installed-R comparison

`qgcomp_survcurve_boot` returns four tables from a
`CoxBootstrapResult`, with the column names and table names of installed R
qgcomp 2.19.6. `mdfq` gives marginal-model curves at each joint exposure score;
`mdfpop` averages those curves equally. `cdfq` gives conditional-model curves
at each score for the **first** original adjustment profile, as R does;
`cdfpop` averages conditional curves over every original adjustment profile
and score. The `q` column is one-based. The marginal and conditional tables
have separate observed-time grids. No table has uncertainty bands.

The [optional checker](check_cox_survcurve_r.py) reruns the installed R
`qgcomp.survcurve.boot` helper on the same original conditional rows and the
same saved Python synthetic outcomes used in the
[marginal Cox stage check](COX_BOOTSTRAP_RESULTS.md). It covers degree-one and
degree-two marginal curves at four intervention scores: eight tables in all.
The [fixed R tables](../tests/fixtures/r_cox_survcurve_2_19_6.json),
[R exporter](export_cox_survcurve_r.R) and
[comparison report](cox-survcurve-reference-results.json) retain the inputs
and hashes. The largest absolute survival-probability difference is
`1.67e-10`; all time and score labels agree exactly. A separate direct
calculation checks that `cdfpop` averages all conditional profiles while
`cdfq` uses only the first.

Replay with an isolated installation of R qgcomp 2.19.6, `survival` and
`jsonlite`:

```sh
PYTHONPATH=src python validation/check_cox_survcurve_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library \
  --output /path/to/new-report.json
```

These results check the curve extraction and averaging for two fixed
same-outcome fits. The Python event-time generator still differs from R
qgcomp 2.19.6, so this is not end-to-end bootstrap parity. For prequantized
`q=None` inputs, Python displays the fitted pooled-quartile intervention
levels; R reconstructs zero-based levels from saved breaks. Repeated-dataset
calibration, confidence bands, case weights, clusters, offsets and broader
survival options remain open. The curves do not establish proportional
hazards, independent censoring or a causal interpretation.
