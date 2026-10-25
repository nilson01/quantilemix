# Named covariance helpers

`se_comb` computes the standard error of a named linear combination of model
coefficients. `vc_comb` gives the 2-by-2 covariance of one named coefficient
and that combination. Both use a covariance matrix plus an explicit `names`
sequence giving its row and column order. The default combination gives each
selected exposure coefficient weight one. A scalar or full-vector `grad`
sets other weights.

The saved matrix and outputs come from isolated installed R qgcomp 2.19.6.
The default standard error, two weighted standard errors, and default
2-by-2 covariance agree exactly at the saved precision. Direct matrix
arithmetic independently checks the weighted contrast.

For a non-unit gradient, R `vc_comb` applies the weight to the contrast
variance but omits it from the cross-covariance with the named coefficient.
In the fixed example, R keeps that cross term at `-0.0076297819` when the
weight is `0.5`; the covariance identity gives `-0.00381489095`. Python
uses the latter. This difference is intentional and covered by a test.
It does not change the default unweighted result.

Run the saved comparison without R:

```sh
PYTHONPATH=src python validation/check_covariance_helpers_r.py
```

`--regenerate` runs the exporter with installed R qgcomp 2.19.6. Set
`QGCOMP_R_LIB` and, if needed, `RSCRIPT` to locate it. The fixture SHA-256
is `813c30d98ae663aad2776f793fe3625d8331eaab8cd4472b77ed436d3ee2ea13`.
R is optional for reference work and is not a Python runtime dependency.
