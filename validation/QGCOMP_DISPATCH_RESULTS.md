# Additive formula dispatch check

The public Python `qgcomp` function chooses among existing additive fitters.
It does not estimate a new model. Gaussian and Poisson formulas use
conditional fits. Binary `rr=False` gives a conditional odds ratio. Binary
`rr=True`, as in R's default, gives a marginal bootstrap risk ratio. This
original check covers two joint-score levels; the later
[four-level check](RISK_RATIO_MULTILEVEL_RESULTS.md) covers the default
`q=4` route. A Poisson observation-time
offset remains on the conditional route.

The fixed reference calls the exported `qgcomp` function in an isolated
installation of R qgcomp 2.19.6. `export_qgcomp_dispatch_r.R` verifies the
loaded namespace and saves six routes: adjusted Gaussian, Gaussian with a
factor adjustment, binary conditional odds ratio, binary default marginal
risk ratio, adjusted Poisson, and Poisson with an offset. The categorical
Python formula uses `C(z)` for an R factor column named `z` in a simple
formula. The offset Python call uses `exposure_time="time"` for the R term
`offset(log(time))`.

The R run resolved a routing detail. R removes `offset(log(time))` from the
term labels used by its nonlinear-term check. As a result, the additive
Poisson offset case uses `qgcomp.glm.noboot`; giving it bootstrap-only `B`
and `seed` arguments fails. The Python dispatcher follows the actual
conditional route and rejects bootstrap-only arguments there. R ignores
`rr=TRUE` for nonbinary families; Python does the same.

The pinned fixture is `tests/fixtures/r_qgcomp_dispatch_2_19_6.json`, SHA-256
`22706d0b1ed75c8dcaebb8188059b288cf67d365a8027678f045ae76265d2f22`.
The binary bootstrap case saves all 24 R row draws. The optional checker
rebuilds that fit using the same schedule; Python's public seeded generator
does not claim to reproduce R's random-number stream.

Run the optional check from the repository root with the isolated R library:

```sh
PYTHONPATH=src python validation/check_qgcomp_dispatch_r.py \
  --rscript /path/to/Rscript --library /path/to/isolated-R-library
```

The checker regenerates the installed-R outputs, checks the saved inputs and
routes, then checks Python's public dispatcher and effect values. All six
cases pass. The largest absolute Python/R difference is `4.53e-07`, in a
binary conditional interval bound; fixed Gaussian and Poisson values agree
more closely. The binary bootstrap uncertainty agrees when the row schedule
is shared. Independent tests check the selected result types, distinct
binary effect measures, offset routing, categorical/weighted complete-case
forwarding, and rejection of unsupported formulas and options. Python has
no R runtime dependency.

The [four-level risk-ratio check](RISK_RATIO_MULTILEVEL_RESULTS.md) covers
the default binary marginal route separately. The [coverage map](../docs/r-api-parity.md)
lists current routes and unsupported options. These fixed cases do not
measure statistical performance across repeated datasets.
