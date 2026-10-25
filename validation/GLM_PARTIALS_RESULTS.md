# Held-out binary and count partial effects

`qgcomp_partials(..., family="binomial")` and
`qgcomp_partials(..., family="poisson")` extend the existing one-split
Gaussian interface. They fit all selected exposures on the training rows,
then estimate each nonempty training sign group on separate validation rows.
Every exposure outside that group stays in the validation model as an
adjustment variable. A validation coefficient stays in its training-selected
group even if its sign reverses. The binary effect is a **conditional log
odds ratio**; the Poisson effect is a **conditional log mean ratio** for equal
observation periods. Exponentiating gives the corresponding conditional
ratio, not a population-average risk or count ratio.

## Installed R comparison

`export_glm_partials_r.R` calls the installed R qgcomp 2.19.6 package in the
isolated `../.r-library-qgcomp-2.19.6` library under compatible R 4.5.1.
Four fixed cases use 320 rows each, with 160 training and 160 validation rows:
prequantized and automatically grouped exposures in each family. The raw
cases perturb exposure measurements before applying four quantile groups;
held-out fits reuse training cuts. Every training fit selects `x1` for the
positive group and `x2` for the negative group. The exporter verifies the
loaded package's namespace path and version. The saved fixture is
`tests/fixtures/r_glm_partials_2_19_6.json`, SHA-256
`cfe113409aeb665f6242508710e1658382db0070cdfdc5f15f950801244a25ee`.

The optional checker re-exports from installed R, compares all saved inputs
and R outputs, then checks Python's training and held-out effects, standard
errors and confidence limits. Run it with:

```sh
PYTHONPATH=src python validation/check_glm_partials_r.py \
  --rscript /Library/Frameworks/R.framework/Resources/bin/Rscript \
  --library ../.r-library-qgcomp-2.19.6
```

All four cases passed. The largest absolute Python/R
difference was **2.18e-7**, inside the existing `2e-6` additive GLM reference
tolerance. The numerical difference is in covariance-derived uncertainty;
R `glm` and the Python solver stop at different tolerances. This comparison
establishes agreement for these fixed inputs, not interval coverage over
repeated datasets.

## Independent calculations and limits

Separate exact-cell constructions check a binary held-out log odds effect of
`log(3)` and a count held-out log mean effect of `log(3)`, with the opposite
sign for the other exposure. Their standard errors are calculated from the
inverse logistic or Poisson information matrix, outside the package's fit
code. A binary case reverses the validation signs and verifies that the
training-selected groups remain unchanged. Failure tests reject unsupported
families and invalid outcomes; checker tests detect changed inputs, changed
R outputs and a missing R executable.

The current interface takes complete numeric arrays and optional numeric
covariates. It does not add formula or categorical partial-effect inputs,
case weights, Poisson observation-time offsets, global cuts, cross-fitting,
Cox or zero-inflated partial variants. The intervals assume the fitted GLM
and independent observations; a group-preserving split alone does not make
them cluster-robust. A causal interpretation needs separate study-design
assumptions. Repeated-sample false-positive rates and coverage for these
partial effects have not been measured.
