# Restricted conditional multinomial comparison

The optional check compares `fit_multinomial` with **installed R qgcomp
2.19.6** using its additive `qgcomp.multinomial.noboot` path. Two fixed
240-row, three-category cases use the same named reference outcome, `low`:
one supplies integer exposure scores, while the other supplies continuous
measurements grouped into four categories. The method reports a separate
joint log odds ratio for `mid` versus `low` and `high` versus `low`, plus
their covariance. It does not combine them into one effect or estimate a
marginal population contrast.

Each case is exported twice from installed R: once with its default `nnet`
optimizer settings, and once with `reltol=1e-13, maxit=1000`. This separates
method agreement from an optimizer stopping difference. The Python fit's
largest gap is about **1.9 × 10⁻⁵** from default R and **9.5 × 10⁻⁷** from
the tighter run; the latter maximum is in a chi-square test statistic.
Coefficient, covariance, category probability, joint effect and interval
differences are smaller. Scores agree exactly. The
[machine-readable report](multinomial-reference-results.json) records every
field and installed version.

Run the optional check from an installed Python checkout:

```sh
python validation/check_multinomial_r.py \
  --r-library ../.r-library-qgcomp-2.19.6 \
  --rscript /Library/Frameworks/R.framework/Resources/bin/Rscript \
  --output /tmp/qgcomp-multinomial-new-check.json
```

The output path must be new. The checker pins the exporter and fixture,
reruns the installed R package and compares both R solver settings before
comparing Python. R, `qgcomp`, `nnet` and `jsonlite` are optional validation
tools; Python fitting uses only NumPy and SciPy. An independent exact-cell
example gives an odds multiplier of four for each nonreference category.
Finite differences of a separately written softmax likelihood check the
analytic score and information in `tests/test_multinomial.py`.

These fixed comparisons do **not** establish interval coverage or a causal
interpretation. Python requires complete numeric exposure/covariate arrays,
named outcome levels and at least three categories. Weights, clusters,
formulas, nonlinear terms, marginal multinomial bootstrap and repeated-data
calibration remain open.
