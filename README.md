# QuantileMix

Quantile g-computation estimates how an outcome changes when several exposure
scores increase together. For example, four exposure groups receive scores
0, 1, 2 and 3. Tied measurements can make the groups unequal.

QuantileMix provides selected models for continuous, binary, count and
survival outcomes. It also has narrower methods for censored, multicategory
and excess-zero outcomes. Each method has its own input limits and effect
measure. The package does not need R for fitting or prediction.

The method follows [Keil et al. (2020)](https://doi.org/10.1289/EHP5838).
Selected calculations were checked against Alexander Keil's
[R qgcomp](https://github.com/alexpkeil1/qgcomp). This is an unofficial Python
package; the original authors have not endorsed it.

[Documentation](https://nilson01.github.io/quantilemix/) · [API](https://nilson01.github.io/quantilemix/reference/)

## Install

With Git and access to the repository, install from GitHub:

```sh
python -m pip install "git+https://github.com/nilson01/quantilemix.git@main"
```

Or run this in a local checkout containing `pyproject.toml`:

```sh
python -m pip install .
```

Use `quantilemix` for installation and imports.
NumPy and SciPy install as dependencies. For an isolated Python environment,
see the [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

## First example

These two exposures already have scores 0–3. `q=None` tells `fit` to use those
scores directly. Omit it when the columns contain raw measurements; the
default groups each exposure into four quantiles.

```python
import numpy as np
from quantilemix import fit

x1, x2 = np.meshgrid(np.arange(4), np.arange(4))
X = np.column_stack([x1.ravel(), x2.ravel()])
y = 1 + 2*X[:, 0] - 0.5*X[:, 1] + 0.2*(X[:, 0] - 1.5)*(X[:, 1] - 1.5)

result = fit(y, X, q=None)
print(result.summary())
print(result.psi, result.se, result.ci)
predicted_means = result.predict(X)
```

Joint-effect row from `result.summary()` (excerpt):

```text
Term  Estimate  SE         CI lower  CI upper  t        p-value
psi   1.5       0.0877058  1.3281    1.6719    17.1026  2.70924e-10
```

![Estimated joint mean difference with a pointwise 95% confidence band.](https://nilson01.github.io/quantilemix/assets/joint-effect.png)

The plot shows `a * result.psi` against the common exposure score `a`,
relative to all scores set to zero. Shading is a pointwise 95% normal
confidence interval for the mean contrast, not an individual prediction interval.
[Plot code](https://github.com/nilson01/quantilemix/blob/main/docs/examples/plot_joint_effect.py).

Here `psi` is the fitted change in the mean outcome when **both** exposure
scores rise by one category. The two exposure coefficients are 2 and −0.5,
so their sum is 1.5. `se` is its standard error, and `ci` contains the
confidence limits. These are changes in *scores*, not in the original
measurement units. A causal interpretation needs assumptions about the
study that the package cannot check.

## Choose a method

| Analysis | Main functions | Effect or output |
| --- | --- | --- |
| Continuous outcome | `fit`, `fit_formula` | Conditional joint mean difference; accepts positive case weights |
| Binary outcome | `fit_binomial`, `fit_binomial_risk_ratio`, `fit_binomial_odds_ratio` | Conditional log odds or separate population-average ratios; check each function's scale |
| Count outcome | `fit_poisson`, `fit_poisson_bootstrap` | Conditional mean or rate ratio, or a population-average count ratio; observation times are supported where documented |
| Resampling and clustered data | `fit_bootstrap`, `fit_gaussian_ee`, `fit_binomial_ee`, `fit_poisson_ee` | Method-specific uncertainty for independent rows or identified clusters |
| Other outcomes | `fit_cox`, `fit_tobit`, `fit_multinomial`, zero-inflated and hurdle fitters | Restricted survival, censored, multicategory and two-part count models |
| Results and figures | `result.summary()`, `plot_weights`, `plot_response` | Saved estimates, diagnostics and selected displays |

The [reference manual](https://nilson01.github.io/quantilemix/manual/reference/) gives signatures, examples and
limits for every supported function. The [coverage map](https://github.com/nilson01/quantilemix/blob/main/docs/r-api-parity.md)
distinguishes implemented methods from open features.

For named data, `fit_formula` accepts additive terms such as
`y ~ x1 + x2 + C(group)`. `C(group)` encodes a text adjustment category.
Use `missing="drop"` only when you want incomplete training rows removed;
the default rejects them. Array inputs require complete numeric values.
Fitted exposure boundaries and category levels are reused for prediction.

Case weights give training rows different influence in supported fits.
They are different from the positive and negative *sign weights* in a result:
sign weights describe fitted exposure coefficients, not causal importance.
Neither kind of weight automatically accounts for a complex survey design.

## Evidence and limits

The [validation record](https://github.com/nilson01/quantilemix/blob/main/validation/RESULTS.md) links fixed comparisons and
repeated-dataset studies. The checks apply to their stated inputs and models:

| Method | Evidence | Limit to keep in mind |
| --- | --- | --- |
| Additive Gaussian | [Nine installed-R cases](https://github.com/nilson01/quantilemix/blob/main/docs/reference.md) and a [24,000-dataset known-score study](https://github.com/nilson01/quantilemix/blob/main/validation/GAUSSIAN_RESULTS.md) | The study did not cover estimated cutpoints or every model assumption. |
| Gaussian row bootstrap | [Fixed R comparisons](https://github.com/nilson01/quantilemix/blob/main/docs/reference.md) and a [24,000-dataset study](https://github.com/nilson01/quantilemix/blob/main/validation/BOOTSTRAP_RESULTS.md) | Three study cells had pointwise coverage intervals below 95%. |
| Binary and count models | [Method-specific reports](https://github.com/nilson01/quantilemix/blob/main/validation/RESULTS.md) | Some settings still lack repeated-dataset checks; sparse-count Poisson results suggest possible undercoverage. |
| Other families | [Coverage map](https://github.com/nilson01/quantilemix/blob/main/docs/r-api-parity.md) | Several marginal methods still need installed-R or repeated-dataset checks. |

The confidence intervals describe uncertainty under each fitted model and
study design. Software agreement with R does not establish general interval
coverage or a causal effect.

## Further information

- [Short tutorials](https://nilson01.github.io/quantilemix/) and [common questions](https://nilson01.github.io/quantilemix/faq/)
- [Markdown manual](https://nilson01.github.io/quantilemix/downloads/quantilemix-manual.md) and [reference PDF](https://nilson01.github.io/quantilemix/downloads/quantilemix-manual.pdf)
- [Method and numerical conventions](https://github.com/nilson01/quantilemix/blob/main/docs/method.md)
- [Contributing and development checks](https://github.com/nilson01/quantilemix/blob/main/CONTRIBUTING.md)
- [Attribution](https://github.com/nilson01/quantilemix/blob/main/NOTICE.md) and [GPL-2.0-or-later license](https://github.com/nilson01/quantilemix/blob/main/LICENSE)

## Citation

Chapagain, N. *QuantileMix: quantile g-computation in Python* [Computer software].
https://github.com/nilson01/quantilemix

```bibtex
@software{chapagain_quantilemix,
  author = {Chapagain, Nilson},
  title = {QuantileMix: quantile g-computation in Python},
  url = {https://github.com/nilson01/quantilemix}
}
```

Record the version or commit used and cite the statistical method used in the
analysis. Machine-readable metadata is in [CITATION.cff](https://github.com/nilson01/quantilemix/blob/main/CITATION.cff).
