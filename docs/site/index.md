# QuantileMix

Quantile g-computation transforms exposures into ordinal quantile scores and estimates the joint effect of increasing all exposures simultaneously by one quantile. The package supports selected models for continuous, binary, count, and survival outcomes, with more limited support for censored, multicategory, and zero-inflated outcomes. Each API entry documents the effect scale, required inputs, and methodological limitations.

Quantile cutpoints estimated during model fitting are reused for prediction. Sign-specific weights summarize fitted exposure coefficients rather than causal importance. Successful model fitting does not establish the validity of model assumptions or identify a causal effect without additional causal identification assumptions.

| Task | Start here |
| --- | --- |
| Install the package | [Installation](install.md) |
| Fit and interpret a model | [Continuous](tutorials/gaussian.md), [binary](tutorials/binary.md), [count](tutorials/count.md) tutorials |
| Find a function | [API by analysis](reference/index.md) |
| Check assumptions and evidence | [Common questions](faq.md) and [validation](evidence.md) |
| Cite the software | [Citation](citation.md) |

![Estimated joint mean change with a pointwise 95% confidence band.](assets/joint-effect.svg)

The Gaussian example compares all scores set to 0 with all scores set to
1, 2 or 3. The band describes uncertainty in the fitted mean contrast.
It is not an interval for an individual outcome or evidence of causality.

The [reference manual](manual/reference.md) contains the full method descriptions
and worked examples. Download its [Markdown source](https://nilson01.github.io/quantilemix/downloads/quantilemix-manual.md)
or [reference PDF](downloads/quantilemix-manual.pdf).
See the [license](https://nilson01.github.io/quantilemix/downloads/LICENSE.txt) and [attribution](https://nilson01.github.io/quantilemix/downloads/NOTICE.md.txt).
