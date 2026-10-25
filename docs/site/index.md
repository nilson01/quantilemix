# QuantileMix

Estimate how an outcome changes when several exposure scores increase together.
QuantileMix fits selected models for continuous, binary, count and survival
outcomes. NumPy and SciPy handle fitting and prediction; R is not required.

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
