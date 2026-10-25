# Evidence and limits

| Question | Evidence |
| --- | --- |
| Do selected calculations agree with R? | [Fixed installed-R comparisons](https://github.com/nilson01/quantilemix/blob/main/docs/reference.md) |
| How do intervals behave across datasets? | [Study results by method](https://github.com/nilson01/quantilemix/blob/main/validation/RESULTS.md) |
| What assumptions and options are supported? | [Coverage map](https://github.com/nilson01/quantilemix/blob/main/docs/r-api-parity.md) |
| How can I rerun the checks? | [Development guide](https://github.com/nilson01/quantilemix/blob/main/docs/development.md) |

Fixed agreement checks do not establish general interval coverage. The Gaussian
known-score studies do not cover estimated cutpoints or every model assumption.
The row-bootstrap study found undercoverage in some conditions. Other methods
have their own evidence and limitations in the linked reports.

The method follows [Keil et al. (2020)](https://doi.org/10.1289/EHP5838).
QuantileMix is an unofficial Python implementation; the method and R-package
authors have not endorsed it.
