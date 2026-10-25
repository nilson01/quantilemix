# Binary outcomes

Binary outcomes must be 0 or 1. The [example CSV](../data/binary.csv)
is a small constructed dataset from a fixed installed-R comparison. Its exposure
scores are already coded as 0 and 1, so the fit uses `q=None`.

Save the CSV as `binary.csv` beside your script, then run:

```python
import numpy as np
from quantilemix import fit_binomial

data = np.genfromtxt("binary.csv", delimiter=",", names=True)
X = np.column_stack([data["x1"], data["x2"]])
result = fit_binomial(data["y"], X, q=None)
print(result.summary())
print(f"Conditional odds ratio: {np.exp(result.psi):.3f}")
```

The final line is:

```text
Conditional odds ratio: 9.000
```

`psi` is on the log scale. `exp(psi)` is the conditional odds ratio for
increasing both scores by one, holding adjustment variables fixed.
This is an odds ratio, not a risk ratio. For population-average effects,
use the separate [risk-ratio or odds-ratio methods](../reference/binary.md).

These data demonstrate a function call; they are not a research application.
[Download the runnable example](https://github.com/nilson01/quantilemix/blob/main/docs/examples/binary.py).
