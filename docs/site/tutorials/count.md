# Count outcomes

Counts must be nonnegative integers. This example uses equal observation times. The [example CSV](../data/count.csv)
is a small constructed dataset from a fixed installed-R comparison. Its exposure
scores are already coded as 0 and 1, so the fit uses `q=None`.

Save the CSV as `count.csv` beside your script, then run:

```python
import numpy as np
from quantilemix import fit_poisson

data = np.genfromtxt("count.csv", delimiter=",", names=True)
X = np.column_stack([data["x1"], data["x2"]])
result = fit_poisson(data["y"], X, q=None)
print(result.summary())
print(f"Conditional mean ratio: {np.exp(result.psi):.3f}")
```

The final line is:

```text
Conditional mean ratio: 6.000
```

`psi` is on the log scale. `exp(psi)` is the conditional mean ratio for
increasing both scores by one, holding adjustment variables fixed.
For rates over unequal observation times, supply `exposure_time` as
described in [fit_poisson](../reference/count.md#quantilemix.fit_poisson).

These data demonstrate a function call; they are not a research application.
[Download the runnable example](https://github.com/nilson01/quantilemix/blob/main/docs/examples/count.py).
