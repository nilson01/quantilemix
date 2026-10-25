from pathlib import Path
import numpy as np
from quantilemix import fit_poisson

path = Path(__file__).resolve().parents[1] / "site" / "data" / "count.csv"
data = np.genfromtxt(path, delimiter=",", names=True)
X = np.column_stack([data["x1"], data["x2"]])
result = fit_poisson(data["y"], X, q=None)
print(result.summary())
print(f"Conditional mean ratio: {np.exp(result.psi):.3f}")
