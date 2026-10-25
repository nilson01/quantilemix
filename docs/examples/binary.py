from pathlib import Path
import numpy as np
from quantilemix import fit_binomial

path = Path(__file__).resolve().parents[1] / "site" / "data" / "binary.csv"
data = np.genfromtxt(path, delimiter=",", names=True)
X = np.column_stack([data["x1"], data["x2"]])
result = fit_binomial(data["y"], X, q=None)
print(result.summary())
print(f"Conditional odds ratio: {np.exp(result.psi):.3f}")
