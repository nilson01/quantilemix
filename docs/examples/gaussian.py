import numpy as np
from quantilemix import fit

x1, x2 = np.meshgrid(np.arange(4), np.arange(4))
X = np.column_stack([x1.ravel(), x2.ravel()])
y = 1 + 2*X[:, 0] - 0.5*X[:, 1] + 0.2*(X[:, 0] - 1.5)*(X[:, 1] - 1.5)

result = fit(y, X, q=None)
print(result.summary())
predicted_means = result.predict(X)
