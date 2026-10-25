# Installation

QuantileMix requires Python 3.10 or newer. NumPy and SciPy install as dependencies.

## From GitHub

With Git installed and access to the repository:

```sh
python -m pip install "git+https://github.com/nilson01/quantilemix.git@main"
```

## From a local checkout

In the folder containing `pyproject.toml`:

```sh
python -m pip install .
```

To draw plots, install the plot dependencies from that checkout:

```sh
python -m pip install '.[plot]'
```

Check the import:

```python
from quantilemix import fit
```

[Fit a continuous outcome](tutorials/gaussian.md) next.
