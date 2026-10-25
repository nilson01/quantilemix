# Contributing

QuantileMix implements selected quantile g-computation methods. See the
[coverage map](docs/r-api-parity.md) before proposing a feature or changing
an existing method.

## Set up and check

Use an isolated Python environment:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python -m ruff check src tests validation docs/examples docs/check_examples.py docs/hooks.py docs/build_manual.py
python -m build
python validation/check_wheel_install.py
```

For minimum dependencies and platform checks, follow the
[development guide](docs/development.md#dependency-and-platform-checks).
Packaging changes require tests from a fresh wheel outside the source
checkout. Report unavailable environments as unverified.

For documentation changes, also run the [site checks](docs/development.md#documentation).
Use the [bug report](https://github.com/nilson01/quantilemix/issues/new?template=bug_report.yml)
or [feature request](https://github.com/nilson01/quantilemix/issues/new?template=feature_request.yml)
form for new issues.

## Submit a focused change

Use a feature branch and pull request. Describe the problem, resulting
behavior and checks performed. Keep code, tests, examples and documentation
consistent. Document changes to inputs, defaults, output fields, warnings
or errors; preserve unaffected public behavior.

The protected `main` branch requires the configured checks and resolved
review conversations. Force-pushes are disabled. New estimands or model
families need a defined statistical target and independent checks.

## Check statistical changes

Use independent known answers, invariance checks, invalid-input cases and
fixed R comparisons where applicable. Preserve reference versions, inputs,
numerical precision and justified tolerances. Do not replace expected values
only to make a failed test pass.

Source-function comparisons, installed-R comparisons and repeated-dataset
studies answer different questions. Fixed agreement does not establish
confidence-interval coverage or causal validity. Record intentional differences
from the reference and retain failed checks. R is needed only for the
[reference tools](docs/reference.md), never for Python fitting.

## Write and credit clearly

Use familiar words and short sentences. Explain necessary terms, symbols,
array shapes, column order and effect scales. Public docstrings use NumPy
style, with parameters, returns, errors and small runnable examples.

Preserve method-paper citations, R-software credit, [NOTICE.md](NOTICE.md)
and [LICENSE](LICENSE). Keep credentials, private data, correspondence and
build caches out of commits and distributions. Pull requests and documentation
should explain the software rather than record personal working instructions.
