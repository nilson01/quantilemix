# Validation

- [Study results](RESULTS.md): repeated-dataset findings and their limits.
- [R reference checks](../docs/reference.md): fixed-input software comparisons.
- [Coverage map](../docs/r-api-parity.md): supported methods and open work.
- [Development guide](../docs/development.md): dependencies and reproduction commands.

Each study report links its protocol. Earlier records retain the former
Python package name, `qgcomp-python`, and original source and artifact hashes.
Current code imports `quantilemix`; R references name `qgcomp`.

Slurm templates require site-specific module, account and quota commands.
They use a Python environment at `python/` in the submission folder, unless
`QGCOMP_PYTHON` is supplied through the scheduler environment. Recorded source
hashes identify the original study scripts; these portable templates do not
change saved results.
