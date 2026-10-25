# Changelog

## 0.1.1

- Use consistent package descriptions across the manual, website and package metadata.
- Add PyPI installation instructions alongside the source-install routes.

## 0.1.0

### Package name

- Rename the package and import namespace to `quantilemix`. Use
  `from quantilemix import fit` in Python. Model functions and statistical
  calculations retain their existing behavior.

### Methods

- Add quantile scoring and additive Gaussian, binary-logit and Poisson-log
  fits, with joint-effect estimates and uncertainty.
- Add fixed-cut row-bootstrap methods for selected Gaussian, binary, count
  and Cox effects. Restricted quadratic Gaussian methods allow selected
  exposure interactions.
- Add clustered Gaussian, binary and Poisson estimating-equation methods,
  conditional Cox fits, and narrower Tobit, multinomial, zero-inflated and
  hurdle fits. The [coverage map](docs/r-api-parity.md) gives each method's
  exact scope.
- Add selected pointwise comparisons, curve bounds, sample-split partial
  effects, synthetic-data and imputation helpers.

### Inputs and results

- Accept positive case weights in supported fits, named additive formulas,
  categorical adjustment variables and explicit complete-case removal where
  documented. Count and Cox methods have their own observation-time or
  offset options.
- Return saved quantile boundaries for prediction, joint estimates and
  uncertainty, descriptive sign weights, readable summaries and selected
  result tables. Two optional plot helpers display sign weights and fitted
  responses without refitting.
- Keep R out of the Python runtime. Optional development checks compare
  selected calculations with installed R qgcomp 2.19.6.

### Evidence and limits

- Independent calculations, fixed R comparisons and repeated-dataset studies
  are linked in the [validation record](validation/RESULTS.md). They establish
  agreement or interval behavior only for the stated cases and study designs.
- Some Gaussian bootstrap cells had low observed interval coverage. The
  sparse-count Poisson null cell showed possible undercoverage. Several
  marginal methods still lack installed-R comparisons or repeated-dataset
  coverage checks.
- The [PDF manual](docs/manual/quantilemix-manual.pdf) and
  [short tutorials](docs/site/index.md) describe the implemented interface. The manual gives a short guide to
  validation evidence; detailed study and execution records remain in
  [validation](validation/). Unsupported features and future candidates are
  tracked separately.
