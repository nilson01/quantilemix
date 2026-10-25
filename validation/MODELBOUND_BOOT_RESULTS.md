# Marginal fitted-curve bounds: installed-R check

Python modelbound_boot covers additive Gaussian and Poisson fixed-cut row-bootstrap fits with a known quantile count. It returns pointwise limits for the fitted marginal curve and the simultaneous limits computed by R qgcomp 2.19.6. Poisson observation-time offsets are included. This is a fixed-input software comparison, not a repeated-data coverage study.

The optional exporter calls modelbound.boot in an isolated installation of R qgcomp 2.19.6. It checks that the loaded package comes from the requested library, then fits three specified datasets with the same 24 resampled-row schedules used by Python. Each fit produces a pointwise-only table and a table with simultaneous limits. The [saved fixture](../tests/fixtures/r_modelbound_boot_2_19_6.json) has SHA-256 c5d3c44f5ed633a331c71952f64785478d6f574f7410acd222784f74d92cbd15.

| Case | Joint score levels | Bootstrap draws | Link | Tables checked |
| --- | ---: | ---: | --- | ---: |
| Gaussian additive | 3 | 24 | Identity | 2 |
| Poisson additive | 4 | 24 | Log | 2 |
| Poisson with unequal observation times | 4 | 24 | Log | 2 |

The installed-R checker compares the inputs, row schedules, original marginal coefficients, every bootstrap coefficient vector, their covariance, fitted means, table column names and all six tables against both the saved fixture and Python. All comparisons pass with an absolute tolerance of 2e-8 for Python/R; the largest observed table difference is 2.15e-9. The R fixture and live re-export agree within 5e-10. Independent Python tests check pointwise normal-limit arithmetic, log versus identity scales, read-only output, omitted simultaneous limits, invalid inputs, a singular coefficient covariance and a nonlinear profile that cannot be represented by the supported degree-one curve.

For each score, the pointwise standard error is the sample standard deviation of bootstrap marginal predictions. Poisson predictions use the log scale for this calculation. The simultaneous calculation takes bootstrap intercept-and-slope vectors inside a chi-square ellipsoid around the original fitted vector, then finds the lowest and highest retained curve values at each score. The Gaussian result is on the mean scale; Poisson limits are exponentiated to expected counts. With pwonly=True, the simultaneous columns are missing, as in R. These are reference-method bounds; neither a fixed-data match nor the earlier joint-effect studies establish their simultaneous repeated-data coverage.

The optional check can be rerun from an installed Python development environment with the matching R executable and an isolated qgcomp 2.19.6 library:

    python validation/check_modelbound_boot_r.py --rscript /path/to/Rscript --library /path/to/isolated-R-library

The Python package itself does not require R. Binary, nonlinear, clustered and other marginal bootstrap families, as well as repeated-data pointwise and simultaneous coverage, remain open.
