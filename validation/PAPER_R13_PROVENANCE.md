# Installed qgcomp 1.3.0 check for the cancellation discrepancy

This is a **post-study, fixed-input diagnostic** for the
four-cell [Gaussian paper comparison](PAPER_REPRODUCTION_RESULTS.md), not a
new repeated-dataset study. The paper reports qgcomp 1.3 results, while the
original optional reference pilot used installed qgcomp 2.19.6. We asked
whether those package versions produce different Gaussian estimates or
variances on the same small set of inputs.

We downloaded [qgcomp 1.3.0 from the CRAN archive](https://cran.r-project.org/src/contrib/Archive/qgcomp/qgcomp_1.3.0.tar.gz)
with SHA-256
`ef81862c126088d8ecb8b31c4c0759d82330118f4c91737d7a6e5f9fd544015d`.
It was installed in a separate library and loaded from that library; R
4.5.1 and the same dependency versions were used for both qgcomp versions.
The optional checker verified the installed package origin and saved MD5
values for its DESCRIPTION, NAMESPACE and installed R code files. The R
dependencies were arm 1.15.3, future 1.80.0, future.apply 1.21.0,
ggplot2 4.0.3, gridExtra 2.3.1 and jsonlite 2.0.0. No R package was added
to Python's runtime dependencies.

The 20 inputs comprise five independent datasets per scenario from the
protocol's **pilot** seed namespace. Both installed R versions and Python
used identical saved score and outcome arrays. For all 20 cases, qgcomp
1.3.0 and 2.19.6 returned **exactly equal** effect estimates and reported
variances at the saved numerical precision. The largest absolute Python/R
difference was `2.9924e-15`. Their per-case input hashes also agree. The
saved qgcomp 1.3.0 report SHA-256 is
`dfda7b0225ffc7eba401f40b31f78a7212a4a4f6ae718c4f9d13390bf71a913d`;
the matching 2.19.6 report SHA-256 is
`9c4ab2fcb5444ad349fecc4a1e426d5729a01058a231cb1e7080b3bef9f41a82`.
Both complete 20-case records are in the private project
`validation-local/paper-reproduction-20261023/` directory.
The read-only [version comparator](compare_paper_r_versions.py) checked all
20 ordered identities, input hashes and 40 saved R values. It reported a
maximum cross-version difference of zero. Its private result SHA-256 is
`5c196e3c6d0f4a8f692ad6f17185c98be4f2271037a129ee9d093509ecb8edce`.

This check rules out a numerical version difference **on these 20 inputs**.
It does not explain why the paper's rounded cancellation root mean variance
is 0.09 while our 1,000-run value is 0.0802. We also inspected the archived
1.3.0 source: its Gaussian `qgcomp.noboot` uses the fitted GLM covariance
and a unit-gradient variance for the sum of exposure coefficients. The
released supplement's later five-run qgcomp 2.1.3 cancellation illustration
has mean variance 0.006380424, whose square root is about 0.0799.
Neither a 20-case check nor a five-run illustration reconstructs the paper's
original 1,000 datasets or production analysis. The cause remains open;
the four-cell study's prespecified results and discrepancy are unchanged.

To repeat the optional fixed-input check, use the [reference checker](check_paper_r.py)
with an isolated installed package library, a dependency library and the
matching Rscript:

```sh
python validation/check_paper_r.py \
  --rscript /path/to/Rscript --r-library /path/to/qgcomp-1.3.0-library \
  --dependency-library /path/to/dependencies --r-version 1.3.0 \
  --output /path/to/fresh/report.json
```

After creating one report for each version, compare their saved values:

```sh
python validation/compare_paper_r_versions.py \
  /path/to/r-1.3.0-report.json /path/to/r-2.19.6-report.json \
  --output /path/to/fresh/version-comparison.json
```

The main reference checker defaults to qgcomp 2.19.6.
