# Development guide

See [Contributing](../CONTRIBUTING.md) for setup and required checks.

## Documentation

Install the documentation tools from a checkout:

```sh
python -m pip install '.[docs,plot]'
python docs/check_examples.py
cffconvert --validate
python -m mkdocs build --strict
```

Run `python -m mkdocs serve` to preview the searchable site locally.
[GitHub Pages](https://nilson01.github.io/quantilemix/) supports either the
checked `main` deployment workflow or prebuilt files on `gh-pages`. The active
publishing source is selected in the repository's Pages settings. For branch
publication, run the checks above, then:

```sh
python -m mkdocs gh-deploy --clean --strict
```

Select GitHub Actions as the Pages source to use the automatic `main`
deployment job. Pull requests cannot deploy. Compiled manuals are copied into
the site at build time; TeX sources remain separate.
API pages use the public docstrings. Edit the small tutorial scripts in
`docs/examples/` and rerun the checks when changing an example.

### Reference manual

Edit `docs/manual/reference.md`; the PDF is a reference copy. Its two images
are in `docs/manual/images/`. With Pandoc and a TeX installation providing
XeLaTeX, rebuild it with:

```sh
python docs/build_manual.py
```

`pypandoc_binary` can supply Pandoc when it is not on PATH. These build tools
are separate from the package's runtime dependencies. Check the equations,
tables, code blocks and rendered PDF after changing the manual.

## Dependency and platform checks

The CI matrix keeps the existing Linux Python 3.10 and 3.14 jobs. It also checks
minimum dependencies on Linux Python 3.10 and current compatible dependencies on
Windows Python 3.14. At most two jobs run together. Each job runs the full suite,
builds both distributions and tests a new wheel installation outside the checkout.
These lanes test selected combinations, not every version allowed by the metadata.

[minimum-constraints.txt](../validation/minimum-constraints.txt) pins the
numerical libraries, test tools and build backend. Install those pins first,
then build with `--no-isolation` so the backend uses the pinned version.

To reproduce in a new Python 3.10, 3.11 or 3.12 development environment:

```sh
python -m pip install --only-binary=:all: -r validation/minimum-constraints.txt
python -m pip install --no-build-isolation -c validation/minimum-constraints.txt '.[dev]'
python -m pytest -q
python -m ruff check src tests validation
python -m build --no-isolation
python validation/check_wheel_install.py --minimum
```

NumPy 2.0.0 supports Python through 3.12, so the minimum check cannot run on
Python 3.14. See the [NumPy release notes](https://numpy.org/doc/2.0/release/2.0.0-notes.html).
For current compatible dependencies, use the normal development installation,
`python -m build`, then `python validation/check_wheel_install.py`.

The wheel checker imports the package from a new environment outside the
checkout. With `--minimum`, it also verifies the installed NumPy, SciPy and
pytest versions. Its report, `validation/wheel-check.json`, records versions,
the wheel hash, test counts and skips. CI retains it as an artifact.

Git preserves LF line endings so reference-file hashes remain stable on
Windows. Live R comparisons run separately from Python CI.

## Gaussian calibration runner

The runner follows the [known-score protocol](../validation/GAUSSIAN_PROTOCOL.md):
12 conditions, 2,000 datasets per condition, Python and installed R fits, and
an independent least-squares contrast. A separate 20-case pilot is excluded
from the study. See the [completed results](../validation/GAUSSIAN_RESULTS.md).

Inspect the configuration and planned workload without fitting or starting R:

```sh
python validation/calibration.py plan
```

Execution needs a Python development environment plus an isolated R library
containing qgcomp 2.19.6 and jsonlite. The
[installed-R instructions](reference.md#installed-r-gaussian-check) describe
the dependency boundary. The runner does not install anything. Python package
runtime dependencies remain unchanged. Use one numerical-library thread and
check storage and memory before a pilot or study.
Set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS` and `NUMEXPR_NUM_THREADS` to `1` before starting
Python. Execution rejects missing or different limits; setting them after
NumPy has loaded may be too late to control its numerical libraries.

To run the timing pilot, supply local paths and a new output folder:

```sh
python validation/calibration.py pilot --execute \
  --rscript /path/to/Rscript --r-library /path/to/isolated-library \
  --output /path/to/pilot-output
```

Use `study` for the full design and a separate output directory. Use
`--resume` only for an interrupted run with the same configuration, source
and environment. Failed fits remain in the record. An exclusive writer lock
prevents concurrent execution in one directory; remove a stale lock only
after confirming that its process has stopped.

Inspect saved records without new fits:

```sh
python validation/calibration.py aggregate --output /path/to/output
```

Keep generated inputs, logs and result directories outside the repository.
The source distribution includes the protocol, configuration and runner.

### Independently check completed study records

The checker uses only the Python standard library. It reads saved
results without starting R, refitting models or generating data:

```sh
python validation/check_calibration_results.py RUN_DIRECTORY NEW_REPORT.json
```

It accepts only the fixed `gaussian-known-scores-v1` study and its recorded
source hashes from revision `8cb4619934e1cf02ed8ade3b09f09b25115f9381`.
The required evidence is 1,200 completed batches containing all 24,000 planned
case identities, plus the manifest and saved summary. Pilot or partial runs,
pending files and an existing writer lock are rejected. A changed study source
needs a separately reviewed identity; it cannot use this check unchanged.

Replace `RUN_DIRECTORY` with the completed run folder. `NEW_REPORT.json` must
be a new file outside that folder, with an existing parent directory. The
checker leaves the run unchanged and never replaces an earlier report.

The checker verifies case identities, seeds and retained input hashes. It
independently recomputes method counts, numerical comparisons, bias, uncertainty
and coverage summaries, then compares them with the saved summary. Failed and
unavailable results remain in their stated denominators. Exit code 0 means
the saved records and summary agree; the report can still describe completed
execution with fitting failures or numerical mismatches. The report always
keeps `calibration_validated` false. Exit code 2 indicates rejected evidence
or an invalid output request.

The checker loads one batch at a time. For transferred results, compare file
hashes with a separately retained source inventory. Record agreement does
not establish statistical calibration or restore the original environment.

### Plot the saved coverage results

The optional plot tool reads a reviewed public JSON record containing the
production summary under `summary`. Use a separate authoring environment with
Matplotlib, Pillow and pypdf; these are not package runtime dependencies.

```sh
python validation/plot_calibration_results.py PUBLIC_RECORD.json NEW_DIRECTORY
```

The output directory must be new and its parent must exist. The tool creates
PNG and PDF figures, a CSV table with all 36 method-by-condition rows, and a
provenance record with source/output hashes and actual dependency versions.
It draws saved coverage and Wilson intervals without fitting models or drawing
data. Both panels use the same scale, and failures remain visible in the counts.
Inspect the exported figure before using it; plotting checks do not replace
independent verification or scientific interpretation.

## Gaussian result display

`QGCompResult.summary(*, digits=6)` returns text from saved fields without
refitting or changing inference. Full-precision attributes remain available
for calculations. Display tests cover the nine fixed R cases, independent
arithmetic, small p-values, undefined inference, sign groups and diagnostics.
See the [reference record](reference.md#run-the-comparison) for the R summary
comparison and its missing t column.
