# Package releases

The `Package release` workflow builds a wheel and source archive, checks metadata
and excludes private author files. Its default `check` target does not publish.
Publication requires all four CI configurations to pass for the selected commit.

## Account setup

Create separate [PyPI](https://pypi.org/account/register/) and
[TestPyPI](https://test.pypi.org/account/register/) accounts. Configure a pending
Trusted Publisher on each account:

| Field | Value |
| --- | --- |
| Project | `quantilemix` |
| Owner | `nilson01` |
| Repository | `quantilemix` |
| Workflow | `release.yml` |
| Environment | `testpypi` on TestPyPI; `pypi` on PyPI |

Create the matching GitHub environments. Require a human reviewer for the `pypi`
environment before enabling its publisher. Trusted Publishing uses short-lived
credentials; repository API-token secrets are not needed.

## Test installation

Run `Package release` on `main` with target `testpypi`. Use a new version for each
upload; index files cannot be overwritten. Check the uploaded description and
install the exact version in a fresh environment. For example:

```sh
python -m pip install numpy scipy
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps 'quantilemix==0.1.0.dev0'
python -c "from quantilemix import fit; print(fit.__name__)"
```

Install dependencies from PyPI first so the test does not mix package indexes.

## Publish a final release

1. Set a final version in `pyproject.toml`, update the changelog and manual, and
   merge the release changes after CI passes.
2. Tag that commit as `vVERSION`. Run `Package release` on the tag with target
   `pypi`, then review and approve the environment deployment.
3. Check the PyPI page and install that exact version in a fresh environment.
   Add the PyPI installation command to user documentation after publication.

See the [PyPA publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
and [pending-publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
