"""Check published examples, their displayed output and public API navigation."""

import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import tempfile
import os

import numpy as np
from numpy.testing import assert_allclose
import quantilemix


ROOT = Path(__file__).resolve().parents[1]


def blocks(path, language):
    return re.findall(r"```" + language + r"\n(.*?)```", path.read_text(), re.S)


def execute(code):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        scope = {}
        exec(compile(code, "<documentation example>", "exec"), scope)
    return scope, stream.getvalue()


def main():
    readme = ROOT / "README.md"
    with tempfile.TemporaryDirectory() as temporary:
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            namespace, output = execute(blocks(readme, "python")[0])
            for excerpt in blocks(readme, "text"):
                assert excerpt.strip() in output, "README output is stale"
            assert_allclose(namespace["result"].psi, 1.5, rtol=0, atol=1e-12)
            for kind, family, expected_ratio in [("binary", "binomial", 9), ("count", "poisson", 6)]:
                data_path = ROOT / "docs/site/data" / f"{kind}.csv"
                data = np.genfromtxt(data_path, delimiter=",", names=True)
                fixture = json.loads((ROOT / f"tests/fixtures/r_{family}_2_19_6.json").read_text())["cases"][0]
                for key in ["y", "x1", "x2"]:
                    assert_allclose(data[key], fixture["inputs"][key], rtol=0, atol=0)
                Path(f"{kind}.csv").write_bytes(data_path.read_bytes())
                page = ROOT / "docs/site/tutorials" / f"{kind}.md"
                scope, output = execute(blocks(page, "python")[0])
                for excerpt in blocks(page, "text"):
                    assert excerpt.strip() in output, f"{kind} tutorial output is stale"
                assert_allclose(np.exp(scope["result"].psi), expected_ratio, rtol=1e-8)
                with contextlib.redirect_stdout(io.StringIO()):
                    script = runpy.run_path(str(ROOT / "docs/examples" / f"{kind}.py"))
                assert_allclose(script["result"].psi, scope["result"].psi, rtol=0, atol=1e-12)
        finally:
            os.chdir(previous)
    with contextlib.redirect_stdout(io.StringIO()):
        gaussian = runpy.run_path(str(ROOT / "docs/examples/gaussian.py"))
    page = ROOT / "docs/site/tutorials/gaussian.md"
    for excerpt in blocks(page, "text"):
        assert excerpt.strip() in gaussian["result"].summary()
    source = (ROOT / "docs/site/assets/joint-effect.csv")
    plotted = np.genfromtxt(source, delimiter=",", names=True)
    result = gaussian["result"]
    assert_allclose(plotted["mean_difference"], plotted["score"] * result.psi)
    assert_allclose(plotted["lower_95"], plotted["score"] * result.ci[0])
    assert_allclose(plotted["upper_95"], plotted["score"] * result.ci[1])
    references = "\n".join(p.read_text() for p in (ROOT / "docs/site/reference").glob("*.md"))
    names = re.findall(r"^::: quantilemix\.([A-Za-z0-9_]+)$", references, re.M)
    assert len(names) == len(set(names)), "Duplicate API entry"
    assert set(names) == set(quantilemix.__all__), "Public API entry missing or stale"
    print(f"README, 3 tutorials, fixed CSVs, plot data and {len(names)} API entries pass.")


if __name__ == "__main__":
    main()
