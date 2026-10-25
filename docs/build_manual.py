"""Build the reference PDF from the editable Markdown manual.

Requires Pandoc and a TeX installation with XeLaTeX. A local installation of
pypandoc_binary can supply Pandoc when it is not on PATH.
"""

import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent / "manual"


def main():
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        import pypandoc

        pandoc = pypandoc.get_pandoc_path()
    source = (HERE / "reference.md").read_text()
    source = re.sub(r'<a id="([^"]+)"></a>\s*\n(#{2,6} [^\n]+)',
                    r'\2 {#\1}', source)
    document = json.loads(subprocess.check_output([
        pandoc, "--from=markdown+tex_math_dollars", "--to=json",
    ], input=source, text=True))
    # The PDF title comes from metadata; its sections start one level lower.
    blocks = []
    for block in document["blocks"][1:]:
        if block["t"] == "Header":
            block["c"][0] -= 1
        blocks.append(block)
    document["blocks"] = blocks
    with tempfile.TemporaryDirectory(prefix="quantilemix-manual-") as temporary:
        source = Path(temporary) / "manual.json"
        source.write_text(json.dumps(document))
        subprocess.run([
            pandoc, str(source), "--from=json", "--pdf-engine=xelatex",
            "--resource-path=" + str(HERE), "--toc", "--toc-depth=3",
            "--metadata=title:QuantileMix Reference Manual", "--metadata=date:",
            "--variable=fontsize:10pt", "--variable=geometry:margin=1in",
            "--variable=colorlinks:true", "--variable=linkcolor:black",
            "--variable=urlcolor:blue", "--listings",
            r"--variable=header-includes:\lstset{basicstyle=\ttfamily\small,breaklines=true,columns=fullflexible,keepspaces=true}",
            "--output=" + str(HERE / "quantilemix-manual.pdf"),
        ], check=True, cwd=temporary)


if __name__ == "__main__":
    main()
