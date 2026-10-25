"""Publish the reader manual and its editable Markdown source."""

from pathlib import Path
import re
import shutil

from mkdocs.structure.files import File


ROOT = Path(__file__).resolve().parent


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == "manual/reference.md":
        return re.sub(r'<a id="([^"]+)"></a>\s*\n(#{2,6} [^\n]+)',
                      r'\2 {#\1}', markdown)
    return markdown


def on_files(files, config):
    for name in ["quantilemix-manual.pdf"]:
        files.append(File.generated(config, f"downloads/{name}",
                                    abs_src_path=str(ROOT / "manual" / name)))
    files.append(File.generated(config, "manual/reference.md",
                                abs_src_path=str(ROOT / "manual/reference.md")))
    for source in sorted((ROOT / "manual/images").glob("*.png")):
        files.append(File.generated(config, f"manual/images/{source.name}",
                                    abs_src_path=str(source)))
    return files


def on_post_build(config):
    downloads = Path(config.site_dir) / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "manual/reference.md", downloads / "quantilemix-manual.md")
    shutil.copytree(ROOT / "manual/images", downloads / "images", dirs_exist_ok=True)
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copy2(ROOT.parent / name, downloads / (name + ".txt"))
