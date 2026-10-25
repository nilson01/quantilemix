# SPDX-License-Identifier: GPL-2.0-or-later
"""Plot a verified public Gaussian study summary without running any models.

Usage: python validation/plot_calibration_results.py PUBLIC_RECORD.json NEW_DIRECTORY

Use a separate figure environment with Matplotlib, Pillow and pypdf. These are
optional authoring dependencies, not package runtime dependencies. The source
must contain ``{"summary": <production summary.json object>}`` for the complete
12-cell, 24,000-dataset study. Independently check the original study records
with check_calibration_results.py before preparing this public record.

The figure uses saved coverage and saved 95% Wilson intervals. It checks their
count consistency, but does not establish source authenticity or calibration.
Failed fits remain visible in every cell's counts; coverage is conditional on
successful fits. Partial studies and unsupported designs fail clearly.

Outputs are coverage.png, coverage.pdf, coverage.csv and provenance.json in a
new directory. No output can overwrite an existing file or directory. Publisher
requirements and visual review remain pending until actual outputs are checked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import sys


METHODS = ("python", "r", "qr_t")
METHOD_LABELS = ("Python (normal)", "R qgcomp (normal)", "QR contrast (t)")
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
OFFSETS = (-0.22, 0.0, 0.22)
BETAS = ("null", "cancellation", "positive")
SAMPLE_SIZES = (100, 500)
PLANNED_PER_CELL = 2000
FIGURE_INCHES = (12.0, 7.5)
DPI = 300
MAX_SOURCE_BYTES = 8_000_000


class PlotError(ValueError):
    """Unsupported or inconsistent evidence, or an unsafe output request."""


def require(condition, message):
    if not condition:
        raise PlotError(message)


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "source contains a duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise PlotError("source contains a nonfinite JSON number")


def finite_tree(value):
    """Reject overflow such as JSON 1e999, including in unplotted fields."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float):
            require(math.isfinite(item), "source contains a nonfinite JSON number")


def number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    require(valid, f"{label}: expected a finite number")
    return float(value)


def integer(value, label):
    require(type(value) is int and value >= 0, f"{label}: expected a nonnegative integer")
    return value


def close(actual, expected, label, tolerance=1e-12):
    require(math.isclose(number(actual, label), expected, rel_tol=0, abs_tol=tolerance),
            f"{label}: inconsistent saved value")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def wilson_check(covered, successful):
    """Compute the 95% Wilson interval only to check the saved interval."""
    z = statistics.NormalDist().inv_cdf(0.975)
    p = covered / successful
    scale = 1 + z*z/successful
    center = (p + z*z/(2*successful)) / scale
    half = z*math.sqrt(p*(1-p)/successful + z*z/(4*successful**2)) / scale
    return max(0.0, center-half), min(1.0, center+half)


def checked_method(entry, label):
    require(isinstance(entry, dict), f"{label}: missing method summary")
    counts = {key: integer(entry.get(key), f"{label}.{key}") for key in (
        "planned", "attempted", "successful", "failed", "missing", "unattempted", "covered")}
    require(counts["planned"] == counts["attempted"] == PLANNED_PER_CELL,
            f"{label}: expected all 2,000 planned datasets to be attempted")
    require(counts["missing"] == counts["unattempted"] == 0,
            f"{label}: incomplete evidence cannot be plotted")
    successful, failed, covered = (counts[key] for key in ("successful", "failed", "covered"))
    require(successful + failed == PLANNED_PER_CELL and covered <= successful,
            f"{label}: invalid success, failure or coverage counts")
    for key in ("failure_classes", "failure_codes"):
        failures = entry.get(key)
        require(isinstance(failures, dict), f"{label}.{key}: missing failure accounting")
        require(all(isinstance(name, str) and name for name in failures),
                f"{label}.{key}: invalid failure labels")
        require(sum(integer(count, f"{label}.{key}") for count in failures.values()) == failed,
                f"{label}.{key}: inconsistent failure accounting")
    close(entry.get("failed_fraction_planned"), failed/PLANNED_PER_CELL,
          f"{label}.failed_fraction_planned")
    close(entry.get("missing_fraction_planned"), 0, f"{label}.missing_fraction_planned")
    close(entry.get("covered_fraction_planned"), covered/PLANNED_PER_CELL,
          f"{label}.covered_fraction_planned")
    bounds = entry.get("coverage_all_planned_bounds")
    require(isinstance(bounds, list) and len(bounds) == 2, f"{label}: missing planned bounds")
    for actual, expected in zip(bounds, (covered/PLANNED_PER_CELL, (covered+failed)/PLANNED_PER_CELL)):
        close(actual, expected, f"{label}.coverage_all_planned_bounds")
    coverage, interval = entry.get("coverage_successful"), entry.get("coverage_wilson")
    if successful:
        close(coverage, covered/successful, f"{label}.coverage_successful")
        require(isinstance(interval, list) and len(interval) == 2, f"{label}: missing Wilson interval")
        for actual, expected in zip(interval, wilson_check(covered, successful)):
            close(actual, expected, f"{label}.coverage_wilson")
        require(0 <= interval[0] <= coverage <= interval[1] <= 1,
                f"{label}: invalid coverage interval")
    else:
        require(coverage is None and interval is None,
                f"{label}: coverage must be undefined when all fits fail")
    return {**counts, "coverage": coverage, "wilson": interval,
            "planned_coverage_bounds": bounds}


def checked_mismatch_cases(entry, checked, label):
    mismatch = integer(entry.get("mismatch"), f"{label}.mismatch")
    names = entry.get("mismatching_cases")
    require(mismatch <= checked and isinstance(names, list)
            and all(isinstance(name, str) and name for name in names),
            f"{label}: invalid mismatch accounting")
    require(len(names) == len(set(names)) == mismatch,
            f"{label}: mismatch names disagree with their count")
    return mismatch


def checked_comparisons(cell, methods, label):
    """Check the comparison counts used to label the figure's study status."""
    paired = cell.get("paired")
    require(isinstance(paired, dict) and set(paired) == {"python_vs_r", "python_vs_qr_t"},
            f"{label}: missing paired comparison accounting")
    mismatches = 0
    for other in ("r", "qr_t"):
        name = f"python_vs_{other}"
        entry = paired[name]
        require(isinstance(entry, dict), f"{label}.{name}: missing comparison counts")
        counts = {key: integer(entry.get(key), f"{label}.{name}.{key}") for key in (
            "planned", "agree", "mismatch", "unavailable", "missing")}
        require(counts["planned"] == PLANNED_PER_CELL and counts["missing"] == 0,
                f"{label}.{name}: incomplete comparison accounting")
        checked = counts["agree"] + counts["mismatch"]
        require(checked + counts["unavailable"] == PLANNED_PER_CELL,
                f"{label}.{name}: inconsistent comparison totals")
        left, right = methods["python"]["successful"], methods[other]["successful"]
        require(max(0, left+right-PLANNED_PER_CELL) <= checked <= min(left, right),
                f"{label}.{name}: comparison availability conflicts with successful fits")
        mismatches += checked_mismatch_cases(entry, checked, f"{label}.{name}")
    conventions = cell.get("reference_conventions")
    require(isinstance(conventions, dict), f"{label}: missing R convention accounting")
    checked = integer(conventions.get("checked"), f"{label}.reference_conventions.checked")
    unavailable = integer(conventions.get("unavailable"), f"{label}.reference_conventions.unavailable")
    require(checked == methods["r"]["successful"] and checked+unavailable == PLANNED_PER_CELL,
            f"{label}: R convention availability conflicts with successful fits")
    return mismatches + checked_mismatch_cases(conventions, checked, f"{label}.reference_conventions")


def read_summary(source):
    require(source.is_file() and not source.is_symlink(), "source must be a regular JSON file")
    require(source.stat().st_size <= MAX_SOURCE_BYTES, "source exceeds the 8 MB input limit")
    raw = source.read_bytes()
    require(len(raw) <= MAX_SOURCE_BYTES, "source exceeds the 8 MB input limit")
    record = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    finite_tree(record)
    require(isinstance(record, dict) and isinstance(record.get("summary"), dict),
            "source must contain a production summary object under 'summary'")
    summary = record["summary"]
    require(type(summary.get("schema_version")) is int and summary["schema_version"] == 1,
            "unsupported summary schema")
    require(summary.get("mode") == "study" and summary.get("execution_complete") is True,
            "only the completed full study is supported; pilot and partial results are rejected")
    require(summary.get("status") in ("complete_numerical_agreement", "completed_with_failures_or_mismatches"),
            "unsupported study status")
    require(summary.get("calibration_validated") is False,
            "expected the production descriptive summary, not a calibration certification")
    for key, expected in (("planned", 24000), ("attempted", 24000), ("terminal", 24000),
                          ("missing", 0), ("completed_batches", 1200)):
        require(integer(summary.get(key), key) == expected, f"{key}: incomplete or unsupported study")
    identity = summary.get("identity_sha256")
    require(isinstance(identity, str) and len(identity) == 64
            and all(character in "0123456789abcdef" for character in identity), "invalid study identity hash")
    cells = summary.get("cells")
    require(isinstance(cells, list) and len(cells) == 12, "expected all twelve study cells")
    indexed = {}
    for cell in cells:
        require(isinstance(cell, dict) and isinstance(cell.get("cell"), str), "invalid study cell")
        require(cell["cell"] not in indexed, "duplicate study cell")
        indexed[cell["cell"]] = cell
    rows, targets, mismatches = [], {}, 0
    for n in SAMPLE_SIZES:
        for rho, rho_name in ((0.0, "0"), (0.75, "0p75")):
            for beta in BETAS:
                identifier = f"n{n}-rho{rho_name}-{beta}"
                require(identifier in indexed, f"missing expected cell {identifier}")
                cell = indexed[identifier]
                close(cell.get("true_psi"), 0.25 if beta == "positive" else 0, f"{identifier}.true_psi")
                analytic = cell.get("analytic")
                require(isinstance(analytic, dict), f"{identifier}: missing analytic targets")
                require(analytic.get("condition") == "correct normal homoscedastic model and full-rank design",
                        f"{identifier}: unsupported analytic assumptions")
                require(integer(analytic.get("df_residual"), "df_residual") == n-7,
                        f"{identifier}: unsupported residual degrees of freedom")
                close(analytic.get("reference_coverage"), 0.94700655 if n == 100 else 0.94943710,
                      f"{identifier}.normal_target", tolerance=5.01e-9)
                close(analytic.get("qr_t_coverage"), 0.95, f"{identifier}.t_target")
                target = analytic["reference_coverage"]
                if n in targets:
                    close(target, targets[n], f"{identifier}.normal_target")
                targets[n] = target
                methods = cell.get("methods")
                require(isinstance(methods, dict) and set(methods) == set(METHODS),
                        f"{identifier}: expected all three methods")
                method_counts = {}
                for method in METHODS:
                    result = checked_method(methods[method], f"{identifier}.{method}")
                    method_counts[method] = result
                    rows.append({"cell": identifier, "n": n, "rho": rho, "beta": beta,
                                 "method": method, "normal_target": target, "t_target": 0.95, **result})
                mismatches += checked_comparisons(cell, method_counts, identifier)
    expected_status = ("completed_with_failures_or_mismatches"
                       if mismatches or any(row["failed"] for row in rows)
                       else "complete_numerical_agreement")
    require(summary["status"] == expected_status,
            "study status conflicts with recorded failures or mismatches")
    return summary, rows, targets, raw


def write_table(rows, destination):
    fields = ["cell", "n", "rho", "beta", "method", "planned", "attempted", "successful",
              "failed", "missing", "unattempted", "covered", "coverage_successful_percent",
              "coverage_wilson_low_percent", "coverage_wilson_high_percent",
              "coverage_all_planned_lower_percent", "coverage_all_planned_upper_percent",
              "normal_analytic_target_percent", "t_analytic_target_percent"]
    with destination.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            interval = row["wilson"]
            writer.writerow({**{key: row[key] for key in fields if key in row},
                             "coverage_successful_percent": 100*row["coverage"] if interval else "",
                             "coverage_wilson_low_percent": 100*interval[0] if interval else "",
                             "coverage_wilson_high_percent": 100*interval[1] if interval else "",
                             "coverage_all_planned_lower_percent": 100*row["planned_coverage_bounds"][0],
                             "coverage_all_planned_upper_percent": 100*row["planned_coverage_bounds"][1],
                             "normal_analytic_target_percent": 100*row["normal_target"],
                             "t_analytic_target_percent": 100*row["t_target"]})


def draw_figure(rows, targets, status, output, plt, Line2D):
    endpoints = [100*value for row in rows if row["wilson"] for value in row["wilson"]]
    endpoints.extend([95.0, *(100*value for value in targets.values())])
    padding = max(0.5, 0.08*(max(endpoints)-min(endpoints)))
    limits = (max(0, math.floor((min(endpoints)-padding)*2)/2),
              min(100, math.ceil((max(endpoints)+padding)*2)/2))
    styles = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
              "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white",
              "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "#111111"}
    with plt.rc_context(styles):
        fig = plt.figure(figsize=FIGURE_INCHES)
        grid = fig.add_gridspec(2, 2, height_ratios=(4.0, 1.0), left=0.075, right=0.985,
                               bottom=0.22, top=0.80, hspace=0.46, wspace=0.20)
        axes = []
        for panel, n in enumerate(SAMPLE_SIZES):
            ax = fig.add_subplot(grid[0, panel], sharey=axes[0] if axes else None)
            axes.append(ax)
            selected = [row for row in rows if row["n"] == n]
            ax.axhline(100*targets[n], color="#555555", linestyle=":", linewidth=1.5, zorder=1)
            ax.axhline(95, color="#111111", linestyle="--", linewidth=1.3, zorder=1)
            for method_index, method in enumerate(METHODS):
                method_rows = [row for row in selected if row["method"] == method]
                for condition, row in enumerate(method_rows):
                    x = condition + OFFSETS[method_index]
                    if row["wilson"] is None:
                        ax.text(x, 0.02, "undefined", transform=ax.get_xaxis_transform(),
                                rotation=90, ha="center", va="bottom", fontsize=8, color="#111111")
                        continue
                    coverage = 100*row["coverage"]
                    low, high = (100*value for value in row["wilson"])
                    ax.errorbar(x, coverage, yerr=[[coverage-low], [high-coverage]],
                                fmt=MARKERS[method_index], color=COLORS[method_index], markersize=5.5,
                                capsize=3, elinewidth=1.2, markeredgewidth=1.0, zorder=3)
            ax.set_title(f"{'AB'[panel]}. n = {n} observations per dataset", loc="left", fontsize=11)
            ax.set_xticks(range(6), [f"{beta.title()}\nρ = {rho:g}" for rho in (0, 0.75) for beta in BETAS],
                          fontsize=8.5)
            ax.set(xlim=(-0.5, 5.5), ylim=limits)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
            if panel == 0:
                ax.set_ylabel("Coverage among successful fits (%)")
            else:
                ax.tick_params(labelleft=False)
            table_ax = fig.add_subplot(grid[1, panel])
            table_ax.axis("off")
            table_ax.set_title("Successful / failed fits (out of 2,000)", fontsize=9, pad=4)
            table = table_ax.table(
                cellText=[[f"{row['successful']} / {row['failed']}" for row in selected if row["method"] == method]
                          for method in METHODS], rowLabels=("Python", "R", "QR t"),
                cellLoc="center", rowLoc="right", bbox=(0.02, 0, 0.98, 0.85))
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for cell in table.get_celld().values():
                cell.set_edgecolor("#CCCCCC")
                cell.set_linewidth(0.4)
        method_handles = [Line2D([], [], marker=marker, color=color, linestyle="none", markersize=6, label=label)
                          for marker, color, label in zip(MARKERS, COLORS, METHOD_LABELS)]
        target_handles = [Line2D([], [], color="#555555", linestyle=":", linewidth=1.5,
                                label=f"Normal interval target: {100*targets[100]:.3f}% (n=100), {100*targets[500]:.3f}% (n=500)"),
                          Line2D([], [], color="#111111", linestyle="--", linewidth=1.3, label="t interval target: 95%")]
        fig.suptitle("Gaussian known-score study: confidence-interval coverage", fontsize=14, y=0.98)
        fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
        fig.legend(handles=target_handles, loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False, fontsize=9)
        caption = ("Points show saved coverage; bars are 95% Wilson intervals for uncertainty in observed coverage.\n"
                   "There are 2,000 independent datasets per condition; all three methods use the same datasets.\n"
                   "Coverage is conditional on successful fits. Counts below each panel preserve failures; undefined means no successful fits.\n"
                   "Analytic targets assume a correct normal, constant-variance model and full-rank design. ρ is residual correlation before scoring.\n"
                   "The shared axis includes every interval and target. coverage.csv contains all values and denominators.\n"
                   f"Execution status: {status.replace('_', ' ')}. Statistical interpretation is given in the accompanying results report.")
        fig.text(0.075, 0.045, caption, ha="left", va="bottom", fontsize=8.5, linespacing=1.45)
        try:
            for extension in ("png", "pdf"):
                with (output / f"coverage.{extension}").open("xb") as stream:
                    metadata = {"Title": "Gaussian known-score coverage", "Creator": "plot_calibration_results.py"}
                    if extension == "png":
                        metadata = {"Title": metadata["Title"], "Software": metadata["Creator"]}
                    fig.savefig(stream, format=extension, dpi=DPI, bbox_inches=None,
                                facecolor="white", transparent=False, metadata=metadata)
        finally:
            plt.close(fig)
    return limits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="public JSON record containing the production summary")
    parser.add_argument("output", type=Path, help="new output directory with an existing parent")
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists() and not args.output.is_symlink(), "output directory already exists")
        require(args.output.parent.is_dir(), "output parent directory does not exist")
        summary, rows, targets, raw = read_summary(args.source)
        # Import authoring tools only after rejecting unsupported evidence.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from PIL import Image
        from pypdf import PdfReader

        args.output.mkdir(exist_ok=False)
        limits = draw_figure(rows, targets, summary["status"], args.output, plt, Line2D)
        write_table(rows, args.output / "coverage.csv")
        with Image.open(args.output / "coverage.png") as raster:
            png_size = list(raster.size)
            png_dpi = list(raster.info.get("dpi", ()))
            require(png_size == [round(size*DPI) for size in FIGURE_INCHES], "PNG dimensions differ from the declared size")
            require("A" not in raster.getbands() or raster.getchannel("A").getextrema() == (255, 255),
                    "PNG contains unexpected transparency")
        pdf = PdfReader(args.output / "coverage.pdf")
        require(len(pdf.pages) == 1, "PDF must have exactly one page")
        pdf_points = [float(pdf.pages[0].mediabox.width), float(pdf.pages[0].mediabox.height)]
        for actual, inches in zip(pdf_points, FIGURE_INCHES):
            close(actual, inches*72, "PDF dimension", tolerance=1e-6)
        outputs = {name: {"sha256": sha256((args.output/name).read_bytes()), "bytes": (args.output/name).stat().st_size}
                   for name in ("coverage.png", "coverage.pdf", "coverage.csv")}
        provenance = {
            "schema_version": 1,
            "source": {"sha256": sha256(raw), "bytes": len(raw),
                       "summary_sha256_canonical": sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()),
                       "study_identity_sha256": summary["identity_sha256"]},
            "script_sha256": sha256(Path(__file__).read_bytes()),
            "outputs": outputs,
            "versions": {"python": platform.python_version(), **{name: version(name) for name in ("matplotlib", "numpy", "Pillow", "pypdf")}},
            "dimensions": {"inches": list(FIGURE_INCHES), "png_pixels": png_size,
                           "png_dpi_metadata": png_dpi, "pdf_points": pdf_points},
            "derivation": {"points": "100 * summary.cells[].methods[].coverage_successful",
                           "bars": "100 * saved coverage_wilson; 95% Wilson interval checked against covered/successful counts",
                           "targets": "100 * saved analytic.reference_coverage and analytic.qr_t_coverage",
                           "ordering": "n=100,500; rho=0,0.75; beta=null,cancellation,positive; method=python,r,qr_t",
                           "coverage_axis_percent": list(limits), "scale": "shared linear axis; all intervals and targets included",
                           "new_model_fits": 0, "new_random_draws": 0},
            "denominators": {"planned_datasets": 24000, "per_condition": 2000,
                             "coverage": "successful fits only", "all_counts_and_planned_bounds": "coverage.csv",
                             "method_fit_failures": sum(row["failed"] for row in rows), "missing_method_results": 0},
            "study_status": summary["status"], "calibration_validated": False,
            "limits": ["Input checks cover plot fields, not independent verification of original case records.",
                       "Analytic targets assume a correct normal, constant-variance model and full-rank design; failure selection may change coverage.",
                       "Wilson intervals are marginal per-method intervals, not simultaneous intervals or paired tests of method differences.",
                       "PNG dimensions, opacity and PDF page size checked; visual review and publisher requirements remain pending.",
                       "CSV blanks mean undefined coverage because no fits succeeded; they do not mean zero coverage."],
        }
        with (args.output / "provenance.json").open("x", encoding="utf-8") as stream:
            json.dump(provenance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (PlotError, OSError, UnicodeError, json.JSONDecodeError, ImportError) as exc:
        print(f"Figure preparation failed: {exc}", file=sys.stderr)
        return 2
    print("Created coverage.png, coverage.pdf, coverage.csv and provenance.json. Visual review remains required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
