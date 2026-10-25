# SPDX-License-Identifier: GPL-2.0-or-later
"""Audit all terminal Cox pilot tasks; run with the same isolated wheel environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_cox_bootstrap_calibration as checker  # noqa: E402


WHEEL = "quantilemix-0.1.0.dev0-py3-none-any.whl"
TASKS = 20
BYTES_LIMIT = 50 * 1024 * 1024
TASK_BYTES_LIMIT = 2 * 1024 * 1024
MEMORY_BYTES_LIMIT = 2 * 1024 * 1024 * 1024
WALL_SECONDS_LIMIT = 45 * 60


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _rss(value: str) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", value)
    if not match:
        raise ValueError(f"unrecognized MaxRSS: {value}")
    return round(float(match.group(1)) * 1024**(" KMGTP".index(match.group(2))))


def audit(root: Path, job: str) -> dict:
    issues = []
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("job must be a Slurm numeric job ID")
    for filename, digest in (("source.bundle", "CODE_BUNDLE_SHA256.txt"),
                             (WHEEL, "WHEEL_SHA256.txt")):
        if not (root / filename).is_file() or not (root / digest).is_file():
            issues.append(f"missing {filename} or {digest}")
        elif _hash(root / filename) != (root / digest).read_text().strip():
            issues.append(f"{filename} digest mismatch")
    revision = (root / "SOURCE_REVISION.txt").read_text().strip()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root / "code",
                          capture_output=True, text=True, check=True).stdout.strip()
    if revision != head:
        issues.append("checkout revision differs from staged revision")
    if subprocess.run(["git", "status", "--porcelain"], cwd=root / "code",
                      capture_output=True, text=True, check=True).stdout.strip():
        issues.append("checkout is dirty")

    inputs = []
    for index in range(TASKS):
        directory = root / f"pilot-case-{index}"
        if not directory.is_dir():
            issues.append(f"task {index}: output directory missing")
            continue
        completed = directory / "result.json"
        partial = directory / "result.json.partial.jsonl"
        if completed.is_file():
            inputs.append(completed)
        elif partial.is_file():
            inputs.append(partial)
            issues.append(f"task {index}: partial journal")
        else:
            issues.append(f"task {index}: no result or journal")
        exit_file = directory / "exit-code.txt"
        if not exit_file.is_file() or exit_file.read_text().strip() != "0":
            issues.append(f"task {index}: nonzero wrapper exit")
        if _size(directory) >= TASK_BYTES_LIMIT:
            issues.append(f"task {index}: output exceeds 2 MiB")
        manifest = directory / "output-sha256.txt"
        if not manifest.is_file():
            issues.append(f"task {index}: output digest manifest missing")
            continue
        for line in manifest.read_text().splitlines():
            digest, filename = line.split(maxsplit=1)
            filename = filename.removeprefix("*")
            if Path(filename).name != filename or not (directory / filename).is_file():
                issues.append(f"task {index}: invalid manifest entry {filename}")
            elif _hash(directory / filename) != digest:
                issues.append(f"task {index}: digest mismatch for {filename}")

    command = ["sacct", "-j", job, "--format=JobID,State,ExitCode,ElapsedRaw,MaxRSS",
               "--parsable2", "--noheader"]
    account = subprocess.run(command, capture_output=True, text=True, check=True)
    (root / f"pilot-sacct-{job}.txt").write_text(account.stdout)
    rows = [line.split("|") for line in account.stdout.splitlines() if line.strip()]
    for index in range(TASKS):
        prefix = f"{job}_{index}"
        task = [row for row in rows if row[0] == prefix]
        if len(task) != 1 or task[0][1] != "COMPLETED" or task[0][2] != "0:0":
            issues.append(f"task {index}: Slurm state or exit not complete/zero")
        if task and (not task[0][3].isdigit() or int(task[0][3]) > WALL_SECONDS_LIMIT):
            issues.append(f"task {index}: elapsed time missing or above 45 minutes")
        rss = [_rss(row[4]) for row in rows if row[0].startswith(prefix)]
        measured = [value for value in rss if value is not None]
        if not measured or max(measured) > MEMORY_BYTES_LIMIT:
            issues.append(f"task {index}: MaxRSS missing or above 2 GiB")

    output_bytes = sum(_size(root / f"pilot-case-{i}") for i in range(TASKS)
                       if (root / f"pilot-case-{i}").is_dir())
    output_bytes += _size(root / "logs") if (root / "logs").is_dir() else 0
    output_bytes += (root / f"pilot-sacct-{job}.txt").stat().st_size
    if output_bytes >= BYTES_LIMIT:
        issues.append("aggregate task outputs and logs exceed 50 MiB")
    try:
        result = checker.audit_files(inputs, complete=True)
    except (ValueError, KeyError, TypeError) as exc:
        issues.append(f"independent case audit failed: {exc}")
        result = None
    if result is not None:
        if (result["status"] != "passed" or result["audited_records"] != TASKS
                or any(part["successful"] != 5 or part["failed"] or part["warnings"]
                       for part in result["summaries"].values())):
            issues.append("case audit has missing, failed, interrupted or warning records")
    return {"accepted": not issues, "job_id": job, "source_revision": revision,
            "output_bytes": output_bytes, "issues": issues, "case_audit": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new path")
    result = audit(args.root.resolve(), args.job)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"pilot accepted: {result['accepted']}; issues: {len(result['issues'])}")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
