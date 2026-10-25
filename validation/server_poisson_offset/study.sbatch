#!/bin/bash -l
#SBATCH --job-name=qgcomp-poi-off-study
#SBATCH --account=stat
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/study-%j.out
#SBATCH --error=logs/study-%j.err
#SBATCH --export=NONE
set -euo pipefail
umask 077
ROOT="$SLURM_SUBMIT_DIR"
test "$(basename "$ROOT")" = qgcomp-poisson-offset-known-scores-v1
test "${SLURM_CPUS_PER_TASK:-0}" = 1
test ! -e "$ROOT/study-001"
mkdir "$ROOT/study-001"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/study-001/exit-code.txt"' EXIT
module purge
module load Python/3.11.5-GCCcore-13.2.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export PYTHONPATH="$ROOT/code/src"
PYTHON="${QGCOMP_PYTHON:-$ROOT/python/bin/python}"
test -x "$PYTHON"
cd "$ROOT"
test "$(sha256sum source.bundle | cut -d ' ' -f 1)" = "$(cat CODE_BUNDLE_SHA256.txt)"
test "$(git -C code rev-parse HEAD)" = "$(cat SOURCE_REVISION.txt)"
test -z "$(git -C code status --porcelain)"
"$PYTHON" -c 'import quantilemix,numpy,scipy; print(quantilemix.__file__,numpy.__version__,scipy.__version__)' > study-001/import-check.txt
/modules/local/bin/showquota > study-001/quota-start.txt
"$PYTHON" - "$ROOT/study-001/quota-start.txt" <<'PY'
import sys
from pathlib import Path
line = next((s for s in Path(sys.argv[1]).read_text().splitlines() if s.startswith(str(Path.home()) + ' ')), None)
if line is None: raise SystemExit('home quota unavailable')
p = line.split()
if len(p) < 4 or not p[1].endswith('G') or not p[2].endswith('G'):
    raise SystemExit('home quota format unexpected')
if float(p[2][:-1])-float(p[1][:-1]) < 5:
    raise SystemExit('home quota headroom below 5 GiB')
PY
date -u +%FT%TZ > study-001/started-utc.txt
hostname > study-001/node.txt
module -t list > study-001/modules.txt 2>&1
lscpu > study-001/cpu.txt
printf "%s\n" "cpus=${SLURM_CPUS_PER_TASK:-}" "memory_mb=${SLURM_MEM_PER_NODE:-}" "job_id=${SLURM_JOB_ID:-}" > study-001/allocation.txt
printf '%s\n' "git_commit=$(cat SOURCE_REVISION.txt)" "bundle_sha256=$(cat CODE_BUNDLE_SHA256.txt)" "job_id=$SLURM_JOB_ID" > study-001/preflight.txt
cd code
"$PYTHON" validation/poisson_offset_calibration.py --mode study --output "$ROOT/study-001/result.json" > "$ROOT/study-001/runner-stdout.txt"
"$PYTHON" validation/check_poisson_offset_calibration.py "$ROOT/study-001/result.json" --output "$ROOT/study-001/check.json" > "$ROOT/study-001/check-stdout.json"
date -u +%FT%TZ > "$ROOT/study-001/ended-utc.txt"
cd "$ROOT/study-001"
sha256sum result.json check.json runner-stdout.txt check-stdout.json preflight.txt import-check.txt cpu.txt allocation.txt quota-start.txt > output-sha256.txt
test "$(du -sk . | cut -f1)" -lt 51200
du -sk . > output-kib.txt
