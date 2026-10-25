#!/bin/bash -l
#SBATCH --job-name=qgcomp-s5-study
#SBATCH --account=stat
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/study-%j.out
#SBATCH --error=logs/study-%j.err
#SBATCH --export=NONE
set -euo pipefail
umask 077
ROOT="$SLURM_SUBMIT_DIR"
test "$(basename "$ROOT")" = qgcomp-paper-scenario5-v1
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
"/modules/local/bin/showquota" > study-001/quota-start.txt
"$PYTHON" - "$ROOT/study-001/quota-start.txt" <<'PY'
import re, sys
from pathlib import Path
line = next((s for s in Path(sys.argv[1]).read_text().splitlines() if s.startswith(str(Path('/scratch/user') / Path.home().name) + ' ')), None)
if line is None: raise SystemExit('scratch quota unavailable')
p = line.split()
def gib(s):
 m = re.fullmatch(r'([0-9.]+)([KMGTP])',s)
 if not m: raise SystemExit('unrecognized quota unit')
 return float(m.group(1))*1024**('KMGTP'.index(m.group(2))-2)
if gib(p[2])-gib(p[1]) < 10: raise SystemExit('scratch quota headroom below 10 GiB')
PY
date -u +%FT%TZ > study-001/started-utc.txt
hostname > study-001/node.txt
module -t list > study-001/modules.txt 2>&1
printf '%s\n' "git_commit=$(cat SOURCE_REVISION.txt)" "bundle_sha256=$(cat CODE_BUNDLE_SHA256.txt)" "job_id=$SLURM_JOB_ID" "command=validation/paper_scenario5.py study --output study-001/result.json" > study-001/preflight.txt
cd code
"$PYTHON" validation/paper_scenario5.py study --output "$ROOT/study-001/result.json" > "$ROOT/study-001/runner-stdout.json"
"$PYTHON" validation/check_paper_scenario5.py "$ROOT/study-001/result.json" > "$ROOT/study-001/check.json"
date -u +%FT%TZ > "$ROOT/study-001/ended-utc.txt"
cd "$ROOT/study-001"
sha256sum result.json result.json.jsonl check.json runner-stdout.json preflight.txt import-check.txt quota-start.txt > output-sha256.txt
du -sk . > output-kib.txt
