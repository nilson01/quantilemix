#!/bin/bash -l
#SBATCH --job-name=qgcomp-cox-pilot
#SBATCH --account=stat
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:45:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/cox-pilot-%A_%a.out
#SBATCH --error=logs/cox-pilot-%A_%a.err
#SBATCH --export=NONE
set -euo pipefail
umask 077
ROOT="$SLURM_SUBMIT_DIR"
case "$(basename "$ROOT")" in
    qgcomp-cox-bootstrap-pilot-[0-9][0-9][0-9]) ;;
    *) exit 2 ;;
esac
test "${SLURM_CPUS_PER_TASK:-0}" = 1
test "${SLURM_ARRAY_TASK_ID:-x}" -ge 0
test "${SLURM_ARRAY_TASK_ID:-x}" -le 19
module purge
module load Python/3.11.5-GCCcore-13.2.0
command -v scancel >/dev/null
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
PYTHON="$ROOT/python/bin/python"
test -x "$PYTHON"
cd "$ROOT"
test "$(sha256sum source.bundle | cut -d ' ' -f 1)" = "$(cat CODE_BUNDLE_SHA256.txt)"
test "$(sha256sum quantilemix-0.1.0.dev0-py3-none-any.whl | cut -d ' ' -f 1)" = "$(cat WHEEL_SHA256.txt)"
test "$(git -C code rev-parse HEAD)" = "$(cat SOURCE_REVISION.txt)"
test -z "$(git -C code status --porcelain)"
CELL=$((SLURM_ARRAY_TASK_ID / 5))
REPETITION=$((SLURM_ARRAY_TASK_ID % 5))
test ! -e "$ROOT/pilot-case-${SLURM_ARRAY_TASK_ID}"
mkdir "$ROOT/pilot-case-${SLURM_ARRAY_TASK_ID}"
OUT="$ROOT/pilot-case-${SLURM_ARRAY_TASK_ID}"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/exit-code.txt"; if [ "$rc" -ne 0 ]; then printf "%s\n" "task failed; cancelling pilot array" > "$OUT/stop-reason.txt"; scancel "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" || true; fi' EXIT
date -u +%FT%TZ > "$OUT/started-utc.txt"
date -u +%s > "$OUT/started-epoch.txt"
hostname > "$OUT/node.txt"
module -t list > "$OUT/modules.txt" 2>&1
printf '%s\n' "commit=$(cat SOURCE_REVISION.txt)" "bundle=$(cat CODE_BUNDLE_SHA256.txt)" "wheel=$(cat WHEEL_SHA256.txt)" "job=$SLURM_JOB_ID" > "$OUT/preflight.txt"
"$PYTHON" -c 'import quantilemix,numpy,scipy; print(quantilemix.__file__,numpy.__version__,scipy.__version__)' > "$OUT/import-check.txt"
case "$(head -n 1 "$OUT/import-check.txt")" in "$ROOT/python/"*) ;; *) exit 2 ;; esac
cd "$ROOT/code"
"$PYTHON" validation/cox_bootstrap_calibration.py --mode pilot --cell "$CELL" --start "$REPETITION" --count 1 --output "$OUT/result.json" > "$OUT/runner-stdout.txt" 2> "$OUT/runner-stderr.txt"
"$PYTHON" validation/check_cox_bootstrap_calibration.py --input "$OUT/result.json" --output "$OUT/check.json" > "$OUT/check-stdout.txt"
date -u +%FT%TZ > "$OUT/ended-utc.txt"
date -u +%s > "$OUT/ended-epoch.txt"
cd "$OUT"
sha256sum result.json check.json runner-stdout.txt runner-stderr.txt check-stdout.txt preflight.txt import-check.txt > output-sha256.txt
test "$(du -sk . | cut -f1)" -lt 2048
du -sk . > output-kib.txt
"$PYTHON" - "$OUT/result.json" <<'PY'
import json
import sys

rows = json.loads(open(sys.argv[1], encoding="utf-8").read())["records"]
if len(rows) != 1 or rows[0]["fit"]["status"] != "ok" or rows[0]["fit"]["warnings"]:
    raise SystemExit("pilot case failed or warned; stop and inspect")
PY
