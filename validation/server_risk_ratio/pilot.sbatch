#!/bin/bash -l
#SBATCH --job-name=qgcomp-rr-pilot
#SBATCH --account=stat
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/pilot-%j.out
#SBATCH --error=logs/pilot-%j.err
#SBATCH --export=NONE
set -euo pipefail
umask 077
ROOT="$SLURM_SUBMIT_DIR"
test "$(basename "$ROOT")" = qgcomp-binary-risk-ratio-known-scores-v1
test "${SLURM_CPUS_PER_TASK:-0}" = 1
test ! -e "$ROOT/pilot-001"
mkdir "$ROOT/pilot-001"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/pilot-001/exit-code.txt"' EXIT
module purge
module load Python/3.11.5-GCCcore-13.2.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
export PYTHONPATH="$ROOT/code/src"
PYTHON="$ROOT/python/bin/python"
test -x "$PYTHON"
cd "$ROOT"
test "$(sha256sum source.bundle | cut -d ' ' -f 1)" = "$(cat CODE_BUNDLE_SHA256.txt)"
test "$(git -C code rev-parse HEAD)" = "$(cat SOURCE_REVISION.txt)"
test -z "$(git -C code status --porcelain)"
"$PYTHON" -c 'import quantilemix,numpy,scipy; print(quantilemix.__file__,numpy.__version__,scipy.__version__)' > pilot-001/import-check.txt
date -u +%FT%TZ > pilot-001/started-utc.txt
hostname > pilot-001/node.txt
module -t list > pilot-001/modules.txt 2>&1
printf '%s\n' "git_commit=$(cat SOURCE_REVISION.txt)" "bundle_sha256=$(cat CODE_BUNDLE_SHA256.txt)" "job_id=$SLURM_JOB_ID" > pilot-001/preflight.txt
cd code
"$PYTHON" validation/risk_ratio_calibration.py --mode pilot --output "$ROOT/pilot-001/result.json" > "$ROOT/pilot-001/runner-stdout.txt"
"$PYTHON" validation/check_risk_ratio_calibration.py "$ROOT/pilot-001/result.json" --output "$ROOT/pilot-001/check.json" > "$ROOT/pilot-001/check-stdout.json"
date -u +%FT%TZ > "$ROOT/pilot-001/ended-utc.txt"
cd "$ROOT/pilot-001"
sha256sum result.json result.json.journal.jsonl result.json.complete.json check.json runner-stdout.txt check-stdout.json preflight.txt import-check.txt > output-sha256.txt
test "$(du -sk . | cut -f1)" -lt 51200
du -sk . > output-kib.txt
