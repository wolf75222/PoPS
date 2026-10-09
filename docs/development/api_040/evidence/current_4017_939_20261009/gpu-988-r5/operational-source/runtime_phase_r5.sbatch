#!/bin/bash
#SBATCH --account=r250127
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --constraint=armgpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
# Scheduler actual short24h limit; no extra per-command wall cap.
set -euo pipefail
umask 077
: "${FUTURE_GPU_ROOT:?Root supplies admitted exact private namespace}"
[[ "$SLURM_JOB_ACCOUNT" == r250127 ]]
cd "$FUTURE_GPU_ROOT"
source "$FUTURE_GPU_ROOT/private_runtime_context.sh"
# Root submits this only with afterok:BUILD_JOB; raw predecessor receipt is checked again here.
/usr/bin/python3 - "$FUTURE_GPU_ROOT" <<'PY_ADMIT'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);plan=json.loads((root/'plan.json').read_text());receipt=root/'results/gpu-native-admission-r5.json'
assert hashlib.sha256(receipt.read_bytes()).hexdigest()==plan['gpu_native_admission_sha256']
assert json.loads(receipt.read_text())['status']=='GPU_NATIVE_PHYSICAL_ADMISSION_PASS'
PY_ADMIT
bash "$FUTURE_GPU_ROOT/run_campaign_r5.sh"
