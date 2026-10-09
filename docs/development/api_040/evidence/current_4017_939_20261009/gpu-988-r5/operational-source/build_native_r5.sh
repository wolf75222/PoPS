#!/bin/bash
# Unexecuted template: official environment preparation must already have completed exactly once.
set -euo pipefail
umask 077
: "${FUTURE_GPU_ROOT:?Root supplies final admitted private namespace}"
: "${SLURM_JOB_ID:?Root admitted allocation required}"
ROOT="$FUTURE_GPU_ROOT"
finish_r5_receipts() {
 local body_status=$?
 set +e
 /usr/bin/python3 "$ROOT/registry_capture.py" "$ROOT" after-r5-native-finally
 local registry_status=$?
 /usr/bin/python3 "$ROOT/namespace_footprint.py" "$ROOT" r5-native-finally
 local footprint_status=$?
 /usr/bin/python3 "$ROOT/quota_guard_resume.py" "$ROOT" after-r5-native-finally 0
 local quota_status=$?
 /usr/bin/python3 - "$ROOT" "$body_status" "$registry_status" "$footprint_status" "$quota_status" <<'PY_STATUS'
import json,sys
from pathlib import Path
with(Path(sys.argv[1])/'results/r5-native-finally-statuses.json').open('x')as f:
 json.dump(dict(zip(('body','registry','footprint','quota'),map(int,sys.argv[2:]))),f,indent=2,sort_keys=True);f.write(chr(10))
PY_STATUS
 local receipt_status=$?
 if ((body_status != 0)); then exit "$body_status"; fi
 if ((registry_status != 0)); then exit "$registry_status"; fi
 if ((footprint_status != 0)); then exit "$footprint_status"; fi
 if ((quota_status != 0)); then exit "$quota_status"; fi
 exit "$receipt_status"
}
trap finish_r5_receipts EXIT

source "$FUTURE_GPU_ROOT/runtime_context_r5.sh"
/usr/bin/python3 "$ROOT/freeze_admission.py"
[[ -f "$ROOT/results/registration-before.json" ]]


[[ $(id -u) == 100267 && "$ROOT" == /gpfs/scratch/rmdraux/PoPS-final-full-native-cuda-dim2-* ]]
export CONDA_REGISTER_ENVS=false CONDA_ENVS_PATH="$ROOT/envs" CONDA_PKGS_DIRS="$ROOT/pkgs" CONDARC="$ROOT/condarc"
source "$ROOT/miniforge/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/pops_final_cuda_dim2"
[[ "$CONDA_PREFIX" == "$ROOT/envs/pops_final_cuda_dim2" ]]
unset PYTHONPATH PYTHONOPTIMIZE
[[ -f "$ROOT/results/official-setup-entered.json" && -x "$ROOT/strict_nvcc_wrapper.sh" ]]
export CXX="$ROOT/strict_nvcc_wrapper.sh" MPICH_CXX="$ROOT/strict_nvcc_wrapper.sh"
export POPS_NATIVE_DIM=2 CMAKE_EXPORT_COMPILE_COMMANDS=ON
# Root must authenticate strict wrapper SM90/O3/FP controls and read-only HOPPER90 Kokkos before entry.
[[ "$Kokkos_ROOT" == $ROOT/kokkos-unified-install ]]
/usr/bin/python3 "$ROOT/phase_quota_r5.py" "$ROOT" before-native
cd "$ROOT/source"
[[ ! -e "$ROOT/results/official-build-entered.json" && ! -e "$ROOT/results/official-build-entered-r4.json" && ! -e "$ROOT/results/official-build-entered-r5.json" ]]
printf '{"job":"%s","dimension":2,"scope":"official build entered; outcome pending"}\n' "$SLURM_JOB_ID" > "$ROOT/results/official-build-entered-r5.json"
env -u PYTHONPATH -u PYTHONOPTIMIZE bash scripts/build_python.sh --dim 2 --mpi --wheel-dir "$ROOT/wheels" -- \
  -C cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -C cmake.define.CMAKE_BUILD_TYPE=Release \
  -C "cmake.define.CMAKE_CXX_FLAGS=--fmad=false -fopenmp -fno-fast-math -ffp-contract=off" \
  -C "cmake.define.Kokkos_ROOT=$Kokkos_ROOT" \
  -C "cmake.define.Kokkos_DIR=$Kokkos_ROOT/lib/cmake/Kokkos" \
  -C cmake.define.CUDAToolkit_ROOT=/apps/2025/manual_install/cuda_eviden/12.6 \
  > "$ROOT/results/official-build-r5.log" 2>&1
# This successful command alone is not full native/scientific acceptance.
# Root must pin retained wheel/installed DSO, every final compile graph TU/object,
# actual runtime backend manifest, source/header origins and Native SDK; then populate native_freeze.

env -u PYTHONPATH -u PYTHONOPTIMIZE "$CONDA_PREFIX/bin/python" "$ROOT/capture_gpu_native_r5.py"
/usr/bin/python3 "$ROOT/namespace_footprint.py" "$ROOT" r5-native
