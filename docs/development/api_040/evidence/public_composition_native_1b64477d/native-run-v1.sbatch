#!/usr/bin/env bash
#SBATCH --job-name=pops-composition15-native
#SBATCH --account=r250127
#SBATCH --qos=normal
#SBATCH --partition=short
#SBATCH --constraint=x64cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -eo pipefail
export API040_ROOT=/project/r250127/api040-composition15-mpich-20261002-v15
export POPS_ENV_NAME=pops_api040_composition15_mpich_v15
export POPS_CONDA_EXE=/scratch_p/rmdraux/hoffart-pops-20260914/miniforge/bin/conda
export CONDA_REGISTER_ENVS=false CONDA_ENVS_PATH="$API040_ROOT/envs" CONDA_PKGS_DIRS="$API040_ROOT/conda-pkgs" CONDARC="$API040_ROOT/condarc"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 POPS_REQUIRE_NATIVE_TESTS=1 POPS_NATIVE_DIM=2
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE _CE_CONDA _CE_M
run="$API040_ROOT/native-$NATIVE_RUN_LABEL-job-$SLURM_JOB_ID"
mkdir "$run"
exec > "$run/run.log" 2>&1
old_prefix=/project/r250127/api040-5582d98a-mpich-20261002-v3/envs/pops_api040_5582d98a_mpich_v3
"$old_prefix/bin/python" "$API040_ROOT/preparation/preserve_existing.py" "$run/preserved-before.json"
finish() {
  result=$?
  set +e
  "$old_prefix/bin/python" "$API040_ROOT/preparation/preserve_existing.py" "$run/preserved-after.json" "$run/preserved-before.json"
  preservation=$?
  printf '{"original_exit":%s,"preservation_exit":%s,"job_id":"%s","source_freeze":"%s","world_request":%s,"label":"%s"}\n' "$result" "$preservation" "$SLURM_JOB_ID" "$NATIVE_SOURCE_FREEZE" "$NATIVE_WORLD_SIZE" "$NATIVE_RUN_LABEL" > "$run/result.json"
  if [[ "$result" == 0 && "$preservation" != 0 ]]; then result=$preservation; fi
  exit "$result"
}
trap finish EXIT
hostname
date -u
uname -a
findmnt -T /tmp -n -o FSTYPE,SOURCE,TARGET
df -h /tmp /project
tmp="/tmp/pops-api040-composition15-$NATIVE_RUN_LABEL-$SLURM_JOB_ID"
mkdir "$tmp"
[[ "$(findmnt -T "$tmp" -n -o FSTYPE)" == xfs ]]
export TMPDIR="$tmp/tmp" POPS_CACHE_DIR="$run/cache/pops" POPS_CODEGEN_DIR="$run/cache/codegen" XDG_CACHE_HOME="$run/cache/xdg" CCACHE_DIR="$run/cache/ccache" PIP_CACHE_DIR="$API040_ROOT/cache/pip"
mkdir -p "$TMPDIR" "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR" "$XDG_CACHE_HOME" "$CCACHE_DIR"
printf '%s\n' "$tmp" > "$run/compute-output-path.txt"
source "$(dirname "$POPS_CONDA_EXE")/../etc/profile.d/conda.sh"
conda_base=$(readlink -f "$(dirname "$POPS_CONDA_EXE")/..")
clean_base_path() {
  local filtered='' entry resolved
  local -a entries
  IFS=: read -ra entries <<< "$PATH"
  for entry in "${entries[@]}"; do
    resolved=$(readlink -f "$entry" 2>/dev/null || true)
    [[ "$resolved" == "$conda_base/bin" || "$resolved" == "$conda_base/condabin" ]] && continue
    filtered="${filtered:+$filtered:}$entry"
  done
  export PATH="$filtered"
}
clean_base_path
if command -v mamba; then echo 'Unexpected mamba exposure' >&2; exit 1; fi
conda activate "$API040_ROOT/envs/$POPS_ENV_NAME"
clean_base_path
if command -v mamba; then echo 'Unexpected mamba exposure after activation' >&2; exit 1; fi
[[ "$(readlink -f "$CONDA_PREFIX")" == "$(readlink -f "$API040_ROOT/envs/$POPS_ENV_NAME")" ]]
export CC="$CONDA_PREFIX/bin/gcc" CXX="$CONDA_PREFIX/bin/g++"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POPS_THREADS=1 OMP_PROC_BIND=false CMAKE_BUILD_PARALLEL_LEVEL=4 POPS_HEAVY_MODULE_TU_POOL=2
cd "$API040_ROOT/source"
[[ "$(git rev-parse HEAD)" == "$NATIVE_SOURCE_FREEZE" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" source "$run/source-before.json" "$PWD"
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" installed "$run/installed-before.json" "$PWD"
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --identity-only --output "$run/before"
set +e
if [[ "$NATIVE_WORLD_SIZE" == 1 ]]; then
  env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --output "$tmp/output" --test "$NATIVE_TEST_NODE" > "$run/test-wrapper.log" 2>&1
  tests_exit=$?
elif [[ "$NATIVE_WORLD_SIZE" == 2 ]]; then
  env -u PYTHONPATH python docs/development/api_040/run_installed_mpi_checks.py --output "$tmp/output" --ranks 2 --dimension 2 --threads 1 --timeout 6000 --test "$NATIVE_TEST_NODE" > "$run/test-wrapper.log" 2>&1
  tests_exit=$?
else
  echo 'Unsupported orchestration request; use another explicit SLURM plan' >&2
  tests_exit=2
fi
printf '{"pytest_wrapper_exit":%s}\n' "$tests_exit" > "$run/test-exit.json"
set -e
# Retain even failed tests and all raw captures before the node allocation ends.
if [[ -d "$tmp/output" ]]; then cp -a "$tmp/output" "$run/output"; else printf '%s\n' 'pytest output directory was never created' > "$run/output-missing.txt"; fi
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --identity-only --output "$run/after"
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" source "$run/source-after.json" "$PWD" "$run/source-before.json"
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" installed "$run/installed-after.json" "$PWD" "$run/installed-before.json"
exit "$tests_exit"
