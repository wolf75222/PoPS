#!/bin/bash
#SBATCH --job-name=pops-sdk26-m18-dim2-build
#SBATCH --account=r250127
#SBATCH --qos=normal
#SBATCH --partition=short
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
# PREPARATION ONLY: ROOT review and submit; no automatic deployment or test job.
set -eo pipefail
DIM=2
BASE=/project/r250127/api040-composition15-mpich-20261002-v15
SOURCE="$BASE/source"
OLD_ENV="$BASE/envs/pops_api040_composition15_mpich_v15"
QUAL="$BASE/qualification-sdk26-m18-dim2-20261003"
export POPS_ENV_NAME="pops_api040_sdk26_m18_dim2"
NEW_ENV="$QUAL/envs/$POPS_ENV_NAME"
RUN="$QUAL/build-job-$SLURM_JOB_ID"
CONDA_TOOL=/scratch_p/rmdraux/hoffart-pops-20260914/miniforge/bin/conda
[[ ! -e "$NEW_ENV" && ! -e "$RUN" ]]
mkdir -p "$RUN" "$QUAL/envs"
exec >"$RUN/build.log" 2>&1
export CONDA_REGISTER_ENVS=false CONDA_ENVS_PATH="$QUAL/envs" CONDA_PKGS_DIRS="$BASE/conda-pkgs" CONDARC="$BASE/condarc"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 POPS_REQUIRE_NATIVE_TESTS=1 POPS_NATIVE_DIM="$DIM"
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE _CE_CONDA _CE_M
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$QUAL/cache/xdg" PIP_CACHE_DIR="$QUAL/cache/pip" CCACHE_DIR="$QUAL/cache/ccache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CCACHE_DIR"
cd "$SOURCE"
[[ "$(git rev-parse HEAD)" == 51b0eec9dcfd660098cc5531eed6c400185c8dad ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OLD3=/project/r250127/api040-5582d98a-mpich-20261002-v3/envs/pops_api040_5582d98a_mpich_v3
"$OLD3/bin/python" "$BASE/preparation/preserve_existing.py" "$RUN/preserved-before.json"
"$OLD_ENV/bin/python" "$BASE/preparation/snapshot_current.py" installed "$RUN/sdk25-original-before.json" "$SOURCE"
finish() {
 status=$?; set +e
 "$OLD3/bin/python" "$BASE/preparation/preserve_existing.py" "$RUN/preserved-after.json" "$RUN/preserved-before.json"
 old3_preserved=$?
 "$OLD_ENV/bin/python" "$BASE/preparation/snapshot_current.py" installed "$RUN/sdk25-original-after.json" "$SOURCE"
 cmp "$RUN/sdk25-original-before.json" "$RUN/sdk25-original-after.json"
 preserved=$?
 printf '{"original_exit":%s,"original_sdk25_preserved_exit":%s,"dimension":%s}
' "$status" "$preserved" "$DIM" >"$RUN/result.json"
 if [[ $status == 0 && $preserved != 0 ]]; then status=$preserved; fi
 if [[ $status == 0 && $old3_preserved != 0 ]]; then status=$old3_preserved; fi
 exit "$status"
}
trap finish EXIT
source "$(dirname "$CONDA_TOOL")/../etc/profile.d/conda.sh"
conda_base=$(readlink -f "$(dirname "$CONDA_TOOL")/..")
clean_base_path() {
 local filtered='' entry resolved
 local -a entries
 IFS=: read -ra entries <<<"$PATH"
 for entry in "${entries[@]}"; do
  resolved=$(readlink -f "$entry" 2>/dev/null || true)
  [[ "$resolved" == "$conda_base/bin" || "$resolved" == "$conda_base/condabin" ]] && continue
  filtered="${filtered:+$filtered:}$entry"
 done
 export PATH="$filtered"
}
clean_base_path
if command -v mamba; then echo 'Unexpected mamba exposure' >&2; exit 1; fi
# Clone actual frozen SDK25. --copy avoids shared writable inodes; offline refuses dependency drift.
"$CONDA_TOOL" create --yes --prefix "$NEW_ENV" --clone "$OLD_ENV" --copy --offline
conda activate "$NEW_ENV"
clean_base_path
if command -v mamba; then echo 'Unexpected mamba exposure after activation' >&2; exit 1; fi
[[ "$(readlink -f "$CONDA_PREFIX")" == "$(readlink -f "$NEW_ENV")" ]]
export CC="$NEW_ENV/bin/gcc" CXX="$NEW_ENV/bin/g++" MPICH_CC="$NEW_ENV/bin/gcc" MPICH_CXX="$NEW_ENV/bin/g++"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=8 POPS_HEAVY_MODULE_TU_POOL=4
# Existing immutable source worktree was already set up: do not repeat setup_env or solve dependencies.
env -u PYTHONPATH python "$BASE/preparation/snapshot_current.py" source "$RUN/source-before.json" "$SOURCE"
env -u PYTHONPATH python "$BASE/preparation/snapshot_current.py" installed "$RUN/clone-before.json" "$SOURCE"
conda list --explicit >"$RUN/environment-explicit.txt"
env -u PYTHONPATH python -m pip list --format=json >"$RUN/pip-list.json"
bash scripts/build_python.sh --dim "$DIM" --mpi --wheel-dir "$RUN/wheels"
clean_base_path
if command -v mamba; then echo "Unexpected mamba exposure after build" >&2; exit 1; fi
env -u PYTHONPATH python "$BASE/preparation/snapshot_current.py" source "$RUN/source-after.json" "$SOURCE" "$RUN/source-before.json"
env -u PYTHONPATH python "$BASE/preparation/snapshot_current.py" installed "$RUN/clone-after.json" "$SOURCE"
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --identity-only --output "$RUN/identity"
find "build" -path "*-dim$DIM/*" -name CMakeCache.txt -exec cp --parents '{}' "$RUN/" \;
find "build" -path "*-dim$DIM/*" -name compile_commands.json -exec cp --parents '{}' "$RUN/" \;
sha256sum "$RUN"/wheels/*.whl >"$RUN/wheel-sha256.txt"
printf '%s\n' BUILD_INSTALL_IDENTITY_ONLY_NO_SCIENCE >"$RUN/scope.txt"
