#!/usr/bin/env bash
#SBATCH --job-name=pops-composition20-build
#SBATCH --account=r250127
#SBATCH --qos=normal
#SBATCH --partition=short
#SBATCH --constraint=x64cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
set -eo pipefail
export API040_ROOT=/project/r250127/api040-composition15-mpich-20261002-v15
export POPS_ENV_NAME=pops_api040_composition15_mpich_v15
export POPS_CONDA_EXE=/scratch_p/rmdraux/hoffart-pops-20260914/miniforge/bin/conda
export CONDA_REGISTER_ENVS=false CONDA_ENVS_PATH="$API040_ROOT/envs" CONDA_PKGS_DIRS="$API040_ROOT/conda-pkgs" CONDARC="$API040_ROOT/condarc"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 POPS_REQUIRE_NATIVE_TESTS=1
unset PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE _CE_CONDA _CE_M
run="$API040_ROOT/build-job-$SLURM_JOB_ID"
mkdir "$run"
exec > "$run/build.log" 2>&1
old_prefix=/project/r250127/api040-5582d98a-mpich-20261002-v3/envs/pops_api040_5582d98a_mpich_v3
"$old_prefix/bin/python" "$API040_ROOT/preparation/preserve_existing.py" "$run/preserved-before.json"
finish() {
  result=$?
  set +e
  "$old_prefix/bin/python" "$API040_ROOT/preparation/preserve_existing.py" "$run/preserved-after.json" "$run/preserved-before.json"
  preservation=$?
  printf '{"original_exit":%s,"preservation_exit":%s,"job_id":"%s","source_freeze":"%s"}\n' "$result" "$preservation" "$SLURM_JOB_ID" "$BUILD_FREEZE" > "$run/result.json"
  if [[ "$result" == 0 && "$preservation" != 0 ]]; then result=$preservation; fi
  exit "$result"
}
trap finish EXIT
hostname
date -u
uname -a
findmnt -T /tmp -n -o FSTYPE,SOURCE,TARGET
df -h /tmp /project
export TMPDIR="$run/tmp" XDG_CACHE_HOME="$API040_ROOT/cache/xdg" PIP_CACHE_DIR="$API040_ROOT/cache/pip" CCACHE_DIR="$API040_ROOT/cache/ccache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$CCACHE_DIR"
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=8 POPS_HEAVY_MODULE_TU_POOL=4
cd "$API040_ROOT/source"
[[ "$(git rev-parse HEAD)" == "$BUILD_FREEZE" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" source "$run/source-before.json" "$PWD"
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" installed "$run/installed-before.json" "$PWD"
env -u PYTHONPATH python - "$run" <<'PYARCHIVE'
import hashlib,json,sys,sysconfig,tarfile
from pathlib import Path
run=Path(sys.argv[1]);site=Path(sysconfig.get_path('purelib'))
before=json.loads((run/'installed-before.json').read_text())
archive=run/'sdk19-installed-preserved.tar'
roots=sorted({Path(name).parts[0] for name in before['files']})
def keep(info):
    return None if '__pycache__' in Path(info.name).parts else info
with tarfile.open(archive,'x',dereference=False) as target:
    for root in roots:target.add(site/root,arcname=root,filter=keep)
with tarfile.open(archive,'r') as source:
    verified=0
    for name,pin in before['files'].items():
        item=source.extractfile(name)
        if item is None:raise RuntimeError('Missing preserved package member: '+name)
        assert hashlib.sha256(item.read()).hexdigest()==pin,name
        verified+=1
receipt={'schema':'root.sdk19-installed-preservation@1','archive':str(archive),
         'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'verified_files':verified,
         'original_inventory_sha256':hashlib.sha256((run/'installed-before.json').read_bytes()).hexdigest(),
         'historical_wheel':'build-job-732041/wheels/pops-1.1.0-cp312-cp312-linux_x86_64.whl'}
with (run/'sdk19-installed-preservation.json').open('x') as f:json.dump(receipt,f,indent=2);f.write(chr(10))
print(json.dumps(receipt))
PYARCHIVE
# Install precisely the test extra declared by this checkout, without rebuilding PoPS here.
env -u PYTHONPATH python -c 'import sys,tomllib; from pathlib import Path; data=tomllib.loads(Path("pyproject.toml").read_text()); Path(sys.argv[1]).write_text("\n".join(data["project"]["optional-dependencies"]["test"])+"\n")' "$run/test-requirements.txt"
env -u PYTHONPATH python -m pip install -r "$run/test-requirements.txt"
env -u PYTHONPATH python -m pip list --format=json > "$run/pip-list.json"
conda list --explicit > "$run/environment-explicit.txt"
"$CC" --version > "$run/compiler-version.txt"
"$CONDA_PREFIX/bin/mpichversion" > "$run/mpich-version.txt"
bash scripts/build_python.sh --dim 2 --mpi --wheel-dir "$run/wheels"
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" source "$run/source-after.json" "$PWD" "$run/source-before.json"
env -u PYTHONPATH python "$API040_ROOT/preparation/snapshot_current.py" installed "$run/installed-after.json" "$PWD"
env -u PYTHONPATH python docs/development/api_040/run_installed_checks.py --identity-only --output "$run/identity"
mkdir "$run/cmake"
find build -name CMakeCache.txt -exec cp --parents '{}' "$run/cmake/" \;
find build -name compile_commands.json -exec cp --parents '{}' "$run/cmake/" \;
sha256sum "$run"/wheels/*.whl > "$run/wheel-sha256.txt"
printf '%s\n' 'BUILD_INSTALL_IDENTITY_ONLY_NO_SCIENTIFIC_ACCEPTANCE' > "$run/build-only.txt"
