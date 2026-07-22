#!/usr/bin/env bash
# One command to build + install the Python module `_pops` for END USERS, applying the build knobs that
# scripts/setup_env.sh only *recommends*. Run `bash scripts/setup_env.sh` ONCE first (it creates the
# `pops` env and pins the per-platform toolchain); then this script, on every (re)build:
#
#   - activates the conda env `pops` (override: POPS_ENV_NAME), without tripping `set -u`;
#   - sizes the production-module heavy-TU Ninja pool (POPS_HEAVY_MODULE_TU_POOL) from cores AND free RAM so the split module TUs
#     compile in PARALLEL without OOM (each -O3 leaf peaks at several GB; the CMake default remains the
#     memory-constrained size-1 guard). Pre-set POPS_HEAVY_MODULE_TU_POOL to pin it by hand.
#   - exports the Kokkos / CMake discovery vars (Kokkos_ROOT, POPS_KOKKOS_ROOT, CMAKE_PREFIX_PATH) and a
#     STABLE, cross-worktree ccache (CCACHE_DIR + CCACHE_BASEDIR -> a file already compiled in another
#     worktree is reused instead of recompiled);
#   - runs `pip install . --no-build-isolation` so the build reuses the env's pinned
#     scikit-build-core / pybind11 (the SAME stack as the toolchain) instead of a fresh pip build env;
#   - ends on the runtime-layer environment doctor.
#
#   bash scripts/build_python.sh            # build + install into the active env
#   bash scripts/build_python.sh --clean    # drop the scikit-build wheel cache first
#   bash scripts/build_python.sh --fresh    # --clean AND `ccache -C`: a true COLD compile (measuring)
#   bash scripts/build_python.sh --mpi      # distributed MPI + native parallel-HDF5 backend
#   bash scripts/build_python.sh --wheel-dir /tmp/wheels
#                                           # build, retain, then install that exact wheel
#   POPS_HEAVY_MODULE_TU_POOL=4 bash scripts/build_python.sh    # pin the pool by hand (skip auto-sizing)
#   bash scripts/build_python.sh -- -e      # pass extra args through to pip (here: editable install)
#
# NOT `set -u`: `conda activate` references unset variables in its own shell hook.
set -eo pipefail

ENV_NAME="${POPS_ENV_NAME:-pops}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/scripts/conda_runtime.sh"

# --- arguments --------------------------------------------------------------------------------------
DO_CLEAN=0
DO_FRESH=0
WITH_MPI=0
WHEEL_DIR=""
EXTRA_PIP=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) DO_CLEAN=1 ;;
    --fresh) DO_CLEAN=1; DO_FRESH=1 ;;
    --mpi)   WITH_MPI=1 ;;
    --wheel-dir)
      shift
      [[ $# -gt 0 ]] || { echo "--wheel-dir requires a directory" >&2; exit 2; }
      WHEEL_DIR="$1"
      ;;
    -h|--help)
      sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    --) shift; EXTRA_PIP=("$@"); break ;;
    *) echo "unknown argument: $1 (use --clean | --fresh | --mpi | --wheel-dir DIR | --help, or -- <pip args>)" >&2
       exit 2 ;;
  esac
  shift
done

# --- conda present + env active ----------------------------------------------------------------------
if ! pops_load_conda; then
  echo "conda not found. Run 'bash scripts/setup_env.sh' first (it bootstraps the env and toolchain)." >&2
  exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' is missing. Create it first: bash scripts/setup_env.sh" >&2
  exit 1
fi
conda activate "$ENV_NAME"
echo "--- env '$ENV_NAME' active (CONDA_PREFIX=$CONDA_PREFIX) ---"

# --- heavy-TU pool: cores capped by RAM (each -O3 leaf peaks ~3-4 GB) --------------------------------
ncpu="$( (nproc 2>/dev/null) || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
if [[ "$(uname)" == "Darwin" ]]; then
  mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
else
  mem_kb="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
  mem_bytes=$(( mem_kb * 1024 ))
fi
mem_gb=$(( mem_bytes / 1024 / 1024 / 1024 ))
if [[ -n "${POPS_HEAVY_MODULE_TU_POOL:-}" ]]; then
  pool="$POPS_HEAVY_MODULE_TU_POOL"
  echo "production module heavy-TU pool: $pool (from POPS_HEAVY_MODULE_TU_POOL)"
else
  ram_cap=$(( mem_gb / 4 )); [[ $ram_cap -lt 1 ]] && ram_cap=1
  pool=$ncpu; [[ $pool -gt $ram_cap ]] && pool=$ram_cap
  echo "production module heavy-TU pool: $pool (min of ${ncpu} cores and ${ram_cap} = ${mem_gb}GB/4; export POPS_HEAVY_MODULE_TU_POOL to override)"
fi

# --- discovery vars + stable cross-worktree ccache --------------------------------------------------
export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export Kokkos_ROOT="${Kokkos_ROOT:-$CONDA_PREFIX}"
export POPS_KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-$CONDA_PREFIX}"
# The build and its doctor must always validate this checkout, not whichever worktree last persisted
# POPS_INCLUDE in the shared conda environment.
export POPS_INCLUDE="$HERE/include"
# A stable cache directory is shared by every checkout. Each worktree uses its own root as base_dir:
# ccache then rewrites its absolute source/build paths to the same relative paths in every worktree.
# Using the main checkout as base_dir does not cover linked worktrees created as siblings.
export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/adc-ccache}"
export CCACHE_BASEDIR="${CCACHE_BASEDIR:-$HERE}"
echo "ccache: dir=$CCACHE_DIR basedir=$CCACHE_BASEDIR"

# --- clean / fresh ----------------------------------------------------------------------------------
if [[ $DO_CLEAN -eq 1 ]]; then
  # scikit-build-core caches under build/<wheel_tag>/ (e.g. build/cp312-cp312-macosx_14_0_arm64). Remove
  # ONLY those tag dirs, never the C++ preset build/ root (its CMakeCache.txt sits at build/).
  shopt -s nullglob
  removed=0
  for d in "$HERE"/build/cp3*/; do rm -rf "$d"; removed=1; done
  [[ $removed -eq 1 ]] && echo "--clean: removed scikit-build wheel cache (build/cp3*/)" \
                       || echo "--clean: no scikit-build wheel cache to remove"
fi
if [[ $DO_FRESH -eq 1 ]]; then
  if command -v ccache >/dev/null 2>&1; then
    ccache -C >/dev/null && echo "--fresh: ccache cleared (cold build)" \
                         || echo "--fresh: ccache -C failed; build may not be fully cold" >&2
  fi
fi

# --- build + install --------------------------------------------------------------------------------
if [[ $WITH_MPI -eq 1 ]]; then
  # The final distributed runtime contract includes its native collective writer.  A serial HDF5
  # discovery is rejected by CMake; there is no reduced-capability `--mpi` artifact.
  export POPS_USE_MPI=ON
  export POPS_USE_HDF5=ON
  echo "MPI backend: ON; native parallel HDF5: ON"
fi
cd "$HERE"
if [[ -n "$WHEEL_DIR" && "$WHEEL_DIR" != /* ]]; then
  WHEEL_DIR="$HERE/$WHEEL_DIR"
fi
cmake_settings=(-C cmake.define.POPS_HEAVY_MODULE_TU_POOL="$pool")
if [[ $WITH_MPI -eq 1 ]]; then
  # Environment seeding applies only to a fresh CMake cache.  These explicit settings also switch
  # an existing serial scikit-build cache to the requested MPI + parallel-HDF5 contract.
  cmake_settings+=(
    -C cmake.define.POPS_USE_MPI=ON
    -C cmake.define.POPS_USE_HDF5=ON
  )
fi
if [[ -n "$WHEEL_DIR" ]]; then
  mkdir -p "$WHEEL_DIR"
  shopt -s nullglob
  existing_wheels=("$WHEEL_DIR"/*.whl)
  if [[ ${#existing_wheels[@]} -ne 0 ]]; then
    echo "--wheel-dir must be empty; refusing stale release artifacts in $WHEEL_DIR" >&2
    exit 2
  fi
  pip_args=(wheel -v . --no-deps --wheel-dir "$WHEEL_DIR" "${cmake_settings[@]}")
else
  pip_args=(install -v . "${cmake_settings[@]}")
fi
if python -c "import scikit_build_core, pybind11" >/dev/null 2>&1; then
  if [[ -n "$WHEEL_DIR" ]]; then
    pip_args=(wheel -v . --no-deps --no-build-isolation --wheel-dir "$WHEEL_DIR" \
      "${cmake_settings[@]}")
  else
    pip_args=(install -v . --no-build-isolation "${cmake_settings[@]}")
  fi
else
  echo "note: scikit-build-core/pybind11 not in '$ENV_NAME'; using pip build isolation"
  echo "      (slower, unpinned build deps). Add 'scikit-build-core' to environment.yml + 'conda env update' to fix."
fi
echo "--- python -m pip ${pip_args[*]} ${EXTRA_PIP[*]} ---"
python -m pip "${pip_args[@]}" "${EXTRA_PIP[@]}"
if [[ -n "$WHEEL_DIR" ]]; then
  built_wheels=("$WHEEL_DIR"/pops-*.whl)
  if [[ ${#built_wheels[@]} -ne 1 ]]; then
    echo "release build must produce exactly one pops wheel in $WHEEL_DIR" >&2
    exit 1
  fi
  echo "--- installing exact retained wheel ${built_wheels[0]} ---"
  python -m pip install --force-reinstall --no-deps "${built_wheels[0]}"
  echo "release wheel: ${built_wheels[0]}"
fi

# --- diagnose ---------------------------------------------------------------------------------------
# ADC-647: pip/scikit-build may rewrite the copied extension after the linker signed its build-tree
# output. Resolve the exact installed module without importing pops, ad-hoc sign it on Darwin, and
# verify both the signature and its ad-hoc identity. Any failure stops before import/doctor.
PYTHONPATH= PYTHONNOUSERSITE=1 python "$HERE/scripts/codesign_pops_extensions.py"
echo ""
echo "--- pops.runtime.doctor.doctor() ---"
PYTHONPATH= PYTHONNOUSERSITE=1 \
  python -c "import pops; from pops.runtime.doctor import doctor; print('pops', pops.__version__); doctor()"
