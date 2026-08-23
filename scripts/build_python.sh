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
#   bash scripts/build_python.sh --dim 2            # build the exact Dim=2 specialization
#   POPS_NATIVE_DIM=1 bash scripts/build_python.sh  # the equivalent environment route
#   bash scripts/build_python.sh --dim 2 --clean    # drop the scikit-build wheel cache first
#   bash scripts/build_python.sh --dim 2 --fresh    # --clean + ccache -C: a true COLD compile
#   bash scripts/build_python.sh --dim 3 --mpi      # MPI + native parallel-HDF5, exactly Dim=3
#   bash scripts/build_python.sh --dim 2 --float32  # compile-time pops::Real = float (not the default ABI)
#   bash scripts/build_python.sh --dim 2 --wheel-dir /tmp/wheels
#                                           # build, retain, then install that exact wheel
#   POPS_HEAVY_MODULE_TU_POOL=4 bash scripts/build_python.sh --dim 2  # pin the pool
#   bash scripts/build_python.sh --dim 2 -- -e  # pass extra args to pip (editable install)
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
WITH_FLOAT32=0
WHEEL_DIR=""
CALLER_NATIVE_DIM="${POPS_NATIVE_DIM-}"
CLI_NATIVE_DIM=""
EXTRA_PIP=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean) DO_CLEAN=1 ;;
    --fresh) DO_CLEAN=1; DO_FRESH=1 ;;
    --mpi)   WITH_MPI=1 ;;
    --float32) WITH_FLOAT32=1 ;;
    --dim)
      shift
      [[ $# -gt 0 ]] || { echo "--dim requires 1, 2, or 3" >&2; exit 2; }
      CLI_NATIVE_DIM="$1"
      ;;
    --wheel-dir)
      shift
      [[ $# -gt 0 ]] || { echo "--wheel-dir requires a directory" >&2; exit 2; }
      WHEEL_DIR="$1"
      ;;
    -h|--help)
      sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    --) shift; EXTRA_PIP=("$@"); break ;;
    *) echo "unknown argument: $1 (use --dim N | --clean | --fresh | --mpi | --float32 | --wheel-dir DIR | --help, or -- <pip args>)" >&2
       exit 2 ;;
  esac
  shift
done

# A PoPS native module is one immutable compile-time spatial specialization.  Do not silently
# manufacture a Dim=2 artifact: every build must name Dim=1, 2, or 3 at its outermost entry point.
if [[ -n "$CLI_NATIVE_DIM" && -n "$CALLER_NATIVE_DIM" \
      && "$CLI_NATIVE_DIM" != "$CALLER_NATIVE_DIM" ]]; then
  echo "conflicting native dimensions: --dim=$CLI_NATIVE_DIM but POPS_NATIVE_DIM=$CALLER_NATIVE_DIM" >&2
  exit 2
fi
NATIVE_DIM="${CLI_NATIVE_DIM:-$CALLER_NATIVE_DIM}"
case "$NATIVE_DIM" in
  1|2|3) ;;
  "")
    echo "native dimension is required: pass --dim 1|2|3 or export POPS_NATIVE_DIM=1|2|3" >&2
    exit 2
    ;;
  *)
    echo "invalid native dimension '$NATIVE_DIM': expected exactly 1, 2, or 3" >&2
    exit 2
    ;;
esac

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
export POPS_NATIVE_DIM="$NATIVE_DIM"
echo "--- env '$ENV_NAME' active (CONDA_PREFIX=$CONDA_PREFIX) ---"
echo "native spatial specialization: Dim=$POPS_NATIVE_DIM"

# Conda's macOS OpenMPI wrappers remember the compiler used to build the package, which is not part
# of this deliberately AppleClang-based environment.  OpenMPI's documented override keeps mpicc,
# mpicxx and therefore h5pcc executable with the exact compiler used by PoPS.  Respect explicit
# caller choices and leave Linux/toolchain-module wrappers untouched.
if [[ "$(uname)" == "Darwin" ]]; then
  export OMPI_CC="${OMPI_CC:-${CC:-/usr/bin/clang}}"
  export OMPI_CXX="${OMPI_CXX:-${CXX:-/usr/bin/clang++}}"
fi

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
  # Each specialization owns build/<wheel_tag>-dimN. Remove only the requested Dim=N cache, never
  # another specialization nor the C++ preset build/ root (its CMakeCache.txt sits at build/).
  shopt -s nullglob
  removed=0
  for d in "$HERE"/build/cp3*-dim"$POPS_NATIVE_DIM"/; do rm -rf "$d"; removed=1; done
  [[ $removed -eq 1 ]] && echo "--clean: removed Dim=$POPS_NATIVE_DIM wheel cache" \
                       || echo "--clean: no Dim=$POPS_NATIVE_DIM wheel cache to remove"
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
  # Conda's macOS MPICH wrappers retain the compiler triplet from their build host.  Route them to
  # the already-selected AppleClang explicitly so h5pcc's real compile probe is authoritative.
  if [[ "$(uname)" == Darwin && -x "$CONDA_PREFIX/bin/mpichversion" ]]; then
    export MPICH_CC="${MPICH_CC:-${CC:-/usr/bin/clang}}"
    export MPICH_CXX="${MPICH_CXX:-${CXX:-/usr/bin/clang++}}"
  fi
  # Keep FindHDF5 inside the active environment.  Honour an explicit caller override byte-for-byte;
  # otherwise a host hdf5-config.cmake (for example Homebrew) could preempt this env's h5pcc.
  if [[ -z "${HDF5_ROOT+x}" ]]; then
    export HDF5_ROOT="$CONDA_PREFIX"
  fi
  echo "MPI backend: ON; native parallel HDF5: ON"
  echo "HDF5 root: ${HDF5_ROOT:-<explicit empty override>}"
else
  # A build script invocation is a complete backend request, not an overlay on the caller's shell.
  # Clear stale feature exports left by a preceding distributed build so the ordinary path cannot
  # accidentally rebuild/install an MPI artifact and only discover the mismatch after installation.
  export POPS_USE_MPI=OFF
  export POPS_USE_HDF5=OFF
  echo "MPI backend: OFF; native parallel HDF5: OFF"
fi
cd "$HERE"
if [[ -n "$WHEEL_DIR" && "$WHEEL_DIR" != /* ]]; then
  WHEEL_DIR="$HERE/$WHEEL_DIR"
fi
cmake_settings=(
  -C build-dir="build/{wheel_tag}-dim${POPS_NATIVE_DIM}"
  -C cmake.define.POPS_NATIVE_DIM="$POPS_NATIVE_DIM"
  -C cmake.define.POPS_HEAVY_MODULE_TU_POOL="$pool"
)
if [[ $WITH_MPI -eq 1 ]]; then
  # Environment seeding applies only to a fresh CMake cache.  These explicit settings switch an
  # existing scikit-build cache to the requested MPI + parallel-HDF5 contract.
  cmake_settings+=(
    -C cmake.define.POPS_USE_MPI=ON
    -C cmake.define.POPS_USE_HDF5=ON
  )
else
  # The ordinary invocation is an exact serial request even after a previous ``--mpi`` build reused
  # this persistent wheel-tag cache.
  cmake_settings+=(
    -C cmake.define.POPS_USE_MPI=OFF
    -C cmake.define.POPS_USE_HDF5=OFF
  )
fi
if [[ $WITH_FLOAT32 -eq 1 ]]; then
  cmake_settings+=(
    -C cmake.define.POPS_REAL_TYPE=float
  )
  echo "Real specialization: float (binary32); Python ABI remains unclosed"
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
NATIVE_VARIANT_SNAPSHOT="$(mktemp -d "${TMPDIR:-/tmp}/pops-native-variants.XXXXXX")"
cleanup_native_snapshot() { rm -rf "$NATIVE_VARIANT_SNAPSHOT"; }
trap cleanup_native_snapshot EXIT
PYTHONPATH='' PYTHONNOUSERSITE=1 \
  python "$HERE/scripts/preserve_native_variants.py" snapshot --dest "$NATIVE_VARIANT_SNAPSHOT"
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
  # Prove the immutable installed payload before Darwin codesign is allowed to repair bytes and
  # refresh their installed-manifest hashes. This build produces exactly the requested one-row set.
  PYTHONPATH='' PYTHONNOUSERSITE=1 \
    python "$HERE/scripts/prove_installed_wheel.py" \
      --wheel "${built_wheels[0]}" --expect-dim "$POPS_NATIVE_DIM"
fi

# ADC-647: pip may leave authenticated sibling files behind even though the new wheel manifest
# declares only Dim=N. Isolate those exact snapshot-authenticated leftovers first; the codesign
# locator remains strict about every unmanifested extension. Scikit-build may also rewrite the new
# extension after the linker signed it, so refresh that row's hash before strict sibling restore.
PYTHONPATH='' PYTHONNOUSERSITE=1 \
  python "$HERE/scripts/preserve_native_variants.py" isolate --src "$NATIVE_VARIANT_SNAPSHOT" \
    --expect-dim "$POPS_NATIVE_DIM"
PYTHONPATH='' PYTHONNOUSERSITE=1 \
  python "$HERE/scripts/codesign_pops_extensions.py" --expect-dim "$POPS_NATIVE_DIM"
PYTHONPATH='' PYTHONNOUSERSITE=1 \
  python "$HERE/scripts/preserve_native_variants.py" restore --src "$NATIVE_VARIANT_SNAPSHOT" \
    --expect-dim "$POPS_NATIVE_DIM"

# --- diagnose ---------------------------------------------------------------------------------------
native_verify_args=()
if [[ $WITH_MPI -eq 1 ]]; then
  native_verify_args=(--expect-dim "$POPS_NATIVE_DIM" --expect-mpi --expect-parallel-hdf5)
else
  native_verify_args=(--expect-dim "$POPS_NATIVE_DIM" --expect-serial)
fi
PYTHONPATH='' PYTHONNOUSERSITE=1 \
  python "$HERE/scripts/verify_installed_native.py" "${native_verify_args[@]}"
echo ""
echo "--- pops.runtime.doctor.doctor() ---"
PYTHONPATH='' PYTHONNOUSERSITE=1 \
  python -c "import pops; from pops._native_selector import select_native_dimension; select_native_dimension($POPS_NATIVE_DIM); from pops.runtime.doctor import doctor; print('pops', pops.__version__, 'Dim=$POPS_NATIVE_DIM'); doctor()"
