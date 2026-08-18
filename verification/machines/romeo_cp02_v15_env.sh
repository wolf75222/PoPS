# Isolated ROMEO env for /scratch_p/rmdraux/pops-cp02-v15.
# Must not use $HOME/kokkos-x64. Must not write caches into $HOME.
# Must not touch /scratch_p/rmdraux/pops-verify or /scratch_p/rmdraux/pops-676.
# shellcheck disable=SC1091

POPS_CP02_ROOT="${POPS_CP02_ROOT:-/scratch_p/rmdraux/pops-cp02-v15}"
POPS_CP02_SRC="${POPS_CP02_SRC:-$POPS_CP02_ROOT/src}"
export POPS_CP02_ROOT POPS_CP02_SRC

set +eu
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set +u
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

unset KOKKOS_CPU_ROOT
export POPS_KOKKOS_ROOT_OPENMP=/scratch_p/rmdraux/kokkos-x64-pic
export POPS_KOKKOS_ROOT_SERIAL=/scratch_p/rmdraux/kokkos-x64-pic-serial
export POPS_KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-$POPS_KOKKOS_ROOT_SERIAL}"
export CMAKE_PREFIX_PATH="${POPS_KOKKOS_ROOT}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export POPS_INCLUDE="$POPS_CP02_SRC/include"

export POPS_CACHE_DIR="$POPS_CP02_ROOT/cache"
export POPS_CODEGEN_DIR="$POPS_CP02_ROOT/cache/codegen"
export XDG_CACHE_HOME="$POPS_CP02_ROOT/xdg-cache"
export CCACHE_DIR="$POPS_CP02_ROOT/ccache"
export TMPDIR="$POPS_CP02_ROOT/tmp"
export PIP_CACHE_DIR="$POPS_CP02_ROOT/pip-cache"
export POPS_PYDEPS="$POPS_CP02_ROOT/pydeps"

mkdir -p "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR" "$XDG_CACHE_HOME" \
  "$CCACHE_DIR" "$TMPDIR" "$PIP_CACHE_DIR" "$POPS_PYDEPS" \
  "$POPS_CP02_ROOT/logs" "$POPS_CP02_ROOT/evidence"

POPS_CP02_MPICH_HASH=tkezxx4y24g7otbf2mtev5zjq7gbhzkr
POPS_CP02_HDF5_HASH=jz6kqcbb76yxszibnrakn2uyzphffpqy

pops_cp02_load_mpi_hdf5() {
  spack load "/${POPS_CP02_MPICH_HASH}"
  spack load "/${POPS_CP02_HDF5_HASH}"
  export POPS_MPI_LIBRARY=MPICH
  export MPI_HOME
  MPI_HOME="$(spack location -i "/${POPS_CP02_MPICH_HASH}")"
  export PATH="${MPI_HOME}/bin:${PATH}"
  export CC="${MPI_HOME}/bin/mpicc"
  export CXX="${MPI_HOME}/bin/mpicxx"
  export HYDRA_LAUNCHER="${HYDRA_LAUNCHER:-fork}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
}

pops_cp02_pythonpath() {
  local build="$1"
  export PYTHONPATH="$POPS_PYDEPS:${build}/python:${POPS_CP02_SRC}/python:${POPS_CP02_SRC}${PYTHONPATH:+:$PYTHONPATH}"
}

pops_cp02_ensure_pydeps() {
  local python="${POPS_PYTHON:?POPS_PYTHON must be set}"
  if PYTHONPATH="$POPS_PYDEPS${PYTHONPATH:+:$PYTHONPATH}" "$python" -c "import jsonschema, matplotlib"; then
    return 0
  fi
  "$python" -m pip install --disable-pip-version-check --no-warn-script-location \
    --target "$POPS_PYDEPS" --cache-dir "$PIP_CACHE_DIR" \
    jsonschema "matplotlib<4" pillow
  # Do not shadow Spack NumPy 1.26, which the compiled leaf is linked against.
  rm -rf "$POPS_PYDEPS/numpy" "$POPS_PYDEPS"/numpy-*.dist-info
}

pops_cp02_select_kokkos() {
  local space="${1:-${POPS_CP02_SPACE:-KokkosSerial}}"
  if [ "$space" = "KokkosOpenMP" ]; then
    export POPS_KOKKOS_ROOT="$POPS_KOKKOS_ROOT_OPENMP"
  else
    export POPS_KOKKOS_ROOT="$POPS_KOKKOS_ROOT_SERIAL"
  fi
  export CMAKE_PREFIX_PATH="${POPS_KOKKOS_ROOT}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
}
