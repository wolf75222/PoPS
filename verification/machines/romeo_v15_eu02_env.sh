# Isolated ROMEO env for /scratch_p/rmdraux/pops-v15-eu02.
# Must not use $HOME/kokkos-x64. Must not write caches into $HOME.
# shellcheck disable=SC1091

POPS_EU02_ROOT="${POPS_EU02_ROOT:-/scratch_p/rmdraux/pops-v15-eu02}"
POPS_EU02_SRC="${POPS_EU02_SRC:-$POPS_EU02_ROOT/src}"
export POPS_EU02_ROOT POPS_EU02_SRC

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
export POPS_INCLUDE="$POPS_EU02_SRC/include"

export POPS_CACHE_DIR="$POPS_EU02_ROOT/cache"
export POPS_CODEGEN_DIR="$POPS_EU02_ROOT/cache/codegen"
export XDG_CACHE_HOME="$POPS_EU02_ROOT/xdg-cache"
export CCACHE_DIR="$POPS_EU02_ROOT/ccache"
export TMPDIR="$POPS_EU02_ROOT/tmp"
export PIP_CACHE_DIR="$POPS_EU02_ROOT/pip-cache"
export POPS_PYDEPS="$POPS_EU02_ROOT/pydeps"

mkdir -p "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR" "$XDG_CACHE_HOME" \
  "$CCACHE_DIR" "$TMPDIR" "$PIP_CACHE_DIR" "$POPS_PYDEPS" \
  "$POPS_EU02_ROOT/logs" "$POPS_EU02_ROOT/evidence"

POPS_EU02_MPICH_HASH=tkezxx4y24g7otbf2mtev5zjq7gbhzkr
POPS_EU02_HDF5_HASH=jz6kqcbb76yxszibnrakn2uyzphffpqy

pops_eu02_load_mpi_hdf5() {
  spack load "/${POPS_EU02_MPICH_HASH}"
  spack load "/${POPS_EU02_HDF5_HASH}"
  export POPS_MPI_LIBRARY=MPICH
  export MPI_HOME
  MPI_HOME="$(spack location -i "/${POPS_EU02_MPICH_HASH}")"
  export HDF5_ROOT="${HDF5_ROOT:-$(spack location -i "/${POPS_EU02_HDF5_HASH}" 2>/dev/null || true)}"
  if [ ! -f "${MPI_HOME}/include/mpi.h" ]; then
    echo "ERROR: mpi.h missing under ${MPI_HOME}/include" >&2
    exit 1
  fi
  export PATH="${MPI_HOME}/bin:${PATH}"
  export CC="${MPI_HOME}/bin/mpicc"
  export CXX="${MPI_HOME}/bin/mpicxx"
  export HYDRA_LAUNCHER="${HYDRA_LAUNCHER:-fork}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
}

pops_eu02_pythonpath() {
  local build="$1"
  export PYTHONPATH="$POPS_PYDEPS:${build}/python:${POPS_EU02_SRC}/python:${POPS_EU02_SRC}${PYTHONPATH:+:$PYTHONPATH}"
}

pops_eu02_select_kokkos() {
  local space="${1:-${POPS_EU02_SPACE:-KokkosSerial}}"
  if [ "$space" = "KokkosOpenMP" ]; then
    export POPS_KOKKOS_ROOT="$POPS_KOKKOS_ROOT_OPENMP"
  else
    export POPS_KOKKOS_ROOT="$POPS_KOKKOS_ROOT_SERIAL"
  fi
  export CMAKE_PREFIX_PATH="${POPS_KOKKOS_ROOT}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
}

pops_eu02_require_pic_kokkos() {
  local lib="$POPS_KOKKOS_ROOT/lib64/libkokkoscore.a"
  if [ ! -f "$lib" ]; then
    echo "ERROR: PIC Kokkos missing: $lib" >&2
    exit 1
  fi
}
