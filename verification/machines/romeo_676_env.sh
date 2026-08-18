# Isolated ROMEO env for /scratch_p/rmdraux/pops-676.
# Must not use $HOME/kokkos-x64. Must not write caches into $HOME.
# shellcheck disable=SC1091

POPS676_ROOT="${POPS676_ROOT:-/scratch_p/rmdraux/pops-676}"
POPS676_SRC="${POPS676_SRC:-$POPS676_ROOT/src}"
export POPS676_ROOT POPS676_SRC

set +eu
# shellcheck source=/dev/null
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set +u
set -e
spack load /3s3hqzq >/dev/null 2>&1 || true

# The sourced CPU env exports KOKKOS_CPU_ROOT=$HOME/kokkos-x64. Override it.
unset KOKKOS_CPU_ROOT
export POPS_KOKKOS_ROOT=/scratch_p/rmdraux/kokkos-x64-pic
export CMAKE_PREFIX_PATH="${POPS_KOKKOS_ROOT}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export POPS_INCLUDE="$POPS676_SRC/include"

export POPS_CACHE_DIR="$POPS676_ROOT/cache"
export POPS_CODEGEN_DIR="$POPS676_ROOT/cache/codegen"
export XDG_CACHE_HOME="$POPS676_ROOT/xdg-cache"
export CCACHE_DIR="$POPS676_ROOT/ccache"
export TMPDIR="$POPS676_ROOT/tmp"
export PIP_CACHE_DIR="$POPS676_ROOT/pip-cache"
export POPS_PYDEPS="$POPS676_ROOT/pydeps"

mkdir -p "$POPS_CACHE_DIR" "$POPS_CODEGEN_DIR" "$XDG_CACHE_HOME" \
  "$CCACHE_DIR" "$TMPDIR" "$PIP_CACHE_DIR" "$POPS_PYDEPS" \
  "$POPS676_ROOT/logs" "$POPS676_ROOT/evidence"

# MPICH 4.2.3 + parallel HDF5 1.10.11, both gcc-14.2.0. Used only for MPI builds/gates.
POPS676_MPICH_HASH=tkezxx4y24g7otbf2mtev5zjq7gbhzkr
POPS676_HDF5_HASH=jz6kqcbb76yxszibnrakn2uyzphffpqy

pops676_load_mpi_hdf5() {
  spack load "/${POPS676_MPICH_HASH}"
  spack load "/${POPS676_HDF5_HASH}"
  export POPS_MPI_LIBRARY=MPICH
  export MPI_HOME
  MPI_HOME="$(spack location -i "/${POPS676_MPICH_HASH}")"
  export HDF5_ROOT="${HDF5_ROOT:-$(spack location -i "/${POPS676_HDF5_HASH}" 2>/dev/null || true)}"
  if [ ! -f "${MPI_HOME}/include/mpi.h" ]; then
    echo "ERROR: mpi.h missing under ${MPI_HOME}/include" >&2
    exit 1
  fi
  if command -v mpicxx >/dev/null 2>&1; then
    export CC=mpicc CXX=mpicxx
    export OMPI_CC="${CC}" OMPI_CXX="${CXX}"
  fi
  export HYDRA_LAUNCHER="${HYDRA_LAUNCHER:-fork}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
}

pops676_pythonpath() {
  local build="$1"
  export PYTHONPATH="$POPS_PYDEPS:${build}/python:${POPS676_SRC}/python:${POPS676_SRC}${PYTHONPATH:+:$PYTHONPATH}"
}

pops676_require_pic_kokkos() {
  local lib="$POPS_KOKKOS_ROOT/lib64/libkokkoscore.a"
  if [ ! -f "$lib" ]; then
    echo "ERROR: PIC Kokkos missing: $lib" >&2
    exit 1
  fi
  case "$POPS_KOKKOS_ROOT" in
    */kokkos-x64-pic) ;;
    *)
      echo "ERROR: refusing non-PIC Kokkos root $POPS_KOKKOS_ROOT" >&2
      exit 1
      ;;
  esac
}
