#!/usr/bin/env bash
# Run a PoPS Catalyst simulation in ParaView's Python runtime without starting pvpython/pvbatch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PARAVIEW_ROOT="${POPS_PARAVIEW_ROOT:-${PARAVIEW_ROOT:-}}"
MPI_RANKS=""

usage() {
  cat <<'EOF'
Usage: scripts/paraview_python.sh [--paraview-root PATH] [--mpi RANKS] SCRIPT [ARG ...]

Runs SCRIPT with ParaView's private Python and Catalyst modules while PoPS initializes MPI first
with MPI_THREAD_MULTIPLE. On macOS, PATH may name either ParaView.app or ParaView.app/Contents.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paraview-root)
      [[ $# -ge 2 ]] || { echo "--paraview-root requires a path" >&2; exit 2; }
      PARAVIEW_ROOT="$2"
      shift 2
      ;;
    --mpi)
      [[ $# -ge 2 ]] || { echo "--mpi requires a rank count" >&2; exit 2; }
      MPI_RANKS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
if [[ -z "$PARAVIEW_ROOT" ]]; then
  shopt -s nullglob
  PARAVIEW_APPS=(/Applications/ParaView*.app)
  shopt -u nullglob
  if [[ ${#PARAVIEW_APPS[@]} -eq 1 ]]; then
    PARAVIEW_ROOT="${PARAVIEW_APPS[0]}"
  fi
fi
[[ -n "$PARAVIEW_ROOT" ]] || {
  echo "set POPS_PARAVIEW_ROOT or pass --paraview-root" >&2
  exit 2
}
if [[ -d "$PARAVIEW_ROOT/Contents" ]]; then
  PARAVIEW_ROOT="$PARAVIEW_ROOT/Contents"
fi
PARAVIEW_ROOT="$(cd "$PARAVIEW_ROOT" && pwd)"

case "$(uname)" in
  Darwin)
    PARAVIEW_LIB="${POPS_PARAVIEW_LIBRARY_DIR:-$PARAVIEW_ROOT/Libraries}"
    PARAVIEW_PYTHONHOME_DEFAULT="$PARAVIEW_LIB"
    LIBPYTHON_SUFFIX=dylib
    ;;
  Linux)
    if [[ -d "$PARAVIEW_ROOT/lib64" ]]; then
      PARAVIEW_LIB="${POPS_PARAVIEW_LIBRARY_DIR:-$PARAVIEW_ROOT/lib64}"
    else
      PARAVIEW_LIB="${POPS_PARAVIEW_LIBRARY_DIR:-$PARAVIEW_ROOT/lib}"
    fi
    PARAVIEW_PYTHONHOME_DEFAULT="$PARAVIEW_ROOT"
    LIBPYTHON_SUFFIX=so
    ;;
  *)
    echo "unsupported platform for the ParaView Python host: $(uname)" >&2
    exit 2
    ;;
esac

[[ -d "$PARAVIEW_LIB" ]] || {
  echo "ParaView library directory is missing: $PARAVIEW_LIB" >&2
  exit 2
}

if [[ -n "${POPS_PARAVIEW_LIBPYTHON:-}" ]]; then
  LIBPYTHON="$POPS_PARAVIEW_LIBPYTHON"
else
  shopt -s nullglob
  LIBPYTHON_CANDIDATES=("$PARAVIEW_LIB"/libpython3.*."$LIBPYTHON_SUFFIX")
  shopt -u nullglob
  if [[ ${#LIBPYTHON_CANDIDATES[@]} -ne 1 ]]; then
    echo "expected exactly one ParaView libpython, found ${#LIBPYTHON_CANDIDATES[@]}" >&2
    echo "set POPS_PARAVIEW_LIBPYTHON explicitly for a non-standard installation" >&2
    exit 2
  fi
  LIBPYTHON="${LIBPYTHON_CANDIDATES[0]}"
fi
[[ -f "$LIBPYTHON" ]] || { echo "ParaView libpython is missing: $LIBPYTHON" >&2; exit 2; }

[[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] || {
  echo "activate the PoPS conda environment before running Catalyst" >&2
  exit 2
}
POPS_PYTHON="$CONDA_PREFIX/bin/python"
POPS_SITE="$(env PYTHONPATH= PYTHONNOUSERSITE=1 "$POPS_PYTHON" -c \
  'import pathlib, pops, sys; p=pathlib.Path(pops.__file__).resolve(); root=pathlib.Path(sys.prefix).resolve(); p.relative_to(root); print(p.parent.parent)')"
POPS_EXTENSION="$(env PYTHONPATH= PYTHONNOUSERSITE=1 "$POPS_PYTHON" -c \
  'from pops import _pops; print(_pops.__file__)')"
POPS_HAS_MPI="$(env PYTHONPATH= PYTHONNOUSERSITE=1 "$POPS_PYTHON" -c \
  'from pops import _pops; print(1 if _pops.__has_mpi__ else 0)')"
POPS_HAS_PARALLEL_HDF5="$(env PYTHONPATH= PYTHONNOUSERSITE=1 "$POPS_PYTHON" -c \
  'from pops import _pops; print(1 if getattr(_pops, "__has_parallel_hdf5__", False) else 0)')"
POPS_MINOR="$(env PYTHONPATH= PYTHONNOUSERSITE=1 "$POPS_PYTHON" -c \
  'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$POPS_HAS_MPI" == 1 ]] || {
  echo "the active PoPS package must be rebuilt with scripts/build_python.sh --mpi --clean" >&2
  exit 2
}
case "$(basename "$LIBPYTHON")" in
  libpython"$POPS_MINOR".*) ;;
  *)
    echo "ParaView and PoPS require the same CPython minor version (PoPS: $POPS_MINOR)" >&2
    exit 2
    ;;
esac

if [[ -n "${POPS_PARAVIEW_PYTHON_DIR:-}" ]]; then
  PARAVIEW_PYTHON="$POPS_PARAVIEW_PYTHON_DIR"
else
  PARAVIEW_PYTHON=""
  PARAVIEW_PYTHON_CANDIDATES=(
    "$PARAVIEW_ROOT/Python"
    "$PARAVIEW_LIB/python"
    "$PARAVIEW_LIB/python$POPS_MINOR/site-packages"
    "$PARAVIEW_LIB/lib/python$POPS_MINOR/site-packages"
    "$PARAVIEW_ROOT/lib/python$POPS_MINOR/site-packages"
  )
  for candidate in "${PARAVIEW_PYTHON_CANDIDATES[@]}"; do
    if [[ -d "$candidate/paraview" ]]; then
      PARAVIEW_PYTHON="$candidate"
      break
    fi
  done
fi
[[ -n "$PARAVIEW_PYTHON" && -d "$PARAVIEW_PYTHON/paraview" ]] || {
  echo "ParaView Python modules are missing; set POPS_PARAVIEW_PYTHON_DIR" >&2
  exit 2
}
PARAVIEW_PYTHON="$(cd "$PARAVIEW_PYTHON" && pwd)"

if [[ -n "${POPS_PARAVIEW_PYTHONHOME:-}" ]]; then
  PARAVIEW_PYTHONHOME="$POPS_PARAVIEW_PYTHONHOME"
else
  PARAVIEW_PYTHONHOME=""
  for candidate in "$PARAVIEW_PYTHONHOME_DEFAULT" "$PARAVIEW_ROOT" "$PARAVIEW_LIB"; do
    if [[ -d "$candidate/lib/python$POPS_MINOR" ]]; then
      PARAVIEW_PYTHONHOME="$candidate"
      break
    fi
  done
fi
[[ -n "$PARAVIEW_PYTHONHOME" && -d "$PARAVIEW_PYTHONHOME/lib/python$POPS_MINOR" ]] || {
  echo "ParaView Python home is missing its Python $POPS_MINOR standard library" >&2
  exit 2
}
PARAVIEW_PYTHONHOME="$(cd "$PARAVIEW_PYTHONHOME" && pwd)"

if [[ "$(uname)" == Darwin ]]; then
  dependency_basename() {
    otool -L "$POPS_EXTENSION" | awk -v prefix="$1" '
      $1 ~ ("/" prefix "\\.[0-9]+\\.dylib$") {
        value=$1; sub(".*/", "", value); print value; exit
      }'
  }
else
  dependency_basename() {
    ldd "$POPS_EXTENSION" | awk -v prefix="$1" '
      $1 ~ ("^" prefix "\\.so\\.[0-9]+$") { print $1; exit }'
  }
fi
POPS_MPI_BASENAME="$(dependency_basename libmpi)"
POPS_PMPI_BASENAME="$(dependency_basename libpmpi)"
POPS_MPICXX_BASENAME="$(dependency_basename libmpicxx)"
[[ -n "$POPS_MPI_BASENAME" && -n "$POPS_PMPI_BASENAME" \
  && -n "$POPS_MPICXX_BASENAME" ]] || {
  echo "cannot authenticate the active PoPS MPI shared-library contract" >&2
  exit 2
}
for mpi_basename in \
  "$POPS_MPI_BASENAME" \
  "$POPS_PMPI_BASENAME" \
  "$POPS_MPICXX_BASENAME"; do
  required_library="$PARAVIEW_LIB/$mpi_basename"
  [[ -f "$required_library" ]] || {
    echo "ParaView/PoPS shared-library ABI mismatch; missing $required_library" >&2
    exit 2
  }
done
POPS_ACTIVE_MPI_LIBRARY="$CONDA_PREFIX/lib/$POPS_MPI_BASENAME"
POPS_ACTIVE_PMPI_LIBRARY="$CONDA_PREFIX/lib/$POPS_PMPI_BASENAME"
POPS_ACTIVE_MPICXX_LIBRARY="$CONDA_PREFIX/lib/$POPS_MPICXX_BASENAME"
for required_library in \
  "$POPS_ACTIVE_MPI_LIBRARY" \
  "$POPS_ACTIVE_PMPI_LIBRARY" \
  "$POPS_ACTIVE_MPICXX_LIBRARY"; do
  [[ -f "$required_library" ]] || {
    echo "active PoPS MPI library is missing: $required_library" >&2
    exit 2
  }
done
POPS_ACTIVE_HDF5_LIBRARY=""
if [[ "$POPS_HAS_PARALLEL_HDF5" == 1 ]]; then
  POPS_HDF5_BASENAME="$(dependency_basename libhdf5)"
  [[ -n "$POPS_HDF5_BASENAME" ]] || {
    echo "cannot authenticate the active PoPS parallel-HDF5 shared library" >&2
    exit 2
  }
  POPS_ACTIVE_HDF5_LIBRARY="$CONDA_PREFIX/lib/$POPS_HDF5_BASENAME"
  [[ -f "$POPS_ACTIVE_HDF5_LIBRARY" ]] || {
    echo "active PoPS parallel-HDF5 library is missing: $POPS_ACTIVE_HDF5_LIBRARY" >&2
    exit 2
  }
fi

if [[ -n "$MPI_RANKS" ]]; then
  [[ "$MPI_RANKS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--mpi must be a positive integer" >&2
    exit 2
  }
  [[ -x "$CONDA_PREFIX/bin/mpiexec" ]] || {
    echo "active PoPS mpiexec is missing: $CONDA_PREFIX/bin/mpiexec" >&2
    exit 2
  }
fi

COMPILER="${CC:-cc}"
COMPILER_PATH="$(command -v "$COMPILER" || true)"
[[ -n "$COMPILER_PATH" ]] || { echo "C compiler not found: $COMPILER" >&2; exit 2; }
if command -v shasum >/dev/null 2>&1; then
  SHA256_COMMAND=(shasum -a 256)
else
  SHA256_COMMAND=(sha256sum)
fi
LIBPYTHON_HASH="$("${SHA256_COMMAND[@]}" "$LIBPYTHON" | awk '{print $1}')"
HOST_SOURCE_HASH="$("${SHA256_COMMAND[@]}" "$HERE/scripts/paraview_python_host.c" | awk '{print $1}')"
COMPILER_HASH="$("$COMPILER_PATH" --version | "${SHA256_COMMAND[@]}" | awk '{print $1}')"
CACHE_BASE="${POPS_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pops}"
CACHE_KEY="${LIBPYTHON_HASH:0:16}-${HOST_SOURCE_HASH:0:16}-${COMPILER_HASH:0:12}-host-v2"
CACHE_DIR="$CACHE_BASE/paraview-python/$CACHE_KEY"
LAUNCHER="$CACHE_DIR/python"
if [[ ! -x "$LAUNCHER" ]]; then
  mkdir -p "$CACHE_DIR"
  TEMP_LAUNCHER="$CACHE_DIR/python.tmp.$$"
  if [[ "$(uname)" == Darwin ]]; then
    "$COMPILER_PATH" "$HERE/scripts/paraview_python_host.c" -o "$TEMP_LAUNCHER"
    codesign --force --sign - "$TEMP_LAUNCHER" >/dev/null 2>&1
  else
    "$COMPILER_PATH" "$HERE/scripts/paraview_python_host.c" -ldl -o "$TEMP_LAUNCHER"
  fi
  mv "$TEMP_LAUNCHER" "$LAUNCHER"
fi

MPI_OVERLAY=""
if [[ "$(uname)" == Linux ]]; then
  MPI_OVERLAY="$CACHE_DIR/mpi-overlay"
  MPI_OVERLAY_ID="$({
    printf 'conda-prefix\0%s\0' "$CONDA_PREFIX"
    for active_library in \
      "$POPS_ACTIVE_MPI_LIBRARY" \
      "$POPS_ACTIVE_PMPI_LIBRARY" \
      "$POPS_ACTIVE_MPICXX_LIBRARY"; do
      active_library_hash="$("${SHA256_COMMAND[@]}" "$active_library" | awk '{print $1}')"
      printf 'library\0%s\0%s\0' "$active_library" "$active_library_hash"
    done
  } | "${SHA256_COMMAND[@]}" | awk '{print substr($1, 1, 32)}')"
  MPI_OVERLAY="$MPI_OVERLAY/$MPI_OVERLAY_ID"
  mkdir -p "$MPI_OVERLAY"
  for active_library in \
    "$POPS_ACTIVE_MPI_LIBRARY" \
    "$POPS_ACTIVE_PMPI_LIBRARY" \
    "$POPS_ACTIVE_MPICXX_LIBRARY"; do
    overlay_entry="$MPI_OVERLAY/$(basename "$active_library")"
    overlay_temporary="$overlay_entry.tmp.$$"
    ln -s "$active_library" "$overlay_temporary"
    mv -f "$overlay_temporary" "$overlay_entry"
  done
fi

export POPS_PARAVIEW_LIBPYTHON="$LIBPYTHON"
export POPS_ACTIVE_MPI_LIBRARY
export POPS_ACTIVE_PMPI_LIBRARY
export POPS_ACTIVE_MPICXX_LIBRARY
export POPS_ACTIVE_HDF5_LIBRARY
export PYTHONHOME="$PARAVIEW_PYTHONHOME"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PARAVIEW_PYTHON:$POPS_SITE"
export CATALYST_IMPLEMENTATION_PATHS="$PARAVIEW_LIB/catalyst"
unset __PYVENV_LAUNCHER__
if [[ "$(uname)" == Darwin ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$CONDA_PREFIX/lib:$PARAVIEW_LIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="$MPI_OVERLAY:$PARAVIEW_LIB"
fi

if [[ -n "$MPI_RANKS" ]]; then
  exec "$CONDA_PREFIX/bin/mpiexec" -n "$MPI_RANKS" \
    "$LAUNCHER" -m pops._paraview_python_bootstrap "$@"
fi
exec "$LAUNCHER" -m pops._paraview_python_bootstrap "$@"
