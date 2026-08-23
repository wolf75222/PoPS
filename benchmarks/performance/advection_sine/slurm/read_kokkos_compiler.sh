#!/usr/bin/env bash
# Resolve the compiler recorded by an installed Kokkos CMake package.

set -euo pipefail

CONFIG="${1:?usage: read_kokkos_compiler.sh KokkosConfigCommon.cmake}"
test -f "${CONFIG}"

# CMake commands are case-insensitive.  Kokkos installations encountered on
# ROMEO use both `set(...)` and `SET(...)`, so the parser must be as well.
EXTRACT_EXPRESSION='s/^[[:space:]]*[Ss][Ee][Tt][[:space:]]*([[:space:]]*Kokkos_CXX_COMPILER[[:space:]]*"\([^"]*\)"[[:space:]]*)[[:space:]]*\(#.*\)\{0,1\}$/\1/p'
MATCH_COUNT="$(sed -n "${EXTRACT_EXPRESSION}" "${CONFIG}" | wc -l | tr -d '[:space:]')"
COMPILER="$(sed -n "${EXTRACT_EXPRESSION}" "${CONFIG}")"

if [[ "${MATCH_COUNT}" != 1 || -z "${COMPILER}" || "${COMPILER}" == *$'\n'* ]]; then
  echo "expected exactly one quoted Kokkos_CXX_COMPILER in ${CONFIG}" >&2
  exit 3
fi

printf '%s\n' "${COMPILER}"
