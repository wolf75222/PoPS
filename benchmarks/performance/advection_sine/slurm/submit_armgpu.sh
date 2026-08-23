#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${HARNESS_DIR}/../../.." && pwd)"
CAMPAIGN="${1:?usage: submit_armgpu.sh CAMPAIGN.json}"
shift
if (( $# != 0 )); then
  echo "submit_armgpu.sh refuses caller-provided sbatch options" >&2
  exit 2
fi

: "${POPS_SLURM_ACCOUNT:?export POPS_SLURM_ACCOUNT with the ROMEO project account}"
: "${POPS_PERF_LOGIN_PYTHON:?export an absolute x86_64 Python 3.10+ used on the login node}"
: "${POPS_PERF_JOB_PYTHON:?export an absolute aarch64 Python 3.10+ used only inside armgpu jobs}"
: "${USER:?the submitting user must be defined}"
: "${HOME:?the submitting home directory must be defined}"
POPS_PERF_SUBMIT_USER="${USER}"
POPS_PERF_SUBMIT_HOME="${HOME}"
case "${POPS_PERF_SUBMIT_USER}" in
  *[!A-Za-z0-9_.-]*|'') echo "submitting user contains unsupported characters" >&2; exit 2 ;;
esac
if [[ "${POPS_PERF_SUBMIT_HOME}" != /* ]] || [[ ! -d "${POPS_PERF_SUBMIT_HOME}" ]]; then
  echo "submitting home directory must be an existing absolute path" >&2
  exit 2
fi
for python_path in "${POPS_PERF_LOGIN_PYTHON}" "${POPS_PERF_JOB_PYTHON}"; do
  if [[ "${python_path}" != /* ]] || [[ ! -x "${python_path}" ]]; then
    echo "ROMEO Python paths must be executable absolute paths: ${python_path}" >&2
    exit 2
  fi
done
"${POPS_PERF_LOGIN_PYTHON}" -c 'import sys; assert sys.version_info >= (3, 10)'
for build_tool in "${POPS_CMAKE:-}" "${POPS_NINJA:-}"; do
  if [[ -n "${build_tool}" ]] && \
     { [[ "${build_tool}" != /* ]] || [[ ! -x "${build_tool}" ]]; }; then
    echo "POPS_CMAKE and POPS_NINJA overrides must be executable absolute paths" >&2
    exit 2
  fi
done

CAMPAIGN="$(cd -- "$(dirname -- "${CAMPAIGN}")" && pwd)/$(basename -- "${CAMPAIGN}")"
EXPECTED_CAMPAIGN="${HARNESS_DIR}/campaigns/$(basename -- "${CAMPAIGN}")"
if [[ "${CAMPAIGN}" != "${EXPECTED_CAMPAIGN}" ]]; then
  echo "campaign must be stored under ${HARNESS_DIR}/campaigns so it is source-authenticated" >&2
  exit 2
fi
test "$("${POPS_PERF_LOGIN_PYTHON}" "${HARNESS_DIR}/validate_campaign.py" "${CAMPAIGN}" --field platform)" = armgpu

EXPORT_ROOT="${POPS_PERF_EXPORT_ROOT:-/scratch_p/${USER}/pops-advection-sine-exports}"
case "${EXPORT_ROOT}" in
  /scratch_p/"${USER}"/*) ;;
  *) echo "POPS_PERF_EXPORT_ROOT must be below /scratch_p/${USER}/" >&2; exit 2 ;;
esac
mkdir -p "${EXPORT_ROOT}"
CANONICAL_SCRATCH_ROOT="$(cd -- "/scratch_p/${USER}" && pwd -P)"
SCHEDULER_ROOT_INPUT="${POPS_PERF_SCHEDULER_ROOT:-${EXPORT_ROOT}/scheduler}"
case "${SCHEDULER_ROOT_INPUT}" in
  /scratch_p/"${USER}"/*|"${CANONICAL_SCRATCH_ROOT}"/*) ;;
  *) echo "POPS_PERF_SCHEDULER_ROOT must be below /scratch_p/${USER}/" >&2; exit 2 ;;
esac
resolve_scheduler_path() {
  "${POPS_PERF_LOGIN_PYTHON}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' "$1"
}
SCHEDULER_ROOT="$(resolve_scheduler_path "${SCHEDULER_ROOT_INPUT}")"
case "${SCHEDULER_ROOT}" in
  "${CANONICAL_SCRATCH_ROOT}"/*) ;;
  *) echo "resolved POPS_PERF_SCHEDULER_ROOT escapes ${CANONICAL_SCRATCH_ROOT}/" >&2; exit 2 ;;
esac
mkdir -p "${SCHEDULER_ROOT}"
SCHEDULER_ROOT="$(cd -- "${SCHEDULER_ROOT}" && pwd -P)"
case "${SCHEDULER_ROOT}" in
  "${CANONICAL_SCRATCH_ROOT}"/*) ;;
  *) echo "resolved POPS_PERF_SCHEDULER_ROOT escapes ${CANONICAL_SCRATCH_ROOT}/" >&2; exit 2 ;;
esac
SCHEDULER_DIRECTORY_INPUT="${SCHEDULER_ROOT}/armgpu"
if [[ -e "${SCHEDULER_DIRECTORY_INPUT}" && ! -d "${SCHEDULER_DIRECTORY_INPUT}" ]]; then
  echo "refusing scheduler-log collision with non-directory ${SCHEDULER_DIRECTORY_INPUT}" >&2
  exit 2
fi
SCHEDULER_DIRECTORY="$(resolve_scheduler_path "${SCHEDULER_DIRECTORY_INPUT}")"
case "${SCHEDULER_DIRECTORY}" in
  "${CANONICAL_SCRATCH_ROOT}"/*) ;;
  *) echo "resolved scheduler-log directory escapes ${CANONICAL_SCRATCH_ROOT}/" >&2; exit 2 ;;
esac
mkdir -p "${SCHEDULER_DIRECTORY}"
SCHEDULER_DIRECTORY="$(cd -- "${SCHEDULER_DIRECTORY}" && pwd -P)"
case "${SCHEDULER_DIRECTORY}" in
  "${CANONICAL_SCRATCH_ROOT}"/*) ;;
  *) echo "resolved scheduler-log directory escapes ${CANONICAL_SCRATCH_ROOT}/" >&2; exit 2 ;;
esac
TEMPORARY_DIRECTORY="$(mktemp -d "${EXPORT_ROOT}/.prepare.XXXXXXXX")"
TEMPORARY_BASE="${TEMPORARY_DIRECTORY}/source"
cleanup_temporary_export() {
  rm -f -- "${TEMPORARY_BASE}.tar" "${TEMPORARY_BASE}.json"
  rmdir -- "${TEMPORARY_DIRECTORY}" 2>/dev/null || true
}
trap cleanup_temporary_export EXIT
"${POPS_PERF_LOGIN_PYTHON}" "${HARNESS_DIR}/prepare_export.py" create \
  --source "${REPO_ROOT}" \
  --archive "${TEMPORARY_BASE}.tar" \
  --manifest "${TEMPORARY_BASE}.json" >/dev/null

manifest_field() {
  "${POPS_PERF_LOGIN_PYTHON}" -c \
    'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]; print(int(value) if isinstance(value, bool) else value)' \
    "$1" "$2"
}
SOURCE_SHA="$(manifest_field "${TEMPORARY_BASE}.json" base_sha)"
SOURCE_DIRTY="$(manifest_field "${TEMPORARY_BASE}.json" source_dirty)"
TREE_SHA256="$(manifest_field "${TEMPORARY_BASE}.json" tree_sha256)"
ARCHIVE_SHA256="$(manifest_field "${TEMPORARY_BASE}.json" archive_sha256)"
EXPORT_ID="${SOURCE_SHA}-${SOURCE_DIRTY}-${TREE_SHA256}"
ARCHIVE="${EXPORT_ROOT}/${EXPORT_ID}.tar"
MANIFEST="${EXPORT_ROOT}/${EXPORT_ID}.json"
if [[ -e "${ARCHIVE}" || -e "${MANIFEST}" ]]; then
  if [[ ! -f "${ARCHIVE}" || ! -f "${MANIFEST}" ]] || \
     ! cmp -s "${TEMPORARY_BASE}.json" "${MANIFEST}" || \
     ! printf '%s  %s\n' "${ARCHIVE_SHA256}" "${ARCHIVE}" | sha256sum --check --status; then
    echo "refusing to replace a conflicting authenticated source export ${EXPORT_ID}" >&2
    exit 2
  fi
else
  mv -n "${TEMPORARY_BASE}.tar" "${ARCHIVE}"
  test ! -e "${TEMPORARY_BASE}.tar"
  mv -n "${TEMPORARY_BASE}.json" "${MANIFEST}"
  test ! -e "${TEMPORARY_BASE}.json"
fi
cleanup_temporary_export
trap - EXIT
MANIFEST_SHA256="$("${POPS_PERF_LOGIN_PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${MANIFEST}")"

RESOURCE_ARGS=()
WALLTIME_OVERRIDE_ARGS=()
if [[ -n "${POPS_PERF_WALLTIME_PARTITION:-}" ]]; then
  WALLTIME_OVERRIDE_ARGS+=(--slurm-partition "${POPS_PERF_WALLTIME_PARTITION}")
  [[ -n "${POPS_PERF_WALLTIME:-}" ]] && WALLTIME_OVERRIDE_ARGS+=(--slurm-time "${POPS_PERF_WALLTIME}")
elif [[ -n "${POPS_PERF_WALLTIME:-}" ]]; then
  echo "POPS_PERF_WALLTIME requires POPS_PERF_WALLTIME_PARTITION=short" >&2
  exit 2
fi
while IFS= read -r argument; do
  RESOURCE_ARGS+=("${argument}")
done < <("${POPS_PERF_LOGIN_PYTHON}" "${HARNESS_DIR}/validate_campaign.py" "${CAMPAIGN}" --slurm-args "${WALLTIME_OVERRIDE_ARGS[@]}")

# The job gets a deliberate, reviewable environment instead of the submitting
# shell.  In particular, credentials, tokens, proxy settings and unrelated
# Python paths cannot enter Slurm or its archived provenance by accident.
append_sbatch_export() {
  local name="$1"
  local value="${!name-}"
  if [[ -z "${value}" ]]; then
    echo "refusing an empty required Slurm export ${name}" >&2
    exit 2
  fi
  if [[ "${value}" == *$'\n'* || "${value}" == *$'\r'* || "${value}" == *,* ]]; then
    echo "refusing unsafe value for exported Slurm variable ${name}" >&2
    exit 2
  fi
  if [[ -z "${SBATCH_EXPORT}" ]]; then
    SBATCH_EXPORT="${name}=${value}"
  else
    SBATCH_EXPORT+=",${name}=${value}"
  fi
}

SBATCH_EXPORT=""
SBATCH_EXPORT_VARIABLES=(
  POPS_PERF_SUBMIT_USER POPS_PERF_SUBMIT_HOME
  POPS_CMAKE POPS_NINJA POPS_NVCC_WRAPPER
  POPS_CMAKE_ARMGPU_SPEC POPS_NINJA_ARMGPU_SPEC POPS_LIBMD_ARMGPU_SPEC
  POPS_PYTHON_DEPENDENCY_ACTIVATION_ARMGPU POPS_PYTHON_NUMPY_ARMGPU_SPEC
  POPS_PYBIND11_ARMGPU_SPEC POPS_PYBIND11_DIR_ARMGPU
  POPS_KOKKOS_CUDA_ROOT POPS_OPENMPI_GPU_SPEC
  POPS_PERF_WORK_ROOT POPS_PERF_RESULTS_ROOT
)
for export_name in "${SBATCH_EXPORT_VARIABLES[@]}"; do
  if [[ -n "${!export_name-}" ]] || [[ "${export_name}" == POPS_PERF_SUBMIT_USER || "${export_name}" == POPS_PERF_SUBMIT_HOME ]]; then
    append_sbatch_export "${export_name}"
  fi
done
[[ -n "${SBATCH_EXPORT}" ]] || { echo "refusing an empty Slurm export allowlist" >&2; exit 2; }

COMMAND=(sbatch "--export=${SBATCH_EXPORT}" --chdir="${SCHEDULER_DIRECTORY}"
  --output="${SCHEDULER_DIRECTORY}/%j.out" --error="${SCHEDULER_DIRECTORY}/%j.err"
  --account="${POPS_SLURM_ACCOUNT}" "${RESOURCE_ARGS[@]}"
  "${SCRIPT_DIR}/armgpu.sbatch" "${ARCHIVE}" "${MANIFEST}" "${ARCHIVE_SHA256}"
  "${MANIFEST_SHA256}" "${SOURCE_SHA}" "${SOURCE_DIRTY}" "${TREE_SHA256}"
  "$(basename -- "${CAMPAIGN}")" "${POPS_PERF_JOB_PYTHON}" "${SCHEDULER_DIRECTORY}")
if [[ "${POPS_PERF_SUBMIT_DRY_RUN:-0}" = 1 ]]; then
  printf 'authenticated source: base=%s dirty=%s tree_sha256=%s\n' \
    "${SOURCE_SHA}" "${SOURCE_DIRTY}" "${TREE_SHA256}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
