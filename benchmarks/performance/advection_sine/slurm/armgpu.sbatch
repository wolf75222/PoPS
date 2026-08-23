#!/usr/bin/env bash
#SBATCH --job-name=pops-sine-gpu
#SBATCH --constraint=armgpu
# Account, time, memory, nodes, tasks, CPUs/task, GPUs, working directory and
# logs come from submit_armgpu.sh.

set -euo pipefail

ARCHIVE="${1:?missing authenticated source archive}"
MANIFEST="${2:?missing source manifest}"
ARCHIVE_SHA256="${3:?missing archive SHA-256}"
MANIFEST_SHA256="${4:?missing manifest SHA-256}"
SOURCE_SHA="${5:?missing source base SHA}"
SOURCE_DIRTY="${6:?missing source dirty state}"
TREE_SHA256="${7:?missing source-tree SHA-256}"
KOKKOS_ARCHIVE="${8:?missing authenticated Kokkos archive}"
KOKKOS_RECEIPT="${9:?missing Kokkos source receipt}"
KOKKOS_ARCHIVE_SHA256="${10:?missing Kokkos archive SHA-256}"
KOKKOS_RECEIPT_SHA256="${11:?missing Kokkos receipt SHA-256}"
CAMPAIGN_NAME="${12:?missing campaign name}"
JOB_PYTHON="${13:?missing aarch64 job Python}"
SCHEDULER_DIRECTORY="${14:?missing scheduler receipt directory}"
: "${SLURM_JOB_ACCOUNT:?submit through submit_armgpu.sh so the account is explicit}"
: "${SLURM_GPUS_ON_NODE:?submit through submit_armgpu.sh so GPU allocation is explicit}"
: "${POPS_PERF_SUBMIT_USER:?submit through submit_armgpu.sh with an explicit user}"
: "${POPS_PERF_SUBMIT_HOME:?submit through submit_armgpu.sh with an explicit home directory}"
PERF_USER="${POPS_PERF_SUBMIT_USER}"
PERF_HOME="${POPS_PERF_SUBMIT_HOME}"
case "${PERF_USER}" in
  *[!A-Za-z0-9_.-]*|'') echo "batch user contains unsupported characters" >&2; exit 3 ;;
esac
if [[ "${PERF_HOME}" != /* ]] || [[ ! -d "${PERF_HOME}" ]]; then
  echo "batch home directory must be an existing absolute path" >&2
  exit 3
fi

# The explicit Slurm allowlist does not inherit shell functions or PATH from
# the login node.  Bootstrap only ROMEO's audited Spack installation from a
# platform constant, after setting the smallest system-tool PATH required by
# this wrapper.  Do not source the user's shell startup files.
PATH="/usr/bin:/bin"
export PATH
ROMEO_SPACK_SETUP="/apps/2025/manual_install/spack-armgpu-1.0.x/share/spack/setup-env.sh"
if [[ ! -r "${ROMEO_SPACK_SETUP}" ]]; then
  echo "ROMEO armgpu Spack setup is unavailable: ${ROMEO_SPACK_SETUP}" >&2
  exit 3
fi
# shellcheck source=/dev/null
source "${ROMEO_SPACK_SETUP}"
if ! command -v spack >/dev/null; then
  echo "ROMEO armgpu Spack setup did not provide spack" >&2
  exit 3
fi
ROMEO_MODULES_SETUP="/etc/profile.d/modules.sh"
if [[ ! -r "${ROMEO_MODULES_SETUP}" ]]; then
  echo "ROMEO module setup is unavailable: ${ROMEO_MODULES_SETUP}" >&2
  exit 3
fi
# shellcheck source=/dev/null
source "${ROMEO_MODULES_SETUP}"
if ! command -v module >/dev/null; then
  echo "ROMEO module setup did not provide module" >&2
  exit 3
fi
module load cuda/12.6
if [[ "${JOB_PYTHON}" != /* ]] || [[ ! -x "${JOB_PYTHON}" ]]; then
  echo "ROMEO job Python must be an executable absolute path: ${JOB_PYTHON}" >&2
  exit 3
fi
printf '%s  %s\n' "${ARCHIVE_SHA256}" "${ARCHIVE}" | sha256sum --check --status
printf '%s  %s\n' "${MANIFEST_SHA256}" "${MANIFEST}" | sha256sum --check --status
printf '%s  %s\n' "${KOKKOS_ARCHIVE_SHA256}" "${KOKKOS_ARCHIVE}" | sha256sum --check --status
printf '%s  %s\n' "${KOKKOS_RECEIPT_SHA256}" "${KOKKOS_RECEIPT}" | sha256sum --check --status

# The ROMEO architecture loader provides Spack, but deliberately does not put
# CMake or Ninja on PATH.  Activate the audited 2025 packages first: location
# alone does not set their runtime-library search paths.  ROMEO compute nodes
# do not provide the login node's system libmd, so the default CMake route also
# activates its architecture-specific Spack libmd and exports its library
# directory.  Operators may instead provide architecture-compatible executables through
# POPS_CMAKE/POPS_NINJA; their runtime is still checked below.
activate_romeo_spack_packages() {
  local activation
  if ! command -v spack >/dev/null; then
    echo "ROMEO Spack is unavailable; cannot activate build-tool packages" >&2
    exit 3
  fi
  if activation="$(spack load --sh "$@" 2>/dev/null)"; then
    :
  else
    echo "ROMEO cannot generate a build-tool package activation" >&2
    exit 3
  fi
  if [[ -z "${activation}" ]]; then
    echo "ROMEO generated an empty build-tool package activation" >&2
    exit 3
  fi
  eval "${activation}"
}

resolve_romeo_build_tool() {
  local name="$1"
  local override="$2"
  local spec="$3"
  local executable="$4"
  local install_root
  local candidate
  if [[ -n "${override}" ]]; then
    candidate="${override}"
    if [[ "${candidate}" != /* ]]; then
      echo "ROMEO ${name} override must be an absolute path: ${candidate}" >&2
      exit 3
    fi
  else
    if ! command -v spack >/dev/null; then
      echo "ROMEO Spack is unavailable; cannot resolve ${name}" >&2
      exit 3
    fi
    if ! install_root="$(spack location -i "${spec}" 2>/dev/null)"; then
      echo "ROMEO cannot resolve ${name} package ${spec}" >&2
      exit 3
    fi
    candidate="${install_root}/bin/${executable}"
  fi
  if [[ ! -x "${candidate}" ]]; then
    echo "ROMEO ${name} is unavailable or not executable: ${candidate}" >&2
    exit 3
  fi
  printf '%s\n' "${candidate}"
}

# See x64cpu.sbatch for the full contract rationale.  This armgpu-specific
# activation is performed inside the allocated aarch64 job, never on x86_64.
activate_romeo_python_dependencies() {
  local activation="${POPS_PYTHON_DEPENDENCY_ACTIVATION_ARMGPU:-}"
  local numpy_spec="${POPS_PYTHON_NUMPY_ARMGPU_SPEC:-/qwvf7fx}"
  local pybind11_spec="${POPS_PYBIND11_ARMGPU_SPEC:-/oshtzly}"
  if [[ -n "${activation}" ]]; then
    if [[ "${activation}" != /* ]] || [[ ! -r "${activation}" ]]; then
      echo "ROMEO Python activation must be an absolute readable file: ${activation}" >&2
      exit 3
    fi
    # shellcheck source=/dev/null
    source "${activation}"
  else
    activate_romeo_spack_packages "${numpy_spec}" "${pybind11_spec}"
  fi
}

resolve_romeo_pybind11_dir() {
  local override="${POPS_PYBIND11_DIR_ARMGPU:-}"
  local candidate
  if [[ -n "${override}" ]]; then
    candidate="${override}"
  else
    candidate="$("${JOB_PYTHON}" -m pybind11 --cmakedir)"
  fi
  if [[ "${candidate}" != /* ]] || [[ ! -f "${candidate}/pybind11Config.cmake" ]]; then
    echo "ROMEO pybind11 CMake directory is unavailable: ${candidate}" >&2
    exit 3
  fi
  printf '%s\n' "${candidate}"
}

CMAKE_SPEC="${POPS_CMAKE_ARMGPU_SPEC:-/phhwuh3m6k5az7cm54dipabxxwuxxxol}"
NINJA_SPEC="${POPS_NINJA_ARMGPU_SPEC:-/vabslnoabb3wyuvemwm7f2xy5emjuebn}"
LIBMD_SPEC="${POPS_LIBMD_ARMGPU_SPEC:-/2fsvnxa6vagggjhy4veycjyxxqwi2vzh}"
DEFAULT_BUILD_TOOL_SPECS=()
if [[ -z "${POPS_CMAKE:-}" ]]; then
  DEFAULT_BUILD_TOOL_SPECS+=("${CMAKE_SPEC}")
  DEFAULT_BUILD_TOOL_SPECS+=("${LIBMD_SPEC}")
fi
if [[ -z "${POPS_NINJA:-}" ]]; then
  DEFAULT_BUILD_TOOL_SPECS+=("${NINJA_SPEC}")
fi
if (( ${#DEFAULT_BUILD_TOOL_SPECS[@]} )); then
  activate_romeo_spack_packages "${DEFAULT_BUILD_TOOL_SPECS[@]}"
fi
if [[ -z "${POPS_CMAKE:-}" ]]; then
  if ! LIBMD_ROOT="$(spack location -i "${LIBMD_SPEC}" 2>/dev/null)"; then
    echo "ROMEO cannot resolve libmd package ${LIBMD_SPEC}" >&2
    exit 3
  fi
  LIBMD_LIBRARY_DIR="${LIBMD_ROOT}/lib"
  if [[ ! -f "${LIBMD_LIBRARY_DIR}/libmd.so.0" ]]; then
    echo "ROMEO libmd is unavailable: ${LIBMD_LIBRARY_DIR}/libmd.so.0" >&2
    exit 3
  fi
  export LD_LIBRARY_PATH="${LIBMD_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
BUILD_CMAKE="$(resolve_romeo_build_tool cmake "${POPS_CMAKE:-}" "${CMAKE_SPEC}" cmake)"
BUILD_NINJA="$(resolve_romeo_build_tool ninja "${POPS_NINJA:-}" "${NINJA_SPEC}" ninja)"
"${BUILD_CMAKE}" --version >/dev/null
"${BUILD_NINJA}" --version >/dev/null
activate_romeo_python_dependencies
"${JOB_PYTHON}" - <<'PY'
import numpy
import pybind11
import sys

assert sys.version_info >= (3, 10)
assert tuple(int(part) for part in pybind11.__version__.split(".")[:2]) >= (2, 13)
PY
PYBIND11_DIR="$(resolve_romeo_pybind11_dir)"

WORK_ROOT="${POPS_PERF_WORK_ROOT:-/scratch_p/${PERF_USER}/${SLURM_JOB_ID}/pops-advection-sine}"
case "${WORK_ROOT}" in
  /scratch_p/"${PERF_USER}"/"${SLURM_JOB_ID}"/pops-advection-sine) ;;
  *) echo "refusing unsafe POPS_PERF_WORK_ROOT=${WORK_ROOT}" >&2; exit 3 ;;
esac
mkdir -p "${WORK_ROOT}/source" "${WORK_ROOT}/kokkos-source" "${WORK_ROOT}/build" "${WORK_ROOT}/raw" "${WORK_ROOT}/report" "${WORK_ROOT}/pops-cache" "${WORK_ROOT}/xdg-cache"
OBSERVED_ALLOCATION="${WORK_ROOT}/report/observed-allocation"
mkdir "${OBSERVED_ALLOCATION}"
scontrol show job -dd "${SLURM_JOB_ID}" >"${OBSERVED_ALLOCATION}/scontrol-job.txt"
scontrol show hostnames "${SLURM_NODELIST}" >"${OBSERVED_ALLOCATION}/nodelist.txt"
while IFS= read -r node; do
  scontrol show node "${node}"
done <"${OBSERVED_ALLOCATION}/nodelist.txt" >"${OBSERVED_ALLOCATION}/scontrol-nodes.txt"
hostname >"${OBSERVED_ALLOCATION}/hostname.txt"
lscpu >"${OBSERVED_ALLOCATION}/lscpu.txt"
nvidia-smi -q >"${OBSERVED_ALLOCATION}/nvidia-smi-q.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,compute_cap,pstate,clocks.current.sm,clocks.current.memory --format=csv,noheader >"${OBSERVED_ALLOCATION}/nvidia-smi-summary.csv"
write_environment_provenance() {
  "${JOB_PYTHON}" - "${OBSERVED_ALLOCATION}/environment-provenance.json" <<'PY'
import json
import os
import sys

allowed = (
    "SLURM_CLUSTER_NAME", "SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION", "SLURM_JOB_USER", "SLURM_NODELIST", "SLURM_NNODES",
    "SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK",
    "SLURM_CPU_BIND", "SLURM_CPU_BIND_LIST", "SLURM_GPU_BIND", "CUDA_VISIBLE_DEVICES",
    "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC", "OMP_NUM_THREADS",
)
values = {}
for name in allowed:
    value = os.environ.get(name)
    if value is None:
        continue
    if any(ord(character) < 32 for character in value):
        raise SystemExit(f"refusing control character in provenance variable {name}")
    values[name] = value
with open(sys.argv[1], "x", encoding="utf-8") as output:
    json.dump({"schema": "pops.environment-provenance.v1", "values": values}, output, sort_keys=True)
    output.write("\n")
PY
}
write_environment_provenance
{
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID}"
  printf 'SLURM_NODELIST=%s\n' "${SLURM_NODELIST}"
  printf 'SLURM_CPU_BIND=%s\n' "${SLURM_CPU_BIND:-}"
  printf 'SLURM_CPU_BIND_LIST=%s\n' "${SLURM_CPU_BIND_LIST:-}"
  printf 'SLURM_GPU_BIND=%s\n' "${SLURM_GPU_BIND:-}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'OMP_PROC_BIND=%s\n' "${OMP_PROC_BIND:-}"
  printf 'OMP_PLACES=%s\n' "${OMP_PLACES:-}"
  printf 'OMP_DYNAMIC=%s\n' "${OMP_DYNAMIC:-}"
} >"${OBSERVED_ALLOCATION}/affinity.txt"
PYBIND11_DIR="${PYBIND11_DIR}" "${JOB_PYTHON}" - <<'PY' >"${OBSERVED_ALLOCATION}/python-dependencies.json"
import json
import sys

import numpy
import pybind11

assert sys.version_info >= (3, 10)
assert tuple(int(part) for part in pybind11.__version__.split(".")[:2]) >= (2, 13)
print(
    json.dumps(
        {
            "python": sys.executable,
            "python_version": sys.version,
            "numpy_version": numpy.__version__,
            "numpy_path": numpy.__file__,
            "pybind11_version": pybind11.__version__,
            "pybind11_cmakedir": pybind11.get_cmake_dir(),
            "selected_pybind11_dir": __import__("os").environ["PYBIND11_DIR"],
        },
        sort_keys=True,
    )
)
PY
test -f "${SCHEDULER_DIRECTORY}/${SLURM_JOB_ID}.out"
test -f "${SCHEDULER_DIRECTORY}/${SLURM_JOB_ID}.err"
cp "${SCHEDULER_DIRECTORY}/${SLURM_JOB_ID}.out" "${OBSERVED_ALLOCATION}/scheduler.stdout"
cp "${SCHEDULER_DIRECTORY}/${SLURM_JOB_ID}.err" "${OBSERVED_ALLOCATION}/scheduler.stderr"
tar --same-permissions -xf "${ARCHIVE}" -C "${WORK_ROOT}/source"
HARNESS="${WORK_ROOT}/source/benchmarks/performance/advection_sine"
export POPS_INCLUDE="${WORK_ROOT}/source/include"
export POPS_CACHE_DIR="${WORK_ROOT}/pops-cache"
export XDG_CACHE_HOME="${WORK_ROOT}/xdg-cache"
test -d "${POPS_INCLUDE}"
# The verifier imports helpers from this extracted tree.  ``-B`` is
# load-bearing: authentication must not create and then reject a pycache file.
"${JOB_PYTHON}" -B "${HARNESS}/prepare_export.py" verify-tree \
  --source "${WORK_ROOT}/source" --manifest "${MANIFEST}" \
  --expect-base-sha "${SOURCE_SHA}" --expect-dirty "${SOURCE_DIRTY}" \
  --expect-tree-sha256 "${TREE_SHA256}" >/dev/null
cp "${MANIFEST}" "${WORK_ROOT}/report/source-export-manifest.json"
"${JOB_PYTHON}" -B "${HARNESS}/prepare_export.py" verify-kokkos-export \
  --archive "${KOKKOS_ARCHIVE}" --receipt "${KOKKOS_RECEIPT}" >/dev/null
tar --same-permissions -xf "${KOKKOS_ARCHIVE}" -C "${WORK_ROOT}/kokkos-source"

CAMPAIGN="${HARNESS}/campaigns/${CAMPAIGN_NAME}"
ROUTE="$("${JOB_PYTHON}" "${HARNESS}/validate_campaign.py" "${CAMPAIGN}" --field route)"
DIMENSION="$("${JOB_PYTHON}" "${HARNESS}/validate_campaign.py" "${CAMPAIGN}" --field dimension)"
CAMPAIGN_ID="$("${JOB_PYTHON}" "${HARNESS}/validate_campaign.py" "${CAMPAIGN}" --field id)"

MPI_ENABLED=OFF
case "${ROUTE}" in
  kokkos_cuda) ;;
  kokkos_cuda_mpi)
    MPI_ENABLED=ON
    spack load "${POPS_OPENMPI_GPU_SPEC:-openmpi@4.1.7+cuda}"
    ;;
  *) echo "armgpu job refuses route ${ROUTE}" >&2; exit 3 ;;
esac
KOKKOS_CUDA_SOURCE="${WORK_ROOT}/kokkos-source"
test -f "${KOKKOS_CUDA_SOURCE}/CMakeLists.txt"
NVCC_WRAPPER="${POPS_NVCC_WRAPPER:-${KOKKOS_CUDA_SOURCE}/bin/nvcc_wrapper}"
test -x "${NVCC_WRAPPER}"
KOKKOS_ROOT="${WORK_ROOT}/kokkos-cuda-install"
"${BUILD_CMAKE}" -S "${KOKKOS_CUDA_SOURCE}" -B "${WORK_ROOT}/kokkos-cuda-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DCMAKE_MAKE_PROGRAM="${BUILD_NINJA}" \
  -DCMAKE_INSTALL_PREFIX="${KOKKOS_ROOT}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SERIAL=ON
"${BUILD_CMAKE}" --build "${WORK_ROOT}/kokkos-cuda-build" --target install \
  --parallel "${SLURM_CPUS_PER_TASK}"
# Preserve the activated aarch64 Python dependency search path while making
# the authenticated build tree and source take import precedence.
PYTHON_DEPENDENCY_PATH="${PYTHONPATH:-}"

"${BUILD_CMAKE}" -S "${WORK_ROOT}/source" -B "${WORK_ROOT}/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DCMAKE_MAKE_PROGRAM="${BUILD_NINJA}" \
  -DPOPS_NATIVE_DIM="${DIMENSION}" \
  -DPOPS_BUILD_PYTHON=ON \
  -DPOPS_BUILD_TESTS=OFF \
  -DPOPS_USE_KOKKOS=ON \
  -DPOPS_USE_MPI="${MPI_ENABLED}" \
  -DPOPS_USE_HDF5=OFF \
  -DPython_EXECUTABLE="${JOB_PYTHON}" \
  -Dpybind11_DIR="${PYBIND11_DIR}" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DKokkos_ROOT="${KOKKOS_ROOT}"
"${BUILD_CMAKE}" --build "${WORK_ROOT}/build" --target _pops \
  --parallel "${SLURM_CPUS_PER_TASK}"

cp "${MANIFEST}" "${WORK_ROOT}/raw/source.manifest.json"
export POPS_NATIVE_VARIANTS_ROOT="${WORK_ROOT}/build/python/pops/_native"
# The imported extension is authenticated in a real Slurm GPU step.  This is
# provenance-only and occurs before every measured public lifecycle.
srun --exclusive --kill-on-bad-exit=1 --nodes=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 \
  env PYTHONPATH="${WORK_ROOT}/build/python:${WORK_ROOT}/source${PYTHON_DEPENDENCY_PATH:+:${PYTHON_DEPENDENCY_PATH}}" \
  POPS_NATIVE_VARIANTS_ROOT="${POPS_NATIVE_VARIANTS_ROOT}" "${JOB_PYTHON}" \
  "${HARNESS}/prepare_export.py" build-receipt \
  --source "${WORK_ROOT}/source" \
  --build "${WORK_ROOT}/build" \
  --manifest "${WORK_ROOT}/raw/source.manifest.json" \
  --campaign "${CAMPAIGN}" \
  --python "${JOB_PYTHON}" \
  --kokkos-root "${KOKKOS_ROOT}" \
  --kokkos-source-receipt "${KOKKOS_RECEIPT}" \
  --kokkos-build "${WORK_ROOT}/kokkos-cuda-build" \
  --output "${WORK_ROOT}/raw/build.receipt.json" >/dev/null

PYTHONPATH="${WORK_ROOT}/build/python:${WORK_ROOT}/source${PYTHON_DEPENDENCY_PATH:+:${PYTHON_DEPENDENCY_PATH}}" "${JOB_PYTHON}" "${HARNESS}/run_campaign.py" \
  --campaign "${CAMPAIGN}" \
  --python "${JOB_PYTHON}" \
  --output "${WORK_ROOT}/raw" \
  --launcher slurm
"${JOB_PYTHON}" "${HARNESS}/collect_results.py" \
  --campaign "${CAMPAIGN}" \
  --input "${WORK_ROOT}/raw" \
  --json "${WORK_ROOT}/report/summary.json" \
  --csv "${WORK_ROOT}/report/measurements.csv"
cp "${SCHEDULER_DIRECTORY}/${SLURM_JOB_ID}.out" "${OBSERVED_ALLOCATION}/scheduler.stdout"
cp "${SCHEDULER_DIRECTORY}/${SLURM_JOB_ID}.err" "${OBSERVED_ALLOCATION}/scheduler.stderr"

RESULTS_ROOT="${POPS_PERF_RESULTS_ROOT:-/scratch_p/${PERF_USER}/pops-sine-results}"
case "${RESULTS_ROOT}" in
  /scratch_p/"${PERF_USER}"/*) ;;
  *) echo "POPS_PERF_RESULTS_ROOT must be below /scratch_p/${PERF_USER}/" >&2; exit 3 ;;
esac
DESTINATION="${RESULTS_ROOT}/${CAMPAIGN_ID}-${SLURM_JOB_ID}"
STAGING="${RESULTS_ROOT}/.${CAMPAIGN_ID}-${SLURM_JOB_ID}.partial"
mkdir -p "${RESULTS_ROOT}"
test ! -e "${DESTINATION}"
test ! -e "${STAGING}"
mkdir "${STAGING}"
cp -R "${WORK_ROOT}/raw" "${WORK_ROOT}/report" "${STAGING}/"
PYTHONPATH="${WORK_ROOT}/source${PYTHON_DEPENDENCY_PATH:+:${PYTHON_DEPENDENCY_PATH}}" "${JOB_PYTHON}" "${HARNESS}/prepare_export.py" complete-receipt \
  --raw "${STAGING}/raw" \
  --report "${STAGING}/report" \
  --campaign "${CAMPAIGN_ID}" \
  --slurm-job-id "${SLURM_JOB_ID}" \
  --source-tree-sha256 "${TREE_SHA256}" \
  --output "${STAGING}/COMPLETE.json" >/dev/null
mv --no-clobber --no-target-directory -- "${STAGING}" "${DESTINATION}"
test ! -e "${STAGING}"
PYTHONPATH="${WORK_ROOT}/source${PYTHON_DEPENDENCY_PATH:+:${PYTHON_DEPENDENCY_PATH}}" "${JOB_PYTHON}" "${HARNESS}/prepare_export.py" verify-complete \
  --root "${DESTINATION}" >/dev/null
rm -rf -- "${WORK_ROOT}"
test ! -e "${WORK_ROOT}"
echo "POPS_SINE_RESULTS=${DESTINATION}"
