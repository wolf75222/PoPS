#!/usr/bin/env bash
#SBATCH --job-name=pops-adc757
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=pops-adc757-%j.out
#SBATCH --error=pops-adc757-%j.err

set -euo pipefail

romeo_load_armgpu_env
module load cuda/12.6
spack load openmpi +cuda

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${POPS_ADC757_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CANDIDATE_REF="${POPS_ADC757_CANDIDATE_REF:-HEAD}"
CANDIDATE_SHA="$(git -C "${REPO_ROOT}" rev-parse "${CANDIDATE_REF}^{commit}")"

WORK_ROOT="${POPS_ADC757_WORK_ROOT:-/scratch_p/${USER}/${SLURM_JOB_ID}/pops-adc757}"
RESULTS_DIR="${POPS_ADC757_RESULTS_DIR:-${HOME}/pops-benchmark-results/adc757}"
KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-${Kokkos_ROOT:-${HOME}/adc_gpu_p1/kinstall}}"
NVCC_WRAPPER="${POPS_NVCC_WRAPPER:-${KOKKOS_ROOT}/bin/nvcc_wrapper}"
RUNTIME_MATRIX_INPUT="${POPS_ADC757_RUNTIME_EVIDENCE:-}"
ABBA_BLOCKS="${POPS_ADC757_ABBA_BLOCKS:-5}"
EXTENT="${POPS_ADC757_EXTENT:-32768}"
INNER_ITERATIONS="${POPS_ADC757_INNER_ITERATIONS:-96}"
MIGRATION_VALUES_PER_TASK="${POPS_ADC757_MIGRATION_VALUES_PER_TASK:-4096}"
MINIMUM_SPEEDUP="${POPS_ADC757_MINIMUM_SPEEDUP:-1.01}"

test -x "${NVCC_WRAPPER}"
test "${SLURM_NTASKS:?}" -ge 2
test "${ABBA_BLOCKS}" -ge 5
case "${WORK_ROOT}" in
  /scratch_p/"${USER}"/*/pops-adc757) ;;
  *) echo "refusing unsafe POPS_ADC757_WORK_ROOT: ${WORK_ROOT}" >&2; exit 3 ;;
esac

cmake -E remove_directory "${WORK_ROOT}"
cmake -E make_directory \
  "${WORK_ROOT}/source" "${WORK_ROOT}/build" "${WORK_ROOT}/wheelhouse" "${RESULTS_DIR}"
git -C "${REPO_ROOT}" archive "${CANDIDATE_SHA}" | tar -xf - -C "${WORK_ROOT}/source"

# Build and retain one exact distributed CUDA wheel through the official installation route. The
# benchmark executable is configured later with the authenticated headers from this installation;
# it must never compile a second source-tree copy of PoPS.
source "${WORK_ROOT}/source/scripts/conda_runtime.sh"
if ! pops_load_conda; then
  echo "ADC-757 requires the official PoPS conda environment to build an exact wheel" >&2
  exit 3
fi
conda activate "${POPS_ENV_NAME:-pops}"
export CC="${CC:-gcc}"
export CXX="${NVCC_WRAPPER}"
export Kokkos_ROOT="${KOKKOS_ROOT}"
export POPS_KOKKOS_ROOT="${KOKKOS_ROOT}"
bash "${WORK_ROOT}/source/scripts/build_python.sh" --mpi \
  --wheel-dir "${WORK_ROOT}/wheelhouse"

WHEELS=("${WORK_ROOT}"/wheelhouse/pops-*.whl)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "ADC-757 expected exactly one retained PoPS wheel" >&2
  exit 3
fi
WHEEL="${WHEELS[0]}"
PYTHON="${CONDA_PREFIX}/bin/python"
WHEEL_PROOF="${WORK_ROOT}/installed-wheel-proof.json"
PYTHONPATH= PYTHONNOUSERSITE=1 \
  "${PYTHON}" "${WORK_ROOT}/source/scripts/prove_installed_wheel.py" \
  --wheel "${WHEEL}" > "${WHEEL_PROOF}"

# A vector kernel can no longer authorize a report. The installed probe calls pops.doctor, verifies
# wheel bytes/header ABI, and requires a four-mode same-scenario runtime matrix with C/G receipts.
# Until ADC-757C/G are integrated and such a matrix is supplied, it writes a precise refusal and the
# job stops before the ABBA microbenchmark.
RUNTIME_EVIDENCE="${WORK_ROOT}/installed-runtime-evidence.json"
RUNTIME_PROBE_ARGS=(
  --wheel-proof "${WHEEL_PROOF}"
  --source-root "${WORK_ROOT}/source"
  --expected-revision "${CANDIDATE_SHA}"
  --output "${RUNTIME_EVIDENCE}"
)
if [[ -n "${RUNTIME_MATRIX_INPUT}" ]]; then
  RUNTIME_PROBE_ARGS+=(--runtime-evidence "${RUNTIME_MATRIX_INPUT}")
fi
set +e
srun --kill-on-bad-exit=1 --ntasks=1 --gpus-per-task=1 \
  env PYTHONPATH= PYTHONNOUSERSITE=1 \
  "${PYTHON}" "${WORK_ROOT}/source/benchmarks/adc757/runtime_probe.py" \
  "${RUNTIME_PROBE_ARGS[@]}"
RUNTIME_PROBE_STATUS=$?
set -e
if [[ "${RUNTIME_PROBE_STATUS}" -ne 0 ]]; then
  if [[ -f "${RUNTIME_EVIDENCE}" ]]; then
    cp "${RUNTIME_EVIDENCE}" \
      "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-runtime-refusal.json"
  fi
  echo "ADC-757 report refused: installed runtime proof exited ${RUNTIME_PROBE_STATUS}" >&2
  exit "${RUNTIME_PROBE_STATUS}"
fi

readarray -t INSTALLED_IDENTITIES < <(
  PYTHONPATH= PYTHONNOUSERSITE=1 "${PYTHON}" - \
    "${WHEEL_PROOF}" "${RUNTIME_EVIDENCE}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from pops import _pops
from pops.codegen import toolchain
import importlib.metadata

wheel_proof = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
canonical = json.dumps(runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
print(wheel_proof["wheel_sha256"])
print(hashlib.sha256(_pops.abi_key().encode("utf-8")).hexdigest())
print(hashlib.sha256(canonical.encode("utf-8")).hexdigest())
print(importlib.metadata.distribution("pops").locate_file(""))
print(toolchain.pops_include())
PY
)
if [[ "${#INSTALLED_IDENTITIES[@]}" -ne 5 ]]; then
  echo "ADC-757 could not resolve exact installed identities" >&2
  exit 3
fi
WHEEL_SHA256="${INSTALLED_IDENTITIES[0]}"
MODULE_ABI_SHA256="${INSTALLED_IDENTITIES[1]}"
RUNTIME_EVIDENCE_SHA256="${INSTALLED_IDENTITIES[2]}"
POPS_INSTALL_PREFIX="${INSTALLED_IDENTITIES[3]}"
POPS_INCLUDE_ROOT="${INSTALLED_IDENTITIES[4]}"

cmake -S "${WORK_ROOT}/source/benchmarks/adc757" -B "${WORK_ROOT}/build" \
  -DPOPS_ADC757_REVISION="${CANDIDATE_SHA}" \
  -DPOPS_ADC757_WHEEL_SHA256="${WHEEL_SHA256}" \
  -DPOPS_ADC757_MODULE_ABI_SHA256="${MODULE_ABI_SHA256}" \
  -DPOPS_ADC757_INCLUDE_ROOT="${POPS_INCLUDE_ROOT}" \
  -DCMAKE_PREFIX_PATH="${POPS_INSTALL_PREFIX};${KOKKOS_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DKokkos_ROOT="${KOKKOS_ROOT}"
cmake --build "${WORK_ROOT}/build" --target adc757_heterogeneous_numerics \
  --parallel "${SLURM_CPUS_PER_TASK}"

EXECUTABLE="${WORK_ROOT}/build/bin/adc757_heterogeneous_numerics"
RAW="${WORK_ROOT}/measurements.jsonl"
INVENTORY="${WORK_ROOT}/device-inventory.txt"
REPORT="${WORK_ROOT}/report.json"
: > "${RAW}"

run_one() {
  local scenario="$1"
  local route="$2"
  local log="${WORK_ROOT}/${scenario}-${route}-${RUN_SERIAL}.log"
  # cudaDeviceGetUuid authenticates the physical device selected by each rank. The executable
  # gathers those UUIDs collectively and refuses aliased Slurm placement before measuring.
  srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 \
    "${EXECUTABLE}" \
    --scenario="${scenario}" \
    --route="${route}" \
    --extent="${EXTENT}" \
    --inner-iterations="${INNER_ITERATIONS}" \
    --migration-values-per-task="${MIGRATION_VALUES_PER_TASK}" \
    --runtime-evidence-sha256="${RUNTIME_EVIDENCE_SHA256}" | tee "${log}"
  grep -E '^\{"schema":"pops\.adc757\.heterogeneous-numerics\.measurement\.v1"' \
    "${log}" >> "${RAW}"
  RUN_SERIAL=$((RUN_SERIAL + 1))
}

RUN_SERIAL=0
for scenario in prepared_local_time cost_aware_load_balance; do
  for ((block = 0; block < ABBA_BLOCKS; ++block)); do
    run_one "${scenario}" baseline
    run_one "${scenario}" candidate
    run_one "${scenario}" candidate
    run_one "${scenario}" baseline
  done
done

python3 "${WORK_ROOT}/source/benchmarks/adc757/assemble.py" \
  --input "${RAW}" \
  --output "${REPORT}" \
  --device-inventory-output "${INVENTORY}" \
  --runtime-evidence "${RUNTIME_EVIDENCE}" \
  --expected-revision "${CANDIDATE_SHA}" \
  --minimum-speedup "${MINIMUM_SPEEDUP}"
python3 "${WORK_ROOT}/source/benchmarks/adc757/verify.py" \
  --input "${REPORT}" \
  --expected-revision "${CANDIDATE_SHA}"

cp "${RAW}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-measurements.jsonl"
cp "${INVENTORY}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-devices.txt"
cp "${WHEEL_PROOF}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-installed-wheel.json"
cp "${RUNTIME_EVIDENCE}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-installed-runtime.json"
cp "${REPORT}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-report.json"
echo "ADC757_REPORT=${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-report.json"
