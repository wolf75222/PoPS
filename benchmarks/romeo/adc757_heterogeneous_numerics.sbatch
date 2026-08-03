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
KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-${Kokkos_ROOT:-${HOME}/pops_gpu_p1/kinstall}}"
NVCC_WRAPPER="${POPS_NVCC_WRAPPER:-${KOKKOS_ROOT}/bin/nvcc_wrapper}"
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
cmake -E make_directory "${WORK_ROOT}/source" "${WORK_ROOT}/build" "${RESULTS_DIR}"
git -C "${REPO_ROOT}" archive "${CANDIDATE_SHA}" | tar -xf - -C "${WORK_ROOT}/source"

cmake -S "${WORK_ROOT}/source/benchmarks/adc757" -B "${WORK_ROOT}/build" \
  -DPOPS_ADC757_SOURCE_ROOT="${WORK_ROOT}/source" \
  -DPOPS_ADC757_REVISION="${CANDIDATE_SHA}" \
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

# Authenticate Slurm's one-GPU-per-rank placement before the timed campaign.
# shellcheck disable=SC2016 # Expanded deliberately by each rank-local bash.
srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 \
  bash -c '
    set -euo pipefail
    selector="${CUDA_VISIBLE_DEVICES:?Slurm did not assign a GPU to this rank}"
    if [[ "${selector}" == *,* ]]; then
      echo "rank ${SLURM_PROCID:?} sees more than one CUDA device: ${selector}" >&2
      exit 4
    fi
    uuid="$(nvidia-smi --id="${selector}" --query-gpu=uuid --format=csv,noheader |
      sed "s/^[[:space:]]*//;s/[[:space:]]*$//")"
    test -n "${uuid}"
    printf "%s\t%s\n" "${SLURM_PROCID}" "${uuid}"
  ' | sort -n -k1,1 > "${INVENTORY}"
test "$(cut -f2 "${INVENTORY}" | sort -u | wc -l)" -eq "${SLURM_NTASKS}"

run_one() {
  local scenario="$1"
  local route="$2"
  local log="${WORK_ROOT}/${scenario}-${route}-${RUN_SERIAL}.log"
  # Query the physical UUID inside each rank's Slurm GPU namespace and pass it to the native
  # harness.  The harness gathers and checks all UUIDs collectively before measuring anything.
  # shellcheck disable=SC2016
  srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 \
    bash -c '
      set -euo pipefail
      executable="$1"
      shift
      selector="${CUDA_VISIBLE_DEVICES:?Slurm did not assign a GPU to this rank}"
      if [[ "${selector}" == *,* ]]; then
        echo "rank ${SLURM_PROCID:?} sees more than one CUDA device: ${selector}" >&2
        exit 4
      fi
      export POPS_ADC757_DEVICE_UUID="$(
        nvidia-smi --id="${selector}" --query-gpu=uuid --format=csv,noheader |
          sed "s/^[[:space:]]*//;s/[[:space:]]*$//"
      )"
      test -n "${POPS_ADC757_DEVICE_UUID}"
      exec "${executable}" "$@"
    ' bash "${EXECUTABLE}" \
      --scenario="${scenario}" \
      --route="${route}" \
      --extent="${EXTENT}" \
      --inner-iterations="${INNER_ITERATIONS}" \
      --migration-values-per-task="${MIGRATION_VALUES_PER_TASK}" | tee "${log}"
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
  --expected-revision "${CANDIDATE_SHA}" \
  --minimum-speedup "${MINIMUM_SPEEDUP}"
python3 "${WORK_ROOT}/source/benchmarks/adc757/verify.py" \
  --input "${REPORT}" \
  --expected-revision "${CANDIDATE_SHA}"

cp "${RAW}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-measurements.jsonl"
cp "${INVENTORY}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-devices.txt"
cp "${REPORT}" "${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-report.json"
echo "ADC757_REPORT=${RESULTS_DIR}/adc757-${SLURM_JOB_ID}-report.json"
