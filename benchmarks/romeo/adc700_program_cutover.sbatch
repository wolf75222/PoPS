#!/usr/bin/env bash
#SBATCH --job-name=pops-adc700
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=pops-adc700-%j.out
#SBATCH --error=pops-adc700-%j.err

set -euo pipefail

module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${POPS_ADC700_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
BASELINE_REF="${POPS_ADC700_BASELINE_REF:-db3d390f43dfb14f12e88db31a9b3e631ff50488}"
CANDIDATE_REF="${POPS_ADC700_CANDIDATE_REF:-35203a76957cdfe1491e237d714f452a50a266ff}"
BASELINE_SHA="$(git -C "${REPO_ROOT}" rev-parse "${BASELINE_REF}^{commit}")"
CANDIDATE_SHA="$(git -C "${REPO_ROOT}" rev-parse "${CANDIDATE_REF}^{commit}")"

WORK_ROOT="${POPS_ADC700_WORK_ROOT:-/scratch_p/${USER}/${SLURM_JOB_ID}/pops-adc700}"
RESULTS_DIR="${POPS_ADC700_RESULTS_DIR:-${HOME}/pops-benchmark-results/adc700}"
KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-${Kokkos_ROOT:-${HOME}/pops_gpu_p1/kinstall}}"
NVCC_WRAPPER="${POPS_NVCC_WRAPPER:-${KOKKOS_ROOT}/bin/nvcc_wrapper}"
ABBA_BLOCKS="${POPS_ADC700_ABBA_BLOCKS:-5}"
DOMAIN_SIZE="${POPS_ADC700_DOMAIN_SIZE:-128}"
WARMUPS="${POPS_ADC700_WARMUPS:-4}"
MEASURED_STEPS="${POPS_ADC700_MEASURED_STEPS:-40}"
STEP_DT="${POPS_ADC700_DT:-5e-4}"

test -x "${NVCC_WRAPPER}"
test "${ABBA_BLOCKS}" -ge 1
case "${WORK_ROOT}" in
  /scratch_p/"${USER}"/*/"pops-adc700") ;;
  *) echo "refusing unsafe POPS_ADC700_WORK_ROOT: ${WORK_ROOT}" >&2; exit 3 ;;
esac

cmake -E remove_directory "${WORK_ROOT}"
cmake -E make_directory "${WORK_ROOT}/baseline" "${WORK_ROOT}/candidate" \
  "${WORK_ROOT}/build-baseline" "${WORK_ROOT}/build-candidate" "${RESULTS_DIR}"
git -C "${REPO_ROOT}" archive "${BASELINE_SHA}" | tar -xf - -C "${WORK_ROOT}/baseline"
git -C "${REPO_ROOT}" archive "${CANDIDATE_SHA}" | tar -xf - -C "${WORK_ROOT}/candidate"

configure_and_build() {
  local route="$1"
  local source_root="$2"
  local revision="$3"
  local build_root="$4"
  cmake -S "${REPO_ROOT}/benchmarks/adc700" -B "${build_root}" \
    -DPOPS_ADC700_SOURCE_ROOT="${source_root}" \
    -DPOPS_ADC700_ROUTE="${route}" \
    -DPOPS_ADC700_REVISION="${revision}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
    -DKokkos_ROOT="${KOKKOS_ROOT}"
  cmake --build "${build_root}" --target adc700_program_cutover \
    --parallel "${SLURM_CPUS_PER_TASK}"
}

configure_and_build pre_cutover_native "${WORK_ROOT}/baseline" "${BASELINE_SHA}" \
  "${WORK_ROOT}/build-baseline"
configure_and_build program_only "${WORK_ROOT}/candidate" "${CANDIDATE_SHA}" \
  "${WORK_ROOT}/build-candidate"

RAW="${WORK_ROOT}/measurements.jsonl"
INVENTORY="${WORK_ROOT}/device-inventory.txt"
REPORT="${WORK_ROOT}/report.json"
: > "${RAW}"
# shellcheck disable=SC2016 # Expanded deliberately by each rank-local bash.
srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 \
  bash -c '
    set -euo pipefail
    device_selector="${CUDA_VISIBLE_DEVICES:?Slurm did not assign a GPU to this rank}"
    if [[ "${device_selector}" == *,* ]]; then
      echo "rank ${SLURM_PROCID:?} sees more than one CUDA device: ${device_selector}" >&2
      exit 4
    fi
    mapfile -t device_uuids < <(
      nvidia-smi --id="${device_selector}" --query-gpu=uuid --format=csv,noheader |
        sed "s/^[[:space:]]*//;s/[[:space:]]*$//"
    )
    if [[ "${#device_uuids[@]}" -ne 1 || -z "${device_uuids[0]}" ]]; then
      echo "rank ${SLURM_PROCID:?} does not see exactly one assigned GPU" >&2
      exit 4
    fi
    printf "%s\t%s\n" "${SLURM_PROCID}" "${device_uuids[0]}"
  ' | sort -n -k1,1 > "${INVENTORY}"

run_one() {
  local label="$1"
  local executable="$2"
  local log="${WORK_ROOT}/${label}-${RUN_SERIAL}.log"
  srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 \
    "${executable}" \
    --n="${DOMAIN_SIZE}" \
    --warmups="${WARMUPS}" \
    --steps="${MEASURED_STEPS}" \
    --dt="${STEP_DT}" | tee "${log}"
  grep -E '^\{"schema":"pops\.adc700\.program_cutover\.measurement\.v1"' "${log}" >> "${RAW}"
  RUN_SERIAL=$((RUN_SERIAL + 1))
}

RUN_SERIAL=0
for ((block = 0; block < ABBA_BLOCKS; ++block)); do
  run_one baseline "${WORK_ROOT}/build-baseline/bin/adc700_program_cutover"
  run_one candidate "${WORK_ROOT}/build-candidate/bin/adc700_program_cutover"
  run_one candidate "${WORK_ROOT}/build-candidate/bin/adc700_program_cutover"
  run_one baseline "${WORK_ROOT}/build-baseline/bin/adc700_program_cutover"
done

python3 "${REPO_ROOT}/benchmarks/adc700/compare.py" \
  --input "${RAW}" \
  --output "${REPORT}" \
  --device-inventory "${INVENTORY}" \
  --baseline-revision "${BASELINE_SHA}" \
  --candidate-revision "${CANDIDATE_SHA}" \
  --threshold 0.98

cp "${RAW}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-measurements.jsonl"
cp "${INVENTORY}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-devices.txt"
cp "${REPORT}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-report.json"
echo "ADC700_REPORT=${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-report.json"
