#!/usr/bin/env bash
#SBATCH --job-name=pops-adc700
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
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
CANDIDATE_REF="${POPS_ADC700_CANDIDATE_REF:-HEAD}"
BASELINE_SHA="$(git -C "${REPO_ROOT}" rev-parse "${BASELINE_REF}^{commit}")"
CANDIDATE_SHA="$(git -C "${REPO_ROOT}" rev-parse "${CANDIDATE_REF}^{commit}")"
if [[ "${BASELINE_SHA}" != "db3d390f43dfb14f12e88db31a9b3e631ff50488" ]]; then
  echo "ADC-700 baseline ref must resolve to db3d390f43dfb14f12e88db31a9b3e631ff50488" >&2
  exit 3
fi
if [[ "${CANDIDATE_SHA}" == "${BASELINE_SHA}" ]]; then
  echo "ADC-700 candidate ref must differ from the pinned baseline" >&2
  exit 3
fi

# A ROMEO receipt is valid only for an immutable archive.  In particular, HEAD must already
# contain the Python candidate and every harness consumed below; an untracked local script must
# never disappear silently when git archive creates the candidate tree.  Campaign-owned dirty files
# are rejected even if unrelated repository work is present for another lane.
REQUIRED_CANDIDATE_PATHS=(
  benchmarks/adc700/CMakeLists.txt
  benchmarks/adc700/compare.py
  benchmarks/adc700/program_cutover.cpp
  benchmarks/adc700/program_cutover.py
  benchmarks/adc700/archive_receipt.py
  benchmarks/adc700/verify.py
  benchmarks/adc700/with_gpu_identity.sh
  benchmarks/manifest.toml
  benchmarks/romeo/adc700_program_cutover.sbatch
  benchmarks/romeo/submit_adc700_program_cutover.sh
  scripts/build_python.sh
  scripts/conda_runtime.sh
  scripts/prove_installed_wheel.py
  scripts/preserve_native_variants.py
  scripts/write_native_variant_manifest.py
  scripts/codesign_pops_extensions.py
  scripts/verify_installed_native.py
)
for relative_path in "${REQUIRED_CANDIDATE_PATHS[@]}"; do
  if ! git -C "${REPO_ROOT}" cat-file -e "${CANDIDATE_SHA}:${relative_path}"; then
    echo "ADC-700 candidate revision ${CANDIDATE_SHA} does not contain ${relative_path}; refusing archive" >&2
    exit 3
  fi
  if [[ "$(git -C "${REPO_ROOT}" cat-file -t "${CANDIDATE_SHA}:${relative_path}")" != blob ]]; then
    echo "ADC-700 candidate path ${relative_path} is not a regular committed file; refusing archive" >&2
    exit 3
  fi
  tree_entry="$(git -C "${REPO_ROOT}" ls-tree "${CANDIDATE_SHA}" -- "${relative_path}")"
  tree_mode="${tree_entry%% *}"
  tree_type="${tree_entry#* }"
  tree_type="${tree_type%% *}"
  if [[ "${tree_mode}" != 100644 && "${tree_mode}" != 100755 ]] || [[ "${tree_type}" != blob ]]; then
    echo "ADC-700 candidate path ${relative_path} is not a regular committed file mode; refusing archive" >&2
    exit 3
  fi
done
if ! git -C "${REPO_ROOT}" diff --quiet "${CANDIDATE_SHA}" -- "${REQUIRED_CANDIDATE_PATHS[@]}"; then
  echo "ADC-700 campaign-owned files are dirty relative to ${CANDIDATE_SHA}; refusing receipt" >&2
  exit 3
fi
if git -C "${REPO_ROOT}" status --porcelain --untracked-files=all -- "${REQUIRED_CANDIDATE_PATHS[@]}" | grep -q .; then
  echo "ADC-700 campaign-owned files have untracked/working-tree overlays; refusing receipt" >&2
  exit 3
fi

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
test "${SLURM_NTASKS:?}" -eq 4
test "${ABBA_BLOCKS}" -ge 5
case "${WORK_ROOT}" in
  /scratch_p/"${USER}"/*/"pops-adc700") ;;
  *) echo "refusing unsafe POPS_ADC700_WORK_ROOT: ${WORK_ROOT}" >&2; exit 3 ;;
esac

cmake -E remove_directory "${WORK_ROOT}"
cmake -E make_directory "${WORK_ROOT}/baseline" "${WORK_ROOT}/candidate" \
  "${WORK_ROOT}/build-baseline" "${WORK_ROOT}/wheelhouse" "${RESULTS_DIR}"
git -C "${REPO_ROOT}" archive "${BASELINE_SHA}" | tar -xf - -C "${WORK_ROOT}/baseline"
git -C "${REPO_ROOT}" archive "${CANDIDATE_SHA}" | tar -xf - -C "${WORK_ROOT}/candidate"
CAMPAIGN_ROOT="${WORK_ROOT}/candidate"
GPU_WRAPPER="${CAMPAIGN_ROOT}/benchmarks/adc700/with_gpu_identity.sh"
test -f "${CAMPAIGN_ROOT}/benchmarks/adc700/program_cutover.py"
test -f "${CAMPAIGN_ROOT}/benchmarks/adc700/verify.py"
test -f "${GPU_WRAPPER}"
# Keep one helper copy outside both extracted trees.  Its bytes are copied from the committed
# candidate archive and rechecked against that archived harness before every receipt verification;
# the pinned baseline intentionally does not contain archive_receipt.py.
ARCHIVE_HELPER="${WORK_ROOT}/archive-receipt.py"
cp "${CAMPAIGN_ROOT}/benchmarks/adc700/archive_receipt.py" "${ARCHIVE_HELPER}"
chmod a-w "${ARCHIVE_HELPER}"
test "$(sha256sum "${ARCHIVE_HELPER}" | awk '{print $1}')" = \
     "$(sha256sum "${CAMPAIGN_ROOT}/benchmarks/adc700/archive_receipt.py" | awk '{print $1}')"
# The baseline archive is read-only for its native build.  The candidate remains writable only
# until the official wheel build has completed; it is then frozen before any MODULE compilation,
# measurement, comparison, or verification can run against a mutable checkout overlay.
chmod -R a-w "${WORK_ROOT}/baseline"

# Build and retain one exact Dim=2 MPI CUDA wheel.  The candidate driver imports only this wheel,
# while its generated MODULE is compiled through the same NVCC wrapper/include root recorded below.
source "${WORK_ROOT}/candidate/scripts/conda_runtime.sh"
if ! pops_load_conda; then
  echo "ADC-700 requires the official PoPS conda environment for the candidate MODULE" >&2
  exit 3
fi
conda activate "${POPS_ENV_NAME:-pops}"
export CC="${CC:-gcc}"
export CXX="${NVCC_WRAPPER}"
export Kokkos_ROOT="${KOKKOS_ROOT}"
export POPS_KOKKOS_ROOT="${KOKKOS_ROOT}"
export POPS_NATIVE_DIM=2
bash "${WORK_ROOT}/candidate/scripts/build_python.sh" --dim 2 --mpi \
  --wheel-dir "${WORK_ROOT}/wheelhouse"
WHEELS=("${WORK_ROOT}"/wheelhouse/pops-*.whl)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "ADC-700 expected exactly one candidate PoPS wheel" >&2
  exit 3
fi
WHEEL="${WHEELS[0]}"
PYTHON="${CONDA_PREFIX}/bin/python"
WHEEL_PROOF="${WORK_ROOT}/installed-wheel-proof.json"
PYTHONPATH= PYTHONNOUSERSITE=1 "${PYTHON}" \
  "${WORK_ROOT}/candidate/scripts/prove_installed_wheel.py" \
  --wheel "${WHEEL}" --expect-dim 2 > "${WHEEL_PROOF}"
POPS_INCLUDE_ROOT="${WORK_ROOT}/candidate/include"

# Inspect the exact imported MPI/Kokkos targets with the same wrapper/root used for the wheel.  The
# resulting immutable contract supplies target options/definitions/libraries that the installed
# Python extension cannot expose by itself; the Python toolchain receipt cross-checks every common
# field before it is handed to the pinned C++ oracle.
TOOLCHAIN_PROBE="${WORK_ROOT}/cmake-target-contract.json"
cmake -S "${CAMPAIGN_ROOT}/benchmarks/adc700" -B "${WORK_ROOT}/build-toolchain-probe" \
  -DPOPS_ADC700_TOOLCHAIN_PROBE_ONLY=ON \
  -DPOPS_ADC700_TOOLCHAIN_PROBE_OUTPUT="${TOOLCHAIN_PROBE}" \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DKokkos_ROOT="${KOKKOS_ROOT}"
test -s "${TOOLCHAIN_PROBE}"
chmod a-w "${TOOLCHAIN_PROBE}"

# Capture one canonical toolchain object after the wheel has been proven.  The four-rank
# toolchain-only invocation uses the installed extension's authenticated NVCC/MPI/Kokkos facts;
# the pinned C++ oracle reads this exact immutable object and emits it in every baseline JSONL row.
TOOLCHAIN_RECEIPT="${WORK_ROOT}/toolchain-receipt.json"
srun --kill-on-bad-exit=1 --ntasks=4 --gpus-per-task=1 \
  bash "${GPU_WRAPPER}" env PYTHONPATH= PYTHONNOUSERSITE=1 \
  POPS_ADC700_INCLUDE_ROOT="${POPS_INCLUDE_ROOT}" \
  POPS_ADC700_NVCC_WRAPPER="${NVCC_WRAPPER}" \
  POPS_ADC700_WHEEL_PROOF="${WHEEL_PROOF}" \
  "${PYTHON}" "${CAMPAIGN_ROOT}/benchmarks/adc700/program_cutover.py" \
  --route=program_only --revision="${CANDIDATE_SHA}" \
  --include="${WORK_ROOT}/candidate/include" --cxx="${NVCC_WRAPPER}" \
  --wheel-proof="${WHEEL_PROOF}" --toolchain-probe="${TOOLCHAIN_PROBE}" --toolchain-only \
  --toolchain-receipt="${TOOLCHAIN_RECEIPT}"
test -s "${TOOLCHAIN_RECEIPT}"
chmod a-w "${TOOLCHAIN_RECEIPT}"
TOOLCHAIN_RECEIPT_SHA256="$(sha256sum "${TOOLCHAIN_RECEIPT}" | awk '{print $1}')"

# The baseline is the only C++ binary.  Configure it only after the candidate wheel has issued the
# shared receipt: ADC-700/CMake then compares its real MPI::MPI_CXX and Kokkos::kokkos target
# properties, compiler wrapper, options, definitions, libraries, and exact C++20 standard against
# that receipt before compiling the pinned oracle.  A second C++ candidate executable is forbidden.
cmake -S "${CAMPAIGN_ROOT}/benchmarks/adc700" -B "${WORK_ROOT}/build-baseline" \
  -DPOPS_ADC700_SOURCE_ROOT="${WORK_ROOT}/baseline" \
  -DPOPS_ADC700_ROUTE=pre_cutover_native \
  -DPOPS_ADC700_REVISION="${BASELINE_SHA}" \
  -DPOPS_ADC700_TOOLCHAIN_RECEIPT="${TOOLCHAIN_RECEIPT}" \
  -DPOPS_ADC700_TOOLCHAIN_RECEIPT_SHA256="${TOOLCHAIN_RECEIPT_SHA256}" \
  -DPOPS_ADC700_TOOLCHAIN_REVISION="${CANDIDATE_SHA}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DKokkos_ROOT="${KOKKOS_ROOT}"
cmake --build "${WORK_ROOT}/build-baseline" --target adc700_program_cutover \
  --parallel "${SLURM_CPUS_PER_TASK}"

# Freeze both extracted source trees before any campaign executable is launched.  The receipt
# helper records every regular file/mode/hash and a canonical tree digest; run_one re-verifies both
# manifests immediately before every ABBA route invocation.
chmod -R a-w "${WORK_ROOT}/candidate"
BASELINE_ARCHIVE_RECEIPT="${WORK_ROOT}/baseline-archive-receipt.json"
CANDIDATE_ARCHIVE_RECEIPT="${WORK_ROOT}/candidate-archive-receipt.json"
python3 "${ARCHIVE_HELPER}" \
  --build --root "${WORK_ROOT}/baseline" --role baseline --revision "${BASELINE_SHA}" \
  --receipt "${BASELINE_ARCHIVE_RECEIPT}" --helper "${ARCHIVE_HELPER}"
python3 "${ARCHIVE_HELPER}" \
  --build --root "${WORK_ROOT}/candidate" --role candidate --revision "${CANDIDATE_SHA}" \
  --receipt "${CANDIDATE_ARCHIVE_RECEIPT}" --helper "${ARCHIVE_HELPER}"
chmod a-w "${BASELINE_ARCHIVE_RECEIPT}" "${CANDIDATE_ARCHIVE_RECEIPT}"

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
  local route="$2"
  local log="${WORK_ROOT}/${label}-${RUN_SERIAL}.log"
  python3 "${ARCHIVE_HELPER}" \
    --root "${WORK_ROOT}/baseline" --role baseline --revision "${BASELINE_SHA}" \
    --receipt "${BASELINE_ARCHIVE_RECEIPT}" --helper "${ARCHIVE_HELPER}" >/dev/null
  python3 "${ARCHIVE_HELPER}" \
    --root "${WORK_ROOT}/candidate" --role candidate --revision "${CANDIDATE_SHA}" \
    --receipt "${CANDIDATE_ARCHIVE_RECEIPT}" --helper "${ARCHIVE_HELPER}" >/dev/null
  if [[ "$(sha256sum "${TOOLCHAIN_RECEIPT}" | awk '{print $1}')" != "${TOOLCHAIN_RECEIPT_SHA256}" ]]; then
    echo "ADC-700 toolchain receipt changed before ${label}" >&2
    exit 3
  fi
  if [[ "${route}" == baseline ]]; then
    srun --kill-on-bad-exit=1 --ntasks=4 --gpus-per-task=1 \
      bash "${GPU_WRAPPER}" env \
      POPS_ADC700_TOOLCHAIN_RECEIPT="${TOOLCHAIN_RECEIPT}" \
      POPS_ADC700_TOOLCHAIN_RECEIPT_SHA256="${TOOLCHAIN_RECEIPT_SHA256}" \
      POPS_ADC700_TOOLCHAIN_RECEIPT_REVISION="${CANDIDATE_SHA}" \
      "${WORK_ROOT}/build-baseline/bin/adc700_program_cutover" \
      --n="${DOMAIN_SIZE}" --warmups="${WARMUPS}" --steps="${MEASURED_STEPS}" \
      --dt="${STEP_DT}" | tee "${log}"
  elif [[ "${route}" == candidate ]]; then
    # Empty PYTHONPATH is intentional: this must be the wheel just proved above, never a source
    # checkout overlay.  The Python driver emits one rank-zero JSONL record or exits non-zero.
    srun --kill-on-bad-exit=1 --ntasks=4 --gpus-per-task=1 \
      bash "${GPU_WRAPPER}" env PYTHONPATH= PYTHONNOUSERSITE=1 \
      POPS_ADC700_INCLUDE_ROOT="${POPS_INCLUDE_ROOT}" \
      POPS_ADC700_NVCC_WRAPPER="${NVCC_WRAPPER}" \
      POPS_ADC700_WHEEL_PROOF="${WHEEL_PROOF}" \
      POPS_ADC700_ARCHIVE_RECEIPT="${CANDIDATE_ARCHIVE_RECEIPT}" \
      POPS_ADC700_ARCHIVE_HELPER="${ARCHIVE_HELPER}" \
      POPS_ADC700_BASELINE_TOOLCHAIN_ATTESTED=1 \
      POPS_ADC700_TOOLCHAIN_RECEIPT="${TOOLCHAIN_RECEIPT}" \
      "${PYTHON}" "${CAMPAIGN_ROOT}/benchmarks/adc700/program_cutover.py" \
      --route=program_only --revision="${CANDIDATE_SHA}" \
      --include="${POPS_INCLUDE_ROOT}" --cxx="${NVCC_WRAPPER}" \
      --wheel-proof="${WHEEL_PROOF}" --toolchain-probe="${TOOLCHAIN_PROBE}" \
      --archive-receipt="${CANDIDATE_ARCHIVE_RECEIPT}" \
      --archive-helper="${ARCHIVE_HELPER}" \
      --toolchain-receipt="${TOOLCHAIN_RECEIPT}" \
      --n="${DOMAIN_SIZE}" --warmups="${WARMUPS}" --steps="${MEASURED_STEPS}" \
      --dt="${STEP_DT}" | tee "${log}"
  else
    echo "unknown ADC-700 route: ${route}" >&2
    exit 3
  fi
  grep -E '^\{"schema":"pops\.adc700\.program_cutover\.measurement\.v1"' "${log}" >> "${RAW}"
  RUN_SERIAL=$((RUN_SERIAL + 1))
}

RUN_SERIAL=0
for ((block = 0; block < ABBA_BLOCKS; ++block)); do
  run_one baseline baseline
  run_one candidate candidate
  run_one candidate candidate
  run_one baseline baseline
done

python3 "${CAMPAIGN_ROOT}/benchmarks/adc700/compare.py" \
  --input "${RAW}" \
  --output "${REPORT}" \
  --device-inventory "${INVENTORY}" \
  --baseline-revision "${BASELINE_SHA}" \
  --candidate-revision "${CANDIDATE_SHA}" \
  --threshold 0.98

VERIFIED_REPORT="${WORK_ROOT}/verified-report.json"
python3 "${CAMPAIGN_ROOT}/benchmarks/adc700/verify.py" \
  --input "${REPORT}" --raw "${RAW}" --device-inventory "${INVENTORY}" \
  --output "${VERIFIED_REPORT}" \
  --baseline-revision "${BASELINE_SHA}" --candidate-revision "${CANDIDATE_SHA}" \
  --source-root "${CAMPAIGN_ROOT}" --wheel-proof "${WHEEL_PROOF}" \
  --archive-receipt "${CANDIDATE_ARCHIVE_RECEIPT}" --archive-helper "${ARCHIVE_HELPER}"

cp "${RAW}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-measurements.jsonl"
cp "${INVENTORY}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-devices.txt"
cp "${WHEEL_PROOF}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-installed-wheel.json"
cp "${TOOLCHAIN_PROBE}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-cmake-target-contract.json"
cp "${TOOLCHAIN_RECEIPT}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-toolchain.json"
cp "${BASELINE_ARCHIVE_RECEIPT}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-baseline-archive.json"
cp "${CANDIDATE_ARCHIVE_RECEIPT}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-candidate-archive.json"
cp "${ARCHIVE_HELPER}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-archive-receipt.py"
cp "${REPORT}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-comparison-report.json"
cp "${VERIFIED_REPORT}" "${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-report.json"
echo "ADC700_REPORT=${RESULTS_DIR}/adc700-${SLURM_JOB_ID}-report.json"
