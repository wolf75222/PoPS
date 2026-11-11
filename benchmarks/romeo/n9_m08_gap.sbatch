#!/usr/bin/env bash
# SV-T1 (spec 0.3 §7.7, N9, ROM-90): M08 Poisson, Program CG against native CG, A B B A.
# CPU:  benchmarks/romeo/submit_n9_m08_gap.sh                       (x64cpu, one rank)
# GPU:  benchmarks/romeo/submit_n9_m08_gap.sh --constraint=armgpu --gpus-per-task=1
# One rank only: a Uniform Cartesian layout lowers to one box, so a second rank would own no cell
# (benchmarks/n9_m08/README.md). The sizes are the per-rank sizes of SV-T1, 256² to 4096².
#SBATCH --job-name=pops-n9-m08
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=pops-n9-m08-%j.out
#SBATCH --error=pops-n9-m08-%j.err

set -euo pipefail

ARCH="$(uname -m)"
if [[ "${ARCH}" == "aarch64" ]]; then
  romeo_load_armgpu_env
  module load cuda/12.6
  spack load openmpi +cuda
else
  romeo_load_x64cpu_env
  spack load openmpi
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${POPS_N9_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CANDIDATE_SHA="$(git -C "${REPO_ROOT}" rev-parse "${POPS_N9_REF:-HEAD}^{commit}")"
WORK_ROOT="${POPS_N9_WORK_ROOT:-/scratch_p/${USER}/${SLURM_JOB_ID}/pops-n9-m08}"
RESULTS_DIR="${POPS_N9_RESULTS_DIR:-${HOME}/pops-benchmark-results/n9-m08}"
SIZES="${POPS_N9_SIZES:-256 512 1024 2048 4096}"
BLOCKS="${POPS_N9_ABBA_BLOCKS:-3}"
STEPS="${POPS_N9_STEPS:-2}"
case "${WORK_ROOT}" in
  /scratch_p/"${USER}"/*/pops-n9-m08) ;;
  *) echo "refusing unsafe POPS_N9_WORK_ROOT: ${WORK_ROOT}" >&2; exit 3 ;;
esac
test "${SLURM_NTASKS:?}" -eq 1

cmake -E remove_directory "${WORK_ROOT}"
cmake -E make_directory "${WORK_ROOT}/source" "${WORK_ROOT}/wheelhouse" "${RESULTS_DIR}"
git -C "${REPO_ROOT}" archive "${CANDIDATE_SHA}" | tar -xf - -C "${WORK_ROOT}/source"

source "${WORK_ROOT}/source/scripts/conda_runtime.sh"
pops_load_conda
conda activate "${POPS_ENV_NAME:-pops}"
if [[ "${ARCH}" == "aarch64" ]]; then
  KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-${Kokkos_ROOT:-${HOME}/adc_gpu_p1/kinstall}}"
  export CXX="${POPS_NVCC_WRAPPER:-${KOKKOS_ROOT}/bin/nvcc_wrapper}"
  export Kokkos_ROOT="${KOKKOS_ROOT}" POPS_KOKKOS_ROOT="${KOKKOS_ROOT}"
fi
bash "${WORK_ROOT}/source/scripts/build_python.sh" --mpi --wheel-dir "${WORK_ROOT}/wheelhouse"
WHEELS=("${WORK_ROOT}"/wheelhouse/pops-*.whl)
test "${#WHEELS[@]}" -eq 1
python -m pip install --force-reinstall --no-deps "${WHEELS[0]}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OMP_PROC_BIND=spread OMP_PLACES=threads
cd "${WORK_ROOT}/source"
for size in ${SIZES}; do
  out="${RESULTS_DIR}/n9-m08-${SLURM_JOB_ID}-${ARCH}-${size}x${size}.json"
  srun --kill-on-bad-exit=1 --ntasks=1 env PYTHONNOUSERSITE=1 \
    python benchmarks/n9_m08/m08_gap.py "${size}x${size}" --rtol 1e-10 \
    --steps "${STEPS}" --blocks "${BLOCKS}" --out "${out}"
done
echo "results: ${RESULTS_DIR}/n9-m08-${SLURM_JOB_ID}-*.json (revision ${CANDIDATE_SHA})"
