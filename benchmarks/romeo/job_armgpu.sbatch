#!/usr/bin/env bash
#SBATCH --job-name=pops-bench
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=pops-bench-%j.out
#SBATCH --error=pops-bench-%j.err

set -euo pipefail

module load cuda/12.6
romeo_load_armgpu_env
spack load openmpi +cuda

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${POPS_BENCH_SOURCE_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${POPS_BENCH_WORK_DIR:-/scratch_p/${USER}/${SLURM_JOB_ID}}"
BUILD_ROOT="${POPS_BENCH_BUILD_DIR:-${WORK_ROOT}/build}"
RESULTS_DIR="${POPS_BENCH_RESULTS_DIR:-${HOME}/pops-benchmark-results}"
KOKKOS_ROOT="${POPS_KOKKOS_ROOT:-${Kokkos_ROOT:-${HOME}/pops_gpu_p1/kinstall}}"
NVCC_WRAPPER="${POPS_NVCC_WRAPPER:-${KOKKOS_ROOT}/bin/nvcc_wrapper}"
RESULT_FILE="${RESULTS_DIR}/pops-benchmark-${SLURM_JOB_ID}.jsonl"

test -x "${NVCC_WRAPPER}"
mkdir -p "${WORK_ROOT}/tmp" "${BUILD_ROOT}" "${RESULTS_DIR}"
export TMPDIR="${WORK_ROOT}/tmp"
cd "${WORK_ROOT}"

cmake -S "${REPO_ROOT}/benchmarks" -B "${BUILD_ROOT}" \
  -DPOPS_BENCH_POPS_SOURCE_DIR="${REPO_ROOT}" \
  -DPOPS_BENCH_ENABLE_MPI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
  -DKokkos_ROOT="${KOKKOS_ROOT}"
cmake --build "${BUILD_ROOT}" --target pops_benchmark --parallel "${SLURM_CPUS_PER_TASK}"

srun --kill-on-bad-exit=1 --ntasks="${SLURM_NTASKS}" --gpus-per-task=1 \
  "${BUILD_ROOT}/bin/pops_benchmark" \
  --case=all \
  --warmups=2 \
  --repetitions=7 \
  --arith-n=2048 \
  --arith-tile=128 \
  --arith-components=4 \
  --krylov-n=256 \
  --krylov-tile=64 \
  --krylov-max-iters=400 \
  --krylov-rel-tol=1e-9 \
  --krylov-abs-tol=0 \
  --output="${RESULT_FILE}"
