#!/usr/bin/env bash
#SBATCH --account=r250127
#SBATCH --constraint=x64cpu
#SBATCH --partition=instant
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=pops-tr01
#SBATCH --output=/scratch_p/rmdraux/pops-verify/job.%J.out
#SBATCH --error=/scratch_p/rmdraux/pops-verify/job.%J.err

set +e
source /scratch_p/rmdraux/pops-validate/env_x64cpu.sh
set -e

export POPS_KOKKOS_ROOT="${KOKKOS_CPU_ROOT:-$HOME/kokkos-x64}"
export POPS_TEST_CXX="${CXX}"
export POPS_INCLUDE=/scratch_p/rmdraux/pops-verify/include
export PYTHONPATH=/scratch_p/rmdraux/pops-verify/python:/scratch_p/rmdraux/pops-verify
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

ROOT=/scratch_p/rmdraux/pops-verify
cd "$ROOT"

# numpy matching python@3.11.9 %gcc@14.2.0 (compute nodes have no pip / no .local numpy)
spack load /3s3hqzq >/dev/null 2>&1 || true

echo "host=$(hostname) arch=$(uname -m) cpus=${SLURM_CPUS_PER_TASK}"
echo "python=$POPS_PYTHON cxx=$CXX kokkos=$POPS_KOKKOS_ROOT"
"$POPS_PYTHON" -c "import sys,numpy; print(sys.version); print('numpy', numpy.__version__, numpy.__file__)"

echo "=== TR-01 native smoke ==="
"$POPS_PYTHON" verification/machines/run_tr01_native.py
echo DONE
