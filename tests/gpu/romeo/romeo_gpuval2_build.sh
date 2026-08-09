#!/bin/bash -l
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=gpuval2
# Build + run "round 2" : validation device (Kokkos Cuda) des chemins exact-ranked.
# Compile via nvcc_wrapper (backend Cuda). Les harness qui restent portables sans Kokkos gardent
# leur oracle Serial; le GeometricMG moderne est valide directement par solution manufacturee et
# convergence spatiale sur le device. NE recompile PAS Kokkos : reutilise kinstall prebuilt.
set -euo pipefail
module load cuda/12.6
romeo_load_armgpu_env
POPS_GPU_ROOT="${POPS_GPU_ROOT:-$HOME/pops_gpu_p1}"
cd "$POPS_GPU_ROOT"
echo "noeud=$(hostname) arch=$(uname -m)"
NW="$PWD/kinstall/bin/nvcc_wrapper"
INC="$PWD/gpuval2_include"          # en-tetes A JOUR (rsync depuis master)
SRC="$PWD/gpuval2_src"              # les .cpp des harness + diff_bin + CMakeLists
RES="$PWD/gpuval2_results"
POPS_NATIVE_DIM="${POPS_NATIVE_DIM:-2}"
case "$POPS_NATIVE_DIM" in
  1|2|3) ;;
  *) echo "POPS_NATIVE_DIM must be 1, 2, or 3" >&2; exit 2 ;;
esac
mkdir -p "$RES"

# --- build device (Kokkos Cuda) ------------------------------------------------------------------
rm -rf gpuval2_build
cmake -S "$SRC" -B gpuval2_build -DCMAKE_CXX_COMPILER="$NW" -DKokkos_ROOT="$PWD/kinstall" \
  -DPOPS_INCLUDE="$INC" -DPOPS_NATIVE_DIM="$POPS_NATIVE_DIM" -DCMAKE_BUILD_TYPE=Release \
  > "$RES/cfg.log" 2>&1 || { echo CFG_FAIL; tail -50 "$RES/cfg.log"; exit 1; }
cmake --build gpuval2_build -j 8 > "$RES/build.log" 2>&1 \
  || { echo BUILD_FAIL; grep -iE "error" "$RES/build.log" | head -40; exit 1; }
echo GPUVAL2_BUILD_OK

# --- build the remaining header-only Serial oracle -----------------------------------------------
echo "=== build Serial oracle (g++) ==="
g++ -std=c++20 -O2 -DPOPS_NATIVE_DIM="$POPS_NATIVE_DIM" -I "$INC" \
  "$SRC/gpu_aux_validate.cpp" -o "$RES/aux_serial" \
  > "$RES/serial_aux.log" 2>&1 \
  || { echo SERIAL_AUX_FAIL; tail -30 "$RES/serial_aux.log"; exit 1; }
g++ -std=c++20 -O2 "$SRC/diff_bin.cpp" -o "$RES/diff_bin" 2>/dev/null \
  || { echo DIFF_BIN_BUILD_FAIL; exit 1; }

cd "$RES" || exit 3
# Pour chaque feature : MEME logique en exec=Cuda (srun 1 GPU) et oracle exec=Serial (g++), puis
# diff_bin -> dmax sur CHAQUE cellule (vise 0 = bit-identique). for_each_cell est ASYNC sous Cuda :
# chaque harness fait device_fence() avant la lecture hote / le dump.
echo "######## exact-ranked GeometricMG constant-scalar MMS ########"
srun --kill-on-bad-exit=1 -n 1 --gpus-per-task=1 \
  "$PWD/../gpuval2_build/gpu_epm_validate" --dump=epm_cuda

echo "######## (1) T_e via load_aux<5> ########"
srun --kill-on-bad-exit=1 -n 1 --gpus-per-task=1 \
  "$PWD/../gpuval2_build/gpu_aux_validate" --dump=aux_cuda
./aux_serial --dump=aux_serial
./diff_bin aux_cuda_te.bin aux_serial_te.bin

echo "######## (7) ordre for_each_cell -> MultiFab::scale -> reduction ########"
# No fence is inserted by the harness before scale or reduce_sum: reduce_sum is
# deliberately the first blocking observation of the CUDA stream.
srun --kill-on-bad-exit=1 -n 1 --gpus-per-task=1 \
  "$PWD/../gpuval2_build/gpu_scale_async_validate" \
  || { echo SCALE_ORDER_FAIL; exit 1; }

echo "######## (8) ordre RHS device -> PoissonFFTSolver::solve -> residu ########"
srun --kill-on-bad-exit=1 -n 1 --gpus-per-task=1 \
  "$PWD/../gpuval2_build/gpu_fft_async_validate" \
  || { echo FFT_ORDER_FAIL; exit 1; }

echo GPUVAL2_DONE
