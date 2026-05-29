#!/usr/bin/env bash
# Preuve que le BACKEND est inclus dans la bibliotheque, pas rajoute par cible :
# on configure l'arbre ENTIER avec -DADC_USE_KOKKOS=ON (nvcc_wrapper) et on compile
# `adc_solver` (libadc : les 3 facades src/) PLUS l'exemple integre. Tout herite de
# Kokkos via la cible d'interface `adc`. Soumettre depuis la racine sur ROMEO armgpu.
#
#SBATCH --account=r250127
#SBATCH --constraint=armgpu
#SBATCH --partition=instant
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --job-name=adc_libgpu
#SBATCH --output=libadc_gpu.%J.out
#SBATCH --error=libadc_gpu.%J.err

romeo_load_armgpu_env
spack load cuda@12.6.3
spack load /vziqyqt          # gcc@11.4.1 (armgpu), pin par hash
spack load cmake 2>/dev/null || spack load /phhwuh3

cd "$SLURM_SUBMIT_DIR"
K=/scratch_p/rmdraux/kokkos-install
rm -rf build-gpu
echo "===== CONFIGURE (arbre entier, ADC_USE_KOKKOS=ON) ====="
cmake -S . -B build-gpu -DADC_USE_KOKKOS=ON \
  -DCMAKE_CXX_COMPILER="$K/bin/nvcc_wrapper" -DKokkos_ROOT="$K" \
  -DADC_USE_EIGEN=OFF -DADC_BUILD_PYTHON=OFF -DADC_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -4
echo "===== BUILD libadc (adc_solver : les 3 facades src/ sous nvcc) ====="
cmake --build build-gpu --target adc_solver -j8 2>&1 | tail -20
echo "===== BUILD exemple integre (herite Kokkos de adc) ====="
cmake --build build-gpu --target two_fluid_ap_kokkos -j8 2>&1 | tail -8
echo "===== RUN GPU ====="
./build-gpu/bin/two_fluid_ap_kokkos
