set -euo pipefail
set +u
source /Users/romaindespoulain/miniforge3/etc/profile.d/conda.sh
conda activate /Users/romaindespoulain/miniforge3/envs/pops
set -u
unset PYTHONPATH PYTHONOPTIMIZE
export CC=/usr/bin/clang CXX=/usr/bin/clang++ MPICH_CC=/usr/bin/clang MPICH_CXX=/usr/bin/clang++ FI_PROVIDER=tcp OMP_NUM_THREADS=2 OMP_PROC_BIND=false
cd /Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004

# configure
cmake --preset mpi -B /Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009/build -DPOPS_BUILD_PYTHON=OFF -DPOPS_USE_KOKKOS=ON -DKokkos_ROOT=/Users/romaindespoulain/miniforge3/envs/pops -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DPOPS_NATIVE_DIM=2

# build
cmake --build /Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009/build --parallel 2 --target test_tensor_fac_conservative_interface test_amr_tensor_fac_provider test_mpi_composite_fac_partitioned_nd test_amr_program_field_publication test_field_nullspace test_mpi_field_nullspace_preflight

# ctest
ctest --test-dir /Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009/build --output-on-failure --no-tests=error -j 1 -R '^(test_tensor_fac_conservative_interface(_np2)?|test_amr_tensor_fac_provider|test_mpi_composite_fac_partitioned_nd_np2|test_amr_program_field_publication|test_field_nullspace|test_mpi_field_nullspace_preflight_np2)$' --output-junit /Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009/ctest.xml
