set -euo pipefail
set +u
source /Users/romaindespoulain/miniforge3/etc/profile.d/conda.sh
conda activate /Users/romaindespoulain/miniforge3/envs/pops
set -u
unset PYTHONPATH PYTHONOPTIMIZE
export FI_PROVIDER=tcp OMP_NUM_THREADS=2 OMP_PROC_BIND=false
ctest --test-dir /Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009/build --output-on-failure --no-tests=error -j 1 -L 'cpp-target:(test_amr_tensor_fac_provider|test_amr_program_field_publication|test_field_nullspace)$' --output-junit /Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009/remaining-world1-e27d/ctest.xml
