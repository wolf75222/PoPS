set -euo pipefail
set +u
source /Users/romaindespoulain/miniforge3/etc/profile.d/conda.sh
conda activate /Users/romaindespoulain/miniforge3/envs/pops
set -u
unset PYTHONPATH PYTHONOPTIMIZE
export POPS_ENV_NAME=pops POPS_NATIVE_DIM=2
cd /Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004
env -u PYTHONPATH -u PYTHONOPTIMIZE bash scripts/build_python.sh --dim 2 --mpi --wheel-dir /Users/romaindespoulain/dev/tmp/PoPS-native-reviewed-integration-4017d29d-r2-20261009/wheel -- -C build.verbose=true -C cmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
