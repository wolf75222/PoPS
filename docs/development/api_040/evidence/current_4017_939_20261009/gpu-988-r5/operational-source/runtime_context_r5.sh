#!/bin/bash
[[ -n "$FUTURE_GPU_ROOT" && -n "$SLURM_JOB_ID" ]]
source "$FUTURE_GPU_ROOT/private_runtime_context.sh"
source "$ROOT/miniforge/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/pops_final_cuda_dim2"
[[ "$CONDA_PREFIX" == "$ROOT/envs/pops_final_cuda_dim2" ]]
export Kokkos_ROOT="$ROOT/kokkos-unified-install"
export POPS_KOKKOS_ROOT="$Kokkos_ROOT"
CUDA_LIBPATH=$(/usr/bin/python3 - "$ROOT/profile-probe-build/CMakeCache.txt" <<'PY_CUDA'
import re,sys
from pathlib import Path
m=re.search(r'^CUDA_CUDART:FILEPATH=(.+)$',Path(sys.argv[1]).read_text(),re.M);assert m
p=Path(m.group(1)).resolve();assert p.is_file();print(p.parent)
PY_CUDA
)
export CUDA_LIBPATH
export LD_LIBRARY_PATH="$CUDA_LIBPATH:$CONDA_PREFIX/lib:$ROOT/kokkos-unified-install/lib:/project/r250127/rmdraux/sol61-gpu-bootstrap-20261004/compiler13/lib"
# Exact cached CUDA first; own environment next; all borrowed dependencies remain readonly.
