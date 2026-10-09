"""Future actual Kokkos/config/allocation/launch reception; never imports old PoPS."""
import hashlib,json,os
from pathlib import Path
root=Path(os.environ['FUTURE_GPU_ROOT']);prefix=root/'kokkos-unified-install'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
config=prefix/'include/KokkosCore_config.h';text=config.read_text()
for macro in ('KOKKOS_ENABLE_CUDA','KOKKOS_ARCH_HOPPER90','KOKKOS_ENABLE_IMPL_CUDA_UNIFIED_MEMORY'):assert '#define '+macro in text
cache=(root/'kokkos-unified-build/CMakeCache.txt').read_text();assert 'Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY:BOOL=ON'in cache
probe_path=root/'results/managed-pointer-launch-probe-r5.json';probe=json.loads(probe_path.read_text())
assert probe['status']==0 and probe['execution_backend']=='Cuda'and probe['field_memory_space_type']=='CudaSpace'
assert probe['HostSpace_accessible']and probe['Cuda_accessible']and probe['pointer_type']==probe['cudaMemoryTypeManaged']
assert probe['launch_status']==0 and probe['stream_sync_status']==0 and probe['stamp']==8192
assert probe['compute_major']==9 and probe['compute_minor']==0 and probe['CUDA_runtime_header_version']>=12020 and probe['CUDA_runtime_library_version']>=12020
files={str(p.relative_to(prefix)): {'bytes':p.stat().st_size,'sha256':sha(p)}for p in prefix.rglob('*')if p.is_file()}
manifest={'version':'5.2.1','prefix':str(prefix),'files':files,'config_sha256':sha(config),'cache_sha256':sha(root/'kokkos-unified-build/CMakeCache.txt'),'probe_source_sha256':sha(root/'managed_cuda_probe.cpp'),'probe_binary_sha256':sha(root/'profile-probe-build/managed_cuda_probe'),'probe_receipt_sha256':sha(probe_path),'probe':probe,'scope':'Actual new ownKokkos library/config and directCUDA allocation/launch probe. Production PDE kernels/numerical qualification remain pending.'}
(root/'results/kokkos-unified-profile-manifest-r5.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
