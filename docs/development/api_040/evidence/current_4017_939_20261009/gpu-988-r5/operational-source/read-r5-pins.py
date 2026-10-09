"""Future physical runtime admission; no ARM execution on login."""
import hashlib,json,os,re,stat
from pathlib import Path
root=Path(os.environ['FUTURE_GPU_ROOT']).resolve();a=json.loads((root/'r5-admission.json').read_text())
assert a['Root_GO']is True and os.getuid()==100267
assert root.stat().st_uid==100267 and stat.S_IMODE(root.stat().st_mode)==0o700
plan=json.loads((root/'plan.json').read_text());assert plan['source_freeze']['commit']==a['Source']and plan['source_freeze']['header_signature']==a['SDK']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for name,expected in a['required_existing_files'].items():assert sha(root/name)==expected,name
probe=root/'profile-probe-build/managed_cuda_probe';assert sha(probe)==a['probe_binary_sha256']
cache=(root/'profile-probe-build/CMakeCache.txt').read_text();match=re.search(r'^CUDA_CUDART:FILEPATH=(.+)$',cache,re.M);assert match
cuda=Path(match.group(1)).resolve();assert str(cuda)==a['cuda_cudart']['resolved']and sha(cuda)==a['cuda_cudart']['sha256']
assert 'Kokkos_ENABLE_IMPL_CUDA_UNIFIED_MEMORY:BOOL=ON'in(root/'kokkos-unified-build/CMakeCache.txt').read_text()
assert(root/'envs/pops_final_cuda_dim2').is_dir()and(root/'kokkos-unified-install').is_dir()
assert not(root/'results/official-build-entered.json').exists()and not(root/'results/official-build-entered-r4.json').exists()and not(root/'results/official-build-entered-r5.json').exists(), 'Native already entered; preNative resume refused'
scope={'Source':a['Source'],'SDK':a['SDK'],'probe_sha256':sha(probe),'cuda_cudart':str(cuda),'cuda_cudart_sha256':sha(cuda),'existing_files':a['required_existing_files'],'scope':'Actual existingenv/Kokkos/probe bytes, no rebuilt Native or scientific GPU qualification'}
with(root/'results/r5-existing-runtime-admission.json').open('x')as f:json.dump(scope,f,indent=2,sort_keys=True);f.write('\n')
print(str(cuda.parent))
