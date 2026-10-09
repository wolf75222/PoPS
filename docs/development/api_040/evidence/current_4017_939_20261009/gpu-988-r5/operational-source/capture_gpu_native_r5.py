"""Future post-build physical GPU admission. Never use a local CPU Native hash."""
import hashlib,importlib.util,json,os,re,shlex,subprocess,sys
from pathlib import Path
from hardware_receipt import capture_hardware
from native_install_rpath_r5 import authenticate_native_install,authenticate_loader_dependencies
root=Path(os.environ['FUTURE_GPU_ROOT']).resolve()
if not(root/'plan.json').is_file():raise SystemExit('Root reviewed materialization/newSource/baseline/GO pending')
root=root;source=root/'source';plan=json.loads((root/'plan.json').read_text())
if plan.get('dispatch_admission',{}).get('authorized')is not True:raise SystemExit('No dispatch/native execution admission: V4 final Source/baseline/RootGO admission pending')
assert 'PYTHONPATH' not in os.environ and 'PYTHONOPTIMIZE' not in os.environ
assert Path(sys.prefix).resolve()==root/'envs/pops_final_cuda_dim2'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(name,value):
 p=root/'results'/name
 with p.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 return sha(p)
hardware=capture_hardware();write('gpu-build-hardware-r5.json',hardware)
# Retain exact official installed/source/header/origin/doctor receipt before any model execution.
driver=source/'docs/development/api_040/run_installed_checks.py'
spec=importlib.util.spec_from_file_location('gpu_installed_identity',driver);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.argv=[str(driver),'--identity-only','--output',str(root/'results/gpu-native-identity-r5')];assert module.main()==0
identity=json.loads((root/'results/gpu-native-identity-r5/identity.json').read_text())
assert identity['source_commit']==plan['source_freeze']['commit']
assert identity['source_files_sha256']==plan['source_freeze']['source_file_manifest_sha256']
import pops
from pops._native_selector import select_native_dimension
from pops.runtime._platform_manifest import native_runtime_backend_for_route,native_device_resource
native=select_native_dimension(2);world=native.mpi_world();assert world.rank==0 and world.size==1
import subprocess
ldd_result=subprocess.run(['ldd',str(native.__file__)],text=True,capture_output=True)
write('r5-installed-native-ldd.json',{'argv':ldd_result.args,'exit':ldd_result.returncode,'stdout':ldd_result.stdout,'stderr':ldd_result.stderr,'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH'),'scope':'Actual allocation loader dependencies, no repair or invented bridges'})
assert ldd_result.returncode==0 and 'not found'not in ldd_result.stdout
admission=json.loads((root/'r5-admission.json').read_text())
assert admission['Source']==plan['source_freeze']['commit'] and admission['SDK']==plan['source_freeze']['header_signature']
match=re.search(r'libcudart\.so\.12[^\n]*=>\s+(\S+)',ldd_result.stdout)
assert match and str(Path(match.group(1)).resolve())==admission['cuda_cudart']['resolved']
assert sha(Path(match.group(1)).resolve())==admission['cuda_cudart']['sha256']

runtime=native.runtime_environment_report();assert runtime['dimension']==2 and runtime['mpi_active'] and runtime['mpi_ranks']==1
assert runtime['kokkos_backend'].lower()=='cuda' and runtime['field_memory_space']=='managed'
profile_path=root/'results/kokkos-unified-profile-manifest-r5.json'
profile=json.loads(profile_path.read_text());assert profile['prefix']==str(root/'kokkos-unified-install')
contract=dict(native.__kokkos_contract__);assert len(contract['abi_sha256'])==64
assert any(Path(include).resolve().is_relative_to(root/'kokkos-unified-install')for include in contract['include_dirs'])
for include in contract['include_dirs']:
 resolved=Path(include).resolve()
 assert any(resolved.is_relative_to(prefix)for prefix in (root/'kokkos-unified-install',Path('/apps/2025/manual_install/cuda_eviden/12.6'))), 'Kokkos includes must be newownprefix or authenticated CUDAtoolkit'

for path,expected in zip(contract['header_paths'],contract['header_sha256']):
 assert Path(path).resolve().is_relative_to(root/'kokkos-unified-install') and sha(path)==expected
write('gpu-native-kokkos-contract-r5.json',contract)
abi=native.abi_key();assert plan['source_freeze']['header_signature'] in abi
assert sha(native.__file__)==identity['native_sha256']
# Authenticate the entire exact native TU list from final CMake source, not a partial Ninja progress number.
required=[]
for cmake,base,pattern in [('src/CMakeLists.txt',source/'src',r'set\(POPS_RUNTIME_[A-Z_]+_SOURCES(.*?)\)'),('python/CMakeLists.txt',source/'python',r'set\(POPS_MODULE_BINDING_SOURCES(.*?)\)')]:
 for group in re.findall(pattern,(source/cmake).read_text(),re.S):
  required += [(base/token).resolve()for token in group.split()if token.endswith('.cpp')]
assert len(required)==24 and sum('/python/bindings/'in str(p)for p in required)==7
commands_files=list((source/'build').glob('**/compile_commands.json'));assert len(commands_files)==1,commands_files
build=commands_files[0].parent;commands=json.loads(commands_files[0].read_text())
ninja=(build/'.ninja_log').read_text().splitlines();logged={line.split('\t')[3]:line for line in ninja if not line.startswith('#') and len(line.split('\t'))==5}
objects=[]
linked=list(build.glob('**/_pops*.so'));assert len(linked)==1
wheels=list((root/'wheels').glob('pops-*.whl'));assert len(wheels)==1
install_proof=authenticate_native_install(linked[0],native.__file__,wheels[0],build,root,admission)
assert install_proof['installed_sha256']==identity['native_sha256']
linked_ldd=subprocess.run(['ldd',str(linked[0])],text=True,capture_output=True)
assert linked_ldd.returncode==0
install_proof['loader_dependencies']=authenticate_loader_dependencies(linked_ldd.stdout,ldd_result.stdout,install_proof['needed_names'])
install_proof['linked_ldd']={'argv':linked_ldd.args,'exit':linked_ldd.returncode,'stdout':linked_ldd.stdout,'stderr':linked_ldd.stderr}
install_proof_hash=write('gpu-native-install-proof-r5.json',install_proof)
linked_rel=str(linked[0].relative_to(build));assert linked_rel in logged
for file in required:
 matches=[c for c in commands if (Path(c['directory'])/c['file']).resolve()==file];assert len(matches)==1,str(file)
 c=matches[0];args=c.get('arguments') or shlex.split(c['command']);assert '-O3' in args and '-O0' not in args and '--use_fast_math' not in args and '-ffast-math' not in args
 assert any(str(root/'strict_nvcc_wrapper.sh')==arg for arg in args),'Exact new strict NVCC wrapper must compile every TU'
 assert any(plan['source_freeze']['header_signature']in arg for arg in args),'Exact SDK must be baked in every TU'
 assert any('sm_90'in arg or 'compute_90'in arg for arg in args),'Actual SM90 compiler graph required'
 output=c.get('output') or args[args.index('-o')+1];obj=(Path(c['directory'])/output).resolve();assert obj.is_file() and obj.stat().st_size>0
 rel=str(obj.relative_to(build));assert rel in logged,'Successful physical Ninja output missing: '+rel
 objects.append({'source':str(file.relative_to(source)),'source_sha256':sha(file),'object':str(obj),'bytes':obj.stat().st_size,'sha256':sha(obj),'command':args,'ninja_success_record':logged[rel]})
object_hash=write('gpu-native-object-manifest-r5.json',{'Source':plan['source_freeze']['commit'],'SDK':plan['source_freeze']['header_signature'],'runtime_count':17,'binding_count':7,'compile_commands_sha256':sha(commands_files[0]),'ninja_log_sha256':sha(build/'.ninja_log'),'objects':objects,'linked_native_path':str(linked[0]),'linked_native_sha256':sha(linked[0]),'link_ninja_success_record':logged[linked_rel]})
wheels=list((root/'wheels').glob('pops-*.whl'));assert len(wheels)==1
manifests={};manifest_hashes={}
for target in sorted({p['native_route_target']for p in plan['runtime']['phases']}):
 backend=native_runtime_backend_for_route('production',target,'MPI_COMM_WORLD');resource=native_device_resource(backend)
 assert resource.device_identity not in ('host','cpu') and resource.execution_backend.lower()=='cuda'
 manifests[target]=backend.to_data();manifest_hashes[target]=canon(manifests[target])
write('gpu-native-runtime-manifests-r5.json',manifests)
assert sha(linked[0])==install_proof['linked_sha256'] and sha(native.__file__)==install_proof['installed_sha256'] and sha(wheels[0])==install_proof['wheel_sha256']
actual={'linked_native_sha256':install_proof['linked_sha256'],'native_install_proof_sha256':install_proof_hash,'actual_source_SDK':plan['source_freeze']['header_signature'],'installed_sha256':sha(native.__file__),'installed_path':str(native.__file__),'abi_key':abi,'wheel_sha256':sha(wheels[0]),'wheel_path':str(wheels[0]),'object_manifest_sha256':object_hash,'runtime_backend_manifest_sha256':manifest_hashes,'Kokkos_manifest_sha256':sha(profile_path),'managed_pointer_launch_probe_sha256':sha(root/'results/managed-pointer-launch-probe-r5.json'),'Kokkos_abi_sha256':contract['abi_sha256']}
receipt={'status':'GPU_NATIVE_PHYSICAL_ADMISSION_PASS','Source':plan['source_freeze'],'Native':actual,'hardware':hardware,'runtime':runtime,'Kokkos_contract':contract,'managed_profile':profile,'production_cuda_kernel_trace':'pending actual runtime profile; direct allocation/launch microprobe is not PDE kernel trace','installed_identity':identity,'compiler_wrapper_sha256':sha(root/'strict_nvcc_wrapper.sh'),'compiler_argument_records':[str(p)for p in sorted((root/'compiler-records').glob('*.json'))],'scope':'Full installed CUDA Native identity and24physical O3 objects; numerical/model qualification still pending'}
receipt_hash=write('gpu-native-admission-r5.json',receipt)
plan['native_freeze'].update(actual);plan['status']='SOURCE_FROZEN_GPU_NATIVE_PHYSICALLY_ADMITTED_NUMERICS_PENDING';plan['gpu_native_admission_sha256']=receipt_hash
(root/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
