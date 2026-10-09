"""Future authorized phase runner. Refuses NULL freezes; never run during preparation."""
import contextlib,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET

root=Path(os.environ['FUTURE_GPU_ROOT']).resolve()
if not(root/'plan.json').is_file():raise SystemExit('Root reviewed materialization/newSource/baseline/GO pending')
plan=json.loads((root/'plan.json').read_text())
if plan.get('dispatch_admission',{}).get('authorized')is not True:raise SystemExit('No dispatch/native execution admission: V4 final Source/baseline/RootGO admission pending')
source_freeze=plan['source_freeze'];native_freeze=plan['native_freeze']
required=(source_freeze.get('commit'),source_freeze.get('header_signature'),
 source_freeze.get('core_fingerprint'),native_freeze.get('installed_sha256'),
 native_freeze.get('wheel_sha256'),native_freeze.get('object_manifest_sha256'),
 native_freeze.get('abi_key'), source_freeze.get('source_file_manifest_sha256'),
 source_freeze.get('test_fixtures_sha256'), native_freeze.get('actual_source_SDK'),
 native_freeze.get('runtime_backend_manifest_sha256'),
 plan['runtime'].get('frozen_selector_manifest_sha256'),native_freeze.get('Kokkos_manifest_sha256'),native_freeze.get('managed_pointer_launch_probe_sha256'))
if any(v is None for v in required):raise SystemExit('Final Source/Native freeze pending; refuse imports and execution')
if 'PYTHONPATH' in os.environ or 'PYTHONOPTIMIZE' in os.environ:raise SystemExit('Unsanitized Python environment')
phase_id=os.environ['FUTURE_GPU_PHASE']
phase=next(v for v in plan['runtime']['phases'] if v['id']==phase_id)
previous=phase.get('requires_previous_phase')
if previous:
 previous_phase=next(v for v in plan['runtime']['phases'] if v['id']==previous)
 for previous_rank in range(previous_phase['world']):
  prior=json.loads((root/'results/r5'/previous/('rank%d'%previous_rank)/'result.json').read_text())
  assert prior['status']=='passed', 'Predecessor phase did not pass'
rank=int(os.environ['PMI_RANK']);assert 0<=rank<phase['world']
source=root/'source';os.chdir(source)
out=root/'results/r5'/phase_id/('rank%d'%rank);out.mkdir(mode=0o700,parents=True,exist_ok=False)
args=['-x','-ra','--strict-markers','--strict-config','-v','--tb=short',
 '-o','pythonpath=','-o','cache_dir='+str(root/'cache/pytest/r5'/phase_id/('rank%d'%rank)),
 '--junitxml='+str(out/'pytest.xml'),'--basetemp='+str(out/'pytest-tmp'),*phase['selectors']]
environment_keys=('FI_PROVIDER','FI_TCP_IFACE','OMP_NUM_THREADS','OMP_PROC_BIND','POPS_THREADS',
 'POPS_CXX','POPS_KOKKOS_CXX','POPS_KOKKOS_USE_NVCC_WRAPPER','CUDA_VISIBLE_DEVICES',
 'CUDA_DEVICE_ORDER','PYTEST_ADDOPTS','PYTHONDONTWRITEBYTECODE','PMI_RANK','PMI_SIZE',
 'SLURM_JOB_ID','SLURM_STEP_ID','SLURM_LOCALID')
before={'scope':'actual future before-process-run environment, selectors and freezes; not numerical acceptance',
 'phase':phase_id,'rank':rank,'argv':args,'source_freeze':source_freeze,'native_freeze':native_freeze,
 'environment':{k:os.environ.get(k) for k in environment_keys},'PYTHONPATH_present':'PYTHONPATH' in os.environ,
 'PYTHONOPTIMIZE_present':'PYTHONOPTIMIZE' in os.environ}
(out/'before-run.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
from hardware_receipt import capture_hardware
hardware=capture_hardware()
(out/'hardware-before-native.json').write_text(json.dumps(hardware,sort_keys=True,indent=2)+'\n')
driver=source/'docs/development/api_040/run_installed_checks.py'
spec=importlib.util.spec_from_file_location('installed_identity_driver',driver)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.argv=[str(driver),'--identity-only','--output',str(out/'identity')];assert module.main()==0
import pops
from pops._native_selector import select_native_dimension
from pops.runtime._platform_manifest import native_runtime_backend_for_route,native_device_resource
native=select_native_dimension(2);world=native.mpi_world()
assert world.rank==rank and world.size==phase['world']
assert Path(pops.__file__).resolve().is_relative_to(root/'envs')
assert Path(native.__file__).resolve().is_relative_to(root/'envs')
native_sha=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
assert native_sha==native_freeze['installed_sha256'] and native.abi_key()==native_freeze['abi_key']
assert source_freeze['header_signature'] in native.abi_key()
backend=native_runtime_backend_for_route('production',phase['native_route_target'],'MPI_COMM_WORLD')
resource=native_device_resource(backend)
manifest_sha=hashlib.sha256(json.dumps(backend.to_data(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert manifest_sha==native_freeze['runtime_backend_manifest_sha256'][phase['native_route_target']]
runtime=native.runtime_environment_report()
assert runtime['dimension']==2 and runtime['mpi_active'] and runtime['mpi_rank']==rank and runtime['mpi_ranks']==phase['world']
assert runtime['kokkos_backend'].lower()=='cuda' and runtime['field_memory_space']=='managed'
assert resource.device_identity not in ('host','cpu')
facts={'rank':rank,'ranks':world.size,'pid':os.getpid(),'package_file':pops.__file__,
 'native_file':native.__file__,'native_sha256':native_sha,'abi':native.abi_key(),
 'runtime_environment':runtime,'backend_manifest':backend.to_data(),
 'device':resource.device_identity,'memory_space':resource.memory_space_identity,
 'execution_backend':resource.execution_backend,'shared_space':resource.shared_space_identity,
 'stream_identity':resource.stream_identity,'inprocess_identity_then_pytest':True,
 'hardware':hardware,'GPU_profile':'One allocated GPU may be shared by both ranks; actual device/stream identities recorded. No separate-device census inferred.'}
(out/'before-pytest-rank-identity.json').write_text(json.dumps(facts,sort_keys=True,indent=2)+'\n')
import pytest
start=time.monotonic()
with (out/'pytest.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
 code=pytest.main(args)
xml=ET.parse(out/'pytest.xml').getroot()
counts={k:sum(int(s.get(k,0)) for s in xml.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
expected={'tests':phase['expected_tests_per_rank'],'failures':0,'errors':0,'skipped':0}
result={'phase':phase_id,'rank':rank,'ranks':world.size,'pid':os.getpid(),'argv':args,
 'exit':int(code),'counts':counts,'expected_counts':expected,'seconds':time.monotonic()-start,
 'native_sha256':native_sha,'inprocess_identity_then_pytest':True,
 'status':'passed' if code==0 and counts==expected else 'failed',
 'scope':'Native execution/JUnit only. Independent saved-array/CP/JIT reception still required. Failures are not automatically Unsupported.'}
(out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
raise SystemExit(0 if result['status']=='passed' else 1)
