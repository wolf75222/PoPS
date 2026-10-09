import json, hashlib, zipfile, struct, re, sys, subprocess, importlib.util
from pathlib import Path
import numpy as np

B=Path('/Users/romaindespoulain/dev/tmp/PoPS-native-stage-fields-cpp-reception-20261009')
R=B/'stage-fields-representative/run'
O=Path(__file__).resolve().parent
M=Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004')
ORACLE=Path('/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-fields-independent-oracle-20261009')
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def read(p):return json.loads(Path(p).read_text())
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def physical(row):assert pin(row['path'])==row,row
def head(m,n):
 if n<24:return bytes([m*32+n])
 for lim,size,tag in [(255,1,24),(65535,2,25),(4294967295,4,26),(18446744073709551615,8,27)]:
  if n<=lim:return bytes([m*32+tag])+n.to_bytes(size,'big')
 raise ValueError(n)
def cbor(x):
 if x is None:return b'\xf6'
 if x is False:return b'\xf4'
 if x is True:return b'\xf5'
 if isinstance(x,int):return head(0,x)if x>=0 else head(1,-1-x)
 if isinstance(x,str):b=x.encode();return head(3,len(b))+b
 if isinstance(x,bytes):return head(2,len(x))+x
 if isinstance(x,(list,tuple)):return head(4,len(x))+b''.join(cbor(v)for v in x)
 if isinstance(x,dict):
  rows=sorted(((cbor(k),cbor(v))for k,v in x.items()),key=lambda r:(len(r[0]),r[0]));return head(5,len(rows))+b''.join(k+v for k,v in rows)
 raise TypeError(type(x))
def identity(domain,payload):return digest(cbor(dict(protocol='pops.identity',domain=domain,schema_version=1,payload=payload)))
def cpcheck(p):
 with np.load(p,allow_pickle=False)as z:
  m=json.loads(str(z['pops_checkpoint_manifest'].item()))
  assert set(z.files)==set(m['arrays'])|{'pops_checkpoint_manifest','pops_restart_identity'}
  for k,row in m['arrays'].items():
   a=np.asarray(z[k],order='C');assert not a.dtype.hasobject
   actual=dict(dtype=a.dtype.str,shape=list(a.shape),content_sha256=digest(cbor(dict(protocol='pops.array-evidence.v1',dtype=a.dtype.str,shape=list(a.shape)))+a.tobytes()))
   assert actual==row,k
  restart=identity('restart',{k:v for k,v in m.items()if k!='restart_identity'})
  assert restart==m['restart_identity']['hexdigest']
  assert str(z['pops_restart_identity'].item())=='pops.restart.v1:sha256:'+restart
  assert m['clock']==dict(time=float(z['t']).hex(),macro_step=int(z['macro_step']))
  assert float(z['t'])==1e-4 and int(z['macro_step'])==1
  return dict(**pin(p),version=int(z['pops_checkpoint_version']),arrays=len(m['arrays']),clock=m['clock'],restart_identity=restart)
def macho(p):
 b=Path(p).read_bytes();assert struct.unpack_from('<I',b)[0]==0xfeedfacf
 n=struct.unpack_from('<I',b,16)[0];off=32;sections={};deps=[];rpaths=[];uuid=None
 for _ in range(n):
  cmd,size=struct.unpack_from('<II',b,off)
  if cmd==0x19:
   ns=struct.unpack_from('<I',b,off+64)[0]
   for j in range(ns):
    s=off+72+j*80;name=b[s:s+16].split(b'\0')[0].decode();seg=b[s+16:s+32].split(b'\0')[0].decode();length,where=struct.unpack_from('<QI',b,s+40);flags=struct.unpack_from('<I',b,s+64)[0]
    if flags&255 not in (1,12,18):sections[seg+'/'+name]=dict(bytes=length,sha256=digest(b[where:where+length]))
  if cmd in (0xc,0x80000018,0x8000001f,0x80000023):
   start=struct.unpack_from('<I',b,off+8)[0];deps.append(b[off+start:off+size].split(b'\0')[0].decode())
  if cmd==0x8000001c:
   start=struct.unpack_from('<I',b,off+8)[0];rpaths.append(b[off+start:off+size].split(b'\0')[0].decode())
  if cmd==0x1b:uuid=b[off+8:off+24].hex()
  off+=size
 return dict(sections=sections,dependencies=deps,rpaths=rpaths,uuid=uuid)

A=read(B/'native-build-admission.json');C=read(B/'core-manifest.json');E=read(R/'execution-closed.json')
assert sha(B/'native-build-admission.json')=='182f374d82334398832430a146ba9516c995dd27bd827dc76a9624c0211585fb'
assert A['Source']=='8c1f55701cab7c60b3519bd343e3b4c1f193570f'
assert A['SDK']=='087fead68988afda3c730e232f15b58ded9a07a1e4387b212c33248ac7661626'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip()==A['Source']
for name,row in C['files'].items():assert pin(M/name)['sha256']==row['sha256']and (M/name).stat().st_size==row['bytes'],name
assert digest(json.dumps(C['files'],sort_keys=True,separators=(',',':')).encode())==C['core_fingerprint']==A['core_fingerprint']
for row in [A['installed_native'],A['wheel']]+A['cache_native_outputs_after']:physical(row)
assert A['installed_native']['sha256']=='cee947c25c771f8c5a155e5f4cb990de308de14fe77e9f55c654e3b3837dee6f'
sources=read(B/'identity/source-files.json');pack=Path(E['identity']['package'])
with zipfile.ZipFile(A['wheel']['path'])as z:
 for name,d in sources.items():
  relative=name.removeprefix('python/pops/')if name.startswith('python/pops/')else name
  assert sha(M/name)==sha(pack/relative)==digest(z.read('pops/'+relative))==d,name
 # Independently authenticate every recorded installed wheel payload member, excluding distribution metadata generated at installation.
 members=[n for n in z.namelist()if n.startswith('pops/')and not n.endswith('/')]
 for name in members:assert sha(pack/name.removeprefix('pops/'))==digest(z.read(name)),name
cache=macho(A['cache_native_outputs_after'][0]['path']);installed=macho(A['installed_native']['path'])
assert cache['sections']==installed['sections']and cache['dependencies']==installed['dependencies']and cache['uuid']==installed['uuid']
objects=[];pre=read(B/'pre-build.json');before_row_discrepancies=[]
for x in A['objects']:
 physical(x['physical']);assert sha(x['source'])==x['source_sha256']
 assert digest(x['command'].encode())==x['command_sha256']
 assert x['SDK_tokens']==[A['SDK']]
 assert x['ninja_output_record']in (M/'build/cp312-cp312-macosx_26_0_arm64-dim2/.ninja_log').read_text()
 frozen_before=pre['objects'][x['physical']['path']]['physical_before']
 if x['physical_before']!=frozen_before:before_row_discrepancies.append(dict(output=x['physical']['path'],admission_physical_before=x['physical_before'],frozen_pre_build_before=frozen_before))
 assert x['byte_changed']==(frozen_before['sha256']!=x['physical']['sha256'])
 objects.append(dict(output=x['physical'],physical_before_from_frozen_pre_build=frozen_before,source=pin(x['source']),command_sha256=x['command_sha256'],byte_changed=x['byte_changed'],ninja_output_record=x['ninja_output_record']))
assert len(objects)==A['observed_CXX_output_action_count']==24 and A['qualified_reused_object_count']==0
assert read(B/'build-result.json')['exit']==0
prod=read(R.parent/'producer-result.json');assert prod['exit']==0 and sha(R.parent/'producer.log')==prod['producer_log_sha256']
invocation=read(R.parent/'invocation.json');runner=Path(invocation['argv'][1]);assert sha(runner)==invocation['runner_sha256']
assert invocation['build_admission_sha256']==sha(B/'native-build-admission.json')
case=Path('/Users/romaindespoulain/dev/tmp/PoPS-root-stage-fields-native-protocol-20261009/stage_fields_case_observed.py')
original=Path('/Users/romaindespoulain/dev/tmp/PoPS-sol61-independent-stage-fields-case-20261009/stage_fields_case.py')
assert sha(case)==E['case_source_sha256']=='bdaee56ac52b4391ed7b5402a95c00524b5ae9a4b48dd47b44fef73799bfad42'
assert sha(original)=='7fd040d65c44f35c174188a091c50214fb70d24da10090d689c6f999df041dae'
before=read(R/'identity-before.json');after=read(R/'identity-after.json')
assert {k:v for k,v in before.items()if k!='runtime_environment'}=={k:v for k,v in after.items()if k!='runtime_environment'}
for k in ['dimension','mpi_rank','mpi_ranks','kokkos_backend','kokkos_device','field_memory_space','precision','real_bytes']:
 assert before['runtime_environment'][k]==after['runtime_environment'][k]
assert before['runtime_environment']['mpi_ranks']==1
assert before['Native']==A['installed_native']['sha256']and before['SDK']==A['SDK']and before['Source']==A['Source']
assert before['build_result_sha256']==sha(B/'build-result.json')and before['build_admission_sha256']==sha(B/'native-build-admission.json')
provenance=[]
for name,row in E['retained_cpp_and_provenance'].items():assert pin(R/name)['sha256']==row['sha256']and (R/name).stat().st_size==row['bytes']
for name,row in E['binaries'].items():
 physical(row);p=Path(row['path']);side=read(str(p)+'.pops-artifact.json')
 expected=identity('binary',dict(algorithm='sha256',content_digest=bytes.fromhex(sha(p)),size=p.stat().st_size))
 assert side['binary_identity']=='pops.binary.v1:sha256:'+expected
 assert A['SDK'].encode()in p.read_bytes()
 compilepath=Path(str(p)+'.pops-model-compile.json')
 if compilepath.exists():
  comp=read(compilepath);cpp=Path(str(p)+'.pops-model.cpp')
  assert sha(cpp)==comp['source_sha256']and sha(p)==comp['binary_sha256']and comp['header_signature']==A['SDK']
  assert side['model_compile_provenance_sha256']==sha(compilepath)
  assert sha(comp['compiler_file'])==comp['compiler_sha256']
  assert '-fno-fast-math'in comp['command']and '-ffp-contract=off'in comp['command']
 provenance.append(dict(name=name,DSO=row,sidecar=pin(str(p)+'.pops-artifact.json'),actual_model_companion=compilepath.exists(),complete_frontend_graph=False))
P=read(R/'actual-program.json');F=read(R/'actual-field-plans.json')['two-screened-potentials'];S=read(R/'actual-accepted-ssp.json')
assert S['A']==[['0','0','0'],['1','0','0'],['1/4','1/4','0']]and S['b']==['1/6','1/6','2/3']and S['c']==['0','1','1/2']and S['coefficient']=='1'
solves=[x for x in P['nodes']if x['op']=='solve_linear'];assert [x['id']for x in solves]==F['solve_node_ids']==[8,24,40]
for x in solves:
 a=x['attrs'];assert a['max_iter']==2000 and a['ncomp']==2 and not a['has_guess']
 assert float.fromhex(a['abs_tol']['value'])==1e-14 and float.fromhex(a['tol']['value'])==1e-12
assert len(P['histories'])==9 and all(x['lag']==1 for x in P['histories'])
cpp=(R/'generated/explicit-SSPRK3-two-fields.cpp').read_text()
assert cpp.count('ctx.publish_field_components(')==3
assert re.findall(r'ctx.set_stage_time\((\d+), (\d+)\);\s*\*operator_dt',cpp)==[('0','1'),('1','1'),('1','2')]
assert re.findall(r'const auto& field_source_0 = (\w+);',cpp)==['u0','u0','u17','u17','u33','u33']
assert re.findall(r'const auto& field_source_1 = (\w+);',cpp)==['u1']*6
assert E['time']==1e-4 and E['macro_step']==E['accepted_steps']==1
mathcheck=read(O/'strict-array-math.json')
assert mathcheck['oracle_ready_sha256']=='28ca67fc888d78ae3f57cc417759b4b9624692c9499d5d1ecacb1fc793b6b81e'
assert sha(ORACLE/'representative-oracle-ready.json')==mathcheck['oracle_ready_sha256']
assert sha(ORACLE/'pre-Native-tolerances.json')==mathcheck['budget_sha256']
assert sha(R/'actual-states-and-fields.npz')==mathcheck['actual_npz_sha256']
ready=read(ORACLE/'representative-oracle-ready.json')
for p,h in ready['evidence_sha256'].items():assert sha(p)==h,p
spec=importlib.util.spec_from_file_location('independent_stage_fields_oracle',ORACLE/'independent_reference.py')
oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)
actual=np.load(R/'actual-states-and-fields.npz',allow_pickle=False);bind=np.load(R/'actual-bind-inputs.npz',allow_pickle=False)
mapped=dict(q0=actual['q_stage0'],q1=actual['q_stage1'],q2=actual['q_stage2'],qfinal=actual['q_final'],d_initial=bind['donor'],d_final=actual['donor_final'],resident_phi=actual['phi_resident'][None,:,:],resident_psi=actual['psi_resident'][None,:,:])
for j in range(3):
 for name in ['phi','psi']:mapped[name+str(j)]=actual[name+'_stage'+str(j)]
assert oracle.check_arrays(mapped)==mathcheck['comparisons']
assert actual['q_stage0'].tobytes()==bind['receiver'].tobytes()
assert actual['donor_final'].tobytes()==bind['donor'].tobytes()
residuals=[]
for j in range(3):
 for name,sigma,factor in [('phi',3,1),('psi',5,-.75)]:
  v=mapped[name+str(j)];value=float(np.max(np.abs(-oracle.laplacian_scalar(v,oracle.Physics())+sigma*v-(mapped['q'+str(j)]+factor*mapped['d_initial']))))
  residuals.append(dict(stage=j,field=name,max_physical_discrete_residual=value,bound=read(ORACLE/'pre-Native-tolerances.json')['independent_residual_bound']))
controls=[]
def refusal(name,mutator):
 trial={k:v.copy()for k,v in mapped.items()};mutator(trial)
 try:oracle.check_arrays(trial)
 except (AssertionError,KeyError)as ex:controls.append(dict(name=name,refused=True,exception_type=type(ex).__name__,reason=str(ex)));return
 raise AssertionError('Independent receiver accepted deliberate bad capture: '+name)
refusal('missing_q1',lambda a:a.pop('q1'))
refusal('maximum_time_resident_phi',lambda a:a.__setitem__('resident_phi',a['phi1'].copy()))
refusal('endpoint_resident_phi',lambda a:a.__setitem__('resident_phi',oracle.screened(a['qfinal']+a['d_initial'],3)))
refusal('frozen_first_stage_phi1',lambda a:a.__setitem__('phi1',a['phi0'].copy()))
refusal('frozen_field_qfinal',lambda a:a.__setitem__('qfinal',oracle.reference(frozen_fields=True)['qfinal']))
refusal('mutated_decay_qfinal',lambda a:a.__setitem__('qfinal',oracle.reference(oracle.replace(oracle.Physics(),decay=.81))['qfinal']))
def flip_donor(a):a['d_final'].view(np.uint64).flat[0]^=np.uint64(1)
refusal('one_bit_donor_mutation',flip_donor)
budget=read(ORACLE/'pre-Native-tolerances.json');refs=np.load(ORACLE/'representative-reference.npz',allow_pickle=False)
X=read(R/'actual-diffusion-exchanges.json');assert len(X)==16*12*4*3
groups={}
for row in X:groups.setdefault(row['evaluation_context'],[]).append(row)
assert len(groups)==3
exchanges=[]
for stage,(context,rows)in enumerate(groups.items()):
 assert len(rows)==16*12*4
 q=refs['q'+str(stage)][0];seen=set();errflux=erramount=0.
 weight=1e-4*[1/6,1/6,2/3][stage]
 for x in rows:
  m=re.fullmatch(r'cell:(\d+):(\d+)/axis:([01])/side:([01])',x['quadrature_identity']);assert m
  i,j,axis,side=map(int,m.groups());assert (i,j,axis,side)not in seen;seen.add((i,j,axis,side))
  orientation=2*side-1;h=[1/8,1/4][axis];area=[1/4,1/8][axis]
  lower=[i,j];upper=[i,j]
  if side:upper[axis]+=1
  else:lower[axis]-=1
  flux=.15*(q[upper[1]%12,upper[0]%16]-q[lower[1]%12,lower[0]%16])/h
  allowance=2*.15/h*budget['absolute_bounds']['q'+str(stage)]+64*np.finfo(float).eps*.15/h*4
  assert x['orientation']==orientation and x['face_measure']==area and x['multiplicity']==1
  assert abs(x['temporal_weight']-weight)<=np.finfo(float).eps*weight
  ferror=abs(x['numerical_flux']-flux);aerror=abs(x['integrated_amount']-orientation*weight*area*flux)
  assert ferror<=allowance and aerror<=weight*area*allowance
  errflux=max(errflux,ferror);erramount=max(erramount,aerror)
 exchanges.append(dict(stage=stage,count=len(rows),temporal_weight=weight,max_flux_error=errflux,max_amount_error=erramount))
cp=cpcheck(R/'actual-accepted-checkpoint.npz')
cp_histories=[]
with np.load(R/'actual-accepted-checkpoint.npz',allow_pickle=False)as z:
 assert z['state_receiver'].tobytes()==actual['q_final'].tobytes()and z['state_reservoir'].tobytes()==actual['donor_final'].tobytes()
 assert sorted(z['history_names'].tolist())==sorted(x['name']for x in P['histories'])
 for name in z['history_names'].tolist():
  assert bool(z['history_init_'+name])and int(z['history_fill_count_'+name])==1
  key='q'+name[-1]if name.startswith('receiver-stage-')else name.rsplit('-',1)[1]+name.split('-')[2]
  slots=z['history_stored_slots_'+name].tolist()
  for slot in slots:assert z['history_'+name+'_'+str(slot)].tobytes()==mapped[key].tobytes(),(name,slot)
  assert np.all(z['history_slot_dt_'+name]==1e-4)
  cp_histories.append(dict(name=name,fill_count=1,physical_depth=int(z['history_depth_'+name]),stored_slots=slots,all_saved_slots_byte_equal_actual_history=True,sample_identity_sha256=digest(z['history_sample_identity_'+name].tobytes()),sample_identity_scope='Authenticated opaque history sample bytes; not proof of stage publication time.'))
pins=[pin(p)for p in sorted(set(B.glob('*.json'))|set((B/'identity').glob('*.json'))|set(R.rglob('*.json'))|set(R.parent.glob('*.json')))]
out=dict(status='ACCEPT_REPRESENTATIVE_CPU_WORLD1_WITH_LIMITS',Source=A['Source'],Native=A['installed_native']['sha256'],SDK=A['SDK'],core_fingerprint=A['core_fingerprint'],core_files_rehashed=len(C['files']),source_installed_wheel_payload=len(sources),all_installed_wheel_members=len(members),objects=objects,observed_CXX_output_actions=len(objects),object_bytes_changed=sum(x['byte_changed']for x in objects),actual_frontend_compile_count=A['actual_frontend_compile_count'],frontend_limit='Ninja output actions do not prove compiler frontend execution; cache/ccache observations remain separate.',cache_vs_installed=dict(cache=pin(A['cache_native_outputs_after'][0]['path']),installed=A['installed_native'],identical_macho_sections_dependencies_uuid=True,cache_rpaths=cache['rpaths'],installed_rpaths=installed['rpaths'],claim='Distinct packaging hashes preserved; no false cache SHA equality.'),producer=prod,binary_provenance=provenance,retained_provenance_files=len(E['retained_cpp_and_provenance']),math=mathcheck,exchanges=exchanges,checkpoint=cp,tableau=S,solve_node_ids=F['solve_node_ids'],history_count=len(P['histories']),accepted_clock=[E['time'],E['macro_step']],stable_runtime_identity_before_after=True,physical_json_pins=pins,pops_modules_loaded=[x for x in sys.modules if x=='pops'or x.startswith('pops.')],limits=['Representative CPU MPI world1 one macro step only; no renamed/changed variant, MPI2/GPU/3D or refinement/convergence qualification.','No persisted CG iterations/status/true residual reports; independent discrete residual and pre-Native budgets verified, generated success guards inspected separately.','No persisted per-stage field publication clocks; actual Program/CPP stage offsets plus independent stage arrays/last-writer resident fields agree. History sample identity is not a stage clock.','No complete frontend-to-binary graph proof. Retained actual C++/model companion receipts/DSOs/SDK and binary identity verified.','SSP coefficient1 certificate inherits only applicable Forward Euler premises; no unconditional positivity/entropy theorem.'],scope='Independent offline reader and independent NumPy/scalar-FV oracle; no PoPS/Native/case-writer import or execution, Main/env mutation, build/install, or SSH.')
out['admission_physical_before_metadata_discrepancies']=before_row_discrepancies
out['independent_actual_physical_field_residuals']=residuals
out['checkpoint_history_byte_checks']=cp_histories
out['oracle_artifacts_rehashed']=ready['evidence_sha256']
out['offline_receiver_negative_controls']=controls
out['producer_runner']=pin(runner)
out['physics_source_original']=pin(original)
out['diagnostic_source_observations_only']=pin(case)
out['installed_sdk_payload_headers']=len([k for k in sources if k.startswith('include/')and k!='include/pops_headers.manifest'])
out['metadata_discrepancy_scope']='Five admission physical_before subrows contain after hashes; true frozen pre-build rows prove five changes. Original admission is preserved, not rewritten; no scientific failure.'
assert not out['pops_modules_loaded']
(O/'independent-reception.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],core=out['core_files_rehashed'],payload=len(sources),objects=len(objects),changed=out['object_bytes_changed'],exchanges=len(X),checkpoint_arrays=cp['arrays'])))
