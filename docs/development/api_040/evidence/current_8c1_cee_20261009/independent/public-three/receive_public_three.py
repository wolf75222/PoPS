import sys,json,importlib.util,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
import numpy as np
from physical_protocol import pin,sha,read,physical,cpcheck,identity

B=Path('/Users/romaindespoulain/dev/tmp/PoPS-native-stage-fields-cpp-reception-20261009')
PH=B/'public-three-native-tests-valid-e27d'
O=Path(__file__).resolve().parent
M=Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004')
ORACLE=Path('/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-fields-independent-oracle-20261009')
PARENT=O.parent
expected_metadata='e27d065a5eb7c086ced60b751a2468f729361ff0'
prod=read(PH/'producer-result.json');assert prod['exit']==0 and prod['Source']==expected_metadata and sha(PH/'pytest.log')==prod['log_sha256']
A=read(B/'native-build-admission.json');C=read(B/'core-manifest.json')
assert sha(B/'native-build-admission.json')=='182f374d82334398832430a146ba9516c995dd27bd827dc76a9624c0211585fb'
assert prod['qualified_cpu_build_source']==A['Source']=='8c1f55701cab7c60b3519bd343e3b4c1f193570f'
assert prod['core_fingerprint_unchanged']==A['core_fingerprint']==C['core_fingerprint']
for name,row in C['files'].items():assert sha(M/name)==row['sha256']and (M/name).stat().st_size==row['bytes']
physical(A['installed_native']);physical(A['wheel'])
sources=read(B/'identity/source-files.json');pack=Path(A['installed_native']['path']).parents[2]
with zipfile.ZipFile(A['wheel']['path'])as z:
 for name,h in sources.items():
  relative=name.removeprefix('python/pops/')if name.startswith('python/pops/')else name
  assert sha(M/name)==sha(pack/relative)==__import__('hashlib').sha256(z.read('pops/'+relative)).hexdigest()==h
assert sha(PARENT/'independent-reception-ready.json')=='9573299df4fd52b3e192fecc50936247a80f3a284d443c94364871d163045cf6'
assert sha(PARENT/'variants/variant-reception-ready.json')=='2516af9a674cfc2414a48e0dcf14db7be6a93fc8c62fdf1306a8079588c22883'
assert sha(ORACLE/'representative-oracle-ready.json')=='28ca67fc888d78ae3f57cc417759b4b9624692c9499d5d1ecacb1fc793b6b81e'
ready=read(ORACLE/'representative-oracle-ready.json')
for p,h in ready['evidence_sha256'].items():assert sha(p)==h
spec=importlib.util.spec_from_file_location('independent_oracle',ORACLE/'independent_reference.py');oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)
xml=ET.parse(PH/'cohort.xml');cases=list(xml.iter('testcase'));assert len(cases)==3
assert all(all(t.find(k)is None for k in ['failure','error','skipped'])for t in cases)
assert prod['counts']==dict(tests=3,failures=0,errors=0,skipped=0)
base=Path(B/'stage-fields-representative/run');basearrays=np.load(base/'actual-states-and-fields.npz',allow_pickle=False);basebind=np.load(base/'actual-bind-inputs.npz',allow_pickle=False)
baseline=dict(q0=basearrays['q_stage0'],q1=basearrays['q_stage1'],q2=basearrays['q_stage2'],qfinal=basearrays['q_final'],d_initial=basebind['donor'],d_final=basearrays['donor_final'],resident_phi=basearrays['phi_resident'][None,:,:],resident_psi=basearrays['psi_resident'][None,:,:])
for j in range(3):
 for n in ['phi','psi']:baseline[n+str(j)]=basearrays[n+'_stage'+str(j)]
olddecay=np.load(B/'public-three-native-tests/tmp/test_public_three_stage_screen2/decay/actual-arrays.npz',allow_pickle=False)
rows=[];allpins=[]
for receipt in sorted((PH/'tmp').rglob('capture-receipt.json')):
 d=receipt.parent;r=read(receipt);v=r['variant'];assert v in ['representative','rename','decay']
 for k,expected in [('Source',expected_metadata),('Native',A['installed_native']['sha256']),('SDK',A['SDK'])]:assert r['identity'][k]==expected
 assert r['time']==1e-4 and r['accepted_steps']==r['macro_step']==1
 arrays=dict(np.load(d/'actual-arrays.npz',allow_pickle=False));bind=np.load(d/'actual-bind-inputs.npz',allow_pickle=False)
 params=oracle.replace(oracle.Physics(),decay=1.1)if v=='decay'else oracle.Physics()
 comparisons=oracle.check_arrays(arrays,params);budget=read(d.parent/'pre-execution-independent-budget.json');assert budget==oracle.tolerances(params)
 assert (d.parent/'pre-execution-independent-budget.json').stat().st_mtime_ns<(d/'actual-arrays.npz').stat().st_mtime_ns
 assert arrays['q0'].tobytes()==bind['receiver'].tobytes()and arrays['d_initial'].tobytes()==bind['donor'].tobytes()
 expected=olddecay if v=='decay'else baseline
 assert set(arrays)==set(expected)
 bytechecks=[]
 for k in arrays:
  assert arrays[k].dtype==expected[k].dtype and arrays[k].shape==expected[k].shape and arrays[k].tobytes()==expected[k].tobytes(),(v,k)
  bytechecks.append(dict(quantity=k,shape=list(arrays[k].shape),byte_exact_prior_independent_capture=True))
 cp=cpcheck(Path(r['checkpoint']['path']));assert cp['sha256']==r['checkpoint']['sha256'];history=[]
 with np.load(r['checkpoint']['path'],allow_pickle=False)as z:
  state_receiver='renamed_receiver_block'if v=='rename'else 'receiver';state_donor='renamed_donor_block'if v=='rename'else 'reservoir'
  assert z['state_'+state_receiver].tobytes()==arrays['qfinal'].tobytes()and z['state_'+state_donor].tobytes()==arrays['d_final'].tobytes()
  assert len(z['history_names'])==9
  for name in z['history_names'].tolist():
   key='q'+name[-1]if name.startswith('receiver-stage-')else name.rsplit('-',1)[1]+name.split('-')[2]
   assert bool(z['history_init_'+name])and int(z['history_fill_count_'+name])==1
   for slot in z['history_stored_slots_'+name]:assert z['history_'+name+'_'+str(slot)].tobytes()==arrays[key].tobytes()
   assert np.all(z['history_slot_dt_'+name]==1e-4)
   history.append(dict(name=name,fill=1,depth=int(z['history_depth_'+name]),all_saved_slots_byte_match=True))
 binaries=[]
 for path,h in r['binaries'].items():
  p=Path(path);assert sha(p)==h and A['SDK'].encode()in p.read_bytes();side=read(str(p)+'.pops-artifact.json')
  bidentity=identity('binary',dict(algorithm='sha256',content_digest=bytes.fromhex(h),size=p.stat().st_size));assert side['binary_identity']=='pops.binary.v1:sha256:'+bidentity
  row=dict(DSO=pin(p),sidecar=pin(str(p)+'.pops-artifact.json'))
  comppath=Path(str(p)+'.pops-model-compile.json')
  if comppath.exists():
   c=read(comppath);source=Path(str(p)+'.pops-model.cpp');assert sha(source)==c['source_sha256']and c['binary_sha256']==h and c['header_signature']==A['SDK']and sha(comppath)==side['model_compile_provenance_sha256'];assert sha(c['compiler_file'])==c['compiler_sha256']
   row.update(actual_model_source=pin(source),actual_model_compile_receipt=pin(comppath))
  binaries.append(row)
 solves=[x for x in r['program']['nodes']if x['op']=='solve_linear'];assert [x['id']for x in solves]==[8,24,40]
 for x in solves:
  a=x['attrs'];assert a['ncomp']==2 and a['max_iter']==2000 and not a['has_guess']and float.fromhex(a['tol']['value'])==1e-12 and float.fromhex(a['abs_tol']['value'])==1e-14
 generated=next((d.parent/'private/generated').glob('*.cpp'));source=(d/'program0.cpp').read_text();assert source in generated.read_text()
 expected_rate='1.1'if v=='decay'else '0.8'
 assert source.count(' - ('+expected_rate+' * q));')==3
 rows.append(dict(variant=v,case=next(t.attrib['name']for t in cases if '['+v+']'in t.attrib['name']),capture=pin(receipt),actual_arrays=pin(d/'actual-arrays.npz'),bind_arrays=pin(d/'actual-bind-inputs.npz'),budget=pin(d.parent/'pre-execution-independent-budget.json'),independent_math=comparisons,canonical_quantities=len(bytechecks),all_quantities_byte_exact_prior_independent_capture=bytechecks,checkpoint=cp,histories=history,binaries=binaries,actual_program_CPP=pin(d/'program0.cpp'),expanded_CPP=pin(generated),saved_program_exact_subset_of_expanded=True,rate_witness_each_of_three_stages=expected_rate,accepted_clock=[r['time'],r['macro_step']],CG_solver_controls_preserved=True))
 allpins.extend(pin(p)for p in sorted(d.iterdir())if p.is_file())
assert len(rows)==3 and {x['variant']for x in rows}=={'representative','rename','decay'}
out=dict(verdict='ACCEPT_THREE_PUBLIC_NATIVE_TESTS_CURRENT_CAPTURE_WITH_LIMITS',metadata_Source=prod['Source'],qualified_native_build_Source=A['Source'],Native=A['installed_native']['sha256'],SDK=A['SDK'],core_fingerprint=C['core_fingerprint'],core_files_rehashed=len(C['files']),source_install_wheel_payload_rehashed=len(sources),producer=prod,xml=pin(PH/'cohort.xml'),invocation=pin(PH/'invocation.json'),cases=rows,physical_captured_pins=allpins,prior_independent_receipts=[pin(PARENT/'independent-reception-ready.json'),pin(PARENT/'variants/variant-reception-ready.json')],unchanged_oracle=pin(ORACLE/'representative-oracle-ready.json'),unchanged_original_budget=pin(ORACLE/'pre-Native-tolerances.json'),pops_modules_loaded=[x for x in sys.modules if x=='pops'or x.startswith('pops.')],limits=['Three genuine new public tests pass CPUworld1; this is local runtime evidence, not GitHub CI or GPU/MPI2/longtime/refinement qualification.','Native remains compiled Source8c1f; e27d is metadata/CI/test-label Source and is not relabeled as a newly compiled Native build. Core1221 and installed/wheel1175 remain exact.','No persisted CG reports or per-stage publication clocks; no complete frontend-to-binary graph; conditional FE coefficient1 only, not unconditional positivity/entropy.','Original failed public-three cohort is preserved separately; this new3/3 capture does not rewrite its2PASS/1SourcegrammarFAIL.'],scope='Independent offline non-author mathematical and physical reception; own frozen NumPy/scalar-FV oracle only; no Native/PoPS/case-writer import/run, Main/env/build/install/SSH operation.')
assert not out['pops_modules_loaded']
(O/'public-three-reception.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(verdict=out['verdict'],tests=3,canonical_quantities=42,history_rings=27,checkpoint_arrays=sum(x['checkpoint']['arrays']for x in rows),core=out['core_files_rehashed'],payload=len(sources))))
