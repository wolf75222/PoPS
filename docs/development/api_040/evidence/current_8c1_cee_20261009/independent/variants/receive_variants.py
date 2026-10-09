import sys,json,importlib.util,xml.etree.ElementTree as ET,re
from pathlib import Path
import numpy as np
from physical_protocol import pin,sha,read,physical,cpcheck,identity

B=Path('/Users/romaindespoulain/dev/tmp/PoPS-native-stage-fields-cpp-reception-20261009')
BASE=B/'stage-fields-representative/run';R=B/'stage-fields-rename/run'
D=B/'public-three-native-tests/tmp/test_public_three_stage_screen2/decay'
O=Path(__file__).resolve().parent
ORACLE=Path('/Users/romaindespoulain/dev/tmp/PoPS-sol61-stage-fields-independent-oracle-20261009')
ready=read(ORACLE/'representative-oracle-ready.json')
assert sha(ORACLE/'representative-oracle-ready.json')=='28ca67fc888d78ae3f57cc417759b4b9624692c9499d5d1ecacb1fc793b6b81e'
for p,h in ready['evidence_sha256'].items():assert sha(p)==h
spec=importlib.util.spec_from_file_location('independent_oracle',ORACLE/'independent_reference.py');oracle=importlib.util.module_from_spec(spec);sys.modules[spec.name]=oracle;spec.loader.exec_module(oracle)
A=read(B/'native-build-admission.json');assert sha(B/'native-build-admission.json')=='182f374d82334398832430a146ba9516c995dd27bd827dc76a9624c0211585fb'
physical(A['installed_native']);physical(A['wheel'])
def identity_check(x):
 for key in ['Source','Native','SDK']:assert x[key]=={'Source':A['Source'],'Native':A['installed_native']['sha256'],'SDK':A['SDK']}[key]
def binaries(rows):
 result=[]
 for name,row in rows.items():
  if isinstance(row,str):row=pin(name);assert row['sha256']==rows[name]
  physical(row);p=Path(row['path']);assert A['SDK'].encode()in p.read_bytes();side=read(str(p)+'.pops-artifact.json')
  bi=identity('binary',dict(algorithm='sha256',content_digest=bytes.fromhex(sha(p)),size=p.stat().st_size));assert side['binary_identity']=='pops.binary.v1:sha256:'+bi
  comp=Path(str(p)+'.pops-model-compile.json');entry=dict(name=name,binary=row,sidecar=pin(str(p)+'.pops-artifact.json'))
  if comp.exists():
   c=read(comp);cpp=Path(str(p)+'.pops-model.cpp');assert sha(cpp)==c['source_sha256']and sha(p)==c['binary_sha256']and c['header_signature']==A['SDK'];assert side['model_compile_provenance_sha256']==sha(comp)
   entry.update(model_source=pin(cpp),compile_receipt=pin(comp),compiler=pin(c['compiler_file']));assert entry['compiler']['sha256']==c['compiler_sha256']
  result.append(entry)
 return result
def canonical(p):
 a=np.load(p/'actual-states-and-fields.npz',allow_pickle=False);bind=np.load(p/'actual-bind-inputs.npz',allow_pickle=False)
 result=dict(q0=a['q_stage0'],q1=a['q_stage1'],q2=a['q_stage2'],qfinal=a['q_final'],d_initial=bind['donor'],d_final=a['donor_final'],resident_phi=a['phi_resident'][None,:,:],resident_psi=a['psi_resident'][None,:,:])
 for j in range(3):
  for role in ['phi','psi']:result[role+str(j)]=a[role+'_stage'+str(j)]
 return result
def history_check(cp,arrays):
 rows=[]
 with np.load(cp,allow_pickle=False)as z:
  for name in z['history_names'].tolist():
   key='q'+name[-1]if name.startswith('receiver-stage-')else name.rsplit('-',1)[1]+name.split('-')[2]
   assert bool(z['history_init_'+name])and int(z['history_fill_count_'+name])==1
   for slot in z['history_stored_slots_'+name]:assert z['history_'+name+'_'+str(slot)].tobytes()==arrays[key].tobytes()
   rows.append(dict(name=name,depth=int(z['history_depth_'+name]),fill=1,all_saved_slots_match=True))
 return rows

E=read(R/'execution-closed.json');identity_check(E['identity']);assert E['variant']=='rename'and E['decay']=='4/5'
prod=read(R.parent/'producer-result.json');assert prod['exit']==0 and sha(R.parent/'producer.log')==prod['producer_log_sha256']
iv=read(R.parent/'invocation.json');assert sha(iv['argv'][1])==iv['runner_sha256']=='b189d26a3e194a514c2abb4928d7de7b0a33beef75b807849c4f4f6b28a47011'
runner2=Path('/Users/romaindespoulain/dev/tmp/PoPS-root-stage-fields-native-protocol-20261009/mechanically-reviewed-runner-r2/run_native_capture.py')
assert sha(runner2)=='3c4972d8a294b9ff98370fb854fbdd1c707d12a473be1df0ac4edba17e4a8dd5'
assert runner2.read_text().replace("role:'renamed-'+role.replace('_','-')", "role:'renamed_'+role")==Path(iv['argv'][1]).read_text()
assert sha('/Users/romaindespoulain/dev/tmp/PoPS-root-stage-fields-native-protocol-20261009/stage_fields_case_observed.py')=='bdaee56ac52b4391ed7b5402a95c00524b5ae9a4b48dd47b44fef73799bfad42'
assert E['case_source_sha256']=='bdaee56ac52b4391ed7b5402a95c00524b5ae9a4b48dd47b44fef73799bfad42'
assert all(v=='renamed_'+k for k,v in E['labels'].items())
before=read(R/'identity-before.json');after=read(R/'identity-after.json');identity_check(before);identity_check(after)
assert {k:v for k,v in before.items()if k!='runtime_environment'}=={k:v for k,v in after.items()if k!='runtime_environment'}
for key in ['dimension','mpi_rank','mpi_ranks','kokkos_backend','kokkos_device','precision','real_bytes']:assert before['runtime_environment'][key]==after['runtime_environment'][key]
assert after['runtime_environment']['mpi_ranks']==1 and E['time']==1e-4 and E['macro_step']==E['accepted_steps']==1
raw=[]
for file in ['actual-states-and-fields.npz','actual-bind-inputs.npz']:
 with np.load(BASE/file,allow_pickle=False)as base,np.load(R/file,allow_pickle=False)as renamed:
  assert set(base.files)==set(renamed.files)
  for key in base.files:
   assert base[key].dtype==renamed[key].dtype and base[key].shape==renamed[key].shape and base[key].tobytes()==renamed[key].tobytes()
   raw.append(dict(file=file,key=key,shape=list(base[key].shape),byte_exact=True))
rename_math=oracle.check_arrays(canonical(R));basecanonical=canonical(BASE)
assert all(basecanonical[k].tobytes()==canonical(R)[k].tobytes()for k in basecanonical)
for name,row in E['retained_cpp_and_provenance'].items():assert (R/name).stat().st_size==row['bytes']and sha(R/name)==row['sha256']
rename_bins=binaries(E['binaries']);rename_cp=cpcheck(R/'actual-accepted-checkpoint.npz')
cpnumeric=[];id_differences=[]
with np.load(BASE/'actual-accepted-checkpoint.npz',allow_pickle=False)as base,np.load(R/'actual-accepted-checkpoint.npz',allow_pickle=False)as renamed:
 def original_key(k):
  return k.replace('renamed_receiver_block','receiver').replace('renamed_donor_block','reservoir')
 for k in renamed.files:
  v=renamed[k];bk=original_key(k)
  if v.dtype.kind in 'fbiu'and v.dtype!=np.uint8 and k not in ['program_exchange_offsets','program_diagnostics_offsets']:
   assert v.dtype==base[bk].dtype and v.shape==base[bk].shape and v.tobytes()==base[bk].tobytes(),k
   cpnumeric.append(dict(renamed_key=k,representative_key=bk,byte_exact=True))
  else:
   if bk in base.files and (v.shape!=base[bk].shape or v.dtype!=base[bk].dtype or v.tobytes()!=base[bk].tobytes()):id_differences.append(k)
rename_histories=history_check(R/'actual-accepted-checkpoint.npz',canonical(R))
renamed_ssp=read(R/'actual-accepted-ssp.json');base_ssp=read(BASE/'actual-accepted-ssp.json')
for k in ['A','b','c','coefficient']:assert renamed_ssp[k]==base_ssp[k]
rename_exchanges=read(R/'actual-diffusion-exchanges.json');base_exchanges=read(BASE/'actual-diffusion-exchanges.json')
assert len(rename_exchanges)==len(base_exchanges)==2304
for renamed,base in zip(rename_exchanges,base_exchanges):
 for k in ['quadrature_identity','orientation','multiplicity']:assert renamed[k]==base[k]
 for k in ['face_measure','numerical_flux','temporal_weight','integrated_amount']:assert np.float64(renamed[k]).tobytes()==np.float64(base[k]).tobytes()

dec=read(D/'capture-receipt.json');identity_check(dec['identity']);assert dec['variant']=='decay'and dec['decay']=='11/10'and dec['time']==1e-4 and dec['macro_step']==dec['accepted_steps']==1
params=oracle.replace(oracle.Physics(),decay=11/10)
budget=read(D.parent/'pre-execution-independent-budget.json');assert oracle.tolerances(params)==budget
assert (D.parent/'pre-execution-independent-budget.json').stat().st_mtime_ns<(D/'actual-arrays.npz').stat().st_mtime_ns
actual=dict(np.load(D/'actual-arrays.npz',allow_pickle=False));decay_math=oracle.check_arrays(actual,params)
bind=np.load(D/'actual-bind-inputs.npz',allow_pickle=False);assert actual['d_initial'].tobytes()==bind['donor'].tobytes()and actual['q0'].tobytes()==bind['receiver'].tobytes()
assert actual['d_final'].tobytes()==basecanonical['d_final'].tobytes()
sep=float(np.max(np.abs(actual['qfinal']-basecanonical['qfinal'])));assert sep>budget['absolute_bounds']['qfinal']+oracle.tolerances()['absolute_bounds']['qfinal']
decay_bins=binaries(dec['binaries']);decay_cp=cpcheck(Path(dec['checkpoint']['path']));assert decay_cp['sha256']==dec['checkpoint']['sha256']
decay_histories=history_check(Path(dec['checkpoint']['path']),actual)
program=dec['program'];solves=[x for x in program['nodes']if x['op']=='solve_linear'];assert [x['id']for x in solves]==[8,24,40]
for x in solves:
 a=x['attrs'];assert a['ncomp']==2 and a['max_iter']==2000 and not a['has_guess']and float.fromhex(a['tol']['value'])==1e-12 and float.fromhex(a['abs_tol']['value'])==1e-14
cpps=[(p,p.read_text())for p in [D/'program0.cpp',D/'model0.cpp',D/'model1.cpp']]
rate_matches=[dict(path=str(p),lines=[dict(line=i+1,text=line.strip())for i,line in enumerate(s.splitlines())if '1.1' in line or ('pops::Real(11)'in line and 'pops::Real(10)'in line)])for p,s in cpps]
assert any(x['lines']for x in rate_matches)
expanded=next((D.parent/'private/generated').glob('*.cpp'))
assert (D/'program0.cpp').read_text()in expanded.read_text()
assert expanded.read_text().count('outA(index, 0) = (((0.2 * screened_first) + (0.4 * screened_second)) - (1.1 * q));')==3
with np.load(dec['checkpoint']['path'],allow_pickle=False)as z:
 assert z['state_receiver'].tobytes()==actual['qfinal'].tobytes()and z['state_reservoir'].tobytes()==actual['d_final'].tobytes()
xml=ET.parse(B/'public-three-native-tests/cohort.xml');cases=[dict(name=x.attrib['name'],passed=x.find('failure')is None and x.find('error')is None and x.find('skipped')is None,failure=x.find('failure').attrib.get('message')if x.find('failure')is not None else None)for x in xml.iter('testcase')]
assert len(cases)==3 and sum(x['passed']for x in cases)==2
assert next(x for x in cases if '[decay]'in x['name'])['passed']
assert not next(x for x in cases if '[rename]'in x['name'])['passed']
assert read(B/'public-three-native-tests/producer-result.json')['exit']==1
pins=[pin(p)for p in sorted(set(R.rglob('*.json'))|set(R.parent.glob('*.json'))|set(D.glob('*'))|{D.parent/'pre-execution-independent-budget.json',B/'public-three-native-tests/cohort.xml',B/'public-three-native-tests/producer-result.json',B/'public-three-native-tests/invocation.json'})if p.is_file()]
out=dict(verdict='ACCEPT_RENAME_AND_CHANGED_DECAY_CPU_WORLD1_WITH_LIMITS',Source=A['Source'],Native=A['installed_native']['sha256'],SDK=A['SDK'],oracle_ready_sha256=sha(ORACLE/'representative-oracle-ready.json'),unchanged_baseline_budget_sha256=sha(ORACLE/'pre-Native-tolerances.json'),rename=dict(producer=prod,runner=pin(iv['argv'][1]),canonical_oracle_quantities=14,raw_state_field_arrays=13,raw_bind_arrays=2,all_raw_arrays_byte_exact=raw,math=rename_math,binaries=rename_bins,checkpoint=rename_cp,checkpoint_numeric_members_byte_exact=cpnumeric,checkpoint_identity_or_opaque_differences=id_differences,checkpoint_scope='98 numeric members byte-exact after explicit block role-key mapping; identifiers/string envelopes/opaque byte carriers are not claimed byte-identical.',histories=rename_histories),decay=dict(parameter='11/10',budget=budget,budget_pin=pin(D.parent/'pre-execution-independent-budget.json'),pre_execution_evidence='Actual test source writes independent tolerances(params) before capture; preserved file mtime precedes actual arrays. Same independent derivation was frozen before representative execution.',math=decay_math,qfinal_separation_from_representative=sep,donor_byte_exact=True,binaries=decay_bins,checkpoint=decay_cp,histories=decay_histories,changed_CPP_rate_witness=rate_matches),original_public_three_result=cases,physical_pins=pins,pops_modules_loaded=[x for x in sys.modules if x=='pops'or x.startswith('pops.')],limits=['Public-three original cohort is 2PASS/1Source-grammarFAIL with process exit1; valid-label rename is a distinct later closed0 capture, not a rewrite of original failure.','CPU MPI world1 one macro step only; no MPI2/GPU/3D/refinement/longtime result.','No persisted CG reports, stage publication clocks, complete frontend-to-binary graph or unconditional FE positivity theorem.','Rename checkpoint semantic numerical members match; renamed identifiers/envelopes and opaque carriers may differ legitimately.'],scope='Non-author offline mathematical/physical reception; no Native/PoPS/case-author import or execution, no Main/env/build/install/SSH operation.')
assert not out['pops_modules_loaded']
out['rename']['checkpoint_scope']='96 numerical diagnostic members byte-exact after explicit receiver/donor role-key mapping. Two offsets index serialized identifiers and differ legitimately (program_exchange_offsets, program_diagnostics_offsets); strings/envelopes/opaque byte carriers are not claimed byte-identical.'
out['rename']['exchange_numeric_rows_byte_exact']=len(rename_exchanges)
out['rename']['runner_delta_exact_valid_labels_only']=dict(original=pin(runner2),new=pin(iv['argv'][1]),all_other_source_bytes_equal=True)
out['rename']['tableau_unchanged']=True
out['decay']['actual_expanded_CPP']=pin(expanded)
out['decay']['saved_program_source_exact_subset_of_expanded_CPP']=True
out['decay']['checkpoint_live_state_bytes_match_capture']=True
out['pre_execution_budget_chronology_evidence']=dict(preserved_xml_source_snippet='Original retained failure traceback includes budget=tolerances(physics), write pre-execution-independent-budget.json, then capture(variant,...) in that order.',budget_mtime_ns=(D.parent/'pre-execution-independent-budget.json').stat().st_mtime_ns,actual_arrays_mtime_ns=(D/'actual-arrays.npz').stat().st_mtime_ns,scope='Code ordering and retained filesystem observations support timing; no full frontend graph or cryptographic timestamp claim.')
(O/'variant-reception.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(verdict=out['verdict'],rename_canonical=14,raw_states_fields=13,raw_bind=2,checkpoint_numeric=len(cpnumeric),decay_separation=sep)))
