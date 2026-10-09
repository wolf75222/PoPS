import json,hashlib,subprocess,re,xml.etree.ElementTree as ET,sys
from pathlib import Path

B=Path('/Users/romaindespoulain/dev/tmp/PoPS-cpp-private-kernel-nonregression-8c1f-20261009')
W=B/'remaining-world1-e27d';M=Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004');O=Path(__file__).resolve().parent
BUILD='8c1f55701cab7c60b3519bd343e3b4c1f193570f';META='e27d065a5eb7c086ced60b751a2468f729361ff0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def xmlcases(p):
 x=ET.parse(p);rows=[]
 for t in x.iter('testcase'):
  assert all(t.find(k)is None for k in ['failure','error','skipped']),t.attrib
  rows.append(dict(name=t.attrib['name'],classname=t.attrib.get('classname'),seconds=t.attrib.get('time'),passed=True))
 return rows
inv=read(B/'invocation.json');old=read(B/'phase-result.json');new=read(W/'phase-result.json');wi=read(W/'invocation.json')
assert old['Source']==inv['Source']==new['compiled_Source']==wi['compiled_Source']==BUILD
assert new['current_metadata_Source']==wi['current_metadata_Source']==META
assert old['exit']==new['exit']==0 and sha(B/'phase.log')==old['log_sha256']and sha(W/'phase.log')==new['log_sha256']and sha(W/'ctest.xml')==new['xml_sha256']
assert new['counts']==dict(tests=26,failures=0,errors=0,skipped=0)
diff=subprocess.check_output(['git','diff','--name-only',BUILD,META],cwd=M,text=True).splitlines()
assert sorted(diff)==sorted(wi['unchanged_compiled_core_proof'])
expected=['.github/workflows/ci.yml','scripts/ci_pytest_timings.py','tests/python/architecture/test_ci_impacted_selection.py','tests/python/support/stage_fields_public_fixture.py','tests/python/test_durations.json']
assert sorted(diff)==sorted(expected)
cache=(B/'build/CMakeCache.txt').read_text()
for text in ['CMAKE_BUILD_TYPE:STRING=Release','POPS_USE_KOKKOS:BOOL=ON','POPS_USE_MPI:BOOL=ON','POPS_NATIVE_DIM:UNINITIALIZED=2']:assert text in cache
cc=read(B/'build/compile_commands.json');ninja=(B/'build/.ninja_log').read_text().splitlines()[1:]
assert len(ninja)==35
outputs=[r.split('\t')[3]for r in ninja]
objects=[]
for output in outputs:
 p=B/'build'/output;assert p.is_file()
 if output.endswith('.o'):
  records=[x for x in cc if (Path(x['directory'])/x['output']).resolve()==p.resolve()];assert len(records)==1
  x=records[0];source=Path(x['file']);command=x['command'];assert '-O3'in command and '-DNDEBUG'in command
  row=dict(object=pin(p),source=pin(source),command_sha256=hashlib.sha256(command.encode()).hexdigest(),command=command)
  if source.is_relative_to(M):
   relative=str(source.relative_to(M));blob=subprocess.check_output(['git','show',BUILD+':'+relative],cwd=M)
   if relative!='tests/cpp/support/gtest_main.cpp':assert '-DPOPS_HAS_KOKKOS'in command and '-DPOPS_HAS_MPI'in command and '-DPOPS_NATIVE_DIM=2'in command
   else:row['role']='Plain GoogleTest host harness, not a PoPS numerical kernel or Native-module compile.'
   assert hashlib.sha256(blob).hexdigest()==row['source']['sha256']
   row['actual_source_bytes_equal_compiled_commit']=True
  objects.append(row)
bins=[]
for target in inv['targets']:
 p=B/'build/bin'/target;assert str(p.relative_to(B/'build'))in outputs
 row=dict(target=target,binary=pin(p));source=[x for x in cc if Path(x['file']).name==target+'.cpp'];assert len(source)==1
 row['test_TU']=pin(source[0]['file']);rel=str(Path(source[0]['file']).relative_to(M));blob=subprocess.check_output(['git','show',BUILD+':'+rel],cwd=M);assert hashlib.sha256(blob).hexdigest()==row['test_TU']['sha256'];row['test_body_source_unchanged_from_compiled_commit']=True;bins.append(row)
assert len(bins)==6
for row in wi['binaries']:assert pin(row['path'])==row
mpi_groups=xmlcases(B/'ctest.xml');assert len(mpi_groups)==3
expected_groups={'test_mpi_composite_fac_partitioned_nd_np2','test_mpi_field_nullspace_preflight_np2','test_tensor_fac_conservative_interface_np2'}
assert {x['name']for x in mpi_groups}==expected_groups
rank_results=[]
for f in sorted((B/'build/test-results/gtest').glob('*.xml')):
 m=re.fullmatch(r'(.+)\.rank([01])\.xml',f.name);assert m
 assert m.group(1)in expected_groups
 rank_results.append(dict(group=m.group(1),rank=int(m.group(2)),xml=pin(f),cases=xmlcases(f)))
assert len(rank_results)==6
for group in expected_groups:
 rr=[x for x in rank_results if x['group']==group];assert {x['rank']for x in rr}=={0,1}
 assert [x['name']for x in rr[0]['cases']]==[x['name']for x in rr[1]['cases']]
assert sum(len(x['cases'])for x in rank_results if x['rank']==0)==10
assert sum(len(x['cases'])for x in rank_results if x['rank']==1)==10
world1=xmlcases(W/'ctest.xml');assert len(world1)==26
grouping={}
for x in world1:
 suite=x['name'].split('.',1)[0];grouping.setdefault(suite,[]).append(x)
assert {k:len(v)for k,v in grouping.items()}==dict(HierarchyTensorExactRank=11,test_field_nullspace=14,AmrProgramFieldPublication=1)
pins=[pin(p)for p in [B/'phase-result.json',B/'phase.log',B/'ctest.xml',B/'actual-commands.sh',B/'invocation.json',W/'phase-result.json',W/'phase.log',W/'ctest.xml',W/'actual-commands.sh',W/'invocation.json',B/'build/CMakeCache.txt',B/'build/compile_commands.json',B/'build/build.ninja',B/'build/.ninja_log']]
out=dict(verdict='ACCEPT_SELECTED_CPP_NONREGRESSION_EXECUTION_WITH_LIMITS',compiled_Source=BUILD,current_metadata_Source=META,metadata_delta_exact_five_paths=diff,build_configuration=dict(Release=True,optimization='O3 DNDEBUG',MPI=True,Kokkos=True,native_dimension=2),binary_targets=bins,observed_ninja_output_records=35,observed_CXX_object_outputs=len(objects),objects=objects,world1=dict(producer=new,counts=new['counts'],inventory_exact_26=world1,suite_counts={k:len(v)for k,v in grouping.items()}),MPI2=dict(producer=old,CTest_groups=mpi_groups,rank_GTest_results=rank_results,distinct_selected_cases_per_rank=10,rank_case_executions=20),physical_pins=pins,pops_modules_loaded=[x for x in sys.modules if x=='pops'or x.startswith('pops.')],limits=['Six selected binary targets only, not full C++ catalog, GitHub CI or aggregate94closure.','World1 registration has26tests; MPI2 has3CTestgroups and10GoogleTestcases perrank,20rankexecutions. These distinct granularities are preserved; no invented combined36globaltest count.','Standalone C++ template tests include exact-ranked1D/2D/3D constructions as named in inventory; installed Native module remains dimension2. No new3DNative/GPU or physical PDE refinement/convergence qualification.','No fresh numerical oracle for every C++ body; actual existing assertions, source-preservation and real execution outputs authenticated.','Ninja output actions and retained objects/binaries do not prove uncached compiler frontend execution for every action.'],scope='Independent offline non-author reception of existing actual process/XML/physical-source/binary evidence; no Native/PoPS import/run, build/install, Main/env or SSH mutation.')
assert not out['pops_modules_loaded']
(O/'cpp-nonregression-reception.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(verdict=out['verdict'],binary_targets=6,world1_tests=26,MPI2_CTest_groups=3,MPI2_GTest_cases_per_rank=10,MPI2_rank_case_executions=20,observed_ninja_records=35,CXX_objects=len(objects))))
