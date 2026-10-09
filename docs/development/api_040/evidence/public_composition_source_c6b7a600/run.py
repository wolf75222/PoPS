"""Read-only integrated Source/host regression receipt; no Native qualification."""
import hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path('/Users/romaindespoulain/dev/tmp/pops-api040-initial-ghost-native-20261002')
OUT=Path(__file__).resolve().parent
BASE='c6b7a600033cb3ba42ca6adf7ca9c743e216ab87'
PRIOR=Path('/Users/romaindespoulain/dev/tmp/sol61-integrated-source-21f2-evidence/receipt.json')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def production(path):
    return path.startswith(('python/','include/','src/','cmake/','scripts/')) or path in ('CMakeLists.txt','pyproject.toml','setup.py','setup.cfg')
def snapshot():
    rows={};drift=[]
    for entry in git('ls-tree','-r','-z',BASE).split(b'\0'):
        if not entry:continue
        metadata,rawpath=entry.split(b'\t',1);mode,kind,blob=metadata.split()
        path=rawpath.decode()
        if not production(path):continue
        p=ROOT/path
        data=os.fsencode(os.readlink(p)) if mode==b'120000' else p.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=blob.decode():drift.append(path)
        rows[path]=hashlib.sha256(data).hexdigest()
    return {'head':git('rev-parse','HEAD').decode().strip(),'base':BASE,'production_sha256':rows,'base_drift':drift,'dirty':git('status','--porcelain').decode()}

class ImmediateReporter:
    def pytest_collection_finish(self,session):
        save('collected-nodes.json',[item.nodeid for item in session.items])
    def pytest_runtest_logreport(self,report):
        row={'nodeid':report.nodeid,'phase':report.when,'outcome':report.outcome,'duration':report.duration,'trace':str(report.longrepr) if report.failed else None}
        with (OUT/'reports.jsonl').open('a') as stream:
            stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
        if report.failed:
            print('IMMEDIATE_FAILURE '+json.dumps(row,sort_keys=True),flush=True)

def main():
    prior=json.loads(PRIOR.read_text())
    basefiles={case['classname'].replace('.','/')+'.py' for case in prior['cases']}
    added={
     'tests/review/test_sol61_flux_wave_independent.py',
     'tests/review/test_flux_wave_authority.py',
     'tests/python/unit/numerics/test_principal_grouping_contract.py',
     'tests/python/unit/codegen/test_principal_group_codegen.py',
     'tests/python/unit/codegen/test_principal_group_guards.py',
     'tests/python/unit/codegen/test_module_lowering.py',
     'tests/python/unit/codegen/test_module_lowering_coverage.py',
     'tests/python/unit/codegen/test_module_hash.py',
     'tests/review/test_sol61_c25_independent.py',
     'tests/review/test_sol61_model_source_evidence.py',
     'tests/python/unit/codegen/test_compile_provenance.py',
     'tests/python/unit/codegen/test_facade_compile_cache.py',
     'tests/python/unit/codegen/test_compile_cache_lock.py',
     'tests/review/test_sol61_scoped_wave_native_preparation.py',
     'tests/review/test_sol61_m16_second_state_native_preparation.py',
     'tests/review/test_sol61_atomic_capture_contract.py',
     'tests/review/test_sol61_m17_saved_physics_v2.py',
     'tests/review/test_actual_compile_capture.py',
     'tests/review/test_sol61_actual_compile_capture_independent.py',
     'tests/review/test_sol61_cubature_capture_raccord_independent.py',
    }
    added.update({'tests/review/test_sol61_model_policy_program_boundary.py','tests/review/test_sol61_joint_wave_migration.py','tests/review/test_sol61_program_policy_independent.py'})
    files=sorted(basefiles|added)
    assert all((ROOT/path).is_file() for path in files)
    args=['--noconftest','-p','no:cacheprovider','-o',f'pythonpath={ROOT}/python {ROOT}',*files,
          '-k','not failed_program_compile_leaves','-q','--tb=short',f'--junitxml={OUT}/pytest.xml',f'--basetemp={OUT}/pytest-tmp']
    save('command.json',{'executable':sys.executable,'environment':{'PYTHONPATH':'unset','PYTHONDONTWRITEBYTECODE':'1'},'cwd':str(ROOT),'pytest_args':args,'base_files':sorted(basefiles),'added_files':sorted(added),'deduplicated_files':files,'only_exclusion':{'expression':'failed_program_compile_leaves','reason':'requires selected Native platform authority; previously documented C25 node'}})
    before=snapshot();save('before.json',before)
    assert not before['base_drift'],before['base_drift']
    sys.path[:0]=[str(ROOT/'python'),str(ROOT)]
    import pops
    import importlib.machinery
    def classified_modules():
        result=[]
        for name,module in list(sys.modules.items()):
            if not name.rsplit('.',1)[-1].startswith('_pops'):continue
            spec=getattr(module,'__spec__',None);origin=getattr(spec,'origin',None)
            loader=getattr(spec,'loader',None)
            extension=isinstance(loader,importlib.machinery.ExtensionFileLoader) or bool(origin and any(str(origin).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES))
            row={'name':name,'origin':origin,'loader':type(loader).__name__,'extension':extension}
            if origin and Path(origin).is_file():row['sha256']=digest(Path(origin))
            result.append(row)
        return result
    native=lambda:sorted(row['name'] for row in classified_modules() if row['extension'] or row['name'].rsplit('.',1)[-1]=='_pops')
    assert Path(pops.__file__).resolve()==ROOT/'python/pops/__init__.py'
    assert not native()
    save('source-identity.json',{'package_file':pops.__file__,'python':sys.executable,'native_modules_before':native(),'head':before['head'],'scope':'Source and bounded host only, no PoPS Native/JIT simulation'})
    import pytest
    start=time.monotonic();code=int(pytest.main(args,plugins=[ImmediateReporter()]));elapsed=time.monotonic()-start
    after=snapshot();save('after.json',after)
    native_after=native()
    save('module-classification-after.json',classified_modules())
    freeze=before['production_sha256']==after['production_sha256'] and not after['base_drift']
    xml=ET.parse(OUT/'pytest.xml');cases=list(xml.iter('testcase'))
    keys=[(case.get('classname'),case.get('name')) for case in cases]
    totals={'tests':len(cases),'failures':sum(case.find('failure') is not None for case in cases),'errors':sum(case.find('error') is not None for case in cases),'skips':sum(case.find('skipped') is not None for case in cases)}
    save('result.json',{'contract':'sol61.integrated-source-host@1','base':BASE,'exit_code':code,'elapsed_seconds':elapsed,'totals':totals,'unique_cases':len(set(keys)),'duplicate_cases':len(keys)-len(set(keys)),'source_freeze_exact':freeze,'native_modules_after':native_after,'native_qualified':False,'ci_qualified':False,'root_seal':False,'case_manifest':[{'classname':c.get('classname'),'name':c.get('name'),'seconds':c.get('time')} for c in cases]})
    assert freeze,'production drift from d584'
    assert not native_after,native_after
    assert len(keys)==len(set(keys)),'duplicate tests'
    sys.exit(code)

if __name__ == "__main__":
    main()
