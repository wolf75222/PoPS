"""Existing focused Source-harness pattern; actual Main dirty diff, Native blocked."""
import hashlib,importlib.abc,importlib.machinery,json,os,sys,time,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004')
OUT=Path(__file__).parent
assert 'PYTHONPATH' not in os.environ and 'PYTHONOPTIMIZE' not in os.environ
assert Path(sys.prefix).resolve()==Path('/Users/romaindespoulain/miniforge3/envs/pops').resolve()
attempts=[]
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='_pops' or fullname.endswith('._pops') or fullname.startswith('pops._native.dim'):
            attempts.append(fullname)
            raise ModuleNotFoundError('Source-only validation explicitly blocks Native: '+fullname,name=fullname)
sys.meta_path.insert(0,DenyNative())
create=importlib.machinery.ExtensionFileLoader.create_module
def no_native(loader,spec):
    if '/pops/_native/' in str(spec.origin) or '_pops' in str(spec.origin):
        attempts.append(str(spec.origin))
        raise ModuleNotFoundError('Source-only validation explicitly blocks Native',name=spec.name)
    return create(loader,spec)
importlib.machinery.ExtensionFileLoader.create_module=no_native
sys.path[:0]=[str(ROOT/'python'),str(ROOT)]
sys.dont_write_bytecode=True
os.chdir(ROOT)
import pops
assert Path(pops.__file__).resolve().is_relative_to(ROOT/'python')
import pytest
selected=[
 'tests/python/architecture/test_ci_python_dimensions.py',
 'tests/python/architecture/test_native_dimension_build_contract.py',
 'tests/python/architecture/test_ci_plan.py',
 'tests/python/architecture/test_ci_python_module_objects.py',
 'tests/python/architecture/test_installed_native_verifier.py',
 'tests/python/architecture/test_native_variant_manifest.py::test_ci_consumes_only_authenticated_explicit_native_variants',
 'tests/python/architecture/test_native_variant_manifest.py::test_ctest_smokes_and_python_suites_preserve_the_exact_native_dimension',
 'tests/python/architecture/test_native_variant_manifest.py::test_ctest_python_mpi_projection_matches_the_manifest_and_dim2_contract',
 'tests/python/architecture/test_native_variant_manifest.py::test_wheel_proof_accepts_an_explicit_fat_set_and_rejects_a_hidden_subset']
args=['-q','-ra','--strict-markers','--strict-config','--confcutdir='+str(ROOT),
      '-o','pythonpath=','-p','no:cacheprovider','--basetemp='+str(OUT/'pytest-tmp'),
      '--junitxml='+str(OUT/'pytest.xml'),*selected]
start=time.monotonic(); code=pytest.main(args)
loaded=[n for n,m in sys.modules.items() if (n=='_pops' or n.endswith('._pops')) and
        isinstance(getattr(getattr(m,'__spec__',None),'loader',None),importlib.machinery.ExtensionFileLoader)]
assert not loaded,loaded
tree=ET.parse(OUT/'pytest.xml').getroot();suites=list(tree.iter('testsuite'))
counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
failures=[{'name':c.get('name'),'class':c.get('classname'),'text':f.text} for c in tree.iter('testcase')
          for f in c if f.tag in ('failure','error','skipped')]
result={'exit':int(code),'counts':counts,'seconds':time.monotonic()-start,'args':args,
        'selected':selected,'Native_blocked_attempts':attempts,'Native_loaded':loaded,
        'package':pops.__file__,'python':sys.executable,'failures':failures,
        'scope':'Focused Main working-tree Source validation; mocked wheel/selector/subprocess fixtures, no real installation/Native/MPI/GPU/GitHub CI. No global903/904 inference.'}
(OUT/'focused-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'exit':int(code),'counts':counts,'Native_loaded':loaded},indent=2))
raise SystemExit(code)
