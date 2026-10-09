import importlib.abc,importlib.machinery,importlib.util,json,sys
from pathlib import Path
ROOT=Path('/Users/romaindespoulain/dev/tmp/PoPS-api040-integrated-main-20261004')
OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'python'),str(ROOT),str(ROOT/'examples/migration/scientific')];sys.dont_write_bytecode=True
blocked=[]
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='_pops' or fullname.endswith('._pops') or fullname.startswith('pops._native.dim'):
            blocked.append(fullname)
            raise ModuleNotFoundError('Source-only geometry validation forbids Native',name=fullname)
sys.meta_path.insert(0,NoNative())
create=importlib.machinery.ExtensionFileLoader.create_module
def refuse(loader,spec):
    if '/pops/_native/' in str(spec.origin) or '_pops' in str(spec.origin):
        raise ModuleNotFoundError('Source-only geometry validation forbids Native',name=spec.name)
    return create(loader,spec)
importlib.machinery.ExtensionFileLoader.create_module=refuse
import pops
from scripts.ci_python_dimensions import partition
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
m05=load('source_m05_geometry',ROOT/'examples/migration/scientific/api040_m05_periodic_shear.py')
m10=load('source_m10_geometry',ROOT/'tests/python/integration/runtime/test_m10_self_consistent_sg.py')
rows=[]
for name,build in [('M05',lambda:m05.build_case(16)),('M10',lambda:m10._case(16))]:
    case,layout,*_=build();plan=pops.resolve(pops.validate(case),layout=layout)
    assert plan.resolved_dimension==1
    rows.append({'kind':name,'genuine_author_validate_resolve_dimension':plan.resolved_dimension,
                 'case_id':case.name,'Native_run':False})
contract=json.loads((ROOT/'tests/python/native_dimensions.json').read_text())
prior=json.loads(__import__('subprocess').check_output(['rtk','proxy','git','-C',str(ROOT),'show','HEAD:tests/python/native_dimensions.json']))
added={k:v for k,v in contract['files'].items() if k not in prior['files']}
assert added=={'tests/python/integration/runtime/test_api040_m05_periodic_shear_runtime.py':1,
              'tests/python/integration/runtime/test_m10_self_consistent_sg.py':1}
assert all(contract['files'][k]==v for k,v in prior['files'].items()) and contract['default']==prior['default']==2
all_paths=[*contract['files'],'tests/python/integration/runtime/test_public_tensor_diffusion.py',
           'tests/python/integration/mpi/test_dsl_compile_cache.py']
groups=partition(all_paths,contract)
assert set(sum(groups.values(),[]))==set(all_paths) and len(sum(groups.values(),[]))==len(all_paths)
assert groups[2]==all_paths[-2:]
assert not[n for n,m in sys.modules.items() if isinstance(getattr(getattr(m,'__spec__',None),'loader',None),
                                                        importlib.machinery.ExtensionFileLoader) and '_pops' in n]
result={'geometries':rows,'added_overrides':added,'old_routes_preserved':True,'partition':groups,
        'Native_loaded':False,'Native_blocked_attempts':blocked,
        'scope':'Actual Source author/validate/resolve only, no compilation, binding or Native/runtime execution'}
(OUT/'geometry-route-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
