"""Source routing checks; these do not execute a Native checkpoint."""
import ast
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]

def test_actual_ci_partition_has_three_separate_native_process_groups():
    path=ROOT/'scripts/ci_python_dimensions.py'
    spec=importlib.util.spec_from_file_location('dimension_router',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    paths=[f'tests/python/integration/runtime/test_uniform_cp9_dim{d}_runtime.py' for d in (1,2,3)]
    groups=module.partition(paths,json.loads((ROOT/'tests/python/native_dimensions.json').read_text()))
    assert groups=={d:[paths[d-1]] for d in (1,2,3)}

@pytest.mark.parametrize('dimension',(1,2,3))
def test_entry_refuses_wrong_launcher_before_native_or_compilation(dimension,monkeypatch):
    import os
    path=ROOT/f'tests/python/integration/runtime/test_uniform_cp9_dim{dimension}_runtime.py'
    tree=ast.parse(path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    calls=[]
    namespace={'os':os,'run_installed_dimensional_cp9_two_states':lambda *args:calls.append(args)}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    monkeypatch.setenv('POPS_NATIVE_DIM',str(dimension%3+1))
    with pytest.raises(RuntimeError,match='exact POPS_NATIVE_DIM'):
        namespace[function.name](None,None,None,None,None)
    assert calls==[]

@pytest.mark.parametrize('dimension',(1,2,3))
def test_real_public_two_block_case_resolves_in_each_declared_dimension(dimension):
    import pops
    from tests.python.support.dimensional_uniform_cp9_runtime import build
    case,layout,initial,dt=build(dimension)
    resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    assert set(initial)=={'narrow','wide'}
    assert initial['narrow'].shape==(2,)+(4,)*dimension
    assert initial['wide'].shape==(3,)+(4,)*dimension
    assert resolved.initial_condition_plan.bindings
    assert dt==1/64

def test_ci_build_downloads_dim3_from_the_same_build_matrix():
    source=(ROOT/'.github/workflows/ci.yml').read_text()
    assert 'dimension: ${{ fromJSON(needs.set-mode.outputs.python_dimensions) }}' in source
    assert "if: steps.test-plan.outputs.dim3_count != '0'" in source
    assert 'name: gate-python-build-kokkos-py-dim3' in source
    assert 'path: .pops-ci/python-packages/dim3' in source

@pytest.mark.parametrize('mutation',('foreign-path','changed-source','extra-source'))
def test_ci_origin_source_authority_refuses_foreign_or_poisoned_artifacts(tmp_path,mutation):
    from tests.python.support.native_package_test_authority import authenticate_ci_source_files
    source=tmp_path/'python/pops';source.mkdir(parents=True);(source/'__init__.py').write_text('# genuine test source\n')
    package=tmp_path/'.pops-ci/python-packages/dim3';(package/'pops').mkdir(parents=True)
    (package/'pops/__init__.py').write_bytes((source/'__init__.py').read_bytes())
    assert authenticate_ci_source_files(package,3,tmp_path)
    if mutation=='foreign-path':package=tmp_path/'python'
    elif mutation=='changed-source':(package/'pops/__init__.py').write_text('# poisoned\n')
    else:(package/'pops/foreign.py').write_text('# extra\n')
    with pytest.raises(RuntimeError):authenticate_ci_source_files(package,3,tmp_path)

@pytest.mark.parametrize('header', (None,'foreign-signature'))
def test_ci_origin_refuses_missing_or_foreign_baked_header_authority(tmp_path,monkeypatch,header):
    # Source-only authority seam: no extension is loaded or Native fact claimed.
    import types
    from tests.python.support import native_package_test_authority as authority
    import scripts.verify_installed_native as verifier
    import pops.codegen.abi as abi
    import pops.codegen.toolchain as toolchain
    source=tmp_path/'python/pops';source.mkdir(parents=True);(source/'__init__.py').write_text('# source\n')
    package=tmp_path/'.pops-ci/python-packages/dim3';(package/'pops/_native/dim3').mkdir(parents=True)
    (package/'pops/__init__.py').write_bytes((source/'__init__.py').read_bytes())
    extension=package/'pops/_native/dim3/SourceOnly.bin';extension.write_bytes(b'not a native extension')
    calls=[]
    monkeypatch.setattr(authority,'ROOT',tmp_path)
    monkeypatch.setenv('POPS_CI_NATIVE_PACKAGE',str(package))
    monkeypatch.setattr(verifier,'verify_installed_native',lambda **kwargs:(calls.append(kwargs) or extension))
    monkeypatch.setattr(abi,'module_header_signature',lambda:header)
    monkeypatch.setattr(toolchain,'pops_header_signature',lambda path:'authentic-signature')
    with pytest.raises(RuntimeError,match='header signature'):
        authority.authenticate_native_test_package(types.SimpleNamespace(__file__=str(package/'pops/__init__.py')),object(),3)
    assert calls[0]['expect_dimension']==3 and calls[0]['expect_mpi'] is False
