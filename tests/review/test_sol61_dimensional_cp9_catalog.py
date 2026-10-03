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
