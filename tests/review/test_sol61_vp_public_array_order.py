"""Source array-protocol checks; no Native engine or scientific receipt."""
import ast
import subprocess
from pathlib import Path
import numpy as np
from pops.runtime._bind_validation import _check_one_initial_state
from tests.python.support.m19_vlasov_poisson_oracle import initial, step

ROOT=Path(__file__).resolve().parents[2]
FIXTURE='tests/python/integration/runtime/test_m19_vlasov_poisson_runtime.py'

def arrays(text):
    tree=ast.parse(text)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_installed_two_stage_vlasov_poisson')
    assignments=[n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('f','initial_state') for t in n.targets)]
    scope={'np':np,'initial':initial}
    exec(compile(ast.fix_missing_locations(ast.Module(body=assignments,type_ignores=[])),FIXTURE,'exec'),scope)
    return scope['initial_state']

def refused(values):
    lines=[]
    for name,components,cells in (('a_spectator',2,(4,8)),('m_number',1,(8,1)),('z_phase',1,(4,8))):
        _check_one_initial_state(lines,name,values[name],{'components':components},cells,1,{'float64'})
    return lines

def test_actual_parent_bind_arrays_reproduce_three_refusals():
    old=subprocess.check_output(['git','show','ef4c797ce019d7baa971158701c098e7b0ac0fc8:'+FIXTURE],cwd=ROOT,text=True)
    assert len(refused(arrays(old)))==3

def test_actual_fixture_arrays_obey_public_nonsquare_protocol():
    values=arrays((ROOT/FIXTURE).read_text())
    assert refused(values)==[]
    np.testing.assert_array_equal(values['z_phase'][0].T,initial())
    np.testing.assert_array_equal(values['m_number'][0,0],.5*initial().sum(axis=0))
    assert values['a_spectator'].shape==(2,8,4)

def test_actual_step_assertions_use_native_order_without_changing_oracle():
    tree=ast.parse((ROOT/FIXTURE).read_text())
    comparisons=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='assert_allclose']
    assert len(comparisons)==2
    expected,stages=step(initial())
    scope={'expected':expected,'stages':stages}
    refs=[eval(compile(ast.Expression(n.args[1]),FIXTURE,'eval'),scope) for n in comparisons]
    assert [r.shape for r in refs]==[(1,8,4),(1,1,8)]
    np.testing.assert_array_equal(refs[0][0].T,expected)
    np.testing.assert_array_equal(refs[1][0,0],stages['predictor_rho'])
