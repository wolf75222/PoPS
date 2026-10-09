"""Independent Source orchestration only; no Native artifact simulated as evidence."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('count',(0,1,3))
def test_exact_publisher_loop_visits_every_block_or_refuses_empty(count):
    # Execute the exact author loop with explicit Source-only protocol spies.
    tree=ast.parse((ROOT/'tests/python/support/atomic_native_capture.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compile_with_model_tus')
    body=next(n for n in fn.body if isinstance(n,ast.With)).body
    start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rows')
    stop=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='proof')
    code=compile(ast.fix_missing_locations(ast.Module(body=body[start:stop],type_ignores=[])),'actual-publisher-loop','exec')
    seen=[]
    env={'artifact':SimpleNamespace(blocks=[SimpleNamespace(name=f'b{i}',model=SimpleNamespace(so_path=f'p{i}')) for i in range(count)]),
         'capture':SimpleNamespace(require_binary=lambda p:seen.append(p) or {'output':p}), 'sha':lambda p:'hash-'+p}
    if count==0:
        with pytest.raises(ValueError,match='no actual model'):exec(code,env)
    else:
        exec(code,env);assert seen==[f'p{i}' for i in range(count)]
        assert [r['block'] for r in env['rows']]==[f'b{i}' for i in range(count)]


def test_peer_route_never_enters_publisher_observer():
    for filename in ('test_atomic_cubature_public_path_runtime.py','test_atomic_cubature_parametric_refusal_runtime.py'):
        tree=ast.parse((ROOT/'tests/python/integration/runtime'/filename).read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compile_artifact')
        mod=ast.Module(body=[fn],type_ignores=[])
        calls=[];env={'world':SimpleNamespace(rank=1),'directory':Path('/unused'),
            'pops':SimpleNamespace(compile=lambda plan:calls.append(('load',plan)) or 'loaded'),
            'compile_with_model_tus':lambda *args:(_ for _ in ()).throw(AssertionError('peer forged TU'))}
        exec(compile(ast.fix_missing_locations(mod),'actual-peer-callback','exec'),env)
        assert env['compile_artifact']('plan')=='loaded' and calls==[('load','plan')]
