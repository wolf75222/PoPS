"""Public Source authoring only; Native test is collected separately."""
import pops
from tests.python.integration.runtime.test_uniform_checkpoint_provisional_effect import build


def test_public_checkpoint_effect_resolves_on_same_authored_clock():
    case,layout,initial,dt=build()
    resolved=pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    assert set(initial)=={'early_transport','late_transport'} and dt>0
    assert resolved.consumer_graph.nodes
    row,=resolved.consumer_graph.nodes
    assert row.kind.value=='checkpoint' and row.target_uri=='transactional-cp9'


def test_actual_runtime_clock_methods_used_by_capture_expression():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from pops.runtime._runtime_instance import RuntimeInstance
    runtime=object.__new__(RuntimeInstance)
    # Explicit Source-only executor seam, real public RuntimeInstance methods.
    object.__setattr__(runtime,'_executor',SimpleNamespace(time=lambda:0.125,macro_step=lambda:7))
    source=Path(__file__).resolve().parents[2]/'tests/python/integration/runtime/test_uniform_checkpoint_provisional_effect.py'
    tree=ast.parse(source.read_text())
    outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_installed_'))
    capture=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='capture')
    clock=next(n.value for n in capture.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='clock' for t in n.targets))
    observed=eval(compile(ast.Expression(clock),str(source),'eval'),{'runtime':runtime})
    assert observed==(0.125,7)
    assert type(observed[0]) is float and type(observed[1]) is int
