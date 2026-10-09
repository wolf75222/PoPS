"""Source preparation only; no Native load/compile or scientific receipt."""
import json,sys
from pathlib import Path
import numpy as np
import pytest
import pops
from tests.python.support import m17_fan_li_public_native_case as s
from tests.python.integration.runtime import test_fan_li15_full_eight_step_public_runtime as f


@pytest.mark.parametrize('reverse',(False,True))
def test_original_uniform_public_require_resolves_and_emits(reverse):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.layouts import Uniform
    module=s.original();assert (module.N,module.DT,module.STEPS)==(16,1e-4,8)
    order=tuple(reversed(module.INDICES)) if reverse else module.INDICES
    case,layout=s.make_case(order);assert type(layout) is Uniform
    plan=pops.resolve(pops.validate(case),layout=layout,compile_options={'model_source_policy':'require'})
    selected,=plan.blocks
    assert selected.name=='gas'
    model,_=lower_and_validate(selected.model,state_space=selected.state_spaces[0],resolved_operations=selected.resolved_operations,numerics=selected.numerics)
    assert len(model._m.cons_names)==15
    assert model._m._path_conservative['identity'].startswith('pops.numerics.normalized-polynomial-path.v1:sha256:')
    cpp=emit_cpp_program(plan.time,model=ProgramModelGraph.from_resolved_blocks(plan.blocks),target='system')
    # Diagnostic labels need not survive lowering; require the genuine Uniform entry.
    assert 'extern "C" void pops_install_program(' in cpp
    assert np.array_equal(s.canonical(s.initial(order),order),s.initial(module.INDICES))


def test_original_eight_reference_not_two_and_dop853_guard():
    module=s.original();q=s.canonical(s.initial(module.INDICES),module.INDICES)[:,0,:]
    original=q.copy()
    two=None
    for n in range(1,9):
        q=s.ssprk2(q)
        if n==2:two=q.copy()
    assert np.max(np.abs(q-two))>module.CRITERIA['oracle_max_error']
    oracle=sys.modules['api040_m17_oracle']
    dop=oracle.solve_reference(original,8*s.DT)
    assert np.max(np.abs(q-dop))<module.CRITERIA['oracle_max_error']
    for k,index in enumerate(module.INDICES):
        if sum(index)<4:assert abs(q[k].mean()-original[k].mean())<module.CRITERIA['conserved_inventory']


def test_phase_storage_is_json_valid_and_no_fabricated_carrier(tmp_path):
    module=s.original();q=s.initial(module.INDICES)
    phases=('initial',)+tuple('accepted'+str(i) for i in range(1,9))
    for i,phase in enumerate(phases):f.save_phase(tmp_path,phase,(q,(i*s.DT,i)))
    assert len(list(tmp_path.glob('*.npy')))==9 and not list(tmp_path.glob('*.carriers'))
    for i,phase in enumerate(phases):
        data=json.loads((tmp_path/(phase+'.capture.json')).read_text())
        assert data['schema']==f.SCHEMA and data['carrier_capture']['status']=='unavailable'
        assert json.loads((tmp_path/(phase+'.clock.json')).read_text())==[i*s.DT,i]
        assert np.load(tmp_path/(phase+'.npy'),allow_pickle=False).tobytes()==q.tobytes()


def test_preparation_imports_only_this_source_checkout():
    checkout=Path(__file__).resolve().parents[2]
    assert Path(pops.__file__).resolve().is_relative_to(checkout/'python')
    assert not any(name=='_pops' or name.startswith('pops._pops') for name in sys.modules)


@pytest.mark.parametrize('retained',(None,'',b'foreign'))
def test_program_source_capture_refuses_regeneration_before_any_dump(tmp_path,retained):
    from tests.python.support.m16_explicit_native_capture import dump_retained_program
    class AdvancedHandle:
        _generated_cpp=retained
        calls=[]
        def dump_cpp(self,path):
            self.calls.append('regenerated')
            path.write_text('synthetic regenerated CPP')
        def dump_ir(self,path):
            self.calls.append('ir')
    handle=AdvancedHandle()
    with pytest.raises(ValueError,match='regeneration forbidden'):
        dump_retained_program(handle,tmp_path)
    assert handle.calls==[] and not tuple(tmp_path.iterdir())


def test_fixture_provenance_uses_retained_guard_not_direct_dump():
    import ast
    tree=ast.parse(Path(f.__file__).read_text())
    fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='retain_sources')
    calls=[node for node in ast.walk(fn) if isinstance(node,ast.Call)]
    assert any(isinstance(node.func,ast.Name) and node.func.id=='dump_retained_program' for node in calls)
    assert not any(isinstance(node.func,ast.Attribute) and node.func.attr=='dump_cpp'
                   and isinstance(node.func.value,ast.Attribute) and node.func.value.attr=='program' for node in calls)
