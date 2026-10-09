"""Typed facade matrix scope follows explicit State, including a smaller first State."""
from pathlib import Path
import sys
import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.model import LocalLinearOperator
from pops.moments import CartesianMonomialBasis
from pops.time import FixedDt
from pops.codegen.module_lowering import lower_and_validate


def model_and_states():
    frame=Rectangle('operator_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('unequal_state_operators',frame=frame)
    first=model.species('earlier',state=('other_a','other_b'))
    selected=model.species('selected',state=tuple('entry_'+str(i) for i in range(6)))
    return model,first,selected,frame


def test_second_state_matrix_and_registration_use_its_own_exact_space():
    model,first,selected,_=model_and_states()
    matrix=[[0.]*6 for _ in range(6)];matrix[1][3],matrix[3][1]=2.,-2.
    math=model.local_linear_operator('selected_skew',on=selected,matrix=matrix)
    handle=model.operator('selected_skew',returns=math)
    op=model.module.operator_registry().get(handle.name)
    assert handle.signature.output==LocalLinearOperator(selected.space,selected.space)
    assert op.signature.output==handle.signature.output
    assert op.signature.inputs==()
    assert tuple(model._dsl._m._linear_sources)==() # no first-State DSL publication
    earlier=model.operator('earlier_identity',returns=model.local_linear_operator(
        'earlier_identity',on=first,matrix=((1.,0.),(0.,1.))))
    assert earlier.signature.output==LocalLinearOperator(first.space,first.space)


def test_facade_selected_second_state_resolves_and_emits_whole_program():
    from pops._balance_due_contract import BalanceDueContract
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    assert Path(pops.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[4]/'python')
    assert 'pops._pops' not in sys.modules
    model,first,selected,frame=model_and_states()
    matrix=[[0.]*6 for _ in range(6)];matrix[1][3],matrix[3][1]=2.,-2.
    operator=model.operator('selected_skew',returns=model.local_linear_operator('selected_skew',on=selected,matrix=matrix))
    case=pops.Case('unequal_case')
    earlier=case.block('earlier',model,states=(first,));other=case.block('selected',model,states=(selected,))
    program=pops.Program('unequal_states');a=program.state(earlier[first]);q=program.state(other[selected])
    indices=tuple((p,r) for r in range(3) for p in range(3-r))
    pushed=program.affine_moment_update(q.n,q.n,linear_operator=operator,theta_dt=program.dt/2,
        basis=CartesianMonomialBasis(indices),components=dict(zip(indices,selected.components)))
    program.commit(q.next,program.value('selected_next',pushed,at=q.next.point))
    program.commit(a.next,program.value('earlier_next',a.n,at=a.next.point))
    program.step_strategy(FixedDt(.125));case.program(program)
    resolved=pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(frame=frame,cells=(4,4),periodic=PeriodicAxes(frame.axes))))
    detached=detach_compiled_program(resolved.time);authority=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    cpp=_emit_resolved_program_graph(detached.to_graph(),lowering_program=detached,model_graph=authority,
        field_plans=resolved.field_plans,balance_due_contract=BalanceDueContract.from_consumer_graph(resolved.consumer_graph),
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(resolved))
    assert 'affine_velocity_push_forward<2>' in cpp
    assert tuple(authority.model_for_block('selected').cons_names)==selected.components
    assert tuple(authority.model_for_block('earlier').cons_names)==first.components


@pytest.mark.parametrize('wrong',('missing','foreign','wrong_size','undeclared_field'))
def test_invalid_authority_refuses_without_registry_publication(wrong):
    model,first,selected,_=model_and_states()
    before=model.module.module_hash()
    with pytest.raises((TypeError,ValueError)):
        if wrong=='missing':model.local_linear_operator('bad',matrix=((1.,0.),(0.,1.)))
        elif wrong=='foreign':
            foreign,_,state,_=model_and_states()
            model.local_linear_operator('bad',on=state,matrix=[[0.]*6 for _ in range(6)])
        elif wrong=='wrong_size':model.local_linear_operator('bad',on=selected,matrix=((1.,0.),(0.,1.)))
        else:model.operator('bad',returns=model.local_linear_operator('bad',on=first,matrix=((1.,0.),(0.,1.))),inputs=('foreign_fields',))
    assert model.module.module_hash()==before


def test_declared_field_input_keeps_real_typed_dependency():
    model,_,selected,_=model_and_states()
    coefficient=model.field('coefficient',components=('angular',))
    angular,=model.module.field_symbols(model.module.field_spaces()['coefficient'])
    matrix=[[0.]*6 for _ in range(6)];matrix[1][3],matrix[3][1]=angular,-angular
    handle=model.operator('field_skew',returns=model.local_linear_operator('field_skew',on=selected,matrix=matrix),inputs=('coefficient',))
    assert handle.signature.inputs==(model.module.field_spaces()['coefficient'],)
    assert handle.signature.output==LocalLinearOperator(selected.space,selected.space)


def test_mono_optional_on_preserves_canonical_module_identity():
    def declaration(explicit):
        model=pops.Model('mono_same')
        state=model.state('U',components=('a','b'))
        math=model.local_linear_operator('skew',on=state if explicit else None,matrix=((0.,2.),(-2.,0.)))
        handle=model.operator('skew',returns=math)
        return model.module.module_hash(),handle.signature
    assert declaration(False)==declaration(True)
