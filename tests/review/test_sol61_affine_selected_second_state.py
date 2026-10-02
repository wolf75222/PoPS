"""SOURCE_ONLY two genuine States: resolved selected second-State lowering."""
from pathlib import Path
import sys
import re
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import Const, ddt, div
from pops.mesh import CartesianGrid,PeriodicAxes
from pops.moments import CartesianMonomialBasis
from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt
from pops.codegen.module_lowering import lower_and_validate


def build(*, operator_route="module"):
    from pops.model import LocalLinearOperator, Signature
    frame=Rectangle('two_state_box',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('two_independent_states',frame=frame)
    first=model.species('spectator',state=tuple('spectator_'+str(i) for i in range(6)))
    names=('slot_c','slot_f','slot_a','slot_e','slot_b','slot_d')
    second=model.species('selected',state=names)
    indices=((0,1),(2,0),(0,0),(1,1),(1,0),(0,2))
    basis=CartesianMonomialBasis(indices)
    binding=dict(zip(indices,names[::-1],strict=True))
    blocks=[];case=pops.Case('two_state_case')
    for state,label in ((first,'first'),(second,'second')):
        flux=model.flux('zero_'+label,state=state,frame=frame,
            components={axis:tuple(0*v for v in state) for axis in frame.axes},
            waves={axis:tuple(Const(0.) for _ in state.components) for axis in frame.axes})
        rate=model.rate('balance_'+label,equation=ddt(state)==-div(flux))
        plan=DiscretizationPlan();plan.rates.add(rate,FiniteVolume(flux=flux,
            variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
        block=case.block(label,model,states=(state,));case.numerics(plan,block=block);blocks.append(block)
    matrix=[[0.]*6 for _ in range(6)]
    x,y=names.index(binding[(1,0)]),names.index(binding[(0,1)])
    matrix[x][y],matrix[y][x]=2.,-2.
    if operator_route=='facade':
        operator=model.operator('selected_skew',returns=model.local_linear_operator('selected_skew',on=second,matrix=tuple(map(tuple,matrix))))
    else:
        module=model.module
        space=module.state_spaces()[second.name]
        module.operator(name='selected_skew',kind='local_linear_operator',
            signature=Signature((),LocalLinearOperator(space,space)),expr=[[Const(v) for v in row] for row in matrix])
        operator=module.operator_handle('selected_skew')
    program=pops.Program('two_states')
    spectator=program.state(blocks[0][first]);q=program.state(blocks[1][second])
    source=program.affine_moment_update(q.n,q.n,linear_operator=operator,
        theta_dt=program.dt/2,basis=basis,components=binding)
    program.commit(spectator.next,program.value('first_unchanged',spectator.n,at=spectator.next.point))
    program.commit(q.next,program.value('second_accepted',source,at=q.next.point))
    program.step_strategy(FixedDt(.125));case.program(program)
    return model,program,case,Uniform(CartesianGrid(frame=frame,cells=(4,4),periodic=PeriodicAxes(frame.axes))),binding,names


def test_authentic_resolved_second_state_emits_without_first_state_selection():
    from pops._balance_due_contract import BalanceDueContract
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    assert Path(pops.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2]/'python')
    assert 'pops._pops' not in sys.modules
    model,program,case,layout,binding,names=build()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    second=next(block for block in resolved.blocks if block.name=='second')
    assert len(model.module.state_spaces())==2
    selected,module=lower_and_validate(second.model,state_space=second.state_spaces[0],resolved_operations=second.resolved_operations)
    assert tuple(selected.cons_names)==names
    detached=detach_compiled_program(resolved.time)
    authority=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    cpp=_emit_resolved_program_graph(detached.to_graph(),lowering_program=detached,
        model_graph=authority,field_plans=resolved.field_plans,
        balance_due_contract=BalanceDueContract.from_consumer_graph(resolved.consumer_graph),
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(resolved))
    assert 'affine_velocity_push_forward<2>' in cpp
    source_view=re.search(r'old_moments\[0\] = (u\d+)A\(index,',cpp).group(1)
    affine=next(value for value in detached._values if value.op=='affine_moment_update')
    assert affine.block.name=='second'
    block_index=detached._block_indices()[affine.block]
    assert f'& {source_view} = ctx.state({block_index});' in cpp
    assert tuple(authority.model_for_block(affine.block).cons_names)==names
    for i,index in enumerate((p,q) for q in range(3) for p in range(3-q)):
        slot=names.index(binding[index])
        assert re.search(r'old_moments\['+str(i)+r'\] = u\d+A\(index, '+str(slot)+r'\);',cpp)
    assert detached._ir_hash()==resolved.time._ir_hash()


def test_facade_multi_state_local_operator_now_uses_typed_registry():
    model,program,*_=build(operator_route='facade')
    signature=model.module.operator_registry().get('selected_skew').signature
    assert signature.output.domain.name=='selected'
    assert signature.output.range==signature.output.domain
