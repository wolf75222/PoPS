"""Actual public Source authoring/emission; no Native or JIT execution."""
from pathlib import Path
import sys,numpy as np,pops
from tests.python.support import m16_second_state_native_case as fixture

def test_two_distinct_states_second_domain_resolves_and_emits():
    from pops._balance_due_contract import BalanceDueContract
    from pops.codegen._shared_interface_evidence import _issue_shared_interface_codegen_evidence
    from pops.codegen.program_graph_lowering import _emit_resolved_program_graph
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    assert Path(pops.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2]/'python')
    assert 'pops._pops' not in sys.modules
    case,layout,subjects,model,operator=fixture.build()
    assert tuple(model.module.state_spaces())==('earlier_state','selected_state')
    assert operator.signature.output.domain.name=='selected_state'
    validated=pops.validate(case);resolved=pops.resolve(validated,layout=layout)
    assert tuple(block.name for block in resolved.blocks)==fixture.PARTITION
    assert len(resolved.initial_condition_plan.bindings)==2
    for subject in subjects.values():assert validated.resolve(subject) is not None
    detached=detach_compiled_program(resolved.time)
    cpp=_emit_resolved_program_graph(detached.to_graph(),lowering_program=detached,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),field_plans=resolved.field_plans,
        balance_due_contract=BalanceDueContract.from_consumer_graph(resolved.consumer_graph),
        shared_interface_codegen_evidence=_issue_shared_interface_codegen_evidence(resolved))
    assert 'affine_velocity_push_forward<5>' in cpp
    value=next(v for v in detached._values if v.op=='affine_moment_update')
    assert value.block.name=='population'
    authority=ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    assert tuple(authority.model_for_block(value.block).cons_names)==fixture.a.NAMES
    import re
    source_view=re.search(r'old_moments\[0\] = (u\d+)A\(index,',cpp).group(1)
    index=detached._block_indices()[value.block]
    assert f'& {source_view} = ctx.state({index});' in cpp
    arrays=fixture.initials();assert arrays['spectator'].shape==(2,4,4) and arrays['population'].shape==(21,4,4)
    assert np.isfinite(arrays['population']).all() and np.signbit(arrays['spectator'][1,0,0])


def test_exact_full_partition_selector_refuses_subset_or_foreign_layout(monkeypatch):
    import pytest
    from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture,CanonicalValue
    from tests.python.support.m16_explicit_native_capture import select_partition_program
    platform=CanonicalValue('SOURCE-only-no-platform')
    monkeypatch.setattr('pops.codegen._compiled_artifact._common_platform_manifest',lambda **kwargs:platform)
    artifact=artifact_fixture(target='amr_system',block_names=fixture.PARTITION)
    row=select_partition_program(artifact,artifact.plan,fixture.PARTITION)
    assert set(row.block_names)==set(fixture.PARTITION) and row.program is artifact.program
    for partition in (('population',),('spectator','foreign'),('population','population')):
        with pytest.raises(ValueError):select_partition_program(artifact,artifact.plan,partition)
