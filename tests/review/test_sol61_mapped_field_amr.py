"""AMR realization admission: Source only, no Native inference."""
import pytest
from pops.runtime._mapped_field_capability import require_mapped_field_native_facts


def test_amr_requires_distinct_capability_and_abi10():
    for facts in ({'abi_version':9,'mapped_consumed_field_output':True,
                   'mapped_consumed_field_output_amr':True},
                  {'abi_version':10,'mapped_consumed_field_output':True}):
        with pytest.raises(RuntimeError):
            require_mapped_field_native_facts(adaptive=True, capability_reader=lambda _:facts)
    require_mapped_field_native_facts(adaptive=True, capability_reader=lambda _:{
        'abi_version':10,'mapped_consumed_field_output':True,'mapped_consumed_field_output_amr':True})


def test_public_amr_route_resolves_actual_hierarchy_ports(tmp_path):
    from tests.review.test_sol61_mapped_field_route import route
    resolved = route(tmp_path, adaptive=True)
    assert len(resolved.layout_plan.layouts) == 2


def test_amr_route_emits_private_scalar_then_publication(tmp_path):
    from tests.review.test_sol61_mapped_field_route import route
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph
    resolved=route(tmp_path, adaptive=True)
    assignments={r.subject.local_id:r.layout for r in resolved.layout_plan.assignments if r.subject_kind=="block"}
    sources=[]
    for layout in resolved.layout_plan.layouts:
        blocks=tuple(b for b in resolved.blocks if assignments[b.name]==layout.handle)
        p=detach_compiled_program(slice_program(resolved.time,tuple(b.name for b in blocks)))
        sources.append(emit_program_graph(p.to_graph(),lowering_program=p,model_graph=ProgramModelGraph.from_resolved_blocks(blocks),target="amr_system",field_plans={}))
    assert any("mapped_field_" in source for source in sources)
    assert sum("stage_field_components" in source for source in sources)==1
