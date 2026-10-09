from tests.python.support.m19_vlasov_poisson_case import build

def test_public_vlasov_poisson_full_chain(tmp_path):
    resolved=build(tmp_path)
    assert len(resolved.layout_plan.layouts)==2
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph
    texts=[]
    for layout in resolved.layout_plan.layouts:
        names=tuple(row.subject.local_id for row in resolved.layout_plan.assignments if row.subject_kind=='block' and row.layout==layout.handle)
        blocks=tuple(b for b in resolved.blocks if b.name in names)
        program=detach_compiled_program(slice_program(resolved.time,names))
        texts.append(emit_program_graph(program.to_graph(),lowering_program=program,model_graph=ProgramModelGraph.from_resolved_blocks(blocks),target='system',field_plans={}))
    assert sum(t.count('publish_field_components(') for t in texts)==2
