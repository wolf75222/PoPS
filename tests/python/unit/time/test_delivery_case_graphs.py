"""Coverage at public Model/Case validate/resolve boundaries, not only checker calls."""
import pytest
import pops
from pops.codegen.realization_delivery import delivered_frontiers
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from examples.api_v02.realization_loop_checks import assemble_delivery_case


def exact_frontiers(candidate, outputs):
    ids = {value.id for value in outputs}
    return delivered_frontiers(candidate, lambda node: node.id in ids)


@pytest.mark.parametrize('kind,last_iteration', [('loop1', 1), ('overwrite2', 2)])
def test_public_case_accepts_and_emits_the_single_final_loop_delivery(kind, last_iteration):
    spec, program, use, candidate, outputs = assemble_delivery_case(kind)
    frontiers = exact_frontiers(candidate, outputs)
    assert len(frontiers) == 1 and len(frontiers[0]) == 1
    reference = next(iter(frontiers[0]))
    assert reference.value_id == outputs[-1].id
    assert reference.invocation[-1][1] == last_iteration
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert len(plan.blocks[0].numerics.rates[0].method.uses) == 1
    assert plan.blocks[0].numerics.rates[0].method.uses[0].to_data() == use.to_data()
    cpp = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert 'owned_exchange' in cpp and 'for (' in cpp


@pytest.mark.parametrize('kind,sizes,message', [
    ('loop0', [0], 'exact state commit'),
    ('dynamic_empty', [0, 1, 1], 'exact state commit'),
    ('accumulate2', [2], 'exactly one owned SSA delivery'),
    ('legacy_recovery', [0, 1], 'exact state commit'),
    ('where_predicate', [0], 'exact state commit'),
    ('expression_where_predicate', [0], 'exact state commit'),
])
def test_public_case_refuses_the_exact_missing_or_duplicate_frontier(kind, sizes, message):
    spec, program, use, candidate, outputs = assemble_delivery_case(kind)
    assert sorted(map(len, exact_frontiers(candidate, outputs))) == sizes
    # The inspection above explains this public refusal; it does not replace the refusal.
    with pytest.raises(ValueError, match=message):
        pops.resolve(pops.validate(spec.case), layout=spec.layout)


def test_public_case_accepts_dynamic_zero_only_with_an_initial_delivery():
    spec, program, use, candidate, outputs = assemble_delivery_case('dynamic_owned')
    frontiers = exact_frontiers(candidate, outputs)
    assert sorted(map(len, frontiers)) == [1, 1, 1]
    refs = set().union(*frontiers)
    assert any(ref.value_id == outputs[0].id and ref.invocation == () for ref in refs)
    assert sorted(ref.invocation[-1][1] for ref in refs if ref.invocation) == [1, 2]
    pops.resolve(pops.validate(spec.case), layout=spec.layout)


@pytest.mark.parametrize('kind', ['legacy_guarded', 'typed_guarded'])
@pytest.mark.parametrize('nested_failure', [False, True])
def test_public_case_guarded_error_has_no_spurious_successful_empty_frontier(kind, nested_failure):
    spec, program, use, candidate, outputs = assemble_delivery_case(kind, nested_failure=nested_failure)
    frontiers = exact_frontiers(candidate, outputs)
    assert len(frontiers) == 1 and len(frontiers[0]) == 1
    assert next(iter(frontiers[0])).value_id == outputs[0].id
    assert 'realization_use_v02' in outputs[0].attrs
    # The error arm rejects the attempt; it cannot manufacture a committed empty
    # delivery. The same error recovered to a value remains rejected above.
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    assert plan.blocks[0].numerics.rates[0].method.uses[0].to_data() == use.to_data()
    emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
