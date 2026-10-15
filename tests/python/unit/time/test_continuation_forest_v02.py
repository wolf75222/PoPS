"""Closed forest source contracts; native AMR acceptance is tested separately."""
import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.continuations import Candidate, ContinuationSchema, Assess, assess, accept
from pops.graph_functions import graph
from pops.time.variant_types import Product, Bool, type_of
from test_continuations_v10 import candidate_fixture


def _approve(p, base, proposal, continuation, observations):
    candidate = Candidate(p, base=base, proposal=proposal, continuation=continuation,
                          increment=p.product(), observations=observations, checks=p.product())
    envelope = p.continuation_candidate_value(candidate)
    read = p.continuation_read(base)

    @graph(inputs=Product(base=type_of(read), proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product()))
    def check(q, args):
        return q.product(accepted=args["candidate"]["continuation"]["step"] > args["base"]["step"],
                         checks=q.product())

    return p.require_success(accept(p, candidate, assess(p, candidate, Assess(check))))


def forest_fixture(**options):
    parts, schema, base, proposal, candidate, policy = candidate_fixture()
    p = parts[5]
    approved = p.require_success(accept(p, candidate, assess(p, candidate, policy)))

    def synchronize(q, base, proposal):
        entry = q.continuation_read(base)
        continuation = q.product(state=proposal["synchronized"]["field_0"],
                                 time=proposal["interval"]["end"], step=entry["step"]+q.int64(1),
                                 counter=proposal["candidate"]["counter"])
        return _approve(q, base, proposal, continuation, q.product(counter=continuation["counter"]))

    def initialize(q):
        initial = q.product(time=q.float64(0), step=q.int64(0), counter=q.int64(0))
        return initial, ContinuationSchema(initial, time=("time",), step=("step",))

    def parent(q, base, proposal):
        entry = q.continuation_read(base)
        count = q.reduce(proposal["children"], q.int64(0),
                         lambda b, total, child: total+child["counter"], order="ascending")
        continuation = q.product(time=proposal["interval"]["end"], step=entry["step"]+q.int64(1),
                                 counter=entry["counter"]+count)
        return _approve(q, base, proposal, continuation, q.product(children=count))

    p.continuation_forest(approved, synchronize=synchronize, initialize=initialize,
                          parent=parent, level_capacity=8, **options)
    return p


def test_complete_amr_forest_emits_every_phase_and_exact_child_dependencies():
    p = forest_fixture()
    assert p.validate()
    source = emit_cpp_program(p, target="amr_system")
    for required in ("begin_continuation_forest", "begin_level_continuation_sync",
                     "prepare_level_continuation_sync", "begin_continuation_parent",
                     "continuation_child_dependencies()", "continuation_child_reports",
                     "install_continuation_forest", "pops_program_checkpoint_continuation_root_schema_identity"):
        assert required in source
    assert source.index("ctx.advance_hierarchy") < source.index("ctx.begin_continuation_parent")


def test_forest_is_not_a_uniform_program_with_silently_ignored_children():
    with pytest.raises(ValueError, match="AMR execution target"):
        emit_cpp_program(forest_fixture(), target="system")


def test_missing_child_publication_is_rejected_before_native_execution():
    p = forest_fixture()
    from pops.codegen.program_lowerability import all_ops
    node = next(v for v in all_ops(p) if v.op == "continuation_sync_publish")
    from types import MappingProxyType
    object.__setattr__(node, "attrs", MappingProxyType(dict(node.attrs, schema=dict(node.attrs["schema"], identity="foreign"))))
    with pytest.raises((ValueError, TypeError)):
        p.validate()


@pytest.mark.parametrize("options", ({}, {"reject_parent": True}, {"reject_last_child": True}))
def test_authored_transport_and_composite_mass_vote_remain_in_the_generated_graph(options):
    import pops
    from examples.api_v02.continuation_forest_amr import build_continuation_forest_amr
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program

    spec = build_continuation_forest_amr(**options)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(plan.time, target="amr_system",
                             model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "prepare_level_continuation_sync" in source and "continuation_child_dependencies()" in source
    assert "complete_forest_mass" in source and "forest_initial_mass" in source
    assert "continuation_bind_field_provenance" in source
    assert "ctx.neg_div_flux_default_into(" in source
    # A native control-flow join retains the exact accepted capability type.
    # It must not enter the legacy scalar branch and turn the capability into Real.
    from pops.codegen.program_emit_versioned import capability_cpp_type
    from pops.codegen.program_lowerability import all_ops
    from pops.time.value_signatures import value_signature
    for value in all_ops(plan.time):
        if value.op == "branch" and value.vtype == "continuation_accepted":
            cpp_type = capability_cpp_type(value_signature(value))
            assert "%s matched_%d_s0{};" % (cpp_type, value.id) in source
            assert "pops::Real s%d;" % value.id not in source
    assert plan.time._ir_hash() == detach_compiled_program(plan.time)._ir_hash()
