"""Regrid authoring/lowering contracts only; native transfer has a separate receiver."""
import pytest

from pops.continuations import Candidate, Assess, assess, accept
from pops.graph_functions import graph
from pops.time.variant_types import Product, Bool, type_of
from pops.time._program.distributed_arrays import Distributed
from pops.codegen.program_codegen import emit_cpp_program
from pops.time._program.detach import detach_compiled_program
from test_continuation_forest_v02 import forest_fixture


def approve(p, base, proposal, after, report):
    before = p.continuation_read(base)
    candidate = Candidate(p, base=base, proposal=proposal, continuation=after,
                          increment=p.product(), observations=report, checks=p.product())
    envelope = p.continuation_candidate_value(candidate)

    @graph(inputs=Product(base=type_of(before), proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product()))
    def unchanged_frontier(q, args):
        return q.product(accepted=args["candidate"]["continuation"]["step"] == args["base"]["step"],
                         checks=q.product())

    return p.require_success(accept(p, candidate, assess(p, candidate, Assess(unchanged_frontier))))


def child(p, proposal):
    Distributed(p, storage_bytes=8*1024*1024, storage_versions=128)
    source_level = p.int64(0)
    base = p.regrid_base(source_level)
    before = p.continuation_read(base)
    other = p.regrid_source(source_level)
    source_topology = p.regrid_topology(other["state"], level=source_level, side="source", dimension=2)
    source = p.field_bundle(other["state"], topology=source_topology, cell_capacity=4096, patch_capacity=128)
    template = proposal["templates"]["field_0"]
    target_topology = p.regrid_topology(template, level=proposal["level"], side="target", dimension=2)
    target = p.field_bundle(template, topology=target_topology, cell_capacity=4096,
                            patch_capacity=128, family=source)
    restored = p.bundle_field(target, template=template, topology=target_topology)
    # This fixture checks representation and authority code generation, not a transfer equation.
    after = p.product(state=restored, time=before["time"], step=before["step"], counter=before["counter"])
    return approve(p, base, proposal, after, p.product(counter=before["counter"]))


def parent(p, base, proposal):
    return approve(p, base, proposal, p.continuation_read(base), p.product())


def test_regrid_is_an_explicit_closed_graph_between_capture_and_publication():
    p = forest_fixture(regrid=child, regrid_parent=parent, dimension=2)
    assert p.validate()
    source = emit_cpp_program(p, target="amr_system")
    for fragment in ("install_continuation_regrid_graph", "begin_regrid_invocation",
                     "continuation_regrid_bind_target_", "continuation_regrid_base",
                     "continuation_regrid_source", "prepare_continuation_regrid_child",
                     "continuation_source_topology", "continuation_target_topology",
                     "ctx.export_field_bundle", "ctx.import_field_bundle",
                     "post_synchronization(0.)", "continuation_child_dependencies()"):
        assert fragment in source
    assert "static_assert(pops::kNativeDimension == 2" in source
    assert detach_compiled_program(p)._ir_hash() == p._ir_hash()


@pytest.mark.parametrize("kwargs", [{"regrid": child}, {"regrid_parent": parent},
                                    {"regrid": child, "regrid_parent": parent},
                                    {"regrid": child, "regrid_parent": parent, "dimension": 4}])
def test_partial_or_dimensionless_regrid_is_not_implicitly_completed(kwargs):
    with pytest.raises((ValueError, TypeError), match="regrid"):
        forest_fixture(**kwargs)


def test_source_views_cannot_be_requested_outside_a_closed_regrid_region():
    from pops.time import Program
    p = Program("no_free_regrid_authority")
    with pytest.raises(ValueError, match="regrid region"):
        p.regrid_source(p.int64(0))
