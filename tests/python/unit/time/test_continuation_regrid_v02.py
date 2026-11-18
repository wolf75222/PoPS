"""Regrid authoring/lowering contracts only; native transfer has a separate receiver."""
import pytest

from pops.continuations import Candidate, Assess, assess, accept
from pops.graph_functions import graph
from pops.time.variant_types import Product, Bool, type_of
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
    source_level = p.int64(0)
    base = p.regrid_base(source_level)
    before = p.continuation_read(base)
    other = p.regrid_source(source_level)
    # Each target cell reads its parent source cell; the runtime allocates the target storage.
    restored = p.regrid_read(other["state"])
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
                     "ctx.regrid_read(", "pops_relayout_byte_count",
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


# -- N10 lot 3b: regrid_read, regrid_integral and the H/H' refusal (addendum of lot 3) ----------

def _child(body):
    def author(p, proposal):
        source_level = p.int64(0)
        base = p.regrid_base(source_level)
        before = p.continuation_read(base)
        other = p.regrid_source(source_level)
        restored = body(p, proposal, other)
        after = p.product(state=restored, time=before["time"], step=before["step"],
                          counter=before["counter"])
        return approve(p, base, proposal, after, p.product(counter=before["counter"]))
    return author


def test_regrid_read_types_its_result_over_the_target_and_integrates_one_component():
    seen = {}

    def body(p, proposal, other):
        restored = p.regrid_read(other["state"])
        seen["mass"] = p.regrid_integral(restored[0])
        assert restored.vtype == other["state"].vtype
        return restored

    p = forest_fixture(regrid=_child(body), regrid_parent=parent, dimension=2)
    assert p.validate()
    from pops.time._program.continuation_regrid import TARGET_INSTANCE, hierarchy_instance
    mass = seen["mass"]
    assert mass.op == "continuation_regrid_integral" and mass.attrs["component"] == 0
    assert hierarchy_instance(mass.inputs[0]) == TARGET_INSTANCE


def test_regrid_read_reads_only_a_source_field_of_the_child():
    def template(p, proposal, other):
        return p.regrid_read(proposal["templates"]["field_0"])

    with pytest.raises(ValueError, match="regrid_source image"):
        forest_fixture(regrid=_child(template), regrid_parent=parent, dimension=2)

    def in_parent(p, base, proposal):
        p.regrid_read(p.regrid_source(p.int64(0))["state"])

    with pytest.raises(ValueError, match="target child region"):
        forest_fixture(regrid=child, regrid_parent=in_parent, dimension=2)


def test_regrid_integral_integrates_a_read_field():
    def unread(p, proposal, other):
        p.regrid_integral(other["state"][0])
        return p.regrid_read(other["state"])

    with pytest.raises(ValueError, match="computed from regrid_read"):
        forest_fixture(regrid=_child(unread), regrid_parent=parent, dimension=2)


def test_a_child_that_mixes_h_and_h_prime_is_refused():
    from pops._ir.refusals import HierarchyInstancesMixed

    def mixed(p, proposal, other):
        restored = p.regrid_read(other["state"])
        return p.value("mixed", restored + other["state"], at=restored.point)

    with pytest.raises(HierarchyInstancesMixed, match="DM2") as caught:
        forest_fixture(regrid=_child(mixed), regrid_parent=parent, dimension=2)
    assert caught.value.op and caught.value.rule == "DM2"
