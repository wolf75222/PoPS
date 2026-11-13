"""N10 lot 4 (spec 0.3 DM3, DM4; addendum of lot 4): the storage the plan chooses.

* Origin: a storage choice the case writes is a checked constraint, the others are the plan's,
  and the layout record says which (§1).
* Every case of the sample conforms without its constraints, and its values keep their types on
  every admissible layout (§2, §3).
* A replicated value given to the distributed builder is a layout fact, ``LayoutMismatch`` (§4).
* ``csr`` is ``Product(offsets, indices)`` over range sets (§5); an array over a product set is
  stored ``ProductLayout(Patches, Local)`` (§6).
"""
from __future__ import annotations

import importlib

import pops
import pops.algebra
import pytest
from pops._ir import arrays as A
from pops._ir import index_sets as S
from pops._ir.data_model import _layout, data_layout_and_distribution, ir_type, support_placements
from pops._ir.refusals import LayoutMismatch, RegionRefusal
from pops.time._graph.value_traversal import walk_program_nodes

from tests.python.support.unconstrained import without_storage_constraints

REPLICATED = (
    ("examples.api_v02.finite_bdf2_v14", "build_finite_bdf2_v14", {}),
    ("examples.api_v02.methods_patankar_network", "build_t11_network", {}),
    ("examples.api_v02.extension_closure_v14", "build_extension_closure_v14", {}),
    ("examples.api_v02.finite_amr", "build_finite_amr", {}),
)
DISTRIBUTED = tuple(
    (module, builder, {"ranks": ranks})
    for module, builder in (("examples.api_v02.distributed_algebra_v14", "build_distributed_algebra"),
                            ("examples.api_v02.distributed_program_checks",
                             "build_distributed_program"))
    for ranks in (2, 3))
SAMPLE = REPLICATED + DISTRIBUTED


def _plan(module, builder, kwargs, *, constrained):
    def build():
        spec = getattr(importlib.import_module(module), builder)(**kwargs)
        return pops.resolve(pops.validate(spec.case), layout=spec.layout)
    if constrained:
        return build()
    with without_storage_constraints():
        return build()


# Written offsets are Int64 constants of the Program, one per rank and the end, where the plan's
# partition reads one size; the values that carry data are the same, in the same order.
_OFFSET_SCALARS = frozenset(("int64", "scalar", "bool"))


def _types(plan):
    placements = support_placements(plan)
    program = str(getattr(plan.time, "name", None) or "program")
    return [(value.vtype, repr(ir_type(value, placements, program=program)))
            for value in walk_program_nodes(plan.time._values)
            if value.vtype not in _OFFSET_SCALARS]


def _ident(case):
    module, _, kwargs = case
    return module.rsplit(".", 1)[1] + "".join("_%s%s" % item for item in kwargs.items())


@pytest.mark.parametrize("case", SAMPLE, ids=_ident)
def test_a_case_types_alike_with_and_without_its_storage_constraints(case):
    constrained = _plan(*case, constrained=True)
    free = _plan(*case, constrained=False)
    left, right = _types(constrained), _types(free)
    assert len(left) == len(right)
    assert left == right
    assert not any("Untyped" in kind and "issues family" in kind for _, kind in left)


@pytest.mark.parametrize("case", SAMPLE, ids=_ident)
def test_without_its_constraints_every_storage_choice_is_the_plans(case):
    before = data_layout_and_distribution(_plan(*case, constrained=True))
    after = data_layout_and_distribution(_plan(*case, constrained=False))
    assert before["schema_version"] == after["schema_version"] == 6
    assert before["constraints"] > 0
    assert after["constraints"] == 0
    data = [[row for row in report["values"] if row["spelling"] not in _OFFSET_SCALARS]
            for report in (before, after)]
    assert len(data[0]) == len(data[1])
    for written, chosen in zip(*data, strict=True):
        assert {key: value for key, value in written.items()
                if key not in ("origin", "distribution", "node", "value")} == \
            {key: value for key, value in chosen.items()
             if key not in ("origin", "distribution", "node", "value")}
        if written["origin"] == "constraint":
            assert chosen["origin"] == "plan", written
            assert written["layout"] in ("Replicated", "Contiguous", "members"), written
    finite = [row for row in after["values"] if row["layout"] in ("Replicated", "Contiguous")]
    assert finite
    for row in finite:
        assert row["distribution"] == ("every rank" if row["layout"] == "Replicated"
                                       else "balanced over the execution lane"), row


def test_written_offsets_and_ownership_are_constraints_with_their_distribution():
    report = data_layout_and_distribution(_plan(*DISTRIBUTED[0], constrained=True))
    contiguous = [row for row in report["values"] if row["layout"] == "Contiguous"]
    assert contiguous and all(row["origin"] == "constraint" for row in contiguous)
    assert {row["distribution"] for row in contiguous} == {
        "partitioned by the offsets the case wrote"}
    report = data_layout_and_distribution(_plan(*REPLICATED[0], constrained=True))
    replicated = [row for row in report["values"] if row["layout"] == "Replicated"]
    assert replicated and {(row["origin"], row["distribution"]) for row in replicated} == {
        ("constraint", "every rank")}


@pytest.mark.parametrize("case", REPLICATED, ids=_ident)
def test_an_unconstrained_replicated_build_emits_the_same_program_but_its_identity(case):
    # Only the replicated cases: a distributed case emits issue_balanced(...) where its written
    # offsets emit issue(...) (addendum rev 2.1, A1).
    from pops.codegen.program_codegen import emit_cpp_program

    module, builder, kwargs = case
    constrained = emit_cpp_program(_plan(module, builder, kwargs, constrained=True).time)
    free = emit_cpp_program(_plan(module, builder, kwargs, constrained=False).time)
    assert len(constrained.splitlines()) == len(free.splitlines())
    differ = [(a, b) for a, b in zip(constrained.splitlines(), free.splitlines(), strict=True)
              if a != b]
    assert all(_identity_only(a, b) for a, b in differ), differ[:3]


def _identity_only(left, right):
    import re

    hexes = re.compile(r"[0-9a-f]{64}|[A-Za-z0-9+/=]{40,}")
    return hexes.sub("#", left) == hexes.sub("#", right)


def test_the_replicated_builder_admits_a_set_without_ownership():
    p = pops.Program("unconstrained")
    stated = p.range_set(2, capacity=2, ownership="replicated")
    chosen = p.range_set(2, capacity=2)
    assert stated.attrs.get("storage_origin") is None
    assert chosen.attrs["storage_origin"] == "plan"
    assert stated.attrs["signature"]["ownership"] == chosen.attrs["signature"]["ownership"] == \
        "replicated"
    family = pops.algebra.IndexFamily(3)
    assert family.ownership == "replicated" and not family._ownership_written
    assert p.range_set(1, family=family).attrs["storage_origin"] == "plan"
    written = pops.algebra.IndexFamily(3, ownership="replicated")
    assert "storage_origin" not in p.range_set(1, family=written).attrs
    with pytest.raises(NotImplementedError, match="DM3"):
        p.range_set(2, capacity=2, ownership="contiguous_distributed")
    with pytest.raises(ValueError, match="use Distributed"):
        pops.algebra.IndexFamily(3, ownership="contiguous_distributed")


def _mixed():
    from pops.time._program.distributed_arrays import Distributed

    p = pops.Program("mixed")
    replicated = p.range_set(2, capacity=2, ownership="replicated")
    values = p.map(replicated, lambda b, i: b.float64(1.))
    d = Distributed(p)
    distributed = d.balanced_domain(2, capacity=2)
    return p, d, replicated, values, distributed


def _mismatch(call):
    with pytest.raises(LayoutMismatch) as caught:
        call()
    refusal = caught.value
    assert isinstance(refusal, TypeError) and isinstance(refusal, ValueError)
    assert isinstance(refusal, RegionRefusal) and refusal.rule == "DM3"
    assert {refusal.expected, refusal.found} == {"Contiguous", "Replicated"}
    assert "Replicated" in str(refusal) and "Contiguous" in str(refusal)


def test_the_distributed_gate_refuses_a_replicated_value_as_a_layout_mismatch():
    _, d, replicated, values, _ = _mixed()
    _mismatch(lambda: d.size(replicated))
    _mismatch(lambda: d.domain_of(values))


def test_same_domain_of_two_layouts_is_a_layout_mismatch():
    p, _, replicated, _, distributed = _mixed()
    _mismatch(lambda: p.same_domain(replicated, distributed))


def test_at_of_a_replicated_array_with_a_distributed_index_is_a_layout_mismatch():
    p, d, _, values, distributed = _mixed()

    def body(b, index):
        return b.capture(lambda q: q.at(values, index))
    _mismatch(lambda: d.local_map(distributed, body, byte_capacity=64))


def test_gather_and_scatter_across_layouts_are_layout_mismatches():
    p, d, replicated, values, distributed = _mixed()
    requests = d.requests(distributed, distributed, lambda b, i: i)
    _mismatch(lambda: p.gather(values, requests))
    _mismatch(lambda: p.scatter_reduce(values, p.map(replicated, lambda b, i: i), distributed,
                                       b_initial(p), lambda b, x, y: x, order="ascending"))


def b_initial(p):
    return p.float64(0.)


def test_the_two_form_refusals_stay_type_errors():
    # A global distributed index and a scan of a distributed domain are forms phase 1 lacks,
    # not two layouts meeting: they keep their TypeError (addendum §4).
    p, d, _, values, distributed = _mixed()
    with pytest.raises(TypeError, match="lexical distributed local body") as caught:
        d.index(distributed, 0)
    assert not isinstance(caught.value, LayoutMismatch)
    with pytest.raises(TypeError, match="materialized Array values") as caught:
        p.scan(distributed, p.float64(0.), lambda b, s, i: s, order="ascending")
    assert not isinstance(caught.value, LayoutMismatch)


def test_a_csr_is_offsets_local_to_each_rank_and_the_requests_it_reads():
    plan = _plan("examples.api_v02.distributed_program_checks", "build_distributed_program", {},
                 constrained=True)
    placements = support_placements(plan)
    program = str(plan.time.name)
    report = {row["node"]: row for row in data_layout_and_distribution(plan)["values"]}
    rows = [value for value in walk_program_nodes(plan.time._values) if value.vtype == "csr"]
    assert rows
    for value in rows:
        value_type = ir_type(value, placements, program=program)
        assert isinstance(value_type, A.Product)
        members = dict(value_type.fields)
        assert set(members) == {"offsets", "indices"}
        offsets, indices = members["offsets"], members["indices"]
        assert isinstance(offsets, A.Array) and offsets.element == A.Int64
        assert offsets.domain.path.endswith("/local_rows+1")
        requests = value.inputs[1]
        assert indices == ir_type(requests, placements, program=program)
        assert isinstance(indices.element, A.Index)
        row = report[int(value.id)]
        assert row["layout"] == "members" and row["origin"] == "constraint"
        assert row["distribution"].startswith("offsets Local")
        assert report[int(requests.id)]["layout"] == "Contiguous"


def test_a_family_with_instances_of_both_origins_is_read_per_instance():
    from pops.time._program.distributed_arrays import Distributed

    def build():
        p = pops.Program("mixed_family")
        d = Distributed(p)
        balanced = d.balanced_domain(4, capacity=4)
        written = d.domain((0, 1, 4), family=balanced)
        return p, p.require_success(d.local_map(
            balanced, lambda b, i: b.capture(lambda c: c.float64(1.)), byte_capacity=64)), \
            p.require_success(d.local_map(
                written, lambda b, i: b.capture(lambda c: c.float64(2.)), byte_capacity=64))

    from pops._ir.data_model import _Origins

    p, over_balanced, over_written = build()
    nodes = tuple(walk_program_nodes(p._values))
    origins = _Origins(nodes)
    for value, expected in ((over_balanced, ("plan", "balanced over the execution lane")),
                            (over_written, ("constraint",
                                            "partitioned by the offsets the case wrote"))):
        found = _value_origin(origins, value)
        assert found == expected, (value, found)


def _value_origin(origins, value):
    from pops.time.value_signatures import value_signature

    return origins.of(value, value_signature(value).get("family"))


def test_the_plans_choice_survives_detach_and_resolve():
    from pops.time._program.detach import detach_compiled_program

    p = pops.Program("detached")
    p.range_set(2, capacity=2)
    detached = detach_compiled_program(p)
    issued = [value for value in walk_program_nodes(detached._values)
              if value.op == "finite_range_set"]
    assert [value.attrs.get("storage_origin") for value in issued] == ["plan"]
    report = data_layout_and_distribution(_plan(*REPLICATED[0], constrained=False))
    assert report["constraints"] == 0


def test_an_array_over_a_product_set_is_stored_product_layout_patches_local():
    plan = _plan("examples.api_v02.finite_amr", "build_finite_amr", {}, constrained=True)
    placements = support_placements(plan)
    base = next(iter(placements.configurations.values()))
    phase_space = S.product(S.cells(base), S.range_set("velocity"))
    value = type("Value", (), {"vtype": "array", "attrs": {}})()
    row = _layout(value, A.Array(phase_space, A.Float64), placements, plan)
    assert row[:3] == ("ProductLayout(Patches, Local)",
                       "plan box decomposition of the base, the fibre on every rank", "plan")
