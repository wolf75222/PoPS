"""N10 phase 1, lot 1a (ROM-85): index sets of IR2, descriptors of IR3, the value types of IR1.

Source-only: no program is resolved and no native module is loaded.
"""
from __future__ import annotations

import pytest

from pops._ir import arrays as A
from pops._ir import index_sets as S
from pops._ir.index_maps import cell, face


def _configuration(identity="case/left", extents=(32, 16), periodic=(False, False)):
    return S.Configuration(identity, extents, periodic)


def test_cells_compare_by_configuration_instance_not_by_grid():
    left, right = _configuration("case/left"), _configuration("case/right")
    # Co-located blocks share one support, so one configuration and one set of cells (§4.1).
    assert S.cells(left) == S.cells(_configuration("case/left"))
    # Two supports with equal grids keep two sets of cells.
    assert S.cells(left) != S.cells(right)
    assert hash(S.cells(left)) == hash(S.cells(_configuration("case/left")))


def test_entities_carry_their_lattice_descriptor_only_from_the_constructor():
    κ = _configuration(extents=(8, "ny"), periodic=(True, False))
    assert S.cells(κ) == S.entities(κ, cell)
    assert S.faces(κ, 1) == S.entities(κ, face(1))
    assert S.cells(κ).descriptors == (S.Lattice(2, (8, "ny"), (True, False), cell),)
    assert S.faces(κ, 0).descriptor(S.Lattice).entity == face(0)
    with pytest.raises(ValueError, match="not an axis"):
        S.faces(κ, 2)
    with pytest.raises(TypeError, match="cell or face"):
        S.entities(κ, face(0).__class__("face", 0, "low"))


def test_levels_is_a_union_of_levels_that_carry_the_hierarchy_descriptor():
    κ = _configuration()
    H = S.HierarchyInstance("case/layout#H0", κ, 3, (2, 4))
    union = S.levels(H)
    assert type(union) is S.Union
    assert union.parts == tuple(S.level(H, index) for index in range(3))
    assert union.descriptors == ()
    for part in union.parts:
        assert part.descriptors == (S.Hierarchy(3, (2, 4)),)
    assert S.composite(H).descriptor(S.Hierarchy) == S.Hierarchy(3, (2, 4))
    # A regrid issues a new hierarchy instance, so new index sets.
    regridded = S.HierarchyInstance("case/layout#H1", κ, 3, (2, 4))
    assert S.level(regridded, 1) != S.level(H, 1)
    assert S.levels(S.HierarchyInstance("case/layout#H2", κ, 1, ())) == S.level(
        S.HierarchyInstance("case/layout#H2", κ, 1, ()), 0)
    with pytest.raises(ValueError, match="not a level"):
        S.level(H, 3)
    with pytest.raises(ValueError, match="ratio"):
        S.HierarchyInstance("case/layout#H3", κ, 2, (1,))


def test_product_of_lattices_carries_both_descriptors_and_concatenated_axes():
    x = S.box((16, 8), periodic=(True, False))
    v = S.box(("nv",))
    p = S.product(x, v)
    assert p.descriptor(S.ProductSet) == S.ProductSet(x, v)
    assert p.descriptor(S.Lattice) == S.Lattice(3, (16, 8, "nv"), (True, False, False), cell)
    finite = S.product(x, S.range_set("case/particles"))
    assert finite.descriptor(S.Lattice) is None
    assert finite.descriptor(S.ProductSet) is not None


def test_range_sets_union_and_constructor_refusals():
    a, b = S.range_set("case/a"), S.range_set("case/b")
    assert a == S.range_set("case/a") and a != b
    assert S.union(a, b).parts == (a, b)
    with pytest.raises(ValueError, match="each part once"):
        S.union(a, a)
    with pytest.raises(TypeError, match="two index sets"):
        S.union(a)
    with pytest.raises(TypeError, match="identity"):
        S.range_set("")
    with pytest.raises(ValueError, match="positive"):
        S.box((0,))
    with pytest.raises(TypeError, match="bool per axis"):
        S.box((4, 4), periodic=(True,))


def test_one_array_type_whatever_the_index_set():
    κ = _configuration()
    field = A.Array(S.cells(κ), A.Product(rho=A.Float64, mx=A.Float64))
    assert field == A.Array(S.cells(_configuration()), A.Product(rho=A.Float64, mx=A.Float64))
    # A state and its rate have the same type; the role is not part of it.
    assert field != A.Array(S.cells(κ), A.Product(mx=A.Float64, rho=A.Float64))
    finite = A.Array(S.range_set("case/particles"), A.Float64)
    assert type(finite) is type(field)
    nested = A.Array(S.range_set("case/blocks"), field)
    assert nested.element == field
    assert A.Index(S.cells(κ)) != A.Index(S.range_set("case/particles"))
    assert A.Result(A.Float64, A.Variant(diverged=A.Int64)).failure == A.Variant(
        diverged=A.Int64)
    assert A.Iteration(field).state is field
    assert A.IndexSetType(S.range_set("case/a")).of == S.range_set("case/a")
    assert repr(A.Float64) == "Float64"


def test_type_constructors_refuse_what_is_not_a_type():
    with pytest.raises(TypeError, match="index set"):
        A.Array((4, 4), A.Float64)
    with pytest.raises(TypeError, match="IR1 type"):
        A.Array(S.box((4,)), float)
    with pytest.raises(ValueError, match="each field once"):
        A.Product(("a", A.Float64), ("a", A.Int64))
    with pytest.raises(ValueError, match="at least one field"):
        A.Product()
    with pytest.raises(ValueError, match="Bool, Int64 or Float64"):
        A.Scalar("Float32")
    assert not A.is_type(A.NotData("operator"))


# -- the data model of a resolved plan: placements, IR types, layout records ---------------------

def _plan(identifier):
    import pops
    from examples.api_v02.profile_dispatch import build_profile

    spec = build_profile(identifier)
    return pops.resolve(pops.validate(spec.case), layout=spec.layout)


def test_co_located_blocks_share_one_set_of_cells_and_state_and_rate_share_one_type():
    from pops._ir.data_model import data_layout_and_distribution, ir_type, support_placements

    plan = _plan("M12")
    placements = support_placements(plan)
    assert len(set(placements.configurations.values())) == 1
    assert placements.domain("five") == placements.domain("EM") == S.cells(placements.default)
    types = {}
    for value in plan.time._values:
        if value.vtype in ("state", "rhs") and value.block is not None:
            types.setdefault((value.block.local_id, value.vtype), ir_type(value, placements))
    for block in ("five", "ten", "EM"):
        if (block, "rhs") in types:
            assert types[(block, "state")] == types[(block, "rhs")]
        assert type(types[(block, "state")]) is A.Array
    report = data_layout_and_distribution(plan)
    assert report["untyped"] == 0
    assert {row["layout"] for row in report["values"] if row["role"]} == {"Patches"}


def test_glued_domains_get_two_placements_and_an_adaptive_field_lives_on_levels():
    import pops
    from examples.api_v02.method_relaxed_multiblock import (
        build_t10_multiblock, transport_trace_flux)
    from pops._ir.data_model import data_layout_and_distribution, support_placements

    flux = transport_trace_flux()
    spec = build_t10_multiblock(flux)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout, components=(flux,))
    placements = support_placements(plan)
    left, right = placements.configurations["left"], placements.configurations["right"]
    assert left != right and left.extents == right.extents
    assert placements.default is None
    domain = placements.domain("left")
    assert type(domain) is S.Union and all(type(part) is S.Level for part in domain.parts)
    report = data_layout_and_distribution(plan)
    fields = [row for row in report["values"] if row["role"]]
    assert fields and {row["layout"] for row in fields} == {"LevelPatches"}
    # The plan partitions the coarse level of a cross-block interface by its own choice (C50).
    assert {(row["distribution"], row["origin"]) for row in fields} == {
        ("coarse level partitioned", "plan")}


def test_what_the_plan_cannot_type_is_written_untyped_and_never_refused():
    from types import SimpleNamespace

    from pops._ir.data_model import Placements, ir_type

    κ, other = _configuration("case/a"), _configuration("case/b")
    several = Placements({"a": κ, "b": other}, {}, {"a": False, "b": False},
                         {"a": "layout", "b": "layout"}, None)
    field = SimpleNamespace(vtype="scalar_field", block=None, space=None,
                            logical_shape={"n_comp": 1}, attrs={}, state_ref=None,
                            field_context=None, point=None)
    value_type = ir_type(field, several)
    assert type(value_type) is A.Untyped and "several supports" in value_type.reason
    one = Placements({"a": κ}, {}, {"a": False}, {"a": "layout"}, κ)
    assert ir_type(field, one) == A.Array(S.cells(κ), A.Float64)
    strange = SimpleNamespace(vtype="mask", block=None, space=None, logical_shape=None,
                              attrs={}, state_ref=None, field_context=None, point=None)
    assert type(ir_type(strange, one)) is A.Untyped
    operator = SimpleNamespace(vtype="operator", block=None, space=None, logical_shape=None,
                               attrs={})
    assert ir_type(operator, one) == A.NotData("operator")


def test_dmt1_compare_gates_undeclared_mesh_differences_and_reports_array_ones(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[4] / "scripts" / "dmt1.py"
    spec = importlib.util.spec_from_file_location("dmt1_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def write(name, changed=()):
        rows = []
        for identifier in module.PROFILES:
            sha = ("changed" if identifier in changed else "same") + identifier
            rows.append(json.dumps({"id": identifier, "program": {"bytes": 1, "sha": sha},
                                    "models": {}}))
        target = tmp_path / name
        target.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return target

    base = write("base.jsonl")
    assert module.compare(base, write("same.jsonl"), None) == 0
    assert module.compare(base, write("array.jsonl", ("M23",)), None) == 0
    mesh = write("mesh.jsonl", ("T08",))
    assert module.compare(base, mesh, None) == 1
    declared = tmp_path / "declared.txt"
    declared.write_text("# kernels of T08 change on purpose\nT08\n", encoding="utf-8")
    assert module.compare(base, mesh, declared) == 0
    declared.write_text("T99\n", encoding="utf-8")
    assert module.compare(base, mesh, declared) == 2
    assert len(module.GATED) == 29 and len(module.PROFILES) == 40


def test_the_compiled_plan_record_reports_the_layouts_outside_its_identity():
    from dataclasses import replace

    from pops._ir.data_model import data_layout_and_distribution
    from pops.codegen._compiled_artifact import CompiledPlanRecord

    plan = _plan("M01")
    record = CompiledPlanRecord.from_resolved(plan)
    assert record.data_layout == data_layout_and_distribution(plan)
    assert record.data_layout["placements"] and record.data_layout["unplaced"] is None
    # Reported only: the contract identity of the record does not depend on it.
    assert replace(record, data_layout={}).contract_identity == record.contract_identity
