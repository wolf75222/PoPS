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

    from pops._frozen_data import thaw_data

    plan = _plan("M01")
    record = CompiledPlanRecord.from_resolved(plan)
    # Frozen like the other mappings of the record, and equal to the report once thawed.
    assert thaw_data(record.data_layout) == data_layout_and_distribution(plan)
    assert record.data_layout["placements"] and record.data_layout["unplaced"] is None
    # Reported only: the contract identity of the record does not depend on it.
    assert replace(record, data_layout={}).contract_identity == record.contract_identity


def test_dmt1_emit_runs_the_real_compile_path_on_a_short_profile(tmp_path):
    # Review of #749, T1: the identity scope that ``pops.compile`` opens changed under the harness
    # (#732); only an emission sees it, ``compare`` alone does not.
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "m15.jsonl"
    done = subprocess.run([sys.executable, str(root / "scripts" / "dmt1.py"), "emit", str(out),
                           "M15"], cwd=root, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout + done.stderr
    row = json.loads(out.read_text(encoding="utf-8"))
    assert "failed" not in row and row["program"]["bytes"] > 0
    assert row["models"] and all(value is not None for value in row["models"].values())


def test_the_manifest_reports_the_values_of_the_control_bodies():
    # T2: a structured iterate keeps its field leaves in its body, not at the first level.
    from pops._ir.data_model import data_layout_and_distribution

    plan = _plan("M18")
    first = {str(value.name if getattr(value, "name", None) else value.id)
             for value in plan.time._values}
    report = data_layout_and_distribution(plan)
    nested = [row for row in report["values"] if row["value"] not in first]
    assert len(report["values"]) > len(first) and nested
    assert any(row["role"] == "state" and row["layout"] == "Patches" for row in nested)
    assert report["untyped"] == sum(1 for row in report["values"] if row["layout"] == "untyped")


def test_a_field_without_a_block_reads_its_component_count_from_its_attributes():
    # T3: ``scalar_field()`` without a block writes ``attrs["ncomp"]`` and nothing else.
    from pops._ir.data_model import data_layout_and_distribution

    report = data_layout_and_distribution(_plan("M10"))
    fields = [row for row in report["values"] if row["spelling"] == "scalar_field"]
    assert fields and all(row["type"].startswith("Array(") for row in fields)
    assert report["untyped"] == 0


def test_every_record_states_a_residency_read_from_the_rule_of_g13():
    # T4: field leaves live in views of the default memory space; a finite array where
    # ``device_array`` puts it; what G13 says nothing about is written untyped.
    from pops._ir.data_model import data_layout_and_distribution
    from pops.codegen.program_emit_finite_kokkos import device_array
    from pops.time.value_signatures import value_signature

    plan = _plan("M12")
    report = data_layout_and_distribution(plan)
    assert all(row["residency"] for row in report["values"])
    assert {row["residency"] for row in report["values"] if row["role"]} == {
        "device: views of the default memory space"}
    from pops.time._graph.value_traversal import walk_program_nodes

    plan = _plan("T06")  # replicated finite arrays, some of flat elements and some structured
    rows = {row["value"]: row for row in data_layout_and_distribution(plan)["values"]}
    finite = [value for value in walk_program_nodes(plan.time._values) if value.vtype == "array"]
    residencies = set()
    for value in finite:
        row = rows[str(value.name if getattr(value, "name", None) else value.id)]
        device = device_array(value_signature(value))
        assert row["residency"].startswith("device" if device else "host"), row
        residencies.add(row["residency"])
    assert residencies == {"device: columns of the default memory space",
                           "host: structural cells"}


def test_one_function_reads_the_kind_of_a_value_from_the_head_of_its_type():
    # Lot 1b (note §3.4): a field leaf is an Array over the cells of a support, a finite array an
    # Array over a range set; the spelling of a signature, a live value or the bare spelling agree.
    from types import SimpleNamespace

    from pops._ir.data_model import (
        field_role, is_field_leaf, is_finite_array, is_finite_index, value_kind)

    kinds = {"state": "field", "rhs": "field", "scalar_field": "field", "array": "finite",
             "index_set": "index_set", "index": "index", "scalar": "other",
             "vector_field": "other", "product": "other", "operator": "other"}
    for spelling, kind in kinds.items():
        assert value_kind(spelling) == kind
        assert value_kind({"vtype": spelling}) == kind
        assert value_kind(SimpleNamespace(vtype=spelling)) == kind
        assert is_field_leaf(spelling) is (kind == "field")
        assert is_finite_array(spelling) is (kind == "finite")
        assert is_finite_index(spelling) is (kind in ("finite", "index_set", "index"))
    assert [field_role(spelling) for spelling in ("state", "rhs", "scalar_field", "array")] == [
        "state", "rhs", "field", None]
    assert value_kind({}) == value_kind(None) == "other"


def test_a_live_field_value_asks_the_one_function():
    from pops._ir.data_model import is_field_leaf

    plan = _plan("M01")
    values = list(plan.time._values)
    assert values and all(value.is_field() is is_field_leaf(value) for value in values)
    assert any(value.is_field() for value in values)
