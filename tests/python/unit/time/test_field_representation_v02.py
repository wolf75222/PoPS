"""Source contracts for native representation ports; not native qualification."""

from types import SimpleNamespace
import pytest
from pops.time import Program
from pops.time._program.distributed_arrays import Distributed
from pops.time._program.detach import detach_compiled_program
from pops.time._program.structured_control import new_typed
from pops.time.field_representation import (
    validate_field_representation,
    validate_field_representation_program,
)
from pops.time.value_signatures import value_signature, same_signature
from pops.codegen.program_emit_field_representation import emit_field_representation
from typed_program_support import typed_state


def fixture(*, policy=True, dimension=2):
    p = Program("field_representation_source_contract")
    field = typed_state(p, "fluid")
    if policy:
        Distributed(p, storage_bytes=1024 * 1024, storage_versions=128)
    # Only authoring is under test. The closed regrid graph supplies this marker;
    # the native callback alone can issue an actual authenticated topology view.
    p.__dict__["_continuation_regrid_authoring"] = {
        "blocks": (field.block,),
        "dimension": dimension,
    }
    topology = p.regrid_topology(field, level=p.int64(0), side="source", dimension=dimension)
    return p, field, topology


def export(p, field, topology, **kw):
    return p.field_bundle(field, topology=topology, cell_capacity=512, patch_capacity=16, **kw)


def test_export_schema_preserves_distributed_storage_and_physical_support():
    p, field, topology = fixture()
    value = export(p, field, topology)
    signature = value_signature(value)
    roles = dict(signature["members"])
    assert tuple(roles) == ("instance", "fields", "geometry", "maps", "histories")
    assert roles["instance"]["ownership"] == "contiguous_distributed"
    geometry = dict(roles["geometry"]["members"])
    assert set(geometry) == {
        "coordinates",
        "indices",
        "measure",
        "volume_fraction",
        "is_valid",
        "is_active",
        "is_covered",
    }
    assert all(item["ownership"] == "contiguous_distributed" for item in geometry.values())
    assert dict(geometry["coordinates"]["element"]["members"]).keys() == {"x0", "x1"}
    maps = dict(roles["maps"]["members"])
    assert maps["patch_instance"]["ownership"] == maps["patches"]["ownership"] == "replicated"
    patches = dict(maps["patches"]["element"]["members"])
    assert set(patches) == {
        "owner_rank",
        "global_patch",
        "global_offset",
        "count",
        "lo0",
        "lo1",
        "hi0",
        "hi1",
        "ghost0",
        "ghost1",
        "is_replica",
    }
    assert patches["is_replica"]["vtype"] == "bool"
    validate_field_representation_program(p)


def test_ports_preserve_full_family_witness_and_detach_exactly():
    p, field, source = fixture()
    before = export(p, field, source)
    target = p.regrid_topology(field, level=p.int64(1), side="target", dimension=2)
    after = export(p, field, target, family=before)
    assert same_signature(value_signature(before), value_signature(after))
    result = p.bundle_field(after, template=field, topology=target)
    assert same_signature(value_signature(result), value_signature(field))
    detached = detach_compiled_program(p)
    validate_field_representation_program(detached)
    assert detached._ir_hash() == p._ir_hash()


def test_export_requires_explicit_budget_and_atomic_capacity_validation():
    p, field, source = fixture(policy=False)
    with pytest.raises(ValueError, match="explicit Distributed"):
        export(p, field, source)
    Distributed(p, storage_bytes=1024, storage_versions=16)
    old = (len(p._values), p._next_finite_domain_identity)
    with pytest.raises(ValueError, match="capacity"):
        p.field_bundle(field, topology=source, cell_capacity=-1, patch_capacity=8)
    assert (len(p._values), p._next_finite_domain_identity) == old


def test_topology_is_not_inferred_from_state_width_and_cannot_be_forged():
    p, field, source = fixture()
    with pytest.raises(ValueError, match="dimension"):
        p.regrid_topology(field, level=p.int64(0), side="target", dimension=3)
    forged = new_typed(p, value_signature(source), "product", (), {})
    with pytest.raises(TypeError, match="only be issued"):
        validate_field_representation_program(p)
    assert forged.vtype == "field_topology"


def test_family_capacity_mismatch_and_cross_field_authority_rejected():
    p, field, source = fixture()
    before = export(p, field, source)
    with pytest.raises(ValueError, match="capacities"):
        p.field_bundle(field, topology=source, cell_capacity=513, patch_capacity=16, family=before)
    other = typed_state(p, "other")
    with pytest.raises(TypeError, match="exact topology"):
        export(p, other, source)


def test_detached_port_rejects_changed_budget():
    p, field, source = fixture()
    value = export(p, field, source)
    bad = SimpleNamespace(
        op=value.op, attrs={**value.attrs, "storage_bytes": 0}, inputs=value.inputs
    )
    with pytest.raises(ValueError, match="resource budget"):
        validate_field_representation(bad)


def test_lowering_calls_real_native_adapter_and_keeps_owned_field_lifetime():
    p, field, source = fixture()
    before = export(p, field, source)
    target = p.regrid_topology(field, level=p.int64(1), side="target", dimension=2)
    result = p.bundle_field(before, template=field, topology=target)
    var = {field.id: "source_field", source.id: "source_topology", target.id: "target_topology"}
    lines = []
    blocks = {field.block: 0}
    emit_field_representation(p, before, var, None, lines, [], blocks, {}, "amr_system")
    emit_field_representation(p, result, var, None, lines, [], blocks, {}, "amr_system")
    source_cpp = "\n".join(lines)
    assert "ctx.export_field_bundle(source_field, source_topology," in source_cpp
    assert "ctx.import_field_bundle(" in source_cpp
    assert "static_assert(pops::kNativeDimension == 2" in source_cpp
    assert "ctx.distributed_registry().configure(1048576, 128)" in source_cpp
    assert "Field import storage validation" in source_cpp
    assert var[result.id].startswith("(*field_representation_")
    assert "for (" not in source_cpp
    assert "ctx.continuation_bind_field_provenance(source_field" not in source_cpp
    assert "source_topology.program_block() != 0" in source_cpp
    assert "target_topology.components() != 1" in source_cpp


def test_import_rejects_incomplete_geometry_and_unmarked_products():
    p, field, source = fixture()
    value = export(p, field, source)
    signature = dict(value_signature(value))
    members = list(signature["members"])
    geo = dict(members[2][1])
    geo["members"] = tuple((k, v) for k, v in geo["members"] if k != "measure")
    members[2] = ("geometry", geo)
    signature["members"] = tuple(members)
    malformed = new_typed(p, signature, "region_argument", (), {})
    with pytest.raises((TypeError, ValueError), match="(binding|schema|Bundle)"):
        p.bundle_field(malformed, template=field, topology=source)
    unmarked = p.product(
        instance=value["instance"],
        fields=value["fields"],
        geometry=value["geometry"],
        maps=value["maps"],
        histories=value["histories"],
    )
    with pytest.raises((TypeError, ValueError), match="(schema|Bundle)"):
        p.bundle_field(unmarked, template=field, topology=source)


def test_collective_export_cannot_execute_in_a_rank_local_map():
    p, field, source = fixture()
    value = export(p, field, source)
    d = Distributed(p)
    with pytest.raises((TypeError, ValueError), match="(local|collective|field_bundle_export)"):
        d.local_map(
            value["instance"],
            lambda q, i: q.field_bundle(
                field, topology=source, cell_capacity=512, patch_capacity=16
            ),
            byte_capacity=1048576,
        )


def test_ephemeral_topology_cannot_be_retained_in_continuation():
    from pops.time._program.versioned_values import _persistable

    p, field, source = fixture()
    with pytest.raises(TypeError, match="(capability|persist|Continuation)"):
        _persistable(value_signature(source))


def test_topology_copies_through_product_branch_fold_capture_and_iteration():
    p, field, source = fixture()
    holder = p.product(view=source, count=p.int64(3))
    selected = p.branch(p.boolean(True), lambda q: holder["view"], lambda q: source)
    carried = p.fold(p.product(view=selected), 3, lambda q, x, i: q.product(view=x["view"]))
    captured = p.require_success(p.capture(lambda q: carried["view"]))
    iteration = p.require_success(
        p.iterate(
            p.product(view=captured), lambda q, x: x, lambda q, x: q.boolean(True), max_steps=2
        )
    )
    exported = export(p, field, iteration["state"]["view"])
    validate_field_representation_program(p)
    detached = detach_compiled_program(p)
    validate_field_representation_program(detached)
    assert exported.op == "field_bundle_export"


def test_structural_copy_cannot_change_issued_topology_type():
    from pops.time.field_representation import validate_topology_origin

    p, field, source = fixture()
    wrong = {**value_signature(source), "dimension": 3}
    projection = new_typed(p, wrong, "product_get", (p.product(view=source),), {"field": "view"})
    with pytest.raises(TypeError, match="exact issued"):
        validate_topology_origin(projection)
