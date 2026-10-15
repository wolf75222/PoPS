"""Explicit, bounded representation ports between native Fields and finite Bundles.

Topology is an ephemeral capability issued by a regrid transaction. Export preserves
all valid and ghost storage in distributed rank/patch/cell order. Import copies the
provided values into the authenticated target layout; interpolation is Python algebra.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pops.bundles import BundleSchema, validate_bundle_signature
from pops.time._authoring import atomic_authoring
from pops.time._program.structured_control import new_typed, product_signature, scalar_signature
from pops.time._program.value_validation import require_owned
from pops.time.value_signatures import same_signature, value_signature
from pops.time.variant_types import ValueType

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object

FIELD_REPRESENTATION_OPS = frozenset(("field_bundle_export", "field_bundle_import", "field_storage_allocate"))
TOPOLOGY_OPS = frozenset(("continuation_source_topology", "continuation_target_topology"))


def validate_topology_signature(signature):
    if (not isinstance(signature, Mapping) or set(signature) != {
            "vtype", "field_signature", "side", "dimension"}
            or signature["vtype"] != "field_topology"
            or signature["side"] not in ("source", "target")
            or type(signature["dimension"]) is not int
            or not 1 <= signature["dimension"] <= 3):
        raise TypeError("FieldTopology requires an exact native dimension and source/target authority")
    field = signature["field_signature"]
    if (not isinstance(field, Mapping) or field.get("vtype") not in ("state", "scalar_field")
            or field.get("block") is None or field.get("space") is None
            or not getattr(field["space"], "components", ())):
        raise TypeError("FieldTopology requires an authenticated complete Field signature")
    return signature


def _capacity(value, name):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(name + " requires a nonnegative exact Int64 capacity")
    return value


def _policy(program):
    policy = program.__dict__.get("_distributed_storage_policy")
    if not program.__dict__.get("_distributed_storage_explicit", False) or policy is None:
        raise ValueError("Field representation requires explicit Distributed storage_bytes/storage_versions")
    if any(type(value) is not int or not 0 < value < 2**63 for value in policy):
        raise ValueError("Field representation has an invalid distributed storage policy")
    return {"storage_bytes": policy[0], "storage_versions": policy[1],
            "storage_policy_binding": "explicit"}


def _record(**members):
    return product_signature(tuple(sorted(members.items())))


def field_bundle_signature(*, dimension, components, cell_domain, patch_domain):
    """One closed schema shared by export, detached validation and native import."""
    real, integer, boolean = (scalar_signature(kind) for kind in ("scalar", "int64", "bool"))
    def array(domain, element):
        return dict(domain, vtype="array", element=element)
    fields = _record(values=array(cell_domain, _record(**{
        "c%d" % i: real for i in range(components)})))
    coordinates = _record(**{"x%d" % i: real for i in range(dimension)})
    indices = _record(**{"i%d" % i: integer for i in range(dimension)})
    geometry = _record(coordinates=array(cell_domain, coordinates), indices=array(cell_domain, indices),
        measure=array(cell_domain, real), volume_fraction=array(cell_domain, real),
        is_valid=array(cell_domain, boolean), is_active=array(cell_domain, boolean),
        is_covered=array(cell_domain, boolean))
    patch_fields = {name: integer for name in ("owner_rank", "global_patch", "global_offset", "count")}
    patch_fields["is_replica"] = boolean
    patch_fields.update({"%s%d" % (name, i): integer
                         for name in ("lo", "hi", "ghost") for i in range(dimension)})
    maps = _record(patch_instance=patch_domain, patches=array(patch_domain, _record(**patch_fields)))
    raw = product_signature((("instance", cell_domain), ("fields", fields),
                             ("geometry", geometry), ("maps", maps), ("histories", _record())))
    bindings = {("fields", "values"): "cells", ("maps", "patches"): "patches"}
    bindings.update({("geometry", name): "cells" for name, _ in geometry["members"]})
    return BundleSchema(ValueType(raw), domains={"cells": ("instance",),
        "patches": ("maps", "patch_instance")}, bindings=bindings).value_type.signature


def _domains(signature):
    validate_bundle_signature(signature)
    members = dict(signature["members"])
    cells = members["instance"]
    patches = dict(members["maps"]["members"])["patch_instance"]
    for domain, ownership in ((cells, "contiguous_distributed"), (patches, "replicated")):
        if domain.get("vtype") != "index_set" or domain.get("ownership") != ownership:
            raise TypeError("Field Bundle has an incompatible storage-domain ownership")
        _capacity(domain.get("capacity"), "Field Bundle domain")
        if type(domain.get("family")) is not int or domain["family"] < 0:
            raise TypeError("Field Bundle requires authenticated domain families")
    return cells, patches


def _field_topology(program, field, topology):
    require_owned(program, field, "Field representation")
    require_owned(program, topology, "Field topology", vtype="field_topology")
    signature = validate_topology_signature(value_signature(topology))
    if field.vtype not in ("state", "scalar_field") or not same_signature(
            value_signature(field), signature["field_signature"]):
        raise TypeError("Field representation requires its exact topology field authority")
    return signature


def validate_field_representation(value):
    op, attrs, inputs = value.op, value.attrs, value.inputs
    if op not in FIELD_REPRESENTATION_OPS:
        raise TypeError("not a Field representation operation")
    if op == "field_storage_allocate":
        return validate_field_storage_allocation(value), None
    if attrs.get("storage_policy_binding") != "explicit" or any(
            type(attrs.get(key)) is not int or not 0 < attrs[key] < 2**63
            for key in ("storage_bytes", "storage_versions")):
        raise ValueError("Field representation lost its explicit resource budget")
    exporting = op == "field_bundle_export"
    if len(inputs) not in ((2, 3) if exporting else (3,)):
        raise TypeError("Field representation has an incomplete input contract")
    field, topology = inputs[:2] if exporting else inputs[1:]
    top = validate_topology_signature(value_signature(topology))
    if field.vtype not in ("state", "scalar_field") or not same_signature(
            value_signature(field), top["field_signature"]):
        raise TypeError("Field representation changed its exact field/topology pair")
    bundle = value_signature(value) if exporting else value_signature(inputs[0])
    cells, patches = _domains(bundle)
    expected = field_bundle_signature(dimension=top["dimension"],
        components=len(field.space.components), cell_domain=cells, patch_domain=patches)
    if not same_signature(bundle, expected):
        raise TypeError("Field representation changed its complete values/geometry/maps schema")
    if exporting and len(inputs) == 3:
        family = value_signature(inputs[2])
        if family["vtype"] == "product":
            before_cells, before_patches = _domains(family)
            if not same_signature(cells, before_cells) or not same_signature(patches, before_patches):
                raise TypeError("Field Bundle changed a family witness")
        elif not same_signature(cells, family):
            raise TypeError("Field Bundle changed its distributed family witness")
    if not exporting and not same_signature(value_signature(value), value_signature(field)):
        raise TypeError("Field import changed its explicit template authority")
    return top, bundle


def validate_field_storage_allocation(value):
    from pops.model.spaces import FieldSpace

    if value.op != "field_storage_allocate" or len(value.inputs) != 3:
        raise TypeError("Field storage allocation requires prototype, target topology and Float64 initial")
    prototype, topology, initial = value.inputs
    top = validate_topology_signature(value_signature(topology))
    if top["side"] != "target":
        raise TypeError("Field storage allocation requires a target topology")
    if (prototype.vtype != "scalar_field" or type(prototype.space) is not FieldSpace
            or prototype.block is None
            or not same_signature(value_signature(prototype), top["field_signature"])):
        raise TypeError("Field storage allocation requires its exact scalar-field topology authority")
    if initial.vtype != "scalar":
        raise TypeError("Field storage allocation requires an owned Float64 initial value")
    if (value.attrs.get("storage_allocation") != "pops.field-storage-topology@1"
            or type(value.attrs.get("ncomp")) is not int
            or value.attrs["ncomp"] != len(prototype.space.components)):
        raise TypeError("Field storage allocation changed its exact component storage")
    for key in ("scalar_storage", "scalar_storage_component", "spatial_port_storage"):
        expected = prototype.attrs.get(key)
        if key == "spatial_port_storage" and prototype.attrs.get("scalar_storage") is None:
            expected = "pops.computed-cell-field@1"
        if not same_signature(value.attrs.get(key), expected):
            raise TypeError("Field storage allocation changed its declared storage authority")
    if not same_signature(value_signature(value), value_signature(prototype)):
        raise TypeError("Field storage allocation changed its exact prototype signature")
    return top


@atomic_authoring
def allocate_field_storage(program, prototype, topology, initial, *, ncomp=None, name=None):
    """Allocate target storage; the authored initializer is the only numerical input."""
    from pops.model.spaces import FieldSpace
    from pops.time.stencil import StencilAccess

    _field_topology(program, prototype, topology)
    require_owned(program, initial, "Field storage Float64 initial", vtype="scalar")
    if prototype.vtype != "scalar_field" or type(prototype.space) is not FieldSpace:
        raise TypeError("Field storage allocation requires a typed scalar-field prototype")
    width = len(prototype.space.components)
    if ncomp is not None and (type(ncomp) is not int or ncomp != width):
        raise ValueError("Field storage allocation must preserve the exact component shape")
    attrs = {"storage_allocation": "pops.field-storage-topology@1", "ncomp": width,
             "stencil_access": StencilAccess.pointwise()}
    # These identify declared storage; equations and source flux expressions
    # are deliberately not inherited by a freshly initialized allocation.
    for key in ("scalar_storage", "scalar_storage_component", "spatial_port_storage"):
        if key in prototype.attrs:
            attrs[key] = prototype.attrs[key]
    if attrs.get("scalar_storage") is None:
        attrs["spatial_port_storage"] = "pops.computed-cell-field@1"
    value = new_typed(program, value_signature(prototype), "field_storage_allocate",
                      (prototype, topology, initial), attrs, point=prototype.point, name=name)
    validate_field_storage_allocation(value)
    return value


def validate_topology_origin(value):
    """Prove native issuance through structural copies without making views persistable.

    The worklist follows the selected Product/Variant path, all branch alternatives
    and both loop entry/body values. It is iterative and rejects cycles or arbitrary
    constructors even if their output metadata claims the right type.
    """
    expected = validate_topology_signature(value_signature(value))
    active, checked = set(), set()
    pending = [(value, (), False)]
    while pending:
        node, path, exiting = pending.pop()
        key = (id(node), path)
        if exiting:
            active.remove(key)
            checked.add(key)
            continue
        if key in checked:
            continue
        if key in active:
            raise TypeError("FieldTopology origin contains a cyclic structural copy")
        if node.vtype == "iteration_result":
            signature = {"vtype": "result", "success": product_signature((
                ("state", node.attrs["state_signature"]),
                ("iterations", scalar_signature("int64"))))}
        else:
            signature = value_signature(node)
        selected = signature
        for part in path:
            if selected["vtype"] in ("product", "variant"):
                selected = dict(selected["members"])[part]
            elif selected["vtype"] == "result" and part == "ok":
                selected = selected["success"]
            else:
                raise TypeError("FieldTopology origin has an invalid structural projection")
        if not same_signature(selected, expected):
            raise TypeError("FieldTopology copy changed its exact issued scientific type")
        active.add(key)
        pending.append((node, path, True))
        op, attrs, inputs = node.op, node.attrs, node.inputs
        dependencies = []
        if op in TOPOLOGY_OPS:
            if (path or len(inputs) != 2 or inputs[1].vtype != "int64"
                    or attrs.get("side") != expected["side"]
                    or attrs.get("dimension") != expected["dimension"]
                    or op != "continuation_%s_topology" % expected["side"]
                    or not same_signature(value_signature(inputs[0]), expected["field_signature"])):
                raise TypeError("FieldTopology issuer changed its source/target contract")
        elif op == "product" and path and node.vtype == "product":
            fields = tuple(attrs["fields"])
            if len(fields) != len(inputs) or len(set(fields)) != len(fields):
                raise TypeError("FieldTopology Product origin has invalid member bindings")
            dependencies.append((inputs[fields.index(path[0])], path[1:]))
        elif op == "product_get" and len(inputs) == 1:
            dependencies.append((inputs[0], (attrs["field"],) + path))
        elif op == "region_argument" and len(inputs) == 1:
            dependencies.append((inputs[0], path))
        elif op in ("branch", "result_match"):
            dependencies.extend((attrs[name], path) for name in ("true_result", "false_result"))
        elif op == "structured_fold":
            dependencies.extend(((inputs[0], path), (attrs["body"], path)))
        elif op in ("result_success", "variant_require_success", "iteration_success"):
            dependencies.append((inputs[0], ("ok",) + path))
        elif op in ("result_payload", "iteration_payload"):
            dependencies.append((inputs[0], (attrs["variant"],) + path))
        elif op == "variant_payload":
            dependencies.append((inputs[0], (attrs["tag"],) + path))
        elif op == "variant_construct" and path:
            if attrs["tag"] == path[0]:
                dependencies.append((inputs[0], path[1:]))
            # A projection of the inactive arm necessarily rejects before use.
        elif op in ("result_ok", "result_error") and path and path[0] == "ok":
            dependencies.append((inputs[0], path[1:]))
        elif op == "capture_result" and path and path[0] == "ok":
            wrapped = attrs.get("body_is_result") or attrs.get("body_is_variant_result")
            dependencies.append((attrs["body"], (("ok",) if wrapped else ()) + path[1:]))
        elif op == "structured_iterate" and path[:2] == ("ok", "state"):
            wrapped = attrs.get("body_is_result") or attrs.get("body_is_variant_result")
            dependencies.extend(((inputs[0], path[2:]),
                (attrs["body"], (("ok",) if wrapped else ()) + path[2:])))
        elif op != "variant_unreachable":
            raise TypeError("FieldTopology can only be issued by the regrid transaction and copied structurally")
        pending.extend((child, location, False) for child, location in reversed(dependencies))


def validate_field_representation_program(program):
    pending = list(program._values)
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.vtype == "field_topology":
            validate_topology_origin(value)
        if value.op in FIELD_REPRESENTATION_OPS:
            validate_field_representation(value)
        pending.extend(program._subblock_value_refs(value))


class _ProgramFieldRepresentation(_ProgramBase):
    def regrid_topology(self, field, *, level, side, dimension):
        """Request an ephemeral source/target topology from the active regrid transaction."""
        from pops.time._program.continuation_regrid import topology
        return topology(self, field, level=level, side=side, dimension=dimension)

    def regrid_source(self, level):
        """Read one immutable source image inside the explicit regrid transaction."""
        from pops.time._program.continuation_regrid import source
        return source(self, level)

    def regrid_base(self, source_level):
        """Choose explicitly the source whose assessed descendant this target becomes."""
        from pops.time._program.continuation_regrid import base
        return base(self, source_level)

    @atomic_authoring
    def field_bundle(self, field, *, topology, cell_capacity, patch_capacity, family=None):
        """Export distributed valid/ghost storage with an explicit complete finite schema."""
        top = _field_topology(self, field, topology)
        cells, patches = _capacity(cell_capacity, "cell_capacity"), _capacity(patch_capacity, "patch_capacity")
        attrs = _policy(self)
        witness = None
        cell_domain = patch_domain = None
        if family is not None:
            require_owned(self, family, "Field Bundle family")
            fs = value_signature(family)
            if fs["vtype"] == "product":
                cell_domain, patch_domain = _domains(fs)
            elif fs["vtype"] == "index_set" and fs.get("ownership") == "contiguous_distributed":
                cell_domain = fs
            else:
                raise TypeError("family requires an owned Field Bundle or distributed IndexSet")
            if cell_domain["capacity"] != cells or (patch_domain is not None and patch_domain["capacity"] != patches):
                raise ValueError("Field representation cannot change family capacities")
            witness = family
        def fresh(capacity, ownership):
            identity = self._next_finite_domain_identity
            self._next_finite_domain_identity += 1
            return dict(vtype="index_set", family=identity, capacity=capacity,
                        ownership=ownership, point=None)
        if cell_domain is None:
            cell_domain = fresh(cells, "contiguous_distributed")
        if patch_domain is None:
            patch_domain = fresh(patches, "replicated")
        signature = field_bundle_signature(dimension=top["dimension"],
            components=len(field.space.components), cell_domain=cell_domain, patch_domain=patch_domain)
        inputs = (field, topology) if witness is None else (field, topology, witness)
        return new_typed(self, signature, "field_bundle_export", inputs, attrs, point=field.point)

    @atomic_authoring
    def bundle_field(self, bundle, *, template, topology):
        """Import authored values into exact target storage, with no implicit remap or interpolation."""
        _field_topology(self, template, topology)
        require_owned(self, bundle, "Field Bundle input", vtype="product")
        attrs = _policy(self)
        value = new_typed(self, value_signature(template), "field_bundle_import",
                          (bundle, template, topology), attrs, point=template.point)
        validate_field_representation(value)
        return value
