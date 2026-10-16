"""Lower Field representation ports to the authenticated native topology adapter."""
from __future__ import annotations

import json
from pops._structural_data import canonical_json
from pops.time.canonical_data import _json_ready
from pops.time.field_topology import field_components

from pops.time._program.field_representation import validate_field_representation
from pops.time.value_signatures import value_signature

from .program_emit_versioned import pack_owned, unpack_owned, schema_cpp


def emit_field_representation(program, value, var, model, lines, prelude,
                              block_idx, field_plans, target):
    del program, model, prelude, field_plans
    if target != "amr_system":
        raise ValueError("FieldTopology is issued by an authenticated AMR regrid transaction")
    topology_type, bundle_type = validate_field_representation(value)
    prefix = "field_representation_%d" % value.id
    lines.append('static_assert(pops::kNativeDimension == %d, "FieldTopology dimension mismatch");'
                 % topology_type["dimension"])
    allocation = value.op == "field_storage_allocate"
    field_node = value.inputs[0] if allocation or value.op == "field_bundle_export" else value.inputs[1]
    topology_node = value.inputs[1] if allocation or value.op == "field_bundle_export" else value.inputs[2]
    signature = value_signature(field_node)
    authority = signature.get("field_authority")
    components = field_components(signature)
    topology = var[topology_node.id]
    if authority is None:
        if field_node.block not in block_idx:
            raise ValueError("Field representation has no resolved scientific storage owner")
        owner_test = "%s.program_block() != %d || !%s.field_provider().empty()" % (
            topology, block_idx[field_node.block], topology)
    else:
        contract = canonical_json(_json_ready(authority))
        owner_test = "%s.program_block() != -1 || %s.field_provider() != %s" % (
            topology, topology, json.dumps(contract))
    lines.append('ctx.continuation_local_phase([&] { ctx.require_field_topology(%s); '
                 'if (%s || %s.components() != %d) '
                 'throw std::logic_error("Field representation scientific owner mismatch"); }, '
                 '"Field representation owner validation");'
                 % (topology, owner_test, topology, components))
    if allocation:
        owned = prefix + "_owned"
        lines.append("const auto %s = ctx.allocate_field_storage(%s, %s, %s);"
                     % (owned, var[field_node.id], topology, var[value.inputs[2].id]))
        lines.append('ctx.continuation_local_phase([&] { if (!%s) throw std::logic_error("Field allocation returned no owned storage"); }, "Field allocation storage validation");' % owned)
        var[value.id] = "(*%s)" % owned
        return
    capacity = value.attrs["storage_bytes"]
    versions = value.attrs["storage_versions"]
    lines.append("ctx.distributed_registry().configure(%d, %d);" % (capacity, versions))
    schema = prefix + "_schema"
    lines.append("const auto %s = ctx.continuation_local_phase([&] { return %s; }, "
                 '"Field representation schema preparation");' % (schema, schema_cpp(bundle_type, block_idx)))
    if value.op == "field_bundle_export":
        field, topology = (var[item.id] for item in value.inputs[:2])
        if len(value.inputs) == 3:
            witness = value.inputs[2]
            signature, tokens = value_signature(witness), var[witness.id]
            if signature["vtype"] == "product":
                members = dict(signature["members"])
                pack_owned(members["instance"], tokens["instance"], lines,
                           prefix + "_cell_witness", block_idx, capacity=capacity)
                patch = dict(members["maps"]["members"])["patch_instance"]
                pack_owned(patch, tokens["maps"]["patch_instance"], lines,
                           prefix + "_patch_witness", block_idx, capacity=capacity)
            else:
                pack_owned(signature, tokens, lines, prefix + "_cell_witness",
                           block_idx, capacity=capacity)
        reference = prefix + "_exported"
        lines.append("const auto %s = ctx.export_field_bundle(%s, %s, %s, %d);"
                     % (reference, field, topology, schema, capacity))
        var[value.id] = unpack_owned(bundle_type, reference, lines, prefix + "_values",
                                     capacity=capacity, block_indices=block_idx)
    else:
        bundle, template, topology = value.inputs
        reference = pack_owned(bundle_type, var[bundle.id], lines, prefix + "_bundle",
                               block_idx, capacity=capacity)
        owned = prefix + "_owned"
        lines.append("const auto %s = ctx.import_field_bundle(%s, %s, %s, %s, %d);"
                     % (owned, reference, var[template.id], var[topology.id], schema, capacity))
        lines.append('ctx.continuation_local_phase([&] { if (!%s) throw std::logic_error("Field import returned no owned storage"); }, "Field import storage validation");' % owned)
        var[value.id] = "(*%s)" % owned
