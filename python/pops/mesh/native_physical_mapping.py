"""Package typed physical-map data for the shared native reduction mechanism."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any


def physical_map_identity(physical: Any) -> str:
    """The exact Python physical descriptor authenticated by the native integral callback."""
    from pops.identity import make_identity
    from pops.mesh.physical_mapping import PhysicalSupportMap
    if type(physical) is not PhysicalSupportMap:
        raise TypeError("physical map identity requires an exact PhysicalSupportMap")
    return make_identity("physical-support-map", physical.to_data()).token


def _native_source(physical: Any, manifest: Any) -> bytes:
    """Emit immutable descriptor data and ABI wiring; the numerical kernel is shared."""
    contract = physical.native_contract()
    weights = []
    cells, offsets = [0] * 3, [0] * 3
    for reduction in physical.reductions:
        cells[reduction.axis] = reduction.cells
        offsets[reduction.axis] = len(weights)
        for exact in reduction.weights:
            value = float(exact)
            if not math.isfinite(value) or (exact and value == 0):
                raise ValueError("physical quadrature weight must be representable as finite float64")
            weights.append(value)
    def array(values: Any, fill: int = 0) -> str:
        row = tuple(values)
        return "{" + ", ".join(str(value) for value in row + (fill,) * (3 - len(row))) + "}"
    descriptor = ",\n  ".join((str(physical.native_dimension), str(physical.operation_abi),
        array(contract["physical_source_to_target"], -1),
        array(contract["physical_source_active"]), array(contract["physical_target_active"]),
        array(cells), array(offsets), "weights", str(len(weights))))
    weight_data = ", ".join(repr(value) for value in weights) or "0.0"
    lines = [
        '#include <pops/runtime/dynamic/physical_support_transfer.hpp>',
        '#include <cstring>',
        '#include <algorithm>',
        'namespace {',
        'const double weights[] = {' + weight_data + '};',
        'const pops::component::PhysicalSupportTransfer descriptor = {\n  ' + descriptor + '};',
        'int apply(void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* status) {',
        '  return pops::component::apply_physical_support_transfer(descriptor, request, status);',
        '}',
        'int integral(void*, const PopsTransferIntegralRequestV2* request, PopsComponentStatusV1* status) {',
        '  if (!request || request->struct_size != sizeof(PopsTransferIntegralRequestV2) ||',
        '      request->dimension != descriptor.dimension || request->operation != descriptor.operation ||',
        '      !request->physical_contract_identity || std::strcmp(request->physical_contract_identity, ' +
            json.dumps(physical_map_identity(physical)) + ') != 0) {',
        '    if (status) *status = {sizeof(PopsComponentStatusV1), 2, POPS_COMPONENT_ABORT_RUN_V1,',
        '                           "physical Transfer V2 integral contract mismatch"};',
        '    return 2;',
        '  }',
        '  pops::component::PhysicalSupportIntegral operation{};',
        '  operation.dimension = request->dimension;',
        '  operation.weights = request->axis_weights;',
        '  operation.weight_count = request->weight_count;',
        '  std::copy_n(request->weight_offsets, 3, operation.weight_offsets);',
        '  return pops::component::apply_physical_support_integral(operation, request->source, request->destination, status);',
        '}',
        'const PopsTransferApiV2 table = {',
        '  {sizeof(PopsTransferApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,',
        '   POPS_NATIVE_INTERFACE_TRANSFER_V2, 2, nullptr, nullptr}, &apply, &integral};',
        'const PopsComponentInterfaceEntryV1 entry = {',
        '  POPS_NATIVE_INTERFACE_TRANSFER_V2, 2, sizeof(PopsTransferApiV2), &table};',
        'const PopsComponentApiV1 component = {',
        '  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1, POPS_ABI_KEY_LITERAL,',
        '  POPS_COMPONENT_CATALOG_SHA256_V1, ' + json.dumps(manifest.component_id) + ', ' +
            json.dumps(manifest.semantic_digest.token) + ', ' + json.dumps(manifest.manifest_digest.token) + ', 1, &entry};',
        '}',
        'extern "C" const PopsComponentApiV1* pops_component_interface_v1() { return &component; }',
    ]
    return ("\n".join(lines) + "\n").encode()


def native_physical_mapping(requirement: Any, directory: Any) -> Any:
    """Materialize an authenticated Transfer over explicit reduction/extension data.

    Returned provider and ``provider.component`` are passed to LayoutPlan.resolve
    and pops.resolve respectively. No numerical field data passes through Python.
    """
    from pops import interfaces
    from pops.external import SourceComponentPackage, build_source_package_manifest, load
    from pops.model import ComponentManifest
    from ._layout_plan_contracts import LayoutMappingRequirement
    from .layout_mapping import NativeLayoutMapping
    if type(requirement) is not LayoutMappingRequirement or requirement.physical_map is None:
        raise TypeError("native_physical_mapping requires an explicit physical map requirement")
    physical = requirement.physical_map
    interface = interfaces.Transfer
    manifest = ComponentManifest(
        uri="pops://physical-maps/" + requirement.qualified_id.rsplit("::", 1)[-1],
        component_type="transfer", version="2.0.0", facets=interface.facets,
        signature={"generic": True, "native_interface": interface.signature_declaration(),
                   "physical_map": physical.to_data(),
                   "physical_integral": {"interface_version": 2,
                                         "physical_contract_identity": physical_map_identity(physical)}},
        interfaces=interface.manifest_declarations(),
        target={"variants": [{"dimension": physical.native_dimension, "scalar": "float64", "device": "cpu",
                              "features": []}]},
        entry_points={"interface_table": "pops_component_interface_v1"})
    source = _native_source(physical, manifest)
    root = Path(directory) / requirement.qualified_id.rsplit("::", 1)[-1]
    root.mkdir(parents=True, exist_ok=True)
    filename = "physical_map.cpp"
    (root / filename).write_bytes(source)
    package = build_source_package_manifest(components={"map": manifest},
        payloads={filename: ("source", source)})
    path = root / "physical-map.pops.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    loaded_package = load(path)
    if type(loaded_package) is not SourceComponentPackage:
        raise TypeError("physical map source manifest did not load as a source component package")
    component = loaded_package.require("map", interface=interface)()
    return NativeLayoutMapping(component, (requirement,))
