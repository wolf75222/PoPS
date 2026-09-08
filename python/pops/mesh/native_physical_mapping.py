"""Build the authenticated C++/Kokkos provider for an exact physical map."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any


def native_physical_mapping(requirement: Any, directory: Any) -> Any:
    """Materialize a reproducible native Transfer package for an authored physical map.

    Returned provider and ``provider.component`` are passed to LayoutPlan.resolve
    and pops.resolve respectively. No numerical field data passes through Python.
    """
    from pops import interfaces
    from pops.external import build_source_package_manifest, load
    from pops.model import ComponentManifest
    from ._layout_plan_contracts import LayoutMappingRequirement
    from .layout_mapping import NativeLayoutMapping
    if type(requirement) is not LayoutMappingRequirement or requirement.physical_map is None:
        raise TypeError("native_physical_mapping requires an explicit physical map requirement")
    physical = requirement.physical_map
    interface = interfaces.Transfer
    manifest = ComponentManifest(
        uri="pops://physical-maps/" + requirement.qualified_id.rsplit("::", 1)[-1],
        component_type="transfer", version="1.0.0", facets=interface.facets,
        signature={"generic": True, "native_interface": interface.signature_declaration(),
                   "physical_map": physical.to_data()},
        interfaces=interface.manifest_declarations(),
        target={"variants": [{"dimension": 2, "scalar": "float64", "device": "cpu",
                              "features": []}]},
        entry_points={"interface_table": "pops_component_interface_v1"})
    quadrature = physical.quadrature
    weight = 1.0 if quadrature is None else float(quadrature.weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("physical quadrature weight must be representable as positive finite float64")
    cells = 0 if quadrature is None else quadrature.cells
    source = _SOURCE.replace("@OP@", str(physical.operation_abi)).replace("@WEIGHT@", repr(weight))
    source = source.replace("@CELLS@", str(cells))
    source = source.replace("@COMPONENT@", json.dumps(manifest.component_id))
    source = source.replace("@SEMANTIC@", json.dumps(manifest.semantic_digest.token))
    source = source.replace("@MANIFEST@", json.dumps(manifest.manifest_digest.token)).encode()
    root = Path(directory) / requirement.qualified_id.rsplit("::", 1)[-1]
    root.mkdir(parents=True, exist_ok=True)
    filename = "physical_map.cpp"
    (root / filename).write_bytes(source)
    package = build_source_package_manifest(components={"map": manifest},
        payloads={filename: ("source", source)})
    path = root / "physical-map.pops.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    component = load(path).require("map", interface=interface)()
    return NativeLayoutMapping(component, (requirement,))


_SOURCE = r"""
#include <pops/runtime/config/generated_component_abi.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstddef>
#include <cstdint>
#include <cmath>
namespace {
int apply(void*, const PopsTransferRequestV1* r, PopsComponentStatusV1* status) {
  if (!r || !status || r->dimension != 2 || r->operation != @OP@ ||
      r->source.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      r->destination.memory_space != POPS_MEMORY_SPACE_HOST_V1) return 2;
  if (r->struct_size < sizeof(PopsTransferRequestV1)) return 2;
  const auto s = r->source;
  const auto d = r->destination;
  if (s.extents[0] != d.extents[0] ||
      (@OP@ == 2 && (d.extents[1] != 1 || s.extents[1] != @CELLS@)) ||
      (@OP@ == 3 && s.extents[1] != 1)) return 3;
  for (int axis = 0; axis < 2; ++axis)
    if (s.ghost_lower[axis] || s.ghost_upper[axis] ||
        d.ghost_lower[axis] || d.ghost_upper[axis]) return 3;
  const auto* source = static_cast<const double*>(s.data);
  auto* destination = static_cast<double*>(d.data);
  const std::size_t count = d.component_count * d.extents[0] * d.extents[1];
  using Policy = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::size_t>>;
  Kokkos::parallel_for("pops_explicit_physical_map", Policy(0, count),
    KOKKOS_LAMBDA(const std::size_t i) {
      const auto x = i % d.extents[0];
      const auto v = (i / d.extents[0]) % d.extents[1];
      const auto c = i / (d.extents[0] * d.extents[1]);
      double value = 0.0;
      if (@OP@ == 2) {
        for (std::size_t j = 0; j < s.extents[1]; ++j)
          value += source[c * s.component_stride + x * s.axis_strides[0] +
                          j * s.axis_strides[1]] * @WEIGHT@;
      } else {
        value = source[c * s.component_stride + x * s.axis_strides[0]];
      }
      destination[c * d.component_stride + x * d.axis_strides[0] +
                  v * d.axis_strides[1]] = value;
    });
  Kokkos::fence();
  int invalid = 0;
  Kokkos::parallel_reduce("pops_physical_map_finite", Policy(0, count),
    KOKKOS_LAMBDA(const std::size_t i, int& bad) {
      const auto x = i % d.extents[0];
      const auto v = (i / d.extents[0]) % d.extents[1];
      const auto c = i / (d.extents[0] * d.extents[1]);
      const double value = destination[c * d.component_stride + x * d.axis_strides[0] +
                                       v * d.axis_strides[1]];
      if (!Kokkos::isfinite(value)) bad = 1;
      else if (bad < 0) bad = 0;
    }, Kokkos::Max<int>(invalid));
  if (invalid) return 4;
  *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}
const PopsTransferApiV1 table = {
  {sizeof(PopsTransferApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
   POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, nullptr, nullptr}, &apply};
const PopsComponentInterfaceEntryV1 entry = {
  POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1), &table};
const PopsComponentApiV1 component = {
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1, POPS_ABI_KEY_LITERAL,
  POPS_COMPONENT_CATALOG_SHA256_V1, @COMPONENT@, @SEMANTIC@, @MANIFEST@, 1, &entry};
}
extern "C" const PopsComponentApiV1* pops_component_interface_v1() { return &component; }
"""
