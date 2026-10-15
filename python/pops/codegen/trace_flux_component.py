"""Compile a closed scalar trace DAG through the real NumericalFlux component ABI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from pops.identity import canonical_bytes


def make_component(roots, evidence, dimension, components):
    from pops import interfaces
    from pops.model import ComponentManifest
    from pops.identity.semantic import semantic_value
    from pops.external.packages import (
        SourceComponentPackage,
        PackagePayload,
        build_source_package_manifest,
    )
    from pops.external._package_data import content_identity, package_identity

    evidence = semantic_value(evidence, where="TraceFlux")
    token = hashlib.sha256(canonical_bytes(evidence)).hexdigest()
    interface = interfaces.NumericalFlux
    manifest = ComponentManifest(
        uri="pops://generated/trace-flux/" + token,
        component_type=interface.name,
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "state_components": components,
            "native_interface": interface.signature_declaration(),
            "trace_flux": evidence,
        },
        interfaces=interface.manifest_declarations(),
        target={
            "variants": [
                {"dimension": dimension, "scalar": "float64", "device": device, "features": []}
                for device in ("cpu", "cuda", "hip", "sycl")
            ]
        },
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = emit_source(manifest, roots, dimension, components).encode()
    data = build_source_package_manifest(
        components={"trace_flux": manifest}, payloads={"trace_flux.cpp": ("source", source)}
    )
    package = SourceComponentPackage(
        (manifest,),
        MappingProxyType({"trace_flux": manifest.component_id}),
        (PackagePayload("trace_flux.cpp", "source", content_identity("source", source), source),),
        package_identity(data),
        Path("trace-flux-" + token + ".pops.json"),
    )
    return package.require("trace_flux", interface=interface)()


def emit_source(manifest, roots, dimension, components):
    from pops.codegen.cpp_writer import _cse_emit

    bindings = {"trace_time": "time", "trace_dt": "dt"}
    for side, width in (("left", components), ("right", components), ("normal", dimension)):
        bindings.update({f"trace_{side}_{i}": f"read({side}, point, {i})" for i in range(width)})
    body, values, observed = _cse_emit(
        roots, "double", "    ", materialize_all=True, return_names=True, scalar_bindings=bindings
    )
    checks = " && ".join(f"Kokkos::isfinite({v})" for v in (*observed, *values))
    bindings_data = manifest.to_data()
    replacements = {
        "@DIM@": str(dimension),
        "@WIDTH@": str(components),
        "@BODY@": "\n".join(body),
        "@CHECK@": f"({checks}) && {values[-1]} >= 0.0",
        "@STORES@": "\n".join(
            f"    scratch(point * {components} + {i}) = {v};"
            for i, v in enumerate(values[:components])
        )
        + f"\n    bounds(point) = {values[-1]};",
        "@PARAMS@": json.dumps(
            json.dumps(bindings_data["parameters"], sort_keys=True, separators=(",", ":"))
        ),
        "@TARGET@": json.dumps(
            json.dumps(bindings_data["target"], sort_keys=True, separators=(",", ":"))
        ),
        "@COMPONENT@": json.dumps(manifest.component_id),
        "@SEMANTIC@": json.dumps(manifest.semantic_digest.token),
        "@MANIFEST@": json.dumps(manifest.manifest_digest.token),
    }
    source = _TEMPLATE
    for key, value in replacements.items():
        source = source.replace(key, value)
    return source


_TEMPLATE = r"""#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/abi_key.hpp>
#include <pops/runtime/dynamic/component_consumers.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
namespace pops_generated_trace_flux {
using Exec = Kokkos::DefaultExecutionSpace;
using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::size_t>>;
using Values = Kokkos::View<double*>;
using Flags = Kokkos::View<int*>;
constexpr std::size_t width = @WIDTH@;
template <class View>
KOKKOS_INLINE_FUNCTION std::ptrdiff_t offset(const View& view, std::size_t point, std::size_t component) {
  std::ptrdiff_t out = static_cast<std::ptrdiff_t>(component) * view.component_stride;
  for (int axis = @DIM@ - 1; axis >= 0; --axis) {
    out += static_cast<std::ptrdiff_t>(point % view.extents[axis]) * view.axis_strides[axis];
    point /= view.extents[axis];
  }
  return out;
}
KOKKOS_INLINE_FUNCTION double read(const PopsConstFieldViewV1& v, std::size_t point, std::size_t component) {
  return static_cast<const double*>(v.data)[offset(v, point, component)];
}
struct Evaluate {
  PopsConstFieldViewV1 left, right, normal;
  double time, dt;
  Values scratch, bounds;
  Flags invalid;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t point) const {
@BODY@
    invalid(point) = (@CHECK@) ? 0 : 1;
@STORES@
  }
};
struct Publish {
  Values scratch;
  PopsFieldViewV1 output;
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t point) const {
    for (std::size_t component = 0; component < width; ++component)
      static_cast<double*>(output.data)[offset(output, point, component)] = scratch(point * width + component);
  }
};
template <class View>
void check_span(const View& view) {
  std::size_t span = 0;
  const auto maximum = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  auto add = [&](std::size_t n, std::ptrdiff_t signed_stride) {
    if (signed_stride <= 0) throw std::invalid_argument("TraceFlux stride");
    const auto stride = static_cast<std::size_t>(signed_stride);
    if (n && stride > (maximum-span)/n) throw std::overflow_error("TraceFlux span");
    span += n*stride;
  };
  add(view.component_count-1, view.component_stride);
  for (int axis=0; axis<@DIM@; ++axis) add(view.extents[axis]-1, view.axis_strides[axis]);
}
int prepare(const PopsComponentPrepareRequestV1* request, void** state, PopsComponentStatusV1* status) {
  if (!request || !state || !status) return 1;
  *state = nullptr;
  try {
    if (!request->parameters_json || !request->target_json ||
        std::strcmp(request->parameters_json, @PARAMS@) || std::strcmp(request->target_json, @TARGET@))
      throw std::invalid_argument("TraceFlux preparation identity");
    *state = new bool(true);
    *status = {sizeof(*status), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
    return 0;
  } catch (...) {
    *status = {sizeof(*status), 1, POPS_COMPONENT_ABORT_RUN_V1, "TraceFlux preparation failed"};
    return 1;
  }
}
void destroy(void* state) { delete static_cast<bool*>(state); }
int evaluate(void* state, const PopsNumericalFluxRequestV1* request, PopsNumericalFluxResultV1* result) {
  if (!result) return 1;
  try {
    if (!state || !*static_cast<bool*>(state) || !request || request->struct_size != sizeof(*request) ||
        result->struct_size != sizeof(*result) || !result->stability_bounds || !result->actions)
      throw std::invalid_argument("TraceFlux request");
    const auto& r = *request;
    namespace component = pops::component;
    component::validate_execution_context(r.execution);
    component::validate_logical_time(r.logical_time);
    component::validate_execution_field<@DIM@>(r.execution, r.left, "TraceFlux left");
    component::validate_execution_field<@DIM@>(r.execution, r.right, "TraceFlux right");
    component::validate_execution_field<@DIM@>(r.execution, r.normals, "TraceFlux normal");
    component::validate_execution_field<@DIM@>(r.execution, result->normal_flux, "TraceFlux output");
    if (r.left.scalar_type != POPS_SCALAR_FLOAT64_V1 || r.left.component_count != width ||
        r.normals.component_count != @DIM@ || !component::same_field_domain(r.left, r.right) ||
        !component::same_field_domain(r.left, result->normal_flux) ||
        !component::same_spatial_domain(r.left, r.normals))
      throw std::invalid_argument("TraceFlux trace shapes");
    constexpr bool host = Kokkos::SpaceAccessibility<Exec, Kokkos::HostSpace>::accessible;
    if ((host && r.execution.memory_space == POPS_MEMORY_SPACE_DEVICE_V1) ||
        (!host && r.execution.memory_space == POPS_MEMORY_SPACE_HOST_V1))
      throw std::invalid_argument("TraceFlux execution memory differs from compiled Kokkos backend");
    check_span(r.left); check_span(r.right); check_span(r.normals); check_span(result->normal_flux);
    const std::size_t count = component::field_point_count(r.left);
    if (count > std::numeric_limits<std::size_t>::max() / width / sizeof(double))
      throw std::overflow_error("TraceFlux scratch size");
    Values scratch("TraceFlux values", count*width);
    Values bounds("TraceFlux bounds", count);
    Flags invalid("TraceFlux invalid", count);
    Kokkos::parallel_for("TraceFlux evaluate authored DAG", Policy(0,count),
        Evaluate{r.left,r.right,r.normals,r.logical_time.physical_time,r.logical_time.dt,scratch,bounds,invalid});
    auto errors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, invalid);
    for (std::size_t i=0; i<count; ++i)
      if (errors(i)) throw std::domain_error("TraceFlux nonfinite expression or negative bound");
    auto host_bounds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, bounds);
    Kokkos::parallel_for("TraceFlux publish validated batch", Policy(0,count), Publish{scratch,result->normal_flux});
    Kokkos::fence("TraceFlux publication");
    for (std::size_t i=0; i<count; ++i) {
      result->stability_bounds[i] = host_bounds(i);
      result->actions[i] = POPS_COMPONENT_CONTINUE_V1;
    }
    result->status = {sizeof(result->status), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
    return 0;
  } catch (...) {
    result->status = {sizeof(result->status), 1, POPS_COMPONENT_REJECT_STEP_V1,
                      "TraceFlux expression or execution contract failed"};
    return 1;
  }
}
const PopsNumericalFluxApiV1 flux_table = {{sizeof(PopsNumericalFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, &prepare, &destroy}, &evaluate};
const PopsComponentInterfaceEntryV1 entry = {POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1,1,sizeof(PopsNumericalFluxApiV1),&flux_table};
const PopsComponentApiV1 api = {sizeof(PopsComponentApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_ABI_KEY_LITERAL,
  POPS_COMPONENT_CATALOG_SHA256_V1,@COMPONENT@,@SEMANTIC@,@MANIFEST@,1,&entry};
}
extern "C" const PopsComponentApiV1* pops_component_interface_v1() { return &pops_generated_trace_flux::api; }
"""
