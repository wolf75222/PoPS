"""One real Case -> compile -> bind -> Program step through a native shared NumericalFlux."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pops

from pops import interfaces
from pops.external import build_source_package_manifest, compile_component, load
from pops.mesh import CartesianGrid
from pops.mesh.boundaries import (
    BlockInterfaceSide,
    ConservativeInterface,
)
from pops.model import ComponentManifest
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt, StagePoint, TimePoint


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_shared_interface_scalar", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flux_component(tmp_path: Path):
    interface = interfaces.NumericalFlux
    manifest = ComponentManifest(
        uri="pops://external.test/shared-interface/average",
        component_type="numerical_flux", version="1.0.0", facets=interface.facets,
        signature={
            "generic": True,
            "state_components": 1,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    expected_parameters_json = json.dumps(
        manifest.to_data()["parameters"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=True)
    expected_target_json = json.dumps(
        manifest.to_data()["target"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=True)
    source = f'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstddef>
#include <cstring>

namespace {{
int prepare(const PopsComponentPrepareRequestV1* request, void** state,
            PopsComponentStatusV1* status) {{
  if (!request || !state || !status || !request->parameters_json ||
      !request->target_json ||
      std::strcmp(request->parameters_json, {json.dumps(expected_parameters_json)}) != 0 ||
      std::strcmp(request->target_json, {json.dumps(expected_target_json)}) != 0) {{
    if (status)
      *status = {{sizeof(PopsComponentStatusV1), 31,
                  POPS_COMPONENT_ABORT_RUN_V1, "unauthenticated prepare JSON"}};
    return 31;
  }}
  *state = new int(73);
  *status = {{sizeof(PopsComponentStatusV1), 0,
              POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

void destroy(void* state) {{ delete static_cast<int*>(state); }}

int evaluate(void* state, const PopsNumericalFluxRequestV1* request,
             PopsNumericalFluxResultV1* result) {{
  if (!state || *static_cast<int*>(state) != 73 || !request || !result ||
      request->left.component_count != 1 ||
      request->right.component_count != 1 || request->execution.execution_identity == nullptr)
    return 2;
  const auto* left = static_cast<const double*>(request->left.data);
  const auto* right = static_cast<const double*>(request->right.data);
  const auto* normal = static_cast<const double*>(request->normals.data);
  auto* flux = static_cast<double*>(result->normal_flux.data);
  const std::size_t count = request->left.extents[0];
  for (std::size_t point = 0; point < count; ++point) {{
    const std::size_t state_offset = point * request->left.axis_strides[0];
    const std::size_t normal_offset = point * request->normals.axis_strides[0];
    flux[point * result->normal_flux.axis_strides[0]] =
        0.5 * (left[state_offset] + right[state_offset]) * normal[normal_offset];
    result->stability_bounds[point] = 1.0;
    result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
  }}
  result->status = {{sizeof(PopsComponentStatusV1), 0,
                     POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

const PopsNumericalFluxApiV1 flux_table = {{
  {{sizeof(PopsNumericalFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, &prepare, &destroy}},
  &evaluate
}};
const PopsComponentInterfaceEntryV1 entry = {{
  POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
  sizeof(PopsNumericalFluxApiV1), &flux_table
}};
const PopsComponentApiV1 api = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)},
  1, &entry
}};
}}
extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{ return &api; }}
'''.encode()
    source_name = "shared_average.cpp"
    (tmp_path / source_name).write_bytes(source)
    package = build_source_package_manifest(
        components={"average": manifest}, payloads={source_name: ("source", source)})
    package_path = tmp_path / "shared-average.pops.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    component = load(package_path).require(
        "average", interface=interfaces.NumericalFlux)()
    return compile_component(component, include=str(ROOT / "include"))


def _program(left_state, right_state, rate):
    program = pops.Program("shared_interface_forward_euler")
    left = program.state(left_state)
    right = program.state(right_state)
    # State declarations materialize lazily on first value access.  Materialize both endpoints
    # before either RHS so the two default-flux evaluations form the one contiguous atomic group
    # required by the shared NumericalFlux scheduler.
    left_n = left.n
    right_n = right.n
    stage = StagePoint("shared_stage", {"main": TimePoint(program.clock, 0)})
    left_rate = program.value("left_rate", rate(left_n), at=stage)
    right_rate = program.value("right_rate", rate(right_n), at=stage)
    left_next = program.value(
        "left_next", left_n + program.dt * left_rate, at=left.next.point)
    right_next = program.value(
        "right_next", right_n + program.dt * right_rate, at=right.next.point)
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)
    program.step_strategy(FixedDt(1.0e-3))
    return program


def test_runtime_instance_executes_one_two_sided_shared_flux(tmp_path):
    example = _load_example()
    core = example.build_authoring(output_root=tmp_path / "unused")
    right = core.case.block("right", model=core.model)
    right_state = right[core.state]
    finite_volume = FiniteVolume(
        flux=core.flux,
        variables=variables.Conservative(core.state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.ScalarUpwind(velocity=core.velocity),
    )
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow
    boundaries = core.frame.boundaries

    def numerics(state):
        plan = DiscretizationPlan()
        plan.rates.add(core.rate, finite_volume)
        plan.boundaries.add(TransportBoundarySet({
            boundaries.x_min: Inflow(state=state, value=core.inlet_x_value),
            boundaries.x_max: Outflow(state=state),
            boundaries.y_min: Inflow(state=state, value=core.inlet_y_value),
            boundaries.y_max: Outflow(state=state),
        }))
        return plan

    left_numerics = numerics(core.tracer_state)
    right_numerics = numerics(right_state)
    component = _flux_component(tmp_path)
    ConservativeInterface(
        "tracer_to_right",
        left=BlockInterfaceSide(core.tracer_state, boundaries.x_max),
        right=BlockInterfaceSide(right_state, boundaries.x_min),
        numerical_flux=component,
        permutation=(0,),
        right_normal_translation=1.0,
    ).attach(left_numerics, right_numerics)
    core.case.numerics(left_numerics, block=core.tracer)
    core.case.numerics(right_numerics, block=right)
    program = _program(core.tracer_state, right_state, core.rate)
    core.case.program(program)
    validated = pops.validate(core.case)
    from pops.layouts import Uniform
    resolved = pops.resolve(
        validated,
        layout=Uniform(CartesianGrid(frame=core.frame, cells=(8, 8))),
        components=(component,),
        compile_options={"include": str(ROOT / "include")},
    )
    endpoint_interfaces = tuple(
        block.numerics.boundaries[0].interfaces[0] for block in resolved.blocks)
    assert endpoint_interfaces[0].canonical_identity() == \
        endpoint_interfaces[1].canonical_identity()
    interface = endpoint_interfaces[0]
    assert interface.left.boundary.owner_path != interface.right.boundary.owner_path
    artifact = pops.compile(resolved)
    initial = {
        "tracer": np.ones((1, 8, 8), dtype=np.float64),
        "right": np.full((1, 8, 8), 3.0, dtype=np.float64),
    }
    runtime = pops.bind(
        artifact, initial_state=initial, params=example.build_bind_params(core))

    pops.run(runtime, t_end=1.0e-3, max_steps=1)

    left = np.asarray(runtime.get_state("tracer")).reshape(1, 8, 8)
    right_values = np.asarray(runtime.get_state("right")).reshape(1, 8, 8)
    # This native count is an integration-only witness that the installed adapter ran exactly
    # once. Public state and advancement remain on RuntimeInstance/pops.run; ``_executor`` is
    # consulted only as an internal integration witness here.
    assert runtime._executor._s._interface_evaluation_count(
        interface.qualified_id, 0) == 1
    # On interior rows, zeroing each former physical face and scattering the
    # unique average flux gives the exact first-order update below.  The paired
    # +/- shared contribution itself is covered independently by the native
    # scheduler conservation test; this assertion proves that the real Program
    # executes the installed adapter instead of a Python callback.
    np.testing.assert_allclose(
        left[0, 1:-1, -1], 0.992,
        rtol=0.0, atol=1.0e-14,
    )
    np.testing.assert_allclose(
        right_values[0, 1:-1, 0], 2.992,
        rtol=0.0, atol=1.0e-14,
    )
