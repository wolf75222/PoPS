"""Authenticated BoundaryFlux admission beside a real conservative shared interface.

The native assembly harness varies only call order. Both orders retain the resolved
state, face and layout identities and use the loaded BoundaryFlux ABI table.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pops
import pytest

from pops import interfaces
from pops.external import build_source_package_manifest, compile_component, load
from pops.layouts import Uniform
from pops.mesh import CartesianGrid
from pops.mesh.boundaries import BlockInterfaceSide, ConservativeInterface
from pops.model import ComponentManifest, Handle
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from tests.python.integration.runtime.test_shared_interface_runtime import (
    INCLUDE,
    _ExternalGhostFaceExecutionAuthority,
    _flux_component,
    _load_example,
    _program,
)


class _ExternalFaceAuthoring:
    """Explicit external x-min authority used by the native BoundaryFlux fixture."""

    def __init__(self, base):
        self.base = base

    def inspect(self):
        return {
            "schema_version": 1,
            "authority_type": "boundary_flux_test_external_face",
            "base": self.base.inspect(),
        }

    def resolve_for_numerics(self, context):
        return _ResolvedExternalFace(self.base.resolve_for_numerics(context))


class _ResolvedExternalFace:
    def __init__(self, base):
        self.base = base

    def canonical_identity(self):
        return {
            "schema_version": 1,
            "authority_type": "boundary_flux_test_external_face",
            "base": self.base.canonical_identity(),
        }

    def ghost_plan_composer_capability(self):
        return {"schema_version": 1, "scope": "self"}

    def compose_ghost_plan(self, context):
        from pops.mesh.boundaries.composition import compose_transport_boundary

        plan = compose_transport_boundary(self.base, context=context)
        production = next(
            row
            for row in plan.productions
            if row.region.boundary is not None
            and row.region.boundary.orientation.axis == 0
            and row.region.boundary.orientation.outward_sign == -1
        )
        target = production.producer.boundary_providers[0].handle
        return replace(
            plan,
            execution_authority=_ExternalGhostFaceExecutionAuthority(
                plan.execution_authority, target.qualified_id
            ),
        )


def _boundary_flux_component(directory):
    directory.mkdir(parents=True, exist_ok=True)
    interface = interfaces.resolve("boundary_flux")
    manifest = ComponentManifest(
        uri="pops://external.test/shared-interface/boundary-flux-ownership",
        component_type="boundary_flux",
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "state_components": 1,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={
            "variants": [{"dimension": 2, "scalar": "float64", "device": "cpu", "features": []}]
        },
        entry_points={"interface_table": "pops_component_interface_v1"},
    )

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    parameters_json = canonical(manifest.to_data()["parameters"])
    target_json = canonical(manifest.to_data()["target"])
    events_path = directory / "callback-events.txt"
    source = f"""#include <pops/runtime/config/generated_component_abi.hpp>
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
namespace {{
int prepare(const PopsComponentPrepareRequestV1* request, void** state, PopsComponentStatusV1* status) {{
  if (!request || !state || !status || !request->parameters_json || !request->target_json ||
      std::strcmp(request->parameters_json, {json.dumps(parameters_json)}) != 0 ||
      std::strcmp(request->target_json, {json.dumps(target_json)}) != 0) return 71;
  const auto* events_path = std::getenv("POPS_TEST_BOUNDARY_FLUX_EVENTS");
  if (!events_path) return 75;
  std::ofstream events(events_path, std::ios::app);
  if (!events) return 73;
  events << "prepare\\n";
  *state = new int(91);
  *status = {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}
void destroy(void* state) {{ delete static_cast<int*>(state); }}
int transform(void* state, const PopsBoundaryFluxRequestV1* request, PopsBoundaryFluxResultV1* result) {{
  if (!state || *static_cast<int*>(state) != 91 || !request || !result ||
      !request->provider_identity || !request->state_identity || !request->execution.execution_identity ||
      request->region.kind != POPS_BOUNDARY_FACE_V1 || request->region.dimension != 2 ||
      request->region.codimension != 1 || request->region.axis_count != 1 ||
      !request->region.axes || !request->region.sides || !request->region.region_identity ||
      request->region.axes[0] != 0 || request->dependency_count != 0 || request->parameter_count != 0 ||
      request->base_outward_normal_flux.component_count != 1 || !request->face_measures ||
      !result->outward_normal_flux.data || !result->actions) return 72;
  const auto* events_path = std::getenv("POPS_TEST_BOUNDARY_FLUX_EVENTS");
  if (!events_path) return 75;
  std::ofstream events(events_path, std::ios::app);
  if (!events) return 74;
  events << "transform\\t" << request->region.sides[0] << "\\t"
         << request->region.region_identity << "\\t" << request->state_identity << "\\n";
  auto* output = static_cast<double*>(result->outward_normal_flux.data);
  std::size_t points = 1;
  for (unsigned axis = 0; axis < result->outward_normal_flux.dimension; ++axis)
    points *= result->outward_normal_flux.extents[axis];
  for (std::size_t point = 0; point < points; ++point) {{
    std::size_t remainder = point;
    std::ptrdiff_t offset = 0;
    for (unsigned axis = 0; axis < result->outward_normal_flux.dimension; ++axis) {{
      offset += (remainder % result->outward_normal_flux.extents[axis]) * result->outward_normal_flux.axis_strides[axis];
      remainder /= result->outward_normal_flux.extents[axis];
    }}
    output[offset] = -4.0 * request->face_measures[point];
    result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
  }}
  result->status = {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}
const PopsBoundaryFluxApiV1 table = {{
  {{sizeof(PopsBoundaryFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1, 1, &prepare, &destroy}}, &transform}};
const PopsComponentInterfaceEntryV1 entry = {{POPS_NATIVE_INTERFACE_BOUNDARY_FLUX_V1, 1,
                                            sizeof(PopsBoundaryFluxApiV1), &table}};
const PopsComponentApiV1 api = {{sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL, POPS_COMPONENT_CATALOG_SHA256_V1, {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)}, {json.dumps(manifest.manifest_digest.token)}, 1, &entry}};
}}
extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{ return &api; }}
""".encode()
    name = "boundary_flux_ownership.cpp"
    (directory / name).write_bytes(source)
    package = build_source_package_manifest(
        components={"flux": manifest}, payloads={name: ("source", source)}
    )
    package_path = directory / "boundary-flux.pops.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    artifact = compile_component(
        load(package_path).require("flux", interface=interface)(), include=INCLUDE
    )
    # Each rank authenticates one identical executable image. Linkers can embed
    # their temporary output path, so equal source packages alone do not prove
    # equal binary identities for this native prepared-component contract.
    from pops._native_selector import selected_native_module
    from pops.identity import Identity

    world = selected_native_module(required=True).mpi_world()
    artifact = replace(
        artifact,
        binary=world.broadcast_bytes(artifact.binary if world.rank == 0 else b"", root=0),
        binary_identity=Identity.from_token(
            world.broadcast_bytes(
                artifact.binary_identity.token.encode() if world.rank == 0 else b"", root=0
            ).decode()
        ),
    )
    installed = artifact.install(directory / "installed").load()
    return SimpleNamespace(
        installed=installed,
        events_path=events_path,
        parameters_json=parameters_json,
        target_json=target_json,
    )


@pytest.fixture(scope="module")
def boundary_flux_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("boundary_flux_ownership")
    example = _load_example()
    core = example.build_authoring(output_root=root / "unused")
    right = core.case.block("right", model=core.model)
    right_state = right[core.state]
    scheme = FiniteVolume(
        flux=core.flux,
        variables=variables.Conservative(core.state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.ScalarUpwind(velocity=core.velocity),
    )
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow

    boundaries = core.frame.boundaries

    def numerics(state, *, external=False):
        plan = DiscretizationPlan()
        plan.rates.add(core.rate, scheme)
        boundary = TransportBoundarySet(
            {
                boundaries.x_min: Inflow(state=state, value=core.inlet_x_value),
                boundaries.x_max: Outflow(state=state),
                boundaries.y_min: Inflow(state=state, value=core.inlet_y_value),
                boundaries.y_max: Outflow(state=state),
            }
        )
        plan.boundaries.add(_ExternalFaceAuthoring(boundary) if external else boundary)
        return plan

    left_plan = numerics(core.tracer_state, external=True)
    right_plan = numerics(right_state)
    shared = _flux_component(root / "shared")
    ConservativeInterface(
        "tracer_to_right",
        left=BlockInterfaceSide(core.tracer_state, boundaries.x_max),
        right=BlockInterfaceSide(right_state, boundaries.x_min),
        numerical_flux=shared,
        permutation=(0,),
        right_normal_translation=1.0,
    ).attach(left_plan, right_plan)
    core.case.numerics(left_plan, block=core.tracer)
    core.case.numerics(right_plan, block=right)
    core.case.program(_program(core.tracer_state, right_state, core.rate))
    resolved = pops.resolve(
        pops.validate(core.case),
        layout=Uniform(CartesianGrid(frame=core.frame, cells=(8, 8))),
        components=(shared,),
        compile_options={"include": INCLUDE},
    )
    artifact = pops.compile(resolved)
    params = {
        core.case.resolve(handle, block=block): value
        for block in (core.tracer, right)
        for handle, value in (
            (core.velocity_x_param, 1.0),
            (core.velocity_y_param, 1.0e-12),
            (core.inlet_x_param, 0.0),
            (core.inlet_y_param, 0.0),
        )
    }
    params.update(
        {
            core.case.resolve(core.refine_threshold): 0.10,
            core.case.resolve(core.coarsen_threshold): 0.04,
        }
    )
    plan = next(block for block in artifact.plan.blocks if block.name == "tracer").boundaries[0]
    payload = plan.runtime_boundary_data(params)
    assert payload["omitted_interface_faces"] == [1]
    assert payload["faces"][0]["type"] == payload["faces"][1]["type"] == "external"
    initial = {"tracer": np.ones((1, 8, 8)), "right": np.full((1, 8, 8), 3.0)}
    return SimpleNamespace(
        example=example,
        artifact=artifact,
        params=params,
        plan=plan,
        resolved_boundary=next(
            block for block in resolved.blocks if block.name == "tracer"
        ).numerics.boundaries[0],
        initial=initial,
        component=_boundary_flux_component(root / "boundary"),
    )


def _component_row(case, ordinal):
    production = next(
        row
        for row in case.resolved_boundary.productions
        if row.region.boundary is not None
        and 2 * row.region.boundary.orientation.axis
        + (row.region.boundary.orientation.outward_sign > 0)
        == ordinal
    )
    region = production.region
    target = Handle(
        "native_boundary_flux_ownership",
        kind="boundary_flux_provider",
        owner=region.selector.owner_path,
    )
    return {
        "target": target.canonical_identity(),
        "component_id": case.component.installed.component_id,
        "component_manifest_identity": case.component.installed.component_manifest.token,
        "interface_version": 1,
        "producer_identity": production.producer.qualified_id,
        "state_identity": region.subject.qualified_id,
        "ghost_identity": region.selector.qualified_id,
        "region": {
            "kind": "face",
            "dimension": 2,
            "codimension": 1,
            "axes": [0],
            "sides": [-1 if ordinal == 0 else 1],
            "region_identity": region.selector.qualified_id,
            "layout_identity": region.layout.qualified_id,
        },
        "states": [],
        "directions": [],
        "fields": [],
        "parameters": [],
        "outputs": [region.subject.qualified_id],
        "rate": None,
        "nonlinear_iterate": None,
    }


@pytest.mark.parametrize(
    "component_first", [False, True], ids=["boundary-first", "component-first"]
)
@pytest.mark.parametrize(
    "ordinal", [1, 0], ids=["reserved-face-refused", "other-external-face-executes"]
)
def test_authenticated_boundary_flux_respects_exact_interface_ownership(
    boundary_flux_case, monkeypatch, component_first, ordinal
):
    case = boundary_flux_case
    from pops.runtime import _runtime_authorities
    from pops.runtime._component_execution_context import component_execution_data
    from pops._native_selector import selected_native_module

    events_path = case.component.events_path
    events_path.unlink(missing_ok=True)
    monkeypatch.setenv("POPS_TEST_BOUNDARY_FLUX_EVENTS", str(events_path))

    def callback_events():
        return events_path.read_text().splitlines() if events_path.exists() else []

    row = _component_row(case, ordinal)
    original = _runtime_authorities._install_boundary_authorities
    observed = {}

    def install(engine, install_plan):
        native = engine._s
        deferred = []

        class NativeAssemblyOrder:
            def __getattr__(self, name):
                return getattr(native, name)

            def _install_boundary_plan(self, *arguments):
                if arguments[0] == "tracer" and component_first:
                    deferred.append(arguments)
                else:
                    native._install_boundary_plan(*arguments)

        engine._s = NativeAssemblyOrder()
        try:
            original(engine, install_plan)
        finally:
            engine._s = native
        observed["native"] = native
        native._install_boundary_flux_component(
            "tracer",
            case.component.installed.native_handle,
            row,
            case.component.parameters_json,
            case.component.target_json,
            component_execution_data(install_plan.execution_context),
        )
        for arguments in deferred:
            native._install_boundary_plan(*arguments)

    monkeypatch.setattr(_runtime_authorities, "_install_boundary_authorities", install)
    if ordinal == 1:
        with pytest.raises((ValueError, RuntimeError), match="reserved|rolled back|preflight"):
            case.example._bind_artifact(
                case.artifact, initial_state=case.initial, params=case.params
            )
        assert callback_events() == []
        assert observed["native"].block_names() == []
        return
    runtime = case.example._bind_artifact(
        case.artifact, initial_state=case.initial, params=case.params
    )
    assert callback_events() == ["prepare"]
    pops.run(runtime, t_end=1.0e-3, max_steps=1)
    transforms = [
        event.split("\t") for event in callback_events() if event.startswith("transform\t")
    ]
    assert all(event[1] == "-1" for event in transforms)
    if selected_native_module(required=True).my_rank() == 0:
        # This one-box Uniform layout is owned by rank zero; get_state reads
        # its local compact image and performs no MPI gather.
        state = np.asarray(runtime.get_state("tracer"), dtype=np.float64)
        assert transforms
        assert all(event[2] == row["region"]["region_identity"] for event in transforms)
        assert all(event[3] == row["state_identity"] for event in transforms)
        state = state.reshape(1, 8, 8)
        np.testing.assert_allclose(state[0, 1:-1, 0], 1.024, rtol=0.0, atol=1.0e-14)
        # The opposite face receives exactly one shared average flux (2), so the
        # value is 1 - dt/h*(2-1), not a restored local flux plus shared flux.
        np.testing.assert_allclose(state[0, 1:-1, -1], 0.992, rtol=0.0, atol=1.0e-14)
    assert runtime.time() == pytest.approx(1.0e-3, rel=0.0, abs=1.0e-15)
