"""One real Case -> compile -> bind -> Program step through a native shared NumericalFlux."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

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
from pops.output import Checkpoint, ConsumerGraph, RegridOnRestart
from pops.time import FixedDt, StagePoint, TimePoint, every


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
  POPS_ABI_KEY_LITERAL,
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
    # State declarations materialize lazily on first value access. Materialize both endpoints before
    # either RHS; the pure reduction deliberately separates them and exercises the shared resolve +
    # compile coherence planner instead of relying on adjacency in the authored Program.
    left_n = left.n
    right_n = right.n
    stage = StagePoint("shared_stage", {"main": TimePoint(program.clock, 0)})
    left_rate = program.value("left_rate", rate(left_n), at=stage)
    program.norm2(left_n)
    right_rate = program.value("right_rate", rate(right_n), at=stage)
    left_next = program.value(
        "left_next", left_n + program.dt * left_rate, at=left.next.point)
    right_next = program.value(
        "right_next", right_n + program.dt * right_rate, at=right.next.point)
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)
    program.step_strategy(FixedDt(1.0e-3))
    return program


def _ssprk2_program(left_state, right_state, rate):
    program = pops.Program("shared_interface_ssprk2")
    left = program.state(left_state)
    right = program.state(right_state)
    stage_0 = StagePoint("shared_stage_0", {"main": TimePoint(program.clock, 0)})
    left_k0 = program.value("left_k0", rate(left.n), at=stage_0)
    right_k0 = program.value("right_k0", rate(right.n), at=stage_0)
    stage_1 = StagePoint("shared_stage_1", {"main": TimePoint(program.clock, 1)})
    left_stage = program.value(
        "left_stage", left.n + program.dt * left_k0, at=stage_1)
    right_stage = program.value(
        "right_stage", right.n + program.dt * right_k0, at=stage_1)
    left_k1 = program.value("left_k1", rate(left_stage), at=stage_1)
    right_k1 = program.value("right_k1", rate(right_stage), at=stage_1)
    left_next = program.value(
        "left_next",
        left.n + 0.5 * program.dt * left_k0 + 0.5 * program.dt * left_k1,
        at=left.next.point,
    )
    right_next = program.value(
        "right_next",
        right.n + 0.5 * program.dt * right_k0 + 0.5 * program.dt * right_k1,
        at=right.next.point,
    )
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)
    program.step_strategy(FixedDt(1.0e-3))
    return program


def _shared_interface_accepted_image(runtime):
    native = runtime._executor._s
    levels = int(runtime.n_levels())
    return {
        "time": float(runtime.time()),
        "step": int(runtime.macro_step()),
        "boxes": tuple(tuple(int(value) for value in row) for row in runtime.patch_boxes()),
        "regrid_count": int(native.checkpoint_regrid_count()),
        "topology_epoch": int(native.checkpoint_topology_epoch()),
        "program_state": bytes(native.program_accepted_state()),
        "states": tuple(
            np.asarray(runtime.block_level_state_global(block, level), dtype=np.float64).copy()
            for block in ("tracer", "right")
            for level in range(levels)
        ),
    }


def _assert_same_shared_interface_image(runtime, expected):
    actual = _shared_interface_accepted_image(runtime)
    assert {key: value for key, value in actual.items() if key != "states"} == {
        key: value for key, value in expected.items() if key != "states"
    }
    assert len(actual["states"]) == len(expected["states"])
    for current, recorded in zip(actual["states"], expected["states"], strict=True):
        np.testing.assert_array_equal(current, recorded)


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
    assert interface.left.trace_provider == "limiter.none"
    assert interface.right.trace_provider == "limiter.none"
    assert interface.left.trace_operation.value == "cell_average"
    assert interface.right.trace_operation.value == "cell_average"
    assert interface.left.required_depth == interface.right.required_depth == 1
    for resolved_block, authored_block in zip(
            resolved.blocks, (core.tracer, right), strict=True):
        expected = core.case.resolve(core.inlet_x_param, block=authored_block)
        x_min = resolved_block.numerics.boundaries[0].compile_boundary_data()["faces"][0]
        assert x_min["values"] == [["handle_value", expected.qualified_id]]
    artifact = pops.compile(resolved)
    initial = {
        "tracer": np.ones((1, 8, 8), dtype=np.float64),
        "right": np.full((1, 8, 8), 3.0, dtype=np.float64),
    }
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
    params.update({
        core.case.resolve(core.refine_threshold): 0.10,
        core.case.resolve(core.coarsen_threshold): 0.04,
    })
    compiled_endpoint_owners = {
        block.name: block.boundaries[0].runtime_boundary_data(params)[
            "interface_endpoints"
        ][0]["owned_sides"]
        for block in artifact.plan.blocks
    }
    assert compiled_endpoint_owners == {"tracer": ["left"], "right": ["right"]}
    runtime = example._bind_artifact(
        artifact, initial_state=initial, params=params)

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


def test_runtime_instance_executes_dynamic_three_level_shared_flux(tmp_path, monkeypatch):
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        Tag,
    )
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow
    from pops.initial import InitialCondition
    from pops.layouts import AMR
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import BindArray
    from pops.math import ValueExpr
    from pops.projection import ConservativeCellAverage

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
    core.case.initials.add(InitialCondition(
        state=core.tracer_state,
        value=BindArray(),
        projection=ConservativeCellAverage(),
    ))
    core.case.initials.add(InitialCondition(
        state=right_state,
        value=BindArray(),
        projection=ConservativeCellAverage(),
    ))
    program = _ssprk2_program(core.tracer_state, right_state, core.rate)
    core.case.program(program)
    core.case.consumers(
        ConsumerGraph.from_consumers(
            (
                Checkpoint(
                    schedule=every(10_000, clock=program.clock),
                    target="unused/shared-interface-restart",
                    hierarchy=RegridOnRestart(),
                ),
            )
        )
    )

    transfer = AMRTransfer()
    transfer.state(core.tracer_state, StateTransfer())
    transfer.state(right_state, StateTransfer())
    tagging = AMRTagging(
        rules=(
            Tag(ValueExpr(core.tracer_state) > core.case.value(core.refine_threshold)),
            Tag(ValueExpr(right_state) > core.case.value(core.refine_threshold)),
            Buffer(cells=1),
        ),
        hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )
    resolved = pops.resolve(
        pops.validate(core.case),
        layout=AMR(
            grid=CartesianGrid(frame=core.frame, cells=(8, 8)),
            hierarchy=AMRHierarchy(max_levels=3, ratios=(2, 2)),
            tagging=tagging,
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.subcycled((
                AMRClockRelation(0, 1, 2),
                AMRClockRelation(1, 2, 2),
            )),
        ),
        components=(component,),
        compile_options={"include": str(ROOT / "include")},
    )
    artifact = pops.compile(resolved)
    left_initial = np.zeros((1, 8, 8), dtype=np.float64)
    right_initial = np.zeros((1, 8, 8), dtype=np.float64)
    # The first public refined route requires an already matched fine interface: refine one
    # full-height coarse-cell band on both mapped faces while keeping the domain interior coarse.
    # One-sided tag propagation across a BlockInterface is a separate capability and must not be
    # implied by this proof.
    left_initial[0, :, -1:] = 1.0
    # Keep the two traces distinct: the shared component must publish its average flux to both
    # consumers.  Equal traces would let a one-sided publication pass by coincidence.
    right_initial[0, :, :1] = 3.0
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
    params.update({
        core.case.resolve(core.refine_threshold): 0.10,
        core.case.resolve(core.coarsen_threshold): 0.04,
    })
    interface = resolved.blocks[0].numerics.boundaries[0].interfaces[0]
    # Dynamic shared interfaces cannot create a missing route after bind: the complete configured
    # prefix must already be materialized by the authenticated bootstrap transaction.
    with pytest.raises(
        NotImplementedError, match="complete configured prefix materialized at bind"
    ):
        example._bind_artifact(
            artifact,
            initial_values={
                core.tracer_state: np.zeros_like(left_initial),
                right_state: np.zeros_like(right_initial),
            },
            params=params,
        )

    # A shared hierarchy does not imply that one endpoint's boundary tags are mirrored to its peer.
    # With only the left x-high band tagged, the materialized L1 layout cannot tile the right x-low
    # face.  The incremental finalizer must reject that incomplete pair before bind freezes.
    with pytest.raises(ValueError, match="does not tile its declared physical face"):
        example._bind_artifact(
            artifact,
            initial_values={
                core.tracer_state: left_initial,
                right_state: np.zeros_like(right_initial),
            },
            params=params,
        )

    runtime = example._bind_artifact(
        artifact,
        initial_values={
            core.tracer_state: left_initial,
            right_state: right_initial,
        },
        params=params,
    )

    assert runtime.n_levels() == 3
    fine_boxes = tuple(row for row in runtime.patch_boxes() if int(row[0]) == 1)
    assert fine_boxes
    assert any(
        int(row[1]) == 0 and int(row[2]) == 0 and int(row[4]) == 15
        for row in fine_boxes
    )
    assert any(
        int(row[3]) == 15 and int(row[2]) == 0 and int(row[4]) == 15
        for row in fine_boxes
    )
    assert not any(int(row[1]) <= 7 <= int(row[3]) for row in fine_boxes)
    initial_left = runtime.integral("tracer")
    initial_right = runtime.integral("right")
    initial_integral = initial_left + initial_right

    pops.run(runtime, t_end=1.0e-3, max_steps=1)

    refined_authority = runtime._executor._interface_authorities[interface.qualified_id]
    assert refined_authority["levels"] == (0, 1, 2)
    assert len(refined_authority["declaration_identity"]) == 64
    assert runtime._executor._s._interface_evaluation_count(
        interface.qualified_id, 0) == 2
    assert runtime._executor._s._interface_evaluation_count(
        interface.qualified_id, 1) == 4
    assert runtime._executor._s._interface_evaluation_count(
        interface.qualified_id, 2) == 8
    final_left = runtime.integral("tracer")
    final_right = runtime.integral("right")
    lost_by_left = initial_left - final_left
    gained_by_right = final_right - initial_right
    assert lost_by_left > 0.0
    assert gained_by_right > 0.0
    np.testing.assert_allclose(gained_by_right, lost_by_left, rtol=0.0, atol=2.0e-13)
    final_integral = final_left + final_right
    np.testing.assert_allclose(final_integral, initial_integral, rtol=0.0, atol=2.0e-13)

    # The three-level route above proves arbitrary-depth execution. Use the independently compiled
    # two-level route for the restart transaction: replacing its only fine transition is the exact
    # dynamic topology capability currently authenticated by the interface scheduler.
    restart_resolved = pops.resolve(
        pops.validate(core.case),
        layout=AMR(
            grid=CartesianGrid(frame=core.frame, cells=(8, 8)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=tagging,
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
        ),
        components=(component,),
        compile_options={"include": str(ROOT / "include")},
    )
    restart_artifact = pops.compile(restart_resolved)
    restart_interface = restart_resolved.blocks[0].numerics.boundaries[0].interfaces[0]
    restart_source = example._bind_artifact(
        restart_artifact,
        initial_values={
            core.tracer_state: left_initial,
            right_state: right_initial,
        },
        params=params,
    )
    assert restart_source.n_levels() == 2
    restart_initial_integral = restart_source.integral("tracer") + restart_source.integral("right")
    source_report = pops.run(
        restart_source,
        t_end=1.0e-3,
        max_steps=1,
        console=False,
        output_dir=tmp_path / "restart-source-output",
    )
    assert source_report.accepted_steps == 1
    assert restart_source._executor._s._interface_evaluation_count(
        restart_interface.qualified_id, 0) == 2
    assert restart_source._executor._s._interface_evaluation_count(
        restart_interface.qualified_id, 1) == 4
    checkpoint_time = float(restart_source.time())
    checkpoint_step = int(restart_source.macro_step())
    checkpoint_integral = (
        restart_source.integral("tracer") + restart_source.integral("right")
    )
    np.testing.assert_allclose(
        checkpoint_integral,
        restart_initial_integral,
        rtol=0.0,
        atol=2.0e-13,
    )
    checkpoint = restart_source.checkpoint(tmp_path / "accepted-shared-interface")

    # RegridOnRestart must enter the serial native tag/cluster/regrid boundary; a deliberately
    # rejected post-transform validation must restore the fresh runtime exactly before the same
    # restart is retried and committed.
    restarted = example._bind_artifact(
        restart_artifact,
        initial_values={
            core.tracer_state: left_initial,
            right_state: right_initial,
        },
        params=params,
    )
    priming_report = pops.run(
        restarted,
        t_end=1.0e-3,
        max_steps=1,
        console=False,
        output_dir=tmp_path / "restart-candidate-output",
    )
    assert priming_report.accepted_steps == 1
    rollback_image = _shared_interface_accepted_image(restarted)
    from pops.runtime import _amr_checkpoint_v3 as checkpoint_codec

    original_conservation_check = checkpoint_codec._require_restart_conservation
    transformed_images = []

    def fail_after_native_regrid(before, after):
        del before, after
        transformed_images.append(_shared_interface_accepted_image(restarted))
        raise RuntimeError("injected shared-interface restart validation failure")

    monkeypatch.setattr(
        checkpoint_codec,
        "_require_restart_conservation",
        fail_after_native_regrid,
    )
    with pytest.raises(RuntimeError, match="injected shared-interface restart validation failure"):
        restarted.restart(checkpoint)
    assert transformed_images
    assert transformed_images[0]["boxes"] != rollback_image["boxes"]
    _assert_same_shared_interface_image(restarted, rollback_image)

    monkeypatch.setattr(
        checkpoint_codec,
        "_require_restart_conservation",
        original_conservation_check,
    )
    restarted.restart(checkpoint)
    receipt = restarted._executor.last_restart_regrid_receipt()
    assert receipt is not None
    assert receipt["changed"] is True
    assert float(restarted.time()) == checkpoint_time
    assert int(restarted.macro_step()) == checkpoint_step
    assert tuple(restarted.patch_boxes()) == tuple(transformed_images[0]["boxes"])
    np.testing.assert_allclose(
        [row["value"] for row in receipt["composite_integrals_after"]],
        [row["value"] for row in receipt["composite_integrals_before"]],
        rtol=2.0e-12,
        atol=2.0e-13,
    )
    restarted_integral = restarted.integral("tracer") + restarted.integral("right")
    np.testing.assert_allclose(
        restarted_integral,
        checkpoint_integral,
        rtol=2.0e-12,
        atol=2.0e-13,
    )

    counts_before_continuation = tuple(
        restarted._executor._s._interface_evaluation_count(restart_interface.qualified_id, level)
        for level in range(2)
    )
    pops.run(restarted, t_end=2.0e-3, max_steps=1, console=False)
    counts_after_continuation = tuple(
        restarted._executor._s._interface_evaluation_count(restart_interface.qualified_id, level)
        for level in range(2)
    )
    assert tuple(
        after - before
        for before, after in zip(counts_before_continuation, counts_after_continuation, strict=True)
    ) == (2, 4)
    np.testing.assert_allclose(
        restarted.integral("tracer") + restarted.integral("right"),
        checkpoint_integral,
        rtol=0.0,
        atol=2.0e-13,
    )
