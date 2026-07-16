"""Real two-layout RuntimeInstance: sliced Programs, native transfer and atomic restart."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

import pops
from pops import interfaces
from pops.external import build_source_package_manifest, load
from pops.layouts import Uniform
from pops.mesh import (
    LayoutMappingOperation,
    LayoutPlanBuilder,
    LayoutRepresentation,
    LayoutSynchronization,
    NativeLayoutMapping,
)
from pops.model import ComponentManifest
from pops.time import FixedDt, StagePoint, TimePoint
from tests.python.support.layout_plan import cartesian_grid


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"
DT = 1.0e-3


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_multi_layout_scalar", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _transfer_manifest() -> ComponentManifest:
    interface = interfaces.Transfer
    return ComponentManifest(
        uri="pops://runtime.test/transfers/conservative-cell-average",
        component_type="transfer",
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        target={
            "variants": [
                {
                    "dimension": 2,
                    "scalar": "float64",
                    "device": "cpu",
                    "features": [],
                }
            ]
        },
        entry_points={"interface_table": "pops_component_interface_v1"},
    )


def _transfer_source(manifest: ComponentManifest) -> bytes:
    source = r"""#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstddef>
#include <cstdint>

namespace {
int apply(void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* status) {
  if (!request || !status || request->struct_size < sizeof(PopsTransferRequestV1) ||
      request->dimension != 2 || request->source.dimension != 2 ||
      request->destination.dimension != 2 || !request->source.data ||
      !request->destination.data || !request->refinement_ratio ||
      request->source.component_count != request->destination.component_count ||
      request->operation != POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1) return 2;
  const auto ratio_y = request->refinement_ratio[0];
  const auto ratio_x = request->refinement_ratio[1];
  if (ratio_y <= 0 || ratio_x <= 0 ||
      request->source.extents[0] != request->destination.extents[0] * ratio_y ||
      request->source.extents[1] != request->destination.extents[1] * ratio_x) return 3;
  const auto* source = static_cast<const double*>(request->source.data);
  auto* destination = static_cast<double*>(request->destination.data);
  const double scale = 1.0 / static_cast<double>(ratio_y * ratio_x);
  for (std::size_t component = 0; component < request->source.component_count; ++component)
    for (std::int64_t y = 0; y < request->destination.extents[0]; ++y)
      for (std::int64_t x = 0; x < request->destination.extents[1]; ++x) {
        double total = 0.0;
        for (std::int32_t dy = 0; dy < ratio_y; ++dy)
          for (std::int32_t dx = 0; dx < ratio_x; ++dx)
            total += source[component * request->source.component_stride +
                            (y * ratio_y + dy) * request->source.axis_strides[0] +
                            (x * ratio_x + dx) * request->source.axis_strides[1]];
        destination[component * request->destination.component_stride +
                    y * request->destination.axis_strides[0] +
                    x * request->destination.axis_strides[1]] = total * scale;
      }
  *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}

const PopsTransferApiV1 transfer_table = {
  {sizeof(PopsTransferApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
   POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, nullptr, nullptr},
  &apply
};
const PopsComponentInterfaceEntryV1 interface_entry = {
  POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1), &transfer_table
};
const PopsComponentApiV1 component_api = {
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_COMPONENT_CATALOG_SHA256_V1,
  __COMPONENT_ID__, __SEMANTIC_ID__, __MANIFEST_ID__, 1, &interface_entry
};
}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {
  return &component_api;
}
"""
    return (
        source.replace("__COMPONENT_ID__", json.dumps(manifest.component_id))
        .replace("__SEMANTIC_ID__", json.dumps(manifest.semantic_digest.token))
        .replace("__MANIFEST_ID__", json.dumps(manifest.manifest_digest.token))
        .encode("utf-8")
    )


def _transfer_component(tmp_path: Path):
    manifest = _transfer_manifest()
    source = _transfer_source(manifest)
    source_name = "conservative_cell_average.cpp"
    (tmp_path / source_name).write_bytes(source)
    package = build_source_package_manifest(
        components={"cell_average": manifest},
        payloads={source_name: ("source", source)},
    )
    package_path = tmp_path / "cell_average.pops.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    return load(package_path).require("cell_average", interface=interfaces.Transfer)()


def _two_layout_program(states, rate):
    program = pops.Program("multi_layout_ssprk2")
    for block_name, state in states:
        temporal = program.state(state)
        stage_0 = StagePoint(block_name + "_stage_0", {"main": TimePoint(program.clock, 0)})
        k0 = program.value(block_name + "_k0", rate(temporal.n), at=stage_0)
        stage_1 = StagePoint(block_name + "_stage_1", {"main": TimePoint(program.clock, 1)})
        staged = program.value(block_name + "_stage", temporal.n + program.dt * k0, at=stage_1)
        k1 = program.value(block_name + "_k1", rate(staged), at=stage_1)
        half = Fraction(1, 2)
        advanced = program.value(
            block_name + "_next",
            temporal.n + program.dt * half * k0 + program.dt * half * k1,
            at=temporal.next.point,
        )
        program.commit(temporal.next, advanced)
    program.step_strategy(FixedDt(DT))
    return program


def _initial(n: int, phase: float) -> np.ndarray:
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    y = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return (1.0 + 0.25 * np.sin(2.0 * np.pi * (xx + phase)) * np.cos(2.0 * np.pi * yy))[None, :, :]


@pytest.fixture(scope="module")
def compiled_multi_layout(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi-layout")
    example = _load_example()
    core = example.build_authoring(output_root=root / "unused")
    coarse_block = core.case.block("coarse", model=core.model)
    coarse_state = coarse_block[core.state]
    core.case.numerics(core.numerics, block=core.tracer)
    core.case.numerics(core.numerics, block=coarse_block)
    core.case.program(
        _two_layout_program((("fine", core.tracer_state), ("coarse", coarse_state)), core.rate)
    )
    validated = pops.validate(core.case)

    subjects = validated.layout_subjects()
    blocks = {row.local_id: row for row in subjects.blocks}
    states = {row.block_ref.local_id: row for row in subjects.states}
    builder = LayoutPlanBuilder(validated.owner_path.canonical())
    coarse_descriptor = Uniform(cartesian_grid(n=8, periodic=True, name="coarse-grid"))
    fine_descriptor = Uniform(cartesian_grid(n=16, periodic=True, name="fine-grid"))
    coarse = builder.layout("coarse", coarse_descriptor)
    fine = builder.layout("fine", fine_descriptor)
    builder.assign_block(blocks["coarse"], coarse)
    builder.assign_state(states["coarse"], coarse)
    builder.assign_block(blocks["tracer"], fine)
    builder.assign_state(states["tracer"], fine)
    (requirement,) = builder.require_mapping(
        fine,
        coarse,
        source=states["tracer"],
        target=states["coarse"],
        operation=LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1,
        synchronization=LayoutSynchronization.BEFORE_STEP_V1,
        source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
        target_representation=LayoutRepresentation.CELL_AVERAGE_V1,
    )
    component = _transfer_component(root)
    provider = NativeLayoutMapping(component, (requirement,))
    plan = builder.resolve(**subjects.to_dict(), providers=(provider,))
    resolved = pops.resolve(
        validated,
        layout=plan,
        layout_providers={coarse: coarse_descriptor, fine: fine_descriptor},
        components=(component,),
        compile_options={"include": str(ROOT / "include")},
    )
    artifact = pops.compile(resolved)
    return example, core, artifact, coarse.qualified_id, fine.qualified_id, requirement.qualified_id


def _bind(compiled_multi_layout):
    _example, core, artifact, coarse_id, fine_id, mapping_id = compiled_multi_layout
    fine_initial = _initial(16, 0.13)
    coarse_initial = np.full((1, 8, 8), -4.0, dtype=np.float64)
    blocks = core.case.blocks()
    params = {
        core.case.resolve(handle, block=blocks[block_name]): value
        for block_name in ("tracer", "coarse")
        for handle, value in (
            (core.velocity_x_param, 1.0),
            (core.velocity_y_param, 0.25),
            (core.inlet_x_param, 0.0),
            (core.inlet_y_param, 0.0),
        )
    }
    params.update({
        core.case.resolve(core.refine_threshold): 0.10,
        core.case.resolve(core.coarsen_threshold): 0.04,
    })
    instance = pops.bind(
        artifact,
        initial_state={"tracer": fine_initial, "coarse": coarse_initial},
        params=params,
    )
    return instance, artifact, coarse_id, fine_id, mapping_id, fine_initial, coarse_initial


def test_two_native_layouts_execute_sliced_programs_and_exact_transfer(compiled_multi_layout):
    instance, artifact, coarse_id, fine_id, mapping_id, fine_initial, coarse_initial = _bind(
        compiled_multi_layout
    )

    assert artifact.program is None
    assert {row.block_names for row in artifact.layout_programs} == {("coarse",), ("tracer",)}
    assert instance._executor_for_layout(coarse_id).nx() == 8
    assert instance._executor_for_layout(fine_id).nx() == 16
    assert instance._executor_for_block("tracer") is instance._executor_for_layout(fine_id)
    with pytest.raises(ValueError, match="executor_for_layout"):
        instance.nx()

    transfer = instance._runtime_plan.communication.transfers[0]
    receipt = instance._executor._apply_mapping(transfer)
    expected = fine_initial.reshape(1, 8, 2, 8, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(
        np.asarray(instance.get_state("coarse")).reshape(1, 8, 8), expected, rtol=0.0, atol=1.0e-15
    )
    assert receipt["applied"] is True
    assert instance._executor.mapping_report() == {mapping_id: 1}

    # The same installed native provider must honor both axes and every component; this rectangular
    # probe catches implementations that accidentally assume a square grid or component zero.
    from pops.runtime._component_execution_context import component_execution_data
    from pops.runtime._multi_layout_executor import _transfer_descriptor

    rectangular = np.arange(2 * 12 * 8, dtype=np.float64).reshape(2, 12, 8)
    reduced = np.empty((2, 4, 4), dtype=np.float64)
    rectangular_receipt = instance._installed_components[
        transfer.component_id
    ].native_handle._transfer_apply(
        _transfer_descriptor(rectangular, layout_id=fine_id, block="rectangular-source"),
        _transfer_descriptor(reduced, layout_id=coarse_id, block="rectangular-target"),
        (3, 2),
        int(LayoutMappingOperation.CONSERVATIVE_CELL_AVERAGE_V1),
        component_execution_data(instance._execution_context),
    )
    expected_rectangular = rectangular.reshape(2, 4, 3, 4, 2).mean(axis=(2, 4))
    np.testing.assert_array_equal(reduced, expected_rectangular)
    assert rectangular_receipt["destination_element_count"] == reduced.size

    instance._executor.set_state("coarse", coarse_initial.reshape(-1))
    run_report = pops.run(instance, t_end=DT, max_steps=1)
    assert run_report.accepted_steps == 1
    assert instance._executor.mapping_report() == {mapping_id: 2}
    assert (
        np.max(
            np.abs(
                np.asarray(instance.get_state("tracer")).reshape(fine_initial.shape) - fine_initial
            )
        )
        > 1.0e-8
    )
    assert (
        np.max(
            np.abs(
                np.asarray(instance.get_state("coarse")).reshape(coarse_initial.shape)
                - coarse_initial
            )
        )
        > 1.0
    )

    snapshot = instance.bound_snapshot.to_dict()
    assert snapshot["layout"]["report_type"] == "layout_plan"
    assert {row["name"] for row in snapshot["blocks"]} == {"coarse", "tracer"}
    inspection = instance.inspect()
    assert len(inspection.instance["layout_plan"]["layouts"]) == 2
    assert len(inspection.instance["installed_components"]) == 1


def test_multi_layout_checkpoint_restart_restores_every_layout_and_mapping_count(
    compiled_multi_layout, tmp_path
):
    instance, _artifact, _coarse_id, _fine_id, mapping_id, _fine, _coarse = _bind(
        compiled_multi_layout
    )
    pops.run(instance, t_end=DT, max_steps=1)
    expected = {name: np.asarray(instance.get_state(name)).copy() for name in ("coarse", "tracer")}
    expected_time = instance.time()
    expected_count = instance._executor.mapping_report()
    checkpoint = instance.checkpoint(tmp_path / "multi-layout")

    pops.run(instance, t_end=2.0 * DT, max_steps=1)
    assert instance._executor.mapping_report()[mapping_id] == expected_count[mapping_id] + 1
    instance.restart(checkpoint)

    assert instance.time() == expected_time
    assert instance._executor.mapping_report() == expected_count
    for name, values in expected.items():
        np.testing.assert_array_equal(np.asarray(instance.get_state(name)), values)


def test_failed_child_restart_rolls_back_already_restored_layouts(compiled_multi_layout, tmp_path):
    instance, _artifact, _coarse_id, _fine_id, _mapping_id, _fine, _coarse = _bind(
        compiled_multi_layout
    )
    pops.run(instance, t_end=DT, max_steps=1)
    checkpoint = instance.checkpoint(tmp_path / "rollback-source")
    pops.run(instance, t_end=2.0 * DT, max_steps=1)
    before = {name: np.asarray(instance.get_state(name)).copy() for name in ("coarse", "tracer")}
    before_time = instance.time()
    before_counts = instance._executor.mapping_report()

    native = instance._executor
    second_layout = tuple(native._engines)[1]
    original = native._engines[second_layout]

    class FailFirstRestart:
        def __init__(self, engine):
            object.__setattr__(self, "engine", engine)
            object.__setattr__(self, "failed", False)

        def __getattr__(self, name):
            return getattr(self.engine, name)

        def __setattr__(self, name, value):
            if name in {"engine", "failed"}:
                object.__setattr__(self, name, value)
            else:
                setattr(self.engine, name, value)

        def restart(self, path):
            if not self.failed:
                self.failed = True
                raise RuntimeError("injected second-layout restart failure")
            return self.engine.restart(path)

    native._engines[second_layout] = FailFirstRestart(original)
    with pytest.raises(RuntimeError, match="second-layout restart failure"):
        instance.restart(checkpoint)

    assert instance.time() == before_time
    assert instance._executor.mapping_report() == before_counts
    for name, values in before.items():
        np.testing.assert_array_equal(np.asarray(instance.get_state(name)), values)
    assert all(
        state is engine._temporal_restart_state
        for state, engine in zip(
            native._temporal_restart_state.states, native._engines.values(), strict=True
        )
    )


def test_post_load_composite_failure_rolls_back_children_counters_and_identity(
    compiled_multi_layout, tmp_path
):
    instance, _artifact, _coarse_id, _fine_id, _mapping_id, _fine, _coarse = _bind(
        compiled_multi_layout
    )
    pops.run(instance, t_end=DT, max_steps=1)
    checkpoint = instance.checkpoint(tmp_path / "post-load-source")
    pops.run(instance, t_end=2.0 * DT, max_steps=1)
    before = {name: np.asarray(instance.get_state(name)).copy() for name in ("coarse", "tracer")}
    before_time = instance.time()
    before_counts = instance._executor.mapping_report()
    native = instance._executor
    before_identity = native.last_restart_identity
    original = native._rebuild_composite_temporal_state
    calls = 0

    def fail_once_after_child_loads():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected post-load temporal divergence")
        return original()

    native._rebuild_composite_temporal_state = fail_once_after_child_loads
    with pytest.raises(RuntimeError, match="post-load temporal divergence"):
        instance.restart(checkpoint)

    assert calls == 2
    assert instance.time() == before_time
    assert instance._executor.mapping_report() == before_counts
    assert native.last_restart_identity == before_identity
    for name, values in before.items():
        np.testing.assert_array_equal(np.asarray(instance.get_state(name)), values)
    assert all(
        state is engine._temporal_restart_state
        for state, engine in zip(
            native._temporal_restart_state.states, native._engines.values(), strict=True
        )
    )
