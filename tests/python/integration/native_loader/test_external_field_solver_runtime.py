"""Real FieldTopology + FieldSolver components across resolve/compile/bind/run."""
from __future__ import annotations

import json

import numpy as np
import pops
import pytest
from pops import interfaces
from pops.external import build_source_package_manifest, load
from pops.fields import ExternalFieldSolver
from pops.model import ComponentManifest
from pops.time import FailRun, FixedDt
from tests.python.integration._final_field_program import (
    passive_field_model,
    resolve_periodic_field_program,
)


def _manifest(name, interface, parameters=()):
    return ComponentManifest(
        uri="pops://external.test/fields/%s" % name,
        component_type=interface.name,
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        parameters=parameters,
        target={"variants": [{
            "dimension": 2,
            "scalar": "float64",
            "device": "cpu",
            "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )


def _component(
    tmp_path, *, name, interface, source_factory,
    manifest_parameters=(), instance_parameters=None,
):
    root = tmp_path / name
    root.mkdir()
    alias = name.replace("-", "_")
    manifest = _manifest(name, interface, manifest_parameters)
    source = source_factory(manifest).encode()
    source_name = name + ".cpp"
    (root / source_name).write_bytes(source)
    package = build_source_package_manifest(
        components={alias: manifest}, payloads={source_name: ("source", source)})
    manifest_path = root / (name + ".pops.json")
    manifest_path.write_text(json.dumps(package), encoding="utf-8")
    factory = load(manifest_path).require(alias, interface=interface)
    return factory(**({} if instance_parameters is None else instance_parameters))


def _topology_source(manifest):
    return f'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstddef>
#include <cstring>

namespace {{
struct State {{ int prepare_count; int topology_count; }};

PopsComponentStatusV1 ok() {{
  return {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
}}

int prepare(const PopsComponentPrepareRequestV1* request, void** output,
            PopsComponentStatusV1* status) {{
  if (!request || !output || !status || request->execution.context_version != 1 ||
      !request->parameters_json ||
      std::strcmp(request->parameters_json, "{{}}") != 0) return 2;
  *output = new State{{1, 0}};
  *status = ok();
  return 0;
}}

void destroy(void* value) {{ delete static_cast<State*>(value); }}

int prepare_topology(void* value, const PopsFieldTopologyRequestV2* request,
                     PopsFieldTopologyResultV2* result) {{
  auto* state = static_cast<State*>(value);
  if (!state || state->prepare_count != 1 || ++state->topology_count != 1 ||
      !request || !result || !request->topology.topology_recipe_identity ||
      !request->topology.source_layout_identity ||
      !request->topology.materialized_layout_identity ||
      request->topology.dimension != 2 || request->topology.periodic_axes != 3 ||
      request->topology.patch_count == 0 ||
      request->local_patch_count > request->topology.patch_count) return 3;
  for (std::size_t local = 0; local < request->local_patch_count; ++local) {{
    const auto& patch = request->local_patches[local];
    if (patch.metadata_index >= request->topology.patch_count ||
        patch.material_representation != POPS_FIELD_MATERIAL_FULL_V1 ||
        patch.material_coverage.data || patch.cut_cell_volume_fraction.data ||
        patch.material_ids.data ||
        patch.material_mask.size != patch.component_labels.size) return 4;
    const auto& metadata = request->topology.patches[patch.metadata_index];
    if (metadata.dimension != 2 || metadata.cell_spacing[0] <= 0.0 ||
        metadata.cell_spacing[1] <= 0.0 || !metadata.layout_identity ||
        !metadata.patch_identity ||
        std::strcmp(metadata.layout_identity,
                    request->topology.source_layout_identity) != 0) return 5;
    for (std::size_t point = 0; point < patch.material_mask.size; ++point) {{
      patch.material_mask.data[point] = 1;
      patch.component_labels.data[point] = 1;
    }}
  }}
  static const PopsTopologyLabelV2 labels[] = {{
    {{sizeof(PopsTopologyLabelV2), 1, "material", "external-test-topology"}}
  }};
  result->label_count = 1;
  result->labels = labels;
  result->provenance = "external-test-topology";
  result->topology_digest = "external-test-topology-digest-v2";
  result->status = ok();
  return 0;
}}

const PopsFieldTopologyApiV2 table = {{
  {{sizeof(PopsFieldTopologyApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2, &prepare, &destroy}},
  &prepare_topology
}};
const PopsComponentInterfaceEntryV1 entry = {{
  POPS_NATIVE_INTERFACE_FIELD_TOPOLOGY_V2, 2,
  sizeof(PopsFieldTopologyApiV2), &table
}};
const PopsComponentApiV1 component = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL,
  POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)},
  1, &entry
}};
}}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{
  return &component;
}}
'''


def _solver_source(manifest, *, solution_expression="7.0"):
    expected_parameters_json = json.dumps(
        {"answer": 7}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstddef>
#include <cstring>
#include <limits>

namespace {{
struct State {{ int prepare_count; int solve_count; }};

PopsComponentStatusV1 ok() {{
  return {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
}}

int prepare(const PopsComponentPrepareRequestV1* request, void** output,
            PopsComponentStatusV1* status) {{
  if (!request || !output || !status || request->execution.context_version != 1 ||
      !request->parameters_json ||
      std::strcmp(request->parameters_json,
                  {json.dumps(expected_parameters_json)}) != 0) return 2;
  *output = new State{{1, 0}};
  *status = ok();
  return 0;
}}

void destroy(void* value) {{ delete static_cast<State*>(value); }}

int solve(void* value, const PopsFieldSolverRequestV2* request,
          PopsSolveReportV2* report) {{
  auto* state = static_cast<State*>(value);
  if (!state || state->prepare_count != 1 || !request || !report ||
      !request->topology.topology_recipe_identity ||
      !request->topology.source_layout_identity ||
      !request->topology.materialized_layout_identity ||
      request->topology.patch_count == 0 ||
      request->local_patch_count > request->topology.patch_count ||
      request->topology_label_count != 1 || !request->topology_labels ||
      request->topology_labels[0].struct_size < sizeof(PopsFieldSolverTopologyLabelV2) ||
      request->topology_labels[0].id != 1 || !request->topology_labels[0].label ||
      !request->topology_labels[0].provenance || !request->topology_provenance ||
      std::strcmp(request->topology_labels[0].label, "material") != 0 ||
      std::strcmp(request->topology_labels[0].provenance,
                  "external-test-topology") != 0 ||
      std::strcmp(request->topology_provenance,
                  "external-test-topology") != 0 ||
      !request->topology_digest ||
      std::strcmp(request->topology_digest,
                  "external-test-topology-digest-v2") != 0 ||
      !request->boundary_contract_json ||
      std::strstr(request->boundary_contract_json, "identity") == nullptr)
    return 3;
  ++state->solve_count;
  for (std::size_t local = 0; local < request->local_patch_count; ++local) {{
    const auto& patch = request->local_patches[local];
    if (patch.metadata_index >= request->topology.patch_count ||
        patch.rhs.dimension != 2 || patch.solution.dimension != 2 ||
        patch.material_mask.size != patch.component_labels.size) return 4;
    const auto& metadata = request->topology.patches[patch.metadata_index];
    if (std::strcmp(patch.rhs.layout_identity, metadata.layout_identity) != 0 ||
        std::strcmp(patch.solution.layout_identity, metadata.layout_identity) != 0 ||
        std::strcmp(patch.rhs.patch_identity, metadata.patch_identity) != 0 ||
        std::strcmp(patch.solution.patch_identity, metadata.patch_identity) != 0) return 6;
    const auto* mask = patch.material_mask.data;
    const auto* labels = patch.component_labels.data;
    auto* solution = static_cast<double*>(patch.solution.data);
    for (std::size_t j = 0; j < patch.solution.extents[1]; ++j) {{
      for (std::size_t i = 0; i < patch.solution.extents[0]; ++i) {{
        const std::size_t point = j * patch.solution.extents[0] + i;
        if (mask[point] != 1 || labels[point] != 1) return 5;
        const auto index = static_cast<std::ptrdiff_t>(i) *
                               patch.solution.axis_strides[0] +
                           static_cast<std::ptrdiff_t>(j) *
                               patch.solution.axis_strides[1];
        solution[index] = {solution_expression};
      }}
    }}
  }}
  report->status = POPS_SOLVE_SOLVED_V2;
  report->action = POPS_SOLVE_ACTION_NONE_V2;
  report->iterations = state->solve_count;
  report->relative_residual = 0.0;
  report->reference_residual_norm = 1.0;
  report->residual_norm = 0.0;
  report->reason = "tolerance reached";
  return 0;
}}

const PopsFieldSolverApiV2 table = {{
  {{sizeof(PopsFieldSolverApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2, &prepare, &destroy}},
  &solve
}};
const PopsComponentInterfaceEntryV1 entry = {{
  POPS_NATIVE_INTERFACE_FIELD_SOLVER_V2, 2,
  sizeof(PopsFieldSolverApiV2), &table
}};
const PopsComponentApiV1 component = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL,
  POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)},
  1, &entry
}};
}}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{
  return &component;
}}
'''


def _nonfinite_solver_source(manifest):
    return _solver_source(
        manifest,
        solution_expression="std::numeric_limits<double>::quiet_NaN()",
    )


def _program(state, rate, field):
    from pops.lib.time import ForwardEuler

    program = ForwardEuler(
        state, rate=rate, fields=field, solve_action=FailRun())
    program.step_strategy(FixedDt(1.0e-4))
    return program


def test_external_field_pair_executes_and_reports_materialized_topology(tmp_path):
    topology = _component(
        tmp_path, name="topology", interface=interfaces.FieldTopology,
        source_factory=_topology_source)
    solver = _component(
        tmp_path, name="solver", interface=interfaces.FieldSolver,
        source_factory=_solver_source,
        manifest_parameters=({"name": "answer", "kind": "runtime"},),
        instance_parameters={"answer": 7})
    provider = ExternalFieldSolver(
        topology=topology, solver=solver, relative_tolerance=1.0e-11,
        absolute_tolerance=0.0, max_iterations=23)
    model = passive_field_model("external-field-runtime", coefficient=0.0)
    resolved = resolve_periodic_field_program(
        model, _program, name="external-field-runtime", block_name="material",
        target="system", n=8, field_solver=provider,
        components=(topology, solver))

    artifact = pops.compile(resolved)
    simulation = pops.bind(
        artifact,
        initial_state={"material": np.ones((1, 8, 8), dtype=np.float64)},
    )
    slot, = simulation.field_provider_slots()
    before = simulation.inspect().to_dict()["instance"]["field_providers"]
    assert len(before) == 1
    assert before[0]["provider_slot"] == slot
    assert before[0]["provider"]["provider_id"] == "pops.fields.external-field-solver"
    assert "provider_kind" not in before[0]
    assert before[0]["materialized"] is False
    assert before[0]["topology_digest"] is None
    assert before[0]["patches"] == []
    assert not hasattr(simulation, "field_topology_report")

    report = pops.run(simulation, t_end=1.0e-4, max_steps=1)
    assert report.accepted_steps == 1
    potential = np.asarray(simulation.field_potential_global(slot)).reshape(-1)
    # The provider's constant 7.0 solution is published only after the resolved mean-zero gauge.
    assert potential.size == 64 and np.all(potential == 0.0)
    providers = report.to_data()["field_providers"]
    assert len(providers) == 1
    provider_report = providers[0]
    assert provider_report["provider_slot"] == slot
    assert provider_report["materialized"] is True
    assert provider_report["topology_digest"] == "external-test-topology-digest-v2"
    assert provider_report["provenance"] == "external-test-topology"
    assert provider_report["source_layout_identity"] == artifact.layout_plan.qualified_id
    assert provider_report["materialized_layout_identity"].startswith(
        "pops.runtime-field-layout.v1:sha256:")
    assert provider_report["patches"] == [{
        "patch_identity": provider_report["patches"][0]["patch_identity"],
        "topology_digest": "external-test-topology-digest-v2",
        "provenance": "external-test-topology",
        "material_points": 64,
        "connected_components": 1,
        "source_layout_identity": artifact.layout_plan.qualified_id,
        "materialized_layout_identity": provider_report["materialized_layout_identity"],
    }]
    assert provider_report["patches"][0]["patch_identity"].startswith(
        "pops.runtime-field-patch.v1:sha256:")
    assert simulation.inspect().to_dict()["instance"]["field_providers"] == providers


def test_external_field_solver_rejects_converged_nonfinite_solution_without_publishing(
    tmp_path,
):
    topology = _component(
        tmp_path, name="nonfinite-topology", interface=interfaces.FieldTopology,
        source_factory=_topology_source)
    solver = _component(
        tmp_path, name="nonfinite-solver", interface=interfaces.FieldSolver,
        source_factory=_nonfinite_solver_source,
        manifest_parameters=({"name": "answer", "kind": "runtime"},),
        instance_parameters={"answer": 7})
    provider = ExternalFieldSolver(
        topology=topology, solver=solver, relative_tolerance=1.0e-11,
        absolute_tolerance=0.0, max_iterations=23)
    model = passive_field_model("external-field-nonfinite", coefficient=0.0)
    resolved = resolve_periodic_field_program(
        model, _program, name="external-field-nonfinite", block_name="material",
        target="system", n=8, field_solver=provider,
        components=(topology, solver))

    simulation = pops.bind(
        pops.compile(resolved),
        initial_state={"material": np.ones((1, 8, 8), dtype=np.float64)},
    )
    slot, = simulation.field_provider_slots()
    before = np.asarray(simulation.field_potential_global(slot)).copy()
    assert before.size == 64 and np.all(before == 0.0)

    with pytest.raises(
        RuntimeError,
        match=r"field_solve failed: invalid_evaluation action=fail_run",
    ):
        pops.run(simulation, t_end=1.0e-4, max_steps=1)

    after = np.asarray(simulation.field_potential_global(slot))
    np.testing.assert_array_equal(after, before)
    assert np.all(np.isfinite(after))
