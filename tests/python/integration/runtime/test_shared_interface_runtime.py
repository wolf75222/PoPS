"""One real Case -> compile -> bind -> Program step through a native shared NumericalFlux."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pops
import pytest

from pops import interfaces
from pops.external import build_source_package_manifest, compile_component, load
from pops.linalg import LinearOperatorProperties, LinearProblem
from pops.mesh import CartesianGrid
from pops.mesh.boundaries import (
    BlockInterfaceSide,
    BoundaryComponentBinding,
    ConservativeInterface,
)
from pops.model import ComponentManifest
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import Flux
from pops.output import Checkpoint, ConsumerGraph, RegridOnRestart
from pops.solvers import GMRES
from pops.time import FailRun, FixedDt, StagePoint, TimePoint, every


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_shared_interface_scalar", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _flux_source_component(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    return load(package_path).require(
        "average", interface=interfaces.NumericalFlux)()


def _flux_component(tmp_path: Path):
    return compile_component(
        _flux_source_component(tmp_path),
        include=str(ROOT / "include"),
    )


def _tagger_source_component(tmp_path: Path):
    """Build the source Tagger used by the existing AMR retry scenario."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    interface = interfaces.Tagger
    from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI

    capability = {
        "schema_version": 1,
        "capability_type": "amr_tagging_program",
        "leaf_opcodes": list(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"]),
        "logical_opcodes": list(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"]),
        "candidate_outputs": list(NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"]),
        "indicator_stencil_routes": list(NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]),
        "maximum_stencil_terms": NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"],
        "maximum_instruction_count": NATIVE_TAGGING_PROGRAM_ABI["maximum_instruction_count"],
        "non_finite_policy": "reject",
        "persistent_hysteresis": False,
        "execution_mode": "host",
        "collective_scope": "none",
        "memory_spaces": ["host"],
    }
    manifest = ComponentManifest(
        uri="pops://external.test/shared-interface/tagger",
        component_type="tagger",
        version="1.0.0",
        facets=interface.facets,
        signature={"generic": True, "native_interface": interface.signature_declaration()},
        interfaces=interface.manifest_declarations(),
        capabilities=(capability,),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    expected_parameters_json = json.dumps(
        manifest.to_data()["parameters"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    expected_target_json = json.dumps(
        manifest.to_data()["target"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    source = f'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace {{
std::atomic<int> fail_once{{0}};

int prepare(const PopsComponentPrepareRequestV1* request, void** state,
            PopsComponentStatusV1* status) {{
  if (!request || !state || !status || !request->parameters_json || !request->target_json ||
      std::strcmp(request->parameters_json, {json.dumps(expected_parameters_json)}) != 0 ||
      std::strcmp(request->target_json, {json.dumps(expected_target_json)}) != 0) {{
    if (status) *status = {{sizeof(PopsComponentStatusV1), 71,
                            POPS_COMPONENT_ABORT_RUN_V1, "unauthenticated Tagger prepare JSON"}};
    return 71;
  }}
  *state = new int(771);
  *status = {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

void destroy(void* state) {{ delete static_cast<int*>(state); }}

int tag_batch(void* state, const PopsTaggerRequestV2* request, PopsComponentStatusV1* status) {{
  if (!state || *static_cast<int*>(state) != 771 || !request || !status ||
      request->execution_mode != POPS_TAGGER_EXECUTION_HOST_V2 ||
      request->collective_scope != POPS_TAGGER_COLLECTIVE_NONE_V2 ||
      request->execution.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      request->state_count != 2 || request->refine_candidates.size == 0 ||
      request->refine_candidates.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      request->coarsen_candidates.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      request->refine_equalities.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      request->coarsen_equalities.memory_space != POPS_MEMORY_SPACE_HOST_V1 ||
      request->coarsen_candidates.size != request->refine_candidates.size ||
      request->refine_equalities.size != request->refine_candidates.size ||
      request->coarsen_equalities.size != request->refine_candidates.size ||
      request->logical_time.tick < 0) return 72;
  if (request->logical_time.tick > 0 && fail_once.fetch_add(1) == 0) {{
    *status = {{sizeof(PopsComponentStatusV1), 73, POPS_COMPONENT_RETRY_STEP_V1,
                "injected rank-local Tagger failure"}};
    return 0;
  }}
  for (size_t point = 0; point < request->refine_candidates.size; ++point) {{
    bool refine = false;
    for (size_t state_index = 0; state_index < request->state_count; ++state_index) {{
      const PopsConstFieldViewV1& field = request->states[state_index].values;
      if (!field.data || field.dimension != 2 || field.component_count != 1 ||
          field.memory_space != POPS_MEMORY_SPACE_HOST_V1) return 74;
      size_t quotient = point;
      size_t offset = 0;
      for (int axis = 0; axis < 2; ++axis) {{
        const size_t interior = field.extents[axis] - field.ghost_lower[axis] - field.ghost_upper[axis];
        if (interior == 0) return 75;
        offset += (quotient % interior + field.ghost_lower[axis]) * field.axis_strides[axis];
        quotient /= interior;
      }}
      const double value = static_cast<const double*>(field.data)[offset];
      if (!std::isfinite(value)) return 76;
      refine = refine || value > 0.10;
    }}
    request->refine_candidates.data[point] = refine ? 1u : 0u;
    request->coarsen_candidates.data[point] = 0u;
    request->refine_equalities.data[point] = 0u;
    request->coarsen_equalities.data[point] = 0u;
  }}
  *status = {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

const PopsTaggerApiV2 tagger_table = {{
  {{sizeof(PopsTaggerApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_TAGGER_V2, 2, &prepare, &destroy}},
  &tag_batch
}};
const PopsComponentInterfaceEntryV1 entry = {{
  POPS_NATIVE_INTERFACE_TAGGER_V2, 2, sizeof(PopsTaggerApiV2), &tagger_table
}};
const PopsComponentApiV1 api = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL, POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)}, {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)}, 1, &entry
}};
}}
extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{ return &api; }}
'''.encode()
    source_name = "shared_tagger.cpp"
    (tmp_path / source_name).write_bytes(source)
    package = build_source_package_manifest(
        components={"tagger": manifest}, payloads={source_name: ("source", source)}
    )
    package_path = tmp_path / "shared-tagger.pops.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    return load(package_path).require("tagger", interface=interface)()


def _tagger_component(tmp_path: Path):
    return compile_component(_tagger_source_component(tmp_path), include=str(ROOT / "include"))


def _ghost_source_component(tmp_path: Path):
    """Build one real GhostBoundary library that fails once after dirtying ABI scratch."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    interface = interfaces.GhostBoundary
    manifest = ComponentManifest(
        uri="pops://external.test/uniform-boundary/fail-once",
        component_type="ghost_boundary",
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "state_components": 1,
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
    expected_parameters_json = json.dumps(
        manifest.to_data()["parameters"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    expected_target_json = json.dumps(
        manifest.to_data()["target"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    source = f"""#include <pops/runtime/config/generated_component_abi.hpp>
#include <atomic>
#include <cstddef>
#include <cstring>

namespace {{
std::atomic<int> apply_count{{0}};

int prepare(const PopsComponentPrepareRequestV1* request, void** state,
            PopsComponentStatusV1* status) {{
  if (!request || !state || !status || !request->parameters_json ||
      !request->target_json ||
      std::strcmp(request->parameters_json, {json.dumps(expected_parameters_json)}) != 0 ||
      std::strcmp(request->target_json, {json.dumps(expected_target_json)}) != 0) {{
    if (status)
      *status = {{sizeof(PopsComponentStatusV1), 51,
                  POPS_COMPONENT_ABORT_RUN_V1, "unauthenticated ghost prepare JSON"}};
    return 51;
  }}
  *state = new int(91);
  *status = {{sizeof(PopsComponentStatusV1), 0,
              POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

void destroy(void* state) {{ delete static_cast<int*>(state); }}

int apply(void* state, const PopsGhostBoundaryRequestV1* request,
          PopsComponentStatusV1* status) {{
  if (!state || *static_cast<int*>(state) != 91 || !request || !status ||
      !request->producer_identity || !request->state_identity ||
      !request->ghost_identity || !request->execution.execution_identity ||
      request->region.kind != POPS_BOUNDARY_FACE_V1 ||
      request->region.dimension != 2 || request->region.codimension != 1 ||
      request->region.axis_count != 1 || !request->region.axes ||
      !request->region.sides || request->region.axes[0] != 0 ||
      request->region.sides[0] != -1 || request->dependency_count != 0 ||
      request->parameter_count != 0 || !request->ghosts.data ||
      request->ghosts.dimension != 2 || request->ghosts.component_count != 1) {{
    if (status)
      *status = {{sizeof(PopsComponentStatusV1), 52,
                  POPS_COMPONENT_ABORT_RUN_V1, "incomplete exact ghost request"}};
    return 52;
  }}
  auto* ghosts = static_cast<double*>(request->ghosts.data);
  std::size_t points = 1;
  for (int axis = 0; axis < request->ghosts.dimension; ++axis)
    points *= request->ghosts.extents[axis];
  const bool fail = apply_count.fetch_add(1) == 0;
  for (std::size_t point = 0; point < points; ++point)
    ghosts[point] = fail ? -1234.0 : 9.0;
  if (fail) {{
    *status = {{sizeof(PopsComponentStatusV1), 53,
                POPS_COMPONENT_ABORT_RUN_V1, "injected ghost failure"}};
    return 53;
  }}
  *status = {{sizeof(PopsComponentStatusV1), 0,
              POPS_COMPONENT_CONTINUE_V1, nullptr}};
  return 0;
}}

const PopsGhostBoundaryApiV1 ghost_table = {{
  {{sizeof(PopsGhostBoundaryApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1, 1, &prepare, &destroy}},
  &apply
}};
const PopsComponentInterfaceEntryV1 entry = {{
  POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1, 1,
  sizeof(PopsGhostBoundaryApiV1), &ghost_table
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
""".encode()
    source_name = "uniform_fail_once_ghost.cpp"
    (tmp_path / source_name).write_bytes(source)
    package = build_source_package_manifest(
        components={"ghost": manifest}, payloads={source_name: ("source", source)}
    )
    package_path = tmp_path / "uniform-fail-once-ghost.pops.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    return load(package_path).require("ghost", interface=interfaces.GhostBoundary)()


class _ExternalGhostFaceExecutionAuthority:
    """Keep the resolved physical authority, but reserve one exact face for its component."""

    def __init__(self, base, producer_identity: str):
        self._base = base
        self._producer_identity = producer_identity

    def canonical_identity(self):
        return {
            "schema_version": 1,
            "authority_type": "external_ghost_face_test",
            "base": self._base.canonical_identity(),
            "producer_identity": self._producer_identity,
        }

    def _externalize(self, data):
        result = deepcopy(data)
        matches = [face for face in result["faces"] if face["producer"] == self._producer_identity]
        if len(matches) != 1:
            raise ValueError("external GhostBoundary test requires one exact physical face")
        matches[0]["type"] = "external"
        matches[0]["values"] = []
        return result

    def compile_boundary_data(self):
        return self._externalize(self._base.compile_boundary_data())

    def runtime_boundary_data(self, params):
        return self._externalize(self._base.runtime_boundary_data(params))


class _ExternalGhostBoundaryAuthority:
    """Open authoring composer used to attach one real source-built x-min provider."""

    def __init__(self, base, component):
        self._base = base
        self._component = component

    def inspect(self):
        return {
            "schema_version": 1,
            "authority_type": "external_ghost_boundary_test_authoring",
            "base": self._base.inspect(),
            "component_id": self._component.component_manifest.component_id,
            "component_manifest_identity": (
                self._component.component_manifest.manifest_digest.token
            ),
        }

    def resolve_for_numerics(self, context):
        return _ResolvedExternalGhostBoundaryAuthority(
            self._base.resolve_for_numerics(context), self._component
        )


class _ResolvedExternalGhostBoundaryAuthority:
    def __init__(self, base, component):
        self._base = base
        self._component = component

    def canonical_identity(self):
        return {
            "schema_version": 1,
            "authority_type": "external_ghost_boundary_test",
            "base": self._base.canonical_identity(),
            "component_id": self._component.component_manifest.component_id,
            "component_manifest_identity": (
                self._component.component_manifest.manifest_digest.token
            ),
        }

    def ghost_plan_composer_capability(self):
        return {"schema_version": 1, "scope": "self"}

    def compose_ghost_plan(self, context):
        from pops.mesh.boundaries.composition import compose_transport_boundary

        boundary = compose_transport_boundary(self._base, context=context)
        matches = [
            production
            for production in boundary.productions
            if production.region.boundary is not None
            and production.region.boundary.orientation.axis == 0
            and production.region.boundary.orientation.outward_sign == -1
        ]
        assert len(matches) == 1
        providers = matches[0].producer.boundary_providers
        assert len(providers) == 1
        target = providers[0].handle
        return replace(
            boundary,
            execution_authority=_ExternalGhostFaceExecutionAuthority(
                boundary.execution_authority, target.qualified_id
            ),
            component_bindings=(BoundaryComponentBinding(target, self._component),),
        )


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


def _implicit_pair_program(left_state, right_state, rate, packed_state=None):
    del rate
    program = pops.Program("shared_interface_implicit_pair")
    left = program.state(left_state)
    right = program.state(right_state)
    stage = StagePoint("shared_implicit_stage", {"main": TimePoint(program.clock, 0)})
    left_iterate = program.value("left_iterate", left.n, at=stage)
    right_iterate = program.value("right_iterate", right.n, at=stage)
    left_r0 = program.rhs("left_r0", state=left_iterate, terms=(Flux(),))
    right_r0 = program.rhs("right_r0", state=right_iterate, terms=(Flux(),))
    operator = program.matrix_free_operator(
        "shared_interface_jacobian", domain="state", range_="state", ncomp=2
    )

    def apply(builder, out, direction):
        builder.rhs_jacvec(
            out, direction, iterate=left_iterate, r0=left_r0, c_dt=builder.dt,
            sources=(), field_coupled=False,
        )
        return builder.rhs_jacvec(
            out, direction, iterate=right_iterate, r0=right_r0, c_dt=builder.dt,
            sources=(), field_coupled=False,
        )

    program.set_apply(operator, apply)
    if packed_state is not None:
        packed = program.state(packed_state)
        program.solve(
            LinearProblem(
                operator,
                packed.n,
                properties=LinearOperatorProperties.general(),
                nullspace=None,
            ),
            solver=GMRES(max_iter=8, restart=4, rel_tol=1.0e-12),
            name="shared_interface_correction",
        ).consume(action=FailRun())
    left_next = program.value("left_next", left.n + program.dt * left_r0, at=left.next.point)
    right_next = program.value("right_next", right.n + program.dt * right_r0, at=right.next.point)
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


def test_runtime_instance_executes_external_ghost_with_rollback_and_retry(tmp_path):
    example = _load_example()
    core = example.build_authoring(output_root=tmp_path / "unused")
    finite_volume = FiniteVolume(
        flux=core.flux,
        variables=variables.Conservative(core.state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.ScalarUpwind(velocity=core.velocity),
    )
    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow

    boundaries = core.frame.boundaries
    x_boundaries = boundaries.pair(core.frame.x)
    y_boundaries = boundaries.pair(core.frame.y)
    component = _ghost_source_component(tmp_path / "component")
    numerics = DiscretizationPlan()
    numerics.rates.add(core.rate, finite_volume)
    numerics.boundaries.add(
        _ExternalGhostBoundaryAuthority(
            TransportBoundarySet(
                {
                    x_boundaries.lower: Outflow(state=core.tracer_state),
                    x_boundaries.upper: Outflow(state=core.tracer_state),
                    y_boundaries.lower: Inflow(
                        state=core.tracer_state, value=core.inlet_y_value
                    ),
                    y_boundaries.upper: Outflow(state=core.tracer_state),
                }
            ),
            component,
        )
    )
    core.case.numerics(numerics, block=core.tracer)

    program = pops.Program("external_ghost_forward_euler")
    tracer = program.state(core.tracer_state)
    stage = StagePoint("external_ghost_stage", {"main": TimePoint(program.clock, 0)})
    rate = program.value("external_ghost_rate", core.rate(tracer.n), at=stage)
    next_state = program.value(
        "external_ghost_next", tracer.n + program.dt * rate, at=tracer.next.point
    )
    program.commit(tracer.next, next_state)
    program.step_strategy(FixedDt(1.0e-3))
    core.case.program(program)

    validated = pops.validate(core.case)
    from pops.layouts import Uniform

    resolved = pops.resolve(
        validated,
        layout=Uniform(CartesianGrid(frame=core.frame, cells=(8, 8))),
        components=(component,),
        compile_options={"include": str(ROOT / "include")},
    )
    resolved.verify()
    artifact = pops.compile(resolved)
    params = {
        core.case.resolve(handle, block=core.tracer): value
        for handle, value in (
            (core.velocity_x_param, 1.0),
            (core.velocity_y_param, 0.0),
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
    runtime = example._bind_artifact(
        artifact,
        initial_state={"tracer": np.ones((1, 8, 8), dtype=np.float64)},
        params=params,
    )
    native = runtime._executor._s
    before = (
        float(runtime.time()),
        int(runtime.macro_step()),
        bytes(native.program_accepted_state()),
        np.asarray(runtime.get_state("tracer"), dtype=np.float64).copy(),
    )

    with pytest.raises(RuntimeError, match="ghost|GhostBoundary|component"):
        pops.run(runtime, t_end=1.0e-3, max_steps=1)

    assert (float(runtime.time()), int(runtime.macro_step())) == before[:2]
    assert bytes(native.program_accepted_state()) == before[2]
    np.testing.assert_array_equal(
        np.asarray(runtime.get_state("tracer"), dtype=np.float64), before[3]
    )

    pops.run(runtime, t_end=1.0e-3, max_steps=1)

    assert runtime.time() == pytest.approx(1.0e-3, rel=0.0, abs=1.0e-15)
    assert runtime.macro_step() == 1
    state = np.asarray(runtime.get_state("tracer"), dtype=np.float64).reshape(1, 8, 8)
    np.testing.assert_allclose(
        state[0, 1:-1, 0],
        1.064,
        rtol=0.0,
        atol=1.0e-14,
    )


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


def _shared_interface_amr_authoring(
    tmp_path,
    *,
    component_root=None,
    component=None,
    tagger_component=None,
    program_factory=_ssprk2_program,
    with_checkpoint=True,
    with_implicit_solve=False,
):
    from pops.amr import (
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
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import BindArray
    from pops.math import ValueExpr, ddt, div
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.spaces import CellState

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
    component_root = tmp_path if component_root is None else Path(component_root)
    component_root.mkdir(parents=True, exist_ok=True)
    if component is None:
        component = _flux_component(component_root)
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
    packed_state = None
    packed_initial = None
    if with_implicit_solve:
        packed_model = pops.Model("shared_interface_packed_vector", frame=core.frame)
        packed = packed_model.state(
            "U",
            components=("left_direction", "right_direction"),
            representation=Conservative(),
            space=CellState(frame=core.frame),
        )
        left_direction, right_direction = packed
        zero_flux = packed_model.flux(
            "zero_packed_transport",
            frame=core.frame,
            state=packed,
            components={
                axis: (0.0 * left_direction, 0.0 * right_direction)
                for axis in core.frame.axes
            },
            waves={axis: (0.0, 0.0) for axis in core.frame.axes},
        )
        packed_rate = packed_model.rate(
            "zero_packed_rate", equation=ddt(packed) == -div(zero_flux)
        )
        packed_block = core.case.block(
            "implicit_vector", model=packed_model, states=(packed,)
        )
        packed_state = packed_block[packed]
        packed_numerics = DiscretizationPlan()
        packed_numerics.rates.add(
            packed_rate,
            FiniteVolume(
                flux=zero_flux,
                variables=variables.Conservative(packed),
                reconstruction=reconstruction.FirstOrder(),
                riemann=riemann.Rusanov(),
            ),
        )
        packed_numerics.boundaries.add(TransportBoundarySet({
            boundary: Outflow(state=packed_state)
            for boundary in (
                boundaries.x_min,
                boundaries.x_max,
                boundaries.y_min,
                boundaries.y_max,
            )
        }))
        core.case.numerics(packed_numerics, block=packed_block)
        core.case.initials.add(InitialCondition(
            state=packed_state,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        ))
        packed_initial = np.empty((2, 8, 8), dtype=np.float64)
        packed_initial[0, :, :] = 0.25
        packed_initial[1, :, :] = -0.125
        program = program_factory(
            core.tracer_state, right_state, core.rate, packed_state
        )
    else:
        program = program_factory(core.tracer_state, right_state, core.rate)
    core.case.program(program)
    if with_checkpoint:
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
    if packed_state is not None:
        transfer.state(packed_state, StateTransfer())
    tagging = AMRTagging(
        rules=(
            Tag(ValueExpr(core.tracer_state) > core.case.value(core.refine_threshold)),
            Tag(ValueExpr(right_state) > core.case.value(core.refine_threshold)),
            Buffer(cells=1),
        ),
        hysteresis=Hysteresis(min_cycles=0, equality=EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    )
    left_initial = np.zeros((1, 8, 8), dtype=np.float64)
    right_initial = np.zeros((1, 8, 8), dtype=np.float64)
    # The first public refined route requires an already matched fine interface: refine one
    # full-height coarse-cell band on both mapped faces while keeping the domain interior coarse.
    # One-sided tag propagation across a BlockInterface is a separate capability and must not be
    # implied by this proof.
    left_initial[0, :, -1:] = 1.0
    # Keep the two traces distinct: the shared component must publish its average flux to both
    # consumers. Equal traces would let a one-sided publication pass by coincidence.
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
    return SimpleNamespace(
        example=example,
        core=core,
        right=right,
        right_state=right_state,
        component=component,
        tagger_component=tagger_component,
        program=program,
        transfer=transfer,
        tagging=tagging,
        left_initial=left_initial,
        right_initial=right_initial,
        packed_state=packed_state,
        packed_initial=packed_initial,
        params=params,
    )


def _resolve_shared_interface_amr(
    authoring, *, max_levels, patch_layout=None, frozen=False, regrid_interval=100
):
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        TaggerProvider,
    )
    from pops.layouts import AMR

    if not isinstance(max_levels, int) or max_levels < 2:
        raise ValueError("shared-interface AMR proof requires at least two levels")
    return pops.resolve(
        pops.validate(authoring.core.case),
        layout=AMR(
            grid=CartesianGrid(frame=authoring.core.frame, cells=(8, 8)),
            hierarchy=AMRHierarchy(
                max_levels=max_levels,
                ratios=tuple(2 for _ in range(max_levels - 1)),
            ),
            tagging=authoring.tagging,
            tagger=(
                TaggerProvider(authoring.tagger_component)
                if authoring.tagger_component is not None
                else None
            ),
            regrid=(
                AMRRegrid.frozen() if frozen else
                AMRRegrid(schedule=every(regrid_interval, clock=authoring.program.clock))
            ),
            transfer=authoring.transfer,
            execution=AMRExecution.subcycled(
                tuple(
                    AMRClockRelation(level, level + 1, 2)
                    for level in range(max_levels - 1)
                )
            ),
            patch_layout=patch_layout,
        ),
        components=tuple(
            component
            for component in (authoring.component, authoring.tagger_component)
            if component is not None
        ),
        compile_options={"include": str(ROOT / "include")},
    )


def test_frozen_two_level_shared_interface_implicit_pair_compiles_native_route(tmp_path):
    authoring = _shared_interface_amr_authoring(
        tmp_path,
        program_factory=_implicit_pair_program,
        with_checkpoint=False,
    )
    resolved = _resolve_shared_interface_amr(authoring, max_levels=2, frozen=True)
    assert resolved.resolved_hierarchy.plan.level_count == 2
    assert resolved.capabilities["shared_interfaces"] == {
        "implicit_jacvec_pair": True,
    }
    from pops.codegen._shared_interface_evidence import (
        _ResolvedSharedInterfaceCodegenEvidence,
        _issue_shared_interface_codegen_evidence,
    )

    with pytest.raises(
        TypeError, match="issued only from an exact resolved plan"
    ):
        _ResolvedSharedInterfaceCodegenEvidence()
    evidence = _issue_shared_interface_codegen_evidence(resolved)
    assert type(evidence) is _ResolvedSharedInterfaceCodegenEvidence
    with pytest.raises(ValueError, match="belongs to another Program graph"):
        evidence.require(pops.Program("foreign_program"), target="amr_system")

    artifact = pops.compile(resolved)

    assert artifact.target == "amr_system"
    assert artifact.program is not None
    generated_path = artifact.program.dump_cpp(tmp_path / "implicit_pair.cpp")
    source = Path(generated_path).read_text(encoding="utf-8")
    assert source.count("ctx.rhs_jacvec_pair_into_at(") == 1
    assert source.count("ctx.copy_component_span(") >= 7
    assert "ctx.rhs_core_into_at(" not in source
    assert "PreparedOperatorConcurrency::Exclusive" in source
    group_identity = re.search(r"ctx\.rhs_group\((\d+),", source)
    assert group_identity is not None
    left_r0 = next(
        value for value in resolved.time._values if value.name == "left_r0"
    )
    from pops.codegen.program_emit_solve import _rhs_evaluation_identity

    assert str(_rhs_evaluation_identity(resolved.time, left_r0)) == group_identity.group(1)


def test_frozen_two_level_generated_program_executes_shared_interface_implicit_pair(tmp_path):
    authoring = _shared_interface_amr_authoring(
        tmp_path,
        program_factory=_implicit_pair_program,
        with_checkpoint=False,
        with_implicit_solve=True,
    )
    resolved = _resolve_shared_interface_amr(authoring, max_levels=2, frozen=True)
    artifact = pops.compile(resolved)
    interface = resolved.blocks[0].numerics.boundaries[0].interfaces[0]
    runtime = authoring.example._bind_artifact(
        artifact,
        initial_values={
            authoring.core.tracer_state: authoring.left_initial,
            authoring.right_state: authoring.right_initial,
            authoring.packed_state: authoring.packed_initial,
        },
        params=authoring.params,
    )

    assert runtime.n_levels() == 2
    initial_packed = np.asarray(runtime.get_state("implicit_vector")).copy()
    report = pops.run(runtime, t_end=1.0e-3, max_steps=1, console=False)

    assert report.accepted_steps == 1
    for level in range(2):
        assert runtime._executor._s._interface_evaluation_count(
            interface.qualified_id, level
        ) > 1
    solved_packed = np.asarray(runtime.get_state("implicit_vector"))
    assert np.isfinite(solved_packed).all()
    np.testing.assert_array_equal(solved_packed, initial_packed)


def test_runtime_instance_executes_dynamic_three_level_shared_flux(tmp_path, monkeypatch):
    authoring = _shared_interface_amr_authoring(
        tmp_path, tagger_component=_tagger_component(tmp_path / "tagger")
    )
    example = authoring.example
    core = authoring.core
    right_state = authoring.right_state
    left_initial = authoring.left_initial
    right_initial = authoring.right_initial
    params = authoring.params
    resolved = _resolve_shared_interface_amr(authoring, max_levels=3, regrid_interval=1)
    artifact = pops.compile(resolved)
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
    # face. The incremental finalizer must reject that incomplete pair before bind freezes.
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

    rollback_before_tagger_retry = _shared_interface_accepted_image(runtime)
    from pops._bootstrap import StepAttemptRejected

    with pytest.raises(StepAttemptRejected) as rejected:
        pops.run(runtime, t_end=1.0e-3, max_steps=1, console=False)
    assert rejected.value.status == "invalid_evaluation"
    assert rejected.value.disposition == "retry"
    assert rejected.value.reason_code == 73
    assert rejected.value.phase == "amr_tagger"
    assert rejected.value.detail == "injected rank-local Tagger failure"
    _assert_same_shared_interface_image(runtime, rollback_before_tagger_retry)

    retry_report = pops.run(runtime, t_end=1.0e-3, max_steps=1, console=False)
    assert retry_report.accepted_steps == 1

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
    restart_resolved = _resolve_shared_interface_amr(authoring, max_levels=2)
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

    # RegridOnRestart enters the native tag/cluster/regrid boundary. A deliberately rejected
    # post-transform validation must restore the fresh runtime exactly before the same restart is
    # retried and committed. This remains independent of the typed Tagger retry proved above.
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
