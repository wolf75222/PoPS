"""Exact external AMR provider authoring and resolve contracts."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pops
import pytest

from pops import interfaces
from pops.amr import ClusteringProvider, RefluxProvider, TaggerProvider
from pops.external import build_source_package_manifest, load
from pops.layouts import AMR
from pops.model import ComponentManifest
from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"
TAGGER_CAPABILITY = {
    "schema_version": 1,
    "capability_type": "amr_tagging_program",
    "leaf_opcodes": list(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"]),
    "logical_opcodes": list(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"]),
    "candidate_outputs": list(NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"]),
    "indicator_stencil_routes": list(NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]),
    "maximum_stencil_terms": NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"],
    "maximum_instruction_count": NATIVE_TAGGING_PROGRAM_ABI["maximum_instruction_count"],
    "non_finite_policy": NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"],
    # The external component evaluates candidates only; the canonical runtime owns persistent
    # hysteresis and accepted-state publication.
    "persistent_hysteresis": False,
    "execution_mode": "native_backend",
    "collective_scope": "none",
    "memory_spaces": ["host"],
}


def _example():
    spec = importlib.util.spec_from_file_location("pops_external_amr_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tagging_opcode_catalog_is_the_single_python_cpp_authority():
    from pops.amr import providers
    from pops.runtime import _runtime_mesh_lowering

    assert providers._TAGGER_LEAF_OPCODES == dict(NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"])
    assert providers._TAGGER_LOGICAL_OPCODES == dict(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])
    assert _runtime_mesh_lowering._TAG_LEAF_OPS == providers._TAGGER_LEAF_OPCODES
    assert _runtime_mesh_lowering._TAG_LOGICAL_OPS == providers._TAGGER_LOGICAL_OPCODES
    header = (ROOT / "include/pops/runtime/config/generated_component_abi.hpp").read_text()
    for name, opcode in {
        **providers._TAGGER_LEAF_OPCODES,
        **providers._TAGGER_LOGICAL_OPCODES,
    }.items():
        assert "POPS_TAGGING_%s_V1 = %d" % (name.upper(), opcode) in header
    assert NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"] == ["linear_axis_stencil_l2_v1"]
    assert (
        "POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 %d"
        % (NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"])
        in header
    )
    assert "POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1" in header
    assert NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"] == "reject"
    assert NATIVE_TAGGING_PROGRAM_ABI["persistent_hysteresis"] is True
    assert "POPS_TAGGING_NON_FINITE_REJECT_V1 1" in header


def _component(
    tmp_path: Path,
    *,
    name: str,
    interface,
    tagger_capability=TAGGER_CAPABILITY,
    dimension: int = 2,
    device: str = "cpu",
    alias: str | None = None,
    manifest_parameters=(),
    instance_parameters=None,
):
    export_alias = name if alias is None else alias
    root = tmp_path / name
    root.mkdir()
    manifest = ComponentManifest(
        uri="pops://external.test/amr/%s" % name,
        component_type=interface.name,
        version="1.0.0",
        facets=interface.facets,
        signature={
            "generic": True,
            "native_interface": interface.signature_declaration(),
        },
        interfaces=interface.manifest_declarations(),
        parameters=manifest_parameters,
        capabilities=(tagger_capability,)
        if interface is interfaces.Tagger and tagger_capability is not None
        else (),
        target={
            "variants": [
                {
                    "dimension": dimension,
                    "scalar": "float64",
                    "device": device,
                    "features": [],
                }
            ]
        },
        determinism={"classification": "bitwise", "scope": ["same-input"]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = b"// resolve-only external AMR component\n"
    source_name = name + ".cpp"
    (root / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={export_alias: manifest}, payloads={source_name: ("source", source)}
    )
    package_path = root / (name + ".pops.json")
    package_path.write_text(json.dumps(package_data), encoding="utf-8")
    factory = load(package_path).require(export_alias, interface=interface)
    return factory(**({} if instance_parameters is None else instance_parameters))


def _runtime_tagger_component(tmp_path: Path, *, fault_marker: Path):
    name = "runtime-tagger"
    root = tmp_path / name
    root.mkdir()
    manifest = ComponentManifest(
        uri="pops://external.test/amr/%s" % name,
        component_type=interfaces.Tagger.name,
        version="1.0.0",
        facets=interfaces.Tagger.facets,
        signature={
            "generic": True,
            "native_interface": interfaces.Tagger.signature_declaration(),
        },
        interfaces=interfaces.Tagger.manifest_declarations(),
        capabilities=(TAGGER_CAPABILITY,),
        target={
            "variants": [
                {
                    "dimension": 2,
                    "scalar": "float64",
                    "device": "cpu",
                    "features": ["mpi"],
                }
            ]
        },
        determinism={"classification": "bitwise", "scope": ["same-input"]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = f"""#include <pops/runtime/config/generated_component_abi.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace {{
PopsComponentStatusV1 ok() {{
  return {{sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr}};
}}

int prepare(const PopsComponentPrepareRequestV1*, void** output,
            PopsComponentStatusV1* status) {{
  *output = new std::uint64_t{{0}};
  *status = ok();
  return 0;
}}

void destroy(void* state) {{ delete static_cast<std::uint64_t*>(state); }}

int tag_batch(void* raw, const PopsTaggerRequestV2* request,
              PopsComponentStatusV1* status) {{
  auto* calls = static_cast<std::uint64_t*>(raw);
  if (!calls || !request || !status) return 2;
  ++*calls;
  if (std::filesystem::exists({json.dumps(str(fault_marker))})) {{
    if (request->refine_candidates.data && request->refine_candidates.size != 0)
      request->refine_candidates.data[0] = std::uint8_t{{1}};
    *status = {{sizeof(PopsComponentStatusV1), 31, POPS_COMPONENT_ABORT_RUN_V1,
               "injected native AMR Tagger failure"}};
    return 31;
  }}
  const std::size_t points = request->refine_candidates.size;
  std::fill_n(request->refine_candidates.data, points, std::uint8_t{{0}});
  std::fill_n(request->coarsen_candidates.data, points, std::uint8_t{{0}});
  std::fill_n(request->refine_equalities.data, points, std::uint8_t{{0}});
  std::fill_n(request->coarsen_equalities.data, points, std::uint8_t{{0}});
  const auto evaluate = [&](const std::int32_t* opcodes, const std::int32_t* arguments,
                            std::size_t instruction_count,
                            PopsTaggerMaskViewV2 candidates,
                            PopsTaggerMaskViewV2 equalities) -> bool {{
    std::size_t interior[3]{{1, 1, 1}};
    for (std::int32_t axis = 0; axis < request->states[0].values.dimension; ++axis)
      interior[axis] = request->states[0].values.extents[axis] -
                       request->states[0].values.ghost_lower[axis] -
                       request->states[0].values.ghost_upper[axis];
    for (std::size_t point = 0; point < points; ++point) {{
      std::size_t coordinate[3]{{0, 0, 0}};
      std::size_t quotient = point;
      for (std::int32_t axis = 0; axis < request->states[0].values.dimension; ++axis) {{
        coordinate[axis] = quotient % interior[axis];
        quotient /= interior[axis];
      }}
      bool matches[POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1]{{}};
      bool equality[POPS_TAGGING_MAXIMUM_INSTRUCTION_COUNT_V1]{{}};
      std::size_t depth = 0;
      for (std::size_t instruction = 0; instruction < instruction_count; ++instruction) {{
        const std::int32_t opcode = opcodes[instruction];
        const std::int32_t argument = arguments[instruction];
        if (pops_tagging_opcode_is_leaf_v1(opcode)) {{
          const auto& leaf = request->program.leaves[argument];
          const auto& view = request->states[leaf.state_index].values;
          const auto* values = static_cast<const double*>(view.data);
          const auto read = [&](const std::ptrdiff_t offset_axis[3]) {{
            std::ptrdiff_t offset =
                static_cast<std::ptrdiff_t>(leaf.component) * view.component_stride;
            for (std::int32_t axis = 0; axis < view.dimension; ++axis)
              offset += (static_cast<std::ptrdiff_t>(coordinate[axis]) +
                         static_cast<std::ptrdiff_t>(view.ghost_lower[axis]) +
                         offset_axis[axis]) * view.axis_strides[axis];
            return values[offset];
          }};
          const std::ptrdiff_t zero[3]{{0, 0, 0}};
          double sample = read(zero);
          if (opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1) sample = std::abs(sample);
          if (opcode == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
              opcode == POPS_TAGGING_GRADIENT_BELOW_V1) {{
            const auto& stencil = request->program.stencils[leaf.stencil_index];
            double squared_norm = 0.0;
            for (std::size_t row = 0; row < stencil.axis_count; ++row) {{
              const auto& axis = stencil.axes[row];
              double derivative = 0.0;
              for (std::size_t term = 0; term < axis.term_count; ++term) {{
                std::ptrdiff_t offset[3]{{0, 0, 0}};
                offset[axis.axis] = axis.offsets[term];
                derivative += axis.coefficients[term] * read(offset);
              }}
              derivative /= request->cell_size[axis.axis];
              squared_norm += derivative * derivative;
            }}
            sample = std::sqrt(squared_norm);
          }}
          if (!std::isfinite(sample)) return false;
          const bool greater = opcode == POPS_TAGGING_ABOVE_V1 ||
                               opcode == POPS_TAGGING_MAGNITUDE_ABOVE_V1 ||
                               opcode == POPS_TAGGING_GRADIENT_ABOVE_V1;
          matches[depth] = greater ? sample > leaf.threshold : sample < leaf.threshold;
          equality[depth] = sample == leaf.threshold;
          ++depth;
        }} else if (opcode == POPS_TAGGING_NOT_V1) {{
          if (!equality[depth - 1]) matches[depth - 1] = !matches[depth - 1];
        }} else {{
          const std::size_t begin = depth - static_cast<std::size_t>(argument);
          bool any_true = false, any_false = false, any_unknown = false;
          for (std::size_t child = begin; child < depth; ++child) {{
            any_unknown = any_unknown || equality[child];
            any_true = any_true || (matches[child] && !equality[child]);
            any_false = any_false || (!matches[child] && !equality[child]);
          }}
          depth = begin + 1;
          matches[begin] = opcode == POPS_TAGGING_ANY_OF_V1
                               ? any_true : !any_false && !any_unknown;
          equality[begin] = opcode == POPS_TAGGING_ANY_OF_V1
                                ? !any_true && any_unknown : !any_false && any_unknown;
        }}
      }}
      if (instruction_count != 0) {{
        candidates.data[point] = matches[0] ? 1u : 0u;
        equalities.data[point] = equality[0] ? 1u : 0u;
      }}
    }}
    return true;
  }};
  if (!evaluate(request->program.refine_opcodes, request->program.refine_arguments,
                request->program.refine_instruction_count, request->refine_candidates,
                request->refine_equalities) ||
      !evaluate(request->program.coarsen_opcodes, request->program.coarsen_arguments,
                request->program.coarsen_instruction_count, request->coarsen_candidates,
                request->coarsen_equalities)) {{
    *status = {{sizeof(PopsComponentStatusV1), 32, POPS_COMPONENT_ABORT_RUN_V1,
               "non-finite native AMR Tagger sample"}};
    return 32;
  }}
  *status = ok();
  return 0;
}}

const PopsTaggerApiV2 table = {{
  {{sizeof(PopsTaggerApiV2), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_TAGGER_V2, 2, &prepare, &destroy}},
  &tag_batch
}};
const PopsComponentInterfaceEntryV1 entry = {{
  POPS_NATIVE_INTERFACE_TAGGER_V2, 2, sizeof(PopsTaggerApiV2), &table
}};
const PopsComponentApiV1 component = {{
  sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
  POPS_ABI_KEY_LITERAL, POPS_COMPONENT_CATALOG_SHA256_V1,
  {json.dumps(manifest.component_id)},
  {json.dumps(manifest.semantic_digest.token)},
  {json.dumps(manifest.manifest_digest.token)},
  1, &entry
}};
}}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {{
  return &component;
}}
""".encode()
    source_name = name + ".cpp"
    (root / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={"runtime_tagger": manifest},
        payloads={source_name: ("source", source)},
    )
    package_path = root / (name + ".pops.json")
    package_path.write_text(json.dumps(package_data), encoding="utf-8")
    return load(package_path).require("runtime_tagger", interface=interfaces.Tagger)()


def test_external_amr_provider_accepts_a_3d_native_target_without_selecting_2d(tmp_path):
    component = _component(
        tmp_path, name="tagger-3d", alias="tagger_3d", interface=interfaces.Tagger, dimension=3
    )

    provider = TaggerProvider(component)
    assert provider.inspect()["component_id"] == component.component_manifest.component_id


def test_external_amr_provider_refuses_a_target_outside_ranked_dimensions(tmp_path):
    component = _component(
        tmp_path, name="tagger-4d", alias="tagger_4d", interface=interfaces.Tagger, dimension=4
    )

    with pytest.raises(ValueError, match="dimension 1, 2, or 3"):
        TaggerProvider(component)


def test_external_tagger_native_backend_accepts_an_exact_gpu_target(tmp_path):
    component = _component(
        tmp_path,
        name="tagger-cuda",
        alias="tagger_cuda",
        interface=interfaces.Tagger,
        device="cuda",
        tagger_capability={**TAGGER_CAPABILITY, "memory_spaces": ["managed"]},
    )

    provider = TaggerProvider(component)
    assert provider.inspect()["tagging_capability"]["execution_mode"] == "native_backend"
    assert provider.inspect()["tagging_capability"]["memory_spaces"] == ["managed"]

    mismatched = _component(
        tmp_path,
        name="tagger-cuda-host",
        alias="tagger_cuda_host",
        interface=interfaces.Tagger,
        device="cuda",
    )
    with pytest.raises(ValueError, match="requires 'managed' field memory"):
        TaggerProvider(mismatched)


def _layout(authored, *, tagger, clustering, tagging=None, reflux=None):
    return AMR(
        grid=authored.grid,
        hierarchy=authored.hierarchy,
        tagging=authored.tagging if tagging is None else tagging,
        tagger=tagger,
        clustering=clustering,
        reflux=authored.reflux if reflux is None else reflux,
        regrid=authored.regrid,
        transfer=authored.transfer,
        execution=authored.execution,
    )


def _runtime_tagger_case(component):
    from pops.amr import (
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
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.lib.amr import BergerRigoutsos, StateTransfer
    from pops.lib.initial import BindArray
    from pops.lib.time import ForwardEuler
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.params import RuntimeParam
    from pops.time import FixedDt, every

    domain = Rectangle("external-tagger-domain", lower=(0.0, 0.0), upper=(1.0, 1.0))
    frame = domain.frame(Cartesian2D())
    axis_x, axis_y = frame.axes
    model = pops.Model("external_tagger_model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={axis_x: (0.0 * rho,), axis_y: (0.0 * rho,)},
        waves={axis_x: (0.0 * rho,), axis_y: (0.0 * rho,)},
    )
    source = model.source("default", on=state, value=(0.01 * rho,))
    rate = model.rate("rhs", equation=ddt(state) == -div(flux) + source)
    case = pops.Case("external-tagger-case")
    block = case.block("material", model)
    block_state = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block_state, rate=rate)
    program.step_strategy(FixedDt(1.0e-3))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=block_state,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    threshold = case.param(RuntimeParam("refine_threshold", default=0.5))
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(8, 8),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block_state) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        tagger=TaggerProvider(component),
        clustering=BergerRigoutsos(),
        regrid=AMRRegrid(schedule=every(5, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    return case, layout, threshold, block_state


def test_external_tagger_resolves_while_unimplemented_provider_roles_fail_closed(tmp_path):
    target = _example().build_final_case()
    tagger_component = _component(tmp_path, name="tagger", interface=interfaces.Tagger)
    clustering_component = _component(tmp_path, name="clustering", interface=interfaces.Clustering)
    reflux_component = _component(tmp_path, name="reflux", interface=interfaces.Reflux)
    cases = (
        (
            "clustering",
            _layout(
                target.layout,
                tagger=target.layout.tagger,
                clustering=ClusteringProvider(clustering_component),
            ),
            (clustering_component,),
            "BergerRigoutsosProvider<Dim>",
        ),
        (
            "reflux",
            _layout(
                target.layout,
                tagger=target.layout.tagger,
                clustering=target.layout.clustering,
                reflux=RefluxProvider(reflux_component),
            ),
            (reflux_component,),
            "transactional AmrRuntime<Dim> reflux ledger",
        ),
    )
    for role, layout, components, authority in cases:
        with pytest.raises(
            NotImplementedError,
            match=r"external AMR %s component installation.*%s" % (role, authority),
        ):
            pops.resolve(
                pops.validate(target.authoring.case),
                layout=layout,
                components=components,
            )
    supported_case, supported_layout, _, _ = _runtime_tagger_case(tagger_component)
    resolved = pops.resolve(
        pops.validate(supported_case),
        layout=supported_layout,
        components=(tagger_component,),
    )
    assert resolved.amr_providers["tagger"]["component_id"] == (
        tagger_component.component_manifest.component_id
    )


def test_third_party_external_authority_uses_open_lowering_but_fails_closed(tmp_path):
    target = _example().build_final_case()
    clustering_component = _component(
        tmp_path, name="third_party_clustering", interface=interfaces.Clustering
    )

    @dataclass(frozen=True, slots=True)
    class ThirdPartyClusteringAuthority:
        delegate: ClusteringProvider
        __pops_ir_immutable__ = True

        def inspect(self):
            return self.delegate.inspect()

        def resolve_references(self, resolver):
            if not callable(resolver):
                raise TypeError("resolver must be callable")
            return self

        def lower_amr_provider(self, context):
            return self.delegate.lower_amr_provider(context)

    layout = _layout(
        target.layout,
        tagger=target.layout.tagger,
        clustering=ThirdPartyClusteringAuthority(ClusteringProvider(clustering_component)),
    )
    with pytest.raises(
        NotImplementedError,
        match=r"external AMR clustering component installation.*"
        r"BergerRigoutsosProvider<Dim>",
    ):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(clustering_component,),
        )


def test_incomplete_third_party_authority_is_rejected_explicitly(tmp_path):
    target = _example().build_final_case()
    tagger_component = _component(tmp_path, name="incomplete_tagger", interface=interfaces.Tagger)
    clustering_component = _component(
        tmp_path, name="incomplete_clustering", interface=interfaces.Clustering
    )

    @dataclass(frozen=True, slots=True)
    class IncompleteClusteringAuthority:
        delegate: ClusteringProvider
        __pops_ir_immutable__ = True

        def inspect(self):
            return self.delegate.inspect()

        def resolve_references(self, resolver):
            return self

    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=IncompleteClusteringAuthority(ClusteringProvider(clustering_component)),
    )
    with pytest.raises(TypeError, match="lower_amr_provider"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(tagger_component, clustering_component),
        )


def test_external_amr_providers_require_exact_resolve_inputs(tmp_path):
    target = _example().build_final_case()
    tagger_component = _component(tmp_path, name="tagger", interface=interfaces.Tagger)
    clustering_component = _component(tmp_path, name="clustering", interface=interfaces.Clustering)
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=ClusteringProvider(clustering_component),
    )

    with pytest.raises(ValueError, match="requires its exact ExternalComponent"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(tagger_component,),
        )


def test_external_clustering_options_remain_inspectable_but_native_lowering_fails_closed(tmp_path):
    from pops.amr.providers import (
        amr_provider_binding_identity,
        prepare_amr_provider_native_config,
    )

    component = _component(
        tmp_path,
        name="parameterized_clustering",
        interface=interfaces.Clustering,
        manifest_parameters=({"name": "options", "kind": "runtime"},),
        instance_parameters={"options": {"target_boxes": 12, "strict": True}},
    )
    binding = ClusteringProvider(component).inspect()
    assert binding["component"]["parameters"] == {
        "options": {"target_boxes": 12, "strict": True},
    }
    binding["layout_identity"] = "test::layout"
    binding["provider_identity"] = amr_provider_binding_identity("clustering", binding)

    with pytest.raises(
        NotImplementedError,
        match=r"external AMR clustering component installation.*"
        r"BergerRigoutsosProvider<Dim>",
    ):
        prepare_amr_provider_native_config(binding)

    forged = dict(binding)
    forged_component = dict(forged["component"])
    forged_component["parameters"] = {
        "options": {"target_boxes": 99, "strict": False},
    }
    forged["component"] = forged_component
    from pops.amr import ResolvedAMRProviderBinding

    with pytest.raises(ValueError, match="provider_identity"):
        ResolvedAMRProviderBinding("clustering", forged)


def test_external_amr_provider_roles_are_not_interchangeable(tmp_path):
    tagger_component = _component(tmp_path, name="tagger", interface=interfaces.Tagger)
    clustering_component = _component(tmp_path, name="clustering", interface=interfaces.Clustering)

    with pytest.raises(TypeError, match="requires exact interface"):
        TaggerProvider(clustering_component)
    with pytest.raises(TypeError, match="requires exact interface"):
        ClusteringProvider(tagger_component)


def test_external_tagger_requires_exact_candidate_program_capability(tmp_path):
    missing = _component(
        tmp_path, name="missing_capability", interface=interfaces.Tagger, tagger_capability=None
    )
    with pytest.raises(ValueError, match="exactly one amr_tagging_program"):
        TaggerProvider(missing)
    non_finite_fallback = _component(
        tmp_path,
        name="non_finite_fallback",
        interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "non_finite_policy": "false"},
    )
    with pytest.raises(ValueError, match="reject every non-finite"):
        TaggerProvider(non_finite_fallback)
    implicit_execution = _component(
        tmp_path,
        name="implicit_execution",
        interface=interfaces.Tagger,
        tagger_capability={
            key: value for key, value in TAGGER_CAPABILITY.items() if key != "execution_mode"
        },
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        TaggerProvider(implicit_execution)
    collective_execution = _component(
        tmp_path,
        name="collective_execution",
        interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "collective_scope": "rank"},
    )
    with pytest.raises(ValueError, match="explicitly noncollective"):
        TaggerProvider(collective_execution)
    disguised_host_fallback = _component(
        tmp_path,
        name="disguised_host_fallback",
        interface=interfaces.Tagger,
        tagger_capability={
            **TAGGER_CAPABILITY,
            "execution_mode": "host",
            "memory_spaces": ["host", "managed"],
        },
    )
    with pytest.raises(ValueError, match="exactly the host memory space"):
        TaggerProvider(disguised_host_fallback)
    advertised_but_unsupported = _component(
        tmp_path,
        name="persistent_capability",
        interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "persistent_hysteresis": True},
    )
    from pops.amr import AMRTagging, EqualityPolicy, Hysteresis

    supported_case, supported_layout, _, _ = _runtime_tagger_case(advertised_but_unsupported)

    persistent_tagging = AMRTagging(
        rules=supported_layout.tagging.rules,
        hysteresis=Hysteresis(3, EqualityPolicy.HOLD),
        conflict_policy=supported_layout.tagging.conflict_policy,
    )
    layout = _layout(
        supported_layout,
        tagger=TaggerProvider(advertised_but_unsupported),
        clustering=supported_layout.clustering,
        tagging=persistent_tagging,
    )
    resolved = pops.resolve(
        pops.validate(supported_case),
        layout=layout,
        components=(advertised_but_unsupported,),
    )
    assert resolved.amr_providers["tagger"]["tagging_capability"]["persistent_hysteresis"] is True


def test_external_tagger_refuses_graph_opcode_outside_manifest(tmp_path):
    target = _example().build_final_case()
    capability = {**TAGGER_CAPABILITY, "leaf_opcodes": ["above", "below"]}
    tagger_component = _component(
        tmp_path, name="limited_tagger", interface=interfaces.Tagger, tagger_capability=capability
    )
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=target.layout.clustering,
    )
    with pytest.raises(NotImplementedError, match="lacks resolved opcode"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(tagger_component,),
        )


def test_external_tagger_refuses_resolved_stencil_beyond_its_capacity(tmp_path):
    target = _example().build_final_case()
    capability = {**TAGGER_CAPABILITY, "maximum_stencil_terms": 1}
    tagger_component = _component(
        tmp_path,
        name="thin_stencil_tagger",
        interface=interfaces.Tagger,
        tagger_capability=capability,
    )
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=target.layout.clustering,
    )
    with pytest.raises(NotImplementedError, match="maximum_stencil_terms"):
        pops.resolve(
            pops.validate(target.authoring.case), layout=layout, components=(tagger_component,)
        )


def test_external_amr_provider_bind_fails_closed_without_native_mutation():
    from pops.amr.providers import (
        _normalize_tagger_capability,
        amr_provider_binding_identity,
        prepare_amr_provider_installation,
    )
    from pops.runtime._runtime_authorities import _install_amr_provider_authorities

    layout_identity = "test::layout"
    clock_identity = "test::case::clock"
    graph_identity = "test::case::tagging-graph"
    normalized_capability = _normalize_tagger_capability((TAGGER_CAPABILITY,))

    def binding(slot, interface, component_id, manifest):
        row = {
            "schema_version": 1,
            "provider_type": "external_amr_%s" % slot,
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "external_component",
            },
            "provider_identity": "test::%s-provider" % slot,
            "component_id": component_id,
            "component_manifest_identity": manifest,
            "component": {
                "component_id": component_id,
                "component_manifest": manifest,
                "interface": interface.to_data(),
            },
            "native_interface": interface.to_data(),
            "interface_version": interface.version,
            "layout_identity": layout_identity,
        }
        if slot in {"tagger", "reflux"}:
            row["clock_identity"] = clock_identity
        if slot == "tagger":
            row.update(
                {
                    "tagging_graph_identity": graph_identity,
                    "tagging_capability": normalized_capability,
                }
            )
        row["provider_identity"] = amr_provider_binding_identity(slot, row)
        return row

    providers = {
        "clustering": binding(
            "clustering", interfaces.Clustering, "test::clustering", "manifest::clustering"
        ),
        "tagger": binding("tagger", interfaces.Tagger, "test::tagger", "manifest::tagger"),
        "reflux": binding("reflux", interfaces.Reflux, "test::reflux", "manifest::reflux"),
    }
    executable_authorities = {
        "clustering": "BergerRigoutsosProvider<Dim>",
        "reflux": "transactional AmrRuntime<Dim> reflux ledger",
    }
    prepared_tagger = prepare_amr_provider_installation(
        role="tagger",
        frozen_binding=providers["tagger"],
        layout_identity=layout_identity,
        resolved_tagging_identity=graph_identity,
    )
    assert prepared_tagger.binding == providers["tagger"]
    for role in ("clustering", "reflux"):
        frozen = providers[role]
        with pytest.raises(
            NotImplementedError,
            match=r"external AMR %s component installation.*%s"
            % (role, executable_authorities[role]),
        ):
            prepare_amr_provider_installation(
                role=role,
                frozen_binding=frozen,
                layout_identity=layout_identity,
                resolved_tagging_identity=graph_identity,
            )

    native = SimpleNamespace(calls=[])
    engine = SimpleNamespace(_s=native)
    plan = SimpleNamespace(
        amr_providers=providers,
        artifact=SimpleNamespace(layout_plan=SimpleNamespace(qualified_id=layout_identity)),
        bootstrap_plan=SimpleNamespace(tagging=SimpleNamespace(qualified_id=graph_identity)),
    )
    with pytest.raises(
        NotImplementedError,
        match=r"external AMR clustering component installation.*"
        r"BergerRigoutsosProvider<Dim>",
    ):
        _install_amr_provider_authorities(engine, plan)
    assert native.calls == []
    assert not hasattr(engine, "_amr_provider_authorities")


def test_source_built_external_tagger_regrids_rolls_back_and_retries(tmp_path):
    fault_marker = tmp_path / "tagger-fault"
    component = _runtime_tagger_component(tmp_path, fault_marker=fault_marker)
    case, layout, threshold, initial_subject = _runtime_tagger_case(component)
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        components=(component,),
    )
    artifact = pops.compile(resolved)
    from tests.python.support.native_execution_context import artifact_execution_context

    simulation = pops.bind(
        artifact,
        params={case.resolve(threshold): 0.5},
        initial_values={
            initial_subject: np.ascontiguousarray(np.ones((1, 8, 8), dtype=np.float64))
        },
        resources={"execution_context": artifact_execution_context(artifact)},
    )
    authority = simulation._executor._amr_provider_authorities["tagger"]
    assert authority["component_id"] == component.component_manifest.component_id
    assert authority["provider_identity"].startswith("resolved-amr-tagger-provider:")

    first = pops.run(simulation, t_end=1.0, max_steps=4, console=False)
    assert first.accepted_steps == 4
    accepted = {
        "time": simulation.time(),
        "step": simulation.macro_step(),
        "levels": tuple(
            np.asarray(
                simulation.block_level_state_global("material", level),
                dtype=np.float64,
            ).copy()
            for level in range(simulation.n_levels())
        ),
        "boxes": tuple(simulation.patch_boxes()),
        "regrid": simulation.amr.explain_regrid().to_dict(),
        "program": simulation.program_report().to_dict(),
        "providers": dict(simulation._executor._amr_provider_authorities),
    }
    fault_marker.write_text("fail the next candidate evaluation", encoding="utf-8")
    with pytest.raises(RuntimeError, match="injected native AMR Tagger failure"):
        pops.run(simulation, t_end=1.0, max_steps=1, console=False)
    assert simulation.time() == accepted["time"]
    assert simulation.macro_step() == accepted["step"]
    assert tuple(simulation.patch_boxes()) == accepted["boxes"]
    assert simulation.amr.explain_regrid().to_dict() == accepted["regrid"]
    assert simulation.program_report().to_dict() == accepted["program"]
    assert dict(simulation._executor._amr_provider_authorities) == accepted["providers"]
    for level, reference in enumerate(accepted["levels"]):
        np.testing.assert_array_equal(
            np.asarray(
                simulation.block_level_state_global("material", level),
                dtype=np.float64,
            ),
            reference,
        )

    fault_marker.unlink()
    retried = pops.run(simulation, t_end=1.0, max_steps=1, console=False)
    assert retried.accepted_steps == 1
    assert simulation.macro_step() == accepted["step"] + 1
    assert simulation.amr.explain_regrid().regrid_count > accepted["regrid"]["regrid_count"]
