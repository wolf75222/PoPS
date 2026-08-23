"""Exact external AMR provider authoring and resolve contracts."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

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
    "leaf_opcodes": [
        opcode for opcode in NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"]
        if opcode != "prescribed_window"
    ],
    "logical_opcodes": list(NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"]),
    "candidate_outputs": list(NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"]),
    "indicator_stencil_routes": list(
        NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]),
    "maximum_stencil_terms": NATIVE_TAGGING_PROGRAM_ABI[
        "maximum_stencil_terms"],
    "maximum_instruction_count": NATIVE_TAGGING_PROGRAM_ABI[
        "maximum_instruction_count"],
    "non_finite_policy": NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"],
    # The external component evaluates candidates only.  The current adapter deliberately
    # refuses non-zero hysteresis even though the builtin runtime owns that accepted state.
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

    assert providers._TAGGER_LEAF_OPCODES == dict(
        NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"])
    assert providers._EXTERNAL_TAGGER_LEAF_OPCODES == {
        name: opcode
        for name, opcode in NATIVE_TAGGING_PROGRAM_ABI["leaf_opcodes"].items()
        if name != "prescribed_window"
    }
    assert providers._TAGGER_LOGICAL_OPCODES == dict(
        NATIVE_TAGGING_PROGRAM_ABI["logical_opcodes"])
    assert _runtime_mesh_lowering._TAG_LEAF_OPS == providers._TAGGER_LEAF_OPCODES
    assert _runtime_mesh_lowering._TAG_LOGICAL_OPS == providers._TAGGER_LOGICAL_OPCODES
    header = (ROOT / "include/pops/runtime/config/generated_component_abi.hpp").read_text()
    for name, opcode in {
            **providers._TAGGER_LEAF_OPCODES,
            **providers._TAGGER_LOGICAL_OPCODES}.items():
        assert "POPS_TAGGING_%s_V1 = %d" % (name.upper(), opcode) in header
    assert NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"] == [
        "linear_axis_stencil_l2_v1"]
    assert "POPS_TAGGING_MAXIMUM_STENCIL_TERMS_V1 %d" % (
        NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"]) in header
    assert "POPS_TAGGING_STENCIL_ROUTE_LINEAR_AXIS_STENCIL_L2_V1" in header
    assert NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"] == "reject"
    assert NATIVE_TAGGING_PROGRAM_ABI["persistent_hysteresis"] is True
    assert "POPS_TAGGING_NON_FINITE_REJECT_V1 1" in header


def _component(
    tmp_path: Path, *, name: str, interface, tagger_capability=TAGGER_CAPABILITY,
    dimension: int = 2, device: str = "cpu", alias: str | None = None, manifest_parameters=(),
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
        if interface is interfaces.Tagger and tagger_capability is not None else (),
        target={"variants": [{
            "dimension": dimension,
            "scalar": "float64",
            "device": device,
            "features": [],
        }]},
        determinism={"classification": "bitwise", "scope": ["same-input"]},
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    source = b"// resolve-only external AMR component\n"
    source_name = name + ".cpp"
    (root / source_name).write_bytes(source)
    package_data = build_source_package_manifest(
        components={export_alias: manifest}, payloads={source_name: ("source", source)})
    package_path = root / (name + ".pops.json")
    package_path.write_text(json.dumps(package_data), encoding="utf-8")
    factory = load(package_path).require(export_alias, interface=interface)
    return factory(**({} if instance_parameters is None else instance_parameters))


def test_external_amr_provider_accepts_a_3d_native_target_without_selecting_2d(tmp_path):
    component = _component(
        tmp_path, name="tagger-3d", alias="tagger_3d",
        interface=interfaces.Tagger, dimension=3)

    provider = TaggerProvider(component)
    assert provider.inspect()["component_id"] == component.component_manifest.component_id


def test_external_amr_provider_refuses_a_target_outside_ranked_dimensions(tmp_path):
    component = _component(
        tmp_path, name="tagger-4d", alias="tagger_4d",
        interface=interfaces.Tagger, dimension=4)

    with pytest.raises(ValueError, match="dimension 1, 2, or 3"):
        TaggerProvider(component)


def test_external_tagger_native_backend_accepts_an_exact_gpu_target(tmp_path):
    component = _component(
        tmp_path, name="tagger-cuda", alias="tagger_cuda",
        interface=interfaces.Tagger, device="cuda",
        tagger_capability={**TAGGER_CAPABILITY, "memory_spaces": ["managed"]})

    provider = TaggerProvider(component)
    assert provider.inspect()["tagging_capability"]["execution_mode"] == "native_backend"
    assert provider.inspect()["tagging_capability"]["memory_spaces"] == ["managed"]

    mismatched = _component(
        tmp_path, name="tagger-cuda-host", alias="tagger_cuda_host",
        interface=interfaces.Tagger, device="cuda")
    with pytest.raises(ValueError, match="requires 'managed' field memory"):
        TaggerProvider(mismatched)


def test_external_tagger_host_mode_publishes_staging_for_cpu_and_device_targets(tmp_path):
    host_capability = {
        **TAGGER_CAPABILITY,
        "execution_mode": "host",
        "memory_spaces": ["host"],
    }
    cpu_component = _component(
        tmp_path,
        name="tagger-cpu-host-execution",
        alias="tagger_cpu_host_execution",
        interface=interfaces.Tagger,
        device="cpu",
        tagger_capability=host_capability,
    )
    assert TaggerProvider(cpu_component).inspect()["tagging_capability"][
        "execution_mode"
    ] == "host"

    component = _component(
        tmp_path,
        name="tagger-cuda-host-execution",
        alias="tagger_cuda_host_execution",
        interface=interfaces.Tagger,
        device="cuda",
        tagger_capability=host_capability,
    )

    provider = TaggerProvider(component)
    assert provider.inspect()["tagging_capability"]["execution_mode"] == "host"
    assert provider.inspect()["tagging_capability"]["memory_spaces"] == ["host"]


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


def test_external_amr_provider_installers_fail_closed_at_resolution(tmp_path):
    target = _example().build_final_case()
    clustering_component = _component(
        tmp_path, name="clustering", interface=interfaces.Clustering)
    reflux_component = _component(
        tmp_path, name="reflux", interface=interfaces.Reflux)
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
                match=r"external AMR %s component installation.*%s"
                % (role, authority)):
            pops.resolve(
                pops.validate(target.authoring.case),
                layout=layout,
                components=components,
            )


def test_third_party_external_authority_uses_open_lowering_but_fails_closed(tmp_path):
    target = _example().build_final_case()
    clustering_component = _component(
        tmp_path, name="third_party_clustering", interface=interfaces.Clustering)

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
        clustering=ThirdPartyClusteringAuthority(
            ClusteringProvider(clustering_component)),
    )
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR clustering component installation.*"
                  r"BergerRigoutsosProvider<Dim>"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(clustering_component,),
        )


def test_incomplete_third_party_authority_is_rejected_explicitly(tmp_path):
    target = _example().build_final_case()
    tagger_component = _component(
        tmp_path, name="incomplete_tagger", interface=interfaces.Tagger)
    clustering_component = _component(
        tmp_path, name="incomplete_clustering", interface=interfaces.Clustering)

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
        clustering=IncompleteClusteringAuthority(
            ClusteringProvider(clustering_component)),
    )
    with pytest.raises(TypeError, match="lower_amr_provider"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(tagger_component, clustering_component),
        )


def test_external_amr_providers_require_exact_resolve_inputs(tmp_path):
    target = _example().build_final_case()
    tagger_component = _component(
        tmp_path, name="tagger", interface=interfaces.Tagger)
    clustering_component = _component(
        tmp_path, name="clustering", interface=interfaces.Clustering)
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


def test_external_clustering_options_remain_inspectable_but_native_lowering_fails_closed(
        tmp_path):
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
    binding["provider_identity"] = amr_provider_binding_identity(
        "clustering", binding)

    with pytest.raises(
            NotImplementedError,
            match=r"external AMR clustering component installation.*"
                  r"BergerRigoutsosProvider<Dim>"):
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
    tagger_component = _component(
        tmp_path, name="tagger", interface=interfaces.Tagger)
    clustering_component = _component(
        tmp_path, name="clustering", interface=interfaces.Clustering)

    with pytest.raises(TypeError, match="requires exact interface"):
        TaggerProvider(clustering_component)
    with pytest.raises(TypeError, match="requires exact interface"):
        ClusteringProvider(tagger_component)


def test_external_tagger_requires_exact_candidate_program_capability(tmp_path):
    missing = _component(
        tmp_path, name="missing_capability", interface=interfaces.Tagger,
        tagger_capability=None)
    with pytest.raises(ValueError, match="exactly one amr_tagging_program"):
        TaggerProvider(missing)
    non_finite_fallback = _component(
        tmp_path, name="non_finite_fallback", interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "non_finite_policy": "false"})
    with pytest.raises(ValueError, match="reject every non-finite"):
        TaggerProvider(non_finite_fallback)
    implicit_execution = _component(
        tmp_path, name="implicit_execution", interface=interfaces.Tagger,
        tagger_capability={key: value for key, value in TAGGER_CAPABILITY.items()
                           if key != "execution_mode"})
    with pytest.raises(ValueError, match="unsupported schema"):
        TaggerProvider(implicit_execution)
    collective_execution = _component(
        tmp_path, name="collective_execution", interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "collective_scope": "rank"})
    with pytest.raises(ValueError, match="explicitly noncollective"):
        TaggerProvider(collective_execution)
    disguised_host_fallback = _component(
        tmp_path, name="disguised_host_fallback", interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "execution_mode": "host",
                           "memory_spaces": ["host", "managed"]})
    with pytest.raises(ValueError, match="exactly the host memory space"):
        TaggerProvider(disguised_host_fallback)
    advertised_but_unsupported = _component(
        tmp_path, name="persistent_capability", interface=interfaces.Tagger,
        tagger_capability={**TAGGER_CAPABILITY, "persistent_hysteresis": True})
    target = _example().build_final_case()
    from pops.amr import AMRTagging, EqualityPolicy, Hysteresis

    persistent_tagging = AMRTagging(
        rules=target.layout.tagging.rules,
        hysteresis=Hysteresis(3, EqualityPolicy.HOLD),
        conflict_policy=target.layout.tagging.conflict_policy,
    )
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(advertised_but_unsupported),
        clustering=target.layout.clustering,
        tagging=persistent_tagging,
    )
    with pytest.raises(
            NotImplementedError,
            match="external AMR Tagger persistent_hysteresis is not implemented"):
        pops.resolve(
            pops.validate(target.authoring.case), layout=layout,
            components=(advertised_but_unsupported,))


def test_external_tagger_refuses_graph_opcode_outside_manifest(tmp_path):
    target = _example().build_final_case()
    capability = {**TAGGER_CAPABILITY, "leaf_opcodes": ["above", "below"]}
    tagger_component = _component(
        tmp_path, name="limited_tagger", interface=interfaces.Tagger,
        tagger_capability=capability)
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


def test_external_tagger_refuses_prescribed_window_without_a_geometry_abi(tmp_path):
    from pops.amr import PrescribedWindow
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian
    from pops.time import Clock

    component = _component(tmp_path, name="window_tagger", interface=interfaces.Tagger)
    frame = CartesianDomain("window", (0.0,), (1.0,)).frame(Cartesian(1))
    window = PrescribedWindow(
        frame=frame,
        clock=Clock("window"),
        center=(0.25,),
        half_width=(0.1,),
        velocity=(1.0,),
    )
    graph = SimpleNamespace(
        registrations=(SimpleNamespace(node_type="prescribed_window"),),
        graph=SimpleNamespace(
            refine=window,
            coarsen=None,
            hysteresis=SimpleNamespace(min_cycles=0),
        ),
    )

    with pytest.raises(NotImplementedError, match="PopsTaggingLeafV1 carries no window geometry"):
        TaggerProvider(component).require_tagging_graph(graph)


def test_external_tagger_capability_cannot_advertise_builtin_window_opcode(tmp_path):
    capability = {
        **TAGGER_CAPABILITY,
        "leaf_opcodes": [*TAGGER_CAPABILITY["leaf_opcodes"], "prescribed_window"],
    }
    component = _component(
        tmp_path, name="forged_window_capability", interface=interfaces.Tagger,
        tagger_capability=capability,
    )

    with pytest.raises(ValueError, match="cannot advertise prescribed_window"):
        TaggerProvider(component)


def test_external_tagger_refuses_resolved_stencil_beyond_its_capacity(tmp_path):
    target = _example().build_final_case()
    capability = {**TAGGER_CAPABILITY, "maximum_stencil_terms": 1}
    tagger_component = _component(
        tmp_path, name="thin_stencil_tagger", interface=interfaces.Tagger,
        tagger_capability=capability)
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=target.layout.clustering,
    )
    with pytest.raises(NotImplementedError, match="maximum_stencil_terms"):
        pops.resolve(
            pops.validate(target.authoring.case), layout=layout,
            components=(tagger_component,))


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
            row.update({
                "tagging_graph_identity": graph_identity,
                "tagging_capability": normalized_capability,
            })
        row["provider_identity"] = amr_provider_binding_identity(slot, row)
        return row

    providers = {
        "clustering": binding(
            "clustering", interfaces.Clustering,
            "test::clustering", "manifest::clustering"),
        "tagger": binding(
            "tagger", interfaces.Tagger, "test::tagger", "manifest::tagger"),
        "reflux": binding(
            "reflux", interfaces.Reflux, "test::reflux", "manifest::reflux"),
    }
    executable_authorities = {
        "clustering": "BergerRigoutsosProvider<Dim>",
        "tagger": "PreparedTaggingExecutionPlan<Dim>",
        "reflux": "transactional AmrRuntime<Dim> reflux ledger",
    }
    for role, frozen in providers.items():
        if role == "tagger":
            prepared = prepare_amr_provider_installation(
                role=role,
                frozen_binding=frozen,
                layout_identity=layout_identity,
                resolved_tagging_identity=graph_identity,
            )
            assert prepared.role == "tagger"
            assert prepared.binding["runtime_installation"] == {
                "schema_version": 1,
                "protocol": "external_component",
            }
            continue
        with pytest.raises(
                NotImplementedError,
                match=r"external AMR %s component installation.*%s"
                % (role, executable_authorities[role])):
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
        artifact=SimpleNamespace(
            layout_plan=SimpleNamespace(qualified_id=layout_identity)),
        bootstrap_plan=SimpleNamespace(
            tagging=SimpleNamespace(qualified_id=graph_identity)),
    )
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR clustering component installation.*"
                  r"BergerRigoutsosProvider<Dim>"):
        _install_amr_provider_authorities(engine, plan)
    assert native.calls == []
    assert not hasattr(engine, "_amr_provider_authorities")
