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
from pops.amr import ClusteringProvider, TaggerProvider
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
    "indicator_stencil_routes": list(
        NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]),
    "maximum_stencil_terms": NATIVE_TAGGING_PROGRAM_ABI[
        "maximum_stencil_terms"],
    "maximum_instruction_count": NATIVE_TAGGING_PROGRAM_ABI[
        "maximum_instruction_count"],
    "non_finite_policy": NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"],
    "persistent_hysteresis": NATIVE_TAGGING_PROGRAM_ABI["persistent_hysteresis"],
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


def test_external_amr_provider_refuses_a_3d_only_native_target(tmp_path):
    component = _component(
        tmp_path, name="tagger-3d", alias="tagger_3d",
        interface=interfaces.Tagger, dimension=3)

    with pytest.raises(ValueError, match="supported 2D float64"):
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


def _layout(authored, *, tagger, clustering):
    return AMR(
        grid=authored.grid,
        hierarchy=authored.hierarchy,
        tagging=authored.tagging,
        tagger=tagger,
        clustering=clustering,
        regrid=authored.regrid,
        transfer=authored.transfer,
        execution=authored.execution,
    )


def test_external_amr_providers_survive_resolution_with_exact_components(tmp_path):
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

    resolved = pops.resolve(
        pops.validate(target.authoring.case),
        layout=layout,
        components=(tagger_component, clustering_component),
    )

    assert tuple(resolved.amr_providers) == ("clustering", "tagger")
    tagger = resolved.amr_providers["tagger"]
    clustering = resolved.amr_providers["clustering"]
    assert tagger["provider_type"] == "external_amr_tagger"
    assert tagger["component_id"] == tagger_component.component_manifest.component_id
    assert tagger["tagging_graph_identity"] == resolved.bootstrap_plan.tagging.qualified_id
    assert tuple(tagger["tagging_capability"]["candidate_outputs"]) == tuple(
        TAGGER_CAPABILITY["candidate_outputs"])
    assert tagger["native_interface"] == interfaces.Tagger.to_data()
    assert clustering["provider_type"] == "external_amr_clustering"
    assert clustering["component_id"] == clustering_component.component_manifest.component_id
    assert clustering["native_interface"] == interfaces.Clustering.to_data()
    from pops.identity.semantic import semantic_value

    assert resolved.resolved_hierarchy.plan.clustering.options.to_data() == {
        "provider": semantic_value(dict(clustering), where="test clustering provider"),
    }


def test_third_party_authority_uses_the_open_provider_lowering_protocol(tmp_path):
    target = _example().build_final_case()
    tagger_component = _component(
        tmp_path, name="third_party_tagger", interface=interfaces.Tagger)
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
        tagger=TaggerProvider(tagger_component),
        clustering=ThirdPartyClusteringAuthority(
            ClusteringProvider(clustering_component)),
    )
    resolved = pops.resolve(
        pops.validate(target.authoring.case),
        layout=layout,
        components=(tagger_component, clustering_component),
    )

    assert resolved.amr_providers["clustering"]["component_id"] \
        == clustering_component.component_manifest.component_id
    assert resolved.amr_providers["clustering"]["runtime_installation"] == {
        "schema_version": 1,
        "protocol": "external_component",
    }


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


def test_external_clustering_options_survive_prepared_native_lowering(tmp_path):
    from pops.amr.providers import (
        AMRProviderLoweringContext,
        prepare_amr_provider_native_config,
    )

    component = _component(
        tmp_path,
        name="parameterized_clustering",
        interface=interfaces.Clustering,
        manifest_parameters=({"name": "options", "kind": "runtime"},),
        instance_parameters={"options": {"target_boxes": 12, "strict": True}},
    )
    graph = SimpleNamespace(qualified_id="test::tagging-graph")
    binding = ClusteringProvider(component).lower_amr_provider(
        AMRProviderLoweringContext(
            layout_identity="test::layout",
            components=(component,),
            tagging_graph=graph,
            clock_identity="test::clock",
        )
    )
    from pops.identity.semantic import semantic_value

    prepared = prepare_amr_provider_native_config(
        semantic_value(binding.data, where="test external clustering binding"))

    assert prepared.role == "clustering"
    assert prepared.config == {}
    assert prepared.provider_options == {
        "options": {"target_boxes": 12, "strict": True},
    }

    forged = dict(binding.data)
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
    persistent_clustering = _component(
        tmp_path, name="persistent_clustering", interface=interfaces.Clustering)
    target = _example().build_final_case()
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(advertised_but_unsupported),
        clustering=ClusteringProvider(persistent_clustering),
    )
    with pytest.raises(NotImplementedError, match="persistent_hysteresis is not implemented"):
        pops.resolve(
            pops.validate(target.authoring.case), layout=layout,
            components=(advertised_but_unsupported, persistent_clustering))


def test_external_tagger_refuses_graph_opcode_outside_manifest(tmp_path):
    target = _example().build_final_case()
    capability = {**TAGGER_CAPABILITY, "leaf_opcodes": ["above", "below"]}
    tagger_component = _component(
        tmp_path, name="limited_tagger", interface=interfaces.Tagger,
        tagger_capability=capability)
    clustering_component = _component(
        tmp_path, name="clustering", interface=interfaces.Clustering)
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=ClusteringProvider(clustering_component),
    )
    with pytest.raises(NotImplementedError, match="lacks resolved opcode"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(tagger_component, clustering_component),
        )


def test_external_tagger_refuses_resolved_stencil_beyond_its_capacity(tmp_path):
    target = _example().build_final_case()
    capability = {**TAGGER_CAPABILITY, "maximum_stencil_terms": 1}
    tagger_component = _component(
        tmp_path, name="thin_stencil_tagger", interface=interfaces.Tagger,
        tagger_capability=capability)
    clustering_component = _component(
        tmp_path, name="thin_stencil_clustering", interface=interfaces.Clustering)
    layout = _layout(
        target.layout,
        tagger=TaggerProvider(tagger_component),
        clustering=ClusteringProvider(clustering_component),
    )
    with pytest.raises(NotImplementedError, match="maximum_stencil_terms"):
        pops.resolve(
            pops.validate(target.authoring.case), layout=layout,
            components=(tagger_component, clustering_component))


def test_external_amr_provider_install_is_prevalidated_and_transactional():
    from pops._platform_contracts import (
        ExecutionContext,
        ExecutionResource,
        proven_serial_manifest,
    )
    from pops.runtime._runtime_authorities import _install_amr_provider_authorities

    layout_identity = "test::layout"
    clock_identity = "test::case::clock"
    graph_identity = "test::case::tagging-graph"
    from pops.amr.providers import (
        _normalize_tagger_capability,
        amr_provider_binding_identity,
    )
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
        if slot == "tagger":
            row.update({
                "clock_identity": clock_identity,
                "tagging_graph_identity": graph_identity,
                "tagging_capability": normalized_capability,
            })
        row["provider_identity"] = amr_provider_binding_identity(slot, row)
        return row

    tagger_handle, clustering_handle = object(), object()
    tagger_id, clustering_id = "test::tagger", "test::clustering"
    tagger_manifest, clustering_manifest = "manifest::tagger", "manifest::clustering"
    installed = {
        tagger_id: SimpleNamespace(
            component_manifest=SimpleNamespace(token=tagger_manifest),
            interface=interfaces.Tagger,
            native_handle=tagger_handle,
            runtime_contract=SimpleNamespace(capabilities=(TAGGER_CAPABILITY,)),
        ),
        clustering_id: SimpleNamespace(
            component_manifest=SimpleNamespace(token=clustering_manifest),
            interface=interfaces.Clustering,
            native_handle=clustering_handle,
            runtime_contract=SimpleNamespace(capabilities=()),
        ),
    }
    execution = ExecutionContext(
        backend=proven_serial_manifest(
            backend="production", target="amr_system",
            abi="test|external-amr-providers|v1", runtime=True),
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"),
    )
    plan = SimpleNamespace(
        amr_providers={
            "clustering": binding(
                "clustering", interfaces.Clustering,
                clustering_id, clustering_manifest),
            "tagger": binding(
                "tagger", interfaces.Tagger, tagger_id, tagger_manifest),
        },
        components=installed,
        execution_context=execution,
        artifact=SimpleNamespace(
            layout_plan=SimpleNamespace(qualified_id=layout_identity),
            plan=SimpleNamespace(blocks=()),
        ),
        bootstrap_plan=SimpleNamespace(
            tagging=SimpleNamespace(qualified_id=graph_identity)),
    )

    class Native:
        def __init__(self):
            self.calls = []
            self.discarded = False

        def _install_amr_clustering_component(self, *args):
            self.calls.append(("clustering", args))

        def _install_amr_tagger_component(self, *args):
            self.calls.append(("tagger", args))

        def _discard_amr_provider_components(self):
            self.calls.clear()
            self.discarded = True

    native = Native()
    engine = SimpleNamespace(_s=native)
    _install_amr_provider_authorities(engine, plan)
    assert [name for name, _ in native.calls] == ["clustering", "tagger"]
    assert native.calls[0][1][0] is clustering_handle
    assert native.calls[1][1][0] is tagger_handle
    assert tuple(engine._amr_provider_authorities) == ("clustering", "tagger")

    missing = SimpleNamespace(**vars(plan))
    missing.components = {clustering_id: installed[clustering_id]}
    untouched = Native()
    with pytest.raises(ValueError, match="not installed"):
        _install_amr_provider_authorities(SimpleNamespace(_s=untouched), missing)
    assert untouched.calls == []
    assert not untouched.discarded
