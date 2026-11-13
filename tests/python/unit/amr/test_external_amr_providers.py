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
from pops.amr import ClusteringProvider, RefluxProvider
from pops.external import build_source_package_manifest, load
from pops.layouts import AMR
from pops.model import ComponentManifest


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"
def _example():
    spec = importlib.util.spec_from_file_location("pops_external_amr_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _component(
    tmp_path: Path, *, name: str, interface,
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
        capabilities=(),
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
        tmp_path, name="clustering-3d", alias="clustering_3d",
        interface=interfaces.Clustering, dimension=3)

    provider = ClusteringProvider(component)
    assert provider.inspect()["component_id"] == component.component_manifest.component_id


def test_external_amr_provider_refuses_a_target_outside_ranked_dimensions(tmp_path):
    component = _component(
        tmp_path, name="clustering-4d", alias="clustering_4d",
        interface=interfaces.Clustering, dimension=4)

    with pytest.raises(ValueError, match="dimension 1, 2, or 3"):
        ClusteringProvider(component)


def _layout(authored, *, clustering, tagging=None, reflux=None):
    return AMR(
        grid=authored.grid,
        hierarchy=authored.hierarchy,
        tagging=authored.tagging if tagging is None else tagging,
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
                clustering=ClusteringProvider(clustering_component),
            ),
            (clustering_component,),
            "BergerRigoutsosProvider<Dim>",
        ),
        (
            "reflux",
            _layout(
                target.layout,
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
        clustering=IncompleteClusteringAuthority(
            ClusteringProvider(clustering_component)),
    )
    with pytest.raises(TypeError, match="lower_amr_provider"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(clustering_component,),
        )


def test_external_amr_providers_require_exact_resolve_inputs(tmp_path):
    target = _example().build_final_case()
    clustering_component = _component(
        tmp_path, name="clustering", interface=interfaces.Clustering)
    reflux_component = _component(
        tmp_path, name="reflux", interface=interfaces.Reflux)
    layout = _layout(
        target.layout,
        clustering=ClusteringProvider(clustering_component),
        reflux=RefluxProvider(reflux_component),
    )

    with pytest.raises(ValueError, match="requires its exact ExternalComponent"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(reflux_component,),
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
    reflux_component = _component(
        tmp_path, name="reflux", interface=interfaces.Reflux)
    clustering_component = _component(
        tmp_path, name="clustering", interface=interfaces.Clustering)

    with pytest.raises(TypeError, match="requires exact interface"):
        RefluxProvider(clustering_component)
    with pytest.raises(TypeError, match="requires exact interface"):
        ClusteringProvider(reflux_component)


def test_external_amr_provider_bind_fails_closed_without_native_mutation():
    from pops.amr.providers import (
        amr_provider_binding_identity,
        prepare_amr_provider_installation,
    )
    from pops.runtime._runtime_authorities import _install_amr_provider_authorities

    layout_identity = "test::layout"
    clock_identity = "test::case::clock"

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
        if slot == "reflux":
            row["clock_identity"] = clock_identity
        row["provider_identity"] = amr_provider_binding_identity(slot, row)
        return row

    providers = {
        "clustering": binding(
            "clustering", interfaces.Clustering,
            "test::clustering", "manifest::clustering"),
        "reflux": binding(
            "reflux", interfaces.Reflux, "test::reflux", "manifest::reflux"),
    }
    executable_authorities = {
        "clustering": "BergerRigoutsosProvider<Dim>",
        "reflux": "transactional AmrRuntime<Dim> reflux ledger",
    }
    for role, frozen in providers.items():
        with pytest.raises(
                NotImplementedError,
                match=r"external AMR %s component installation.*%s"
                % (role, executable_authorities[role])):
            prepare_amr_provider_installation(
                role=role,
                frozen_binding=frozen,
                layout_identity=layout_identity,
            )

    native = SimpleNamespace(calls=[])
    engine = SimpleNamespace(_s=native)
    plan = SimpleNamespace(
        amr_providers=providers,
        artifact=SimpleNamespace(
            layout_plan=SimpleNamespace(qualified_id=layout_identity)),
    )
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR clustering component installation.*"
                  r"BergerRigoutsosProvider<Dim>"):
        _install_amr_provider_authorities(engine, plan)
    assert native.calls == []
    assert not hasattr(engine, "_amr_provider_authorities")
