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
from pops.amr import RefluxProvider
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
    instance_parameters=None, native_signature=None,
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
            "native_interface": (interface.signature_declaration()
                                 if native_signature is None else native_signature),
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


@pytest.mark.parametrize("prior_catalogue", [True, False])
def test_retired_clustering_manifest_cannot_select_a_native_interface(tmp_path, prior_catalogue):
    assert not hasattr(interfaces, "Clustering")
    assert not hasattr(pops.amr, "ClusteringProvider")
    with pytest.raises(KeyError, match="unknown native component interface 'clustering'"):
        interfaces.resolve("clustering")

    literal = interfaces.ComponentInterface(
        name="clustering", abi_id=4, uri="pops://interfaces/clustering", version=1,
        cpp_table="PopsClusteringApiV1", hot_path=False,
        facets=("lowering", "effects"), operations=("cluster",),
    )
    signature = literal.signature_declaration()
    if prior_catalogue:
        signature["catalog_sha256"] = (
            "3300f2e69852d0b12cbcb12b2c90c1c1dbdd9cddde375d4633062a64d0857985")
    with pytest.raises(ValueError, match=r"pops://interfaces/clustering@1.*closed catalog"):
        _component(tmp_path, name="legacy_cluster", interface=literal, native_signature=signature)


def test_external_amr_provider_accepts_a_3d_native_target_without_selecting_2d(tmp_path):
    component = _component(tmp_path, name="reflux_3d", interface=interfaces.Reflux, dimension=3)
    assert RefluxProvider(component).inspect()["component_id"] == component.component_manifest.component_id


def test_external_amr_provider_refuses_a_target_outside_ranked_dimensions(tmp_path):
    component = _component(tmp_path, name="reflux_4d", interface=interfaces.Reflux, dimension=4)
    with pytest.raises(ValueError, match="dimension 1, 2, or 3"):
        RefluxProvider(component)


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
    component = _component(tmp_path, name="reflux", interface=interfaces.Reflux)
    layout = _layout(target.layout, clustering=target.layout.clustering,
                     reflux=RefluxProvider(component))
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR reflux component installation.*transactional AmrRuntime<Dim>"):
        pops.resolve(pops.validate(target.authoring.case), layout=layout, components=(component,))


def test_third_party_external_authority_uses_open_lowering_but_fails_closed(tmp_path):
    target = _example().build_final_case()
    reflux_component = _component(
        tmp_path, name="third_party_reflux", interface=interfaces.Reflux)

    @dataclass(frozen=True, slots=True)
    class ThirdPartyRefluxAuthority:
        delegate: RefluxProvider
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
        clustering=target.layout.clustering,
        reflux=ThirdPartyRefluxAuthority(
            RefluxProvider(reflux_component)),
    )
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR reflux component installation.*"
                  r"transactional AmrRuntime<Dim> reflux ledger"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(reflux_component,),
        )


def test_incomplete_third_party_authority_is_rejected_explicitly(tmp_path):
    target = _example().build_final_case()
    reflux_component = _component(
        tmp_path, name="incomplete_reflux", interface=interfaces.Reflux)

    @dataclass(frozen=True, slots=True)
    class IncompleteRefluxAuthority:
        delegate: RefluxProvider
        __pops_ir_immutable__ = True

        def inspect(self):
            return self.delegate.inspect()

        def resolve_references(self, resolver):
            return self

    layout = _layout(
        target.layout,
        clustering=target.layout.clustering,
        reflux=IncompleteRefluxAuthority(
            RefluxProvider(reflux_component)),
    )
    with pytest.raises(TypeError, match="lower_amr_provider"):
        pops.resolve(
            pops.validate(target.authoring.case),
            layout=layout,
            components=(reflux_component,),
        )


def test_external_amr_providers_require_exact_resolve_inputs(tmp_path):
    target = _example().build_final_case()
    component = _component(tmp_path, name="reflux", interface=interfaces.Reflux)
    other = _component(tmp_path, name="other_reflux", interface=interfaces.Reflux)
    layout = _layout(target.layout, clustering=target.layout.clustering,
                     reflux=RefluxProvider(component))
    with pytest.raises(ValueError, match="requires its exact ExternalComponent"):
        pops.resolve(pops.validate(target.authoring.case), layout=layout, components=(other,))


def test_external_reflux_options_remain_inspectable_but_installation_fails_closed(tmp_path):
    from pops.amr import ResolvedAMRProviderBinding, amr_provider_binding_identity
    from pops.amr.providers import prepare_amr_provider_installation

    component = _component(
        tmp_path,
        name="parameterized_reflux",
        interface=interfaces.Reflux,
        manifest_parameters=({"name": "options", "kind": "runtime"},),
        instance_parameters={"options": {"strict": True}},
    )
    binding = RefluxProvider(component).inspect()
    assert binding["component"]["parameters"] == {"options": {"strict": True}}
    binding["layout_identity"] = "test::layout"
    binding["clock_identity"] = "test::clock"
    binding["provider_identity"] = amr_provider_binding_identity("reflux", binding)
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR reflux component installation.*transactional AmrRuntime<Dim>"):
        prepare_amr_provider_installation(
            role="reflux", frozen_binding=binding, layout_identity="test::layout")

    forged = dict(binding, component=dict(binding["component"], parameters={"options": {"strict": False}}))
    with pytest.raises(ValueError, match="provider_identity"):
        ResolvedAMRProviderBinding("reflux", forged)


def test_external_amr_provider_roles_are_not_interchangeable(tmp_path):
    component = _component(tmp_path, name="transfer", interface=interfaces.Transfer)
    with pytest.raises(TypeError, match="requires exact interface"):
        RefluxProvider(component)

    target = _example().build_final_case()
    reflux = _component(tmp_path, name="reflux", interface=interfaces.Reflux)
    layout = _layout(target.layout, clustering=RefluxProvider(reflux))
    with pytest.raises(TypeError, match="built-in BergerRigoutsos policy"):
        pops.resolve(pops.validate(target.authoring.case), layout=layout, components=(reflux,))


def test_external_amr_provider_bind_fails_closed_without_native_mutation(tmp_path):
    from pops.amr import amr_provider_binding_identity
    from pops.amr.providers import resolve_clustering_binding
    from pops.runtime._runtime_authorities import _install_amr_provider_authorities

    layout_identity = "test::layout"
    target = _example().build_final_case()
    component = _component(tmp_path, name="reflux", interface=interfaces.Reflux)
    reflux = dict(RefluxProvider(component).inspect(), layout_identity=layout_identity,
                  clock_identity="test::clock")
    reflux["provider_identity"] = amr_provider_binding_identity("reflux", reflux)
    clustering = resolve_clustering_binding(target.layout.clustering, layout_identity)
    plan = SimpleNamespace(
        amr_providers={"clustering": clustering, "reflux": reflux},
        artifact=SimpleNamespace(layout_plan=SimpleNamespace(qualified_id=layout_identity)),
    )
    engine = SimpleNamespace(_s=SimpleNamespace(calls=[]))
    with pytest.raises(
            NotImplementedError,
            match=r"external AMR reflux component installation.*transactional AmrRuntime<Dim>"):
        _install_amr_provider_authorities(engine, plan)
    assert engine._s.calls == []
    assert not hasattr(engine, "_amr_provider_authorities")

    # A detached old clustering row is refused before a provider report is published.
    plan.amr_providers["clustering"] = dict(clustering, native_interface={
        "id": 4, "uri": "pops://interfaces/clustering", "version": 1,
    })
    with pytest.raises(ValueError, match="exactly the built-in policy fields"):
        _install_amr_provider_authorities(engine, plan)
    assert engine._s.calls == []
    assert not hasattr(engine, "_amr_provider_authorities")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"component_id": "external"}, "exactly the built-in policy fields"),
        ({"extra": 1}, "exactly the built-in policy fields"),
        ({"layout_identity": "other"}, "built-in policy of this layout"),
        ({"runtime_installation": {"schema_version": 1, "protocol": "external_component"}},
         "built-in policy of this layout"),
        ({"minimum_efficiency": True}, "canonical binary64"),
        ({"minimum_efficiency": {"binary64": "0x1.0p-1"}}, "canonical binary64"),
        ({"minimum_efficiency": float("inf")}, r"must be in \(0, 1\]"),
        ({"minimum_box_size": True}, "box-size controls are invalid"),
        ({"minimum_box_size": 20, "maximum_box_size": 12}, "box-size controls are invalid"),
        ({"maximum_box_size": 13}, "provider_identity does not authenticate"),
    ],
)
def test_intrinsic_clustering_refuses_unsupported_or_forged_data(changes, message):
    from pops.amr.providers import clustering_controls, resolve_clustering_binding
    from pops.lib.amr import BergerRigoutsos

    layout_identity = "test::layout"
    policy = BergerRigoutsos(minimum_efficiency=0.91, minimum_box_size=2, maximum_box_size=12)
    binding = resolve_clustering_binding(policy, layout_identity)
    assert clustering_controls(binding, layout_identity=layout_identity) == (0.91, 2, 12)
    canonical = dict(binding, minimum_efficiency={"binary64": float(0.91).hex()})
    assert clustering_controls(canonical, layout_identity=layout_identity) == (0.91, 2, 12)
    with pytest.raises((TypeError, ValueError), match=message):
        clustering_controls({**binding, **changes}, layout_identity=layout_identity)
