"""ADC-680 strict source/fixed package and phase-registry contracts."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from pops import interfaces
from pops.external import (
    CompiledArtifactRegistry,
    ComponentPackageError,
    SourcePackageRegistry,
    build_fixed_binary_manifest,
    build_source_package_manifest,
    load,
)
from pops.model import ComponentManifest
from pops.runtime.platform_manifest import proven_serial_manifest


def _manifest(*, uri="pops://external.test/fluxes/average", generic=True, entry_points=None):
    return ComponentManifest(
        uri=uri, component_type="numerical_flux", version="1.0.0",
        facets=("lowering", "stability"),
        signature={"generic": generic, "state_components": 2,
                   "inputs": ["left", "right", "face", "providers"]},
        interfaces=interfaces.NumericalFlux.manifest_declarations(),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points=entry_points or {
            "header": "average.hpp", "component": "AverageFlux",
            "numerical_flux": "numerical_flux", "stability_bound": "stability_bound",
        },
    )


def _write_source(tmp_path, *, manifest=None, header=b"struct AverageFlux {};\n"):
    component = manifest or _manifest()
    (tmp_path / "average.hpp").write_bytes(header)
    data = build_source_package_manifest(
        components={"average": component}, payloads={"average.hpp": ("header", header)})
    path = tmp_path / "average.pops.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def test_source_package_verifies_content_before_authoring_registry(tmp_path):
    path, _ = _write_source(tmp_path)
    package = load(path)
    factory = package.require("average", interface=interfaces.NumericalFlux)
    component = factory()
    assert component.package_identity == package.identity
    assert component.to_data()["component_id"] == _manifest().component_id

    (tmp_path / "average.hpp").write_text("tampered", encoding="utf-8")
    with pytest.raises(ComponentPackageError) as error:
        load(path)
    assert error.value.code == "source_digest"


def test_tampered_manifest_digest_is_rejected(tmp_path):
    path, data = _write_source(tmp_path)
    changed = deepcopy(data)
    changed["exports"] = {"other": _manifest().component_id}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ComponentPackageError) as error:
        load(path)
    assert error.value.code == "package_digest"


def test_source_registry_is_atomic_idempotent_collision_safe_and_frozen(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = load(_write_source(first_dir, header=b"// first\n")[0])
    same = load(_write_source(first_dir, header=b"// first\n")[0])
    other = load(_write_source(second_dir, header=b"// other\n")[0])
    registry = SourcePackageRegistry()
    assert registry.register(first) is first
    assert registry.register(same) is first
    assert registry.revision == 1
    with pytest.raises(ValueError, match="identity collision"):
        registry.register(other)
    assert registry.revision == 1 and registry.resolve(_manifest().component_id) is first
    registry.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(first)


def test_fixed_binary_cannot_claim_template_genericity():
    platform = proven_serial_manifest(
        backend="aot-component", target="component", abi="headers|clang|c++20")
    with pytest.raises(ComponentPackageError) as error:
        build_fixed_binary_manifest(
            components={"average": _manifest(generic=True)}, platform=platform,
            binary_path="average.so", binary=b"not-a-binary",
            symbols=("numerical_flux", "stability_bound"))
    assert error.value.code == "fixed_generic_claim"


def test_compiled_registry_refuses_source_values_and_freezes():
    registry = CompiledArtifactRegistry()
    with pytest.raises(TypeError, match="CompiledComponentArtifact"):
        registry.register(object())
    registry.freeze()
    assert registry.frozen


def test_external_authoring_surface_has_no_native_escape_hatch(tmp_path):
    package = load(_write_source(tmp_path)[0])
    component = package.require("average", interface=interfaces.NumericalFlux)()
    assert not hasattr(component, "native_id")
    assert "_native" not in component.to_data()
