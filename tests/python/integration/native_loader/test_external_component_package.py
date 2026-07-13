"""ADC-680 real installed-package AOT compile, audit, install and CPU invocation."""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pops
from pops import interfaces
from pops.codegen import compile_component, pops_include
from pops.external import (
    CompiledArtifactRegistry,
    CompiledComponentArtifact,
    SourcePackageRegistry,
    build_fixed_binary_manifest,
    build_source_package_manifest,
    load,
)
from pops.model import ComponentManifest
from pops.runtime.platform_manifest import CapabilityProof


POPS_PROCESS_TIMEOUT = 300


HEADER = br'''
#pragma once
#include <pops/runtime/program/component_package.hpp>
#include <algorithm>
#include <cmath>

struct AverageFlux {
  template <class Left, class Right, class Face, class Providers>
  auto numerical_flux(const Left& left, const Right& right, const Face& face,
                      const Providers&) const -> typename Providers::normal_flux_type {
    typename Providers::normal_flux_type result{};
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = 0.5 * (left[i] + right[i]) * face.normal[0];
    return result;
  }
  template <class Left, class Right, class Face, class Providers>
  auto stability_bound(const Left& left, const Right& right, const Face&,
                       const Providers&) const -> typename Providers::normal_speed_type {
    double result = 0.0;
    for (std::size_t i = 0; i < left.values.size(); ++i)
      result = std::max(result, std::max(std::abs(left[i]), std::abs(right[i])));
    return result;
  }
};
POPS_REGISTER_COMPONENT(AverageFlux);
'''


def manifest(*, generic=True, entry_points=None, requirements=()):
    return ComponentManifest(
        uri="pops://external.test/fluxes/average", component_type="numerical_flux",
        version="1.0.0", facets=("lowering", "stability"),
        signature={"generic": generic, "state_components": 2},
        interfaces=(interfaces.NumericalFlux.manifest_ref(),),
        requirements=requirements,
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points=entry_points or {
            "header": "average.hpp", "component": "AverageFlux",
            "numerical_flux": "numerical_flux", "stability_bound": "stability_bound",
        })


def main():
    with tempfile.TemporaryDirectory(prefix="pops-adc680-") as directory:
        root = Path(directory)
        (root / "average.hpp").write_bytes(HEADER)
        source_data = build_source_package_manifest(
            components={"average": manifest()}, payloads={"average.hpp": ("header", HEADER)})
        source_path = root / "average.pops.json"
        source_path.write_text(json.dumps(source_data), encoding="utf-8")

        package = load(source_path)
        source_registry = SourcePackageRegistry()
        source_registry.register(package)
        source_registry.freeze()
        component = package.require("average", interface=interfaces.NumericalFlux)()
        artifact = compile_component(component)
        compiled_registry = CompiledArtifactRegistry()
        assert compiled_registry.register(artifact) is artifact
        assert compiled_registry.register(artifact) is artifact
        compiled_registry.freeze()
        installed = artifact.install(root / "installed")
        assert artifact.install(root / "installed").path == installed.path
        flux, speed = installed.bind(interfaces.NumericalFlux).evaluate(
            (2.0, 4.0), (6.0, 8.0), (1.0, 0.0))
        assert flux == (4.0, 6.0) and speed == 8.0
        assert installed.runtime_contract.reads == ()
        assert Path(pops.__file__).is_file()
        assert Path(pops_include()).resolve() == (Path(pops.__file__).parent / "include").resolve()

        unresolved_data = build_source_package_manifest(
            components={"average": manifest(requirements=("pressure",))},
            payloads={"average.hpp": ("header", HEADER)})
        unresolved_path = root / "unresolved.pops.json"
        unresolved_path.write_text(json.dumps(unresolved_data), encoding="utf-8")
        unresolved = load(unresolved_path).require(
            "average", interface=interfaces.NumericalFlux)()
        try:
            compile_component(unresolved)
        except Exception as exc:
            assert "requirements" in str(exc)
        else:
            raise AssertionError("component with unresolved providers was compiled")

        installed.path.write_bytes(b"competing-content")
        try:
            artifact.install(root / "installed")
        except Exception as exc:
            assert "install_collision" in str(exc)
        else:
            raise AssertionError("content-addressed installation collision was overwritten")

        # Repackage the exact linked bytes under the separate fixed ABI contract.
        fixed_manifest = manifest(generic=False, entry_points=dict(artifact.entry_symbols))
        binary_name = "average" + artifact.suffix
        (root / binary_name).write_bytes(artifact.binary)
        fixed_data = build_fixed_binary_manifest(
            components={"average": fixed_manifest}, platform=artifact.platform_manifest,
            binary_path=binary_name, binary=artifact.binary, symbols=artifact.symbols)
        fixed_path = root / "average-fixed.pops.json"
        fixed_path.write_text(json.dumps(fixed_data), encoding="utf-8")
        fixed = load(fixed_path)
        fixed_artifact = CompiledComponentArtifact.from_fixed(
            fixed, "average", interface=interfaces.NumericalFlux)
        fixed_installed = fixed_artifact.install(root / "fixed-installed")
        assert fixed_installed.bind(interfaces.NumericalFlux).evaluate(
            (1.0, 3.0), (3.0, 5.0), (1.0, 0.0))[0] == (2.0, 4.0)

        # A self-consistent package digest cannot hide a target mismatch.
        bad_platform = replace(
            artifact.platform_manifest,
            device=CapabilityProof.proven("cuda", "tampered-target-test"))
        bad_target_data = build_fixed_binary_manifest(
            components={"average": fixed_manifest}, platform=bad_platform,
            binary_path=binary_name, binary=artifact.binary, symbols=artifact.symbols)
        bad_target_path = root / "bad-target.pops.json"
        bad_target_path.write_text(json.dumps(bad_target_data), encoding="utf-8")
        try:
            load(bad_target_path)
        except Exception as exc:
            assert "target" in str(exc)
        else:
            raise AssertionError("fixed binary with mismatching target was accepted")

        # Declaring a symbol in a valid manifest does not make it exist in the image.
        missing_manifest = manifest(generic=False, entry_points={
            "numerical_flux": artifact.entry_symbols["numerical_flux"],
            "stability_bound": "pops_component_missing_symbol",
        })
        bad_symbol_data = build_fixed_binary_manifest(
            components={"average": missing_manifest}, platform=artifact.platform_manifest,
            binary_path=binary_name, binary=artifact.binary,
            symbols=(artifact.entry_symbols["numerical_flux"], "pops_component_missing_symbol"))
        bad_symbol_path = root / "bad-symbol.pops.json"
        bad_symbol_path.write_text(json.dumps(bad_symbol_data), encoding="utf-8")
        bad_symbol_package = load(bad_symbol_path)
        try:
            CompiledComponentArtifact.from_fixed(
                bad_symbol_package, "average", interface=interfaces.NumericalFlux)
        except Exception as exc:
            assert "symbols" in str(exc)
        else:
            raise AssertionError("fixed binary with a missing exported symbol was accepted")

        # Actual binary tampering is refused before a second installation can publish anything.
        (root / binary_name).write_bytes(artifact.binary + b"tamper")
        try:
            load(fixed_path)
        except Exception as exc:
            assert "binary_digest" in str(exc)
        else:
            raise AssertionError("tampered fixed binary was accepted")

    print("OK ADC-680 source AOT + fixed ABI package compiled, audited, installed and ran")


if __name__ == "__main__":
    main()
