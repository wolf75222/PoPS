"""ADC-681: a new native interface plugs in without editing central dispatch."""
from __future__ import annotations

import ctypes
import json
import tempfile
from pathlib import Path

from pops.codegen.component_packages import compile_component
from pops.external import build_source_package_manifest, load
from pops.interfaces import ComponentInterface
from pops.mesh.cartesian import CartesianMesh
from pops.mesh.layouts import AMR, Uniform
from pops.model import ComponentManifest


class ProbeBinding:
    __slots__ = ("_artifact", "_handle", "_probe")

    def __init__(self, installed):
        installed.verify()
        handle = ctypes.CDLL(str(installed.path))
        probe = getattr(handle, installed.entry_symbols["probe"])
        probe.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        probe.restype = ctypes.c_int
        self._artifact = installed
        self._handle = handle
        self._probe = probe

    def evaluate(self, layout, value: float) -> float:
        capabilities = layout.capabilities()
        layout_name = capabilities.get("layout")
        if layout_name not in self._artifact.runtime_contract.layouts:
            raise ValueError("external probe does not support layout %r" % layout_name)
        output = ctypes.c_double()
        status = self._probe(
            int(bool(capabilities.get("supports_amr"))), value, ctypes.byref(output))
        if status:
            raise RuntimeError("external probe returned status %d" % status)
        return output.value


class ProbeNativeBackend:
    """Test-owned ABI provider: no PoPS compiler/installer branch knows this interface."""

    def resolve_target(self, component):
        return {"dimension": 2, "scalar": "float64", "device": "cpu", "features": []}

    def wrapper_source(self, component, symbols):
        return '''
extern "C" int %s(int adaptive, double input, double* output) {
  if (!output) return 2;
  *output = (adaptive ? 5.0 : 3.0) * input;
  return 0;
}
''' % symbols["probe"]

    def bind_installed(self, component):
        return ProbeBinding(component)


Probe = ComponentInterface(
    "pops://interfaces/test-probe",
    1,
    (("fallible_evaluation", "probe"),),
    ("header", "component", "probe"),
    ("probe",),
    ProbeNativeBackend(),
)


def main():
    manifest = ComponentManifest(
        uri="pops://external.test/components/probe",
        component_type="test_probe",
        version="1.0.0",
        facets=Probe.facets,
        signature={"generic": True},
        interfaces=Probe.manifest_declarations(),
        layouts=("uniform", "amr"),
        target={"variants": [{
            "dimension": 2, "scalar": "float64", "device": "cpu", "features": [],
        }]},
        entry_points={"header": "probe.hpp", "component": "Probe", "probe": "probe"},
    )
    with tempfile.TemporaryDirectory(prefix="pops-interface-probe-") as directory:
        root = Path(directory)
        payload = b"#pragma once\nstruct Probe {};\n"
        (root / "probe.hpp").write_bytes(payload)
        package_data = build_source_package_manifest(
            components={"probe": manifest}, payloads={"probe.hpp": ("header", payload)})
        package_path = root / "probe.pops.json"
        package_path.write_text(json.dumps(package_data), encoding="utf-8")
        component = load(package_path).require("probe", interface=Probe)()
        artifact = compile_component(component)
        installed = artifact.install(root / "installed")
        binding = installed.bind(Probe)
        mesh = CartesianMesh(n=4, periodic=True)
        assert binding.evaluate(Uniform(mesh), 4.0) == 12.0
        assert binding.evaluate(AMR(base=mesh), 4.0) == 20.0
        assert installed.runtime_contract.layouts == ("amr", "uniform")

    print("OK ADC-681/687 external C++ interface executed through Uniform and AMR")


if __name__ == "__main__":
    main()
