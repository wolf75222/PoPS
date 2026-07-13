"""ADC-681: a new native interface plugs in without editing central dispatch."""
from __future__ import annotations

import ctypes
import json
import tempfile
from pathlib import Path

from pops.codegen import compile_component
from pops.external import build_source_package_manifest, load
from pops.interfaces import ComponentInterface
from pops.model import ComponentManifest


class ProbeBinding:
    __slots__ = ("_handle", "_probe")

    def __init__(self, installed):
        installed.verify()
        handle = ctypes.CDLL(str(installed.path))
        probe = getattr(handle, installed.entry_symbols["probe"])
        probe.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        probe.restype = ctypes.c_int
        self._handle = handle
        self._probe = probe

    def evaluate(self, value: float) -> float:
        output = ctypes.c_double()
        status = self._probe(value, ctypes.byref(output))
        if status:
            raise RuntimeError("external probe returned status %d" % status)
        return output.value


class ProbeNativeBackend:
    """Test-owned ABI provider: no PoPS compiler/installer branch knows this interface."""

    def resolve_target(self, component):
        return {"dimension": 2, "scalar": "float64", "device": "cpu", "features": []}

    def wrapper_source(self, component, symbols):
        return '''
extern "C" int %s(double input, double* output) {
  if (!output) return 2;
  *output = 3.0 * input;
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
        assert installed.bind(Probe).evaluate(4.0) == 12.0

    print("OK ADC-681 external native interface backend compiled, installed and executed")


if __name__ == "__main__":
    main()
