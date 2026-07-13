"""Fail-closed native-report and absolute-memory-estimate contracts."""
from types import SimpleNamespace

import pytest

from pops import _capabilities_report as capability_reports
from pops.codegen import inspect_compiled
from pops.codegen import toolchain
from pops.runtime import defaults


def test_capability_report_is_explicitly_source_only_only_without_extension(monkeypatch):
    monkeypatch.setattr(capability_reports, "_native_extension", lambda: None)
    report = capability_reports.native_capability_report()
    assert report.routes
    assert {row.source for row in report.routes} == {"source-only"}
    assert any(row.status == "unknown" for row in report.routes)


def test_capability_report_does_not_hide_native_call_failure(monkeypatch):
    class BrokenExtension:
        @staticmethod
        def capability_report(_target):
            raise RuntimeError("native boom")

    monkeypatch.setattr(capability_reports, "_native_extension", lambda: BrokenExtension())
    with pytest.raises(capability_reports.NativeCapabilityReportError,
                       match="capability_report") as excinfo:
        capability_reports._native_capability_report_from_extension()
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_defaults_source_only_is_not_used_for_a_loaded_broken_extension(monkeypatch):
    monkeypatch.setattr(defaults, "_native_extension", lambda: None)
    assert defaults.numerical_defaults_report()["source"] == "source-only"

    class BrokenExtension:
        @staticmethod
        def numerical_defaults_report():
            return object()

    monkeypatch.setattr(defaults, "_native_extension", lambda: BrokenExtension())
    with pytest.raises(defaults.NativeDefaultsReportError, match="malformed"):
        defaults.numerical_defaults_report()


def test_toolchain_does_not_treat_a_broken_extension_as_absent(monkeypatch):
    def broken_import(name):
        if name == "_pops":
            raise ImportError("missing dependent dylib")
        raise AssertionError("relative import must not be attempted after a broken top-level extension")

    monkeypatch.setattr(toolchain.importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="dependent dylib"):
        toolchain._pops_module()


def test_absolute_memory_estimate_refuses_unknown_native_precision(monkeypatch):
    def absent_extension(name):
        raise ModuleNotFoundError("absent", name=name)

    monkeypatch.setattr(inspect_compiled.importlib, "import_module", absent_extension)
    with pytest.raises(inspect_compiled.MemoryEstimateCapabilityError,
                       match="source-only") as excinfo:
        inspect_compiled.build_memory_estimate(SimpleNamespace(), SimpleNamespace())
    assert excinfo.value.field == "runtime.precision"


def test_absolute_memory_estimate_refuses_untyped_shape_before_any_formula(monkeypatch):
    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16, "amr_refinement_ratio": 3}

    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    with pytest.raises(inspect_compiled.MemoryEstimateCapabilityError,
                       match="typed CartesianMesh") as excinfo:
        inspect_compiled.build_memory_estimate(SimpleNamespace(), 32)
    assert excinfo.value.field == "mesh"


def test_absolute_memory_estimate_uses_reported_native_byte_width(monkeypatch):
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16, "amr_refinement_ratio": 2}

    class Program:
        @staticmethod
        def estimate():
            return {"buffer_count": 0, "heavy_kernels": 0}

    mesh = CartesianMesh(n=4)
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U"))
    estimate = inspect_compiled.build_memory_estimate(
        SimpleNamespace(program=Program(), install_plan=None), mesh, layout=Uniform(mesh))
    assert estimate.categories["state"] == 2 * 4 * 4 * 16
    assert "16 bytes per cell value" in estimate.assumptions[0]
