"""Fail-closed native-report and absolute-memory-estimate contracts."""
from types import SimpleNamespace

import pytest

from pops import _capabilities_report as capability_reports
from pops.codegen import inspect_compiled
from pops.codegen import toolchain
from pops.runtime import defaults


def _memory_artifact(*, program=None):
    """Exact minimum resolved-plan evidence required by the strict memory estimator."""
    block = SimpleNamespace(name="test", numerics=None, spatial={"ghost_depth": 2})
    return SimpleNamespace(
        program=program,
        install_plan=None,
        plan=SimpleNamespace(blocks=(block,)),
    )


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


@pytest.mark.parametrize(
    ("supports_mpi", "expected"),
    ((False, "unavailable"), (True, "available")),
)
def test_mpi_world_route_reports_only_proved_native_availability(supports_mpi, expected):
    report = capability_reports.native_capability_report(
        flags={"supports_mpi": supports_mpi, "supports_amr": True}, source="test-manifest")
    routes = {row.feature: row for row in report.routes}
    route = routes["parallel:mpi_world_communicator"]
    assert route.status == expected
    assert route.mpi is supports_mpi
    assert route.available_route == (
        "ExecutionContext.mpi_world()" if supports_mpi else "serial ExecutionContext"
    )
    assert bool(route.alternative) is (not supports_mpi)
    assert "ParallelContext" not in routes["parallel:custom_communicator"].alternative
    assert "PrecisionPolicy is representable" in routes["precision:single_or_mixed"].limitation
    assert routes["checkpoint:accepted_state_v3"].status == "available"
    assert routes["checkpoint:accepted_state_v3"].layout == "uniform|amr"
    assert routes["checkpoint:amr_dynamic_regrid"].status == "available"
    assert "checkpoint:system_v1" not in routes
    weno = routes["reconstruction:weno5"]
    assert weno.layout == "uniform"
    assert "order-5 coarse/fine provider" in weno.limitation


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
                       match="CartesianGrid") as excinfo:
        inspect_compiled.build_memory_estimate(SimpleNamespace(), 32)
    assert excinfo.value.field == "mesh"


def test_absolute_memory_estimate_uses_reported_native_byte_width(monkeypatch):
    from pops.layouts import Uniform
    from tests.python.support.layout_plan import cartesian_grid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16, "amr_refinement_ratio": 2}

    class Program:
        @staticmethod
        def estimate():
            return {"buffer_count": 0, "heavy_kernels": 0}

    mesh = cartesian_grid(n=4)
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U"))
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(program=Program()), mesh, layout=Uniform(mesh))
    assert estimate.categories["state"] == 2 * 4 * 4 * 16
    assert "16 bytes per cell value" in estimate.assumptions[0]


def test_absolute_memory_estimate_accepts_final_cartesian_grid_cells(monkeypatch):
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid
    from pops.layouts import Uniform

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16, "amr_refinement_ratio": 2}

    frame = Rectangle("estimate-grid", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    grid = CartesianGrid(frame=frame, cells=(3, 5))
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U"))
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(), grid, layout=Uniform(grid))
    assert estimate.mesh_shape == (3, 5)
    assert estimate.cells == 15
    assert estimate.categories["state"] == 2 * 3 * 5 * 16


def test_absolute_memory_estimate_accepts_strict_final_amr_protocol(monkeypatch):
    from pops.descriptors_report import CapabilitySet
    from tests.python.support.layout_plan import cartesian_grid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16, "amr_refinement_ratio": 2}

    class FinalAMRProtocol:
        """The public final-AMR capability shape, without relying on a legacy layout class."""

        @staticmethod
        def capabilities():
            return CapabilitySet({
                "layout": "amr",
                "dim": 2,
                "max_levels": 3,
                "ratio": 2,
                "transition_ratios": [2, 2],
                "supports_amr": True,
            })

    mesh = cartesian_grid(n=4)
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U"))
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(), mesh, layout=FinalAMRProtocol())
    assert estimate.layout == "amr"
    assert estimate.categories["amr_patch"] == (2 ** 2 + 2 ** 4) * (2 * 4 * 4 * 16)


def test_absolute_memory_estimate_refuses_amr_without_transition_ratios(monkeypatch):
    from pops.descriptors_report import CapabilitySet
    from tests.python.support.layout_plan import cartesian_grid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16, "amr_refinement_ratio": 2}

    class IncompleteAMR:
        @staticmethod
        def capabilities():
            return CapabilitySet({"layout": "amr", "dim": 2, "max_levels": 2, "ratio": 2})

    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 1, {}, (), 0, "U"))
    with pytest.raises(inspect_compiled.MemoryEstimateCapabilityError,
                       match="transition_ratios") as excinfo:
        inspect_compiled.build_memory_estimate(
            _memory_artifact(), cartesian_grid(n=4),
            layout=IncompleteAMR())
    assert excinfo.value.field == "layout.transition_ratios"
