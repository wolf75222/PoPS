"""Fail-closed native-report and absolute-memory-estimate contracts."""

from types import SimpleNamespace

import pytest

from pops import _capabilities_report as capability_reports
from pops import _native_selector
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
    with pytest.raises(
        capability_reports.NativeCapabilityReportError, match="capability_report"
    ) as excinfo:
        capability_reports._native_capability_report_from_extension()
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_amr_transfer_report_iterates_every_canonical_oriented_face_centering():
    from pops.mesh._amr._transfer_contracts import (
        CELL_CENTERED,
        NODE_CENTERED,
        ORIENTED_FACE_CENTERINGS,
    )

    report = capability_reports.native_capability_report(
        flags={"supports_mpi": False, "supports_gpu": False, "supports_amr": True},
        source="test-manifest",
    )
    transfer = {row.feature: row for row in report.routes}["amr:transfer_contracts"]
    expected = "/".join(
        (
            CELL_CENTERED.name,
            *(centering.name for centering in ORIENTED_FACE_CENTERINGS),
            NODE_CENTERED.name,
        )
    )
    assert "exact dense %s contracts" % expected in transfer.limitation
    assert "face_z" in transfer.limitation


def test_amr_transition_envelope_reports_exact_rank_isotropic_defaults():
    report = capability_reports.native_capability_report(
        flags={"supports_mpi": False, "supports_gpu": False, "supports_amr": True},
        source="test-manifest",
    )
    envelope = {row.feature: row for row in report.routes}["amr:transition_envelope"]
    assert "(2,2,2)" in envelope.limitation
    assert "anisotropic Dim-length ratios" in envelope.limitation
    assert "2D isotropic ratio-(2,2)" not in envelope.limitation


@pytest.mark.parametrize(
    ("supports_mpi", "expected"),
    ((False, "unavailable"), (True, "available")),
)
def test_mpi_world_route_reports_only_proved_native_availability(supports_mpi, expected):
    report = capability_reports.native_capability_report(
        flags={"supports_mpi": supports_mpi, "supports_amr": True}, source="test-manifest"
    )
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
    assert routes["checkpoint:uniform_accepted_state_v5"].status == "available"
    assert routes["checkpoint:uniform_accepted_state_v5"].layout == "uniform"
    assert routes["checkpoint:amr_accepted_state_v7"].status == "available"
    assert routes["checkpoint:amr_accepted_state_v7"].layout == "amr"
    assert routes["checkpoint:amr_dynamic_regrid"].status == "available"
    assert "checkpoint:system_v1" not in routes
    weno = routes["reconstruction:weno5"]
    assert weno.layout == "uniform|amr"
    assert "hierarchy-selected AMR in the compile-selected native rank" in weno.limitation
    assert "order-5" in weno.limitation
    for feature in ("limiter:mc", "limiter:superbee"):
        limiter = routes[feature]
        assert limiter.status == "available"
        assert limiter.backend == "production"
        assert limiter.layout == "uniform|amr"
        assert "formal_order=2" in limiter.limitation
        assert "ghost_depth=2" in limiter.limitation
        assert limiter.available_route == ""
        assert limiter.alternative == ""
    amr_implicit = routes["amr:source_implicit_program"]
    assert amr_implicit.status == "partial"
    assert amr_implicit.backend == "production"
    assert amr_implicit.mpi is supports_mpi
    assert amr_implicit.gpu is False
    assert "prepared LocalNewton" in amr_implicit.limitation
    assert "SolveOutcome/FailRun rollback is exact" in amr_implicit.limitation
    assert "subcycled local solves" in amr_implicit.limitation
    assert amr_implicit.layout == "amr"
    assert amr_implicit.available_route == (
        "generated Program local implicit source solve with LocalNewton and a consumed "
        "SolveOutcome on synchronous two-level 2D AMR"
    )
    cell_local = routes["amr:cell_local_temporal_transport"]
    assert cell_local.status == "partial"
    assert cell_local.layout == "amr"
    assert cell_local.backend == "production"
    assert cell_local.mpi is supports_mpi
    assert cell_local.gpu is False
    assert "multi-block, multi-level and distributed multi-box" in cell_local.limitation
    assert "authoritative reflux" in cell_local.limitation
    assert "Program.cell_local_time" in cell_local.limitation
    assert "integral power-of-two temporal ratios" in cell_local.limitation
    assert "one homogeneous rung per level-group" in cell_local.limitation
    assert "Heterogeneous rungs" in cell_local.limitation
    assert "Same-rank restart and regrid" in cell_local.limitation
    assert "global/interface-coupled blocks" in cell_local.limitation
    assert "physical/non-periodic boundaries" in cell_local.limitation
    assert "every GPU default execution or memory space" in cell_local.limitation
    assert "fail collectively during exact-lane preparation" in cell_local.limitation
    external_amr = routes["amr:external_field_solver_v2"]
    assert external_amr.status == "available"
    assert external_amr.layout == "amr"
    assert external_amr.mpi is supports_mpi
    assert external_amr.gpu is False
    assert "hierarchy-selected AMR ratios" in external_amr.limitation
    assert "both components to declare MPI_COMM_WORLD" in external_amr.limitation
    assert "distributed coarse level" in external_amr.limitation
    assert external_amr.available_route == (
        "authenticated FieldTopology@2 + FieldSolver@2 composite hierarchy batch with "
        "metadata.level, binary coarse/fine coverage, one collective solve, exact "
        "materialization/report consensus and transactional candidate publication"
    )
    implicit_pair = routes["amr:shared_interface_implicit_jacvec_pair"]
    assert implicit_pair.status == "partial"
    assert implicit_pair.layout == "amr"
    assert implicit_pair.backend == "production"
    assert implicit_pair.mpi is supports_mpi
    assert implicit_pair.gpu is False
    assert "compiles, binds and runs GMRES" in implicit_pair.limitation
    assert "independent packed-vector carrier block" in implicit_pair.limitation
    assert "dynamic hierarchy mutation" in implicit_pair.limitation
    assert "ExecutionLane communicator" in implicit_pair.limitation
    assert "DefaultExecutionSpace" in implicit_pair.limitation
    assert implicit_pair.available_route == (
        "generated GMRES solve with an authenticated two-sided shared-interface "
        "JVP on a frozen two-level Cartesian Dim-in-{1,2,3} AMR hierarchy via "
        "level_rhs_jacvec_pair and the prepared ExecutionLane communicator"
    )
    assert "mixed apply operators" in implicit_pair.alternative
    field_coupled = routes["amr:field_coupled_rhs_jacvec"]
    assert field_coupled.status == "available"
    assert field_coupled.layout == "amr"
    assert field_coupled.backend == "production"
    assert field_coupled.mpi is False
    assert field_coupled.gpu is False
    assert "evaluate_with_field_state_at" in field_coupled.limitation
    assert "point.level == active_level" in field_coupled.limitation
    assert "cannot forge a coarse point" in field_coupled.limitation
    assert field_coupled.available_route == (
        "field_coupled rhs_jacvec on every AMR level via the level-qualified "
        "tangent-field provider"
    )


def test_transport_boundary_routes_report_exact_supported_envelope_and_missing_kernels():
    report = capability_reports.native_capability_report(
        flags={"supports_mpi": True, "supports_gpu": False, "supports_amr": True},
        source="test-manifest",
    )
    routes = {row.feature: row for row in report.routes}

    prepared = routes["boundary:prepared_transport"]
    assert prepared.status == "partial"
    assert prepared.layout == "uniform|amr"
    assert prepared.backend == "production"
    assert prepared.mpi is True
    assert prepared.gpu is False
    assert "one prepared Cartesian Dim-in-{1,2,3} model-aware plan" in prepared.limitation
    assert "Geometry<Dim>::cell_center" in prepared.limitation
    assert "origin/spacing" in prepared.limitation
    assert "typed-role slip wall" in prepared.limitation
    assert "typed no-flux faces" in prepared.limitation
    assert "before divergence/reflux" in prepared.limitation
    assert "model primitive-to-conservative" in prepared.limitation
    assert "coarse-fine ghosts under the prepared transfer authority" in prepared.limitation
    assert "corners explicitly not required" in prepared.limitation

    conversion = routes["boundary:representation_conversion"]
    assert conversion.status == "partial"
    assert conversion.layout == "uniform|amr"
    assert conversion.backend == "production"
    assert "Cartesian Dim-in-{1,2,3}" in conversion.limitation
    assert "to_conservative provider" in conversion.limitation
    assert "recovery" in conversion.limitation

    analytic = routes["boundary:analytic_xtp"]
    assert analytic.status == "partial"
    assert analytic.layout == "uniform|amr"
    assert analytic.backend == "production"
    assert "analytic ScalarExpr" in analytic.limitation
    assert "exact logical Clock" in analytic.limitation
    assert "Input slots 1..ncomp" in analytic.limitation
    assert "to_conservative" in analytic.limitation
    assert "axis-permuted periodic coordinates" in analytic.limitation
    assert "Geometry<Dim>::cell_center" in analytic.limitation
    assert "x/y/z coordinates" in analytic.limitation
    assert "origin/spacing" in analytic.limitation
    assert "refuses rather than inventing zeros" in analytic.limitation
    assert "setup-program discrete inputs" in analytic.limitation

    characteristic = routes["boundary:characteristic_no_inflow"]
    assert characteristic.status == "partial"
    assert characteristic.layout == "uniform|amr"
    assert characteristic.backend == "production"
    assert characteristic.mpi is True
    assert characteristic.gpu is False
    assert "Cartesian Dim-in-{1,2,3}" in characteristic.limitation
    assert "m.roe_from_jacobian()" in characteristic.limitation
    assert "sonic subspace as neutral" in characteristic.limitation
    assert "rolls back ghosts" in characteristic.limitation
    assert "ExecutionLane communicator" in characteristic.limitation
    assert "same kernel" in characteristic.limitation
    assert "DefaultExecutionSpace" in characteristic.limitation
    assert "Cartesian cut-cell and staircase no-inflow" in characteristic.limitation
    assert "EB interface normal" in characteristic.limitation
    assert "polar geometry remain unavailable" in characteristic.limitation
    assert "polar/embedded geometry remain unavailable" not in characteristic.limitation
    assert "PreparedMetricProvider metric ABI" not in characteristic.limitation
    assert "primitive/analytic references" in characteristic.limitation
    assert "qualified MPI/GPU execution remain unavailable" not in characteristic.limitation
    post_riemann = routes["boundary:post_riemann_flux"]
    assert post_riemann.status == "partial"
    assert post_riemann.layout == "uniform|amr"
    assert post_riemann.backend == "production"
    assert "outward-normal face flux" in post_riemann.limitation
    assert "Riemann solve and divergence/reflux" in post_riemann.limitation
    assert "Cartesian Dim-in-{1,2,3} host-batch" in post_riemann.limitation
    gpu_report = capability_reports.native_capability_report(
        flags={"supports_mpi": True, "supports_gpu": True, "supports_amr": True},
        source="test-gpu-manifest",
    )
    gpu_routes = {row.feature: row for row in gpu_report.routes}
    gpu_post_riemann = gpu_routes["boundary:post_riemann_flux"]
    assert gpu_post_riemann.gpu is False
    gpu_characteristic = gpu_routes["boundary:characteristic_no_inflow"]
    assert gpu_characteristic.mpi is True
    assert gpu_characteristic.gpu is True
    gpu_jacvec = gpu_routes["amr:shared_interface_implicit_jacvec_pair"]
    assert gpu_jacvec.mpi is True
    assert gpu_jacvec.gpu is True


def test_riemann_recovery_routes_distinguish_typed_rejection_from_missing_policy():
    report = capability_reports.native_capability_report(
        flags={"supports_mpi": True, "supports_gpu": False, "supports_amr": True},
        source="test-manifest",
    )
    routes = {row.feature: row for row in report.routes}

    typed = routes["riemann:typed_failure_outcome"]
    assert typed.status == "partial"
    assert typed.layout == "uniform|amr"
    assert typed.backend == "production"
    assert typed.mpi is True
    assert typed.gpu is False
    assert "one device-copyable FluxEvaluation" in typed.limitation
    assert "requested/used/last solver identity" in typed.limitation
    assert "single-solver routes remain explicit" in typed.limitation
    assert "fallback counters and restart" in typed.limitation

    policy = routes["riemann:prepared_recovery_policy"]
    assert policy.status == "partial"
    assert policy.layout == "uniform|amr"
    assert policy.backend == "production"
    assert policy.mpi is False
    assert policy.gpu is False
    assert "typed public riemann.Recovery descriptor" in policy.limitation
    assert "Uniform and AMR Cartesian face kernels" in policy.limitation
    assert "only typed candidate rejection" in policy.limitation
    assert "polar geometry is refused" in policy.limitation
    assert "GPU qualification" in policy.limitation
    assert "riemann.Recovery(primary=Roe()" in policy.available_route
    assert "consume rejection through the step retry/failure policy" in policy.alternative


def test_variable_recovery_routes_separate_delivered_consumers_from_complete_cutover():
    report = capability_reports.native_capability_report(
        flags={"supports_mpi": True, "supports_gpu": False, "supports_amr": True},
        source="test-manifest",
    )
    routes = {row.feature: row for row in report.routes}

    prepared = routes["recovery:prepared_variable"]
    assert prepared.status == "partial"
    assert prepared.layout == "uniform|amr"
    assert prepared.backend == "production"
    assert prepared.mpi is True
    assert prepared.gpu is False
    assert "one block-prepared closed-form method" in prepared.limitation
    assert "device-copyable RecoveryOutcome/RecoveryReport" in prepared.limitation
    assert "selected and last-attempted method kinds" in prepared.limitation
    assert "consume publication permission" in prepared.limitation
    assert "transactional analytic initial-state materialization" in prepared.limitation
    assert "primitive-to-conservative setup conversion" in prepared.limitation
    assert "AMR regrid prolongation and restriction" in prepared.limitation
    assert "AMR bootstrap commits" in prepared.limitation
    assert "rematerialized history slots" in prepared.limitation
    assert "physical boundary traces" in prepared.limitation
    assert "generated Program terminal commits" in prepared.limitation
    assert "model-local and coupled sources" in prepared.limitation
    assert "no implicit repair or fallback" in prepared.limitation
    assert "generation-qualified warm-start slot per local cell" in prepared.limitation
    assert "invalidates every slot after a refused batch" in prepared.limitation

    cutover = routes["recovery:complete_consumer_cutover"]
    assert cutover.status == "unavailable"
    assert cutover.layout == "uniform|amr"
    assert cutover.backend == "none"
    assert "manual in-place Program writes" in cutover.limitation
    assert "initial and analytic materialization" not in cutover.limitation
    assert "fallible primitive-to-conservative conversion" not in cutover.limitation
    assert "AMR bootstrap/history transfer" not in cutover.limitation
    assert "primitive boundary traces" not in cutover.limitation
    assert (
        "persistent warm starts outside the host Uniform diagnostic materializer"
        in cutover.limitation
    )
    assert "transactional analytic initial-state materialization" in cutover.available_route
    assert "spatial face reconstruction" in cutover.available_route
    assert "fallible primitive-to-conservative setup conversion" in cutover.available_route
    assert "transactional AMR regrid prolongation/restriction" in cutover.available_route
    assert "bootstrap/history" in cutover.available_route
    assert "physical boundary-trace publication" in cutover.available_route
    assert "model-local and coupled-source endpoints" in cutover.available_route
    assert "generation-qualified warm starts" in cutover.available_route
    assert "missing in-place-write, AMR/spatial warm-start" in cutover.alternative
    assert cutover.error_message


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
    def broken_selection(*, required=False):
        assert required is False
        raise ImportError("missing dependent dylib")

    monkeypatch.setattr(_native_selector, "selected_native_module", broken_selection)
    with pytest.raises(ImportError, match="dependent dylib"):
        toolchain._pops_module()


def test_absolute_memory_estimate_refuses_unknown_native_precision(monkeypatch):
    def absent_extension(name):
        raise ModuleNotFoundError("absent", name=name)

    monkeypatch.setattr(inspect_compiled.importlib, "import_module", absent_extension)
    with pytest.raises(
        inspect_compiled.MemoryEstimateCapabilityError, match="source-only"
    ) as excinfo:
        inspect_compiled.build_memory_estimate(SimpleNamespace(), SimpleNamespace())
    assert excinfo.value.field == "runtime.precision"


def test_absolute_memory_estimate_refuses_untyped_shape_before_any_formula(monkeypatch):
    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16}

    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    with pytest.raises(
        inspect_compiled.MemoryEstimateCapabilityError, match="CartesianGrid"
    ) as excinfo:
        inspect_compiled.build_memory_estimate(SimpleNamespace(), 32)
    assert excinfo.value.field == "mesh"


def test_absolute_memory_estimate_uses_reported_native_byte_width(monkeypatch):
    from pops.layouts import Uniform
    from tests.python.support.layout_plan import cartesian_grid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16}

    class Program:
        @staticmethod
        def estimate():
            return {"buffer_count": 0, "heavy_kernels": 0}

    mesh = cartesian_grid(n=4)
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U")
    )
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(program=Program()), mesh, layout=Uniform(mesh)
    )
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
            return {"dimension": 2, "real_bytes": 16}

    frame = Rectangle("estimate-grid", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    grid = CartesianGrid(frame=frame, cells=(3, 5))
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U")
    )
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(), grid, layout=Uniform(grid)
    )
    assert estimate.mesh_shape == (3, 5)
    assert estimate.cells == 15
    assert estimate.categories["state"] == 2 * 3 * 5 * 16


@pytest.mark.parametrize("shape", ((7,), (3, 5), (2, 3, 4)))
def test_absolute_memory_estimate_is_rank_generic(monkeypatch, shape):
    from math import prod

    from pops.domain import CartesianDomain
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid

    dimension = len(shape)

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": dimension, "real_bytes": 8}

    domain = CartesianDomain("estimate", (0.0,) * dimension, (1.0,) * dimension)
    grid = CartesianGrid(frame=domain.frame(), cells=shape)
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U")
    )
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(), grid, layout=Uniform(grid)
    )

    cells = prod(shape)
    ghost_cells = prod(extent + 4 for extent in shape) - cells
    assert estimate.cells == cells
    assert estimate.mesh_shape == shape
    assert estimate.categories["state"] == 2 * cells * 8
    assert estimate.categories["halo"] == ghost_cells * 2 * 8


def test_absolute_memory_estimate_accepts_strict_final_amr_protocol(monkeypatch):
    from pops.descriptors_report import CapabilitySet
    from tests.python.support.layout_plan import cartesian_grid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16}

    class FinalAMRProtocol:
        """The public final-AMR capability shape, without relying on a legacy layout class."""

        @staticmethod
        def capabilities():
            return CapabilitySet(
                {
                    "layout": "amr",
                    "dim": 2,
                    "max_levels": 3,
                    "transition_ratios": [[3, 3], [3, 3]],
                    "supports_amr": True,
                }
            )

    mesh = cartesian_grid(n=4)
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U")
    )
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(), mesh, layout=FinalAMRProtocol()
    )
    assert estimate.layout == "amr"
    assert estimate.categories["amr_patch"] == (3**2 + 3**4) * (2 * 4 * 4 * 16)


def test_absolute_memory_estimate_uses_spatial_rank_for_amr_refinement(monkeypatch):
    from pops.descriptors_report import CapabilitySet
    from pops.domain import CartesianDomain
    from pops.mesh import CartesianGrid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 3, "real_bytes": 8}

    class ThreeDimensionalAMR:
        @staticmethod
        def capabilities():
            return CapabilitySet(
                {
                    "layout": "amr",
                    "dim": 3,
                    "max_levels": 3,
                    "transition_ratios": [[2, 1, 3], [1, 4, 1]],
                    "supports_amr": True,
                }
            )

    domain = CartesianDomain("volume", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    mesh = CartesianGrid(frame=domain.frame(), cells=(2, 3, 4))
    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 2, {}, (), 0, "U")
    )
    estimate = inspect_compiled.build_memory_estimate(
        _memory_artifact(), mesh, layout=ThreeDimensionalAMR()
    )
    state = 2 * 2 * 3 * 4 * 8
    assert estimate.categories["amr_patch"] == (6 + 24) * state


def test_absolute_memory_estimate_refuses_amr_without_transition_ratios(monkeypatch):
    from pops.descriptors_report import CapabilitySet
    from tests.python.support.layout_plan import cartesian_grid

    class Extension:
        @staticmethod
        def runtime_environment_report():
            return {"dimension": 2, "real_bytes": 16}

    class IncompleteAMR:
        @staticmethod
        def capabilities():
            return CapabilitySet({"layout": "amr", "dim": 2, "max_levels": 2})

    monkeypatch.setattr(inspect_compiled.importlib, "import_module", lambda _name: Extension())
    monkeypatch.setattr(
        inspect_compiled, "_model_metadata", lambda _compiled: ((), 1, {}, (), 0, "U")
    )
    with pytest.raises(
        inspect_compiled.MemoryEstimateCapabilityError, match="transition_ratios"
    ) as excinfo:
        inspect_compiled.build_memory_estimate(
            _memory_artifact(), cartesian_grid(n=4), layout=IncompleteAMR()
        )
    assert excinfo.value.field == "layout.transition_ratios"
