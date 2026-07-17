#!/usr/bin/env python3
"""Final public scientific-output contract under a real MPI world.

This script is an MPI entrypoint, not a pytest-launched nested MPI job.  It proves both Uniform and
AMR ``Case -> validate -> resolve -> compile -> ExecutionContext.mpi_world -> bind -> run`` routes
against the exact HDF5 COLLECTIVE writer.  The AMR witness includes a replicated coarse level, one
sparse fine patch, and therefore at least one rank with no fine-level piece.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from _compile_once import compile_resolved_plan_once
from tests.python.support.requirements import require_mpi_or_skip


try:
    import h5py  # noqa: F401 -- serial HDF5 reopen on rank zero
    import numpy as np

    import pops
    from pops import _pops
    from pops._native_collectives import (
        allgather_value,
        barrier,
        broadcast_value,
        rank as world_rank,
        size as world_size,
    )
    from pops.amr import (
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        PatchLayout,
        Tag,
    )
    from pops.codegen import Production
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.initial import InitialCondition
    from pops.layouts import AMR, Uniform
    from pops.lib.amr import StateTransfer
    from pops.lib.initial import Gaussian
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.output import (
        ConsumerGraph,
        HDF5,
        ParallelMode,
        ScientificOutput,
        read_hdf5,
    )
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.output._writers.hdf5 import _collective_temporary_owner
    from pops.time import FixedDt, StagePoint, TimePoint, every
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("scientific-output MPI/HDF5 runtime import failed: %s" % exc)


N = 16
DT = 1.0e-3


if getattr(_pops, "__has_mpi__", False) is not True:
    require_mpi_or_skip("scientific-output contract requires a native MPI build")
if getattr(_pops, "__has_parallel_hdf5__", False) is not True:
    require_mpi_or_skip("scientific-output contract requires native parallel HDF5")

COMM = _pops.mpi_world()
RANK = world_rank(COMM)
SIZE = world_size(COMM)


if SIZE < 2:
    require_mpi_or_skip("scientific-output contract needs mpiexec -n 2")
def _collective_local(label: str, operation: Any) -> Any:
    """Run a noncollective local operation and report its error before the next MPI phase."""
    result = None
    error = None
    try:
        result = operation()
    except BaseException as exc:  # noqa: BLE001 -- test must report every rank before continuing
        error = "%s: %s" % (type(exc).__name__, exc)
    errors = allgather_value(COMM, error)
    failures = [
        "rank %d: %s" % (rank, value)
        for rank, value in enumerate(errors) if value is not None
    ]
    if failures:
        raise RuntimeError("%s failed: %s" % (label, "; ".join(failures)))
    return result


def _shared_directory() -> Path:
    path = None
    if RANK == 0:
        path = tempfile.mkdtemp(prefix="pops-scientific-output-mpi-")
    path = broadcast_value(COMM, path, root=0)
    if not isinstance(path, str):
        raise RuntimeError("rank zero did not publish the shared test directory")
    return Path(path)


def _validate_native_binding_error_consensus(root: Path) -> None:
    """One malformed rank must fail before HDF5 while every peer receives the same cause."""
    values = (
        [[1.0, 2.0], [3.0, 4.0]]
        if RANK == 0
        else np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    )
    error = None
    try:
        _pops._write_parallel_hdf5(
            COMM,
            str(root / "binding-must-not-enter-hdf5.h5"),
            "{}",
            {"geometry/0000/coverage": np.zeros((2, 2), dtype=np.bool_)},
            ({
                "dataset": "fields/0000/values",
                "dtype": np.dtype(np.float64).str,
                "shape": (2, 2),
                "pieces": ({"lower": (0, 0), "upper": (2, 2), "values": values},),
            },),
        )
    except RuntimeError as exc:
        error = str(exc)
    errors = allgather_value(COMM, error)
    if not all(item is not None and "binding input validation" in item for item in errors):
        raise AssertionError("rank-local binding fault did not reach all ranks: %r" % (errors,))
    if len(set(errors)) != 1:
        raise AssertionError("binding fault consensus differs across ranks: %r" % (errors,))
    if (root / "binding-must-not-enter-hdf5.h5").exists():
        raise AssertionError("rank-local binding fault entered HDF5")


def _validate_temporary_owner_error_consensus(root: Path) -> None:
    """A real rank-zero filesystem failure must reach every peer before another collective."""
    missing = root / "missing-collective-staging-file.h5"
    error = None
    try:
        _collective_temporary_owner(COMM, missing)
    except RuntimeError as exc:
        error = str(exc)
    errors = allgather_value(COMM, error)
    if not all(
        item is not None
        and "temporary inode authentication failed" in item
        and "FileNotFoundError" in item
        for item in errors
    ):
        raise AssertionError(
            "rank-zero staging failure did not reach every rank: %r" % (errors,)
        )
    if len(set(errors)) != 1:
        raise AssertionError(
            "rank-zero staging failure consensus differs across ranks: %r" % (errors,)
        )


def _authored_case(*, adaptive: bool) -> tuple[pops.Case, Any, np.ndarray | None]:
    label = "amr" if adaptive else "uniform"
    frame = Rectangle(
        "scientific-output-%s-domain" % label,
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("scientific-output-%s-model" % label, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    flux = model.flux(
        "stationary_flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    rate = model.rate("stationary_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )

    case = pops.Case("scientific-output-%s-case" % label)
    block = case.block("fluid", model=model)
    block_state = block[state]
    case.numerics(numerics, block=block)
    program = pops.Program("scientific-output-%s-forward-euler" % label)
    temporal = program.state(block_state)
    stage = StagePoint("main_stage", {"main": TimePoint(program.clock, 0)})
    derivative = program.value("rhs", rate(temporal.n), at=stage)
    accepted = program.value(
        "accepted",
        temporal.n + program.dt * derivative,
        at=temporal.next.point,
    )
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.consumers(ConsumerGraph.from_consumers((ScientificOutput(
        format=HDF5(mode=ParallelMode.COLLECTIVE),
        schedule=every(1, clock=program.clock),
        fields=(block_state,),
        target="density",
    ),)))

    grid = CartesianGrid(
        frame=frame,
        cells=(N, N),
        periodic=PeriodicAxes(frame.axes),
    )
    if not adaptive:
        initial = np.arange(N * N, dtype=np.float64).reshape(1, N, N)
        initial += np.float64(1.125)
        return case, Uniform(grid), initial

    case.initials.add(InitialCondition(
        state=block_state,
        value=Gaussian(
            frame=frame,
            center={x_axis: 0.5, y_axis: 0.5},
            background=1.0,
            amplitude=0.5,
            inverse_width=1.0 / (0.10 * 0.10),
        ),
        projection=ConservativeCellAverage(),
    ))
    threshold = case.param(RuntimeParam(
        "scientific_output_refine_threshold", default=1.30))
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    layout = AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(block_state) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
        patch_layout=PatchLayout(distribute_coarse=False, coarse_max_grid=N),
    )
    return case, layout, None


def _artifact(case: pops.Case, layout: Any, *, route: str) -> Any:
    resolved = _collective_local(
        route + " resolution",
        lambda: pops.resolve(pops.validate(case), layout=layout, backend=Production()),
    )
    return compile_resolved_plan_once(
        COMM,
        resolved,
        route=route,
        compile_artifact=pops.compile,
    )


def _run_case(root: Path, *, adaptive: bool) -> tuple[Any, Path, np.ndarray | None]:
    label = "amr" if adaptive else "uniform"
    case, layout, initial = _collective_local(
        label + " authoring", lambda: _authored_case(adaptive=adaptive))
    artifact = _artifact(case, layout, route="scientific-output-" + label)
    context = pops.ExecutionContext.mpi_world(artifact)
    if initial is None:
        runtime = pops.bind(artifact, resources={"execution_context": context})
    else:
        runtime = pops.bind(
            artifact,
            initial_state={"fluid": initial},
            resources={"execution_context": context},
        )
    levels = allgather_value(COMM, int(runtime.n_levels()))
    expected_levels = 2 if adaptive else 1
    if levels != (expected_levels,) * SIZE:
        raise AssertionError("%s hierarchy differs across ranks: %r" % (label, levels))
    output_root = root / label
    report = pops.run(runtime, t_end=DT, max_steps=1, output_dir=output_root)
    reports = allgather_value(COMM, (
        report.accepted_steps,
        report.run_identity.token,
        report.bind_identity.token,
    ))
    if any(row != reports[0] for row in reports[1:]) or reports[0][0] != 1:
        raise AssertionError("%s run report differs across ranks: %r" % (label, reports))
    files = tuple(str(path) for path in output_root.rglob("*.h5")) if RANK == 0 else ()
    files = broadcast_value(COMM, files, root=0)
    if len(files) != 1:
        raise AssertionError("%s produced %d HDF5 artifacts" % (label, len(files)))
    return runtime, Path(files[0]), initial


def _validate_uniform(path: Path, initial: np.ndarray) -> None:
    def validate() -> None:
        if RANK != 0:
            return
        reopened = read_hdf5(path)
        datasets = tuple(reopened.manifest["datasets"]["fields"].values())
        if len(datasets) != 1:
            raise AssertionError("uniform output did not contain exactly one field")
        np.testing.assert_array_equal(reopened.arrays[datasets[0]], initial)
        selection = reopened.manifest["snapshot"]["selection"]
        assert selection["parallel_mode"] == "collective"
        assert selection["ranks"] == list(range(SIZE))

    _collective_local("uniform HDF5 verification", validate)


def _validate_amr(path: Path) -> None:
    def validate() -> None:
        if RANK != 0:
            return
        reopened = read_hdf5(path)
        snapshot = reopened.manifest["snapshot"]
        fields = {row["key"]["level"]: row for row in snapshot["fields"]}
        geometries = {row["level"]: row for row in snapshot["geometries"]}
        if set(fields) != {0, 1} or set(geometries) != {0, 1}:
            raise AssertionError("AMR output did not preserve both exact levels")
        coarse = fields[0]["pieces"]
        fine = fields[1]["pieces"]
        if not coarse or not all(piece["replicated"] for piece in coarse):
            raise AssertionError("AMR coarse output lost its replicated ownership contract")
        if {piece["owner_rank"] for piece in coarse} != {0}:
            raise AssertionError("collective AMR coarse authority was not assigned to rank zero")
        if not fine or any(piece["replicated"] for piece in fine):
            raise AssertionError("AMR fine output did not retain distributed ownership")
        fine_owners = {piece["owner_rank"] for piece in fine}
        if not fine_owners < set(range(SIZE)):
            raise AssertionError(
                "AMR witness did not retain a rank with no fine-level output piece")
        fine_geometry = geometries[1]
        represented = sum(
            (box[2] - box[0]) * (box[3] - box[1])
            for box in fine_geometry["boxes"]
        )
        ny, nx = fine_geometry["cell_shape"]
        if represented >= ny * nx:
            raise AssertionError("AMR fine geometry is not a sparse hierarchy witness")
        if {piece["global_box_index"] for piece in fine} != set(
                range(len(fine_geometry["boxes"]))):
            raise AssertionError("AMR fine pieces do not authenticate every geometry box")
        selection = snapshot["selection"]
        assert selection["parallel_mode"] == "collective"
        assert selection["ranks"] == list(range(SIZE))

    _collective_local("AMR HDF5 verification", validate)


def main() -> None:
    root = _shared_directory()
    os.environ["POPS_CACHE_DIR"] = str(root / "cache")
    try:
        _validate_native_binding_error_consensus(root)
        _validate_temporary_owner_error_consensus(root)
        _uniform_runtime, uniform_path, initial = _run_case(root, adaptive=False)
        if initial is None:
            raise AssertionError("uniform case lost its exact bind array")
        _validate_uniform(uniform_path, initial)
        _amr_runtime, amr_path, _ = _run_case(root, adaptive=True)
        _validate_amr(amr_path)
        if RANK == 0:
            print("PASS test_scientific_output_mpi")
    finally:
        barrier(COMM)
        if RANK == 0:
            shutil.rmtree(root, ignore_errors=True)
        barrier(COMM)


if __name__ == "__main__":
    main()
