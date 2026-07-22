#!/usr/bin/env python3
"""Final public scientific-output contract under a real MPI world.

This script is an MPI entrypoint, not a pytest-launched nested MPI job.  It proves both Uniform and
AMR ``Case -> validate -> resolve -> compile -> ExecutionContext.mpi_world -> bind -> run`` routes
against the exact HDF5 COLLECTIVE writer, both synchronously and on a duplicated async MPI lane.
The Uniform route also proves the public ParaView PER_RANK transaction: one VTU per rank, one
authenticated PVTU, and one temporal PVD. The AMR route repeats ParaView on the async lane. The AMR
witness includes a replicated coarse level, one sparse fine patch, and therefore at least one rank
with no fine-level piece.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

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
        AsyncScientificOutput,
        ConsumerGraph,
        HDF5,
        ParaView,
        ParallelMode,
        ScientificOutput,
        read_hdf5,
        read_paraview,
        read_paraview_parallel,
        read_paraview_series,
    )
    from pops.params import RuntimeParam
    from pops.projection import ConservativeCellAverage
    from pops.output._writers.hdf5 import _collective_temporary_owner
    from pops.time import FixedDt, StagePoint, TimePoint, every
except Exception as exc:  # noqa: BLE001 -- optional outside the required MPI lane
    require_mpi_or_skip("scientific-output MPI/HDF5 runtime import failed: %s" % exc)


N = 16
DT = 1.0e-3
VTK_DUPLICATE_CELL = 1
VTK_REFINED_CELL = 8


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
    consumers = [ScientificOutput(
        format=HDF5(mode=ParallelMode.COLLECTIVE),
        schedule=every(1, clock=program.clock),
        fields=(block_state,),
        target="density",
    )]
    consumers.append(AsyncScientificOutput(
        format=HDF5(mode=ParallelMode.COLLECTIVE),
        schedule=every(1, clock=program.clock),
        fields=(block_state,),
        target="density_async",
        queue_capacity=1,
    ))
    # Two collective HDF5 observers on the same accepted tick exercise the process-local
    # shared worker FIFO before the asynchronous PVTU observer below.
    consumers.append(AsyncScientificOutput(
        format=HDF5(mode=ParallelMode.COLLECTIVE),
        schedule=every(1, clock=program.clock),
        fields=(block_state,),
        target="density_async_second",
        queue_capacity=1,
    ))
    # The Uniform rank-one publication fault below is injected below the public writer facade.
    # Both Uniform and sparse AMR then publish standard PVTU/PVD companions for their rank leaves.
    consumers.append(ScientificOutput(
        format=ParaView(mode=ParallelMode.PER_RANK),
        schedule=every(1, clock=program.clock),
        fields=(block_state,),
        target="paraview",
    ))
    if adaptive:
        consumers.append(AsyncScientificOutput(
            format=ParaView(mode=ParallelMode.PER_RANK),
            schedule=every(1, clock=program.clock),
            fields=(block_state,),
            target="paraview_async",
            queue_capacity=1,
        ))
    case.consumers(ConsumerGraph.from_consumers(tuple(consumers)))

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


def _run_case(root: Path, *, adaptive: bool) -> tuple[
    Any, Path, Path, Path, np.ndarray | None,
]:
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
    if not adaptive:
        _validate_paraview_collective_rollback(runtime, output_root, initial)
    expected_steps = 1 if adaptive else 2
    report = pops.run(
        runtime,
        t_end=expected_steps * DT,
        max_steps=expected_steps,
        output_dir=output_root,
    )
    reports = allgather_value(COMM, (
        report.accepted_steps,
        report.run_identity.token,
        report.bind_identity.token,
    ))
    if any(row != reports[0] for row in reports[1:]) \
            or reports[0][0] != expected_steps:
        raise AssertionError("%s run report differs across ranks: %r" % (label, reports))
    _validate_paraview(
        output_root / "paraview",
        adaptive=adaptive,
        expected_steps=expected_steps,
    )
    if adaptive:
        _validate_paraview(
            output_root / "paraview_async",
            adaptive=True,
            expected_steps=expected_steps,
        )
    synchronous = tuple(
        str(path) for path in sorted((output_root / "density").rglob("*.h5"))
    ) if RANK == 0 else ()
    asynchronous = tuple(
        str(path) for path in sorted((output_root / "density_async").rglob("*.h5"))
    ) if RANK == 0 else ()
    asynchronous_second = tuple(
        str(path)
        for path in sorted((output_root / "density_async_second").rglob("*.h5"))
    ) if RANK == 0 else ()
    synchronous = broadcast_value(COMM, synchronous, root=0)
    asynchronous = broadcast_value(COMM, asynchronous, root=0)
    asynchronous_second = broadcast_value(COMM, asynchronous_second, root=0)
    if len(synchronous) != expected_steps \
            or len(asynchronous) != expected_steps \
            or len(asynchronous_second) != expected_steps:
        raise AssertionError(
            "%s produced HDF5 sync/async/async-second counts %d/%d/%d "
            "instead of %d/%d/%d"
            % (
                label,
                len(synchronous),
                len(asynchronous),
                len(asynchronous_second),
                expected_steps,
                expected_steps,
                expected_steps,
            ))
    return (
        runtime,
        Path(synchronous[-1]),
        Path(asynchronous[-1]),
        Path(asynchronous_second[-1]),
        initial,
    )


def _validate_paraview_collective_rollback(
    runtime: Any,
    output_root: Path,
    initial: np.ndarray | None,
) -> None:
    """Inject one rank-local publication fault and prove whole-step compensation."""
    if initial is None:
        raise AssertionError("ParaView rollback witness requires the exact Uniform bind array")
    from pops.output._writers.common import _StagedOutputFile

    before = np.asarray(runtime.state_global("fluid"), dtype=np.float64).copy()
    original_publish = _StagedOutputFile.publish
    injected = False

    def fail_one_leaf(staged: Any) -> Any:
        nonlocal injected
        if not injected and staged.target.suffix == ".vtu":
            injected = True
            raise OSError("injected rank-one ParaView leaf publication failure")
        return original_publish(staged)

    if RANK == 1:
        _StagedOutputFile.publish = fail_one_leaf
    error = None
    try:
        pops.run(runtime, t_end=DT, max_steps=1, output_dir=output_root)
    except RuntimeError as exc:
        error = str(exc)
    finally:
        if RANK == 1:
            _StagedOutputFile.publish = original_publish
    errors = allgather_value(COMM, error)
    if not all(
            item is not None and "rank 1" in item
            and "injected rank-one ParaView leaf publication failure" in item
            for item in errors):
        raise AssertionError(
            "rank-local ParaView publication fault did not reach every rank: %r" % (errors,))
    if len(set(errors)) != 1:
        raise AssertionError(
            "ParaView publication-fault consensus differs across ranks: %r" % (errors,))
    if runtime.macro_step() != 0 or runtime.time() != 0.0:
        raise AssertionError("failed ParaView publication committed the accepted clock")
    np.testing.assert_array_equal(
        np.asarray(runtime.state_global("fluid"), dtype=np.float64), before)

    def validate_residue() -> None:
        if RANK != 0:
            return
        files = tuple(sorted(
            path.relative_to(output_root).as_posix()
            for path in output_root.rglob("*") if path.is_file()
        ))
        if files:
            raise AssertionError(
                "collective ParaView rollback left published/staged residue: %r" % (files,))

    _collective_local("ParaView rollback residue verification", validate_residue)
    barrier(COMM)


def _validate_paraview(
    output_root: Path,
    *,
    adaptive: bool,
    expected_steps: int,
) -> None:
    """Authenticate the complete PVD -> PVTU -> rank-VTU hierarchy on rank zero."""
    def validate() -> None:
        if RANK != 0:
            return
        leaves = tuple(sorted(output_root.rglob("*.vtu")))
        parallel = tuple(sorted(output_root.rglob("*.pvtu")))
        collections = tuple(sorted(output_root.rglob("*.pvd")))
        if len(leaves) != expected_steps * SIZE:
            raise AssertionError(
                "%d PER_RANK sample(s) produced %d VTU leaves instead of %d"
                % (expected_steps, len(leaves), expected_steps * SIZE))
        if len(parallel) != expected_steps or len(collections) != expected_steps:
            raise AssertionError(
                "%d ParaView sample(s) produced PVTU/PVD counts %d/%d"
                % (expected_steps, len(parallel), len(collections)))

        series = read_paraview_series(collections[-1])
        generic_series = read_paraview(collections[-1])
        if generic_series.output_identity != series.output_identity:
            raise AssertionError("generic ParaView reopen disagrees with the exact PVD reader")
        if series.kind != "pvd" or series.paths != parallel:
            raise AssertionError("latest PVD does not reference every immutable PVTU sample")
        entries = series.manifest["entries"]
        expected_macro_steps = list(range(1, expected_steps + 1))
        expected_times = [step * DT for step in expected_macro_steps]
        if [row["macro_step"] for row in entries] != expected_macro_steps:
            raise AssertionError("PVD did not preserve every accepted macro step")
        if [float.fromhex(row["time_hex"]) for row in entries] != expected_times:
            raise AssertionError("PVD did not preserve every physical output time")

        xml_collection = ET.parse(collections[-1]).getroot()
        datasets = xml_collection.findall("./Collection/DataSet")
        if [Path(node.attrib["file"]).suffix for node in datasets] \
                != [".pvtu"] * expected_steps:
            raise AssertionError("PVD does not expose standard PVTU temporal components")
        if any(Path(node.attrib["file"]).is_absolute() for node in datasets):
            raise AssertionError("PVD companion references must remain portable relative paths")

        all_leaf_paths = []
        for macro_step, pvtu_path in enumerate(parallel, start=1):
            reopened_parallel = read_paraview_parallel(pvtu_path)
            if reopened_parallel.kind != "pvtu" or len(reopened_parallel.paths) != SIZE:
                raise AssertionError("PVTU does not authenticate one piece per MPI rank")
            if reopened_parallel.manifest["clock"]["macro_step"] != macro_step:
                raise AssertionError("PVTU clock differs from its temporal catalogue entry")
            if [row["rank"] for row in reopened_parallel.manifest["pieces"]] \
                    != list(range(SIZE)):
                raise AssertionError("PVTU rank-piece order is not deterministic")
            all_leaf_paths.extend(reopened_parallel.paths)

            xml_parallel = ET.parse(pvtu_path).getroot()
            sources = [
                node.attrib["Source"]
                for node in xml_parallel.findall("./PUnstructuredGrid/Piece")
            ]
            if sources != [path.name for path in reopened_parallel.paths]:
                raise AssertionError("PVTU XML pieces differ from authenticated rank leaves")
            if any(Path(source).is_absolute() for source in sources):
                raise AssertionError("PVTU piece references must remain portable relative paths")
            public_field = xml_parallel.find(
                "./PUnstructuredGrid/PCellData/PDataArray[@Name='U']")
            if public_field is None or public_field.attrib.get("ComponentName0") != "rho":
                raise AssertionError("PVTU lost the user-authored U/rho field names")

        if tuple(sorted(all_leaf_paths)) != leaves:
            raise AssertionError("PVTU catalogues do not cover every emitted VTU leaf exactly")

        # The exact PoPS reopen above is mandatory and authenticates every component.  When the
        # independently maintained VTK Python reader is installed in the MPI lane, also prove that
        # the standard PVTU is directly consumable without any PoPS-specific adapter.
        try:
            from vtkmodules.vtkIOXML import vtkXMLPUnstructuredGridReader
        except ImportError:
            vtkXMLPUnstructuredGridReader = None
        if vtkXMLPUnstructuredGridReader is not None:
            for pvtu_path in parallel:
                reader = vtkXMLPUnstructuredGridReader()
                reader.SetFileName(str(pvtu_path))
                reader.Update()
                grid = reader.GetOutput()
                if grid.GetNumberOfCells() < 1 \
                        or grid.GetCellData().GetArray("U") is None:
                    raise AssertionError("the native VTK reader could not consume the PVTU")

        observed_ranks = set()
        observed_steps = set()
        fine_ranks_by_step = {step: set() for step in expected_macro_steps}
        for leaf_path in leaves:
            leaf = read_paraview(leaf_path)
            selection = leaf.manifest["snapshot"]["selection"]
            rank = selection["rank"]
            macro_step = leaf.manifest["snapshot"]["clock"]["macro_step"]
            observed_ranks.add(rank)
            observed_steps.add(macro_step)
            if selection["parallel_mode"] != "per_rank" or selection["size"] != SIZE:
                raise AssertionError("VTU leaf lost its public PER_RANK selection evidence")
            names = {
                row["name"]
                for row in leaf.manifest["datasets"]["fields"].values()
            }
            if names != {"U"} or "U" not in leaf.arrays or "field_0000" in leaf.arrays:
                raise AssertionError("VTU did not retain the user-authored field name U")

            levels = np.asarray(leaf.arrays["pops_level"], dtype=np.int64)
            ghosts = np.asarray(leaf.arrays["vtkGhostType"], dtype=np.uint8)
            coverage = np.asarray(leaf.arrays["pops_coverage"], dtype=np.uint8)
            if np.any(ghosts & np.uint8(~(VTK_DUPLICATE_CELL | VTK_REFINED_CELL) & 0xFF)):
                raise AssertionError("VTU emitted an unsupported VTK ghost bit")
            if not np.array_equal(
                    (ghosts & np.uint8(VTK_REFINED_CELL)) != 0,
                    coverage != 0):
                raise AssertionError("VTU refined ghost bits differ from AMR coverage")
            if adaptive:
                if not set(np.unique(levels)).issubset({0, 1}) or 0 not in levels:
                    raise AssertionError("AMR VTU leaf lost its coarse level")
                if np.any(levels == 1):
                    fine_ranks_by_step[macro_step].add(rank)
                    if np.any(ghosts[levels == 1] & np.uint8(VTK_DUPLICATE_CELL)):
                        raise AssertionError("distributed fine cells were marked replicated")
                if rank > 0 and np.any(
                        (ghosts[levels == 0] & np.uint8(VTK_DUPLICATE_CELL)) == 0):
                    raise AssertionError("replicated non-root coarse cells lost duplicate flags")
            elif levels.size and (set(np.unique(levels)) != {0} or np.any(ghosts)):
                raise AssertionError("Uniform VTU leaf invented AMR levels or ghost flags")

            xml_leaf = ET.parse(leaf_path).getroot()
            if xml_leaf.attrib.get("compressor") != "vtkZLibDataCompressor":
                raise AssertionError("VTU leaf is not using standard VTK zlib compression")
            piece = xml_leaf.find("./UnstructuredGrid/Piece")
            if piece is None:
                raise AssertionError("VTU leaf has no UnstructuredGrid piece")
            cell_count = int(piece.attrib["NumberOfCells"])
            point_count = int(piece.attrib["NumberOfPoints"])
            if cell_count == 0:
                if point_count != 0:
                    raise AssertionError("empty VTU piece retained unattached points")
            elif not 0 < point_count < 4 * cell_count:
                raise AssertionError("VTU quadrilaterals do not share their common points")
            public_field = piece.find("./CellData/DataArray[@Name='U']")
            if public_field is None or public_field.attrib.get("ComponentName0") != "rho":
                raise AssertionError("VTU lost the user-authored U/rho field names")
        if observed_ranks != set(range(SIZE)) \
                or observed_steps != set(expected_macro_steps):
            raise AssertionError("VTU leaves do not cover every rank and temporal sample")
        if adaptive and any(
                not owners or not owners < set(range(SIZE))
                for owners in fine_ranks_by_step.values()):
            raise AssertionError(
                "AMR PVTU witness did not retain a rank with no fine-level piece")

    _collective_local("ParaView PVD/PVTU/VTU verification", validate)


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
        (
            _uniform_runtime,
            uniform_path,
            uniform_async_path,
            uniform_async_second_path,
            initial,
        ) = _run_case(root, adaptive=False)
        if initial is None:
            raise AssertionError("uniform case lost its exact bind array")
        _validate_uniform(uniform_path, initial)
        _validate_uniform(uniform_async_path, initial)
        _validate_uniform(uniform_async_second_path, initial)
        (
            _amr_runtime,
            amr_path,
            amr_async_path,
            amr_async_second_path,
            _,
        ) = _run_case(root, adaptive=True)
        _validate_amr(amr_path)
        _validate_amr(amr_async_path)
        _validate_amr(amr_async_second_path)
        if RANK == 0:
            print("PASS test_scientific_output_mpi")
    finally:
        barrier(COMM)
        if RANK == 0:
            shutil.rmtree(root, ignore_errors=True)
        barrier(COMM)


if __name__ == "__main__":
    main()
