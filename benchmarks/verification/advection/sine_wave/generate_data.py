#!/usr/bin/env python3
"""Generate the periodic sine-wave advection verification data.

Read this file from top to bottom: constants and command-line choices come
first, then sections 1--9 describe the complete PoPS case and public runtime
lifecycle.  Diagnostics, provenance, and atomic files are delegated to the
private ``_case_support.py`` module so they do not hide the numerical case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CASE_DIRECTORY = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.verification.advection.sine_wave._case_support import (  # noqa: E402
    collect_snapshot,
    compute_metrics,
    coverage_witnesses,
    execution_provenance,
    publish_result,
    source_provenance,
)
from helpers.verification import direction_velocity  # noqa: E402


# Scientific constants: change them here, never inside diagnostics or plotting.
EPSILON = 0.10
PROBE_TIME = 0.37
FINAL_TIME = 1.0
BASE_CFL = 0.40
MAX_STEPS = 100_000
WAVE_NUMBERS = (1, 2, 3)

# AMR constants shared by the readable layout section and the private diagnostics.
REFINEMENT_RATIO = 2
MAX_LEVELS = 2
REGRID_EVERY = 4
PATCH_CENTER = (0.25, 0.40, 0.55)
PATCH_VELOCITY = (0.50, 0.25, 0.125)
PATCH_HALF_WIDTH = (0.18, 0.15, 0.13)
WITNESS_REFERENCE_POINT = (0.137, 0.137, 0.137)

# Command-line defaults and authenticated data schemas.
DEFAULT_RESOLUTION = 32
DEFAULT_BLOCK_SIZE = 16
DEFAULT_TIME_SNAPSHOTS = 17
SCHEMA_VERSION = "pops.sine-wave.v3"
SOURCE_SCHEMA_VERSION = "pops.sine-wave.source.v2"


def _effective_cfl(dimension: int) -> float:
    """Use the dimensionally conservative CFL expected by unsplit ND advection."""
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    return BASE_CFL / dimension


def _parse_resolution(raw: str, dimension: int) -> tuple[int, ...]:
    values = tuple(int(token) for token in raw.split(","))
    if len(values) == 1:
        values *= dimension
    if len(values) != dimension or any(value < 4 for value in values):
        raise ValueError("resolution must be one integer or %d integers, all >= 4" % dimension)
    return values


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--resolution", default=str(DEFAULT_RESOLUTION))
    parser.add_argument("--mode", choices=("x", "y", "z", "xy", "diagonal"), default="diagonal")
    parser.add_argument(
        "--layout", choices=("uniform", "amr-frozen", "amr-mobile"), default="uniform"
    )
    parser.add_argument("--subcycling", choices=("synchronous", "subcycled"), default="synchronous")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="number of complete periodic crossings before the final snapshot",
    )
    parser.add_argument(
        "--time-snapshots",
        type=int,
        default=DEFAULT_TIME_SNAPSHOTS,
        help=(
            "number of uniformly spaced visualization snapshots, including t=0 and t=T (minimum: 9)"
        ),
    )
    parser.add_argument("--mpi", action="store_true", help="bind through the native MPI context")
    parser.add_argument(
        "--mpi-topology",
        default=None,
        help="required Cartesian rank decomposition, for example 2,2,2",
    )
    parser.add_argument(
        "--obligation",
        action="append",
        default=[],
        choices=(
            "block_face",
            "block_edge_3d",
            "block_corner_3d",
            "coarse_fine_interface",
            "periodic_boundary",
            "prescribed_mobile_regrid",
            "repeated_patch_crossing",
            "second_order_convergence",
        ),
        help="named coverage obligation authenticated with this complete run",
    )
    parser.add_argument("--output", type=Path, default=CASE_DIRECTORY / "results")
    args = parser.parse_args()
    args.resolution = _parse_resolution(args.resolution, args.dimension)
    wave_numbers = WAVE_NUMBERS[: args.dimension]
    if any(
        count <= 2 * abs(wave) for count, wave in zip(args.resolution, wave_numbers, strict=True)
    ):
        parser.error("resolution must keep every wave strictly below the Nyquist limit")
    if args.block_size < 2:
        parser.error("--block-size must be at least 2")
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.time_snapshots < 9:
        parser.error("--time-snapshots must be at least 9")
    if args.mode == "xy":
        if args.dimension != 3:
            parser.error("mode 'xy' is reserved for the 3D block-edge witness")
        args.velocity = (1.0, 1.0, 0.0)
    else:
        try:
            args.velocity = direction_velocity(args.mode, args.dimension)
        except ValueError as error:
            parser.error(str(error))
    if args.layout == "uniform" and args.subcycling != "synchronous":
        parser.error("uniform layout has no AMR clock; use --subcycling synchronous")
    if args.mpi_topology is None:
        if args.mpi:
            parser.error("--mpi requires an explicit --mpi-topology receipt contract")
    else:
        try:
            args.mpi_topology = tuple(int(value) for value in args.mpi_topology.split(","))
        except ValueError:
            parser.error("--mpi-topology must be comma-separated positive integers")
        if (
            not args.mpi
            or len(args.mpi_topology) != args.dimension
            or any(value < 1 for value in args.mpi_topology)
        ):
            parser.error("--mpi-topology must match --mpi and the requested dimension")
    return args


def main() -> None:
    args = _arguments()

    # Public PoPS vocabulary used below; no native time step is implemented in Python.
    import pops
    from pops._native_collectives import allgather_value
    from pops.amr import (
        AMRClockRelation,
        AMRExecution,
        AMRHierarchy,
        AMRRegrid,
        AMRTagging,
        AMRTransfer,
        Buffer,
        Coarsen,
        ConflictPolicy,
        EqualityPolicy,
        Hysteresis,
        PatchLayout,
        PrescribedWindow,
        Tag,
    )
    from pops.analytic import coordinates, sin
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian
    from pops.initial import InitialCondition
    from pops.layouts import AMR, Uniform
    from pops.lib.amr import BergerRigoutsos, StateTransfer
    from pops.lib.initial import Analytic
    from pops.lib.time import SSPRK2
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes, RegularBlocks
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.reconstruction import limiters
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.representations import Conservative
    from pops.runtime_environment import runtime_environment_report
    from pops.spaces import CellState
    from pops.time import AdaptiveCFL, every

    velocity = args.velocity
    wave_numbers = WAVE_NUMBERS[: args.dimension]
    final_time = FINAL_TIME * args.cycles

    # 1. Domaine périodique [0, 1]^d et grille cartésienne de volumes finis.
    domain = CartesianDomain(
        "periodic_unit_box", (0.0,) * args.dimension, (1.0,) * args.dimension
    ).tag("fluid")
    frame = domain.frame(Cartesian(args.dimension))
    axes = frame.axes
    if args.layout == "uniform":
        grid = CartesianGrid(
            frame=frame,
            cells=args.resolution,
            periodic=PeriodicAxes(axes),
            blocks=RegularBlocks(max_cells=args.block_size),
        )
    else:
        # Le lowering AMR possède sa hiérarchie et ne reçoit donc aucun pavage uniforme.
        grid = CartesianGrid(
            frame=frame,
            cells=args.resolution,
            periodic=PeriodicAxes(axes),
        )

    # 2. Modèle traceur : dq/dt + a.grad(q) = 0, soit div(a q) pour a constant.
    model = pops.Model("periodic_sine_advection", frame=frame)
    q_state = model.state(
        "U", components=("q",), representation=Conservative(), space=CellState(frame=frame)
    )
    (q,) = q_state
    advection_velocity = model.vector(
        "a", frame=frame, components=dict(zip(axes, velocity, strict=True))
    )
    flux = model.flux(
        "advection_flux",
        frame=frame,
        state=q_state,
        components={axis: (speed * q,) for axis, speed in zip(axes, velocity, strict=True)},
        waves={axis: (speed,) for axis, speed in zip(axes, velocity, strict=True)},
    )
    rate = model.rate("advection_rate", equation=ddt(q_state) == -div(flux))

    # 3. Volumes finis : reconstruction MUSCL-VanLeer, flux amont et Case publique.
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(q_state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.ScalarUpwind(velocity=advection_velocity),
        ),
    )
    case = pops.Case("benchmark_periodic_sine_wave")
    tracer = case.block("tracer", model=model)
    tracer_q = tracer[q_state]
    case.numerics(numerics, block=tracer)

    # 4. Horloge et fenêtre AMR prescrite : aucun marqueur n'est transporté.
    program = SSPRK2(tracer_q, rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=_effective_cfl(args.dimension)))
    case.program(program)
    patch_velocity = (
        (0.0,) * args.dimension if args.layout == "amr-frozen" else PATCH_VELOCITY[: args.dimension]
    )
    prescribed_patch = (
        None
        if args.layout == "uniform"
        else PrescribedWindow(
            frame=frame,
            clock=program.clock,
            center=PATCH_CENTER[: args.dimension],
            half_width=PATCH_HALF_WIDTH[: args.dimension],
            velocity=patch_velocity,
        )
    )

    # 5. Conditions initiales analytiques projetées en moyennes de cellules conservatives.
    analytic_coordinates = coordinates(frame)
    phase = sum(
        coefficient * coordinate
        for coefficient, coordinate in zip(wave_numbers, analytic_coordinates, strict=True)
    )
    case.initials.add(
        InitialCondition(
            state=tracer_q,
            value=Analytic(
                frame=frame,
                components=(1.0 + EPSILON * sin(2.0 * np.pi * phase),),
            ),
            projection=ConservativeCellAverage(),
        )
    )

    # 6. Layout : grille uniforme ou AMR taggée, regriddée et éventuellement sous-cyclée.
    bind_params = {}
    if args.layout == "uniform":
        layout = Uniform(grid)
    else:
        if prescribed_patch is None:
            raise RuntimeError("AMR layout lost its prescribed geometric patch")
        tagging = AMRTagging(
            rules=(
                Tag(prescribed_patch),
                Coarsen(~prescribed_patch),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        )
        transfer = AMRTransfer()
        transfer.state(tracer_q, StateTransfer())
        regrid = (
            AMRRegrid.frozen()
            if args.layout == "amr-frozen"
            else AMRRegrid(schedule=every(REGRID_EVERY, clock=program.clock))
        )
        execution = AMRExecution.synchronous()
        if args.subcycling == "subcycled":
            execution = AMRExecution.subcycled((AMRClockRelation(0, 1, REFINEMENT_RATIO),))
        layout = AMR(
            grid=grid,
            hierarchy=AMRHierarchy(
                max_levels=MAX_LEVELS,
                ratios=(REFINEMENT_RATIO,),
            ),
            tagging=tagging,
            regrid=regrid,
            transfer=transfer,
            execution=execution,
            patch_layout=PatchLayout(
                distribute_coarse=True,
                coarse_max_grid=args.block_size,
            ),
            clustering=BergerRigoutsos(maximum_box_size=args.block_size),
        )

    # 7. Cycle public explicite : valider, résoudre puis compiler le cas et son layout.
    validated = pops.validate(case)
    resolved = pops.resolve(validated, layout=layout)
    resolved_bind_params = {
        validated.resolve(parameter): value for parameter, value in bind_params.items()
    }
    artifact = pops.compile(resolved)
    if int(artifact.resolved_dimension) != args.dimension:
        raise RuntimeError(
            "requested Dim%d but the compiled artifact resolves Dim%d"
            % (args.dimension, artifact.resolved_dimension)
        )

    # 8. Contexte MPI éventuel, puis bind public du même artefact sur chaque rang.
    resources = {}
    rank = 0
    mpi_ranks = 1
    runtime_report = None
    mpi_topology_receipt = None
    if args.mpi:
        context = pops.ExecutionContext.mpi_world(artifact)
        resources["execution_context"] = context
        rank = int(context.communicator.handle.rank)
        mpi_ranks = int(context.communicator.handle.size)
    simulation = pops.bind(artifact, resources=resources, params=resolved_bind_params)
    runtime_report = dict(runtime_environment_report())
    if args.mpi:
        if mpi_ranks != int(np.prod(args.mpi_topology)):
            raise RuntimeError("native MPI rank count does not match the requested Cartesian topology")
        local_boxes = simulation.local_boxes("tracer")
        local_receipt = {
            "rank": rank,
            "mpi_compiled": runtime_report.get("mpi_compiled"),
            "mpi_active": runtime_report.get("mpi_active"),
            "mpi_ranks": runtime_report.get("mpi_ranks"),
            "local_boxes": [[list(lower), list(upper)] for lower, upper in local_boxes],
        }
        gathered = allgather_value(context.communicator.handle, local_receipt)
        if len(gathered) != mpi_ranks:
            raise RuntimeError("native MPI allgather did not return one ownership receipt per rank")
        if any(
            type(row) is not dict
            or row.get("rank") != index
            or row.get("mpi_compiled") is not True
            or row.get("mpi_active") is not True
            or row.get("mpi_ranks") != mpi_ranks
            for index, row in enumerate(gathered)
        ):
            raise RuntimeError("native MPI runtime facts are not active and coherent on every rank")
        occupancy = np.full(tuple(reversed(args.resolution)), -1, dtype=np.int64)
        for row in gathered:
            boxes = row["local_boxes"]
            if not isinstance(boxes, list) or not boxes:
                raise RuntimeError("every MPI rank must own an active local box")
            for bounds in boxes:
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or any(not isinstance(edge, list) or len(edge) != args.dimension for edge in bounds)
                ):
                    raise RuntimeError("native MPI local-box receipt has an invalid dimensional shape")
                lower, upper = bounds
                if any(type(value) is not int for edge in bounds for value in edge) or any(
                    lo < 0 or hi <= lo or hi > extent
                    for lo, hi, extent in zip(lower, upper, args.resolution, strict=True)
                ):
                    raise RuntimeError("native MPI local-box receipt is outside the exact domain")
                slices = tuple(slice(lo, hi) for lo, hi in zip(reversed(lower), reversed(upper), strict=True))
                if np.any(occupancy[slices] != -1):
                    raise RuntimeError("native MPI local-box ownership overlaps between ranks")
                occupancy[slices] = row["rank"]
        if np.any(occupancy == -1):
            raise RuntimeError("native MPI local-box ownership leaves part of the domain unowned")
        rank_coordinates = []
        for coordinate in np.ndindex(*args.mpi_topology):
            expected = np.zeros_like(occupancy, dtype=bool)
            lower = tuple(
                coordinate[index] * args.resolution[index] // args.mpi_topology[index]
                for index in range(args.dimension)
            )
            upper = tuple(
                (coordinate[index] + 1) * args.resolution[index] // args.mpi_topology[index]
                for index in range(args.dimension)
            )
            expected[tuple(slice(lo, hi) for lo, hi in zip(reversed(lower), reversed(upper), strict=True))] = True
            owners = [candidate for candidate in range(mpi_ranks) if np.array_equal(occupancy == candidate, expected)]
            if len(owners) != 1:
                raise RuntimeError(
                    "native MPI ownership does not realize the requested Cartesian decomposition"
                )
            rank_coordinates.append({"rank": owners[0], "coordinate": list(coordinate)})
        if len({row["rank"] for row in rank_coordinates}) != mpi_ranks:
            raise RuntimeError("native MPI Cartesian ownership maps one rank to multiple regions")
        corner_crossing = None
        if args.dimension == 3 and args.mpi_topology == (2, 2, 2):
            corner = tuple(args.resolution[index] // 2 for index in range(args.dimension))
            adjacent = []
            for coordinate in np.ndindex(*args.mpi_topology):
                rank_coordinate = next(
                    row["rank"] for row in rank_coordinates if row["coordinate"] == list(coordinate)
                )
                adjacent.append(rank_coordinate)
            start = WITNESS_REFERENCE_POINT[: args.dimension]
            arrival_times = tuple(
                (corner[index] / args.resolution[index] - start[index]) % 1.0
                / velocity[index]
                for index in range(args.dimension)
            )
            if len({round(value, 12) for value in arrival_times}) != 1:
                raise RuntimeError("diagonal characteristic does not meet the 3D ownership corner")
            corner_crossing = {
                "observed": True,
                "corner_index": list(corner),
                "corner_coordinate": [corner[index] / args.resolution[index] for index in range(args.dimension)],
                "participating_ranks": sorted(adjacent),
                "characteristic_start": list(start),
                "arrival_time": arrival_times[0],
                "velocity": list(velocity),
            }
            if corner_crossing["participating_ranks"] != list(range(8)):
                raise RuntimeError("3D MPI corner is not owned by eight distinct active ranks")
        mpi_topology_receipt = {
            "requested_ranks": int(np.prod(args.mpi_topology)),
            "observed_ranks": mpi_ranks,
            "expected_spatial_decomposition": list(args.mpi_topology),
            "ownership_active": True,
            "rank_ownership": [
                {"rank": row["rank"], "local_boxes": row["local_boxes"]} for row in gathered
            ],
            "rank_coordinates": rank_coordinates,
            "inter_rank_corner_crossing": corner_crossing,
        }
    execution = execution_provenance(runtime_report) if rank == 0 else None
    source = (
        source_provenance(
            repository_root=REPOSITORY_ROOT,
            generator_path=Path(__file__).resolve(),
            support_path=CASE_DIRECTORY / "_case_support.py",
            source_schema_version=SOURCE_SCHEMA_VERSION,
        )
        if rank == 0
        else None
    )

    # 9. Chronologie native, collectes collectives, métriques puis publication au rang zéro.
    timeline_times = tuple(
        float(value) for value in np.linspace(0.0, final_time, args.time_snapshots)
    )
    initial_mass = float(simulation.integral("tracer"))
    initial_snapshot = collect_snapshot(
        simulation,
        cells=args.resolution,
        velocity=velocity,
        waves=wave_numbers,
        layout=args.layout,
        time=0.0,
        epsilon=EPSILON,
        refinement_ratio=REFINEMENT_RATIO,
    )
    snapshots_by_time = {0.0: initial_snapshot}
    masses_by_time = {0.0: initial_mass}
    run_targets = sorted({*timeline_times[1:], PROBE_TIME})
    for target_time in run_targets:
        pops.run(simulation, t_end=target_time, max_steps=MAX_STEPS, console=False)
        masses_by_time[target_time] = float(simulation.integral("tracer"))
        snapshots_by_time[target_time] = collect_snapshot(
            simulation,
            cells=args.resolution,
            velocity=velocity,
            waves=wave_numbers,
            layout=args.layout,
            time=target_time,
            epsilon=EPSILON,
            refinement_ratio=REFINEMENT_RATIO,
        )

    probe_mass = masses_by_time[PROBE_TIME]
    probe_snapshot = snapshots_by_time[PROBE_TIME]
    final_mass = masses_by_time[final_time]
    final_snapshot = snapshots_by_time[final_time]
    timeline_snapshots = tuple(snapshots_by_time[time] for time in timeline_times)
    timeline_masses = tuple(masses_by_time[time] for time in timeline_times)

    # Toutes les intégrales/extractions ci-dessus sont collectives : aucun rang ne sort avant.
    if rank != 0:
        return
    metrics = compute_metrics(
        initial_snapshot,
        final_snapshot,
        probe_snapshot,
        cells=args.resolution,
        layout=args.layout,
        initial_mass=initial_mass,
        probe_mass=probe_mass,
        final_mass=final_mass,
        final_time=final_time,
        probe_phase_cycles=PROBE_TIME
        * sum(wave * speed for wave, speed in zip(wave_numbers, velocity, strict=True)),
        probe_time=PROBE_TIME,
        base_cfl=BASE_CFL,
        refinement_ratio=REFINEMENT_RATIO,
        history_times=timeline_times,
        history_snapshots=timeline_snapshots,
        history_masses=timeline_masses,
    )
    if execution is None or source is None:
        raise RuntimeError("rank zero did not capture execution/source provenance")
    coverage = {
        "requested_obligations": tuple(args.obligation),
        "mpi_topology": mpi_topology_receipt,
        "witnesses": coverage_witnesses(
            dimension=args.dimension,
            velocity=velocity,
            layout=args.layout,
            resolution=args.resolution,
            mode=args.mode,
            cycles=args.cycles,
            obligations=tuple(args.obligation),
            timeline_times=timeline_times,
            timeline_snapshots=timeline_snapshots,
            metrics=metrics,
            witness_reference_point=WITNESS_REFERENCE_POINT,
            patch_velocity=PATCH_VELOCITY,
            patch_center=PATCH_CENTER[: args.dimension],
            patch_half_width=PATCH_HALF_WIDTH[: args.dimension],
            wave_numbers=wave_numbers,
            final_time=final_time,
            refinement_ratio=REFINEMENT_RATIO,
        ),
    }

    patch_marker = (
        None
        if prescribed_patch is None
        else {
            "kind": "prescribed_window",
            "trajectory": "constant_velocity_layout_periodicity",
            "center": PATCH_CENTER[: args.dimension],
            "half_width": PATCH_HALF_WIDTH[: args.dimension],
            "velocity": patch_velocity,
            "clock": program.clock.qualified_id,
            "coarsen_outside": True,
        }
    )
    configuration = {
        "case": "periodic_sine_wave_advection",
        "dimension": args.dimension,
        "resolution": args.resolution,
        "mode": args.mode,
        "wave_numbers": wave_numbers,
        "velocity": velocity,
        "epsilon": EPSILON,
        "probe_time": PROBE_TIME,
        "period": FINAL_TIME,
        "cycles": args.cycles,
        "final_time": final_time,
        "layout": args.layout,
        "subcycling": args.subcycling,
        "block_size": args.block_size,
        "patch_marker": patch_marker,
        "coverage": coverage,
        "mpi": args.mpi,
        "mpi_ranks": mpi_ranks,
        "mpi_topology": args.mpi_topology,
    }
    artifact_identities = {
        "semantic_identity": artifact.semantic_identity.token,
        "artifact_identity": artifact.artifact_identity.token,
        "bind_identity": simulation.bind_identity.token,
    }
    data_path, metadata_path = publish_result(
        output=args.output,
        schema_version=SCHEMA_VERSION,
        pops_version=pops.__version__,
        configuration=configuration,
        metrics=metrics,
        execution=execution,
        source=source,
        artifact=artifact_identities,
        initial_snapshot=initial_snapshot,
        probe_snapshot=probe_snapshot,
        final_snapshot=final_snapshot,
        timeline_times=timeline_times,
        timeline_snapshots=timeline_snapshots,
        segmented_native_runs=len(run_targets),
    )
    print("data: %s" % data_path)
    print("metadata: %s" % metadata_path)
    if not metrics["qualification"]["run_integrity_passed"]:
        raise RuntimeError(
            "sine-wave run integrity failed: %s; data were retained for diagnosis"
            % metrics["qualification"]["reason"]
        )


if __name__ == "__main__":
    main()
