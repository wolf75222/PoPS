#!/usr/bin/env python3
"""Final executable PoPS target: 15-moment Vlasov--Poisson--Lorentz dynamics.

The provided HyQMOM15 factory returns an ordinary ``Model``.  This script composes its exact
transport and magnetic operators with a model-owned Poisson field, ordinary spatial numerics and
an ordinary IMEX ``Program``.  It then executes the sole public lifecycle, reopens scientific
artifacts and proves an independently rebound checkpoint continues bit-identically.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.layouts import Uniform
from pops.lib.models.moments import HyQMOM15
from pops.lib.time import IMEX
from pops.math import laplacian
from pops.mesh import CartesianMesh
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.output import Checkpoint, HDF5, ParaView, ScientificOutput
from pops.params import ConstParam
from pops.runtime import ConsumerGraph
from pops.solvers.elliptic import GeometricMG
from pops.time import AdaptiveCFL, RejectAttempt, every


DEFAULT_CELLS = 8
DEFAULT_T_END = 1.0e-5


@dataclass(frozen=True, slots=True)
class HyQMOM15Authoring:
    """All exact declarations retained across the public lifecycle."""

    model: Any
    case: Any
    state: Any
    state_instance: Any
    field: Any
    field_instance: Any
    explicit_rate: Any
    implicit_operator: Any
    field_provider: Any
    program: Any
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Restart-sensitive public state used by the acceptance proof."""

    time: float
    macro_step: int
    state: np.ndarray
    fields: dict[str, np.ndarray]
    program_hash: str
    consumer_graph_identity: str
    consumer_cursors: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Scientific artifacts and exact states produced by one final execution."""

    hdf5_path: Path
    paraview_path: Path
    scheduled_checkpoint_path: Path
    manual_checkpoint_path: Path
    hdf5_identity: str
    paraview_identity: str
    accepted: RuntimeSnapshot
    restored: RuntimeSnapshot
    continuous: RuntimeSnapshot
    restarted: RuntimeSnapshot


def build_authoring() -> HyQMOM15Authoring:
    """Compose provided physics with generic field, numerical and time interfaces."""

    model = HyQMOM15.vlasov_lorentz(
        q_over_m=ConstParam("q_over_m", -1.0),
        omega_c=ConstParam("omega_c", 0.5),
    )
    state = model.states["U"]

    # A fixed unit ion background makes the periodic Poisson right-hand side neutral.  The output
    # names are the canonical electrostatic FieldContext consumed by the provided electric source.
    density = state["M00"]
    potential = model.field("electrostatic_potential")
    field_operator = model.field_operator(
        "fields",
        unknown=potential,
        equation=(-laplacian(potential) == density - 1.0),
        outputs=(
            FieldOutput("phi", potential),
            GradientOutput("grad", potential, sign=-1),
        ),
    )
    field_provider = next(iter(field_operator.providers)).provider
    # The added field completes the model's FieldSpace. Read all operator handles only after this
    # final authoring mutation so their immutable signatures authenticate that settled space.
    flux = model.fluxes["transport"]
    explicit_rate = model.operators["transport"]
    implicit_operator = model.operators["magnetic_rotation"]

    numerics = DiscretizationPlan()
    numerics.rates.add(
        explicit_rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.MUSCL(limiters.VanLeer()),
            riemann=riemann.HLL(waves=riemann.waves.ExplicitPair()),
        ),
    )

    case = pops.Case("hyqmom15_vlasov_poisson_lorentz")
    plasma = case.block("plasma", model)
    state_instance = plasma[state]
    case.numerics(numerics, block=plasma)
    field_instance = case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=GeometricMG(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )

    program = IMEX(
        state_instance,
        explicit_operator=explicit_rate,
        implicit_operator=implicit_operator,
        fields_operator=field_provider,
        solve_action=RejectAttempt(),
    )
    program.step_strategy(AdaptiveCFL(0.35))
    case.program(program)

    schedule = every(1, clock=program.clock)
    case.consumers(ConsumerGraph.from_consumers((
        ScientificOutput(
            format=HDF5(), schedule=schedule,
            fields=(state_instance, field_instance), target="state/hyqmom15.h5"),
        ScientificOutput(
            format=ParaView(), schedule=schedule,
            fields=(state_instance, field_instance), target="visualization/hyqmom15.vtu"),
        Checkpoint(
            schedule=schedule, target="checkpoints/hyqmom15", bit_identical=True),
    )))
    return HyQMOM15Authoring(
        model=model,
        case=case,
        state=state,
        state_instance=state_instance,
        field=potential,
        field_instance=field_instance,
        explicit_rate=explicit_rate,
        implicit_operator=implicit_operator,
        field_provider=field_provider,
        program=program,
        components=tuple(state.components),
    )


def build_initial_state(*, cells: int = DEFAULT_CELLS) -> dict[str, np.ndarray]:
    """Return a smooth realizable Gaussian moment field with exactly neutral mean density."""

    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) / float(cells)
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    density = 1.0 + 0.01 * np.sin(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    state = np.zeros((len(HyQMOM15.components), cells, cells), dtype=np.float64)
    component = {name: index for index, name in enumerate(HyQMOM15.components)}
    for name, coefficient in {
        "M00": 1.0,
        "M20": 0.1,
        "M02": 0.1,
        "M40": 0.03,
        "M22": 0.01,
        "M04": 0.03,
    }.items():
        state[component[name]] = coefficient * density
    return {"plasma": state}


def compile_final_case(*, cells: int = DEFAULT_CELLS) -> tuple[HyQMOM15Authoring, Any, Any]:
    """Run validate, resolve and compile without any alternate runtime route."""

    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 4:
        raise ValueError("cells must be an integer >= 4")
    target = build_authoring()
    validated = pops.validate(target.case)
    resolved = pops.resolve(
        validated,
        layout=Uniform(CartesianMesh(n=cells, L=1.0, periodic=True)),
    )
    return target, resolved, pops.compile(resolved)


def _snapshot(simulation: Any) -> RuntimeSnapshot:
    fields = {
        slot: np.asarray(simulation.field_potential_global(slot), dtype=np.float64).copy()
        for slot in simulation.field_provider_slots()
    }
    return RuntimeSnapshot(
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
        state=np.asarray(simulation.get_state("plasma"), dtype=np.float64).copy(),
        fields=fields,
        program_hash=str(simulation.installed_program_hash()),
        consumer_graph_identity=simulation.consumer_graph.identity.token,
        consumer_cursors=simulation.consumer_cursors.to_data(),
    )


def _require_same_snapshot(left: RuntimeSnapshot, right: RuntimeSnapshot, *, where: str) -> None:
    for name in (
        "time", "macro_step", "program_hash", "consumer_graph_identity", "consumer_cursors",
    ):
        if getattr(left, name) != getattr(right, name):
            raise RuntimeError("%s changed %s across restart" % (where, name))
    if not np.array_equal(left.state, right.state):
        raise RuntimeError("%s changed the 15-moment state across restart" % where)
    if tuple(left.fields) != tuple(right.fields) or any(
        not np.array_equal(left.fields[name], right.fields[name]) for name in left.fields
    ):
        raise RuntimeError("%s changed the solved electrostatic field across restart" % where)


def _one_artifact(root: Path, suffix: str) -> Path:
    paths = tuple(sorted(root.rglob("*%s" % suffix)))
    if not paths:
        raise RuntimeError("accepted run did not publish a %s artifact under %s" % (suffix, root))
    return paths[-1]


def run_and_restart(
    output_dir: Any,
    *,
    cells: int = DEFAULT_CELLS,
) -> ExecutionEvidence:
    """Run once, reopen artifacts, restore independently and continue exactly."""

    from pops.output import read_hdf5, read_paraview

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _target, _resolved, artifact = compile_final_case(cells=cells)
    initial = build_initial_state(cells=cells)
    simulation = pops.bind(artifact, initial_state=initial)
    accepted_root = root / "accepted"
    if pops.run(
        simulation, t_end=DEFAULT_T_END, max_steps=1, output_dir=accepted_root,
    ) != 1:
        raise RuntimeError("the accepted segment did not execute exactly one macro-step")

    hdf5_path = _one_artifact(accepted_root, ".h5")
    paraview_path = _one_artifact(accepted_root, ".vtu")
    scheduled_checkpoint_path = _one_artifact(accepted_root, ".npz")
    hdf5 = read_hdf5(hdf5_path)
    paraview = read_paraview(paraview_path)
    if not hdf5.arrays or not paraview.arrays:
        raise RuntimeError("scientific outputs reopened without arrays")
    if not all(np.isfinite(value).all() for value in hdf5.arrays.values()):
        raise RuntimeError("HDF5 output contains a non-finite value")

    manual_checkpoint_path = Path(simulation.checkpoint(root / "manual_restart"))
    accepted = _snapshot(simulation)
    resumed = pops.bind(artifact, initial_state=build_initial_state(cells=cells))
    resumed.restart(manual_checkpoint_path)
    restored = _snapshot(resumed)
    _require_same_snapshot(accepted, restored, where="independent reopen")
    if resumed.last_restart_identity is None:
        raise RuntimeError("restart did not expose an authenticated identity")

    final_time = 2.0 * DEFAULT_T_END
    pops.run(
        simulation, t_end=final_time, max_steps=1, output_dir=root / "continuous")
    pops.run(
        resumed, t_end=final_time, max_steps=1, output_dir=root / "restarted")
    continuous, restarted = _snapshot(simulation), _snapshot(resumed)
    _require_same_snapshot(continuous, restarted, where="bit-identical continuation")

    return ExecutionEvidence(
        hdf5_path=hdf5_path,
        paraview_path=paraview_path,
        scheduled_checkpoint_path=scheduled_checkpoint_path,
        manual_checkpoint_path=manual_checkpoint_path,
        hdf5_identity=hdf5.output_identity.token,
        paraview_identity=paraview.output_identity.token,
        accepted=accepted,
        restored=restored,
        continuous=continuous,
        restarted=restarted,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/hyqmom15"),
        help="directory receiving accepted scientific output and restart artifacts",
    )
    args = parser.parse_args(argv)
    evidence = run_and_restart(args.output_dir, cells=args.cells)
    print("HDF5: %s" % evidence.hdf5_path)
    print("ParaView: %s" % evidence.paraview_path)
    print("checkpoint: %s" % evidence.manual_checkpoint_path)
    print("bit-identical restart: True")
    print("report: " + json.dumps({
        "finite": bool(np.isfinite(evidence.restarted.state).all()),
        "n_moments": int(evidence.restarted.state.shape[0]),
        "runtime_steps": evidence.restarted.macro_step,
        "runtime_time": evidence.restarted.time,
        "scheduled_checkpoint": str(evidence.scheduled_checkpoint_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
