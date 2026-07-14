#!/usr/bin/env python3
"""Final executable PoPS target: two-fluid transport, electrostatics and collisions.

The script executes the complete public lifecycle with no mock, fallback or compatibility route:
typed authoring, explicit ``LayoutPlan``, resolved ``ConsumerGraph``, native compilation, HDF5 and
ParaView publication, independent file reopening, strict checkpoint/restart and bit-identical
continuation.  The two states, field context and collision join remain owner-qualified throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pops
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOperator,
    FieldOutput,
    FieldProviderContribution,
    FieldProviderPack,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.ir import Const, ValueExpr, Var
from pops.math import ddt, div, laplacian
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.physics import Density, Momentum
from pops.solvers.elliptic import GeometricMG
from pops.solvers.nonlinear import LocalNewton
from pops.time import CoupledImplicitEuler, Dense, FailRun, FixedDt, RejectAttempt


DEFAULT_CELLS = 8
DEFAULT_DT = 1.0e-3


@dataclass(frozen=True, slots=True)
class MultiphysicsAuthoring:
    """Typed declarations retained across every public lifecycle phase."""

    model: Any
    case: Any
    electron_space: Any
    ion_space: Any
    field_space: Any
    electron_block: Any
    ion_block: Any
    electron_state: Any
    ion_state: Any
    field: Any
    program: Any
    explicit_electrons: Any
    explicit_ions: Any
    collision: Any
    electron_numerics: Any
    ion_numerics: Any


@dataclass(frozen=True, slots=True)
class FinalMultiphysicsCase:
    """Validated authoring plus its explicit one-layout authority and runtime provider."""

    authoring: MultiphysicsAuthoring
    layout_plan: Any
    layout_handle: Any
    layout_provider: Any


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Exact restart-sensitive state captured only through public runtime accessors."""

    time: float
    macro_step: int
    states: dict[str, np.ndarray]
    fields: dict[str, np.ndarray]
    histories: dict[str, tuple[np.ndarray, ...]]
    program_hash: str
    consumer_graph_identity: str
    consumer_cursors: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    """Artifacts and exact snapshots proving the final example ran and restarted."""

    hdf5_path: Path
    paraview_path: Path
    checkpoint_path: Path
    hdf5_identity: str
    paraview_identity: str
    accepted: RuntimeSnapshot
    restored: RuntimeSnapshot
    continuous: RuntimeSnapshot
    restarted: RuntimeSnapshot


def build_authoring() -> MultiphysicsAuthoring:
    """Build the complete two-state transactional Program and accepted-side-effect graph."""

    frame = Rectangle("unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("electrostatic_two_fluid", frame=frame)
    electrons = model.species(
        "electrons",
        state=("ne", "pex", "pey"),
        roles={
            "ne": Density(),
            "pex": Momentum(axis=x_axis),
            "pey": Momentum(axis=y_axis),
        },
    )
    ions = model.species(
        "ions",
        state=("ni", "pix", "piy"),
        roles={
            "ni": Density(),
            "pix": Momentum(axis=x_axis),
            "piy": Momentum(axis=y_axis),
        },
    )
    fields = model.field(
        "electrostatic_fields",
        components=("potential", "electric_x", "electric_y"),
    )

    ne, pex, pey = electrons
    ni, pix, piy = ions
    electric_x, electric_y = Var("electric_x", "aux"), Var("electric_y", "aux")
    sound_speed_squared = Const(0.2)

    electron_flux = model.flux(
        "electron_transport_flux",
        frame=frame,
        state=electrons,
        components={
            x_axis: (pex, pex * pex / ne + sound_speed_squared * ne, pex * pey / ne),
            y_axis: (pey, pex * pey / ne, pey * pey / ne + sound_speed_squared * ne),
        },
        waves={
            x_axis: (Const(-1.0), Const(0.0), Const(1.0)),
            y_axis: (Const(-1.0), Const(0.0), Const(1.0)),
        },
    )
    ion_flux = model.flux(
        "ion_transport_flux",
        frame=frame,
        state=ions,
        components={
            x_axis: (pix, pix * pix / ni + sound_speed_squared * ni, pix * piy / ni),
            y_axis: (piy, pix * piy / ni, piy * piy / ni + sound_speed_squared * ni),
        },
        waves={
            x_axis: (Const(-1.0), Const(0.0), Const(1.0)),
            y_axis: (Const(-1.0), Const(0.0), Const(1.0)),
        },
    )
    electron_force = model.source(
        "electron_electric_force",
        on=electrons,
        fields=fields,
        value=(Const(0.0), -ne * electric_x, -ne * electric_y),
    )
    ion_force = model.source(
        "ion_electric_force",
        on=ions,
        fields=fields,
        value=(Const(0.0), ni * electric_x, ni * electric_y),
    )
    explicit_electrons = model.rate(
        "explicit_electrons",
        equation=ddt(electrons) == -div(electron_flux) + electron_force,
    )
    explicit_ions = model.rate(
        "explicit_ions",
        equation=ddt(ions) == -div(ion_flux) + ion_force,
    )

    electron_numerics = DiscretizationPlan()
    electron_numerics.rates.add(
        explicit_electrons,
        FiniteVolume(
            flux=electron_flux,
            variables=variables.Conservative(electrons),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    ion_numerics = DiscretizationPlan()
    ion_numerics.rates.add(
        explicit_ions,
        FiniteVolume(
            flux=ion_flux,
            variables=variables.Conservative(ions),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )

    # Each model-owned provider contributes one signed source-density term to the same Poisson RHS.
    electron_charge = model.field_provider(
        "electron_charge", on=electrons, into=fields, value=-ne,
    )
    ion_charge = model.field_provider(
        "ion_charge", on=ions, into=fields, value=ni,
    )

    exchange_x = Const(2.0) * (pix - pex)
    exchange_y = Const(2.0) * (piy - pey)
    collision = model.coupled_rate(
        "implicit_collision",
        inputs=(electrons, ions),
        outputs={
            electrons: (Const(0.0), exchange_x, exchange_y),
            ions: (Const(0.0), -exchange_x, -exchange_y),
        },
    )

    case = pops.Case("two_fluid_transport_field_collision")
    electron_block = case.block("electrons", model, states=(electrons,))
    ion_block = case.block("ions", model, states=(ions,))
    case.numerics(electron_numerics, block=electron_block)
    case.numerics(ion_numerics, block=ion_block)
    electron_state = electron_block[electrons]
    ion_state = ion_block[ions]

    # The field declaration is model-owned; the Case qualification chooses the electron block as
    # its storage anchor.  The physical RHS below remains a genuine two-block provider pack.
    unknown = electron_block[fields]
    field_operator = FieldOperator(
        "electrostatic",
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == (-ne + ni),
        providers=FieldProviderPack(
            (
                FieldProviderContribution(electron_block[electron_charge]),
                FieldProviderContribution(ion_block[ion_charge]),
            )
        ),
        outputs=(
            FieldOutput("potential", unknown),
            GradientOutput("electric", unknown, sign=-1),
        ),
    )
    field_discretization = FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
        solver=GeometricMG(),
        nullspace=ConstantNullspace(),
        gauge=MeanValueGauge(0.0),
    )
    field = case.field(field_operator, field_discretization)

    program = pops.Program("transport_field_collision")
    electron_time = program.state(electron_state)
    ion_time = program.state(ion_state)
    program.keep_history(electron_time, depth=1, checkpoint_policy=Dense())
    program.keep_history(ion_time, depth=1, checkpoint_policy=Dense())

    exact_fields = field(
        electron_time.n, ion_time.n, name="electrostatic_at_n"
    ).consume(action=FailRun())
    electron_rate = explicit_electrons(
        electron_time.n, exact_fields, name="electron_explicit_rate")
    ion_rate = explicit_ions(ion_time.n, exact_fields, name="ion_explicit_rate")
    electron_predictor = program.value(
        "electron_predictor",
        electron_time.n + program.dt * electron_rate,
        at=electron_time.next.point,
    )
    ion_predictor = program.value(
        "ion_predictor",
        ion_time.n + program.dt * ion_rate,
        at=ion_time.next.point,
    )

    collided = program.solve(
        CoupledImplicitEuler(
            collision,
            (electron_predictor, ion_predictor),
            at={
                electron_block: electron_time.next.point,
                ion_block: ion_time.next.point,
            },
        ),
        solver=LocalNewton(
            tolerance=1.0e-11,
            max_iterations=12,
            finite_difference_step=1.0e-7,
        ),
        name="collision_at_next",
    ).consume(action=RejectAttempt())
    program.commit_many(
        {
            electron_time.next: collided[electron_block],
            ion_time.next: collided[ion_block],
        }
    )
    program.step_strategy(FixedDt(DEFAULT_DT))
    case.program(program)

    from pops.output import Checkpoint, HDF5, ParaView, ScientificOutput
    from pops.runtime import ConsumerGraph
    from pops.time import every, on_end, on_start

    case.consumers(ConsumerGraph.from_consumers((
        ScientificOutput(
            format=ParaView(),
            schedule=on_start(clock=program.clock),
            fields=(electron_state, ion_state),
            target="visualization/two_fluid.vtu",
        ),
        ScientificOutput(
            format=HDF5(parallel=False),
            schedule=on_end(clock=program.clock),
            fields=(electron_state, ion_state),
            target="state/two_fluid.h5",
        ),
        Checkpoint(
            schedule=every(100, clock=program.clock),
            target="checkpoints/restart",
            bit_identical=True,
        ),
    )))

    return MultiphysicsAuthoring(
        model=model,
        case=case,
        electron_space=electrons,
        ion_space=ions,
        field_space=fields,
        electron_block=electron_block,
        ion_block=ion_block,
        electron_state=electron_state,
        ion_state=ion_state,
        field=field,
        program=program,
        explicit_electrons=explicit_electrons,
        explicit_ions=explicit_ions,
        collision=collision,
        electron_numerics=electron_numerics,
        ion_numerics=ion_numerics,
    )


def build_final_case(*, cells: int = DEFAULT_CELLS) -> FinalMultiphysicsCase:
    """Validate all declarations, then assign every block/state/field exactly once."""

    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 4:
        raise ValueError("cells must be an integer >= 4")
    from pops.mesh import CartesianMesh, LayoutPlanBuilder
    from pops.layouts import Uniform

    authoring = build_authoring()
    pops.validate(authoring.case)
    subjects = authoring.case.layout_subjects()
    provider = Uniform(CartesianMesh(n=cells, L=1.0, periodic=True))
    builder = LayoutPlanBuilder(authoring.case.owner_path.canonical())
    layout = builder.layout("uniform_two_fluid", provider)
    for block in subjects.blocks:
        builder.assign_block(block, layout)
    for state in subjects.states:
        builder.assign_state(state, layout)
    for field in subjects.fields:
        builder.assign_field(field, layout)
    plan = builder.resolve(**subjects.to_dict())
    return FinalMultiphysicsCase(authoring, plan, layout, provider)


def build_initial_state(*, cells: int = DEFAULT_CELLS) -> dict[str, np.ndarray]:
    """Create positive, neutral two-fluid data without selecting any resolved semantics."""

    coordinate = (np.arange(cells, dtype=np.float64) + 0.5) / float(cells)
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    density = 1.0 + 0.05 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    momentum_x = 0.08 * density
    momentum_y = -0.03 * density
    state = np.stack((density, momentum_x, momentum_y))
    return {
        "electrons": state.copy(),
        "ions": state.copy(),
    }


def build_initial_fields(*, cells: int = DEFAULT_CELLS) -> dict[str, np.ndarray]:
    """Allocate the declared field-space buffers; the Program solve supplies their values."""

    zeros = np.zeros((cells, cells), dtype=np.float64)
    return {
        "potential": zeros.copy(),
        "electric_x": zeros.copy(),
        "electric_y": zeros.copy(),
    }


def compile_final_case(*, cells: int = DEFAULT_CELLS) -> tuple[FinalMultiphysicsCase, Any]:
    """Resolve and compile the exact final Case with its authenticated layout provider."""

    from pops.codegen import Production

    target = build_final_case(cells=cells)
    resolved = pops.resolve(
        target.authoring.case,
        layout=target.layout_plan,
        layout_providers={target.layout_handle: target.layout_provider},
        backend=Production(),
    )
    return target, pops.compile(resolved)


def _snapshot(simulation: Any) -> RuntimeSnapshot:
    """Capture every checkpoint-sensitive quantity needed by the acceptance proof."""

    states = {
        block: np.asarray(simulation.state_global(block), dtype=np.float64).copy()
        for block in simulation.block_names()
    }
    fields = {
        slot: np.asarray(simulation.field_potential_global(slot), dtype=np.float64).copy()
        for slot in simulation.field_provider_slots()
    }
    histories = {
        name: tuple(
            np.asarray(simulation.history_global(name, slot), dtype=np.float64).copy()
            for slot in range(int(simulation.history_depth(name)))
        )
        for name in simulation.history_names()
    }
    return RuntimeSnapshot(
        time=float(simulation.time()),
        macro_step=int(simulation.macro_step()),
        states=states,
        fields=fields,
        histories=histories,
        program_hash=str(simulation.installed_program_hash()),
        consumer_graph_identity=simulation.consumer_graph.identity.token,
        consumer_cursors=simulation.consumer_cursors.to_data(),
    )


def _require_same_snapshot(left: RuntimeSnapshot, right: RuntimeSnapshot, *, where: str) -> None:
    """Fail closed on any restart drift, including identities and hidden history rings."""

    scalar_pairs = {
        "time": (left.time, right.time),
        "macro_step": (left.macro_step, right.macro_step),
        "program_hash": (left.program_hash, right.program_hash),
        "consumer_graph_identity": (
            left.consumer_graph_identity, right.consumer_graph_identity),
        "consumer_cursors": (left.consumer_cursors, right.consumer_cursors),
    }
    for name, (expected, actual) in scalar_pairs.items():
        if expected != actual:
            raise RuntimeError("%s changed %s across restart" % (where, name))
    for category in ("states", "fields", "histories"):
        expected = getattr(left, category)
        actual = getattr(right, category)
        if tuple(expected) != tuple(actual):
            raise RuntimeError("%s changed qualified %s across restart" % (where, category))
        for name in expected:
            expected_rows = expected[name] if category == "histories" else (expected[name],)
            actual_rows = actual[name] if category == "histories" else (actual[name],)
            if len(expected_rows) != len(actual_rows) or any(
                    not np.array_equal(a, b) for a, b in zip(expected_rows, actual_rows)):
                raise RuntimeError("%s changed %s %s across restart" % (where, category, name))


def run_and_restart(
    output_dir: Any,
    *,
    cells: int = DEFAULT_CELLS,
) -> ExecutionEvidence:
    """Run, publish, reopen, checkpoint, restore independently and continue bit-identically."""

    from pops.output.writers import read_hdf5, read_paraview

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _target, artifact = compile_final_case(cells=cells)
    simulation = pops.bind(
        artifact,
        initial_state=build_initial_state(cells=cells),
        aux=build_initial_fields(cells=cells),
    )
    if pops.run(
            simulation,
            t_end=DEFAULT_DT, max_steps=1, output_dir=root / "accepted") != 1:
        raise RuntimeError("the accepted segment did not execute exactly one macro-step")

    hdf5_path = root / "accepted" / "two_fluid.h5"
    paraview_path = root / "accepted" / "two_fluid.vtu"
    hdf5 = read_hdf5(hdf5_path)
    paraview = read_paraview(paraview_path)
    checkpoint_path = Path(simulation.checkpoint(root / "accepted_restart"))
    accepted = _snapshot(simulation)

    resumed = pops.bind(
        artifact,
        initial_state=build_initial_state(cells=cells),
        aux=build_initial_fields(cells=cells),
    )
    resumed.restart(checkpoint_path)
    restored = _snapshot(resumed)
    _require_same_snapshot(accepted, restored, where="independent reopen")
    if simulation.bind_identity != resumed.bind_identity:
        raise RuntimeError("fresh bind changed the authenticated install identity")
    if resumed.last_restart_identity is None:
        raise RuntimeError("restart did not publish an authenticated checkpoint identity")

    final_time = 2.0 * DEFAULT_DT
    pops.run(
        simulation,
        t_end=final_time, max_steps=1, output_dir=root / "continuous")
    pops.run(
        resumed,
        t_end=final_time, max_steps=1, output_dir=root / "restarted")
    continuous, restarted = _snapshot(simulation), _snapshot(resumed)
    _require_same_snapshot(continuous, restarted, where="bit-identical continuation")

    return ExecutionEvidence(
        hdf5_path=hdf5_path,
        paraview_path=paraview_path,
        checkpoint_path=checkpoint_path,
        hdf5_identity=hdf5.output_identity.token,
        paraview_identity=paraview.output_identity.token,
        accepted=accepted,
        restored=restored,
        continuous=continuous,
        restarted=restarted,
    )


def main() -> None:
    """Execute the complete final lifecycle and print only its authenticated evidence."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multiphysics"))
    args = parser.parse_args()

    evidence = run_and_restart(args.output_dir, cells=args.cells)
    print("PoPS final multiphysics acceptance:")
    print("  HDF5: %s" % evidence.hdf5_identity)
    print("  ParaView: %s" % evidence.paraview_identity)
    print("  checkpoint: %s" % evidence.checkpoint_path)
    print("  bit-identical restart: step %d" % evidence.restarted.macro_step)


if __name__ == "__main__":
    main()
