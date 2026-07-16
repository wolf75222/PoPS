"""A typed RuntimeParam crosses bind and changes native execution without recompilation.

One final Production artifact is bound twice with two owner-qualified values.  Its fixed-step
Forward Euler Program advances a spatially uniform scalar according to ``du/dt = -k*u``; therefore
the exact one-step oracle is ``u1 = u0 * (1 - k*dt)``.  Both runs use the same binaries, whose bytes
and timestamps are checked before and after bind/execution.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import pops
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
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.fields import (
    CellCenteredSecondOrder,
    CompositeHierarchySolve,
    FieldDiscretization,
    FieldOutput,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import EllipticRecompute, StateTransfer
from pops.lib.initial import Gaussian
from pops.lib.time import ForwardEuler
from pops.math import ValueExpr, ddt, div, laplacian, unknown
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.solvers.elliptic import GeometricMG
from pops.time import FailRun, FixedDt, every


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
ROOT = Path(__file__).resolve().parents[4]
# The file deliberately compiles three distinct resolved plans (uniform runtime source plus
# uniform/AMR named-field routes).  Keep process isolation below the 30-minute workflow bound while
# allowing a cold compiler cache on slower runners.
POPS_PROCESS_TIMEOUT = 1200

N = 8
DT = 1.0e-2
INITIAL_VALUE = 3.0


def _resolved_runtime_parameter_case():
    frame = Rectangle(
        "runtime-parameter-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("runtime-parameter-model", frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    declaration = model.param(RuntimeParam("decay_rate", default=1.0))
    decay_rate = model.value(declaration)
    flux = model.flux(
        "zero_transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    source = model.source("decay", on=state, value=(-decay_rate * rho,))
    rate = model.rate(
        "explicit_rhs", equation=ddt(state) == -div(flux) + source
    )

    case = pops.Case("runtime-parameter-case")
    block = case.block("scalar", model)
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
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    validated = pops.validate(case)
    bound_parameter = validated.resolve(declaration)
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    resolved = pops.resolve(
        validated,
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )
    return resolved, bound_parameter


def _initial_field_state() -> np.ndarray:
    coordinates = (np.arange(N, dtype=np.float64) + 0.5) / N
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    density = 0.1 + 0.9 * np.exp(-60.0 * ((xx - 0.3) ** 2 + (yy - 0.4) ** 2))
    return density.reshape(1, N, N)


def _resolved_named_field_runtime_parameter_case(*, target: str):
    """Resolve one named-field-only RuntimeParam route for an exact native target."""
    if target not in {"system", "amr_system"}:
        raise ValueError("target must be 'system' or 'amr_system'")

    frame = Rectangle(
        "named-field-runtime-%s-domain" % target,
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("named-field-runtime-%s-model" % target, frame=frame)
    state = model.state("U", components=("rho",))
    (rho,) = state
    parameter = model.param(RuntimeParam("named_rhs_scale", default=1.0))
    scale = model.value(parameter)
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
        waves={x_axis: (0.0 * rho,), y_axis: (0.0 * rho,)},
    )
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux))
    potential = model.field("potential")
    phi = unknown(potential)
    field_operator = model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=-laplacian(phi) + 1.0 * phi == scale * rho,
        outputs=(FieldOutput("phi", potential),),
    )

    case = pops.Case("named-field-runtime-%s-case" % target)
    block = case.block("scalar", model)
    state_instance = block[state]
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
    case.numerics(numerics, block=block)
    field_instance = case.field(
        field_operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=GeometricMG(),
            hierarchy_policy=(
                CompositeHierarchySolve() if target == "amr_system" else None
            ),
        ),
    )
    program = ForwardEuler(
        state_instance,
        rate=rate,
        fields=field_instance,
        solve_action=FailRun(),
    )
    program.step_strategy(FixedDt(DT))
    case.program(program)

    if target == "system":
        layout = Uniform(CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ))
    else:
        case.initials.add(InitialCondition(
            state=state_instance,
            value=Gaussian(
                frame=frame,
                center={frame.x: 0.3, frame.y: 0.4},
                background=0.1,
                amplitude=0.9,
                inverse_width=60.0,
            ),
            projection=ConservativeCellAverage(),
        ))
        refine_threshold = case.param(
            # The analytic Gaussian is bounded by 1.0.  A 0.5 threshold selects a compact,
            # deterministic interior patch on the 8x8 parent grid (and leaves untagged coarse
            # cells), so this test exercises a genuine two-level field solve instead of asking the
            # bootstrap to create a level from an identically-empty tag mask.
            RuntimeParam("named_field_refine_threshold", default=0.5)
        )
        transfer = AMRTransfer()
        transfer.state(state_instance, StateTransfer())
        transfer.field(field_instance, EllipticRecompute())
        layout = AMR(
            grid=CartesianGrid(frame=frame, cells=(N, N)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(
                rules=(
                    Tag(ValueExpr(state_instance) > case.value(refine_threshold)),
                    Buffer(cells=1),
                ),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS,
            ),
            regrid=AMRRegrid(schedule=every(2, clock=program.clock)),
            transfer=transfer,
            execution=AMRExecution.synchronous(),
        )

    validated = pops.validate(case)
    bound_parameter = validated.resolve(parameter)
    resolved = pops.resolve(
        validated,
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include")},
    )
    return resolved, bound_parameter


def _binary_paths(artifact) -> tuple[Path, ...]:
    paths = tuple(Path(block.model.so_path) for block in artifact.blocks)
    paths += tuple(Path(path) for path in artifact.layout_program_paths.values())
    assert paths and len(paths) == len(set(paths)) and all(path.is_file() for path in paths)
    return paths


def _binary_fingerprints(paths: tuple[Path, ...]):
    result = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stat = path.stat()
        result[path] = (digest, stat.st_size, stat.st_mtime_ns)
    return result


def test_runtime_parameter_bind_values_drive_native_execution_without_recompile(
    isolated_native_cache, native_cxx, kokkos_root
):
    del isolated_native_cache, native_cxx, kokkos_root
    resolved, parameter = _resolved_runtime_parameter_case()
    artifact = pops.compile(resolved)
    artifact.verify()
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)
    initial = np.full((1, N, N), INITIAL_VALUE, dtype=np.float64)

    values = (1.25, 4.0)
    instances = tuple(
        pops.bind(
            artifact,
            initial_state={"scalar": initial.copy()},
            params={parameter: value},
        )
        for value in values
    )
    reports = tuple(
        pops.run(instance, t_end=DT, max_steps=1) for instance in instances
    )

    for value, instance, report in zip(values, instances, reports, strict=True):
        assert report.accepted_steps == 1
        assert report.final_time == DT
        actual = np.asarray(instance.get_state("scalar"), dtype=np.float64).reshape(1, N, N)
        expected = np.full_like(actual, INITIAL_VALUE * (1.0 - value * DT))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)

    first = np.asarray(instances[0].get_state("scalar"), dtype=np.float64)
    second = np.asarray(instances[1].get_state("scalar"), dtype=np.float64)
    assert not np.array_equal(first, second)
    assert _binary_fingerprints(paths) == before


@pytest.mark.parametrize(
    "target",
    ("system", "amr_system"),
    ids=("uniform", "amr"),
)
def test_named_elliptic_runtime_parameter_binds_into_native_rhs_without_recompile(
    target, isolated_native_cache, native_cxx, kokkos_root
):
    """The shared BindSchema carrier reaches a standalone named elliptic RHS brick."""
    del isolated_native_cache, native_cxx, kokkos_root
    resolved, parameter = _resolved_named_field_runtime_parameter_case(target=target)
    artifact = pops.compile(resolved)
    artifact.verify()
    assert artifact.plan.target == target
    parameter_slot = artifact.bind_schema.slot(parameter)
    assert parameter_slot.kind == "runtime"
    assert parameter_slot.handle == parameter
    paths = _binary_paths(artifact)
    before = _binary_fingerprints(paths)

    values = (0.75, 2.25)
    simulations = []
    for value in values:
        bind_inputs = {"params": {parameter: value}}
        if target == "system":
            bind_inputs["initial_state"] = {"scalar": _initial_field_state()}
        simulations.append(pops.bind(artifact, **bind_inputs))
    assert simulations[0].bind_identity != simulations[1].bind_identity

    centered_potentials = []
    provider_slots = []
    for simulation in simulations:
        if target == "amr_system":
            assert simulation.n_levels() == 2
        report = pops.run(simulation, t_end=DT, max_steps=1)
        assert report.accepted_steps == 1
        assert report.final_time == DT
        slots = simulation.field_provider_slots()
        assert len(slots) == 1
        if target == "amr_system":
            assert simulation.field_provider_levels(slots[0]) == 2
        provider_slots.append(slots[0])
        potential_values = np.asarray(
            simulation.field_potential_global(slots[0]), dtype=np.float64
        ).reshape(-1)
        assert potential_values.size > 0 and np.all(np.isfinite(potential_values))
        centered = potential_values - potential_values.mean()
        assert np.max(np.abs(centered)) > 1.0e-8
        centered_potentials.append(centered)

    assert provider_slots[0] == provider_slots[1]
    assert not np.array_equal(centered_potentials[0], centered_potentials[1])
    np.testing.assert_allclose(
        centered_potentials[1],
        (values[1] / values[0]) * centered_potentials[0],
        rtol=5.0e-5,
        atol=5.0e-8,
    )
    assert _binary_fingerprints(paths) == before
