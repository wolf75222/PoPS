"""Generated native multi-block implicit phase contract for the M2 gate.

The case deliberately declares its runtime blocks as ``electrons, ions`` while the Program reads
them as ``ions, electrons``.  A correct generated package therefore has to use the exported
name-based block table; positional binding produces the wrong analytic result.
"""
from __future__ import annotations

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import Constant
from pops.math import Const, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.solvers import LocalNewton
from pops.time import CoupledImplicitEuler, FixedDt, RejectAttempt
from tests.python.support.requirements import repo_include


DT = 0.05
ELECTRON_INITIAL = 1.0
ION_INITIAL = 0.25
ELECTRON_FORCING = 0.5
ION_FORCING = -0.25
ELECTRON_RELAXATION = 2.0
ION_RELAXATION = 5.0


def _author_case(*, solve_count: int = 1, route_drift: bool = False):
    frame = Rectangle(
        "implicit-phase-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    model = pops.Model("implicit_phase_model", frame=frame)
    electrons = model.species("electrons", state=("ne",))
    ions = model.species("ions", state=("ni",))
    x_axis, y_axis = frame.axes
    electron_flux = model.flux(
        "electron_transport",
        frame=frame,
        state=electrons,
        components={
            x_axis: (0.0 * electrons["ne"],),
            y_axis: (0.0 * electrons["ne"],),
        },
        waves={
            x_axis: (Const(0.0),),
            y_axis: (Const(0.0),),
        },
    )
    ion_flux = model.flux(
        "ion_transport",
        frame=frame,
        state=ions,
        components={
            x_axis: (0.0 * ions["ni"],),
            y_axis: (0.0 * ions["ni"],),
        },
        waves={
            x_axis: (Const(0.0),),
            y_axis: (Const(0.0),),
        },
    )
    electron_transport = model.rate(
        "electron_transport_rate", equation=ddt(electrons) == -div(electron_flux)
    )
    ion_transport = model.rate(
        "ion_transport_rate", equation=ddt(ions) == -div(ion_flux)
    )
    model.source(
        "electron_forcing",
        on=electrons,
        value=(Const(ELECTRON_FORCING) + 0.0 * electrons["ne"],),
    )
    model.source(
        "ion_forcing",
        on=ions,
        value=(Const(ION_FORCING) + 0.0 * ions["ni"],),
    )
    electron_forcing = model.module.operator_handle("electron_forcing")
    ion_forcing = model.module.operator_handle("ion_forcing")
    collision = model.coupled_rate(
        "density_exchange",
        inputs=(electrons, ions),
        outputs={
            electrons: (
                Const(ELECTRON_RELAXATION) * (ions["ni"] - electrons["ne"]),
            ),
            ions: (
                Const(ION_RELAXATION) * (electrons["ne"] - ions["ni"]),
            ),
        },
    )

    case = pops.Case("multiblock_implicit_phase")
    # Case/runtime insertion order is the opposite of the Program declaration order below.
    electron_block = case.block("electrons", model, states=(electrons,))
    ion_block = case.block("ions", model, states=(ions,))
    electron_state = electron_block[electrons]
    ion_state = ion_block[ions]

    for block, state, flux, rate in (
        (electron_block, electrons, electron_flux, electron_transport),
        (ion_block, ions, ion_flux, ion_transport),
    ):
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

    program = pops.Program("one_multiblock_implicit_phase")
    ion_time = program.state(ion_state)
    electron_time = program.state(electron_state)
    if route_drift:
        # This is deliberately invalid: the typed electron/ion operator inputs cannot be swapped.
        program.solve(
            CoupledImplicitEuler(collision, (ion_time.n, electron_time.n)),
            solver=LocalNewton(),
            name="cross_block_collision",
        )

    ion_predictor = program.value(
        "ion_predictor",
        ion_time.n + program.dt * program.source(ion_forcing, state=ion_time.n),
        at=ion_time.next.point,
    )
    electron_predictor = program.value(
        "electron_predictor",
        electron_time.n
        + program.dt * program.source(electron_forcing, state=electron_time.n),
        at=electron_time.next.point,
    )
    solved = None
    for index in range(solve_count):
        solved = program.solve(
            CoupledImplicitEuler(
                collision,
                (electron_predictor, ion_predictor),
            ),
            solver=LocalNewton(
                tolerance=1.0e-12,
                max_iterations=12,
                finite_difference_step=1.0e-7,
            ),
            name="collision_phase_%d" % index,
        ).consume(action=RejectAttempt())
    if solved is None:
        raise ValueError("the fixture requires at least one implicit solve")
    program.commit_many(
        {
            electron_time.next: solved[electron_block],
            ion_time.next: solved[ion_block],
        }
    )
    program.step_strategy(FixedDt(DT))
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=electron_state,
            value=Constant((ELECTRON_INITIAL,)),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=ion_state,
            value=Constant((ION_INITIAL,)),
            projection=ConservativeCellAverage(),
        )
    )
    layout = Uniform(
        CartesianGrid(
            frame=frame,
            cells=(8, 8),
            periodic=PeriodicAxes(frame.axes),
        )
    )
    return case, layout, program


def _resolve(*, solve_count: int = 1, cxx: str | None = None):
    case, layout, _program = _author_case(solve_count=solve_count)
    compile_options = {"include": repo_include()}
    if cxx is not None:
        compile_options["cxx"] = cxx
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options=compile_options,
    )


def _require_exact_implicit_phase(program):
    """Fail closed unless every operator route and the one solve phase remain exact."""
    from pops.time.references import block_name

    values = tuple(program._values)
    sources = {
        (value.attrs["operator_handle"].name, block_name(value.block))
        for value in values
        if value.op == "source"
    }
    expected_sources = {
        ("ion_forcing", "ions"),
        ("electron_forcing", "electrons"),
    }
    if sources != expected_sources:
        raise AssertionError(
            "multi-block implicit operator routes diverged: %r != %r"
            % (sorted(sources), sorted(expected_sources))
        )
    solves = tuple(value for value in values if value.op == "solve_coupled_implicit")
    if len(solves) != 1:
        raise AssertionError(
            "multi-block implicit fixture must declare exactly one solve phase; got %d"
            % len(solves)
        )
    solve = solves[0]
    output_blocks = tuple(block_name(block) for block in solve.attrs["blocks"])
    input_blocks = tuple(block_name(value.block) for value in solve.inputs)
    if output_blocks != ("electrons", "ions") or input_blocks != ("electrons", "ions"):
        raise AssertionError(
            "multi-block implicit solve mapping diverged: outputs=%r inputs=%r"
            % (output_blocks, input_blocks)
        )
    return solve


def test_multiblock_implicit_fixture_refuses_route_or_phase_multiplicity_drift() -> None:
    with pytest.raises(ValueError, match="space|StateSpace|operator"):
        _author_case(route_drift=True)

    _case, _layout, duplicate_program = _author_case(solve_count=2)
    with pytest.raises(AssertionError, match="exactly one solve phase"):
        _require_exact_implicit_phase(duplicate_program)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_generated_native_multiblock_implicit_phase_uses_exact_name_routes(
    isolated_native_cache, native_cxx, kokkos_root
) -> None:
    del isolated_native_cache, kokkos_root
    resolved = _resolve(cxx=native_cxx)
    _require_exact_implicit_phase(resolved.time)

    artifact = pops.compile(resolved)
    artifact.verify()
    assert tuple(block.name for block in artifact.blocks) == ("electrons", "ions")
    assert artifact.program_block_routes == ((0, "ions"), (1, "electrons"))
    generated = artifact.program._generated_cpp
    assert isinstance(generated, str)
    assert generated.count('"solve_coupled_implicit"') == 2
    assert 'ctx.require_cartesian_generated_operator(1, "solve_coupled_implicit");' in generated
    assert 'ctx.require_cartesian_generated_operator(0, "solve_coupled_implicit");' in generated
    assert 'ctx.require_cartesian_generated_operator(0, "named_source");' in generated
    assert 'ctx.require_cartesian_generated_operator(1, "named_source");' in generated

    simulation = pops.bind(artifact)
    pops.run(simulation, t_end=DT, max_steps=1)
    electron = np.asarray(simulation.get_state("electrons"))
    ion = np.asarray(simulation.get_state("ions"))

    electron_predictor = ELECTRON_INITIAL + DT * ELECTRON_FORCING
    ion_predictor = ION_INITIAL + DT * ION_FORCING
    determinant = 1.0 + DT * (ELECTRON_RELAXATION + ION_RELAXATION)
    electron_expected = (
        (1.0 + DT * ION_RELAXATION) * electron_predictor
        + DT * ELECTRON_RELAXATION * ion_predictor
    ) / determinant
    ion_expected = (
        DT * ION_RELAXATION * electron_predictor
        + (1.0 + DT * ELECTRON_RELAXATION) * ion_predictor
    ) / determinant

    np.testing.assert_allclose(electron, electron_expected, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(ion, ion_expected, rtol=0.0, atol=2.0e-10)
    assert not np.allclose(electron, ion_expected)
    assert not np.allclose(ion, electron_expected)
