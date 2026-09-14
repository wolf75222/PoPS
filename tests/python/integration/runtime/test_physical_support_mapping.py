"""ADC-944 public physical map lifecycle: native Dim2 storage for 1x1v/1x."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
import pops
from pops.model import PhysicalDimension, PhysicalSupport
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.layouts import Uniform
from pops.math import Const, ddt, div, grad, laplacian
from pops.mesh import (CartesianGrid, PeriodicAxes, LayoutPlanBuilder, LayoutMappingOperation,
    LayoutRepresentation, LayoutSynchronization, PhysicalSupportMap, VelocityQuadrature,
    native_physical_mapping)
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.numerics.terms import Flux, SourceTerm
from pops.solvers import CG
from pops.time import FailRun, FixedDt, RejectAttempt
from tests.python.support.native_execution_context import artifact_execution_context

ROOT = Path(__file__).resolve().parents[4]
DT = 0.01
PHASE = PhysicalSupport((("x", "physical-periodic-domain"), ("v", "normalized-velocity-domain")))
PHYSICAL = PhysicalSupport((("x", "physical-periodic-domain"),))
UNITLESS = PhysicalDimension()


def _zero_transport(model, state, frame):
    flux = model.flux(state.local_id + "_flux", frame=frame, state=state,
        components={axis: (0 * state[0],) for axis in frame.axes},
        waves={axis: (Const(0),) for axis in frame.axes})
    rate = model.rate(state.local_id + "_rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    return numerics


def author_physical_case(nx=16, nv=12, *, reject_field=False, field_stage=0):
    phase_frame = Rectangle("phase storage", (0, -2), (1, 2)).frame(Cartesian2D())
    field_frame = Rectangle("field storage", (0, 0), (1, 1)).frame(Cartesian2D())
    phase_model = pops.Model("phase population", frame=phase_frame)
    f = phase_model.species("distribution", state=("value",), support=PHASE,
                            units=(UNITLESS,), sampling="cell_average")
    observation = phase_model.species("observation", state=("value",), support=PHASE,
                                      units=(UNITLESS,), sampling="cell_average")
    f_numerics = _zero_transport(phase_model, f, phase_frame)
    observation_numerics = _zero_transport(phase_model, observation, phase_frame)
    coupling = phase_model.coupled_rate("consume_field", inputs=(f, observation),
        outputs={f: (observation[0],), observation: (0 * observation[0],)})
    density_model = pops.Model("physical moment population", frame=field_frame)
    density = density_model.state("U", components=("value",), support=PHYSICAL,
                                  units=(UNITLESS,), sampling="cell_average")
    density_numerics = _zero_transport(density_model, density, field_frame)
    potential = density_model.field("potential")
    observer_model = pops.Model("physical field observation", frame=field_frame)
    observed = observer_model.state("U", components=("value",), support=PHYSICAL,
                                    units=(UNITLESS,), sampling="cell_average")
    observer_numerics = _zero_transport(observer_model, observed, field_frame)
    observed_potential = observer_model.field("sample")
    source = observer_model.source("sample_field", on=observed, value=(grad(observed_potential).x,))
    case = pops.Case("explicit physical supports")
    specs = (("distribution", phase_model, f, f_numerics),
             ("observation", phase_model, observation, observation_numerics),
             ("density", density_model, density, density_numerics),
             ("field_observation", observer_model, observed, observer_numerics))
    states, blocks = {}, {}
    for name, model, state, numerics in specs:
        block = case.block(name, model, states=(state,))
        case.numerics(numerics, block=block)
        blocks[name], states[name] = block, block[state]
    problem = FieldProblem("physical Poisson", unknowns=(potential,),
        equations=(-laplacian(potential) == density[0] - 1,),
        boundaries=(FieldBoundary(potential, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
        gauge=SharedMeanGauge((potential,)))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CG(max_iter=1 if reject_field else 4000, rel_tol=1e-12, abs_tol=1e-13)))
    program = pops.Program("moment solve pullback evolve")
    current = {name: program.state(state) for name, state in states.items()}
    at = program.stage("moment field stage", c=field_stage)
    density_stage = current["density"].n
    observation_stage = current["field_observation"].n
    if field_stage:
        density_stage = program.value("predicted moment", current["density"].n + field_stage * program.dt *
            program.rhs(state=current["density"].n, terms=[Flux()]), at=at)
        observation_stage = program.value("predicted observation", current["field_observation"].n +
            field_stage * program.dt * program.rhs(state=current["field_observation"].n, terms=[Flux()]), at=at)
    solved = field.observe(program.solve(field, values={states["density"]: density_stage},
        at=at).consume(action=RejectAttempt() if reject_field else FailRun()))
    module = observer_model.module
    carrier = blocks["field_observation"][module.field_handle(module.field_spaces()["fields"])]
    context = solved.publish({(carrier, "sample_grad_x"): (solved.gradient(field[potential], dimension=2), 0)},
                             states={states["field_observation"]: observation_stage})
    sample = program.rhs(state=observation_stage, fields=context,
                         terms=[SourceTerm(blocks["field_observation"][module.operator_handle("sample_field")])])
    program.commit(current["field_observation"].next,
        program.value("observe current field", Fraction(1) * sample,
                      at=current["field_observation"].next.point))
    program.commit(current["density"].next, program.value("retain moment", current["density"].n + program.dt * program.rhs(state=current["density"].n, terms=[Flux()]),
                                                         at=current["density"].next.point))
    rates = coupling(current["distribution"].n, current["observation"].n)
    program.commit(current["distribution"].next, program.value("evolve distribution",
        current["distribution"].n + program.dt * rates[blocks["distribution"]],
        at=current["distribution"].next.point))
    program.commit(current["observation"].next, program.value("retain pullback",
        current["observation"].n + program.dt * rates[blocks["observation"]],
        at=current["observation"].next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    phase_layout = Uniform(CartesianGrid(frame=phase_frame, cells=(nx, nv), periodic=PeriodicAxes(phase_frame.axes)))
    field_layout = Uniform(CartesianGrid(frame=field_frame, cells=(nx, 1), periodic=PeriodicAxes(field_frame.axes)))
    return case, phase_layout, field_layout


def resolve_physical_case(directory, nx=16, nv=12, *, reject_field=False, field_stage=0):
    case, phase_descriptor, field_descriptor = author_physical_case(nx, nv, reject_field=reject_field, field_stage=field_stage)
    validated = pops.validate(case)
    subjects = validated.layout_subjects()
    builder = LayoutPlanBuilder(validated.owner_path.canonical())
    phase = builder.layout("phase", phase_descriptor)
    physical = builder.layout("physical", field_descriptor)
    states = {row.block_ref.local_id: row for row in subjects.states}
    for block in subjects.blocks:
        layout = phase if block.local_id in ("distribution", "observation") else physical
        builder.assign_block(block, layout)
        builder.assign_state(states[block.local_id], layout)
    for field in subjects.fields:
        builder.assign_field(field, physical)
    common = dict(source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
                  target_representation=LayoutRepresentation.CELL_AVERAGE_V1)
    moment, = builder.require_mapping(phase, physical, source=states["distribution"],
        target=states["density"], operation=LayoutMappingOperation.VELOCITY_MOMENT_V1,
        synchronization=LayoutSynchronization.BEFORE_STEP_V1,
        physical_map=PhysicalSupportMap(PHASE, PHYSICAL, VelocityQuadrature(-2, 2, nv, UNITLESS)), **common)
    pullback, = builder.require_mapping(physical, phase, source=states["field_observation"],
        target=states["observation"], operation=LayoutMappingOperation.PHYSICAL_PULLBACK_V1,
        synchronization=LayoutSynchronization.AFTER_SOURCE_STEP_V1,
        physical_map=PhysicalSupportMap(PHYSICAL, PHASE), **common)
    providers = tuple(native_physical_mapping(row, directory) for row in (moment, pullback))
    layout = builder.resolve(**subjects.to_dict(), providers=providers)
    resolved = pops.resolve(validated, layout=layout,
        layout_providers={phase: phase_descriptor, physical: field_descriptor},
        components=tuple(row.component for row in providers),
        compile_options={"include": str(ROOT / "include")})
    return case, resolved


REFINEMENTS = ((16, 12), (32, 24), (64, 48))
STEPS = 20


def initial_states(nx, nv):
    x = (np.arange(nx) + 0.5) / nx
    dv = 4.0 / nv
    v = -2 + (np.arange(nv) + 0.5) * dv
    # Exact cell averages of the polynomial, not point samples masquerading as averages.
    velocity_average = (1 + v * v + dv * dv / 12) / (28 / 3)
    distribution = (velocity_average[:, None] * (1 + 0.2 * np.cos(2 * np.pi * x))[None, :])[None]
    return {"distribution": distribution, "observation": np.full((1, nv, nx), -7.0),
            "density": np.full((1, 1, nx), -11.0),
            "field_observation": np.full((1, 1, nx), 13.0)}


def bind_physical(artifact, nx, nv):
    return pops.bind(artifact, initial_state=initial_states(nx, nv),
        resources={"execution_context": artifact_execution_context(artifact)})


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("nx,nv", REFINEMENTS)
def test_native_physical_lifecycle_quadrature_inventory_field_timing(tmp_path, nx, nv):
    case, resolved = resolve_physical_case(tmp_path, nx, nv)
    artifact = pops.compile(resolved)
    instance = bind_physical(artifact, nx, nv)
    x = (np.arange(nx) + 0.5) / nx
    dx, dv = 1 / nx, 4 / nv
    eigenvalue = 4 * np.sin(np.pi / nx) ** 2 / dx ** 2
    gradient_symbol = np.sin(2 * np.pi / nx) / dx
    amplitude = complex(0.2)
    expected_f = initial_states(nx, nv)["distribution"].copy()
    errors = []
    for step in range(1, STEPS + 1):
        density = 1 + np.real(amplitude * np.exp(2j * np.pi * x))
        electric = np.real(1j * gradient_symbol / eigenvalue * amplitude * np.exp(2j * np.pi * x))
        expected_f += DT * electric[None, None, :]
        pops.run(instance, t_end=step * DT, max_steps=1)
        actual = {name: np.asarray(instance.get_state(name)) for name in initial_states(nx, nv)}
        np.testing.assert_allclose(actual["density"].reshape(-1), density, rtol=0, atol=5e-11)
        np.testing.assert_allclose(actual["field_observation"].reshape(-1), electric, rtol=0, atol=5e-11)
        np.testing.assert_allclose(actual["observation"].reshape(1, nv, nx),
                                  np.broadcast_to(electric, (1, nv, nx)), rtol=0, atol=5e-11)
        np.testing.assert_allclose(actual["distribution"].reshape(1, nv, nx), expected_f, rtol=0, atol=5e-11)
        assert abs(float(actual["distribution"].sum()) * dx * dv - 1.0) < 2e-12
        assert instance.time() == pytest.approx(step * DT, abs=1e-14)
        assert set(instance._executor.mapping_report().values()) == {step}
        amplitude *= 1 + 1j * 4 * DT * gradient_symbol / eigenvalue
        errors.append(float(np.max(np.abs(actual["distribution"].reshape(1, nv, nx) - expected_f))))
    import json
    (tmp_path / "physical-evidence.json").write_text(json.dumps({
        "native_dimension": 2, "supports": ["1x1v", "1x"], "backend": "cpu",
        "ranks": 1, "nx": nx, "nv": nv, "steps": STEPS,
        "max_discrete_oracle_error": max(errors), "quadrature": "exact polynomial cell averages",
        "inventory": "dx*dv*sum(f)=1", "field_timing": "accepted-state c=0 before each update",
        "performance_characterized": False}, indent=2))


@pytest.fixture(scope="module")
def physical_artifact(tmp_path_factory):
    _, resolved = resolve_physical_case(tmp_path_factory.mktemp("physical-transaction"))
    return pops.compile(resolved)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_physical_prescribed_pullback_and_transaction_rollback(physical_artifact):
    instance = bind_physical(physical_artifact, 16, 12)
    native = instance._executor
    before = {name: np.asarray(instance.get_state(name)).copy() for name in initial_states(16, 12)}
    # The prescribed field is the sentinel 13 on the field observation's accepted state.
    route = next(row for row in native._transfer_routes if row.transfer.operation_abi == 3)
    native._begin_step_transaction()
    generation = native._active_transfer_generation
    route.session.capture(generation, 1)
    receipt = route.session.apply(generation, 1)
    np.testing.assert_array_equal(instance.get_state("observation"),
                                  np.full_like(instance.get_state("observation"), 13.0))
    assert receipt.source_element_count == 16
    assert receipt.destination_element_count == 16 * 12
    native._rollback_step_transaction()
    for name, values in before.items():
        np.testing.assert_array_equal(instance.get_state(name), values)
    # Invalid attempts cannot publish state, clocks, field observations or mapping receipts.
    with pytest.raises((ValueError, RuntimeError)):
        pops.run(instance, t_end=float("nan"), max_steps=1)
    for name, values in before.items():
        np.testing.assert_array_equal(instance.get_state(name), values)
    assert instance.time() == 0.0
    assert set(native.mapping_report().values()) == {0}


@pytest.mark.compiler
@pytest.mark.native_loader
def test_physical_restart_preserves_accepted_continuation(physical_artifact, tmp_path):
    instance = bind_physical(physical_artifact, 16, 12)
    pops.run(instance, t_end=DT * 3, max_steps=3)
    checkpoint = instance.checkpoint(tmp_path / "physical-checkpoint")
    pops.run(instance, t_end=DT * 6, max_steps=3)
    expected = {name: np.asarray(instance.get_state(name)).copy() for name in initial_states(16, 12)}
    instance.restart(checkpoint)
    pops.run(instance, t_end=DT * 6, max_steps=3)
    for name, value in expected.items():
        np.testing.assert_array_equal(instance.get_state(name), value)
    assert set(instance._executor.mapping_report().values()) == {6}


@pytest.mark.compiler
@pytest.mark.native_loader
def test_rejected_native_field_solve_restores_moment_and_accepted_state(tmp_path):
    _, resolved = resolve_physical_case(tmp_path, reject_field=True)
    artifact = pops.compile(resolved)
    initial = initial_states(16, 12)
    x = (np.arange(16) + 0.5) / 16
    initial["distribution"] += 0.13 / 4 * np.sin(4 * np.pi * x)[None, None, :]
    instance = pops.bind(artifact, initial_state=initial,
        resources={"execution_context": artifact_execution_context(artifact)})
    from pops._bootstrap import StepAttemptRejected
    for _ in range(2):
        with pytest.raises(StepAttemptRejected):
            pops.run(instance, t_end=DT, max_steps=1)
        for name, value in initial.items():
            np.testing.assert_array_equal(np.asarray(instance.get_state(name)).reshape(value.shape), value)
        assert instance.time() == 0.0
        assert set(instance._executor.mapping_report().values()) == {0}
