"""Consumed solve publication to real source and face-flux providers, N16/32/64.

Uniform Dim2/OpenMP2/MPI1 qualification. A single Forward Euler step compares the
native source against its analytic gradient and the face transport against its
analytic conservative derivative. Refinement orders must be >=1.8 (source) and
>=0.8 (first-order FV transport). Both components remain sourced from the consumed
current-stage solve; no legacy field solver is installed or invoked.
"""
from __future__ import annotations

import json
import numpy as np
import pops
import pytest
from pops.domain import Rectangle
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, grad, laplacian, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.numerics.terms import Flux, SourceTerm
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_public_field_problem import _write_evidence


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
REFINEMENTS = (16, 32, 64)
DT = 0.125
AMPLITUDE = 0.1


def consumer_case(n, *, transport=False):
    frame = Rectangle("consumer domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("published field consumer", frame=frame)
    state = model.state("U", components=("rho", "driver") if transport else ("rho", "mx", "my"))
    rho = state[0]
    potential = model.field("potential")
    gradient = model.vector("gradient", frame=frame,
        components={x_axis: grad(potential).x, y_axis: grad(potential).y})
    if transport:
        driver = state[1]
        flux = model.flux("advection", frame=frame, state=state,
            components={x_axis: (0 * rho, 0 * driver), y_axis: (gradient.y * rho, 0 * driver)},
            waves={x_axis: (0 * rho, 0 * driver), y_axis: (gradient.y, 0 * driver)})
        physical_rate = model.rate("transport", equation=ddt(state) == -div(flux))
    else:
        _rho, mx, my = state
        speed = sqrt(0.5)
        flux = model.flux("Euler", frame=frame, state=state,
            components={x_axis: (mx, mx * mx / rho + 0.5 * rho, mx * my / rho),
                        y_axis: (my, mx * my / rho, my * my / rho + 0.5 * rho)},
            waves={x_axis: (mx / rho - speed, mx / rho, mx / rho + speed),
                   y_axis: (my / rho - speed, my / rho, my / rho + speed)})
        electric = model.source("electric", on=state,
            value=(0 * rho, -rho * gradient.x, -rho * gradient.y))
        physical_rate = model.rate("Euler Poisson", equation=ddt(state) == -div(flux) + electric)
        driver_model = pops.Model("independent Poisson driver", frame=frame)
        driver_state = driver_model.state("U", components=("load",))
        driver_flux = driver_model.flux("stationary", frame=frame, state=driver_state,
            components={axis: (0 * driver_state[0],) for axis in frame.axes},
            waves={axis: (0 * driver_state[0],) for axis in frame.axes})
        driver_rate = driver_model.rate("frozen load", equation=ddt(driver_state) == -div(driver_flux))
        driver = driver_state[0]
    problem = FieldProblem("Poisson", unknowns=(potential,),
        equations=(-laplacian(potential) == driver - 1,),
        boundaries=(FieldBoundary(potential, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
        gauge=SharedMeanGauge((potential,)))
    case = pops.Case("published field lifecycle")
    block = case.block("fluid", model)
    method = FiniteVolume(flux=flux, variables=variables.Conservative(state),
                          reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    numerics = DiscretizationPlan()
    numerics.rates.add(physical_rate, method)
    case.numerics(numerics, block=block)
    if not transport:
        driver_block = case.block("driver", driver_model)
        driver_numerics = DiscretizationPlan()
        driver_numerics.rates.add(driver_rate, FiniteVolume(flux=driver_flux,
            variables=variables.Conservative(driver_state), reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov()))
        case.numerics(driver_numerics, block=driver_block)
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CG(max_iter=4000, rel_tol=1e-11, abs_tol=1e-12)))
    program = pops.Program("published field step")
    current = program.state(block[state])
    point = program.stage("evaluate", c=0)
    driver_current = current if transport else program.state(driver_block[driver_state])
    solve_inputs = {block[state]: current.n} if transport else {driver_block[driver_state]: driver_current.n}
    observations = field.observe(program.solve(field, values=solve_inputs, at=point)
                                  .consume(action=FailRun()))
    solved_gradient = observations.gradient(field[potential], dimension=2)
    module = model.module
    carrier = block[module.field_handle(module.field_spaces()["fields"])]
    context = observations.publish({
        (carrier, "potential_grad_x"): (solved_gradient, 0),
        (carrier, "potential_grad_y"): (solved_gradient, 1),
    }, states=None if transport else {block[state]: current.n})
    terms = [Flux()] if transport else [SourceTerm(block[module.operator_handle("electric")])]
    rhs = program.rhs(state=current.n, fields=context, terms=terms)
    program.store_history("gradient", solved_gradient, depth=1)
    program.commit(current.next, program.value("advanced", current.n + DT * rhs, at=current.next.point))
    if not transport:
        program.commit(driver_current.next, program.value("frozen driver", 1 * driver_current.n,
                                                          at=driver_current.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes))
    return case, Uniform(grid)


def consumer_data(n, *, transport=False):
    coordinate = (np.arange(n, dtype=float) + 0.5) / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    if transport:
        rho = 1 + 0.2 * np.sin(2 * np.pi * y)
        driver = 1 + AMPLITUDE * np.cos(2 * np.pi * y)
        velocity = -AMPLITUDE / (2 * np.pi) * np.sin(2 * np.pi * y)
        derivative = AMPLITUDE * np.cos(2 * np.pi * y) * rho \
                     - velocity * (0.4 * np.pi * np.cos(2 * np.pi * y))
        initial = np.stack((rho, driver))
        expected = np.stack((rho + DT * derivative, driver))
        exact_gradient = np.stack((np.zeros_like(y), velocity))
    else:
        rho = 1 + AMPLITUDE * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
        gradient_x = -AMPLITUDE / (4 * np.pi) * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        gradient_y = -AMPLITUDE / (4 * np.pi) * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        initial = np.stack((rho, np.zeros_like(rho), np.zeros_like(rho)))
        expected = np.stack((rho, -DT * rho * gradient_x, -DT * rho * gradient_y))
        exact_gradient = np.stack((gradient_x, gradient_y))
    return initial, expected, exact_gradient


@pytest.mark.parametrize("transport", (False, True))
def test_full_public_consumed_field_source_and_transport_matrix(
    transport, isolated_native_cache, native_cxx, kokkos_root, record_property,
):
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        initial, expected, exact_gradient = consumer_data(n, transport=transport)
        case, layout = consumer_case(n, transport=transport)
        resolved = pops.resolve(pops.validate(case), layout=layout)
        assert not resolved.field_plans
        claims = resolved.blocks[0].resolved_operations.provider_evidence["program_field_publications"]
        assert len(claims) == 2
        artifact = pops.compile(resolved)
        initial_states = {"fluid": initial} if transport else {"fluid": initial, "driver": initial[0:1]}
        runtime = pops.bind(artifact, initial_state=initial_states,
                           resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=DT, max_steps=1)
        assert report.accepted_steps == 1
        actual = np.asarray(runtime.state_global("fluid")).reshape(initial.shape)
        observed_gradient = np.asarray(runtime.history_global("gradient", 0)).reshape(2, n, n)
        error = actual - expected
        l2 = float(np.sqrt(np.sum(error**2) / n**2))
        gradient_error = float(np.sqrt(np.sum((observed_gradient - exact_gradient)**2) / n**2))
        if not transport:
            np.testing.assert_array_equal(np.asarray(runtime.state_global("driver")).reshape(1, n, n),
                                          initial[0:1])
            np.testing.assert_array_equal(actual[0], initial[0])
            np.testing.assert_allclose(actual[1:], -DT * initial[0] * observed_gradient, rtol=0, atol=1e-12)
        else:
            np.testing.assert_allclose(np.mean(actual, axis=(1, 2)), np.mean(initial, axis=(1, 2)),
                                       rtol=0, atol=1e-12)
            # Independent conservative Rusanov face oracle: left/right field samples differ.
            velocity = observed_gradient[1]
            right_velocity = np.roll(velocity, -1, axis=0)
            right_state = np.roll(initial, -1, axis=1)
            speed = np.maximum(np.abs(velocity), np.abs(right_velocity))
            high_flux = -0.5 * speed * (right_state - initial)
            high_flux[0] += 0.5 * (velocity * initial[0] + right_velocity * right_state[0])
            discrete = initial - DT * n * (high_flux - np.roll(high_flux, 1, axis=1))
            np.testing.assert_allclose(actual, discrete, rtol=0, atol=2e-12)
        rows.append({"n": n, "L2": l2, "Linf": float(np.max(np.abs(error))),
                     "gradient_L2": gradient_error, "accepted_steps": report.accepted_steps})
    orders = np.log2(np.asarray([row["L2"] for row in rows[:-1]]) /
                     np.asarray([row["L2"] for row in rows[1:]]))
    assert np.min(orders) >= (0.8 if transport else 1.8)
    gradient_orders = np.log2(np.asarray([row["gradient_L2"] for row in rows[:-1]]) /
                              np.asarray([row["gradient_L2"] for row in rows[1:]]))
    assert np.min(gradient_orders) >= 1.8
    evidence = {"results": rows, "orders": orders.tolist(), "gradient_orders": gradient_orders.tolist()}
    name = "field_transport" if transport else "Euler_Poisson_source"
    record_property(name, json.dumps(evidence, sort_keys=True))
    _write_evidence("fields-public-" + name, evidence)


def test_full_joint_neumann_consumed_gradient_matrix(
    isolated_native_cache, native_cxx, kokkos_root, record_property,
):
    from tests.python.unit.fields.test_program_field_problem import field_case
    from tests.python.integration.runtime.test_public_field_problem import _data
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        first, second, _ = _data("joint_neumann_smooth", n)
        case, field, problem, program, values, point = field_case(joint=True)
        observations = field.observe(program.solve(field, values=values, at=point).consume(action=FailRun()))
        for index, unknown in enumerate(problem.unknowns):
            program.store_history("gradient%d" % index,
                observations.gradient(field[unknown], dimension=2), depth=1)
        for handle in values:
            state = program.state(handle)
            program.commit(state.next, program.value("unchanged_" + state.n.name,
                1 * state.n, at=state.next.point))
        program.step_strategy(FixedDt(DT))
        case.program(program)
        frame = case._block_registry.spec("first")["model"]._frame
        layout = Uniform(CartesianGrid(frame=frame, cells=(n, n)))
        artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
        runtime = pops.bind(artifact, initial_state={"first": first, "second": second},
                           resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=DT, max_steps=1)
        assert report.accepted_steps == 1
        coordinate = (np.arange(n, dtype=float) + 0.5) / n
        x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
        exact = np.stack((
            np.stack((-np.pi * np.sin(np.pi*x) * np.cos(np.pi*y),
                      -np.pi * np.cos(np.pi*x) * np.sin(np.pi*y))),
            np.stack((-np.pi * np.sin(2*np.pi*x) * np.cos(np.pi*y),
                      -0.5*np.pi * np.cos(2*np.pi*x) * np.sin(np.pi*y)))))
        actual = np.stack(tuple(np.asarray(runtime.history_global("gradient%d" % index, 0))
                                .reshape(2, n, n) for index in range(2)))
        error = actual - exact
        # Frozen input states and joint unknown selection survive the observation path.
        np.testing.assert_array_equal(np.asarray(runtime.state_global("first")).reshape(first.shape), first)
        np.testing.assert_array_equal(np.asarray(runtime.state_global("second")).reshape(second.shape), second)
        rows.append({"n": n, "L2": float(np.sqrt(np.sum(error**2) / n**2)),
                     "Linf": float(np.max(np.abs(error)))})
    orders = np.log2(np.asarray([[row["L2"], row["Linf"]] for row in rows[:-1]]) /
                     np.asarray([[row["L2"], row["Linf"]] for row in rows[1:]]))
    assert np.min(orders) >= 1.8
    evidence = {"results": rows, "orders": orders.tolist(), "physical_boundary": "homogeneous_neumann"}
    record_property("joint_neumann_gradient", json.dumps(evidence, sort_keys=True))
    _write_evidence("fields-public-joint-neumann-gradient", evidence)
