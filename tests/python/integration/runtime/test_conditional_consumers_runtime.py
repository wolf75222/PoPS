"""Installed native C09 consumers: diffusion and affine moment coefficients."""
import numpy as np
import pytest
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.time import FixedDt
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _attempt(runtime, world):
    report, error = None, None
    try:
        report = pops.run(runtime, t_end=1e-4, max_steps=1)
    except Exception as exc:
        error = (type(exc).__name__, str(exc))
    if world is None:
        return report, (error,)
    from pops._native_collectives import allgather_value
    return report, allgather_value(world, error)


def consumer_case(kind):
    from pops.lib.time import ForwardEuler
    from pops.moments import affine_push_forward, moment_indices, moment_names
    from pops.params import RuntimeParam
    frame = Rectangle("conditional_domain", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("conditional_physics", frame=frame)
    state = model.state("U", components=("u",) if kind == "diffusion" else tuple(moment_names(2)))
    plan = DiscretizationPlan()
    parameter = None
    if kind == "diffusion":
        q = state[0]
        danger = model.scalar("danger", 1 / (q - q))
        variable = math.where(q > 0, lambda: danger, lambda: q)
        flux = model.diffusive_flux("heat", state=state, value=.1 * math.grad(variable))
        rate = model.rate("heat_rate", equation=math.ddt(state) == math.div(flux))
        plan.rates.add(rate, Diffusion(flux=flux))
    else:
        parameter = model.param(RuntimeParam("omega", default=-1.))
        p = model.value(parameter)
        omega = math.minimum(math.where(p > 0, lambda: 1 / (p - p), lambda: 2.), 3.)
        matrix = [[0.] * 6 for _ in range(6)]
        matrix[1][3], matrix[3][1] = omega, -omega
        rotation = model.operator("rotation", returns=model.local_linear_operator(
            "rotation", on=state, matrix=matrix))
        if kind == "affine_library":
            # The fixed-step consumer authors its Cayley map using ordinary
            # expressions. The reusable library itself knows no time method,
            # gyro law, positive-density domain, or canonical moment ordering.
            a = omega * (1e-4 / 2)
            cosine, sine = (1-a*a)/(1+a*a), 2*a/(1+a*a)
            mx, my = state[1]/state[0], state[3]/state[0]
            transformed = affine_push_forward(
                state, indices=moment_indices(2),
                matrix=((cosine, sine), (-sine, cosine)),
                offset=(mx-cosine*mx-sine*my, my+sine*mx-cosine*my))
            # Match this consumer's historical rho>0 admission contract. The
            # branch remains in the common DAG; no projection repairs a state.
            model.source("affine_library", on=state, value=tuple(
                math.where(state[0] > 0, lambda value=value, k=k: value-state[k],
                           lambda: 1/(state[0]-state[0])) for k, value in enumerate(transformed)))
        if kind != "affine_library":
            flux = model.flux("zero", state=state, frame=frame,
                              components={axis: tuple(0 * q for q in state) for axis in frame.axes},
                              waves={axis: (0.,) * 6 for axis in frame.axes})
            rate = model.rate("zero_rate", equation=math.ddt(state) == -math.div(flux))
            plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                            reconstruction=reconstruction.FirstOrder(),
                                            riemann=riemann.Rusanov()))
    case = pops.Case("conditional_consumer")
    block = case.block("matter", model)
    if kind != "affine_library":
        case.numerics(plan, block=block)
    if kind == "diffusion":
        program = ForwardEuler(block[state], rate=rate)
    else:
        program = pops.Program("conditional_rotation")
        q = program.state(block[state])
        if kind == "affine_library":
            mapped = q.n + program.source(model.module.operator_handle("affine_library"), q.n)
        else:
            mapped = program.affine_moment_update(q.n, q.n, linear_operator=rotation,
                                                 theta_dt=program.dt / 2, order=2)
        program.commit(q.next, program.value("accepted", mapped, at=q.next.point))
    program.step_strategy(FixedDt(1e-4))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout, None if parameter is None else block[parameter]


@pytest.mark.parametrize("kind", ("diffusion", "affine", "affine_library"))
def test_native_consumer_accepts_inactive_and_rejects_active_invalid_branch(
        isolated_native_cache, native_cxx, kokkos_root, kind):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, parameter = consumer_case(kind)
    artifact, world = _compile(case, layout, "conditional-consumer-" + kind)
    for active in (False, True):
        if kind == "diffusion":
            initial = np.full((1, 4, 4), 1. if active else -1.)
            params = {}
        else:
            initial = np.zeros((6, 4, 4))
            initial[[0, 2, 5]] = 1.  # isotropic centered second moments
            params = {parameter: 1. if active else -1.}
        runtime = pops.bind(artifact, initial_state={"matter": initial.copy()}, params=params,
                            resources={"execution_context": artifact_execution_context(artifact)})
        report, errors = _attempt(runtime, world)
        if active:
            assert all(error is not None and error[0] == "RuntimeError" for error in errors), errors
            gathered = np.asarray(runtime.state_global("matter"))
            _root_check(world, lambda gathered=gathered, initial=initial: np.testing.assert_array_equal(
                gathered.reshape(initial.shape), initial))
        else:
            assert not any(errors), errors
            assert report.accepted_steps == 1
            gathered = np.asarray(runtime.state_global("matter"))
            _root_check(world, lambda gathered=gathered, initial=initial: np.testing.assert_allclose(
                gathered.reshape(initial.shape), initial, rtol=0., atol=4e-15))


def test_native_affine_library_consumer_matches_particles_and_retained_recipe(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    from pops.moments import moment_indices
    particles = np.array([[-.5, .2], [.7, -.3], [.1, .8], [-.2, -.6]])
    weights = np.array([.2, .5, .7, .3])
    indices = moment_indices(2)
    raw = np.array([np.dot(weights, particles[:, 0]**i * particles[:, 1]**j) for i, j in indices])
    initial = np.broadcast_to(raw[:, None, None], (6, 4, 4)).copy()
    a = 2. * 1e-4 / 2
    matrix = np.array([[(1-a*a)/(1+a*a), 2*a/(1+a*a)],
                       [-2*a/(1+a*a), (1-a*a)/(1+a*a)]])
    mean = np.average(particles, axis=0, weights=weights)
    moved = (particles-mean) @ matrix.T + mean
    expected = np.array([np.dot(weights, moved[:, 0]**i * moved[:, 1]**j) for i, j in indices])
    results = []
    for kind in ("affine", "affine_library"):
        case, layout, parameter = consumer_case(kind)
        artifact, world = _compile(case, layout, "affine-particles-" + kind)
        runtime = pops.bind(artifact, initial_state={"matter": initial.copy()}, params={parameter: -1.},
                            resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=1e-4, max_steps=1)
        gathered = np.asarray(runtime.state_global("matter"))
        assert report.accepted_steps == 1

        def check(gathered=gathered):
            result = gathered.reshape(initial.shape)
            np.testing.assert_allclose(result, np.broadcast_to(expected[:, None, None], initial.shape),
                                       rtol=2e-13, atol=2e-14)
            np.testing.assert_array_equal(result[0], initial[0])
            results.append(result)
        _root_check(world, check)
    _root_check(world, lambda: np.testing.assert_allclose(*results, rtol=2e-13, atol=2e-14))
