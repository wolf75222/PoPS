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
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def consumer_case(kind):
    from pops.lib.time import ForwardEuler
    from pops.moments import moment_names
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
        flux = model.flux("zero", state=state, frame=frame,
                          components={axis: tuple(0 * q for q in state) for axis in frame.axes},
                          waves={axis: (0.,) * 6 for axis in frame.axes})
        rate = model.rate("zero_rate", equation=math.ddt(state) == -math.div(flux))
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                        reconstruction=reconstruction.FirstOrder(),
                                        riemann=riemann.Rusanov()))
    case = pops.Case("conditional_consumer")
    block = case.block("matter", model)
    case.numerics(plan, block=block)
    if kind == "diffusion":
        program = ForwardEuler(block[state], rate=rate)
    else:
        program = pops.Program("conditional_rotation")
        q = program.state(block[state])
        mapped = program.affine_moment_update(q.n, q.n, linear_operator=rotation,
                                             theta_dt=program.dt / 2, order=2)
        program.commit(q.next, program.value("accepted", mapped, at=q.next.point))
    program.step_strategy(FixedDt(1e-4))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout, None if parameter is None else block[parameter]


@pytest.mark.parametrize("kind", ("diffusion", "affine"))
def test_native_consumer_accepts_inactive_and_rejects_active_invalid_branch(
        isolated_native_cache, native_cxx, kokkos_root, kind):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, parameter = consumer_case(kind)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
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
        if active:
            with pytest.raises(RuntimeError):
                pops.run(runtime, t_end=1e-4, max_steps=1)
            np.testing.assert_array_equal(np.asarray(runtime.state_global("matter")).reshape(initial.shape), initial)
        else:
            assert pops.run(runtime, t_end=1e-4, max_steps=1).accepted_steps == 1
            np.testing.assert_allclose(np.asarray(runtime.state_global("matter")).reshape(initial.shape),
                                       initial, rtol=0., atol=4e-15)
