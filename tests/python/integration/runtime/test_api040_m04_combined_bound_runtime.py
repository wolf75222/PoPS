"""Prepared W02 native rejection: separate bounds pass, composed bound fails.

Do not run during a concurrent native rebuild; the root reception owns the
installed-artifact check and execution of this test.
"""
import numpy as np
import pops
import pytest

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import CoeffGradient, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


@pytest.mark.parametrize("transverse", (False, True))
def test_dim2_native_rejects_unstable_sum_without_mutating_the_pulse(transverse):
    n, a, diffusivity = 32, 1., .01
    dt = .75 / n
    c = a * dt * n
    r = diffusivity * dt * n * n
    assert c < 1 and 2 * r < 1 and 4 * r < 1
    assert c + 2 * r > 1 and c + 4 * r > 1
    assert 1 - c - 4 * r == pytest.approx(-.71)
    assert 1 - c - 2 * r == pytest.approx(-.23)

    frame = Rectangle("w02_square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("w02_combined", frame=frame)
    state = model.state("pulse", components=("amount",))
    (u,) = state
    transport = model.flux("transport", frame=frame, state=state,
                           components={x_axis: (a * u,), y_axis: (0 * u,)},
                           waves={x_axis: (a,), y_axis: (0.,)})
    diffusion = model.diffusive_flux("diffusion", state=state,
                                     value=CoeffGradient(u, ((diffusivity, 0.),
                                                            (0., diffusivity if transverse else 0.))))
    rate = model.rate("balance", equation=ddt(state) == -div(transport) + div(diffusion))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=diffusion, transport=FiniteVolume(
        flux=transport, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())))
    case = pops.Case("api040_W02_rejection")
    block = case.block("pulse_block", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n),
                                   periodic=PeriodicAxes(frame.axes)))
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    assert artifact.resolved_dimension == 2
    initial = np.zeros((1, n, n), dtype=np.float64)
    initial[0, n // 2, n // 2] = 1.
    context = artifact_execution_context(artifact)
    world = context.communicator.handle
    simulation = pops.bind(artifact, initial_state={"pulse_block": initial},
                           resources={"execution_context": context})
    before = simulation.state_global("pulse_block")
    rejection = ""
    try:
        pops.run(simulation, t_end=dt, max_steps=1, console=False)
    except RuntimeError as error:
        rejection = str(error)
    if world is not None:
        from pops._native_collectives import allgather_value
        rejections = allgather_value(world, rejection)
    else:
        rejections = (rejection,)
    assert all(any(token in error for token in
                   ("combined_transport_diffusion_stability", "pointwise", "rejected"))
               for error in rejections), rejections
    after = simulation.state_global("pulse_block")
    failure = b""
    if world is None or world.rank == 0:
        try:
            np.testing.assert_array_equal(np.asarray(before).reshape(initial.shape), initial)
            np.testing.assert_array_equal(np.asarray(after).reshape(initial.shape), initial)
        except AssertionError as error:
            failure = str(error).encode()
    if world is not None:
        failure = world.broadcast_bytes(failure, root=0)
    assert not failure, failure.decode()
    assert simulation.time() == 0 and simulation.macro_step() == 0
