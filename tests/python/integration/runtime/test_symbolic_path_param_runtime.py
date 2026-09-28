"""One compiled symbolic path, two bind-time speeds, one independent FV oracle."""

import numpy as np
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, maximum
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                           SymbolicPath, reconstruction, riemann, variables)
from pops.params import RuntimeParam
from pops.time import AdaptiveCFL
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.symbolic_path_case import SIMPSON


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _case(cells, dt):
    frame = Rectangle("param_path_box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("param_path_model", frame=frame)
    state = model.state("U", components=("u", "v"))
    u, v = state
    parameter = model.param(RuntimeParam("path_speed_margin", default=1.))
    margin = model.value(parameter)
    flux = model.flux("transport", state=state, frame=frame,
                      components={x_axis: (0.5*u, 0.5*v), y_axis: (0., 0.)})
    product = model.nonconservative_product("variable_velocity", state=state,
        matrices={x_axis: ((v*v, 0.), (0., 0.)),
                  y_axis: ((0., 0.), (0., 0.))},
        conservative_components=("v",))

    def speed(left, right, axis):
        return (margin + maximum(left[1]*left[1], right[1]*right[1])
                if axis == 0 else 0.)

    path = SymbolicPath(product, frame=frame, quadrature=SIMPSON, speed=speed)
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    method = PathConservativeFiniteVolume(flux=flux, path=path,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov())
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case = pops.Case("param_symbolic_path_case")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=0.25, max_dt=dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells),
                                   periodic=PeriodicAxes(frame.axes)))
    return case, layout, block[parameter]


def _one_step_oracle(initial, cells, dt, margin):
    right = np.roll(initial, -1, axis=2)
    speed = margin + np.maximum(initial[1]**2, right[1]**2)
    conservative_flux = 0.25*(initial + right) - 0.5*speed*(right - initial)
    integral = np.zeros_like(initial)
    integral[0] = ((initial[1]**2 + initial[1]*right[1] + right[1]**2) / 3.
                   * (right[0] - initial[0]))
    rhs = cells*(np.roll(conservative_flux, 1, axis=2) - conservative_flux
                 - 0.5*(integral + np.roll(integral, 1, axis=2)))
    return initial + dt*rhs


def test_bind_time_path_speed_changes_native_dissipation_on_same_artifact(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, dt = 8, 1.e-4
    case, layout, parameter = _case(cells, dt)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    x = (np.arange(cells) + 0.5) / cells
    u = np.broadcast_to(1. + 0.1*np.sin(2*np.pi*x), (cells, cells)).copy()
    v = np.broadcast_to(0.3 + 0.1*np.cos(2*np.pi*x), (cells, cells)).copy()
    initial = np.stack((u, v))
    observed = []
    for margin in (1., 2.):
        runtime = pops.bind(artifact, initial_state={"transport": initial.copy()},
                            params={parameter: margin},
                            resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=dt, max_steps=1)
        assert report.accepted_steps == 1
        actual = np.asarray(runtime.state_global("transport")).reshape(initial.shape)
        np.testing.assert_allclose(actual, _one_step_oracle(initial, cells, dt, margin),
                                   rtol=0., atol=4.e-14)
        observed.append(actual)
    assert not np.allclose(observed[0], observed[1], rtol=0., atol=1.e-10)
