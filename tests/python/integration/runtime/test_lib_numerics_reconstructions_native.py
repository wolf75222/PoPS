"""N3 (ROM-87): muscl, the limiters and weno5 written in Python (pops.lib.numerics) against b6's
native routes, on the case of master's tutorial 02 (docs/tutorials/scalar_advection/02).

The case is the tutorial's: 64 x 64 cells, the authored flux a u with a = (1, 0.25), inflow on
x_min and y_min, outflow on x_max and y_max, SSPRK2 written in pops.Program, AdaptiveCFL(0.45)
up to t = 0.2. Each row compiles it twice, with the native
``reconstruction.MUSCL(limiters.<Limiter>())`` and with ``muscl(U, <limiter>)``, the face flux
being master's ``riemann.ScalarUpwind`` in both so that the reconstruction is the only
difference, and runs both artifacts from the same initial state. The two runs take the same steps
and reach the same time; their final states agree to rounding, and their L1 and L-infinity
differences and bitwise equality are recorded. Since N3 lots 2 and 3 ``muscl``, the limiters and
``weno5`` are the b6 bodies of ``examples/api_v02/b6_reconstructions.py`` (D5), which the words of
the surface equal node for node (``tests/python/unit/surface/test_n3_lot2_muscl.py`` and
``test_n3_lot3_weno5.py``). The ``weno5`` row does the same with the native
``reconstruction.WENO5()`` and ``weno5(U)``.

The tutorial with a Python face flux as well is not a row here: under the tutorial's AdaptiveCFL
a riemann.User face is refused at the first step (user_face_numerical_stability, ROM-107), a
cause in the core route of Python faces, which the flux lot of N3 carries.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pops
import pytest

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.reconstruction import limiters
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, StagePoint, TimePoint
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

from examples.api_v02.b6_reconstructions import mc, minmod, muscl, superbee, vanleer, weno5

# The constants of master's tutorial 02.
NX, NY = 64, 64
AX, AY = 1.0, 0.25
FAR_FIELD = 0.05
GAUSSIAN_AMPLITUDE, GAUSSIAN_BETA = 0.95, 120.0
GAUSSIAN_CENTER_X, GAUSSIAN_CENTER_Y = 0.30, 0.35
CFL, MAX_DT, T_END, MAX_STEPS = 0.45, 1.0e-2, 0.20, 10_000

# limiter -> (native descriptor factory, Python limiter)
LIMITERS = {
    "minmod": (limiters.Minmod, minmod),
    "vanleer": (limiters.VanLeer, vanleer),
    "mc": (limiters.MC, mc),
    "superbee": (limiters.Superbee, superbee),
}


def _native_scalar_upwind(flux, U, recon, velocity):
    return FiniteVolume(flux=flux, variables=variables.Conservative(U), reconstruction=recon,
                        riemann=riemann.ScalarUpwind(velocity=velocity))


def _tutorial_02(method):
    """Master's tutorial 02, with ``method(flux, U, velocity)`` as its finite-volume method."""
    domain = Rectangle("unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).tag("fluid")
    frame = domain.frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    grid = CartesianGrid(frame=frame, cells=(NX, NY))
    model = pops.Model("scalar_advection", frame=frame)
    U = model.state("U", components=("u",), representation=Conservative(),
                    space=CellState(frame=frame))
    (u,) = U
    velocity = model.vector("a", frame=frame, components={x_axis: AX, y_axis: AY})
    flux = model.flux("advection_flux", frame=frame, state=U,
                      components={x_axis: (AX*u,), y_axis: (AY*u,)},
                      waves={x_axis: (AX,), y_axis: (AY,)})
    rate = model.rate("advection_rate", equation=ddt(U) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method(flux, U, velocity))
    case = pops.Case("tutorial_scalar_advection")
    tracer = case.block("tracer", model=model)
    tracer_U = tracer[U]
    boundaries = frame.boundaries
    numerics.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=tracer_U, value=FAR_FIELD),
        boundaries.x_max: Outflow(state=tracer_U),
        boundaries.y_min: Inflow(state=tracer_U, value=FAR_FIELD),
        boundaries.y_max: Outflow(state=tracer_U),
    }))
    case.numerics(numerics, block=tracer)
    program = pops.Program("SSPRK2")
    q = program.state(tracer_U)
    stage_0 = StagePoint("ssprk2_stage_0", {"main": TimePoint(program.clock, 0)})
    k0 = program.value("ssprk2_k_0", rate(q.n), at=stage_0)
    stage_1 = StagePoint("ssprk2_stage_1", {"main": TimePoint(program.clock, 1)})
    q_stage = program.value("ssprk2_U1", q.n + program.dt*k0, at=stage_1)
    k1 = program.value("ssprk2_k_1", rate(q_stage), at=stage_1)
    half = Fraction(1, 2)
    program.commit(q.next, program.value("ssprk2_step",
                                         q.n + program.dt*half*k0 + program.dt*half*k1,
                                         at=q.next.point))
    program.step_strategy(AdaptiveCFL(cfl=CFL, max_dt=MAX_DT))
    case.program(program)
    return case, Uniform(grid)


def _initial_state():
    x = (np.arange(NX, dtype=np.float64) + 0.5)/NX
    y = (np.arange(NY, dtype=np.float64) + 0.5)/NY
    xx, yy = np.meshgrid(x, y, indexing="xy")
    initial = FAR_FIELD + GAUSSIAN_AMPLITUDE*np.exp(
        -GAUSSIAN_BETA*((xx - GAUSSIAN_CENTER_X)**2 + (yy - GAUSSIAN_CENTER_Y)**2))
    return np.ascontiguousarray(initial[np.newaxis, :, :], dtype=np.float64)


def _run(world, method, route):
    case, layout = collective_call(world, lambda: _tutorial_02(method))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, resolved, route=route,
                                         compile_artifact=pops.compile)
    initial = _initial_state()
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_state={"tracer": initial},
        resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=T_END, max_steps=MAX_STEPS,
                                                     console=False))
    state = collective_call(world, lambda: runtime.state_global("tracer"))
    if state is not None:
        state = np.asarray(state, dtype=np.float64).reshape(initial.shape)
    return report.accepted_steps, float(runtime.time()), state


def _assert_parity(native, python, record_property):
    native_steps, native_time, native_state = native
    python_steps, python_time, python_state = python
    assert native_steps == python_steps >= 2
    assert native_time == python_time
    if native_state is None:
        return
    difference = python_state - native_state
    l1, linf = float(np.mean(np.abs(difference))), float(np.max(np.abs(difference)))
    record_property("accepted_steps", native_steps)
    record_property("l1", l1)
    record_property("linf", linf)
    record_property("bitwise_equal", bool(np.array_equal(python_state, native_state)))
    # The same formula in the same order: only floating-point contraction may differ.
    scale = max(1., float(np.max(np.abs(native_state))))
    assert linf <= 16*np.finfo(float).eps*scale*native_steps, (l1, linf)
    assert np.all(np.isfinite(python_state))


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("name", tuple(LIMITERS))
def test_python_muscl_matches_the_native_muscl(name, isolated_native_cache, native_cxx,
                                               kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope=f"N3 muscl-{name}")
    native_limiter, python_limiter = LIMITERS[name]
    native = _run(world, lambda flux, U, velocity: _native_scalar_upwind(
        flux, U, reconstruction.MUSCL(native_limiter()), velocity), f"N3 muscl-{name} native")
    python = _run(world, lambda flux, U, velocity: _native_scalar_upwind(
        flux, U, muscl(U, python_limiter), velocity), f"N3 muscl-{name} Python")
    with collective_check(world):
        _assert_parity(native, python, record_property)



def test_python_weno5_matches_the_native_weno5(isolated_native_cache, native_cxx, kokkos_root,
                                               record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="N3 weno5")
    native = _run(world, lambda flux, U, velocity: _native_scalar_upwind(
        flux, U, reconstruction.WENO5(), velocity), "N3 weno5 native")
    python = _run(world, lambda flux, U, velocity: _native_scalar_upwind(
        flux, U, weno5(U), velocity), "N3 weno5 Python")
    with collective_check(world):
        _assert_parity(native, python, record_property)
