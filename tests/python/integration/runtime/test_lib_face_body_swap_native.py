"""CF4 body-swap parity of the native face loop that runs a Python face (N3, ROM-87).

Spec 0.3 CF4: a face loop passes its parity test with two generated bodies from scientifically
distinct cases, the L1 upwind flux and the L5 Scharfetter-Gummel flux, and a native that embeds a
formula fails the second body. The loop under test is the one every ``pops.lib.numerics`` flux
compiles to: the generated ``riemann.User`` face policy (``pops.generated.riemann.face/v2``) inside
the native finite-volume transport. Its two bodies:

* L1: ``upwind(F, U, speed=f)`` of ``pops.lib.numerics``, f the velocity dotted with the face
  normal; the velocity is positive along x and negative along y, so each face takes both
  branches of the row of 2.6.9 (s uL where s > 0, else s uR).
* L5: the Scharfetter-Gummel face of L5 (section 5.10, D8), written in the case as a
  ``riemann.User`` body, with the drift given (a constant potential gradient, so a constant
  drift velocity c): F* = (D/h) (B(-Pe) uL - B(Pe) uR), Pe = (c.n) h / D, B(z) = z / expm1(z),
  the Bernoulli function of M10's fitted face. It reads the normal, the spacing of the face's
  axis, ``where`` and ``expm1``; a loop that held the upwind or the Rusanov formula could not run
  it.

Each body runs two SSPRK2 steps on the same periodic 24 x 8 mesh through the same native loop,
and each final state is compared with its own NumPy oracle, written from the formulas and not
from the generated code, to a few ulps per step. The two runs then differ by the diffusion of
the second body, far above rounding, so the loop ran each body and not one fixed formula.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from examples.api_v02.common import CaseSpec
from examples.api_v02.models_M01_M14 import _domain, _initial, _periodic_metrics, _ssprk2
from pops import math as pm
from pops.lib.numerics import upwind
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.riemann.user import authenticated_user_face
from pops.numerics.spatial import FiniteVolume
from tests.python.integration.runtime.test_api_v02_python_faces_native import (
    _assert_rounding_level, _bind,
)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import native_profile


CELLS = (24, 8)
SPACING = (1./CELLS[0], 1./CELLS[1])
# The drift velocity: positive along x, negative along y.
VELOCITY = (.7, -.4)
DIFFUSION = .05
DT = .002
STEPS = 2


def _normal_velocity(n):
    return VELOCITY[0]*n[0] + VELOCITY[1]*n[1]


def _l1_upwind(flux, state):
    """L1: the library's upwind flux with the velocity of the case dotted with the normal."""
    return upwind(flux, state, speed=lambda face: _normal_velocity(face.normal))


def _bernoulli(z):
    """B(z) = z / (exp(z) - 1), with the branches of M10's fitted face."""
    square = z*z
    return pm.where(abs(z) < 1.e-4,
        lambda: 1 - z/2 + square*(1/12 + square*(-1/720 + square/30240)),
        lambda: pm.where(z > 50, lambda: z*pm.exp(-z)/(1 - pm.exp(-z)),
            lambda: pm.where(z < -50, lambda: -z/(1 - pm.exp(z)), lambda: z/pm.expm1(z))))


def _l5_body(uL, uR, fL, fR, s, *, n):
    del fL, fR, s
    spacing = SPACING[0]*n[0] + SPACING[1]*n[1]
    peclet = _normal_velocity(n)*spacing/DIFFUSION
    rate = DIFFUSION/spacing
    return (rate*(_bernoulli(-peclet)*uL[0] - _bernoulli(peclet)*uR[0]),)


def _l5_speed(uL, uR, fL, fR, s, *, n):
    del uL, uR, fL, fR, s
    spacing = SPACING[0]*n[0] + SPACING[1]*n[1]
    return abs(_normal_velocity(n)) + 2*DIFFUSION/spacing


def _l5_scharfetter_gummel(flux, state):
    """L5: the Scharfetter-Gummel face written in the case, on the same native loop."""
    face = riemann.User(body=_l5_body, state=state, stability=_l5_speed)
    return FiniteVolume(flux=flux, variables=variables.Conservative(state),
                        reconstruction=reconstruction.FirstOrder(), riemann=face)


BODIES = {"L1 upwind": _l1_upwind, "L5 Scharfetter-Gummel": _l5_scharfetter_gummel}


def _initial_state():
    x, y = np.meshgrid((np.arange(CELLS[0]) + .5)/CELLS[0], (np.arange(CELLS[1]) + .5)/CELLS[1])
    return (1 + .2*np.cos(2*np.pi*(x + y)) + .1*np.sin(2*np.pi*(2*x - y)))[None]


def _build(scheme):
    """A periodic scalar drift on the unit square; ``scheme`` realizes its transport."""
    frame, layout = _domain("N3_body_swap_periodic", CELLS)
    model = pops.Model("N3_body_swap_drift", frame=frame)
    U = model.state("density", components=("u",))
    flux = model.flux("drift", state=U, frame=frame,
        components={axis: (c*U[0],) for axis, c in zip(frame.axes, VELOCITY, strict=True)},
        waves={axis: (c,) for axis, c in zip(frame.axes, VELOCITY, strict=True)})
    rate = model.rate("conservation", equation=pm.ddt(U) == -pm.div(flux))
    case = pops.Case("N3_body_swap")
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, scheme(flux, U))
    case.numerics(plan, block=block)
    _ssprk2(case, block, U, lambda p, u, point, label: rate(u), DT, name="N3_body_swap_SSPRK2")
    initial = _initial_state()
    subject = block[U]
    return CaseSpec(case, layout, _initial(case, subject, initial), {"fluid": subject},
                    STEPS*DT, DT, _periodic_metrics(initial, minimum=.5),
                    "N3 body swap: periodic scalar drift, one native face loop, two bodies")


# The oracles, from the formulas: F[i + 1/2] is the flux through the face between cell i and
# cell i + 1 of an axis (periodic), and the rate is -(F[i + 1/2] - F[i - 1/2]) / h on each axis.
def _np_bernoulli(z):
    return z/np.expm1(z)


def _np_upwind(left, right, axis):
    s = VELOCITY[axis]
    return s*left if s > 0 else s*right


def _np_scharfetter_gummel(left, right, axis):
    spacing = SPACING[axis]
    peclet = VELOCITY[axis]*spacing/DIFFUSION
    return DIFFUSION/spacing*(_np_bernoulli(-peclet)*left - _np_bernoulli(peclet)*right)


def _np_rate(u, face):
    rate = np.zeros_like(u)
    for axis, array_axis in ((0, -1), (1, -2)):
        through = face(u, np.roll(u, -1, axis=array_axis), axis)
        rate -= (through - np.roll(through, 1, axis=array_axis))/SPACING[axis]
    return rate


def _np_ssprk2(face):
    u = _initial_state()
    for _ in range(STEPS):
        predictor = u + DT*_np_rate(u, face)
        u = .5*u + .5*(predictor + DT*_np_rate(predictor, face))
    return u


ORACLES = {"L1 upwind": _np_upwind, "L5 Scharfetter-Gummel": _np_scharfetter_gummel}


def test_the_two_bodies_compile_to_the_same_loop_with_distinct_policies():
    """Source level: both faces are the generated face of riemann.User, one native loop; only the
    generated body differs, and the L5 body holds the exponential that the L1 body does not."""
    from pops.codegen.user_riemann_lowering import emit_user_face_policy
    from types import SimpleNamespace

    frame, _ = _domain("N3_body_swap_periodic", CELLS)
    model = pops.Model("N3_body_swap_drift", frame=frame)
    U = model.state("density", components=("u",))
    flux = model.flux("drift", state=U, frame=frame,
        components={axis: (c*U[0],) for axis, c in zip(frame.axes, VELOCITY, strict=True)},
        waves={axis: (c,) for axis, c in zip(frame.axes, VELOCITY, strict=True)})
    faces = {label: authenticated_user_face(scheme(flux, U).riemann)
             for label, scheme in BODIES.items()}
    assert {face.native_id for face in faces.values()} == {"pops.generated.riemann.face/v2"}
    policies = {label: emit_user_face_policy(SimpleNamespace(_user_face=face))
                for label, face in faces.items()}
    assert policies["L1 upwind"] != policies["L5 Scharfetter-Gummel"]
    assert "expm1" in policies["L5 Scharfetter-Gummel"]
    assert "expm1" not in policies["L1 upwind"]
    assert riemann.Rusanov().native_id not in {face.native_id for face in faces.values()}


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_the_native_face_loop_runs_the_l1_and_the_l5_bodies(isolated_native_cache, native_cxx,
                                                             kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="N3 CF4 body swap")
    finals = {}
    for label, scheme in BODIES.items():
        spec = collective_call(world, lambda scheme=scheme: _build(scheme))
        runtime = _bind(world, spec, "N3 CF4 body swap " + label)
        report = collective_call(world, lambda runtime=runtime, spec=spec: pops.run(
            runtime, t_end=spec.t_end, max_steps=8, console=False))
        state = collective_call(world, lambda runtime=runtime: runtime.state_global("fluid"))
        with collective_check(world):
            assert report.accepted_steps == STEPS, (label, report.accepted_steps)
            if state is not None:
                _assert_rounding_level(STEPS, _np_ssprk2(ORACLES[label]), state, record_property,
                                       label + " ")
        finals[label] = state
    with collective_check(world):
        if finals["L1 upwind"] is not None:
            # The second body is not the first: the diffusion of L5 moves the state far above
            # rounding, so the loop ran each body, not one formula.
            gap = float(np.max(np.abs(np.asarray(finals["L1 upwind"])
                                      - np.asarray(finals["L5 Scharfetter-Gummel"]))))
            record_property("body_gap_linf", gap)
            assert gap > 1.e-6, gap


# ---------------------------------------------------------------------------------------------
# The upwind row of ROM-87 on master's tutorial 02 (docs/tutorials/scalar_advection/
# 02_openmp_explicit_ssprk2.py): F(U) = a U with a = (1, 0.25), MUSCL(VanLeer()), inflow on x_min
# and y_min, outflow on x_max and y_max, the authored SSPRK2 under AdaptiveCFL(0.45, max_dt=0.01).
# Master's face is riemann.ScalarUpwind(velocity=a), the native pops::RusanovFlux; the library's
# is upwind(F, U, speed=f), f(face) = a.n. The reconstruction stays the native MUSCL(VanLeer())
# on both sides, so the row isolates the flux; the Python muscl and vanleer are the row of the
# reconstruction family.
#
# Two rows. The first runs both faces at the step that the native tutorial takes, FixedDt(0.45/64)
# (|a| = 1 on x, h = 1/64), after checking on the native route that this step reproduces the
# tutorial's own steps and final state. The second runs the tutorial as written, under its
# AdaptiveCFL, with each face: before ROM-107 every Python face was refused there at its first
# step (reason 503, user_face_numerical_stability), because System::step_cfl proposed the step
# cfl min_a h_a / max_a s_a while the guard of the riemann.User face checked dt sum_a s_a / h_a
# against cfl. The guard now reads the frequency of the bound its strategy proposes from
# (StepStrategy.face_frequency_rule), so both faces take the same steps.
# ---------------------------------------------------------------------------------------------
TUTORIAL_CELLS = (64, 64)
TUTORIAL_VELOCITY = (1.0, 0.25)
TUTORIAL_CFL = 0.45
# The step of the native tutorial: cfl h / max |a|, below max_dt = 0.01.
TUTORIAL_STEP = TUTORIAL_CFL/TUTORIAL_CELLS[0]


def _tutorial_02_native(flux, state, velocity):
    return FiniteVolume(flux=flux, variables=variables.Conservative(state),
                        reconstruction=reconstruction.MUSCL(reconstruction.limiters.VanLeer()),
                        riemann=riemann.ScalarUpwind(velocity=velocity))


def _tutorial_02_library(flux, state, velocity):
    del velocity
    return upwind(flux, state, speed=lambda face: TUTORIAL_VELOCITY[0]*face.normal[0]
                  + TUTORIAL_VELOCITY[1]*face.normal[1],
                  reconstruction=reconstruction.MUSCL(reconstruction.limiters.VanLeer()))


def _tutorial_02(method, *, fixed_step=False):
    from fractions import Fraction

    from pops.boundary import TransportBoundarySet
    from pops.boundary.transport import Inflow, Outflow
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid
    from pops.representations import Conservative
    from pops.spaces import CellState
    from pops.time import AdaptiveCFL, FixedDt, StagePoint, TimePoint

    nx, ny = TUTORIAL_CELLS
    ax, ay = TUTORIAL_VELOCITY
    far_field = 0.05
    frame = Rectangle("unit_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).tag("fluid").frame(
        Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("scalar_advection", frame=frame)
    U = model.state("U", components=("u",), representation=Conservative(),
                    space=CellState(frame=frame))
    (u,) = U
    velocity = model.vector("a", frame=frame, components={x_axis: ax, y_axis: ay})
    flux = model.flux("advection_flux", frame=frame, state=U,
                      components={x_axis: (ax*u,), y_axis: (ay*u,)},
                      waves={x_axis: (ax,), y_axis: (ay,)})
    rate = model.rate("advection_rate", equation=pm.ddt(U) == -pm.div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method(flux, U, velocity))
    case = pops.Case("tutorial_scalar_advection")
    tracer = case.block("tracer", model=model)
    tracer_U = tracer[U]
    boundaries = frame.boundaries
    numerics.boundaries.add(TransportBoundarySet({
        boundaries.x_min: Inflow(state=tracer_U, value=far_field),
        boundaries.x_max: Outflow(state=tracer_U),
        boundaries.y_min: Inflow(state=tracer_U, value=far_field),
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
    program.commit(q.next, program.value(
        "ssprk2_step", q.n + program.dt*half*k0 + program.dt*half*k1, at=q.next.point))
    program.step_strategy(FixedDt(TUTORIAL_STEP) if fixed_step
                          else AdaptiveCFL(cfl=TUTORIAL_CFL, max_dt=1.0e-2))
    case.program(program)
    x = (np.arange(nx, dtype=np.float64) + 0.5)/nx
    y = (np.arange(ny, dtype=np.float64) + 0.5)/ny
    xx, yy = np.meshgrid(x, y, indexing="xy")
    initial = far_field + 0.95*np.exp(-120.0*((xx - 0.30)**2 + (yy - 0.35)**2))
    initial = np.ascontiguousarray(initial[np.newaxis, :, :], dtype=np.float64)
    layout = Uniform(CartesianGrid(frame=frame, cells=TUTORIAL_CELLS))
    return CaseSpec(case, layout, _initial(case, tracer_U, initial), {"tracer": tracer_U},
                    0.20, 1.0e-2, lambda runtime: {"passed": True},
                    "master tutorial 02: scalar advection, MUSCL(VanLeer), inflow/outflow, "
                    "SSPRK2 under AdaptiveCFL")


def _run_tutorial_02(world, label, method, *, fixed_step):
    spec = collective_call(world, lambda: _tutorial_02(method, fixed_step=fixed_step))
    runtime = _bind(world, spec, "N3 tutorial 02 " + label)
    report = collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=10_000,
                                                     console=False))
    clock = collective_call(world, lambda: float(runtime.time()))
    state = collective_call(world, lambda: runtime.state_global("tracer"))
    return report.accepted_steps, clock, state


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_upwind_matches_master_scalar_upwind_on_tutorial_02(isolated_native_cache, native_cxx,
                                                             kokkos_root, record_property):
    """The case of tutorial 02 at the tutorial's own step: master's face, then the library's."""
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="N3 upwind tutorial 02")
    as_written = _run_tutorial_02(world, "native AdaptiveCFL", _tutorial_02_native,
                                  fixed_step=False)
    native = _run_tutorial_02(world, "native FixedDt", _tutorial_02_native, fixed_step=True)
    library = _run_tutorial_02(world, "upwind(speed=) FixedDt", _tutorial_02_library,
                               fixed_step=True)
    with collective_check(world):
        # FixedDt(0.45/64) is the tutorial's own step sequence on the native route.
        assert native[:2] == as_written[:2] and as_written[0] > 10, (native[:2], as_written[:2])
        if as_written[2] is not None:
            # Same step sequence, same face: the two native runs agree bit for bit.
            record_property("tutorial 02 native FixedDt vs AdaptiveCFL bitwise_equal",
                            bool(np.array_equal(as_written[2], native[2])))
            assert np.array_equal(as_written[2], native[2]), (
                float(np.max(np.abs(np.asarray(native[2]) - np.asarray(as_written[2])))))
        assert library[:2] == native[:2], (library[:2], native[:2])
        if native[2] is not None:
            _assert_rounding_level(native[0], native[2], library[2], record_property,
                                   "tutorial 02 upwind vs ScalarUpwind ")


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_upwind_matches_master_scalar_upwind_on_tutorial_02_as_written(isolated_native_cache,
                                                                       native_cxx, kokkos_root,
                                                                       record_property):
    """ROM-107: the tutorial as written, under its AdaptiveCFL, with master's face and the library's.

    The two runs take the same steps to the same clock, and their final states agree to rounding.
    Before ROM-107 the library's face was refused at the first step and its clock stayed at zero.
    """
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="N3 tutorial 02 as written")
    native = _run_tutorial_02(world, "native AdaptiveCFL", _tutorial_02_native, fixed_step=False)
    library = _run_tutorial_02(world, "upwind(speed=) AdaptiveCFL", _tutorial_02_library,
                               fixed_step=False)
    with collective_check(world):
        assert native[0] > 10, native[:2]
        assert library[:2] == native[:2], (library[:2], native[:2])
        if native[2] is not None:
            _assert_rounding_level(native[0], native[2], library[2], record_property,
                                   "tutorial 02 as written, upwind vs ScalarUpwind ")
