"""ROM-99: the face inputs n and (sL, sR) of a Python face, compiled and run against b6's routes.

A face that reads the normal ``n`` must take the native steps, so a wrong binding of the normal
(its sign, an exchanged axis) moves the state. A face that reads the signed interval (sL, sR)
must refuse a face where a trace's interval is not finite or not ordered, as the native
``riemann.HLL()`` does, whatever its body does with the pair: the step is refused, the clock stays
at its start and the state is not published. The rows of the N3a pilot, with the derived
``hll(F, Uh)`` on M01, M03 and M16, are in test_api_v02_python_faces_native.py.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from examples.api_v02.common import CaseSpec
from examples.api_v02.models_M01_M14 import (
    _domain, _initial, _periodic_metrics, _ssprk2, build_m01, native_hll, native_rusanov,
)
from examples.api_v02.models_M15_M18 import build_m16
from pops import math as pm
from pops.lib.numerics import hll, rusanov, upwind
from pops.numerics import DiscretizationPlan
from tests.python.integration.runtime.test_api_v02_python_faces_native import (
    _assert_rounding_level, _bind, _run,
)
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check,
)
from tests.python.support.public_native_profile import native_profile


# The transport velocity a of build_m01: its flux is (a.n) u on each face.
VELOCITY = (.7, .3)


def _smax_along_the_normal(uL, uR, n):
    del uL, uR
    return abs(VELOCITY[0]*n[0] + VELOCITY[1]*n[1])


def _speed_along_the_normal(face):
    return VELOCITY[0]*face.normal[0] + VELOCITY[1]*face.normal[1]


FACES_THAT_READ_THE_NORMAL = {
    "rusanov smax(uL, uR, n)": lambda flux, state: rusanov(flux, state,
                                                            smax=_smax_along_the_normal),
    "upwind speed(face)": lambda flux, state: upwind(flux, state, speed=_speed_along_the_normal),
}


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_faces_that_read_the_normal_match_the_native_rusanov(isolated_native_cache, native_cxx,
                                                             kokkos_root, record_property):
    """``rusanov`` with ``smax(uL, uR, n)`` and ``upwind`` with ``speed(face)`` (section 4.4).

    On M01, |a.n| is the spectral radius that riemann.Rusanov() takes, and a.n > 0 on both axes,
    where the upwind flux of a linear transport is the Rusanov flux: both faces take the native
    steps to rounding. Each face sees an error of the normal that the other does not: the upwind
    face its sign (it would take the downwind trace), the Rusanov face an exchange of the axes
    (|a.n| is .7 on x and .3 on y, while a.n is positive on both).
    """
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="ROM-99 M01 normal")
    native_steps, _, native_state = _run(world, build_m01, native_rusanov, {}, "fluid",
                                         "ROM-99 M01 native Rusanov")
    rows = {label: _run(world, build_m01, scheme, {}, "fluid", "ROM-99 M01 " + label)
            for label, scheme in FACES_THAT_READ_THE_NORMAL.items()}
    with collective_check(world):
        for label, (steps, _, state) in rows.items():
            assert steps == native_steps >= 2, (label, steps, native_steps)
            if native_state is not None:
                _assert_rounding_level(native_steps, native_state, state, record_property,
                                       label + " ")


def _m01_with_an_inverted_pair(scheme):
    """build_m01 with the signed pair (a + 1, a) declared on each axis.

    The pair is not ordered (sL > sR), while the spectral radius |a| of ``waves=`` and the
    physical fluxes stay valid, so only the interval of the face can refuse it. A face that used
    the pair anyway would take sL >= 0, F(uL).n, a finite flux.
    """
    cells = (16, 8)
    frame, layout = _domain("M01_periodic", cells)
    model = pops.Model("M01_inverted_signed_pair", frame=frame)
    U = model.state("amount", components=("u",))
    flux = model.flux("transport", state=U, frame=frame,
        components={axis: (a*U[0],) for axis, a in zip(frame.axes, VELOCITY, strict=True)},
        waves={axis: (a,) for axis, a in zip(frame.axes, VELOCITY, strict=True)})
    model.wave_speeds(flux, frame=frame, values={
        axis: (a + 1., a) for axis, a in zip(frame.axes, VELOCITY, strict=True)})
    rate = model.rate("conservation", equation=pm.ddt(U) == -pm.div(flux))
    case = pops.Case("M01_inverted_signed_pair")
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, scheme(flux, U))
    case.numerics(plan, block=block)
    dt = .05/max(cells)
    _ssprk2(case, block, U, lambda p, u, point, label: rate(u), dt, name="M01_SSPRK2")
    x, y = np.meshgrid((np.arange(cells[0])+.5)/cells[0], (np.arange(cells[1])+.5)/cells[1])
    initial = (1+.2*np.cos(2*np.pi*(x+y)))[None]
    subject = block[U]
    return CaseSpec(case, layout, _initial(case, subject, initial), {"fluid": subject}, 2*dt, dt,
        _periodic_metrics(initial, minimum=.799), "M01 with an inverted signed pair")


def _refused_step(world, spec, block, route):
    """Bind, take one step; return the failures of every rank, the clock and both states."""
    runtime = _bind(world, spec, route)
    initial = collective_call(world, lambda: runtime.state_global(block))
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=spec.dt, max_steps=1,
                                                             console=False))
    final = collective_call(world, lambda: runtime.state_global(block))
    clock = collective_call(world, lambda: (float(runtime.time()), int(runtime.macro_step())))
    return failures, clock, initial, final


def _assert_refused(label, failures, clock, initial, final):
    assert all(failure is not None and failure[2] and
               "hyperbolic face evaluation refused publication" in failure[1]
               for failure in failures), (label, failures)
    assert clock == (0., 0), (label, clock)
    if initial is not None:
        np.testing.assert_array_equal(final, initial, err_msg=label)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_an_inverted_signed_pair_is_refused_by_the_interval_as_by_the_native_hll(
        isolated_native_cache, native_cxx, kokkos_root):
    """The interval guard of the derived face, on a pair that nothing else refuses (ROM-99).

    The native ``riemann.HLL()`` refuses the face before it reads the pair; the derived
    ``hll(F, Uh)`` publishes the same refusal, and keeps the clock and the state.
    """
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="ROM-99 M01 inverted signed pair")
    for label, scheme in (("native HLL", native_hll), ("derived hll", hll)):
        spec = collective_call(world, lambda scheme=scheme: _m01_with_an_inverted_pair(scheme))
        refused = _refused_step(world, spec, "fluid", "ROM-99 M01 inverted pair " + label)
        with collective_check(world):
            _assert_refused(label, *refused)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_derived_hll_refuses_the_nonreal_oblique_spectrum_of_m16(isolated_native_cache,
                                                                  native_cxx, kokkos_root):
    """The signed interval of a trace with a nonreal Jacobian spectrum is not finite.

    The native HLL refuses such a face (test_api_v02_models_M15_M21.py, same profile) when it
    reads the interval, before the stability bound, whose Jacobian majorant is not finite either.
    The derived face reads (sL, sR) first too and refuses it, so the step is refused and the state
    is not published. The interval is not the only guard that refuses this profile; the inverted
    pair above is the one that only the interval refuses.
    """
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="ROM-99 M16 oblique Python HLL")
    spec = collective_call(world, lambda: build_m16(cells=16, steps=4, oblique_refusal=True,
                                                    scheme=hll))
    refused = _refused_step(world, spec, "moments", "ROM-99 M16 oblique Python HLL")
    with collective_check(world):
        _assert_refused("derived hll on M16's oblique profile", *refused)
