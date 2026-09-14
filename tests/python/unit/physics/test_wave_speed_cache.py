#!/usr/bin/env python3
"""Production-package wave-speed cache admission and default HLL transport parity.

ModelSpec now lowers to an authenticated production package. That ABI does not carry
wave_speed_cache, so enabling it must fail explicitly before native publication. The
historical native-composed cache ON/OFF trajectory is not a supported Python route.
Keep the full N32, twenty-step default/OFF bit-parity and nonstationary-state oracle,
and exercise the explicit refusal across the former Riemann/time/geometry routes.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import HLL, Rusanov
from pops.mesh.masks import CutCell
import sys

import numpy as np

from pops.runtime._engine_descriptors import (
    BackgroundDensity, Explicit, FluidState, IMEX, IsothermalFlux, Model, NoSource, Spatial,
)
from pops.runtime._system import System  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import install_forward_euler_program

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn()
        return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


CS2 = 0.5
N = 32


def make_sim(cache, riemann=None, limiter=None, time=None):
    riemann = riemann if riemann is not None else HLL()
    limiter = limiter if limiter is not None else FirstOrder()
    sim = System(n=N, L=1.0, periodicity=(True, True))
    sim.add_equation("ions",
                     Model(state=FluidState("isothermal", cs2=CS2),
                           transport=IsothermalFlux(), source=NoSource(),
                           elliptic=BackgroundDensity(alpha=1.0, n0=1.0)),
                     spatial=Spatial(limiter=limiter, flux=riemann, wave_speed_cache=cache),
                     time=time if time is not None else Explicit())
    return sim


x = (np.arange(N) + 0.5) / N
X, Y = np.meshgrid(x, x, indexing="ij")
U0 = np.stack([1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y),
               0.2 * np.cos(2 * np.pi * X),
               -0.15 * np.sin(2 * np.pi * Y)])

print("== (1) explicit cache OFF: full twenty-step HLL trajectory ==")
s_off = make_sim(cache=False)
s_off.set_state("ions", U0)
install_forward_euler_program(s_off)
for _ in range(20):
    s_off.step_cfl(0.4)
A_off = np.array(s_off.get_state("ions"))
chk(not np.array_equal(A_off, U0), "l'etat a reellement evolue (test non creux)")

print("== (2) defaut inchange : sans wave_speed_cache == cache OFF ==")
s_def = System(n=N, L=1.0, periodicity=(True, True))
s_def.add_equation("ions",
                   Model(state=FluidState("isothermal", cs2=CS2),
                         transport=IsothermalFlux(), source=NoSource(),
                         elliptic=BackgroundDensity(alpha=1.0, n0=1.0)),
                   spatial=Spatial(limiter=FirstOrder(), flux=HLL()),
                   time=Explicit())
s_def.set_state("ions", U0)
install_forward_euler_program(s_def)
for _ in range(20):
    s_def.step_cfl(0.4)
chk(np.array_equal(np.array(s_def.get_state("ions")), A_off),
    "FiniteVolume sans wave_speed_cache == cache OFF (bit-identique)")

def check_unsupported_cache(fn, label):
    message = err_msg(fn)
    chk("wave_speed_cache" in message and "production package ABI" in message,
        f"{label}: explicit unsupported production capability ({message[:100]})")


print("== (3) cache ON is refused for every production-package treatment ==")
check_unsupported_cache(lambda: make_sim(cache=True), "HLL explicit")
message = err_msg(lambda: make_sim(cache=True, riemann=Rusanov()))
chk("wave_speed_cache requires flux=riemann.HLL()" in message,
    "Rusanov is rejected by the spatial contract before package admission")
check_unsupported_cache(lambda: make_sim(cache=True, time=IMEX()), "HLL IMEX")


def install_half_space(sim, mode):
    from pops.analytic import coordinates
    from pops.domain import CartesianDomain
    from pops.mesh.geometry import LevelSet
    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    frame = CartesianDomain("test-wave-speed-cache-eb", (0.0, 0.0), (1.0, 1.0)).frame()
    level_set = LevelSet(coordinates(frame)[0] - 0.5)
    ((opcodes, literals),) = lower_analytic_components(
        (level_set.expression.to_data(),), frame_id=frame.canonical_id
    )
    sim._s._set_analytic_level_set(
        list(opcodes), list(literals), mode, 0.0, 0.0, 0.0
    )


def make_mode_then_cache():
    sim = System(n=N, L=1.0, periodicity=(True, True))
    install_half_space(sim, CutCell().lower())
    sim.add_equation("ions",
                     Model(state=FluidState("isothermal", cs2=CS2),
                           transport=IsothermalFlux(), source=NoSource(),
                           elliptic=BackgroundDensity(alpha=1.0, n0=1.0)),
                     spatial=Spatial(limiter=FirstOrder(), flux=HLL(),
                                     wave_speed_cache=True),  # doit lever (mode disque actif)
                     time=Explicit())


# The reverse order cannot publish a cached block: cache admission already refuses.
# The geometry-first order must retain that same ABI refusal, with its geometry intact.
check_unsupported_cache(make_mode_then_cache, "LevelSet then cached package")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
