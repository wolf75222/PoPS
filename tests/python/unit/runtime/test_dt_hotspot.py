#!/usr/bin/env python3
"""Native CFL-hotspot regression behind the final runtime boundary.

``RuntimeInstance`` intentionally does not expose an ad-hoc diagnostic method, but the native
executor still implements the on-demand reduction used by runtime inspection.  This test keeps the
numerical oracle on that native seam: dominant-cell location, agreement with ``step_cfl``, stable
tie-breaking, and absence of side effects.  It does not reintroduce a public compatibility method.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.runtime._system import System

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn()
        return ""
    except Exception as exc:  # noqa: BLE001 -- the message is the contract under test
        return str(exc)


CS2 = 0.5


def make_sim(n=32):
    sim = System(n=n, L=1.0, periodicity=(True, True))
    sim.add_equation(
        "ions",
        engine.Model(
            state=engine.FluidState("isothermal", cs2=CS2),
            transport=engine.IsothermalFlux(),
            source=engine.NoSource(),
            elliptic=engine.BackgroundDensity(alpha=1.0, n0=1.0),
        ),
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(),
    )
    return sim


def hotspot(sim, block):
    """Read the still-supported native diagnostic without widening RuntimeInstance."""
    return sim._s.dt_hotspot(block)


print("== (1) planted dominant cell ==")
n, i0, j0, u_hot = 32, 21, 9, 7.5
sim = make_sim(n)
rho = np.ones((n, n))
mx = np.zeros((n, n))
mx[j0, i0] = u_hot * rho[j0, i0]
sim.set_state("ions", np.stack([rho, mx, np.zeros((n, n))]))
w, ih, jh = hotspot(sim, "ions")
w_ref = u_hot + np.sqrt(CS2)
chk(abs(w - w_ref) < 1e-12, "w equals the analytic |u| + cs")
chk((int(ih), int(jh)) == (i0, j0), "reported cell is the planted hotspot")

print("== (2) same transport reduction as step_cfl ==")
dt = sim.step_cfl(0.4)
chk(abs(0.4 * (1.0 / n) / dt - w) < 1e-9 * w,
    "cfl*h/dt equals the hotspot wave speed")

print("== (3) querying the diagnostic has no runtime side effect ==")
sa, sb = make_sim(), make_sim()
x = (np.arange(32) + 0.5) / 32
X, Y = np.meshgrid(x, x, indexing="ij")
U0 = np.stack([
    1.0 + 0.3 * np.sin(2 * np.pi * X),
    0.4 * np.cos(2 * np.pi * Y),
    np.zeros((32, 32)),
])
sa.set_state("ions", U0)
sb.set_state("ions", U0)
_ = hotspot(sa, "ions")
dta = sa.step_cfl(0.4)
dtb = sb.step_cfl(0.4)
chk(dta == dtb and np.array_equal(
    np.array(sa.get_state("ions")), np.array(sb.get_state("ions"))),
    "step and state remain bit-identical with or without the query")

print("== (4) deterministic tie break on a uniform state ==")
su = make_sim()
su.set_state("ions", np.stack([
    np.ones((32, 32)),
    0.5 * np.ones((32, 32)),
    np.zeros((32, 32)),
]))
_w, iu, ju = hotspot(su, "ions")
chk((int(iu), int(ju)) == (0, 0), "uniform ties select the first global cell")

print("== (5) unknown block fails loud ==")
msg = err_msg(lambda: hotspot(sim, "ghost"))
chk("ghost" in msg, "unknown block is named in the error")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
