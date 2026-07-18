#!/usr/bin/env python3
"""Uniform low-level checkpoint refusal plus synchronized direct-step state.

``System.step(dt)`` is an exact ``FixedDt`` attempt and therefore owns a declared controller plus an
accepted synchronized temporal boundary. Publication remains stricter: a durable checkpoint requires
the authenticated ``ExecutionContext`` installed by ``pops.bind``, so the direct low-level route must
be refused before checkpoint capture. Authenticated checkpoint/restart remains covered by the public
lifecycle tests.

Verifie :
  (1) trois pas directs produisent une enveloppe FixedDt synchronisee, mais ne peuvent pas publier
      un checkpoint sans ExecutionContext installe par pops.bind.
Invariants par assert ; imprime "OK test_io_checkpoint" en cas de succes.
"""
from pops.numerics.reconstruction.limiters import Minmod
import os
import sys
import tempfile

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import System  # ADC-545 advanced runtime seam

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def build(n=16):
    """Deux blocs couples par le Poisson, dont un cadence en hold-then-catch-up STRIDE=2."""
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    ions = 1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))
    slow = 1.0 + 0.3 * np.exp(-50.0 * ((X - 0.6) ** 2 + (Y - 0.5) ** 2))
    sim = System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    sim.add_equation("ions",
                  engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                            transport=engine.IsothermalFlux(),
                            source=engine.PotentialForce(charge=1.0),
                            elliptic=engine.BackgroundDensity(
                                alpha=1.0, n0=float(ions.mean()))),
                  spatial=engine.Spatial(limiter=Minmod()), time=engine.Explicit())
    sim.add_equation("slow",
                  engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                            transport=engine.IsothermalFlux(),
                            source=engine.PotentialForce(charge=-1.0),
                            elliptic=engine.BackgroundDensity(
                                alpha=-1.0, n0=float(slow.mean()))),
                  spatial=engine.Spatial(limiter=Minmod()),
                  time=engine.Explicit(stride=2))
    sim.set_density("ions", ions.ravel())
    sim.set_density("slow", slow.ravel())
    return sim


tmp = tempfile.mkdtemp()
dt = 2e-3

# --- (1) synchronized direct-step state and strict bind authority ------------------
print("== (1) pas direct : frontiere FixedDt synchronisee, autorite bind stricte ==")
sim = build()
for _ in range(3):
    sim.step(dt)
temporal = sim.program_report().temporal
chk(temporal["strategy"]["strategy"]["kind"] == "fixed_dt",
    "l'enveloppe directe conserve la strategie FixedDt exacte")
chk(temporal["transaction_stats"] == {"accepted": 3, "failed": 0, "rejected": 0},
    "une seule acceptation temporelle par pas direct")
try:
    checkpoint_root = os.path.join(tmp, "chk")
    sim.checkpoint(checkpoint_root)
    chk(False, "checkpoint sans ExecutionContext aurait du etre refuse")
except ValueError as exc:
    chk("authenticated ExecutionContext installed by pops.bind" in str(exc),
        "checkpoint refuse sans ExecutionContext installe par pops.bind")
    chk(not os.path.exists(checkpoint_root + ".npz"),
        "le refus ne laisse aucun checkpoint partiel")

if fails:
    print(f"FAIL test_io_checkpoint : {fails} echec(s)")
    sys.exit(1)
print("OK test_io_checkpoint")
