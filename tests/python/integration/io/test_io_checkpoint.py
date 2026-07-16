#!/usr/bin/env python3
"""Uniform low-level IO: visualization plus synchronized direct-step state.

``System.step(dt)`` is an exact ``FixedDt`` attempt and therefore owns a declared controller plus an
accepted synchronized temporal boundary. Publication remains stricter: a durable checkpoint requires
an installed compiled Program identity, so the direct low-level route must be refused. Compiled-Program
checkpoint/restart remains covered by ``test_time_history_checkpoint.py``.

Verifie :
  (1) trois pas directs produisent une enveloppe FixedDt synchronisee, mais ne peuvent pas publier
      un checkpoint sans identite de Program compile ;
  (2) write npz : champs et horloge presents ; write vtk : .vti lisible (en-tete ImageData).
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
    sim = System(n=n, L=1.0, periodic=True)
    sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    sim.add_equation("ions",
                  engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                            transport=engine.IsothermalFlux(),
                            source=engine.PotentialForce(charge=1.0),
                            elliptic=engine.ChargeDensity(charge=1.0)),
                  spatial=engine.Spatial(limiter=Minmod()), time=engine.Explicit())
    sim.add_equation("slow",
                  engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                            transport=engine.IsothermalFlux(),
                            source=engine.PotentialForce(charge=-1.0),
                            elliptic=engine.ChargeDensity(charge=-1.0)),
                  spatial=engine.Spatial(limiter=Minmod()),
                  time=engine.Explicit(stride=2))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    sim.set_density("ions", (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).ravel())
    sim.set_density("slow", (1.0 + 0.3 * np.exp(-50.0 * ((X - 0.6) ** 2 + (Y - 0.5) ** 2))).ravel())
    return sim


tmp = tempfile.mkdtemp()
dt = 2e-3

# --- (1) synchronized direct-step state and strict checkpoint identity -------------
print("== (1) pas direct : frontiere FixedDt synchronisee, identite stricte ==")
sim = build()
for _ in range(3):
    sim.step(dt)
temporal = sim.program_report().temporal
chk(temporal["strategy"]["kind"] == "fixed_dt",
    "l'enveloppe directe conserve la strategie FixedDt exacte")
chk(temporal["transaction_stats"] == {"accepted": 3, "failed": 0, "rejected": 0},
    "une seule acceptation temporelle par pas direct")
try:
    checkpoint_root = os.path.join(tmp, "chk")
    sim.checkpoint(checkpoint_root)
    chk(False, "checkpoint sans Program compile aurait du etre refuse")
except RuntimeError as exc:
    chk("installed compiled Program hash" in str(exc),
        "checkpoint refuse sans identite de Program compile")
    chk(not os.path.exists(checkpoint_root + ".npz"),
        "le refus ne laisse aucun checkpoint partiel")

# --- (2) write npz / vtk ---------------------------------------------------------------
print("== (2) write npz / vtk ==")
p_npz = sim.write(os.path.join(tmp, "out"), format="npz", step=7)
d = np.load(p_npz)
chk(p_npz.endswith("_000007.npz") and "state_ions" in d and "phi" in d and "macro_step" in d,
    f"npz ecrit avec etats/phi/horloge ({os.path.basename(p_npz)})")
chk(d["state_ions"].shape == (3, 16, 16), "npz : etat (ncomp, ny, nx)")
p_vti = sim.write(os.path.join(tmp, "out"), format="vtk", step=7)
head = open(p_vti).read(200)
chk("ImageData" in head and "VTKFile" in head, f"vti ecrit (en-tete ImageData) ({os.path.basename(p_vti)})")
chk("ions_rho" in open(p_vti).read(), "vti : DataArray par variable (ions_rho)")
try:
    sim.write(os.path.join(tmp, "out"), format="silo")
    chk(False, "format inconnu aurait du lever")
except ValueError as e:
    chk("format" in str(e), f"format inconnu : {str(e)[:60]}")

if fails:
    print(f"FAIL test_io_checkpoint : {fails} echec(s)")
    sys.exit(1)
print("OK test_io_checkpoint")
