#!/usr/bin/env python3
"""Uniform low-level IO: visualization plus strict checkpoint boundary.

The final checkpoint route requires a compiled/bound Program, a declared run controller and an
accepted synchronized transaction. This low-level engine intentionally has none of those identities:
the test therefore proves it cannot manufacture a restart artifact. Exact compiled-Program restart,
including stride/history state, is covered by ``test_time_history_checkpoint.py``.

Verifie :
  (1) un moteur bas niveau ne peut pas contourner la frontiere de checkpoint stricte ;
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
    """Deux blocs couples par le Poisson, le second a STRIDE=2 (cadence hold-then-catch-up) :
    le restart doit reprendre la fenetre stride exactement (macro_step restaure)."""
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

# --- (1) strict checkpoint boundary -------------------------------------------------
print("== (1) checkpoint strict : moteur bas niveau refuse ==")
sim = build()
for _ in range(3):  # 3 pas (IMPAIR) : le bloc stride=2 est au MILIEU de sa fenetre au checkpoint
    sim.step(dt)
try:
    sim.checkpoint(os.path.join(tmp, "chk"))
    chk(False, "checkpoint bas niveau aurait du etre refuse")
except RuntimeError as e:
    chk("desynchronized" in str(e), f"frontiere stricte : {str(e)[:80]}")
chk(not os.path.exists(os.path.join(tmp, "chk.npz")), "aucun fichier partiel publie")

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
