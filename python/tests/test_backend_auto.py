#!/usr/bin/env python3
"""Backend par defaut clean-break : pas de ``backend="auto"`` public.

La route publique est descripteur-first. Omettre le backend revient a demander explicitement
``Production()`` ; un repli AOT implicite n'est plus une API utilisateur. Si un autre chemin est
voulu, il doit etre nomme par ``pops.codegen.AOT()`` / ``JIT()``.
"""
from pops.numerics.reconstruction.limiters import Minmod
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
from pops.ir.ops import sqrt
from pops.physics.facade import Model

fails = 0
INCLUDE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "include"))


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso3(name):
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my",
                                      roles=["Density", "MomentumX", "MomentumY"])
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    c = sqrt(0.5)
    m.flux(x=[mx, mx * u + 0.5 * rho, mx * v], y=[my, my * u, my * v + 0.5 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    return m


def skip_if_kokkos_missing(exc):
    text = str(exc)
    if "Kokkos" in text or "POPS_HAS_KOKKOS" in text:
        print("skip test_backend_auto : Kokkos introuvable -> portion compilee sautee")
        sys.exit(0)
    raise exc


cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if not cxx or not os.path.isdir(INCLUDE):
    print("skip test_backend_auto : compilateur ou en-tetes pops absents")
    sys.exit(0)

tmp = tempfile.mkdtemp()
try:
    try:
        iso3("bad")._m.compile(os.path.join(tmp, "bad.so"), INCLUDE, backend="bogus")
        chk(False, "backend inconnu aurait du lever")
    except TypeError as e:
        chk("typed" in str(e), f"backend string rejete : {str(e)[:80]}")

    print("== (1) backend omis -> Production() explicite par defaut ==")
    try:
        cm = iso3("default_prod")._compile_for_runtime(
            so_path=os.path.join(tmp, "default_prod.so"), include=INCLUDE)
    except RuntimeError as e:
        skip_if_kokkos_missing(e)
    chk(cm.backend == "production", f"compile() sans backend -> {cm.backend!r}")
    chk(cm.backend_auto_reason is None, "pas de raison auto : aucun repli implicite")
    n = 16
    sim = pops.System(n=n, L=1.0, periodic=True)
    sim._set_poisson()
    sim._add_equation("f", model=cm, spatial=pops.FiniteVolume(limiter=Minmod()),
                     time=pops.Explicit())
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    z = np.zeros((n, n))
    sim.set_primitive_state("f", rho=1.0 + 0.3 * np.exp(-40 * ((X - .5) ** 2 + (Y - .5) ** 2)),
                            u=z, v=z)
    sim.step_cfl(0.3)
    chk(np.all(np.isfinite(np.asarray(sim.density("f")))), "bloc production par defaut tourne fini")

    print("== (2) backend explicite inchange ==")
    try:
        cm3 = iso3("explicit_aot")._compile_for_runtime(
            so_path=os.path.join(tmp, "explicit_aot.so"), include=INCLUDE,
            backend=pops.codegen.AOT())
    except RuntimeError as e:
        skip_if_kokkos_missing(e)
    chk(cm3.backend == "aot" and cm3.backend_auto_reason is None,
        "backend=pops.codegen.AOT() explicite : aot, sans raison auto")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_backend_auto : {fails} echec(s)")
    sys.exit(1)
print("OK test_backend_auto")
