#!/usr/bin/env python3
"""V3 compiled capabilities: HLLC, required IMEX Programs, and the source-Jacobian guard.

These independent native compiler checks are split from test_v3_features.py so their four
model packages and one time Program do not share the A-C runtime group's process deadline.
"""

from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
import os
import shutil
import sys
import tempfile

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.math import sqrt
from pops.physics import Density, Momentum
from pops.physics._facade import Model
from pops.runtime._system import System
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.support.physics_roles import X_AXIS, Y_AXIS
from tests.python.support.requirements import (
    missing_compiler_requirement,
    repo_include,
    require_native_or_skip,
)

# Four distinct model capabilities plus one time Program compile on a cold runner.
POPS_PROCESS_TIMEOUT = 900

fails = 0
INCLUDE = repo_include()


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


# --- (D) DSL : enable_hllc + source_jacobian (compilateur requis) ---------------------
missing = missing_compiler_requirement(INCLUDE)
if missing:
    if fails:
        print(f"FAIL test_v3_compiled_capabilities : {fails} echec(s)")
        sys.exit(1)
    require_native_or_skip(f"(D) test_v3_compiled_capabilities : {missing}")


def iso3_dsl(name, hllc=False, jac=False):
    m = Model(name)
    rho, mx, my = m.conservative_vars(
        "rho", "mx", "my",
        roles=[Density(), Momentum(X_AXIS), Momentum(Y_AXIS)],
    )
    cs2 = 0.5
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", cs2 * rho)
    c = sqrt(cs2)
    m.flux(x=[mx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    if hllc:
        m.enable_hllc()
    if jac:
        kk = 50.0
        m.source([0.0 * rho, -kk * mx, -kk * my])  # friction raide lineaire
        m.source_jacobian(
            [
                [0.0 * rho, 0.0 * rho, 0.0 * rho],
                [0.0 * rho, -kk + 0.0 * rho, 0.0 * rho],
                [0.0 * rho, 0.0 * rho, -kk + 0.0 * rho],
            ]
        )
    return m


tmp = tempfile.mkdtemp()
try:
    print("== (D1) enable_hllc : riemann='hllc' sur 3-var NON Euler ==")
    cm_h = iso3_dsl("iso3_hllc", hllc=True).compile(
        os.path.join(tmp, "iso3_hllc.so"), INCLUDE, backend="production"
    )
    chk(getattr(cm_h, "has_hllc", False), "CompiledModel.has_hllc = True (capability emise)")
    sh = System(n=24, L=1.0, periodicity=(True, True))
    sh.set_poisson()
    sh.add_equation(
        "f",
        model=cm_h,
        spatial=engine.Spatial(limiter=Minmod(), flux=HLLC()),
        time=engine.Explicit(),
    )
    z = np.zeros((24, 24))
    sh.set_primitive_state("f", rho=gaussian(24), u=z, v=z)
    install_forward_euler_program(sh)
    for _ in range(5):
        sh.step_cfl(0.3)
    chk(
        np.all(np.isfinite(np.asarray(sh.density("f")))),
        "HLLC capability sur 3-var : 5 pas finis (contact-resolving hors Euler)",
    )
    cm_nh = iso3_dsl("iso3_nohllc").compile(
        os.path.join(tmp, "iso3_nohllc.so"), INCLUDE, backend="production"
    )
    try:
        s2 = System(n=16, L=1.0, periodicity=(True, True))
        s2.add_equation("f", model=cm_nh, spatial=engine.Spatial(limiter=Minmod(), flux=HLLC()))
        chk(False, "hllc sans capability sur 3-var aurait du lever")
    except (ValueError, RuntimeError) as e:
        chk("hllc" in str(e), f"rejet sans capability : {str(e)[:70]}")

    print("== (D2) source_jacobian : aucun ancien IMEX sans Program ==")
    cm_j = iso3_dsl("iso3_jac", jac=True).compile(
        os.path.join(tmp, "iso3_jac.so"), INCLUDE, backend="production"
    )
    cm_f = iso3_dsl("iso3_fd", jac=True)
    cm_f._m._src_jac = None  # meme modele, SANS jacobien emis -> FD historiques
    cm_f = cm_f.compile(os.path.join(tmp, "iso3_fd.so"), INCLUDE, backend="production")

    def expect_imex_program_required(cm, label):
        s = System(n=16, L=1.0, periodicity=(True, True))
        s.set_poisson()
        s.add_equation(
            "f",
            model=cm,
            spatial=engine.Spatial(limiter=Minmod()),
            time=engine.IMEX(),
        )
        z16 = np.zeros((16, 16))
        s.set_primitive_state("f", rho=gaussian(16), u=0.2 + z16, v=z16)
        try:
            s.step(1e-3)
        except TypeError as error:
            chk(
                "exact registered StepStrategy" in str(error)
                and "Program.step_strategy" in str(error),
                f"{label}: un Program avec strategie authentifiee reste obligatoire",
            )
            return
        chk(False, f"{label}: un ancien solveur IMEX a avance sans Program")

    expect_imex_program_required(cm_j, "jacobien analytique")
    expect_imex_program_required(cm_f, "jacobien differences finies")

    print("== (D3) garde CODEGEN : source_jacobian sans source -> erreur (pas de purge muette) ==")
    mg = iso3_dsl("iso3_guard", jac=True)
    mg._m._source = None  # jacobien declare, source retiree : compile() doit lever (pas check())
    try:
        mg.compile(os.path.join(tmp, "iso3_guard.so"), INCLUDE, backend="production")
        chk(False, "source_jacobian sans source aurait du lever au codegen")
    except ValueError as e:
        chk("source_jacobian" in str(e), f"codegen leve : {str(e)[:70]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_v3_compiled_capabilities : {fails} echec(s)")
    sys.exit(1)
print("OK test_v3_compiled_capabilities")
