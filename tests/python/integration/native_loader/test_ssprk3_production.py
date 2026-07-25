#!/usr/bin/env python3
"""SSPRK3 parity for a component produced by the final compilation lifecycle.

The typed time descriptor rejects unknown methods before native installation.  A detached
production component installed through ``System.add_equation`` then matches the equivalent native
ModelSpec under SSPRK3 and demonstrably differs from SSPRK2.
"""
from pops.numerics.variables import Conservative
from pops.numerics.riemann import Rusanov
import sys

import numpy as np

import pops.runtime._engine_descriptors as engine
from test_dsl_coupled import build_euler, compile_euler_component, GAMMA, INCLUDE
from tests.python.support.requirements import (
    missing_compiler_requirement,
    require_native_or_skip,
)
from pops.runtime._system import System  # ADC-545 advanced runtime seam

# Multiple DSL native compiles by design: on a slow CI runner the file can exceed the
# global 300 s process-isolation budget (ADC-627, same class as test_dsl_compile_cache).
POPS_PROCESS_TIMEOUT = 900

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


def _native_spec(rho0):
    """Equivalent native Euler bricks used as the numerical parity oracle."""
    return engine.Model(state=engine.FluidState("compressible", gamma=GAMMA),
                     transport=engine.CompressibleFlux(),
                     source=engine.NoSource(),
                     elliptic=engine.BackgroundDensity(alpha=1.0, n0=rho0))


def _initial_state(n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((4, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    velocity_x = 0.2 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
    velocity_y = -0.15 * np.cos(2.0 * np.pi * X) * np.sin(2.0 * np.pi * Y)
    pressure = 1.0 + 0.1 * np.sin(2.0 * np.pi * X)
    U[1] = U[0] * velocity_x
    U[2] = U[0] * velocity_y
    U[3] = pressure / (GAMMA - 1.0) + 0.5 * U[0] * (
        velocity_x * velocity_x + velocity_y * velocity_y
    )
    return U


# --- (1) Typed authoring guard: SSPRK3 is accepted, an unknown method is rejected. ----------------
print("== (1) Explicit(method='ssprk3') accepte ; une methode inconnue est rejetee ==")
ssprk3 = engine.Explicit(method="ssprk3")
chk(str(ssprk3.kind) == "ssprk3", "SSPRK3 lower to the exact typed native route")
msg_bad = err_msg(lambda: engine.Explicit(method="rk4"))
chk(msg_bad != "" and "ssprk2" in msg_bad and "ssprk3" in msg_bad,
    "une methode inconnue est rejetee avant toute installation native")

# --- (2)/(3) PARITE + NON-TRIVIALITE (necessite un compilateur + en-tetes pops) ---------------------
missing = missing_compiler_requirement(INCLUDE)
if missing:
    if fails:
        print(f"test_ssprk3_production : {fails} ECHEC(S)")
        sys.exit(1)
    require_native_or_skip(f"(2)/(3) test_ssprk3_production : {missing}")

n, L = 48, 1.0
U = _initial_state(n)
Uflat = U.reshape(-1).tolist()
spec = _native_spec(float(U[0].mean()))
model = build_euler("ssprk3-production")


def run_compiled_checks():
    compiled = compile_euler_component(model, cells=16)

    def build_prod(method):
        s = System(n=n, L=L, periodicity=(True, True))
        s.add_equation(
            "gas",
            compiled,
            spatial=engine.Spatial(
                minmod=True, flux=Rusanov(), recon=Conservative()
            ),
            time=engine.Explicit(method=method),
        )
        s.set_state("gas", Uflat)
        return s

    def build_ref_ssprk3():
        s = System(n=n, L=L, periodicity=(True, True))
        s.add_equation(
            "gas",
            spec,
            spatial=engine.Spatial(
                minmod=True, flux=Rusanov(), recon=Conservative()
            ),
            time=engine.Explicit(method="ssprk3"),
        )
        s.set_state("gas", Uflat)
        return s

    # (2a) eval_rhs : production+SSPRK3 == native ModelSpec+SSPRK3 (the spatial residual does not
    # RK -- mais on verifie que les DEUX chemins instancient le meme bloc avant toute avance).
    prod = build_prod("ssprk3")
    ref = build_ref_ssprk3()
    R_prod = np.array(prod.eval_rhs("gas")).reshape(4, n, n)
    R_ref = np.array(ref.eval_rhs("gas")).reshape(4, n, n)
    chk(float(np.max(np.abs(R_prod))) > 1e-3, "(2a) residu non trivial")
    chk(float(np.max(np.abs(R_prod - R_ref))) == 0.0,
        "(2a) eval_rhs production+SSPRK3 BIT-IDENTIQUE au ModelSpec+SSPRK3")

    # (2b) avance SSPRK3 : etat final bit-identique au bloc natif sur 12 pas a dt fixe.
    prod = build_prod("ssprk3")
    ref = build_ref_ssprk3()
    dt = 1e-3
    for _ in range(12):
        prod.step(dt)
        ref.step(dt)
    Up = np.array(prod.get_state("gas")).reshape(4, n, n)
    Ur = np.array(ref.get_state("gas")).reshape(4, n, n)
    chk(np.isfinite(Up).all() and Up[0].min() > 0, "(2b) etat production+SSPRK3 physique (fini, rho>0)")
    chk(float(np.max(np.abs(Up - Ur))) == 0.0,
        "(2b) 12 pas production+SSPRK3 BIT-IDENTIQUE au ModelSpec+SSPRK3")

    # (3) NON-TRIVIALITE : production+SSPRK3 DIFFERE de production+SSPRK2 (ssprk3 bien selectionne).
    p2 = build_prod("ssprk2")
    p3 = build_prod("ssprk3")
    for _ in range(12):
        p2.step(dt)
        p3.step(dt)
    U2 = np.array(p2.get_state("gas")).reshape(4, n, n)
    U3 = np.array(p3.get_state("gas")).reshape(4, n, n)
    diff = float(np.max(np.abs(U2 - U3)))
    chk(diff > 0.0,
        "(3) production+SSPRK3 != production+SSPRK2 (ecart %.2e -> ssprk3 effectivement actif)" % diff)


run_compiled_checks()

print("test_ssprk3_production : tout est vert" if fails == 0 else f"{fails} ECHEC(S)")
sys.exit(0 if fails == 0 else 1)
