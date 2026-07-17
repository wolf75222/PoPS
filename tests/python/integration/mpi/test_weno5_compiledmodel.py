"""WENO5 parity for a final production component.

The component is compiled through ``Case -> resolve -> compile`` and installed through the one
remaining low-level ``System.add_equation`` seam.  Rusanov with first-order, Minmod and WENO5 is
bit-identical to the equivalent native ModelSpec, including a multi-step WENO5 advance.  This
proves that the production package derives its halo from the selected reconstruction.
"""
from tests.python.support.requirements import require_native_or_skip
from pops.numerics.variables import Conservative
from pops.numerics.riemann import Rusanov
import os
import shutil

import numpy as np

import pops.runtime._engine_descriptors as engine
from test_dsl_coupled import build_euler, compile_euler_component, GAMMA, INCLUDE
from pops.runtime._system import System  # ADC-545 advanced runtime seam
# Multiple DSL native compiles by design: on a slow CI runner the file can exceed the
# global 300 s process-isolation budget (ADC-627, same class as test_dsl_compile_cache).
POPS_PROCESS_TIMEOUT = 900


def _native_spec():
    """Equivalent native Euler bricks used only as the numerical parity oracle."""
    return engine.Model(state=engine.FluidState("compressible", gamma=GAMMA),
                     transport=engine.CompressibleFlux(),
                     source=engine.NoSource(),
                     elliptic=engine.ChargeDensity(charge=1.0))


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


def main():
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        require_native_or_skip('skip  compilateur ou en-tetes pops absents')
        print("test_weno5_compiledmodel : OK (rien a compiler)")
        return

    model = build_euler("weno5-production")
    n, L = 48, 1.0
    U = _initial_state(n)
    Uflat = U.reshape(-1).tolist()
    spec = _native_spec()

    def run_compiled_checks():
        compiled_prod = compile_euler_component(model, cells=16, cxx=cxx)
        assert compiled_prod.backend == "production"

        # --- native ModelSpec reference (numerical parity oracle) ---
        def ref(limiter):
            sys = System(n=n, L=L, periodic=True)
            lim = {"none": dict(none=True), "minmod": dict(minmod=True),
                   "weno5": dict(weno5=True)}[limiter]
            sys.add_equation("gas", spec, spatial=engine.Spatial(flux=Rusanov(), recon=Conservative(),
                                                                    **lim), time=engine.Explicit())
            sys.set_state("gas", Uflat)
            return np.array(sys.eval_rhs("gas")).reshape(4, n, n)

        # --- final production component: strict bit-identical WENO5 parity ---
        def prod(limiter):
            sys = System(n=n, L=L, periodic=True)
            lim = {
                "none": dict(none=True),
                "minmod": dict(minmod=True),
                "weno5": dict(weno5=True),
            }[limiter]
            sys.add_equation(
                "gas",
                compiled_prod,
                spatial=engine.Spatial(
                    flux=Rusanov(), recon=Conservative(), **lim
                ),
                time=engine.Explicit(),
            )
            sys.set_state("gas", Uflat)
            return np.array(sys.eval_rhs("gas")).reshape(4, n, n)

        for limiter in ("none", "minmod", "weno5"):
            R_ref = ref(limiter)
            R_prod = prod(limiter)
            assert float(np.max(np.abs(R_prod))) > 1e-3, "%s : residu production trivial" % limiter
            dres = float(np.max(np.abs(R_prod - R_ref)))
            # Both routes reach the same native block and halo provider.
            assert dres == 0.0, "production %s : eval_rhs != add_block (%.2e, attendu 0)" % (limiter,
                                                                                            dres)
            print("OK  production %s : eval_rhs BIT-IDENTIQUE au ModelSpec"
                  % limiter)

        # avance production weno5 : etat final bit-identique au natif sur 12 pas a dt fixe.
        def build_prod_step():
            sys = System(n=n, L=L, periodic=True)
            sys.add_equation(
                "gas",
                compiled_prod,
                spatial=engine.Spatial(
                    weno5=True, flux=Rusanov(), recon=Conservative()
                ),
                time=engine.Explicit(),
            )
            sys.set_state("gas", Uflat)
            return sys

        def build_ref_step():
            sys = System(n=n, L=L, periodic=True)
            sys.add_equation("gas", spec, spatial=engine.Spatial(weno5=True, flux=Rusanov(),
                                                                    recon=Conservative()),
                             time=engine.Explicit())
            sys.set_state("gas", Uflat)
            return sys

        p_sys, r_sys = build_prod_step(), build_ref_step()
        dt = 1e-3
        for _ in range(12):
            p_sys.step(dt)
            r_sys.step(dt)
        Up = np.array(p_sys.get_state("gas")).reshape(4, n, n)
        Ur = np.array(r_sys.get_state("gas")).reshape(4, n, n)
        dstep = float(np.max(np.abs(Up - Ur)))
        assert np.isfinite(Up).all() and Up[0].min() > 0, "production weno5 : etat non physique"
        assert dstep == 0.0, "production weno5 : etat apres 12 pas != add_block (%.2e)" % dstep
        print("OK  production weno5 : 12 pas SSPRK2 BIT-IDENTIQUES au bloc natif add_block")

        print("test_weno5_compiledmodel : tout est vert")

    run_compiled_checks()


if __name__ == "__main__":
    main()
