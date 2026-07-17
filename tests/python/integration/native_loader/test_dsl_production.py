"""Native production-package parity and ABI refusal.

One final board model is compiled through ``Case -> validate -> resolve -> compile``.  Its detached
component is bit-identical to the equivalent ModelSpec for Rusanov and HLLC and over a multi-step
advance.  A deliberately recompiled package with a mismatched header signature must be rejected at
the authenticated native boundary.
"""
import os
import shutil
import tempfile

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.codegen.loader import CompiledModel
from test_dsl_coupled import build_euler, compile_euler_component, GAMMA, INCLUDE
from pops.runtime._system import System  # ADC-545 advanced runtime seam
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    require_native_or_skip,
)
# Multiple DSL native compiles by design: on a slow CI runner the file can exceed the
# global 300 s process-isolation budget (ADC-627, same class as test_dsl_compile_cache).
POPS_PROCESS_TIMEOUT = 900


def _native_spec():
    """Equivalent native Euler bricks used as the numerical parity oracle."""
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
    cxx = default_cxx()
    missing = missing_native_compile_requirement(INCLUDE, cxx)
    if missing is not None:
        require_native_or_skip(missing)
    assert cxx is not None

    model = build_euler("production-parity")
    n, L = 48, 1.0
    U = _initial_state(n)
    Uflat = U.reshape(-1).tolist()
    spec = _native_spec()
    tmp = tempfile.mkdtemp()
    try:
        # The component package is produced only by the final public lifecycle.
        compiled = compile_euler_component(model, cells=16, cxx=cxx)
        assert compiled.backend == "production"

        def spatial(limiter, riemann, recon):
            from pops.numerics.riemann import Rusanov, HLL, HLLC, Roe
            from pops.numerics.reconstruction import FirstOrder
            from pops.numerics.reconstruction.limiters import Minmod, VanLeer
            from pops.numerics.variables import Conservative, Primitive
            return engine.Spatial(
                limiter={"none": FirstOrder(), "minmod": Minmod(), "vanleer": VanLeer()}[limiter],
                flux={"rusanov": Rusanov(), "hll": HLL(), "hllc": HLLC(), "roe": Roe()}[riemann],
                recon={"conservative": Conservative(), "primitive": Primitive()}[recon],
            )

        def build_native(limiter, riemann, recon, evolve=True):
            sys = System(n=n, L=L, periodic=True)
            sys.add_equation(
                "gas", model=compiled, spatial=spatial(limiter, riemann, recon),
                time=engine.Explicit(), evolve=evolve,
            )
            sys.set_state("gas", Uflat)
            return sys

        def build_ref(limiter, riemann, recon, evolve=True):
            sys = System(n=n, L=L, periodic=True)
            sys.add_equation("gas", spec,
                             spatial=spatial(limiter, riemann, recon),
                             time=engine.Explicit(), evolve=evolve)
            sys.set_state("gas", Uflat)
            return sys

        def compare(limiter, riemann, recon):
            prod = build_native(limiter, riemann, recon)
            R_prod = np.array(prod.eval_rhs("gas")).reshape(4, n, n)

            ref = build_ref(limiter, riemann, recon)
            R_ref = np.array(ref.eval_rhs("gas")).reshape(4, n, n)

            assert float(np.max(np.abs(R_prod))) > 1e-3, "%s : residu trivial" % riemann
            dres = float(np.max(np.abs(R_prod - R_ref)))
            # Parite STRICTE : meme chemin compile (install_block), donc bit-identique (pas seulement < 1e-9).
            assert dres == 0.0, "%s : eval_rhs natif != add_block (ecart %.2e, attendu 0)" % (riemann, dres)
            print("OK  bloc production %s+%s : eval_rhs BIT-IDENTIQUE a add_equation(ModelSpec)"
                  % (limiter, riemann))

        compare("minmod", "rusanov", "conservative")
        compare("minmod", "hllc", "primitive")  # flux de production (pressure()/wave_speeds() generes)

        # (2) avance SSPRK2 : etat final bit-identique au bloc natif sur 12 pas a dt fixe (meme dt des
        # deux cotes -> pas de derive numerique possible si la numerique est la meme).
        prod = build_native("minmod", "hllc", "primitive")
        ref = build_ref("minmod", "hllc", "primitive")
        dt = 1e-3
        for _ in range(12):
            prod.step(dt)
            ref.step(dt)
        Up = np.array(prod.get_state("gas")).reshape(4, n, n)
        Ur = np.array(ref.get_state("gas")).reshape(4, n, n)
        dstep = float(np.max(np.abs(Up - Ur)))
        assert np.isfinite(Up).all() and Up[0].min() > 0, "etat de production non physique"
        assert float(np.abs(Up[1]).max()) > 1e-4, "le transport Euler est reste trivial"
        assert dstep == 0.0, "etat apres 12 pas natif != add_block (ecart %.2e, attendu 0)" % dstep
        print("OK  12 pas SSPRK2 : etat de production BIT-IDENTIQUE au bloc natif add_block")

        # (3) GARDE-FOU ABI : on compile un loader dont la SIGNATURE D'EN-TETES bakee est volontairement
        # FAUSSE (-DPOPS_HEADER_SIG different). Sa cle pops_native_abi_key differe alors de celle du module
        # -> add_native_block doit lever une erreur EXPLICITE. (On ne patche PAS le binaire : sur macOS
        # ARM cela invaliderait la signature ad-hoc et le noyau tuerait le process ; on recompile un .so
        # valide a la cle differente, ce qui teste exactement la frontiere d'ABI.)
        bad = _compile_wrong_abi(model, os.path.join(tmp, "euler_wrongabi.so"), cxx)
        bad_component = _component_at(compiled, bad)
        sys = System(n=n, L=L, periodic=True)
        raised = False
        try:
            sys.add_equation(
                "gas",
                bad_component,
                spatial=spatial("minmod", "rusanov", "conservative"),
                time=engine.Explicit(),
            )
        except RuntimeError as ex:
            raised = True
            assert "incompatible native ABI" in str(ex), "message inattendu : %s" % ex
        assert raised, "add_native_block a accepte un loader a cle d'ABI fausse (UB silencieux)"
        print("OK  cle d'ABI divergente REJETEE explicitement par add_native_block")

        print("test_dsl_production : tout est vert")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compile_wrong_abi(model, dst_so, cxx):
    """Compile le MEME loader natif mais avec une signature d'en-tetes FAUSSE (-DPOPS_HEADER_SIG bidon) :
    le .so produit est valide (signe par le compilateur) mais sa cle d'ABI differe de celle du module,
    ce qui doit declencher le rejet d'add_native_block. Renvoie le chemin du .so."""
    import subprocess
    import tempfile
    from pops.codegen.toolchain import pops_loader_build_flags
    lowering = model.__pops_compiler_lowering__()
    src = lowering.native_loader_source()
    # PoPS est Kokkos-only : le loader inclut les en-tetes pops (for_each), il faut donc Kokkos +
    # (macOS) -undefined dynamic_lookup. pops_loader_build_flags fournit compilateur + flags ; on garde
    # une SIGNATURE D'EN-TETES FAUSSE (-DPOPS_HEADER_SIG bidon) pour que le .so compile mais soit REJETE
    # a l'ABI par add_native_block (le but du test).
    cc, kflags_c, kflags_l = pops_loader_build_flags(cxx)
    flags = ["-shared", "-fPIC", "-std=c++20", "-O2",
             "-DPOPS_HEADER_SIG=\"deadbeef_signature_volontairement_fausse\"", *kflags_c]
    with tempfile.TemporaryDirectory() as t:
        cpp = os.path.join(t, "wrong.cpp")
        with open(cpp, "w") as f:
            f.write(src)
        subprocess.run([cc, *flags, "-I", INCLUDE, cpp, "-o", dst_so, *kflags_l], check=True)
    return dst_so


def _component_at(component, so_path):
    """Detach valid metadata while substituting the deliberately bad package path."""
    return CompiledModel(
        so_path=so_path,
        backend=component.backend,
        target=component.target,
        cons_names=component.cons_names,
        state_spaces=component.state_spaces,
        cons_roles=component.cons_roles,
        prim_names=component.prim_names,
        n_vars=component.n_vars,
        gamma=component.gamma,
        n_aux=component.n_aux,
        params=component.params,
        caps=component.caps,
        abi_key=component.abi_key,
        model_hash=component.model_hash,
        cxx=component.cxx,
        std=component.std,
        hllc=component.has_hllc,
        roe=component.has_roe,
        aux_extra_names=component.aux_extra_names,
        wave_speeds=component.has_wave_speeds,
        wave_speed_provider=component.wave_speed_provider,
        elliptic_field_names=component.elliptic_field_names,
        definition_identity=component.definition_identity,
    )


if __name__ == "__main__":
    main()
