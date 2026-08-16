"""Prepared production-package runtime, public-bind parity, and ABI refusal.

One final board model is compiled through ``Case -> validate -> resolve -> compile``.  Its detached
component is materialized only through the authenticated lane/state-route boundary.  The runtime
matches the public bind path bit-for-bit for the artifact's authored Rusanov/MUSCL plan, while an
alternate HLLC/primitive installation preserves a stationary Euler contact that Rusanov
necessarily diffuses.  A deliberately recompiled package with a mismatched header signature must
be rejected at the native boundary.
"""
import os
import shutil
import tempfile

import numpy as np

import pops
import pops.runtime._engine_descriptors as engine
from pops.codegen.loader import CompiledModel
from test_dsl_coupled import build_euler, compile_euler_artifact, GAMMA, INCLUDE
from pops.runtime._system import System, SystemConfig  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    require_native_or_skip,
)
# Multiple DSL native compiles by design: on a slow CI runner the file can exceed the
# global 300 s process-isolation budget (ADC-627, same class as test_dsl_compile_cache).
POPS_PROCESS_TIMEOUT = 900


def _system_config_2d(n, *, length=1.0):
    config = SystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (float(length), float(length))
    config.periodicity = (True, True)
    config.boxes = (((0, 0), (n, n)),)
    return config


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


def _stationary_contact(n):
    """Periodic constant-pressure contact: exact for HLLC, diffusive for Rusanov."""
    U = np.zeros((4, n, n))
    U[0, : n // 2, :] = 1.0
    U[0, n // 2 :, :] = 2.0
    U[3] = 1.0 / (GAMMA - 1.0)
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
    tmp = tempfile.mkdtemp()
    try:
        # The component package and its state/lane identities come from one full public artifact.
        artifact = compile_euler_artifact(model, cells=n, cxx=cxx)
        compiled = artifact.blocks[0].model
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

        def build_native(limiter, riemann, recon, *, initial=U):
            context = artifact_execution_context(artifact)
            sys = System(_system_config_2d(n, length=L))
            sys._s._prepare_boundary_execution_lane(
                context.communicator.handle,
                context.identity.token,
            )
            (state_identity,) = artifact.plan.blocks[0].state_identities
            sys._s._install_block_state_route("gas", state_identity)
            sys.add_equation(
                "gas", model=compiled, spatial=spatial(limiter, riemann, recon),
                time=engine.Explicit(),
            )
            if sys._pending_native_packages:
                sys._s._finalize_native_packages()
                sys._pending_native_packages = 0
            sys.set_state("gas", np.asarray(initial).reshape(-1).tolist())
            install_forward_euler_program(sys)
            return sys

        def compare(limiter, riemann, recon):
            first = build_native(limiter, riemann, recon)
            second = build_native(limiter, riemann, recon)
            first_rhs = np.array(first.eval_rhs("gas")).reshape(4, n, n)
            second_rhs = np.array(second.eval_rhs("gas")).reshape(4, n, n)
            assert float(np.max(np.abs(first_rhs))) > 1e-3, "%s : residu trivial" % riemann
            assert np.array_equal(first_rhs, second_rhs), (
                "%s : le package prepare depend de l'instance runtime" % riemann
            )
            print("OK  bloc production %s+%s : eval_rhs prepare deterministe"
                  % (limiter, riemann))

        compare("minmod", "rusanov", "conservative")

        # HLLC must preserve a stationary contact exactly: pressure and velocity are uniform, so
        # the physical flux is identical on both sides even though density jumps.  Rusanov is an
        # independent control because its scalar dissipation necessarily moves density here.  This
        # fails if the HLLC selection deterministically falls back to Rusanov.
        contact = _stationary_contact(n)
        hllc_contact = build_native(
            "none", "hllc", "primitive", initial=contact
        )
        rusanov_contact = build_native(
            "none", "rusanov", "primitive", initial=contact
        )
        hllc_rhs = np.array(hllc_contact.eval_rhs("gas")).reshape(4, n, n)
        rusanov_rhs = np.array(rusanov_contact.eval_rhs("gas")).reshape(4, n, n)
        assert float(np.max(np.abs(hllc_rhs))) <= 1e-12
        assert float(np.max(np.abs(rusanov_rhs[0]))) > 1e-3
        for _ in range(12):
            hllc_contact.step(1e-3)
        hllc_state = np.array(hllc_contact.get_state("gas")).reshape(4, n, n)
        assert np.array_equal(hllc_state, contact)
        print("OK  HLLC preserve le contact stationnaire que Rusanov diffuse")

        # (2) le plan Rusanov/MUSCL authentifie est identique entre le package detache prepare et
        # le lifecycle public bind/run.  C'est la reference supportee depuis le retrait de ModelSpec.
        prod = build_native("minmod", "rusanov", "conservative")
        public = pops.bind(artifact, initial_state={"gas": np.ascontiguousarray(U)})
        dt = 1e-4
        for _ in range(12):
            prod.step(dt)
        report = pops.run(public, t_end=12 * dt, max_steps=12)
        Up = np.array(prod.get_state("gas")).reshape(4, n, n)
        Ur = np.array(public.state_global("gas")).reshape(4, n, n)
        assert report.accepted_steps == 12
        assert np.isfinite(Up).all() and Up[0].min() > 0, "etat de production non physique"
        assert float(np.abs(Up[1]).max()) > 1e-4, "le transport Euler est reste trivial"
        assert np.array_equal(Up, Ur), "package prepare != public bind apres 12 pas"
        print("OK  12 pas Forward-Euler : package prepare BIT-IDENTIQUE au public bind")

        # (3) GARDE-FOU ABI : on compile un loader dont la SIGNATURE D'EN-TETES bakee est volontairement
        # FAUSSE (-DPOPS_HEADER_SIG different). Sa cle pops_native_abi_key differe alors de celle du module
        # -> add_native_block doit lever une erreur EXPLICITE. (On ne patche PAS le binaire : sur macOS
        # ARM cela invaliderait la signature ad-hoc et le noyau tuerait le process ; on recompile un .so
        # valide a la cle differente, ce qui teste exactement la frontiere d'ABI.)
        bad = _compile_wrong_abi(model, os.path.join(tmp, "euler_wrongabi.so"), cxx)
        bad_component = _component_at(compiled, bad)
        sys = System(_system_config_2d(n, length=L))
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
        native_dimension=component.native_dimension,
        hllc=component.has_hllc,
        roe=component.has_roe,
        provider_components=component.provider_components,
        wave_speeds=component.has_wave_speeds,
        wave_speed_provider=component.wave_speed_provider,
        characteristic_no_inflow=component.has_characteristic_no_inflow,
        hllc_provider=component.hllc_provider,
        roe_provider=component.roe_provider,
        roe_entropy_policy=component.roe_entropy_policy,
        roe_entropy_delta=component.roe_entropy_delta,
        elliptic_field_names=component.elliptic_field_names,
        bind_schema=component.bind_schema,
        definition_identity=component.definition_identity,
        module_manifest=component.module_manifest,
    )


if __name__ == "__main__":
    main()
