#!/usr/bin/env python3
"""Politique de pas GENERIQUE de step_cfl (audit 2026-06, chantier 1).

step_cfl n'est plus une formule transport-only cachee : il AGREGE des bornes par bloc
(stability_speed / stability_dt compilees par le DSL, source_frequency cote C++) et des bornes
GLOBALES (sim.add_dt_bound, hote, une evaluation par pas), avec fallback STRICTEMENT historique
(transport max_wave_speed) quand aucune borne optionnelle n'existe. La borne ACTIVE est consultable
via sim.last_dt_bound().

Verifie :
 (A, composant public compile)
  - NO-DEFAULT-CHANGE : sans borne optionnelle, dt identique et last_dt_bound()=="transport:<bloc>" ;
  - add_dt_bound contraint step_cfl (dt == borne, last_dt_bound()=="global:<label>") ;
  - une borne lache (1e9) / non-positive (-1) ne contraint PAS (dt inchange) ;
 Les variantes DSL de stabilite sont verifiees dans test_dt_bounds_compiled.py.

Invariants par assert ; imprime "OK test_dt_bounds" en cas de succes.
"""
from pops.numerics.reconstruction.limiters import Minmod
from functools import cache
import sys

import numpy as np

import pops
import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import (  # ADC-545 advanced runtime seam
    AmrSystem,
    AmrSystemConfig,
    System,
    SystemConfig,
)
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.integration._final_field_program import (
    density_advection_model,
    forward_euler_program,
    resolve_periodic_field_program,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)

POPS_PROCESS_TIMEOUT = 600  # Global and AMR bounds; compiled stability variants are separate.
INCLUDE = repo_include()
fails = 0

missing = missing_native_compile_requirement(INCLUDE, default_cxx())
if missing:
    require_native_or_skip("test_dt_bounds: %s" % missing)


def _system_config_2d(n):
    config = SystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.boxes = (((0, 0), (n, n)),)
    return config


def _amr_config_2d(n):
    config = AmrSystemConfig()
    config.shape = (n, n)
    config.lower = (0.0, 0.0)
    config.upper = (1.0, 1.0)
    config.periodicity = (True, True)
    config.boxes = (((0, 0), (n, n)),)
    config.regrid_every = 0
    return config


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.5 * np.exp(-80.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


@cache
def _uniform_artifact(
    n,
    name,
    stability_speed=None,
    stability_dt=None,
    source_frequency=None,
):
    model = density_advection_model(
        "%s-%d" % (name, n),
        speed=1.0,
        stability_speed=stability_speed,
        stability_dt=stability_dt,
        source_frequency=source_frequency,
    )
    resolved = resolve_periodic_field_program(
        model,
        forward_euler_program,
        name="%s-%d" % (name, n),
        block_name="s",
        target="system",
        n=n,
        cxx=default_cxx(),
        include=INCLUDE,
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    return artifact


def _build_uniform_artifact(artifact, n, *, block_name="s"):
    context = artifact_execution_context(artifact)
    sim = System(_system_config_2d(n))
    sim._s._prepare_boundary_execution_lane(
        context.communicator.handle,
        context.identity.token,
    )
    (state_identity,) = artifact.plan.blocks[0].state_identities
    sim._s._install_block_state_route(block_name, state_identity)
    sim.add_equation(
        block_name,
        artifact.blocks[0].model,
        spatial=engine.Spatial(limiter=Minmod()),
        time=engine.Explicit(),
    )
    if sim._pending_native_packages:
        sim._s._finalize_native_packages()
        sim._pending_native_packages = 0
    sim.set_poisson(rhs="charge_density", solver="cartesian_cg", bc=Periodic())
    sim.set_density(block_name, gaussian(n).ravel())
    install_forward_euler_program(sim)
    return sim


def build(n=24):
    return _build_uniform_artifact(
        _uniform_artifact(n, "dt-bounds-transport"),
        n,
        block_name="ions",
    )


def build_amr(n=24, *, second_block=False):
    amr = AmrSystem(_amr_config_2d(n))
    from pops.runtime._modelspec_compile import compile_modelspec_package
    packages = {}
    for name in (("ions", "e2") if second_block else ("ions",)):
        packages[name] = compile_modelspec_package(
            engine.Model(
                state=engine.FluidState("isothermal", cs2=0.5),
                transport=engine.IsothermalFlux(), source=engine.NoSource(),
                elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0),
            ), name=name, target="amr_system",
        )
    from pops.runtime._amr_package_lane import ensure_native_block_state_route
    for name, package in packages.items():
        ensure_native_block_state_route(amr._s, name, package)
    amr.set_temporal_relations([2], [1], ["integral_only"])
    amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    amr.add_equation("ions", packages["ions"],
                     spatial=engine.Spatial(limiter=Minmod()),
                     time=engine.Explicit())
    amr.set_density("ions", gaussian(n))
    if second_block:
        amr.add_equation("e2", packages["e2"],
                         spatial=engine.Spatial(limiter=Minmod()),
                         time=engine.Explicit())
        amr.set_density("e2", gaussian(n))
    install_forward_euler_program(amr)
    return amr


def main():
    # --- (A) bornes globales + raison, composant public compile -----------------------
    print("== (A1) fallback historique : transport seul ==")
    sim = build()
    chk(sim.last_dt_bound() == "", "avant tout pas : last_dt_bound() == ''")
    dt0 = sim.step_cfl(0.4)
    chk(np.isfinite(dt0) and dt0 > 0, f"dt transport fini ({dt0:.3e})")
    chk(sim.last_dt_bound() == "transport:ions",
        f"borne active = transport:ions (recu {sim.last_dt_bound()!r})")

    print("== (A2) add_dt_bound contraint le pas ==")
    cap = 0.5 * dt0
    sim2 = build()
    sim2.add_dt_bound("cap_test", lambda: cap)
    dt2 = sim2.step_cfl(0.4)
    chk(abs(dt2 - cap) < 1e-15, f"dt == borne globale ({dt2:.3e} vs {cap:.3e})")
    chk(sim2.last_dt_bound() == "global:cap_test",
        f"borne active = global:cap_test (recu {sim2.last_dt_bound()!r})")

    print("== (A3) bornes laches / non-positives : ne contraignent pas ==")
    sim3 = build()
    sim3.add_dt_bound("loose", lambda: 1e9)
    sim3.add_dt_bound("inactive", lambda: -1.0)
    dt3 = sim3.step_cfl(0.4)
    chk(abs(dt3 - dt0) < 1e-15, "dt inchange (bornes inactives)")
    chk(sim3.last_dt_bound() == "transport:ions", "borne active reste transport")

    # --- (C) AMR : StabilityPolicy cablee (audit vague 2) -------------------------------
    print("== (C1) AMR mono-bloc : transport + borne globale + last_dt_bound ==")


    amr = build_amr()
    chk(amr.last_dt_bound() == "", "AMR avant tout pas : last_dt_bound() == ''")
    amr.mark_bound()
    dta = amr.step_cfl(0.4)
    chk(np.isfinite(dta) and dta > 0, f"AMR dt transport fini ({dta:.3e})")
    chk(amr.last_dt_bound() == "transport:ions",
        f"AMR borne active = transport:ions (recu {amr.last_dt_bound()!r})")
    amr2 = build_amr()
    cap_amr = 0.5 * dta
    amr2.add_dt_bound("cap_amr", lambda: cap_amr)
    amr2.mark_bound()
    dta2 = amr2.step_cfl(0.4)
    chk(abs(dta2 - cap_amr) < 1e-15, f"AMR dt == borne globale ({dta2:.3e})")
    chk(amr2.last_dt_bound() == "global:cap_amr",
        f"AMR borne active = global:cap_amr (recu {amr2.last_dt_bound()!r})")

    print("== (C2) AMR multi-blocs : borne globale via AmrRuntime ==")
    amr3 = build_amr(second_block=True)  # 2e bloc -> moteur multi-blocs (AmrRuntime)
    amr3.add_dt_bound("cap_multi", lambda: cap_amr)
    amr3.mark_bound()
    dta3 = amr3.step_cfl(0.4)
    chk(dta3 <= cap_amr + 1e-15, f"AMR multi-blocs dt <= borne ({dta3:.3e})")
    chk(amr3.last_dt_bound() == "global:cap_multi",
        f"AMR multi-blocs borne active = global:cap_multi (recu {amr3.last_dt_bound()!r})")


    if fails:
        print(f"FAIL test_dt_bounds : {fails} echec(s)")
        sys.exit(1)
    print("OK test_dt_bounds")


if __name__ == "__main__":
    main()
