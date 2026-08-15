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
 (B, avec compilateur -- auto-skip sinon)
  - DSL m.stability_speed(lambda*) : pilote la CFL (dt reduit du ratio attendu) ;
  - DSL m.stability_dt(dt_adm) : borne directe (dt == dt_adm, cfl NON applique,
    last_dt_bound()=="stability_dt:<bloc>").

Invariants par assert ; imprime "OK test_dt_bounds" en cas de succes.
"""
from pops.numerics.reconstruction.limiters import Minmod
from functools import cache
import os
import shutil
import sys
import tempfile

import numpy as np

import pops
import pops.runtime._engine_descriptors as engine
from pops.physics import Density
from pops.physics._facade import Model
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

POPS_PROCESS_TIMEOUT = 600  # Five distinct production DSL artifacts are compiled below.
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


def build_amr(n=24, *, second_block=False):
    amr = AmrSystem(_amr_config_2d(n))
    if second_block:
        amr.set_temporal_relations([2], [1], ["integral_only"])
    amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    amr.add_equation("ions", engine.Model(
                         state=engine.FluidState("isothermal", cs2=0.5),
                         transport=engine.IsothermalFlux(),
                         source=engine.NoSource(),
                         elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0)),
                     spatial=engine.Spatial(limiter=Minmod()),
                     time=engine.Explicit())
    amr.set_density("ions", gaussian(n))
    if second_block:
        amr.add_equation("e2", engine.Model(
                             state=engine.FluidState("isothermal", cs2=0.5),
                             transport=engine.IsothermalFlux(),
                             source=engine.NoSource(),
                             elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0)),
                         spatial=engine.Spatial(limiter=Minmod()),
                         time=engine.Explicit())
        amr.set_density("e2", gaussian(n))
    install_forward_euler_program(amr)
    return amr


amr = build_amr()
chk(amr.last_dt_bound() == "", "AMR avant tout pas : last_dt_bound() == ''")
dta = amr.step_cfl(0.4)
chk(np.isfinite(dta) and dta > 0, f"AMR dt transport fini ({dta:.3e})")
chk(amr.last_dt_bound() == "transport:ions",
    f"AMR borne active = transport:ions (recu {amr.last_dt_bound()!r})")
amr2 = build_amr()
cap_amr = 0.5 * dta
amr2.add_dt_bound("cap_amr", lambda: cap_amr)
dta2 = amr2.step_cfl(0.4)
chk(abs(dta2 - cap_amr) < 1e-15, f"AMR dt == borne globale ({dta2:.3e})")
chk(amr2.last_dt_bound() == "global:cap_amr",
    f"AMR borne active = global:cap_amr (recu {amr2.last_dt_bound()!r})")

print("== (C2) AMR multi-blocs : borne globale via AmrRuntime ==")
amr3 = build_amr(second_block=True)  # 2e bloc -> moteur multi-blocs (AmrRuntime)
amr3.add_dt_bound("cap_multi", lambda: cap_amr)
dta3 = amr3.step_cfl(0.4)
chk(dta3 <= cap_amr + 1e-15, f"AMR multi-blocs dt <= borne ({dta3:.3e})")
chk(amr3.last_dt_bound() == "global:cap_multi",
    f"AMR multi-blocs borne active = global:cap_multi (recu {amr3.last_dt_bound()!r})")

# --- (B) DSL stability_speed / stability_dt (avec compilateur) ---------------------
def scalar_model(name, stab_speed=None, stab_dt=None, src_freq=None):
    """AMR component fixture; Uniform variants use full public artifacts above."""
    model = Model(name)
    (rho,) = model.conservative_vars("rho", roles=[Density()])
    model.flux(x=[1.0 * rho], y=[0.0 * rho])
    model.eigenvalues(x=[1.0 + 0.0 * rho], y=[0.0 * rho])
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.elliptic_rhs(0.0 * rho)
    if stab_speed is not None:
        model.stability_speed(stab_speed + 0.0 * rho)
    if stab_dt is not None:
        model.stability_dt(stab_dt + 0.0 * rho)
    if src_freq is not None:
        model.source([0.0 * rho])
        model.source_frequency(src_freq + 0.0 * rho)
    return model


def build_dsl(artifact, n=16):
    return _build_uniform_artifact(artifact, n)


tmp = tempfile.mkdtemp()
try:
    n, cfl = 16, 0.4
    h = 1.0 / n
    cm_base = _uniform_artifact(n, "scal-base")
    cm_speed = _uniform_artifact(n, "scal-speed", 4.0)
    cm_dt = _uniform_artifact(n, "scal-dt", None, 1e-4)

    print("== (B1) fallback : dt = cfl*h/lambda_max (lambda=1) ==")
    s = build_dsl(cm_base, n)
    dtb = s.step_cfl(cfl)
    chk(abs(dtb - cfl * h / 1.0) < 1e-12, f"dt baseline = cfl*h ({dtb:.3e})")

    print("== (B2) m.stability_speed(4) : dt divise par 4, CFL pilotee par lambda* ==")
    s = build_dsl(cm_speed, n)
    dts = s.step_cfl(cfl)
    chk(abs(dts - cfl * h / 4.0) < 1e-12, f"dt = cfl*h/4 ({dts:.3e})")
    chk(s.last_dt_bound() == "transport:s", "borne active = transport:s (lambda* via max_speed)")

    print("== (B3) m.stability_dt(1e-4) : borne directe, sans cfl ==")
    s = build_dsl(cm_dt, n)
    dtd = s.step_cfl(cfl)
    chk(abs(dtd - 1e-4) < 1e-12, f"dt = 1e-4 ({dtd:.3e})")
    chk(s.last_dt_bound() == "stability_dt:s",
        f"borne active = stability_dt:s (recu {s.last_dt_bound()!r})")

    print("== (B5) m.source_frequency(50) : la 'deuxieme CFL' (source), sans h ==")
    cm_freq = _uniform_artifact(n, "scal-freq", None, None, 50.0)
    s = build_dsl(cm_freq, n)
    dtf = s.step_cfl(cfl)
    chk(abs(dtf - cfl / 50.0) < 1e-12, f"dt = cfl/mu = {cfl / 50.0:.3e} ({dtf:.3e})")
    chk(s.last_dt_bound() == "source_frequency:s",
        f"borne active = source_frequency:s (recu {s.last_dt_bound()!r})")
    try:
        bad = Model("freq_sans_source")
        (r2,) = bad.conservative_vars("rho", roles=[Density()])
        bad.flux(x=[1.0 * r2], y=[0.0 * r2])
        bad.eigenvalues(x=[1.0 + 0.0 * r2], y=[0.0 * r2])
        bad.primitive_vars(r2)
        bad.conservative_from([r2])
        bad.elliptic_rhs(0.0 * r2)
        bad.source_frequency(10.0 + 0.0 * r2)
        bad.check()
        chk(False, "source_frequency sans source aurait du lever")
    except ValueError as e:
        chk("source" in str(e), f"rejet explicite : {str(e)[:70]}")

    print("== (B4) AMR mono-bloc DSL : m.stability_dt cablee (vague 2) ==")
    cm_dt_amr = scalar_model("scal_dt_amr", stab_dt=1e-4).compile(
        os.path.join(tmp, "scal_dt_amr.so"), INCLUDE, backend="production", target="amr_system")
    amr_dsl = AmrSystem(_amr_config_2d(16))
    amr_dsl.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    amr_dsl.add_equation("s", model=cm_dt_amr, spatial=engine.Spatial(limiter=Minmod()),
                         time=engine.Explicit())
    install_forward_euler_program(amr_dsl)
    amr_dsl.set_density("s", gaussian(16))
    dt_amr = amr_dsl.step_cfl(cfl)
    chk(abs(dt_amr - 1e-4) < 1e-12, f"AMR DSL dt = 1e-4 ({dt_amr:.3e})")
    chk(amr_dsl.last_dt_bound() == "stability_dt:s",
        f"AMR borne active = stability_dt:s (recu {amr_dsl.last_dt_bound()!r})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_dt_bounds : {fails} echec(s)")
    sys.exit(1)
print("OK test_dt_bounds")
