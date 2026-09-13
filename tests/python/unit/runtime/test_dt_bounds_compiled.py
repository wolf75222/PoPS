#!/usr/bin/env python3
"""Compiled speed, direct-dt and source-frequency bounds, including AMR.

Keep the five independent stability artifacts out of the global/AMR bounds process
so both groups retain a finite deadline on cold compiler runners.
"""
import os
import shutil
import sys
import tempfile

from pops.numerics.reconstruction.limiters import Minmod
from pops.physics import Density
from pops.physics._facade import Model
import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import AmrSystem
from tests.python.support.explicit_program import install_forward_euler_program
from tests.python.unit.runtime.test_dt_bounds import (
    INCLUDE, _amr_config_2d, _build_uniform_artifact, _uniform_artifact, gaussian,
)

POPS_PROCESS_TIMEOUT = 600
fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


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
    amr_dsl.set_temporal_relations([2], [1], ["integral_only"])
    amr_dsl.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
    amr_dsl.add_equation("s", model=cm_dt_amr, spatial=engine.Spatial(limiter=Minmod()),
                         time=engine.Explicit())
    install_forward_euler_program(amr_dsl)
    amr_dsl.set_density("s", gaussian(16))
    amr_dsl.mark_bound()
    dt_amr = amr_dsl.step_cfl(cfl)
    chk(abs(dt_amr - 1e-4) < 1e-12, f"AMR DSL dt = 1e-4 ({dt_amr:.3e})")
    chk(amr_dsl.last_dt_bound() == "stability_dt:s",
        f"AMR borne active = stability_dt:s (recu {amr_dsl.last_dt_bound()!r})")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_dt_bounds_compiled : {fails} echec(s)")
    sys.exit(1)
print("OK test_dt_bounds_compiled")
