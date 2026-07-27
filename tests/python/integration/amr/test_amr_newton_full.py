#!/usr/bin/env python3
"""AMR source-Newton requests fail closed until a typed Program primitive exists.

The AMR block descriptor may still carry the canonical ``IMEX`` authoring token,
but the spatial runtime owns neither a local implicit step nor a Newton report.
Non-default Newton controls, diagnostics, and partial masks must therefore be
rejected instead of being accepted and ignored.
"""

import sys

import pops.runtime._engine_descriptors as engine
from pops.codegen.abi import module_header_signature
from pops.codegen.loader import CompiledModel
from pops.runtime._system import AmrSystem

fails = 0


def chk(condition, label):
    global fails
    print(f"  [{'OK ' if condition else 'XX '}] {label}")
    if not condition:
        fails += 1


def iso_model():
    return engine.Model(
        state=engine.FluidState("isothermal", cs2=0.5),
        transport=engine.IsothermalFlux(),
        source=engine.NoSource(),
        elliptic=engine.BackgroundDensity(alpha=0.0, n0=0.0),
    )


def fresh_amr():
    system = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
    system.set_temporal_relations([2], [1], ["integral_only"])
    return system


def expect_native_rejection(label, time, *, newton=False):
    system = fresh_amr()
    try:
        system.add_equation("gas", iso_model(), spatial=engine.Spatial(), time=time)
    except (RuntimeError, ValueError) as error:
        message = str(error)
        chk(
            "Newton" in message or "implicit" in message,
            f"{label}: rejet explicite et actionnable",
        )
        if newton:
            chk(
                "LocalLinear" in message and "uniform System" in message,
                f"{label}: alternatives executables indiquees",
            )
        return
    chk(False, f"{label}: la requete non executable aurait du etre rejetee")


def compiled_amr_metadata():
    """Detached metadata used to prove the Python pre-loader guard."""
    return CompiledModel(
        so_path="/inexistant_amr.so",
        backend="production",
        cons_names=["rho", "rho_u", "rho_v"],
        cons_roles=["Density", "MomentumX", "MomentumY"],
        prim_names=["rho", "u", "v"],
        n_vars=3,
        gamma=1.4,
        n_aux=3,
        params={},
        caps={},
        abi_key=f"{module_header_signature()}|c++|c++23",
        model_hash="amr-newton-preloader-guard",
        cxx="c++",
        std="c++23",
        target="amr_system",
    )


print("== AMR source Newton: current unsupported requests fail closed ==")

# The default token remains valid authoring metadata. It does not claim that the
# spatial runtime executed a hidden backward-Euler/Newton step.
default_system = fresh_amr()
default_system.add_equation("gas", iso_model(), spatial=engine.Spatial(), time=engine.IMEX())
chk(True, "IMEX par defaut reste un descripteur d'auteur valide")

newton_knobs = (
    ("newton_max_iters", {"newton_max_iters": 5}),
    ("newton_rel_tol", {"newton_rel_tol": 1e-12}),
    ("newton_abs_tol", {"newton_abs_tol": 1e-12}),
    ("newton_fd_eps", {"newton_fd_eps": 2e-7}),
    ("newton_damping", {"newton_damping": 0.5}),
)

for knob, values in newton_knobs:
    expect_native_rejection(
        f"option Newton native non-defaut {knob}",
        engine.IMEX(**values),
        newton=True,
    )
expect_native_rejection(
    "diagnostics Newton natifs",
    engine.IMEX(newton_diagnostics=True),
    newton=True,
)
expect_native_rejection(
    "masque implicite natif",
    engine.IMEX(implicit_vars=["rho_u"]),
)

compiled_requests = tuple(
    (f"option Newton du package {knob}", engine.IMEX(**values)) for knob, values in newton_knobs
) + (("diagnostics Newton du package", engine.IMEX(newton_diagnostics=True)),)

for label, time in compiled_requests:
    system = fresh_amr()
    try:
        system.add_equation("gas", compiled_amr_metadata(), spatial=engine.Spatial(), time=time)
    except ValueError as error:
        message = str(error)
        chk(
            "Newton" in message and "production" in message,
            f"{label}: garde pre-loader claire",
        )
        chk(
            "LocalLinear" in message and "uniform System" in message,
            f"{label}: alternatives executables indiquees",
        )
    else:
        chk(False, f"{label}: la garde pre-loader aurait du rejeter")

chk(
    not hasattr(default_system, "newton_report"),
    "aucun faux rapport Newton n'est expose par le runtime spatial AMR",
)

print("FAIL test_amr_newton_full" if fails else "OK test_amr_newton_full")
sys.exit(1 if fails else 0)
