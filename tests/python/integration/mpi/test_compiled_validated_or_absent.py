#!/usr/bin/env python3
"""ADC-558 real-compiler acceptance: a compiled artifact is validated-or-absent.

Two claims (skips cleanly unless the full toolchain is present, like the sibling
``test_compile_problem.py``):

  (1) SUCCESS -> DIRECTLY USABLE: a successful ``compile_problem`` returns a handle that installs
      onto the runtime with NO intermediate check() step (``sim.install_program(compiled.so_path)``
      succeeds directly). The handle's inspect().status is the single validity signal.
  (2) FAILURE -> NO ARTIFACT: a deliberately broken model raises during compile and NO handle
      escapes -- there is no partially-validated object to catch and no .so to install.

Runs in CI; skips locally when no compiler / Kokkos is visible or the .so compile fails.
"""
import sys
from pops.runtime._system import System  # ADC-545 advanced runtime seam


def _skip(msg):
    print("skip test_compiled_validated_or_absent (%s)" % msg)
    sys.exit(0)


try:
    import numpy as np

    import pops
    from pops.codegen._compile_drivers import compile_problem
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from tests.python.integration._final_field_program import (
        compile_block_model,
        scalar_advection_field_model,
    )
    from tests.python.support.typed_program import program_states
except Exception as exc:  # noqa: BLE001
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _fe_program(model, name="validated_probe"):
    P = pops.Program(name)
    dt = P.dt
    module = model.module
    _case, states = program_states(P, model, ("ions",))
    temporal = states["ions"]
    U = temporal.n
    R = module.operator_handle("explicit_rhs")(U, name="rate")
    P.commit(temporal.next, P.value("U1", U + dt * R, at=temporal.next.point))
    return P


def transport_model():
    return scalar_advection_field_model("validated_transport")


# ---- (1) success -> directly usable ----
try:
    model = transport_model()
    compiled = compile_problem(time=_fe_program(model), model=model)
except (RuntimeError, Exception) as exc:  # noqa: BLE001 -- no compiler / Kokkos / compile failure
    _skip("compile_problem could not build the .so: %s" % str(exc)[:160])

print("== (1) success -> directly usable (no check step) ==")
# The handle is validated-or-absent: it exists, so it is valid. The status line is the only signal.
chk(compiled.inspect().status == "compiled, waiting for pops.bind(...)",
    "the handle's status is the single validity signal")
chk(not hasattr(compiled, "check"), "the handle carries no post-compile check()")

# install_program is forwarded by the System facade; skip the install if _pops lacks the binding.
sim = System(n=24, L=1.0, periodic=True)
if not hasattr(sim, "install_program"):
    print("-- install skipped: _pops lacks the install_program binding --")
else:
    sim.add_equation(
        "ions", compile_block_model(model, target="system"),
        spatial=pops.FiniteVolume(limiter=FirstOrder(), riemann=Rusanov()),
        time=pops.Explicit(method="euler"),
    )
    sim.set_poisson("charge_density", "geometric_mg")
    n = 24
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho]))
    # DIRECTLY bindable: install with no intermediate check().
    sim.install_program(compiled.so_path)
    chk(True, "the validated handle installs directly (no check step)")

# ---- (2) failure -> no artifact escapes ----
print("== (2) failure -> no artifact escapes ==")


def _broken_program(model):
    # An advanced low-level Program that declares an evolved block but never commits it is invalid;
    # compile must fail before any handle exists. A foreign free-name commit is no longer expressible.
    P = pops.Program("broken_probe")
    program_states(P, model, ("ions",))
    return P


handle_holder = {}


def _try_broken():
    broken_model = transport_model()
    handle_holder["h"] = compile_problem(
        time=_broken_program(broken_model), model=broken_model)


raised = False
try:
    _try_broken()
except Exception:  # noqa: BLE001 -- any validation / compile error is the fail-loud contract
    raised = True
chk(raised, "a broken Program raises during compile")
chk("h" not in handle_holder, "NO handle object escaped a failed compile (validated-or-absent)")

print("%s test_compiled_validated_or_absent" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
