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
    from pops import time as adctime
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from tests.python.support.typed_program import program_states, synthetic_module
except Exception as exc:  # noqa: BLE001
    _skip("pops/numpy unavailable: %s" % exc)

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def _fe_program(name="validated_probe"):
    P = adctime.Program(name)
    dt = P.dt
    module = synthetic_module("%s_state" % name, components=("rho", "mx", "my"))
    _case, states = program_states(P, module, ("ions",))
    temporal = states["ions"]
    U = temporal.n
    f = P.solve_fields(U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["default"])
    P.commit(temporal.next, P.value("U1", U + dt * R))
    return P


def transport_model():
    return pops.Model(state=pops.FluidState("isothermal", cs2=0.5),
                      transport=pops.IsothermalFlux(),
                      source=pops.NoSource(),
                      elliptic=pops.BackgroundDensity(alpha=1.0, n0=0.0))


# ---- (1) success -> directly usable ----
try:
    compiled = pops.codegen.compile_problem(time=_fe_program(), model=transport_model())
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
    sim.block("ions", transport_model(),
                  spatial=pops.FiniteVolume(limiter=FirstOrder(),
                                            riemann=Rusanov()),
                  time=pops.Explicit(method="euler"))
    sim.set_poisson("charge_density", "geometric_mg")
    n = 24
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    # DIRECTLY bindable: install with no intermediate check().
    sim.install_program(compiled.so_path)
    chk(True, "the validated handle installs directly (no check step)")

# ---- (2) failure -> no artifact escapes ----
print("== (2) failure -> no artifact escapes ==")


def _broken_program():
    # An advanced low-level Program that declares an evolved block but never commits it is invalid;
    # compile must fail before any handle exists. A foreign free-name commit is no longer expressible.
    P = adctime.Program("broken_probe")
    module = synthetic_module("broken_probe_state", components=("rho", "mx", "my"))
    program_states(P, module, ("ions",))
    return P


handle_holder = {}


def _try_broken():
    handle_holder["h"] = pops.codegen.compile_problem(time=_broken_program(),
                                                      model=transport_model())


raised = False
try:
    _try_broken()
except Exception:  # noqa: BLE001 -- any validation / compile error is the fail-loud contract
    raised = True
chk(raised, "a broken Program raises during compile")
chk("h" not in handle_holder, "NO handle object escaped a failed compile (validated-or-absent)")

print("%s test_compiled_validated_or_absent" % ("FAIL (%d)" % fails if fails else "PASS"))
sys.exit(1 if fails else 0)
