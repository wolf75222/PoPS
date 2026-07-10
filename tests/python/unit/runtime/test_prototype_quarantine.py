"""ADC-600: the host/prototype routes are quarantined off the compile/bind target surface.

Runtime proofs (need ``pops`` importable; skip otherwise, never fake):

* the target compile route refuses a non-production backend EARLY (the production native route is
  required; a prototype/host route is never a fallback);
* the typed backend descriptors expose an honest ``tier`` (production | prototype | internal);
* ``pops.experimental`` is importable (internal prototyping) but marked ``__experimental__`` and its
  ``PythonFlux`` is NOT reachable as ``pops.PythonFlux``;
* a compiled time-Program step body carries no Python-callback token (no ``PyObject`` / ``py::`` /
  ``std::function`` in the hot step), complementing test_module_codegen's ``pops_module_*`` check.
"""
import sys

try:
    import pops
    from pops.codegen.backends import JIT, AOT, Production
    from pops.codegen.compile_drivers import compile_problem
    from pops.model import OperatorHandle
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_prototype_quarantine (pops unavailable: %s)" % exc)
    sys.exit(0)


def _op(m, name):
    """A typed OperatorHandle for a registered operator (the de-stringed macro selector, ADC-532)."""
    op = m.operator_registry().get(name)
    return OperatorHandle(
        op.name, kind=op.kind, owner=m.operator_registry().owner_path,
        signature=op.signature)


PRODUCTION_ONLY = "require backend='production'"


def test_target_compile_refuses_non_production_backend():
    """compile_problem with a prototype backend refuses BEFORE any model use (no host fallback)."""
    for backend in ("prototype", "aot", JIT(), AOT()):
        try:
            compile_problem(backend=backend, model=None, time=None)
        except ValueError as exc:
            assert PRODUCTION_ONLY in str(exc), (
                "backend=%r must refuse with the production-only message, got: %s" % (backend, exc))
        else:
            raise AssertionError(
                "compile_problem(backend=%r) must REFUSE (no prototype/host fallback)" % (backend,))
    print("OK  compile target refuses prototype/aot backends with the production-only message")


def test_backend_descriptor_tiers():
    """The typed descriptors name their route class honestly (ADC-600)."""
    assert JIT().tier == "prototype", "JIT (add_dynamic_block host virtual dispatch) is a prototype"
    assert AOT().tier == "internal", "AOT (host-marshalled production algorithm) is internal"
    assert Production().tier == "production", "Production (native loader) is the production route"
    # The tier is also on the capabilities() dict (single source: _BACKEND_CAPS).
    assert Production().capabilities().to_dict().get("tier") == "production"
    print("OK  JIT=prototype, AOT=internal, Production=production")


def test_experimental_is_off_the_root():
    """pops.experimental stays an internal prototyping package, not part of the public surface."""
    import importlib

    experimental = importlib.import_module("pops.experimental")
    assert getattr(experimental, "__experimental__", False) is True, (
        "pops.experimental must be marked __experimental__ = True (non-production)")
    assert hasattr(experimental, "PythonFlux"), "PythonFlux lives under pops.experimental"
    assert not hasattr(pops, "PythonFlux"), (
        "PythonFlux (numpy host backend) must NOT be reachable as pops.PythonFlux (ADC-600)")
    print("OK  pops.experimental is __experimental__ and PythonFlux is off the pops root")


def _program_source():
    """A tiny compiled time-Program C++ source, built the way test_module_codegen does."""
    from pops.ir.expr import Const
    from pops.physics.facade import Model
    import pops.lib.time as libtime

    m = Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    P = adctime.Program("pc").bind_operators(m)
    libtime.predictor_corrector_local_linear(
        P, "plasma", fields_operator=_op(m, "fields_from_state"),
        explicit_rate_operator=_op(m, "explicit_rhs"), implicit_operator=_op(m, "lorentz"))
    return P.emit_cpp_program(model=m)


def test_program_step_has_no_python_callback_tokens():
    """The generated step body has no Python/host-callback token (complements module_codegen)."""
    src = _program_source()
    assert "pops_install_program" in src, "the source must define the program install entry"
    body = src.split("pops_install_program", 1)[1]
    for token in ("PyObject", "py::", "std::function"):
        assert token not in body, (
            "the compiled Program step body must not reference %r (no Python callback / host "
            "type-erasure in the hot step, ADC-600)" % token)
    print("OK  compiled Program step body carries no Python-callback token")


def main():
    test_target_compile_refuses_non_production_backend()
    test_backend_descriptor_tiers()
    test_experimental_is_off_the_root()
    test_program_step_has_no_python_callback_tokens()
    print("OK  test_prototype_quarantine")


if __name__ == "__main__":
    main()
