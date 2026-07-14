"""ADC-600: the host/prototype routes are quarantined off the compile/bind target surface.

Runtime proofs (need ``pops`` importable; skip otherwise, never fake):

* the target compile route refuses a non-production backend EARLY (the production native route is
  required; a prototype/host route is never a fallback);
* the typed backend descriptors expose an honest ``tier`` (production | prototype | internal);
* ``pops.experimental`` is importable (internal prototyping) but marked ``__experimental__`` and its
  ``PythonFlux`` is NOT reachable as ``pops.PythonFlux``;
* a compiled operator-first time Program carries no Python callback token in its hot step.
"""
import sys

try:
    import pops
    from pops.codegen._backends import JIT, AOT, Production
    from pops.codegen._compile_drivers import compile_problem
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_prototype_quarantine (pops unavailable: %s)" % exc)
    sys.exit(0)


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
    """Build a tiny final-API Program and return its generated C++ source."""
    from pops.physics._facade import Model
    from pops.problem import Case
    import pops.lib.time as libtime

    model = Model("ep")
    (u,) = model.conservative_vars("u")
    model.source_term("relax", [-u])
    model.linear_source("implicit", [[-1.0]])
    model.rate_operator("explicit_rhs", flux=False, sources=["relax"])

    state_space = next(iter(model.module.state_spaces().values()))
    state = model.module.state_handle(state_space)
    block = Case(name="pc-case").block("plasma", model.module)
    program = libtime.IMEX(
        block[state],
        explicit_operator=model.module.operator_handle("explicit_rhs"),
        implicit_operator=model.module.operator_handle("implicit"),
    )
    return program.emit_cpp_program(model=model)


def test_program_step_has_no_python_callback_tokens():
    """The generated step body has no Python/host-callback token."""
    source = _program_source()
    assert "pops_install_program" in source, "the source must define the program install entry"
    body = source.split("pops_install_program", 1)[1]
    for token in ("PyObject", "py::", "std::function"):
        assert token not in body, (
            "the compiled Program step body must not reference %r (no Python callback / host "
            "type-erasure in the hot step, ADC-600)" % token)


def main():
    test_target_compile_refuses_non_production_backend()
    test_backend_descriptor_tiers()
    test_experimental_is_off_the_root()
    test_program_step_has_no_python_callback_tokens()
    print("OK  test_prototype_quarantine")


if __name__ == "__main__":
    main()
