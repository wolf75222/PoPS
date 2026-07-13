"""Spec 2 (S2-4): operator-first standard macros.

pops.lib.time.predictor_corrector_local_linear / explicit_rk / imex_local_linear take typed
operator HANDLES (pops.model.OperatorHandle, not physical terms or name strings) and compose them
with P.call against the Module bound to the Program (ADC-532). The macros are model-free (their
source mentions no physics) and reusable across any Module with matching signatures. Pure Python
(emit only); skips if pops is not importable.
"""
import inspect
import sys

try:
    from pops.ir.expr import Const
    from pops.model import OperatorHandle
    from pops.physics._facade import Model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
    from typed_program_support import state_refs, typed_state
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_macros (pops unavailable: %s)" % exc)
    sys.exit(0)

_PHYSICS_TOKENS = ("electric", "lorentz", "poisson", "rho", "grad_x", "grad_y", "B_z")


def _model(name, gain=1.0):
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), rho * (-gx) * gain, rho * (-gy) * gain])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0],
                                [0.0, 0.0, bz],
                                [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def _handle(m, name):
    """A typed OperatorHandle for a registered operator (the selector the de-stringed macros take)."""
    op = m.operator_registry().get(name)
    return OperatorHandle(
        op.name, kind=op.kind, owner=m.operator_registry().owner_path,
        signature=op.signature)


def test_macros_are_model_free():
    for macro in (libtime.predictor_corrector_local_linear,
                  libtime.explicit_rk,
                  libtime.imex_local_linear):
        src = inspect.getsource(macro)
        for tok in _PHYSICS_TOKENS:
            assert tok not in src, "%s must not mention %r" % (macro.__name__, tok)
    print("OK  the operator-first macros mention no physics term")


def test_predictor_corrector_macro():
    m = _model("ep")
    P = adctime.Program("pc")._bind_operators(m)
    block, state = state_refs(P, "plasma", model=m)
    libtime.predictor_corrector_local_linear(
        P, block, state, fields_operator=_handle(m, "fields_from_state"),
        explicit_rate_operator=_handle(m, "explicit_rhs"),
        implicit_operator=_handle(m, "lorentz"))
    P.validate()
    src = P.emit_cpp_program(model=m)
    assert "pops_install_program" in src
    print("OK  predictor_corrector_local_linear composes 3 typed handles -> .so source")


def test_explicit_rk_macro():
    m = _model("rk")
    P = adctime.Program("rk")._bind_operators(m)
    block, state = state_refs(P, "plasma", model=m)
    libtime.explicit_rk(P, block, state, rhs_operator=_handle(m, "explicit_rhs"),
                            fields_operator=_handle(m, "fields_from_state"),
                            tableau=libtime.SSPRK2_TABLEAU)
    P.validate()
    assert "pops_install_program" in P.emit_cpp_program(model=m)
    print("OK  explicit_rk over a typed rate handle (SSPRK2 tableau)")


def test_imex_local_linear_macro():
    m = _model("imex")
    P = adctime.Program("imex")._bind_operators(m)
    block, state = state_refs(P, "plasma", model=m)
    libtime.imex_local_linear(P, block, state,
                             explicit_operator=_handle(m, "explicit_rhs"),
                                  implicit_operator=_handle(m, "lorentz"),
                                  fields_operator=_handle(m, "fields_from_state"), theta=1.0)
    P.validate()
    assert "pops_install_program" in P.emit_cpp_program(model=m)
    print("OK  imex_local_linear (theta-implicit local linear solve)")


def test_macro_rejects_string_operator():
    # ADC-532: a stale string operator selector is refused with a clear TypeError, not silently taken.
    m = _model("reject")
    P = adctime.Program("rej")._bind_operators(m)
    block, state = state_refs(P, "plasma", model=m)
    try:
        libtime.imex_local_linear(P, block, state, explicit_operator="explicit_rhs",
                                  implicit_operator=_handle(m, "lorentz"),
                                  fields_operator=_handle(m, "fields_from_state"))
        raise AssertionError("a string operator selector must be refused")
    except TypeError as exc:
        assert "OperatorHandle" in str(exc) and "explicit_operator" in str(exc), str(exc)
    print("OK  a string operator selector is refused pointing at the handle form")


def test_handle_macro_ir_parity_with_name_call():
    # The handle path emits the same executable body as a manual Program built with the same
    # primitive _call selectors (the allowed internal seam). Authoring IR deliberately
    # authenticates block handles, so independently-authored Programs have distinct artifact hashes.
    m = _model("parity")
    P_handles = adctime.Program("imex")._bind_operators(m)
    block, state = state_refs(P_handles, "plasma", model=m)
    libtime.imex_local_linear(P_handles, block, state,
                              explicit_operator=_handle(m, "explicit_rhs"),
                              implicit_operator=_handle(m, "lorentz"),
                              fields_operator=_handle(m, "fields_from_state"), theta=1.0)
    # Manual equivalent using the internal name selectors (the pre-ADC-532 lowering).
    P_manual = adctime.Program("imex")._bind_operators(m)
    u = typed_state(P_manual, "plasma", model=m)
    fields = P_manual._call("fields_from_state", u, name="fields")
    r = P_manual._call("explicit_rhs", u, fields, name="R")
    lin = P_manual._call("lorentz", fields, name="L")
    endpoint = typed_state(P_manual, "plasma", state_name="U", model=m).next
    q = P_manual.value("imex_rhs", u + P_manual.dt * r, at=endpoint.point)
    u1 = P_manual.solve_local_linear("imex_step", operator=P_manual.I - 1.0 * P_manual.dt * lin,
                                     rhs=q, fields=fields)
    P_manual.commit(endpoint, u1)
    def executable_body(source):
        return "\n".join(
            line for line in source.splitlines()
            if "pops_program_hash()" not in line)

    handles_cpp = executable_body(P_handles.emit_cpp_program(model=m))
    manual_cpp = executable_body(P_manual.emit_cpp_program(model=m))
    assert handles_cpp == manual_cpp, "handle-built macro body must equal the manual lowering"
    assert P_handles._ir_hash() != P_manual._ir_hash(), \
        "distinct authenticated block handles must keep distinct artifact identities"
    print("OK  handle-built imex_local_linear body == the manual _call lowering")


def test_macro_reused_across_modules():
    def build(m):
        P = adctime.Program("pc")._bind_operators(m)
        block, state = state_refs(P, "plasma", model=m)
        libtime.predictor_corrector_local_linear(
            P, block, state, fields_operator=_handle(m, "fields_from_state"),
            explicit_rate_operator=_handle(m, "explicit_rhs"),
            implicit_operator=_handle(m, "lorentz"))
        return P.emit_cpp_program(model=m)

    src_a = build(_model("A", 1.0))
    src_b = build(_model("B", 2.0))
    assert "pops_install_program" in src_a and src_a != src_b
    print("OK  the same predictor-corrector macro is reused across two modules")


def main():
    test_macros_are_model_free()
    test_predictor_corrector_macro()
    test_explicit_rk_macro()
    test_imex_local_linear_macro()
    test_macro_rejects_string_operator()
    test_handle_macro_ir_parity_with_name_call()
    test_macro_reused_across_modules()
    print("OK  test_operator_macros")


if __name__ == "__main__":
    main()
