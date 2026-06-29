"""Spec 2 (S2-4): operator-first standard macros.

pops.lib.time.predictor_corrector_local_linear / explicit_rk / imex_local_linear take typed
operator handles (not physical terms or string selectors) and compose them against the Module
bound to the Program. The macros are model-free (their source mentions no physics) and reusable
across any Module with matching signatures. Pure Python (emit only); skips if pops is not importable.
"""
import inspect
import sys

try:
    from pops.ir.expr import Const
    from pops import model
    from pops.ir.expr import Var
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_macros (pops unavailable: %s)" % exc)
    sys.exit(0)

_PHYSICS_TOKENS = ("electric", "lorentz", "poisson", "rho", "grad_x", "grad_y", "B_z")


def _model(name, gain=1.0):
    mod = model.Module(name)
    u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y", "B_z"))
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy, bz = Var("grad_x", "aux"), Var("grad_y", "aux"), Var("B_z", "aux")
    fields_op = mod.operator(
        name="fields_from_state", signature=(u,) >> fields, kind="field_operator",
        capabilities={"default": True}, expr=rho - 1.0)
    mod.operator(
        name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
        expr={"x": [mx, mx * mx / rho, mx * my / rho],
              "y": [my, mx * my / rho, my * my / rho]})
    electric = mod.operator(
        name="electric", signature=(u, fields) >> model.Rate(u), kind="local_source",
        expr=[Const(0.0), rho * (-gx) * gain, rho * (-gy) * gain])
    lorentz = mod.operator(
        name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
        kind="local_linear_operator",
        expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    explicit_rhs = mod.rate_operator("explicit_rhs", flux=True, sources=[electric])
    return mod, {"fields": fields_op, "explicit_rhs": explicit_rhs, "lorentz": lorentz}


def test_macros_are_model_free():
    for macro in (libtime.predictor_corrector_local_linear,
                  libtime.explicit_rk,
                  libtime.imex_local_linear):
        src = inspect.getsource(macro)
        for tok in _PHYSICS_TOKENS:
            assert tok not in src, "%s must not mention %r" % (macro.__name__, tok)
    print("OK  the operator-first macros mention no physics term")


def test_predictor_corrector_macro():
    m, h = _model("ep")
    P = adctime.Program("pc").bind_operators(m)
    libtime.predictor_corrector_local_linear(
        P, "plasma", fields_operator=h["fields"],
        explicit_rate_operator=h["explicit_rhs"], implicit_operator=h["lorentz"])
    P.validate()
    src = P.emit_cpp_program(model=m)
    assert "pops_problem_install" in src
    assert "GeneratedModule::Operators::" in src
    print("OK  predictor_corrector_local_linear composes 3 typed operators -> .so source")


def test_explicit_rk_macro():
    m, h = _model("rk")
    P = adctime.Program("rk").bind_operators(m)
    libtime.explicit_rk(P, "plasma", rhs_operator=h["explicit_rhs"],
                            fields_operator=h["fields"],
                            tableau=libtime.SSPRK2_TABLEAU)
    P.validate()
    src = P.emit_cpp_program(model=m)
    assert "pops_problem_install" in src
    assert "GeneratedModule::Operators::" in src
    print("OK  explicit_rk over a typed rate operator (SSPRK2 tableau)")


def test_ready_explicit_macros_are_operator_first():
    m, h = _model("ready_explicit")
    for macro in (libtime.forward_euler, libtime.ssprk2, libtime.ssprk3, libtime.rk4):
        P = adctime.Program(macro.__name__).bind_operators(m)
        macro(P, "plasma", rhs_operator=h["explicit_rhs"], fields_operator=h["fields"])
        P.validate()
        src = P.emit_cpp_program(model=m)
        assert "GeneratedModule::Operators::" in src
        assert all(v.op != "rhs" for v in P._values)
        assert all(v.op != "source" for v in P._values)
        assert all(v.op != "linear_source" for v in P._values)
    P = adctime.Program("rk_generic").bind_operators(m)
    libtime.rk(P, "plasma", libtime.SSPRK2_TABLEAU,
               rhs_operator=h["explicit_rhs"], fields_operator=h["fields"])
    P.validate()
    assert "GeneratedModule::Operators::" in P.emit_cpp_program(model=m)

    Pab = adctime.Program("ab2").bind_operators(m)
    libtime.adams_bashforth(Pab, "plasma", 2,
                            rhs_operator=h["explicit_rhs"], fields_operator=h["fields"])
    Pab.validate()
    assert any(v.op == "history" for v in Pab._values)
    print("OK  ready explicit macros compose typed operator handles")


def test_imex_local_linear_macro():
    m, h = _model("imex")
    P = adctime.Program("imex").bind_operators(m)
    libtime.imex_local_linear(P, "plasma", explicit_operator=h["explicit_rhs"],
                                  implicit_operator=h["lorentz"],
                                  fields_operator=h["fields"], theta=1.0)
    P.validate()
    src = P.emit_cpp_program(model=m)
    assert "pops_problem_install" in src
    assert "GeneratedModule::Operators::" in src
    print("OK  imex_local_linear (theta-implicit local linear solve)")


def test_public_macros_reject_string_operator_selectors():
    m, h = _model("macro_strings")
    P = adctime.Program("reject_strings").bind_operators(m)
    try:
        libtime.explicit_rk(P, "plasma", rhs_operator="explicit_rhs",
                            fields_operator=h["fields"],
                            tableau=libtime.SSPRK2_TABLEAU)
    except TypeError as exc:
        assert "typed operator handles" in str(exc)
    else:
        raise AssertionError("pops.lib.time accepted a string operator selector")
    try:
        libtime.forward_euler(P, "plasma", rhs_operator="explicit_rhs",
                              fields_operator=h["fields"])
    except TypeError as exc:
        assert "typed operator handles" in str(exc)
    else:
        raise AssertionError("forward_euler accepted a string rate selector")
    print("OK  pops.lib.time macros reject string operator selectors")


def test_macro_reused_across_modules():
    def build(m):
        m, h = m
        P = adctime.Program("pc").bind_operators(m)
        libtime.predictor_corrector_local_linear(
            P, "plasma", fields_operator=h["fields"],
            explicit_rate_operator=h["explicit_rhs"], implicit_operator=h["lorentz"])
        return P.emit_cpp_program(model=m)

    src_a = build(_model("A", 1.0))
    src_b = build(_model("B", 2.0))
    assert "pops_problem_install" in src_a and src_a != src_b
    assert "GeneratedModule::Operators::" in src_a
    print("OK  the same predictor-corrector macro is reused across two modules")


def main():
    test_macros_are_model_free()
    test_predictor_corrector_macro()
    test_explicit_rk_macro()
    test_ready_explicit_macros_are_operator_first()
    test_imex_local_linear_macro()
    test_public_macros_reject_string_operator_selectors()
    test_macro_reused_across_modules()
    print("OK  test_operator_macros")


if __name__ == "__main__":
    main()
