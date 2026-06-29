"""Spec 2 (S2-11 / ADC-447): a pure pops.model.Module compiles directly.

A Module authored directly -- typed spaces + operators with IR (Expr) bodies + eigenvalues --
is a self-contained, compilable model. ``compile_problem(model=module, time=P)`` and
``Program.emit_cpp_program(model=module)`` consume it through the Module-native codegen view, never
through ``Module.to_dsl`` or a rebuilt legacy facade. Pure Python; skips if pops is not importable.
"""
import sys

try:
    from pops import model
    from pops.ir.expr import Const, Expr, Var
    from pops.ir.ops import sqrt
    from pops.codegen.module_view import codegen_model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_module_compile (pops unavailable: %s)" % exc)
    sys.exit(0)


def pure_module():
    mod = model.Module("euler_poisson_lorentz_operator_first")
    u = mod.state_space("U", ("rho", "mx", "my"),
                        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    mod.aux_fields(B_z="cell_scalar")
    # Operator bodies are plain Expr over the state/field names (evaluated at codegen only).
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy = Var("grad_x", "aux"), Var("grad_y", "aux")
    bz = Var("B_z", "aux")
    cs = sqrt(0.5)  # isothermal sound speed (cs2 = 0.5)
    mod.operator(name="fields_from_state", signature=(u,) >> fields,
                 kind="field_operator", expr=rho)
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [mx, mx * mx / rho + 0.5 * rho, mx * my / rho],
                       "y": [my, mx * my / rho, my * my / rho + 0.5 * rho]})
    mod.eigenvalues(x=[mx / rho - cs, mx / rho, mx / rho + cs],
                    y=[my / rho - cs, my / rho, my / rho + cs])
    electric = mod.operator(name="electric", signature=(u, fields) >> model.Rate(u),
                            kind="local_source", expr=[Const(0.0), -rho * gx, -rho * gy])
    mod.operator(name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
                 kind="local_linear_operator",
                 expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    mod.rate_operator("explicit_rhs", flux=True, sources=[electric])
    return mod


def test_module_codegen_view_is_direct():
    mod = pure_module()
    impl = codegen_model(mod)
    assert impl.cons_names == ["rho", "mx", "my"]
    assert "electric" in impl._source_terms
    assert "lorentz" in impl._linear_sources
    assert impl._elliptic is not None
    assert impl._flux and impl._eig
    assert "explicit_rhs" in impl._rate_operators
    assert impl.prim_state == ["rho", "mx", "my"]
    assert not hasattr(mod, "to_dsl")
    print("OK  Module codegen view reads the Module directly")


def test_pure_module_program_emits():
    mod = pure_module()
    P = adctime.Program("pc").bind_operators(mod)
    ops = mod.operator_registry()
    libtime.predictor_corrector_local_linear(
        P, "plasma", fields_operator=ops.get("fields_from_state"),
        explicit_rate_operator=ops.get("explicit_rhs"), implicit_operator=ops.get("lorentz"))
    # compile_problem(model=Module) consumes the Module directly; emit the .so source (no compile).
    src = P.emit_cpp_program(model=mod)
    assert "pops_install_program" in src
    # the GeneratedModule descriptor reflects the pure Module's operators
    assert "pops_module_operator_count() { return" in src
    for op in ("electric", "lorentz", "fields_from_state", "explicit_rhs"):
        assert '"%s"' % op in src, op
    print("OK  a pure operator-first Module + generic macro emits a combined .so source")


def test_module_flux_name_is_default_flux():
    mod = pure_module()
    P = adctime.Program("flux_call").bind_operators(mod)
    U = P.state("plasma")
    F = P._call(mod.operator_registry().get("flux"), U)
    P.commit("plasma", P.linear_combine("u1", U + P.dt * F))
    src = P.emit_cpp_program(model=mod)
    assert "ctx.neg_div_flux_default_into(0," in src
    assert "_emit_flux_kernel" not in src
    print("OK  Module grid operator named 'flux' lowers as the default flux")


def test_module_requires_one_state_space():
    mod = model.Module("two_states")
    mod.state_space("U", ("rho",))
    mod.state_space("V", ("n",))
    try:
        codegen_model(mod)
        raise AssertionError("expected a single-StateSpace requirement error")
    except ValueError as exc:
        assert "exactly one StateSpace" in str(exc)
    print("OK  a Module to compile must declare exactly one StateSpace")


def test_decorator_body_rejected():
    mod = model.Module("deco")
    u = mod.state_space("U", ("rho",))
    fields = mod.field_space("fields", ("phi",))

    @mod.operator(name="electric", signature=(u, fields) >> model.Rate(u), kind="local_source")
    def electric(state, flds):  # a callable body, not an IR expression
        return None

    try:
        codegen_model(mod)
        raise AssertionError("expected a no-IR-body error for a decorator-authored operator")
    except ValueError as exc:
        assert "no IR body" in str(exc)
    print("OK  a decorator/callable operator body is rejected at compile")


def test_multiple_field_operators_are_named_fields():
    mod = model.Module("twofields")
    u = mod.state_space("U", ("rho",))
    f1 = mod.field_space("fields", ("phi",))
    rho = Var("rho", "cons")
    mod.operator(name="fields_from_state", signature=(u,) >> f1, kind="field_operator", expr=rho)
    mod.operator(name="psi", signature=(u,) >> f1, kind="field_operator", expr=rho)
    impl = codegen_model(mod)
    assert impl._elliptic is not None
    assert "psi" in impl._elliptic_fields
    print("OK  multiple Module field_operators lower as default + named fields")


def test_explicit_roles_honored():
    # A non-canonical layout: the StateSpace's explicit roles must reach the dsl model, not be lost.
    mod = model.Module("custom")
    u = mod.state_space("U", ("n", "px", "py"),
                        roles={"n": "density", "px": "momentum_x", "py": "momentum_y"})
    n, px, py = Var("n", "cons"), Var("px", "cons"), Var("py", "cons")
    mod.operator(name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
                 expr={"x": [px, px * px / n, px * py / n], "y": [py, px * py / n, py * py / n]})
    impl = codegen_model(mod)
    assert impl.cons_roles == ["Density", "MomentumX", "MomentumY"], impl.cons_roles
    print("OK  explicit StateSpace roles are mapped through to the dsl model")


def main():
    test_explicit_roles_honored()
    test_module_codegen_view_is_direct()
    test_pure_module_program_emits()
    test_module_flux_name_is_default_flux()
    test_module_requires_one_state_space()
    test_decorator_body_rejected()
    test_multiple_field_operators_are_named_fields()
    print("OK  test_module_compile")


if __name__ == "__main__":
    main()
