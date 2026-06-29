"""Spec 2 (S2-5): operator introspection on a Module, a Model and a CompiledProblem.

list_operators / operator_signature / operator_requirements / operator_capabilities /
list_state_spaces / list_field_spaces return the typed registry metadata. The CompiledProblem
methods read the carried model's metadata -- no need to load or run the .so -- so they are
exercised here on a CompiledProblem built directly (not via the Kokkos-only compile). Pure Python.
"""
import sys

try:
    from pops import model
    from pops.codegen.loader import CompiledProblem
    from pops.ir.expr import Const, Var
    from pops.model import OperatorHandle
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_introspection (pops unavailable: %s)" % exc)
    sys.exit(0)


def _model():
    mod = model.Module("ep")
    u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y", "B_z"))
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    gx, gy, bz = Var("grad_x", "aux"), Var("grad_y", "aux"), Var("B_z", "aux")
    mod.operator(
        name="fields_from_state", signature=(u,) >> fields, kind="field_operator",
        capabilities={"default": True}, expr=rho - 1.0)
    mod.operator(
        name="flux", signature=(u,) >> model.Rate(u), kind="grid_operator",
        expr={"x": [mx, mx * mx / rho, mx * my / rho],
              "y": [my, mx * my / rho, my * my / rho]})
    electric = mod.operator(
        name="electric", signature=(u, fields) >> model.Rate(u), kind="local_source",
        expr=[Const(0.0), rho * (-gx), rho * (-gy)])
    mod.operator(
        name="lorentz", signature=(fields,) >> model.LocalLinearOperator(u, u),
        kind="local_linear_operator",
        capabilities={"solve_i_minus_a": True},
        requirements={"aux": ["B_z"]},
        expr=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    mod.rate_operator("explicit_rhs", flux=True, sources=[electric])
    return mod


def _check(obj):
    ops = obj.list_operators()
    assert "explicit_rhs" in ops and "fields_from_state" in ops and "lorentz" in ops
    assert obj.operator_signature("explicit_rhs").output == model.Rate("U")
    assert obj.operator_capabilities("lorentz")["solve_i_minus_a"] is True
    assert obj.operator_requirements("lorentz")["aux"] == ["B_z"]
    assert "U" in obj.list_state_spaces()
    assert "fields" in obj.list_field_spaces()


def test_module_introspection():
    _check(_model().module)
    print("OK  Module introspection")


def test_compiled_problem_introspection():
    m = _model()
    P = adctime.Program("pc").bind_operators(m)
    libtime.predictor_corrector_local_linear(
        P, "plasma", fields_operator=OperatorHandle("fields_from_state", kind="field_operator"),
        explicit_rate_operator=OperatorHandle("explicit_rhs", kind="local_rate"),
        implicit_operator=OperatorHandle("lorentz", kind="local_linear_operator"))
    # A CompiledProblem built directly: introspection reads model metadata, never the .so.
    compiled = CompiledProblem(so_path="<not built>", program=P, model=m,
                                   abi_key="k", cxx="clang", std="c++23")
    _check(compiled)
    # The matching-the-spec assertion.
    assert compiled.operator_signature("explicit_rhs").output == model.Rate("U")
    # A CompiledProblem with no model raises a clear error rather than guessing.
    bare = CompiledProblem(so_path="x", program=P, model=None,
                               abi_key="k", cxx="c", std="s")
    try:
        bare.list_operators()
        raise AssertionError("expected an error introspecting a model-less CompiledProblem")
    except ValueError as exc:
        assert "no model" in str(exc)
    print("OK  CompiledProblem introspection (metadata only, no .so run)")


def main():
    test_module_introspection()
    test_compiled_problem_introspection()
    print("OK  test_operator_introspection")


if __name__ == "__main__":
    main()
