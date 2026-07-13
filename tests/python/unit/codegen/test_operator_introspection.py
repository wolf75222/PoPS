"""Spec 2 (S2-5): operator introspection on a Module, a Model and a CompiledProblem.

list_operators / operator_signature / operator_requirements / operator_capabilities /
list_state_spaces / list_field_spaces return the typed registry metadata. The CompiledProblem
methods read the carried model's metadata -- no need to load or run the .so -- so they are
exercised here on a CompiledProblem built directly (not via the Kokkos-only compile). Pure Python.
"""
import gc
import sys
import weakref

try:
    from pops import model
    from pops.codegen.loader import CompiledProblem
    from pops.ir.expr import Const
    from pops.physics.facade import Model
    from pops.problem import Case
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_operator_introspection (pops unavailable: %s)" % exc)
    sys.exit(0)


def _op(m, name):
    """A typed OperatorHandle for a registered operator (the de-stringed macro selector, ADC-532)."""
    op = m.operator_registry().get(name)
    return model.OperatorHandle(
        op.name, kind=op.kind, owner=m.operator_registry().owner_path,
        signature=op.signature)


def _model():
    m = Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), rho * (-gx), rho * (-gy)])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_rhs(rho - 1.0)
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def _time_refs(m):
    module = m.module
    block = Case(name="introspection-case").block("plasma", module)
    state = module.state_handle(module.state_spaces()["U"])
    return block, state


def _check(obj):
    ops = obj.list_operators()
    assert "explicit_rhs" in ops and "fields_from_state" in ops and "lorentz" in ops
    signature = obj.operator_signature("explicit_rhs")
    assert signature.output == model.Rate(signature.inputs[0])
    assert obj.operator_capabilities("lorentz")["solve_i_minus_a"] is True
    assert obj.operator_requirements("lorentz")["aux"] == ["B_z"]
    assert "U" in obj.list_state_spaces()
    assert "fields" in obj.list_field_spaces()


def test_module_introspection():
    _check(_model().module)
    print("OK  Module introspection")


def test_dsl_model_introspection():
    m = _model()
    _check(m)
    assert m.list_state_spaces() == ["U"]
    print("OK  Model introspection")


def test_compiled_problem_introspection():
    m = _model()
    block, state = _time_refs(m)
    P = adctime.Program("pc").bind_operators(m)
    libtime.predictor_corrector_local_linear(
        P, block, state, fields_operator=_op(m, "fields_from_state"),
        explicit_rate_operator=_op(m, "explicit_rhs"), implicit_operator=_op(m, "lorentz"))
    # A CompiledProblem built directly: introspection reads model metadata, never the .so.
    compiled = CompiledProblem(so_path="<not built>", program=P, model=m,
                                   abi_key="k", cxx="clang", std="c++23")
    _check(compiled)
    # Public orchestration discards the authoring model after code emission.  Introspection is
    # served by the detached compiled-module view, not by re-entering that builder.
    compiled.model = object()
    _check(compiled)
    # The matching-the-spec assertion.
    signature = compiled.operator_signature("explicit_rhs")
    assert signature.output == model.Rate(signature.inputs[0])
    # A CompiledProblem with no model raises a clear error rather than guessing.
    bare = CompiledProblem(so_path="x", program=P, model=None,
                               abi_key="k", cxx="c", std="s")
    try:
        bare.list_operators()
        raise AssertionError("expected an error introspecting a model-less CompiledProblem")
    except ValueError as exc:
        assert "no model" in str(exc)
    print("OK  CompiledProblem introspection (metadata only, no .so run)")


def test_compiled_introspection_view_does_not_retain_model_registry():
    def build():
        m = _model()
        block, state = _time_refs(m)
        P = adctime.Program("detached-introspection").bind_operators(m)
        libtime.predictor_corrector_local_linear(
            P, block, state, fields_operator=_op(m, "fields_from_state"),
            explicit_rate_operator=_op(m, "explicit_rhs"),
            implicit_operator=_op(m, "lorentz"))
        registry_ref = weakref.ref(m.operator_registry())
        model_ref = weakref.ref(m)
        compiled = CompiledProblem(
            "<not built>", P, m, "k", "clang", "c++23",
            generated_cpp="// detached\n")
        compiled.model = object()
        return compiled, model_ref, registry_ref

    compiled, model_ref, registry_ref = build()
    gc.collect()

    assert compiled.list_operators()
    assert model_ref() is None
    assert registry_ref() is None


def main():
    test_module_introspection()
    test_dsl_model_introspection()
    test_compiled_problem_introspection()
    test_compiled_introspection_view_does_not_retain_model_registry()
    print("OK  test_operator_introspection")


if __name__ == "__main__":
    main()
