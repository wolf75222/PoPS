"""Spec 2 (S2-7): Module.module_hash covers the ModuleSpec (spaces + typed operators).

module_hash folds the spaces, parameters, aux, authenticated operator aliases and -- per operator
-- name, kind, signature, capabilities, requirements and a body identity. It is deterministic for
an identical module and invalidated by an operator body / signature / capability / alias / space
change, so a compiled artifact keyed on it is rebuilt when the operator spec changes. The dsl
codegen sensitivity to a formula change stays with the existing Model._model_hash; module_hash adds
the operator-spec layer.
Pure Python; skips if pops is not importable.
"""
import sys

import pytest

try:
    from pops import model
    from pops._ir.expr import Const, Var
    from pops.math import sqrt
    from pops.physics._facade import Model
    from pops.params import RuntimeParam
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_module_hash (pops unavailable: %s)" % exc)
    sys.exit(0)


def test_deterministic():
    def build():
        mod = model.Module("m")
        u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
        f = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
        mod.parameters(RuntimeParam("alpha", default=1.0))
        mod.aux_fields(B_z="cell_scalar")
        mod.operator(name="fields_from_state", signature=(u,) >> f,
                     kind="field_operator", expr="POISSON")
        return mod

    assert build().module_hash() == build().module_hash()
    print("OK  module_hash is deterministic for an identical module")


def test_signature_change_invalidates():
    m1 = model.Module("m")
    u1 = m1.state_space("U", ("rho", "mx"))
    m1.field_space("fields", ("phi",))
    m1.operator(name="op", signature=(u1,) >> model.Rate(u1), kind="local_rate", expr="E")
    m2 = model.Module("m")
    u2 = m2.state_space("U", ("rho", "mx"))
    f2 = m2.field_space("fields", ("phi",))
    m2.operator(name="op", signature=(u2, f2) >> model.Rate(u2), kind="local_rate", expr="E")
    assert m1.module_hash() != m2.module_hash()
    print("OK  a signature change invalidates module_hash")


def test_public_operator_alias_change_invalidates():
    def build(alias=None):
        module = model.Module("aliased")
        state = module.state_space("U", ("rho",))
        module.operator(
            name="flux_default", signature=(state,) >> model.Rate(state),
            kind="grid_operator", expr="F",
        )
        if alias is not None:
            module.operator_registry().register_alias(alias, "flux_default")
        return module

    assert build("transport").module_hash() != build().module_hash()
    assert build("transport").module_hash() != build("advection").module_hash()
    print("OK  an authenticated public operator alias invalidates module_hash")


def test_expr_body_change_invalidates():
    def build(expr):
        mod = model.Module("m")
        u = mod.state_space("U", ("rho",))
        f = mod.field_space("fields", ("phi",))
        mod.operator(name="op", signature=(u, f) >> model.Rate(u),
                     kind="local_rate", expr=expr)
        return mod

    assert build("BODY_A").module_hash() != build("BODY_B").module_hash()
    print("OK  an operator body change invalidates module_hash")


def test_callable_body_change_invalidates():
    def mod_a():
        mod = model.Module("m")
        u = mod.state_space("U", ("rho",))
        f = mod.field_space("fields", ("phi",))

        @mod.operator(name="op", signature=(u, f) >> model.Rate(u), kind="local_rate")
        def op(state, fields):
            return "alpha"

        return mod

    def mod_b():
        mod = model.Module("m")
        u = mod.state_space("U", ("rho",))
        f = mod.field_space("fields", ("phi",))

        @mod.operator(name="op", signature=(u, f) >> model.Rate(u), kind="local_rate")
        def op(state, fields):
            return "beta"

        return mod

    assert mod_a().module_hash() != mod_b().module_hash()
    print("OK  a decorated-body source change invalidates module_hash")


def test_callable_instances_hash_by_code_and_strict_state_not_address_repr():
    class Scale:
        def __init__(self, factor):
            self.factor = factor

        def __call__(self, state):
            return self.factor, state

    def build(body):
        mod = model.Module("callable-instance")
        state = mod.state_space("U", ("rho",))
        mod.operator(
            "source", signature=(state,) >> model.Rate(state),
            kind="local_source", expr=body)
        return mod.module_hash()

    assert build(Scale(2)) == build(Scale(2))
    assert build(Scale(2)) != build(Scale(3))


def test_callable_private_slots_and_referenced_globals_invalidate_hash():
    class SlottedScale:
        __slots__ = ("__factor",)

        def __init__(self, factor):
            self.__factor = factor

        def __call__(self, state):
            return self.__factor, state

    def build(body):
        mod = model.Module("callable-dependencies")
        state = mod.state_space("U", ("rho",))
        mod.operator(
            "source", signature=(state,) >> model.Rate(state),
            kind="local_source", expr=body)
        return mod.module_hash()

    assert build(SlottedScale(2)) != build(SlottedScale(3))

    namespace = {"__name__": __name__, "FACTOR": 2}
    exec("def source(state):\n    return FACTOR, state\n", namespace)
    source = namespace["source"]
    first = build(source)
    namespace["FACTOR"] = 3
    assert first != build(source)


def test_opaque_body_and_opaque_hash_metadata_fail_loud_without_repr_fallback():
    state = model.StateSpace("U", ("rho",))

    opaque_body = model.Module("opaque-body")
    opaque_body.state_space("U", ("rho",))
    opaque_body.operator(
        "source", signature=(state,) >> model.Rate(state),
        kind="local_source", expr=object())
    with pytest.raises(TypeError, match="opaque.*to_data"):
        opaque_body.module_hash()

    opaque_metadata = model.Module("opaque-metadata")
    opaque_metadata.state_space("U", ("rho",))
    opaque_metadata.operator(
        "source", signature=(state,) >> model.Rate(state),
        kind="local_source", capabilities={"opaque": object()}, expr="source")
    with pytest.raises(TypeError, match="opaque.*to_data"):
        opaque_metadata.module_hash()


def test_capability_and_space_change_invalidate():
    u = model.StateSpace("U", ("rho", "mx"))
    base = model.Module("m")
    base.state_space("U", ("rho", "mx"))
    base.field_space("fields", ("phi",))
    base.operator(name="op", signature=(u,) >> model.Rate(u), kind="local_rate",
                  capabilities={"produces_rate": True}, expr="E")
    other_caps = model.Module("m")
    other_caps.state_space("U", ("rho", "mx"))
    other_caps.field_space("fields", ("phi",))
    other_caps.operator(name="op", signature=(u,) >> model.Rate(u), kind="local_rate",
                        capabilities={"produces_rate": False}, expr="E")
    assert base.module_hash() != other_caps.module_hash()

    other_space = model.Module("m")
    other_space.state_space("U", ("rho", "mx", "my"))  # one more component
    other_space.field_space("fields", ("phi",))
    other_space.operator(name="op", signature=(u,) >> model.Rate(u), kind="local_rate",
                         capabilities={"produces_rate": True}, expr="E")
    assert base.module_hash() != other_space.module_hash()
    print("OK  a capability or a state-space change invalidates module_hash")


def test_layout_storage_roles_and_typed_signature_change_invalidate():
    def build(*, layout="cell", storage="multifab", roles=None, operator_components=("rho",)):
        mod = model.Module("m")
        mod.state_space(
            "U", ("rho",), roles=roles or {"rho": "Density"},
            layout=layout, storage=storage,
        )
        op_space = model.StateSpace(
            "U", operator_components, roles=roles or {"rho": "Density"},
            layout=layout, storage=storage,
        )
        mod.operator(
            name="L", signature=() >> model.LocalLinearOperator(op_space, op_space),
            kind="local_linear_operator", expr="L",
        )
        return mod.module_hash()

    baseline = build()
    assert baseline != build(layout="face")
    assert baseline != build(storage="array")
    assert baseline != build(roles={"rho": "Mass"})
    assert baseline != build(operator_components=("energy",))
    print("OK  layout/storage/roles and full operator spaces invalidate module_hash")


def test_requirements_change_invalidates():
    u = model.StateSpace("U", ("rho",))
    f = model.FieldSpace("fields", ("phi",))

    def build(reqs):
        mod = model.Module("m")
        mod.state_space("U", ("rho",))
        mod.field_space("fields", ("phi",))
        mod.operator(name="op", signature=(f,) >> model.LocalLinearOperator(u, u),
                     kind="local_linear_operator", requirements=reqs, expr="E")
        return mod

    assert build({"aux": ["B_z"]}).module_hash() != build({"aux": ["E_x"]}).module_hash()
    print("OK  a requirements change invalidates module_hash")


def test_eigenvalues_change_invalidates():
    def build(speed):
        mod = model.Module("m")
        mod.state_space("U", ("rho", "mx"))
        mod.field_space("fields", ("phi",))
        rho, mx = Var("rho", "cons"), Var("mx", "cons")
        mod.eigenvalues(x=[mx / rho - speed, mx / rho + speed],
                        y=[mx / rho - speed, mx / rho + speed])
        return mod

    assert build(sqrt(0.5)).module_hash() != build(sqrt(0.7)).module_hash()
    print("OK  an eigenvalue (wave-speed) change invalidates module_hash")


def test_dsl_backed_module_hashes():
    m = Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.flux(x=[mx, mx, mx], y=[my, my, my])
    m.source_term("electric", [Const(0.0), rho, rho])
    h = m.module.module_hash()
    assert isinstance(h, str) and len(h) == 64
    print("OK  a dsl-backed Module produces a module_hash")


def main():
    test_deterministic()
    test_signature_change_invalidates()
    test_public_operator_alias_change_invalidates()
    test_expr_body_change_invalidates()
    test_callable_body_change_invalidates()
    test_callable_instances_hash_by_code_and_strict_state_not_address_repr()
    test_callable_private_slots_and_referenced_globals_invalidate_hash()
    test_opaque_body_and_opaque_hash_metadata_fail_loud_without_repr_fallback()
    test_capability_and_space_change_invalidate()
    test_layout_storage_roles_and_typed_signature_change_invalidate()
    test_requirements_change_invalidates()
    test_eigenvalues_change_invalidates()
    test_dsl_backed_module_hashes()
    print("OK  test_module_hash")


if __name__ == "__main__":
    main()
