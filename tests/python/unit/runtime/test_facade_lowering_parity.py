#!/usr/bin/env python3
"""ADC-529: the physics facade lowers into the ONE operator-first core, not a parallel surface.

A model authored through the PDE facade (``pops.physics._facade.Model``: flux / source_term /
linear_source / rate_operator) exposes the SAME typed operator-first ``pops.model.Module`` a
hand-built Module would -- same state / field spaces, same typed operators (name, kind, signature
inputs / output). This is the ADC-529 acceptance criterion: there is no facade-specific registry;
the facade only POPULATES the shared operator-first registry.

The test compares the STRUCTURAL manifest rows (name / kind / inputs / output) of the facade's
lowered module against a hand-built ``pops.model.Module`` that declares the equivalent operators
directly. The full module_hash also folds each operator's IR BODY (a facade flux carries symbolic
IR a bare declarator does not), so the hash is not expected equal; the structural typed contract is.

Pure Python (no numerics beyond IR construction); skips if pops is not importable.
"""
import sys

try:
    from pops import model
    from pops._ir.expr import Const
    from pops.physics._facade import Model
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_facade_lowering_parity (pops unavailable: %s)" % exc)
    sys.exit(0)


def _facade_model():
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
    return m


def _op_rows(module):
    """The structural typed rows of a Module's operators: {name: (kind, tuple(inputs), output)}."""
    rows = {}
    for row in module.manifest().operators:
        rows[row.name] = (row.kind, tuple(row.inputs), row.output)
    return rows


def _hand_built_module():
    """A pops.model.Module built operator-first with the same typed operators the facade lowers to."""
    mod = model.Module("hand")
    u = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields")
    # explicit_rhs: a composite rate (U, fields) -> Rate(U). The facade's rate_operator carries the
    # field dependency in its signature, so we declare the same (U, fields) -> Rate(U) contract here.
    mod.operator(name="explicit_rhs", kind="local_rate",
                 signature=(u, fields) >> model.Rate(u),
                 expr="<explicit-rate-ir>")
    # electric: a named local_source (U, fields) -> Rate(U).
    mod.operator(name="electric", kind="local_source",
                 signature=(u, fields) >> model.Rate(u),
                 expr="<electric-source-ir>")
    # lorentz: a local_linear_operator fields -> LocalLinearOperator(U, U).
    mod.operator(name="lorentz", kind="local_linear_operator",
                 signature=(fields,) >> model.LocalLinearOperator(u, u),
                 expr="<lorentz-matrix-ir>")
    # fields_from_state: the default field operator U -> fields.
    mod.operator(name="fields_from_state", kind="field_operator",
                 signature=(u,) >> fields,
                 expr="<poisson-ir>")
    return mod


def test_facade_lowers_to_same_typed_operators():
    facade_rows = _op_rows(_facade_model().module)
    hand_rows = _op_rows(_hand_built_module())
    # Every operator the hand-built operator-first Module declares appears in the facade lowering
    # with the identical typed contract (kind, inputs, output). The facade may add its default flux
    # operator (flux_default); we assert the shared operators match structurally.
    for name, contract in hand_rows.items():
        assert name in facade_rows, "facade lowering is missing operator %r" % name
        assert facade_rows[name] == contract, (
            "operator %r typed contract differs: facade=%r hand=%r"
            % (name, facade_rows[name], contract))
    print("OK  every hand-built operator-first operator matches the facade lowering (kind/inputs/output)")


def test_facade_spaces_match_operator_first():
    facade = _facade_model().module
    assert facade.list_state_spaces() == ["U"], facade.list_state_spaces()
    assert facade.list_field_spaces() == ["fields"], facade.list_field_spaces()
    print("OK  the facade lowers to the operator-first StateSpace 'U' and FieldSpace 'fields'")


def test_facade_lowering_is_deterministic():
    m = _facade_model()
    # Lowering the SAME facade twice yields byte-identical operator-first modules (one registry,
    # no hidden state), so the compiled-artifact cache key is stable.
    assert m.module.module_hash() == m.module.module_hash()
    print("OK  the facade lowering is deterministic (stable module_hash)")


def test_no_parallel_facade_registry():
    # ADC-529 forbidden: the facade must not carry a second operator registry. Its typed view is the
    # shared pops.model.OperatorRegistry; assert the facade's registry IS an OperatorRegistry.
    m = _facade_model()
    assert isinstance(m.operator_registry(), model.OperatorRegistry)
    assert isinstance(m.module.operator_registry(), model.OperatorRegistry)
    print("OK  the facade exposes the single operator-first OperatorRegistry, no parallel surface")


def main():
    test_facade_lowers_to_same_typed_operators()
    test_facade_spaces_match_operator_first()
    test_facade_lowering_is_deterministic()
    test_no_parallel_facade_registry()
    print("OK  test_facade_lowering_parity")


if __name__ == "__main__":
    main()
