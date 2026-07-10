#!/usr/bin/env python3
"""ADC-530: the Program IR nodes are inspectable, and inspection metadata stays out of the hash.

Every IR node carries a stable identity (id / type / block / inputs / debug name) plus DERIVED,
INSPECTION-ONLY metadata -- a logical shape (component layout from the operator-first space) and an
optional source location (the authoring call site). The inspection metadata is EXCLUDED from
``_serialize`` / ``_ir_hash``, so two Programs differing only in a call-site line (or a value's space
tag) hash identically and every ``.so`` cache key is unchanged -- the bit-identity guarantee.

Also pins the SSA structural checks (missing commit, double commit, distinct field context per
stage) and the Python-collection refusals (``len(value)`` / ``range(value)``) ADC-530 requires.

Pure Python IR construction (no numerics / no _pops); skips if pops is not importable.
"""
import sys

try:
    from pops import time as adctime
    from pops import model
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_program_ir_nodes (pops unavailable: %s)" % exc)
    sys.exit(0)


def _euler(scale=1.0):
    P = adctime.Program("forward_euler")
    U = P.state("plasma")
    R = P._rhs_legacy(state=U, fields=P.solve_fields(U), flux=True, sources=["default"])
    P.commit(P.state("U", block="plasma").next, P.linear_combine("U1", U + (scale * P.dt) * R))
    return P


def test_ir_node_has_identity_and_inspection_fields():
    P = _euler()
    nodes = P.ir_nodes()
    assert nodes, "expected IR nodes"
    for n in nodes:
        for key in ("name", "op", "vtype", "block", "inputs", "logical_shape", "source_location"):
            assert key in n, "node %r is missing key %r" % (n.get("name"), key)
        # logical_shape is a derived dict; source_location is None unless capture is enabled.
        assert isinstance(n["logical_shape"], dict) and "vtype" in n["logical_shape"], n
        assert n["source_location"] is None, "capture is off by default"
    print("OK  each IR node exposes identity + inspection-only logical_shape / source_location")


def test_logical_shape_reflects_the_space_tag():
    P = adctime.Program("typed")
    u = model.StateSpace("U", ("rho", "mx", "my"))
    state = P.state("plasma", space=u)
    shape = state.logical_shape
    assert shape["space"] == "U" and shape["n_comp"] == 3 and shape["layout"] == "cell", shape
    # An untyped (legacy) value has no space -> n_comp / layout are None (backward compatible).
    untyped = adctime.Program("legacy").state("plasma")
    assert untyped.logical_shape["space"] is None and untyped.logical_shape["n_comp"] is None
    print("OK  logical_shape is derived from the space tag (None for an untyped value)")


def test_source_location_capture_is_opt_in_and_out_of_hash():
    # Capture ON records the authoring line; the IR hash is IDENTICAL to the capture-OFF build.
    off = _euler()
    on = adctime.Program("forward_euler").capture_source_locations(True)
    U = on.state("plasma")
    R = on._rhs_legacy(state=U, fields=on.solve_fields(U), flux=True, sources=["default"])
    on.commit(on.state("U", block="plasma").next, on.linear_combine("U1", U + (1.0 * on.dt) * R))
    located = [n for n in on.ir_nodes() if n["source_location"]]
    assert located, "capture ON must populate at least one source_location"
    loc = located[0]["source_location"]
    file_part, _, line_part = loc.rpartition(":")
    assert file_part.endswith(".py") and line_part.isdigit(), loc
    # BIT-IDENTITY: the source location is inspection-only, excluded from _serialize / _ir_hash.
    assert on._ir_hash() == off._ir_hash(), "source_location must not change the IR hash"
    print("OK  source_location is opt-in and excluded from the IR hash (bit-identity preserved)")


def test_space_tag_changes_the_hash():
    # A StateSpace controls component order/layout and therefore belongs to compiled identity.
    def build(tag):
        P = adctime.Program("forward_euler")
        u = model.StateSpace("U", ("rho", "mx", "my")) if tag else None
        U = P.state("plasma", space=u)
        R = P._rhs_legacy(state=U, fields=P.solve_fields(U), flux=True, sources=["default"])
        P.commit(P.state("U", block="plasma").next, P.linear_combine("U1", U + P.dt * R))
        return P
    assert build(True)._ir_hash() != build(False)._ir_hash()
    print("OK  the operator-first space tag participates in the IR hash")


def test_missing_commit_rejected():
    P = adctime.Program("p")
    U = P.state("plasma")
    P._rhs_legacy(state=U, fields=P.solve_fields(U))
    try:
        P.validate()
        raise AssertionError("a program with no commit must be rejected")
    except ValueError as exc:
        assert "commit" in str(exc), str(exc)
    print("OK  a program with no commit is rejected")


def test_double_commit_rejected():
    P = adctime.Program("p")
    U = P.state("plasma")
    U1 = P.linear_combine("U1", U + P.dt * P._rhs_legacy(state=U, fields=P.solve_fields(U)))
    P.commit(P.state("U", block="plasma").next, U1)
    try:
        P.commit(P.state("U", block="plasma").next, U1)
        raise AssertionError("a double commit must be rejected")
    except ValueError as exc:
        assert "committed more than once" in str(exc), str(exc)
    print("OK  a double commit is rejected")


def test_distinct_field_context_per_stage():
    P = adctime.Program("p")
    U = P.state("plasma")
    f0 = P.solve_fields(U)
    U1 = P.linear_combine("U1", U + P.dt * P._rhs_legacy(state=U, fields=f0))
    f1 = P.solve_fields(U1)
    assert f0 is not f1 and f0.id != f1.id
    # Each stage's FieldContext is tagged with the state it was solved from (no stale global aux).
    assert f0.field_context.stage_sources != f1.field_context.stage_sources, (
        "each stage gets a FieldContext keyed on its own stage state")
    print("OK  each stage's solve_fields is a distinct FieldContext keyed on its own state")


def test_value_refuses_len_and_range():
    P = adctime.Program("p")
    U = P.state("plasma")
    try:
        len(U)
        raise AssertionError("len(value) must raise")
    except TypeError as exc:
        assert "no Python len()" in str(exc), str(exc)
    try:
        range(U)  # uses __index__
        raise AssertionError("range(value) must raise")
    except TypeError as exc:
        assert "Python index" in str(exc), str(exc)
    print("OK  a runtime IR value refuses len() and range()")


def main():
    test_ir_node_has_identity_and_inspection_fields()
    test_logical_shape_reflects_the_space_tag()
    test_source_location_capture_is_opt_in_and_out_of_hash()
    test_space_tag_changes_the_hash()
    test_missing_commit_rejected()
    test_double_commit_rejected()
    test_distinct_field_context_per_stage()
    test_value_refuses_len_and_range()
    print("OK  test_program_ir_nodes")


if __name__ == "__main__":
    main()
