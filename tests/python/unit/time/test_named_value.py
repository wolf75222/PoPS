"""ADC-561: the short named-value API T.value(name, expr).

``T.value("name", expr)`` names an intermediate SSA value and lowers to the EXACT
``program.define(name, expr)`` path, so it produces the byte-identical IR as the long
``T.define("name", expr)`` form. ``U.stage(k)`` stays the temporal-version handle only and
``T.commit(U.next, value)`` is the only end-of-step write door. The SSA invariants (single
definition, no redefine, use-before-define) are unchanged; named values appear in inspection.

Pure Python (``_ir_hash`` is the IR fingerprint; no compilation); skips cleanly if pops is
unavailable. Never fakes the engine.
"""
import sys

try:
    import pytest
    from pops.time import Program
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_named_value (pops unavailable: %s)" % exc)
    sys.exit(0)


def _heun(P, use_value):
    """A Heun-like step written with the short T.value or the long T.define for the intermediate."""
    U = P.state("U", block="plasma")
    R = P._rhs_legacy(state=U.n, fields=P.solve_fields(U.n), sources=["default"])
    name = "rhs_star"
    U_star = (P.value(name, U.n + P.dt * R) if use_value
              else P.define(name, U.n + P.dt * R))
    R_star = P._rhs_legacy(state=U_star, fields=P.solve_fields(U_star), sources=["default"])
    U_next = P.value("U_next", U.n + 0.5 * P.dt * R + 0.5 * P.dt * R_star)
    P.commit(U.next, U_next)


def test_value_ir_byte_identical_to_define():
    """T.value(name, expr) lowers to the byte-identical IR as the long T.define(name, expr)."""
    via_value = Program("heun")
    _heun(via_value, use_value=True)
    via_value.validate()
    via_define = Program("heun")
    _heun(via_define, use_value=False)
    via_define.validate()
    assert via_value._ir_hash() == via_define._ir_hash(), (
        "T.value must lower to the same IR as T.define\n"
        "  value : %s\n  define: %s" % (via_value._ir_hash(), via_define._ir_hash()))
    print("OK  T.value(name, expr) IR hash == T.define(name, expr): %s" % via_value._ir_hash())


def test_named_value_appears_in_inspection():
    """A named value shows up in the structured IR inspection under its name."""
    P = Program("insp")
    U = P.state("U", block="plasma")
    R = P._rhs_legacy(state=U.n, fields=P.solve_fields(U.n), sources=["default"])
    v = P.value("Q", U.n + 0.5 * P.dt * R)
    assert v.name == "Q"
    names = [n["name"] for n in P.ir_nodes()]
    assert "Q" in names, names
    print("OK  named value 'Q' appears in ir_nodes()")


def test_value_returns_composable_handle():
    """The returned handle composes in the affine algebra like any State value."""
    P = Program("compose")
    U = P.state("U", block="plasma")
    R = P._rhs_legacy(state=U.n, fields=P.solve_fields(U.n), sources=["default"])
    U_star = P.value("U_star", U.n + P.dt * R)
    # It reads as a State value and composes further.
    assert U_star.vtype == "state"
    combined = P.value("combined", 0.5 * U.n + 0.5 * U_star)
    assert combined.vtype == "state"
    print("OK  T.value returns a composable named State handle")


def test_value_refuses_a_version_handle():
    """T.value is for free intermediates: a version handle is refused pointing at T.define."""
    P = Program("nover")
    U = P.state("U", block="plasma")
    with pytest.raises(TypeError, match="commit"):
        P.value(U.next, U.n)
    with pytest.raises(TypeError, match="T.define"):
        P.value(U.stage(1), U.n)
    with pytest.raises(ValueError, match="non-empty string"):
        P.value("", U.n)
    print("OK  T.value refuses a version handle / empty name")


def test_ssa_invariants_unchanged():
    """The version-handle SSA guards (read-only n, no redefine, use-before-define) are unchanged."""
    P = Program("ssa")
    U = P.state("U", block="plasma")
    R = P._rhs_legacy(state=U.n, fields=P.solve_fields(U.n), sources=["default"])
    # use-before-define: a stage used before T.define raises
    with pytest.raises(ValueError, match="undefined"):
        _ = U.stage(1) + P.dt * R
    # define once, then no redefine
    P.define(U.stage(1), U.n + P.dt * R)
    with pytest.raises(ValueError, match="already defined"):
        P.define(U.stage(1), U.n)
    # current state is read-only
    with pytest.raises(ValueError, match="read-only"):
        P.define(U.n, U.n + P.dt * R)
    print("OK  SSA invariants (use-before-define / no-redefine / read-only n) unchanged")


def test_next_is_a_commit_only_endpoint():
    """T.commit(U.next, value) is the only end-of-step write door."""
    P = Program("door")
    U = P.state("U", block="plasma")
    R = P._rhs_legacy(state=U.n, fields=P.solve_fields(U.n), sources=["default"])
    U_next = P.value("U_next", U.n + P.dt * R)
    P.commit(U.next, U_next)
    P.validate()
    assert P.commits()["plasma"] is U_next
    print("OK  T.commit(U.next, value) is the commit-only endpoint door")


def main():
    test_value_ir_byte_identical_to_define()
    test_named_value_appears_in_inspection()
    test_value_returns_composable_handle()
    test_value_refuses_a_version_handle()
    test_ssa_invariants_unchanged()
    test_next_is_a_commit_only_endpoint()
    print("OK  test_named_value")


if __name__ == "__main__":
    main()
