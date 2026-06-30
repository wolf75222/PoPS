"""pops.time.Program operator-first IR tests.

These tests cover the pure Python authoring layer.  They intentionally build
rates and fields through typed module operator handles so the Program IR stays
at the final operator-first level: ``call`` + ``linear_combine`` + ``commit``.
"""

from pops import time as adctime

from _module_models import isothermal_transport_module


def _coeff(node, value):
    """Coefficient polynomial (dict power->float) attached to value in a combine node."""
    for v, c in zip(node.inputs, node.attrs["coeffs"], strict=True):
        if v is value:
            return c
    raise AssertionError("value %r not an input of %r" % (value, node))


def _program(name):
    module = isothermal_transport_module("%s_model" % name)
    program = adctime.Program(name).bind_operators(module)
    U = program.state("U", block="plasma", space=module.state_spaces()["U"])
    ops = module.operator_registry()
    return module, program, U, ops.get("fields_from_state"), ops.get("explicit_rate")


def _rate(program, fields_operator, rate_operator, state, suffix):
    fields = program.call(fields_operator, state, name="fields_%s" % suffix)
    rate = program.call(rate_operator, state, fields, name="k_%s" % suffix)
    return rate, fields


def _assert_operator_first(program):
    forbidden = {"r" + "hs", "solve" + "_fields", "linear" + "_source", "source"}
    assert {v.op for v in program._values}.isdisjoint(forbidden)
    dump = program.dump_operator_ir()
    assert "P." + "rhs" not in dump
    assert "solve" + "_fields" not in dump
    assert "linear" + "_source" not in dump


def test_forward_euler_ir():
    _, P, U, fields_operator, rate_operator = _program("forward_euler")
    R, fields = _rate(P, fields_operator, rate_operator, U.n, "n")
    U1 = P.define("U1", U.n + P.dt * R)
    P.commit("plasma", U1, fields=fields)

    P.validate()
    _assert_operator_first(P)
    assert fields.vtype == "fields" and fields.op == "call"
    assert R.vtype == "rhs" and R.op == "call"
    assert U1.vtype == "state" and U1.op == "linear_combine"
    assert set(U1.inputs) == {U.n, R}
    assert _coeff(U1, U.n) == {0: 1.0}
    assert _coeff(U1, R) == {1: 1.0}
    assert P.commits()["plasma"] is U1


def test_ssprk2_ir():
    _, P, U, fields_operator, rate_operator = _program("ssprk2")
    k0, _ = _rate(P, fields_operator, rate_operator, U.n, "0")
    U1 = P.define(U.stage(1), U.n + P.dt * k0)
    k1, fields1 = _rate(P, fields_operator, rate_operator, U.stage(1), "1")
    U2 = P.define(U.next, 0.5 * U.n + 0.5 * (U.stage(1) + P.dt * k1))
    P.commit("plasma", U.next, fields=fields1)

    P.validate()
    _assert_operator_first(P)
    assert U1 is U.stage(1).value
    assert U2 is U.next.value
    assert _coeff(U2, U.n) == {0: 0.5}
    assert _coeff(U2, U1) == {0: 0.5}
    assert _coeff(U2, k1) == {1: 0.5}


def test_rk4_ir():
    _, P, U, fields_operator, rate_operator = _program("rk4")
    k1, _ = _rate(P, fields_operator, rate_operator, U.n, "1")
    U1 = P.define("U1", U.n + 0.5 * P.dt * k1)
    k2, _ = _rate(P, fields_operator, rate_operator, U1, "2")
    U2 = P.define("U2", U.n + 0.5 * P.dt * k2)
    k3, _ = _rate(P, fields_operator, rate_operator, U2, "3")
    U3 = P.define("U3", U.n + P.dt * k3)
    k4, fields4 = _rate(P, fields_operator, rate_operator, U3, "4")
    Unp1 = P.define(
        U.next,
        U.n + P.dt / 6.0 * k1 + P.dt / 3.0 * k2 + P.dt / 3.0 * k3 + P.dt / 6.0 * k4,
    )
    P.commit("plasma", U.next, fields=fields4)

    P.validate()
    _assert_operator_first(P)
    assert _coeff(Unp1, U.n) == {0: 1.0}
    assert abs(_coeff(Unp1, k1)[1] - 1.0 / 6.0) < 1e-15
    assert abs(_coeff(Unp1, k2)[1] - 1.0 / 3.0) < 1e-15
    assert abs(_coeff(Unp1, k4)[1] - 1.0 / 6.0) < 1e-15


def test_commit_once():
    _, P, U, fields_operator, rate_operator = _program("commit_once")
    R, fields = _rate(P, fields_operator, rate_operator, U.n, "n")
    U1 = P.define(U.next, U.n + P.dt * R)
    P.commit("plasma", U.next, fields=fields)
    try:
        P.commit("plasma", U1)
    except ValueError as e:
        assert "committed more than once" in str(e), str(e)
        return
    raise AssertionError("expected ValueError on double commit")


def test_no_commit_rejected():
    _, P, U, fields_operator, rate_operator = _program("no_commit")
    _rate(P, fields_operator, rate_operator, U.n, "n")
    try:
        P.validate()
    except ValueError as e:
        assert "commit" in str(e), str(e)
        return
    raise AssertionError("expected ValueError on missing commit")


def test_value_not_python_bool():
    _, P, U, _, _ = _program("bool")
    try:
        bool(U.n)
    except TypeError as e:
        assert "Program control flow" in str(e) or "Python bool" in str(e), str(e)
        return
    raise AssertionError("expected TypeError on bool(IR value)")


def _build_euler(scale=1.0):
    _, P, U, fields_operator, rate_operator = _program("forward_euler")
    R, fields = _rate(P, fields_operator, rate_operator, U.n, "n")
    P.commit("plasma", P.define(U.next, U.n + (scale * P.dt) * R), fields=fields)
    return P


def test_ir_hash_deterministic_and_sensitive():
    assert _build_euler()._ir_hash() == _build_euler()._ir_hash()
    assert _build_euler(1.0)._ir_hash() != _build_euler(2.0)._ir_hash()


def test_operator_call_nodes_are_distinct():
    _, P, U, fields_operator, _ = _program("distinct_calls")
    f0 = P.call(fields_operator, U.n, name="fields0")
    f1 = P.call(fields_operator, U.n, name="fields1")
    assert f0 is not f1
    assert f0.id != f1.id
    assert f0.vtype == "fields"
    assert f0.op == f1.op == "call"


def test_rate_operator_call_records_handle_and_id():
    module, P, U, fields_operator, rate_operator = _program("rate_metadata")
    rate, _ = _rate(P, fields_operator, rate_operator, U.n, "n")
    operator_id = module.operator_registry().id_of("explicit_rate")
    assert rate.attrs["operator"] == "explicit_rate"
    assert rate.attrs["operator_id"] == operator_id
    assert rate.attrs["operator_handle"] == "explicit_rate"
    assert rate.attrs["output_vtype"] == "rate"
