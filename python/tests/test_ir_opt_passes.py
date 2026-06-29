#!/usr/bin/env python3
"""pops.time IR optimization passes over canonical operator-first Programs."""

import sys

from pops.ir.expr import Const
from pops import model as pm
from pops import time as adctime
import pops.lib.time as libtime


def _ops_module(name="ops", ncomp=1):
    m = pm.Module(name + "_module")
    U = m.state_space("U", tuple("q%d" % i for i in range(ncomp)))
    F = m.field_space("Fields", ("phi",))
    fields = m.operator(
        "fields_from_state",
        signature=(U,) >> F,
        kind="field_operator",
        capabilities={"default": True},
        expr=Const(0.0),
    )
    rhs = m.operator(
        "rhs",
        signature=(U, F) >> pm.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(0.0) for _ in range(ncomp)],
    )
    rhs_plain = m.operator(
        "rhs_plain",
        signature=(U,) >> pm.Rate(U),
        kind="local_source",
        capabilities={"produces_rate": True, "supports_device": True},
        expr=[Const(0.0) for _ in range(ncomp)],
    )
    return m, fields, rhs, rhs_plain


def _program(name, module):
    P = adctime.Program(name).bind_operators(module)
    P._test_model = module
    P._registry._test_model = module
    return P


def _emit(P):
    module = getattr(P, "_test_model", None)
    if module is None and getattr(P, "_registry", None) is not None:
        module = getattr(P._registry, "_test_model", None)
    return P.emit_cpp_program(model=module)


def _state(P, block="plasma"):
    return P.state("U", block=block).n


def _field_call_count(P):
    return sum(1 for v in P._values if v.op == "call" and v.vtype == "fields")


def _rate_call_count(P):
    return sum(1 for v in P._values if v.op == "call" and v.vtype == "rhs")


def _euler_clean():
    """Forward Euler, no optimizable structure."""
    m, fields_op, rhs_op, _ = _ops_module("euler_clean")
    P = _program("forward_euler", m)
    U = _state(P)
    fields = P.call(fields_op, U, name="fields")
    R = P.call(rhs_op, U, fields, name="R")
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    return P


def _cse_dup_program():
    P = adctime.Program("cse")
    U = P._state_value("plasma")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 2.0 * U)
    m1 = P.cell_compare(U, 0.0, ">", name="mask")
    m2 = P.cell_compare(U, 0.0, ">", name="mask_dup")
    w1 = P.where(m1, a, b, name="w1")
    w2 = P.where(m2, b, a, name="w2")
    P.commit("plasma", P.linear_combine("U1", 0.5 * w1 + 0.5 * w2))
    return P


def _cse_handwritten():
    P = adctime.Program("cse")
    U = P._state_value("plasma")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 2.0 * U)
    m = P.cell_compare(U, 0.0, ">", name="mask")
    w1 = P.where(m, a, b, name="w1")
    w2 = P.where(m, b, a, name="w2")
    P.commit("plasma", P.linear_combine("U1", 0.5 * w1 + 0.5 * w2))
    return P


def test_cse_collapses_duplicate_pure_subir():
    P = _cse_dup_program()
    assert sum(1 for v in P._values if v.op == "cell_compare") == 2
    Q = adctime.eliminate_common_subexpressions(P)
    assert Q is not P
    assert sum(1 for v in P._values if v.op == "cell_compare") == 2
    assert sum(1 for v in Q._values if v.op == "cell_compare") == 1
    assert sum(1 for v in Q._values if v.op == "where") == 2


def test_cse_byte_identical_to_handwritten():
    Q = adctime.eliminate_common_subexpressions(_cse_dup_program())
    H = _cse_handwritten()
    assert Q._ir_hash() == H._ir_hash()
    assert _emit(Q) == _emit(H)
    assert _cse_dup_program()._ir_hash() != H._ir_hash()


def test_cse_noop_byte_identical_when_nothing_duplicated():
    P = _euler_clean()
    Q = adctime.eliminate_common_subexpressions(P)
    assert Q._ir_hash() == P._ir_hash()
    assert _emit(Q) == _emit(P)


def test_cse_never_collapses_side_effecting_field_calls():
    m, fields_op, rhs_op, _ = _ops_module("two_fields")
    P = _program("two_fields", m)
    U = _state(P)
    f1 = P.call(fields_op, U, name="f1")
    f2 = P.call(fields_op, U, name="f2")
    R1 = P.call(rhs_op, U, f1, name="R1")
    R2 = P.call(rhs_op, U, f2, name="R2")
    P.commit("plasma", P.linear_combine("U1", U + 0.5 * P.dt * R1 + 0.5 * P.dt * R2))
    Q = adctime.eliminate_common_subexpressions(P)
    assert _field_call_count(Q) == 2
    assert Q._ir_hash() == P._ir_hash()


def test_cse_never_collapses_reduce_or_aux_reading_rate_call():
    m, fields_op, rhs_op, _ = _ops_module("reduce_aux")
    P = _program("reduce_aux", m)
    U = _state(P)
    f = P.call(fields_op, U, name="f")
    R = P.call(rhs_op, U, f, name="R")
    P.record_scalar("n1", P.norm2(R))
    P.record_scalar("n2", P.norm2(R))
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    Q = adctime.eliminate_common_subexpressions(P)
    assert sum(1 for v in Q._values if v.op == "reduce") == 2
    assert Q._ir_hash() == P._ir_hash()

    P2 = _program("aux_rate_cse", m)
    U2 = _state(P2)
    f0 = P2.call(fields_op, U2, name="f0")
    R1 = P2.call(rhs_op, U2, f0, name="R1")
    P2.call(fields_op, U2, name="f_refresh")
    R2 = P2.call(rhs_op, U2, f0, name="R2")
    P2.commit("plasma", P2.linear_combine("U1", U2 + P2.dt * R1 + P2.dt * R2))
    Q2 = P2.optimize()
    assert _rate_call_count(Q2) == 2


def test_redundant_field_call_removed_when_no_mutation():
    m, fields_op, rhs_op, _ = _ops_module("redundant")
    P = _program("redundant", m)
    U = _state(P)
    P.call(fields_op, U, name="f1")
    f2 = P.call(fields_op, U, name="f2")
    R = P.call(rhs_op, U, f2, name="R")
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    Q = adctime.eliminate_redundant_field_solves(P)
    assert _field_call_count(Q) == 1
    assert set(Q.commits()) == {"plasma"}
    assert Q._ir_hash() != P._ir_hash()


def test_redundant_field_call_kept_when_state_mutated():
    m, fields_op, rhs_op, _ = _ops_module("project_barrier")
    P = _program("project_barrier", m)
    U = _state(P)
    P.call(fields_op, U, name="f1")
    P.project(U)
    f2 = P.call(fields_op, U, name="f2")
    R = P.call(rhs_op, U, f2, name="R")
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    Q = adctime.eliminate_redundant_field_solves(P)
    assert _field_call_count(Q) == 2
    assert Q._ir_hash() == P._ir_hash()


def test_redundant_field_call_kept_when_fill_boundary_intervenes():
    m, fields_op, rhs_op, _ = _ops_module("fill_barrier")
    P = _program("fill_barrier", m)
    U = _state(P)
    P.call(fields_op, U, name="f1")
    P.fill_boundary(U)
    f2 = P.call(fields_op, U, name="f2")
    R = P.call(rhs_op, U, f2, name="R")
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    Q = adctime.eliminate_redundant_field_solves(P)
    assert _field_call_count(Q) == 2


def test_redundant_field_call_noop_byte_identical():
    P = _euler_clean()
    Q = adctime.eliminate_redundant_field_solves(P)
    assert Q._ir_hash() == P._ir_hash()
    assert _emit(Q) == _emit(P)


def _rk4():
    m, _, _, rhs = _ops_module("rk4")
    P = _program("rk4", m)
    libtime.rk4(P, "plasma", rhs_operator=rhs)
    return P


def test_scratch_liveness_sane():
    P = _rk4()
    rows = P.scratch_liveness()
    assert rows
    n = len(P._values)
    for r in rows:
        assert 0 <= r["def_index"] <= r["last_use_index"], r
        assert r["live_span"] == r["last_use_index"] - r["def_index"] >= 0, r
        assert r["def_index"] < n, r
    assert sum(1 for r in rows if r["op"] == "call") == 4, rows


def test_buffer_reuse_consistent_and_saves():
    P = _rk4()
    rep = P.buffer_reuse_report()
    assert rep["buffer_count"] <= rep["scratch_count"], rep
    assert rep["reused"] == rep["scratch_count"] - rep["buffer_count"], rep
    assert rep["buffer_count"] >= 1
    ranges = {r["name"]: r for r in P.scratch_liveness()}
    by_buf = {}
    for name, buf in rep["assignment"].items():
        by_buf.setdefault(buf, []).append(name)
    for buf, names in by_buf.items():
        names.sort(key=lambda nm: ranges[nm]["def_index"])
        for earlier, later in zip(names, names[1:], strict=False):
            assert ranges[earlier]["last_use_index"] < ranges[later]["def_index"], (buf, earlier, later)
    assert rep["scratch_count"] - rep["buffer_count"] >= 1, rep


def test_estimate_internally_consistent():
    P = _rk4()
    est = P.estimate()
    assert est["kernel_count"] >= est["small_kernels"] + est["heavy_kernels"]
    assert est["traffic_fields"] == est["field_reads"] + est["field_writes"]
    assert est["buffers_saved"] == est["scratch_count"] - est["buffer_count"] >= 0
    assert est["heavy_kernels"] == 0
    assert est["small_kernels"] >= 4
    rep = P.estimate_report()
    for token in ("kernels", "scratch buffers", "memory traffic", "GPU detectors"):
        assert token in rep, rep


def _pathological():
    P = adctime.Program("patho")
    U = P._state_value("plasma")
    a = P.linear_combine("a", 1.0 * U)
    b = P.linear_combine("b", 2.0 * U)
    acc = 1.0 * U
    for i in range(20):
        mask = P.cell_compare(U, float(i), ">", name="m%d" % i)
        where = P.where(mask, a, b, name="w%d" % i)
        acc = acc + 0.01 * where
    P.commit("plasma", P.linear_combine("U1", acc))
    return P


def test_gpu_detectors_flag_pathological_ir():
    warns = _pathological().gpu_detectors()
    names = {w["detector"] for w in warns}
    assert "too_many_small_kernels" in names, names
    assert "too_many_scratches" in names, names
    assert "excessive_memory_traffic" in names, names
    for w in warns:
        assert w["value"] > w["threshold"]
        assert w["message"]


def test_gpu_detectors_quiet_on_well_behaved_ir():
    assert _euler_clean().gpu_detectors() == []


def test_optimize_byte_identical_when_nothing_optimizable():
    for build in (_euler_clean, _rk4):
        P = build()
        Q = P.optimize()
        assert Q._ir_hash() == P._ir_hash(), P.name
        assert _emit(Q) == _emit(P), P.name


def test_optimize_runs_all_proven_safe_passes():
    m, fields_op, rhs_op, _ = _ops_module("all")
    P = _program("all", m)
    U = _state(P)
    P.call(fields_op, U, name="f1")
    f2 = P.call(fields_op, U, name="f2")
    R = P.call(rhs_op, U, f2, name="R")
    P.call(rhs_op, U, f2, name="dead")
    P.commit("plasma", P.linear_combine("U1", U + P.dt * R))
    Q = P.optimize()
    assert _field_call_count(Q) == 1
    assert "dead" not in {v.name for v in Q._values}
    assert set(Q.commits()) == {"plasma"}
    assert Q.validate()


def test_dump_passes_traces_pipeline():
    P = _euler_clean()
    before = [(v.op, v.name) for v in P._values]
    trace = P.dump_passes()
    assert "dead-node elimination" in trace
    assert "common-subexpression elimination" in trace
    assert "redundant field-solve elimination" in trace
    assert [(v.op, v.name) for v in P._values] == before


def test_method_and_free_function_forms_agree():
    P = _cse_dup_program()
    assert adctime.eliminate_common_subexpressions(P)._ir_hash() == P.eliminate_common_subexpressions()._ir_hash()
    R = _euler_clean()
    assert adctime.optimize(R)._ir_hash() == R.optimize()._ir_hash()

    m, fields_op, rhs_op, _ = _ops_module("free_redundant")
    S = _program("rs", m)
    U = _state(S)
    S.call(fields_op, U, name="f1")
    f2 = S.call(fields_op, U, name="f2")
    Rr = S.call(rhs_op, U, f2, name="R")
    S.commit("plasma", S.linear_combine("U1", U + S.dt * Rr))
    assert (adctime.eliminate_redundant_field_solves(S)._ir_hash()
            == S.eliminate_redundant_field_solves()._ir_hash())


def _run_as_script():
    fails = 0
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    for name, fn in tests:
        try:
            fn()
            print("  [OK ] %s" % name)
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print("  [XX ] %s -- %s" % (name, exc))
    print("FAILS =", fails)
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run_as_script() else 0)
