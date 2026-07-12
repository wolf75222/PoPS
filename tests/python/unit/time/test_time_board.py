"""Spec 3 board-like time programs (pops.time board sugar).

T.fields / T.define / T.solve / T.commit_many are blackboard notation that lowers
to the SAME Program IR as the primitive solve_fields / linear_combine /
solve_local_linear / commit calls. These tests assert that IR identity, plus the
RateBundle / StageStateSet / atomic commit_many behaviour.
"""
from typed_program_support import commits_by_block, state_refs, typed_state

import pytest

from pops import model as _model
from pops.ir import ScalarLiteral
from pops.time import Program
from pops.math import rate, unknown


def _ir(P):
    """Structural IR of a Program: per value (vtype, op, input positions, attrs, block)."""
    from pops.time.program_serialization import _json_ready
    from pops.time.references import handle_data

    idx = {id(v): k for k, v in enumerate(P._values)}
    out = []
    for v in P._values:
        ins = tuple(idx[id(i)] for i in v.inputs)
        block = None if v.block is None else handle_data(v.block)
        out.append((v.vtype, v.op, ins, _json_ready(v.attrs), block))
    return out


def test_fields_define_commit_match_primitive_ir():
    def build(board):
        P = Program("fe")
        dt = P.dt
        u = typed_state(P, "plasma")
        f = P.fields("f", from_state=u) if board else P.solve_fields("f", u)
        r = P._rhs_legacy(name="R", state=u, fields=f, flux=True, sources=["electric"])
        endpoint = typed_state(P, "plasma", state_name="U").next
        if board:
            u1 = P.define("U1", u + dt * r, at=endpoint.point)
        else:
            u1 = P.linear_combine("U1", u + dt * r, at=endpoint.point)
        P.commit(endpoint, u1)
        return P

    assert _ir(build(True)) == _ir(build(False))


def test_solve_matches_linear_combine_plus_solve_local_linear():
    def build(board):
        P = Program("imp")
        dt = P.dt
        u = typed_state(P, "plasma")
        r = P._rhs_legacy(name="R", state=u, flux=True, sources=["electric"])
        endpoint = typed_state(P, "plasma", state_name="U").next
        if board:
            u1 = P.solve(
                "U1",
                (P.I - dt * P._linear_source("lorentz")) @ unknown("U1") == u + dt * r,
                at=endpoint.point,
            )
        else:
            op = P.I - dt * P._linear_source("lorentz")  # primitive private selector seam
            rhs = P.linear_combine("U1_rhs", u + dt * r, at=endpoint.point)
            u1 = P.solve_local_linear(name="U1", operator=op, rhs=rhs)
        P.commit(endpoint, u1)
        return P

    assert _ir(build(True)) == _ir(build(False))


def test_apply_operator_to_state_via_matmul():
    P = Program("apply")
    u = typed_state(P, "plasma")
    lu_board = P._linear_source("lorentz") @ u
    lu_manual = P.apply(operator=P._linear_source("lorentz"), state=u)
    assert lu_board.op == "apply" and lu_board.attrs["linear_source"] == "lorentz"
    assert lu_manual.op == "apply"


def test_define_equation_replaces_and_renames_rhs_immutably():
    P = Program("def")
    u = typed_state(P, "plasma")
    raw = P._rhs_legacy(name="tmp", state=u, flux=True, sources=["electric"])
    r = P.define("R^n", rate(u) == raw)
    assert r is not raw
    assert r.id == raw.id      # same SSA identity, immutable replacement object
    assert r.name == "R^n"     # renamed to the board label
    assert any(value is r for value in P._values)


def test_commit_many_is_atomic():
    P = Program("ms")
    e = typed_state(P, "electrons")
    i = typed_state(P, "ions")
    e_endpoint = typed_state(P, "electrons", state_name="U").next
    i_endpoint = typed_state(P, "ions", state_name="U").next
    e1 = P.linear_combine("e1", 2.0 * e, at=e_endpoint.point)
    i1 = P.linear_combine("i1", 2.0 * i, at=i_endpoint.point)
    P.commit_many({e_endpoint: e1, i_endpoint: i1})
    assert set(commits_by_block(P)) == {"electrons", "ions"}


def test_commit_many_rejects_double_commit_without_partial():
    P = Program("ms")
    e = typed_state(P, "electrons")
    i = typed_state(P, "ions")
    e_endpoint = typed_state(P, "electrons", state_name="U").next
    i_endpoint = typed_state(P, "ions", state_name="U").next
    e1 = P.linear_combine("e1", 2.0 * e, at=e_endpoint.point)
    i1 = P.linear_combine("i1", 2.0 * i, at=i_endpoint.point)
    P.commit(e_endpoint, e1)
    with pytest.raises(ValueError, match="committed more than once"):
        P.commit_many({e_endpoint: e1, i_endpoint: i1})
    # atomic: 'ions' must NOT have been committed because validation failed first
    assert "ions" not in commits_by_block(P)


def test_commit_many_rejects_non_state():
    P = Program("ms")
    e = typed_state(P, "electrons")
    scalar = P.norm2(e)
    with pytest.raises(TypeError, match="needs a State or scalar_field ProgramValue"):
        P.commit_many({typed_state(P, "electrons", state_name="U").next: scalar})


def test_state_set_drives_a_multi_block_field_solve():
    P = Program("ss")
    e = typed_state(P, "electrons")
    i = typed_state(P, "ions")
    n = typed_state(P, "neutrals")
    star = P.state_set("star", {e.block: e, i.block: i, n.block: n})
    assert len(star) == 3
    f = P.fields("fstar", from_state_set=star)
    assert f.vtype == "fields" and f.op == "solve_fields_from_blocks"
    assert len(f.inputs) == 3


def test_state_set_rejects_stringified_or_mismatched_block_identity():
    P = Program("strict_state_set")
    state = typed_state(P, "electrons")

    with pytest.raises(TypeError, match="name must be a non-empty string"):
        P.state_set(object(), {"electrons": state})
    with pytest.raises(TypeError, match="BlockHandle"):
        P.state_set("stage", {object(): state})
    ions, _ = state_refs(P, "ions")
    with pytest.raises(ValueError, match="that block's State"):
        P.state_set("stage", {ions: state})


def test_rate_bundle_typed_multi_output():
    e = _model.StateSpace("electron_state", ["ne", "mex", "mey"])
    i = _model.StateSpace("ion_state", ["ni", "mix", "miy"])
    rb = _model.RateBundle({"electrons": _model.Rate(e), "ions": _model.Rate(i)})
    assert rb["electrons"] == _model.Rate(e)
    rb.require("electrons", e)  # correct StateSpace -> ok
    with pytest.raises(TypeError):
        rb.require("electrons", i)  # wrong Rate on wrong StateSpace -> rejected


def test_record_and_check_invariant_lower_to_record_scalar():
    P = Program("inv")
    e = typed_state(P, "electrons")
    before = P.sum(e)                      # a Program scalar (reduction)
    P.record("mass", before)               # board diagnostic
    e1 = P.linear_combine("e1", 2.0 * e)
    after = P.sum(e1)
    out = P.check_invariant("mass", before=before, after=after, tolerance=1e-9)
    assert out.vtype == "scalar"
    assert out.attrs.get("tolerance") == ScalarLiteral.from_value(1e-9)
    assert [v for v in P._values if v.op == "record_scalar"]  # both recorded


def test_record_rejects_non_scalar():
    P = Program("inv")
    e = typed_state(P, "electrons")
    with pytest.raises(ValueError, match="must be a Program scalar"):
        P.record("bad", e)  # a State, not a scalar


def test_rate_bundle_arbitrary_arity():
    spaces = {name: _model.StateSpace(name + "_state", ["n", "mx", "my"])
              for name in ("a", "b", "c", "d")}
    rb = _model.RateBundle({k: _model.Rate(v) for k, v in spaces.items()})
    assert len(rb) == 4  # no 2-input limit
    for k, v in spaces.items():
        rb.require(k, v)
