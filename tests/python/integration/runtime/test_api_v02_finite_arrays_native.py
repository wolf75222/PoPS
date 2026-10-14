"""Installed PoPS reception of immutable replicated arrays, including MPI fault injection.

Fault injection changes one generated producer/allocation only. It is deliberately separate
from public scientific authoring and tests the collective boundary before a rank diverges.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.time import FixedDt
from examples.api_v02.methods_T01_T12 import _storage, _initial
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics
from tests.python.support.api_v02_native import execute_case
from tests.python.support.public_native_profile import native_profile


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
_FLOAT_MARKER = 970113.125


def _build(kind):
    case, layout, model, block, state, p, q, initial = _storage(
        "finite_"+kind, ("a", "b"), (1., 2.), cells=(8, 4), sampling="cell_average")
    q.n
    expected = initial.copy()
    if kind == "compositions":
        count = p.int64((1 << 53)+3) - (1 << 53)
        domain = p.range_set(count, capacity=5, ownership="replicated")
        values = p.map(domain, lambda b, i: b.product(
            n=b.index_value(i)+1, x=b.float64(b.index_value(i))+0.25))
        seed = p.product(a=values, k=p.int64(0))
        def step(b, old):
            out = b.fold(old["a"], 2, lambda b, a, k: b.map(b.domain(a),
                lambda b, i: b.product(n=a[i]["n"]+1, x=a[i]["x"]+0.5)))
            return b.product(a=out, k=old["k"]+1)
        values = p.require_success(p.iterate(seed, step, lambda b, s: s["k"] >= 2,
                                            max_steps=2))["state"]["a"]
        reversed_indices = p.map(domain, lambda b, i: b.index(domain, 2-b.index_value(i)))
        gathered = p.gather(values, reversed_indices)
        scan = p.scan(domain, p.int64(0), lambda b, s, i:
            b.product(state=s+gathered[i]["n"], value=s+gathered[i]["n"]), order="ascending")
        total = p.reduce(scan["values"], p.int64(0), lambda b, s, x: s+x, order="ascending")
        real_sum = p.reduce(values, p.float64(0), lambda b, s, x: s+x["x"], order="ascending")
        left = p.value("integer_weighted", p.float64(total)*q.n)
        right = p.value("real_weighted", real_sum*q.n)
        selected = p.value("array_result", (left[0],right[1]))
        # Independent ordered NumPy oracle: [1,2,3]+4, reverse, prefix sum then sum.
        expected[0] *= np.cumsum((np.arange(1,4)+4)[::-1]).sum()
        expected[1] *= (np.arange(3)+0.25+2).sum()
    elif kind == "empty":
        domain = p.range_set(0, capacity=0, ownership="replicated")
        def empty_body(b, s, i):
            # Must remain unreachable at runtime, though it is recorded and type checked once.
            impossible = b.index(domain, 0)
            return b.product(state=s+1, value=b.index_value(impossible))
        scan = p.scan(domain, p.int64(17), empty_body, order="ascending")
        count = p.reduce(scan["values"], p.int64(0), lambda b, s, x: s+1, order="ascending")
        left = p.value("empty_state",p.float64(scan["state"])*q.n)
        right = p.value("empty_count",p.float64(count)*q.n)
        selected = p.value("empty_result", (left[0],right[1]))
        expected[0] *= 17
        expected[1] *= 0
    else:
        def trial(b):
            d = b.range_set(2, capacity=2, ownership="replicated")
            if kind == "stale_index":
                indices = b.map(d, lambda b, i:
                    b.index(b.range_set(1, capacity=1, ownership="replicated"), 0))
                first, second = indices[b.index(d,0)], indices[b.index(d,1)]
                a = b.map(b.domain(first), lambda b, i: b.int64(7))
                return a[second]
            if kind == "capacity":
                return b.size(b.range_set(3, capacity=2, ownership="replicated"))
            if kind == "bounds":
                a = b.map(d, lambda b, i: b.index_value(i))
                return a[b.index(d,2)]
            if kind in ("signed_zero", "nonfinite", "allocation"):
                a = b.map(d, lambda b, i: b.float64(_FLOAT_MARKER))
                return a[b.index(d,0)]
            raise ValueError(kind)
        attempt = p.capture(trial)
        selected = p.match(attempt,
            ok=lambda b, _: b.value("unexpected_success", 0.*q.n),
            err=lambda b, e: b.branch(e["origin"] == 2,
                lambda b: b.branch(e["code"] == 3, lambda b: q.n,
                    lambda b: b.value("wrong_code", 0.*q.n)),
                lambda b: b.value("wrong_origin", 0.*q.n)))
    p.commit(q.next, p.value("accepted", selected, at=q.next.point))
    p.step_strategy(FixedDt(.01)); case.program(p)
    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        return {"passed": bool(np.array_equal(actual,expected)), "metrics_owner": True,
                "max_error": float(np.max(np.abs(actual-expected)))}
    return CaseSpec(case, layout, _initial(case,block[state],initial), {"method":block[state]},
                    .01,.01,check,"native finite arrays: "+kind)


@pytest.mark.parametrize("kind", ("compositions", "empty", "stale_index", "capacity", "bounds"))
def test_native_finite_array_composition(kind, isolated_native_cache, native_cxx, kokkos_root,
                                       record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _, world = native_profile(dimension=2,scope="finite arrays "+kind)
    _, artifact, metrics = execute_case(lambda:_build(kind),world,"finite arrays "+kind)
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
    record_property("metrics",str(metrics))


@pytest.mark.parametrize("kind", ("signed_zero", "nonfinite", "allocation"))
def test_native_finite_array_collective_fault(kind, monkeypatch, isolated_native_cache,
                                             native_cxx, kokkos_root, record_property):
    del isolated_native_cache,native_cxx,kokkos_root
    _, world = native_profile(dimension=2,scope="finite arrays fault "+kind)
    if world is None or world.size < 2:
        pytest.skip("replica disagreement/allocation reception requires explicit MPI2")
    import pops.codegen.program_codegen as codegen
    original = codegen._emit_cpp_program_impl
    def inject(*args,**kwargs):
        source = original(*args,**kwargs)
        if kind == "allocation":
            needle = "try { finite_"
            # Domain allocation remains valid; target precisely the first vector allocation.
            lines = source.splitlines()
            matches = [i for i,line in enumerate(lines) if needle in line and "_buffer = std::make_shared" in line]
            assert len(matches) == 1
            n = matches[0]
            lines[n] = lines[n].replace("try { ","try { if (ctx.prepared_execution_lane().rank() == 1) throw std::bad_alloc(); ",1)
            return "\n".join(lines)+"\n"
        literal = "pops::Real(%s)" % _FLOAT_MARKER.hex()
        assert source.count(literal) == 1
        expression = ("(ctx.prepared_execution_lane().rank() == 0 ? pops::Real(0.0) : pops::Real(-0.0))"
                      if kind == "signed_zero" else
                      "(ctx.prepared_execution_lane().rank() == 0 ? pops::Real(0.0) : std::numeric_limits<pops::Real>::quiet_NaN())")
        return source.replace(literal,expression)
    monkeypatch.setattr(codegen,"_emit_cpp_program_impl",inject)
    _, artifact, metrics = execute_case(lambda:_build(kind),world,"finite arrays fault "+kind)
    record_property("fault_injection",kind+" at one exact generated producer/allocation")
    record_property("artifact_identity",artifact.artifact_identity.hexdigest)
    record_property("metrics",str(metrics))
