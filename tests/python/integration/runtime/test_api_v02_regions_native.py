"""Real PoPS region semantics; run under MPI2 to exercise non-owner failure rendezvous.

Only the divergent-count test injects a native fault, at one scalar producer. It is not an
example of public scientific authoring. All other equations, nested algorithms and fallback
branches are ordinary Program algebra compiled and executed by the installed PoPS package.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.fields import CellMidpoint
from pops.time import FixedDt
from examples.api_v02.methods_T01_T12 import _storage, _initial, build_t04_global
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import (
    compile_profile_plan_once, native_profile, profile_gather_value,
)


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
_EXACT_CODE = (1 << 53) + 1
_DIVERGENCE_MARKER = 970031


def _assert_failure(p, error, code, origin, yes, no):
    return p.branch(error["origin"] == origin,
                    lambda q: q.branch(error["code"] == code, yes, no), no)


def _build(kind):
    case, layout, model, block, state, p, q, initial = _storage(
        "region_" + kind, ("a", "b"), (1., 2.), cells=(8, 4), sampling="cell_average")
    q.n  # Issue the accepted read in the enclosing graph before closed regions capture it.
    expected = initial.copy()
    if kind == "overflow":
        # There is one Uniform box: MPI2 gives one owner and one rank with no cells.
        # Only component 1 overflows; a component-0 finite check would miss the defect.
        initial[0] = 1.
        initial[1] = np.broadcast_to(np.where(np.arange(8) % 2, -1.e308, 1.e308), (4, 8))
        expected = initial.copy()
        def trial(builder):
            field = builder.cell_field(q.n, quadrature=CellMidpoint())
            lap = builder.laplacian(builder.scalar_field(like=field), field)
            return builder.cell_mean(lap, template=q.n, quadrature=CellMidpoint())
        attempted = p.capture(trial)
        selected = p.match(attempted,
            ok=lambda builder, value: builder.value("unexpected_success", (0.*value[0], 0.*value[1])),
            err=lambda builder, error: _assert_failure(builder, error, 3, 2,
                lambda inner: q.n, lambda inner: inner.value("bad_cause", 0.*q.n)))
    elif kind == "exact_counter":
        count = (p.int64(1 << 53) + 1) - (1 << 53)
        selected = p.fold(q.n, count,
            lambda builder, value, index: builder.value("increment", (value[0]+5., value[1]+5.)),
            max_steps=1)
        expected += 5.
    elif kind in ("nested_error_fallback", "terminal_error"):
        def trial(builder):
            carry = builder.product(value=q.n, count=builder.int64(0))
            def step(inner, old):
                candidate = inner.product(
                    value=inner.value("discarded_candidate", (old["value"][0]+17., old["value"][1]+17.)),
                    count=old["count"]+1)
                return inner.branch(old["count"] == 0,
                    lambda arm: arm.success(candidate),
                    lambda arm: arm.failure(candidate, _EXACT_CODE))
            iteration = builder.iterate(carry, step,
                lambda inner, old: inner.boolean(False), max_steps=3)
            return builder.require_success(iteration)["state"]["value"]
        # Both captures must preserve the inner user cause, rather than resetting to runtime code 3.
        attempted = p.capture(lambda outer: outer.require_success(outer.capture(trial)))
        if kind == "terminal_error":
            selected = p.require_success(attempted)
        else:
            selected = p.match(attempted,
                ok=lambda builder, value: value,
                err=lambda builder, error: _assert_failure(builder, error, _EXACT_CODE, 1,
                    lambda inner: inner.value("fallback", (q.n[0]+7., q.n[1]+7.)),
                    lambda inner: inner.value("bad_cause", 0.*q.n)))
            expected += 7.
    elif kind == "divergent_count":
        # Test-only code generation injects lane.rank()+1 for this exact constant below.
        count = p.int64(_DIVERGENCE_MARKER)
        attempted = p.capture(lambda builder: builder.fold(q.n, count,
            lambda inner, value, index: inner.value("unreachable", 2.*value),
            max_steps=_DIVERGENCE_MARKER + 1))
        selected = p.match(attempted,
            ok=lambda builder, value: builder.value("unexpected_success", 0.*value),
            err=lambda builder, error: _assert_failure(builder, error, 3, 2,
                lambda inner: q.n, lambda inner: inner.value("bad_cause", 0.*q.n)))
    else:
        raise ValueError(kind)
    p.commit(q.next, p.value("accepted", selected, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        return {"passed": bool(np.array_equal(actual, expected)), "metrics_owner": True}
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    .01, .01, check, "native region reception: " + kind)


def _compile(builder, scope):
    _native, world = native_profile(dimension=2, scope=scope)
    spec = collective_call(world, builder)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values={key: value.copy() for key, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}))
    return spec, artifact, runtime, world


def _check_success(spec, runtime, world, max_steps=1):
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=max_steps, console=False))
    rows = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        owners = [row for row in rows if row.get("metrics_owner")]
        assert owners and all(row["passed"] for row in owners), rows
        assert runtime.time() == spec.t_end
        assert runtime.macro_step() == max_steps
    return rows


@pytest.mark.parametrize("kind", ("overflow", "exact_counter", "nested_error_fallback", "terminal_error"))
def test_native_region_failure_and_exact_control(kind, isolated_native_cache, native_cxx, kokkos_root,
                                                record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, runtime, world = _compile(lambda: _build(kind), "API v0.2 region " + kind)
    if kind == "overflow":
        ownership = profile_gather_value(world, collective_call(world, lambda: runtime.local_boxes("method")))
        with collective_check(world):
            assert sum(bool(row) for row in ownership) == 1, ownership
            if world is not None and world.size > 1:
                assert any(not row for row in ownership), ownership
        record_property("actual_rank_ownership", str(ownership))
    if kind == "terminal_error":
        before = collective_call(world, lambda: runtime.state_global("method"))
        _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
        after = collective_call(world, lambda: runtime.state_global("method"))
        with collective_check(world):
            assert all(error and "origin=1" in error[1] and "code="+str(_EXACT_CODE) in error[1]
                       for error in errors), errors
            assert runtime.time() == 0 and runtime.macro_step() == 0
            if before is not None:
                np.testing.assert_array_equal(after, before)
        record_property("collective_terminal_causes", str(errors))
    else:
        record_property("native_metrics", str(_check_success(spec, runtime, world)))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)


def test_native_rank_divergent_count_rejects_before_body(monkeypatch, isolated_native_cache,
                                                        native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _native, world = native_profile(dimension=2, scope="region count fault injection")
    if world is None or world.size < 2:
        pytest.skip("rank-divergent count reception requires an explicit MPI2 launch")
    import pops.codegen.program_codegen as codegen
    original = codegen._emit_cpp_program_impl
    def inject(*args, **kwargs):
        source = original(*args, **kwargs)
        literal = "std::int64_t(%d)" % _DIVERGENCE_MARKER
        assert source.count(literal) == 1, "fault injection must address one exact producer"
        return source.replace(literal, "std::int64_t(ctx.prepared_execution_lane().rank() + 1)")
    monkeypatch.setattr(codegen, "_emit_cpp_program_impl", inject)
    spec, artifact, runtime, world = _compile(lambda: _build("divergent_count"), "rank count fault injection")
    record_property("fault_injection", "one native Int64 producer replaced by lane.rank()+1; no public rank primitive")
    record_property("native_metrics", str(_check_success(spec, runtime, world)))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)


def test_native_whole_global_schur_and_monolithic_cg(isolated_native_cache, native_cxx, kokkos_root,
                                                   record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, runtime, world = _compile(build_t04_global, "T04 global compiled CG")
    record_property("native_metrics", str(_check_success(spec, runtime, world, max_steps=2)))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-ra", "--maxfail=1"]))
