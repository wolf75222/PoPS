"""Finite native reception programs for user-defined sums; no Python runtime evaluator."""
import numpy as np

from pops.time import FixedDt
from pops.time.variant_types import Failure, Float64, Int64, Product, Result, Variant, type_of
from examples.api_v02.common import CaseSpec, non_owner_metrics, state_array
from examples.api_v02.methods_T01_T12 import _initial, _storage


EXACT_CODE = (1 << 60) + 17
DIVERGENCE_MARKER = 970039


def build_variant_case(kind, *, ranks=1):
    case, layout, model, block, state, p, q, initial = _storage(
        "variant_" + kind, ("a", "b"), (1., 2.), cells=(8, 4), sampling="cell_average")
    q.n
    expected = initial.copy()
    if kind == "array_variants":
        domain = p.range_set(3, capacity=3, ownership="replicated")
        Choice = Variant(number=Int64, field=type_of(q.n))
        def element(t, i):
            k = t.index_value(i)
            return t.branch(k == 1,
                lambda r: r.variant(Choice, "number", r.int64(100)),
                lambda r: r.variant(Choice, "field", r.value("snapshot", r.float64(k+1)*q.n)))
        values = p.map(domain, element)
        scanned = p.scan(domain, p.int64(0), lambda t,s,i:
                         t.product(state=s+t.index_value(i), value=values[i]), order="ascending")
        zero = p.value("zero", 0*q.n)
        total = p.reduce(scanned["values"], zero, lambda t,s,item:
                         t.match(item, field=lambda r,x:r.value("sum_field",s+x),
                                 number=lambda r,x:r.value("sum_number",s+r.float64(x)*q.n)),
                         order="ascending")
        Errors = Failure(invalid=Product(index=Int64, defect=Float64))
        Outcome = Result(type_of(q.n), Errors)
        def checked(t, i):
            k = t.index_value(i)
            attempted = t.branch(k == 1,
                lambda r:r.variant(Outcome, "err", r.variant(Errors,"invalid",r.product(index=k,defect=r.float64(.25)))),
                lambda r:r.variant(Outcome, "ok", r.value("partial",r.float64(k+1)*q.n)))
            return t.require_success(attempted)
        partial = p.capture(lambda t:t.map(domain, checked))
        recovery = p.match(partial, ok=lambda t,x:t.value("partial_array_leaked",0*q.n),
            err=lambda t,e:t.match(e,
                invalid=lambda r,data:r.branch(data["index"] == 1,
                    lambda s:s.branch(data["defect"] == .25,
                        lambda a:a.value("recovered_array",7*q.n),lambda a:a.value("wrong_defect",0*q.n)),
                    lambda s:s.value("wrong_index",0*q.n)),
                runtime=lambda r,data:r.value("unexpected_runtime",0*q.n)))
        selected = p.value("sum_and_recovery",total+recovery)
        # Field element 0 must retain q.n even though the map producer's scratch later stores 3*q.n.
        expected *= 111
    elif kind == "divergent_choice":
        Choice = Variant(advance=Int64, retry=Int64)
        def attempt(t):
            choice = t.variant(Choice, "advance", t.int64(DIVERGENCE_MARKER))
            return t.match(choice, advance=lambda r, x: r.value("advanced", 2*q.n),
                           retry=lambda r, x: r.value("retried", 3*q.n))
        captured = p.capture(attempt)
        selected = p.match(captured, ok=lambda t, x: x,
                           err=lambda t, e: t.branch(e["code"] == 3,
                               lambda r: r.branch(e["origin"] == 2,
                                   lambda s: s.value("uniform_recovery", 7*q.n),
                                   lambda s: s.value("wrong_origin", 0*q.n)),
                               lambda r: r.value("wrong_code", 0*q.n)))
        expected *= 7 if ranks > 1 else 2
    elif kind == "inactive_singular":
        Choice = Variant(advance=Float64, retry=Product(code=Int64), stop=Product())
        chosen = p.variant(Choice, "advance", p.float64(2))
        selected = p.match(chosen,
                           advance=lambda t, x: t.value("scaled", x*q.n),
                           retry=lambda t, x: t.value("singular", (t.float64(1)/t.float64(0))*q.n),
                           stop=lambda t, x: t.value("stopped", 0*q.n))
        expected *= 2
    elif kind == "full_payload":
        Errors = Failure(retry=Product(code=Int64, defect=Float64, checkpoint=type_of(q.n)))
        Outcome = Result(type_of(q.n), Errors)
        def fail(t):
            data = t.product(code=t.int64(EXACT_CODE), defect=t.float64(.25), checkpoint=q.n)
            return t.require_success(t.variant(Outcome, "err", t.variant(Errors, "retry", data)))
        captured = p.capture(lambda t: t.require_success(t.capture(fail)))
        def recover(t, failure):
            def retry(r, data):
                return r.branch(data["code"] == EXACT_CODE,
                                lambda s: s.branch(data["defect"] == .25,
                                                   lambda a: a.value("recovered", (data["checkpoint"][0]+7, data["checkpoint"][1]+7)),
                                                   lambda a: a.value("bad_defect", 0*q.n)),
                                lambda s: s.value("bad_code", 0*q.n))
            return t.match(failure, retry=retry, runtime=lambda r, data: r.value("unexpected_runtime", 0*q.n))
        selected = p.match(captured, ok=lambda t, x: t.value("unexpected_success", 0*x), err=recover)
        expected += 7
    elif kind == "sum_field_loop":
        Choice = Variant(old=type_of(q.n), new=Product(field=type_of(q.n), count=Int64))
        value = p.variant(Choice, "old", q.n)
        def body(t, carried, index):
            return t.match(carried,
                           old=lambda r, x: r.variant(Choice, "new", r.product(field=r.value("doubled", 2*x), count=index)),
                           new=lambda r, x: r.variant(Choice, "old", x["field"]))
        result = p.fold(value, 3, body)
        selected = p.match(result, old=lambda t, x: x, new=lambda t, x: x["field"])
        expected *= 4
    elif kind == "iteration_payload":
        initial_carry = p.product(field=q.n, count=p.int64(0))
        Errors = Failure(retry=Product(attempt=Int64, defect=Float64, checkpoint=type_of(q.n)))
        Step = Result(type_of(initial_carry), Errors)
        def step(t, x):
            return t.branch(x["count"] == 0,
                lambda r: r.variant(Step, "ok", r.product(field=r.value("advanced", (x["field"][0]+17, x["field"][1]+17)), count=x["count"]+1)),
                lambda r: r.variant(Step, "err", r.variant(Errors, "retry", r.product(attempt=x["count"], defect=r.float64(.75), checkpoint=x["field"]))))
        result = p.iterate(initial_carry, step, lambda t, x: x["count"] >= 3, max_steps=4)
        def recover(t, errors):
            return t.match(errors,
                retry=lambda r, e: r.branch(e["attempt"] == 1,
                    lambda s: s.branch(e["defect"] == .75,
                        lambda a: a.value("restored", (e["checkpoint"][0]-10, e["checkpoint"][1]-10)),
                        lambda a: a.value("wrong_defect", 0*q.n)),
                    lambda s: s.value("wrong_attempt", 0*q.n)),
                runtime=lambda r, e: r.value("unexpected_runtime", 0*q.n))
        selected = p.match(result, ok=lambda t, x: t.value("unexpected_success", 0*x["state"]["field"]), err=recover)
        expected += 7
    else:
        raise ValueError(kind)
    p.commit(q.next, p.value("accepted", selected, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        return {"passed": bool(np.array_equal(actual, expected)), "max_error": error, "metrics_owner": True}
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    .01, .01, check, "native arbitrary Variant: " + kind)
