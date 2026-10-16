"""Nested Array/Variant/Product values authored in the public algebra."""
import numpy as np
import pops
from pops.time.expressions import ProgramScalar
from pops.time import FixedDt
from pops.time.variant_types import Int64, Product, Variant, type_of
from .common import CaseSpec, non_owner_metrics
from .methods_T01_T12 import _initial, _storage


def build_finite_structural_depth(*, depth=160):
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    case, layout, model, block, state, p, q, initial = _storage(
        "finite_structural_depth", ("decoded",), (0.,), cells=(4, 4))
    domain = p.range_set(1, capacity=1, ownership="replicated")
    current = p.int64(7)
    sources = []
    for _ in range(depth):
        old = current
        shape = Variant(active=Product(value=type_of(old)), inactive=Int64)
        current = p.map(domain, lambda b, i, old=old, shape=shape:
            b.variant(shape, "active", b.product(value=old)))
        sources.append(old)
    for old in reversed(sources):
        selected = current[p.index(domain, 0)]
        current = p.match(selected, active=lambda b, payload: payload["value"],
                          inactive=lambda b, unused, old=old: old)
    p.record_scalar("structural.decoded", p.float64(current))
    p.commit(q.next, p.value("decoded", (q.n[0]+ProgramScalar(p.float64(current)),), at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    def check(runtime):
        value = runtime.state_global("method")
        if value is None:
            return non_owner_metrics()
        error = float(np.max(abs(np.asarray(value)-7*runtime.macro_step())))
        return {"passed": error == 0., "metrics_owner": True, "error": error}
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    .01, .01, check, "nested host storage with inactive Variant cells left empty")
