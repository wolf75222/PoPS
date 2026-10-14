"""A complete typed error leaves generated C++ through the real PoPS transaction."""
import numpy as np

from pops.time import FixedDt
from pops.time.variant_types import Failure, Float64, Int64, Product, Result, type_of
from examples.api_v02.common import CaseSpec
from examples.api_v02.methods_T01_T12 import _initial, _storage

EXACT_CODE = (1 << 60) + 17


def build_terminal_failure(kind):
    if kind not in ("scalar", "field"):
        raise ValueError(kind)
    case, layout, model, block, state, p, q, initial = _storage(
        "terminal_failure_" + kind, ("a", "b"), (1., 2.), cells=(8, 4), sampling="cell_average")
    initial += np.arange(32, dtype=float).reshape(1, 4, 8) / 128
    fields = {"code": Int64, "defect": Float64}
    values = {"code": p.int64(EXACT_CODE), "defect": p.float64(.25)}
    if kind == "field":
        fields["checkpoint"] = type_of(q.n)
        values["checkpoint"] = q.n
    errors = Failure(retry=Product(**fields))
    outcome = Result(type_of(q.n), errors)
    rejected = p.variant(outcome, "err", p.variant(errors, "retry", p.product(**values)))
    selected = p.require_success(rejected)
    p.commit(q.next, p.value("must_not_publish", selected, at=q.next.point))
    p.step_strategy(FixedDt(.01))
    case.program(p)
    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    .01, .01, lambda runtime: {"passed": False}, "native terminal typed failure " + kind)
