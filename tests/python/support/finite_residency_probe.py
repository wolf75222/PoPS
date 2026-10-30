"""Run one finite-array Program natively under the Kokkos Tools residency library (DM8).

``python -m tests.python.support.finite_residency_probe KIND REPORT`` builds the Program of KIND,
runs it for STEPS steps and writes REPORT, a JSON object with the counters of its Program DSO. The
test points KOKKOS_TOOLS_LIBS at the library built from ``kokkos_residency_tool.cpp`` and
POPS_KOKKOS_RESIDENCY_OUTPUT at the file where that library writes one line per step.

KIND "device" maps, gathers and folds a replicated Array: every array operation has a device
lowering, so no array byte may cross inside a step. KIND "host" adds a map whose body creates a
domain, an operation without a device lowering: its host evaluation copies the device Array it
reads to the host once and publishes its result, the negative control which shows that the
library sees array data cross. KIND "fold" adds a fold whose carry is an Array, which the host
sequences: each iteration reads one element of its carry, and the Array that the fold reads
without redefining it is copied once.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

import numpy as np

STEPS = 2
DT = 0.01
COUNT = 257
FOLDS = 4
KINDS = ("device", "host", "fold")


def build(kind):
    from examples.api_v02.common import CaseSpec, non_owner_metrics, state_array
    from examples.api_v02.methods_T01_T12 import _initial, _storage
    from pops.time import FixedDt

    if kind not in KINDS:
        raise ValueError("finite residency probe kind must be one of %s" % (KINDS,))
    case, layout, _, block, state, p, q, initial = _storage(
        "finite_residency_" + kind, ("a", "b"), (1., 2.), cells=(8, 4), sampling="cell_average")
    _ = q.n
    domain = p.range_set(COUNT, capacity=2**63 - 1, ownership="replicated")
    big = 2**53 + 3
    computed = p.map(domain, lambda b, i: b.product(
        exact=b.index_value(i) + big,
        value=b.branch(b.index_value(i) == 0, lambda c: c.float64(7),
                       lambda c: c.float64(1) / c.float64(c.index_value(i))),
        even=b.index_value(i) % 2 == 0))
    values = p.map(domain, lambda b, i: computed[b.index(domain, COUNT - 1 - b.index_value(i))])
    if kind == "host":
        gathered = values

        # range_set in a map body has no device lowering: the map runs on the host (+2-2 = 0).
        def shifted(b, i):
            element = gathered[i]
            created = b.size(b.range_set(2, capacity=2, ownership="replicated"))
            return b.product(exact=element["exact"] + created - 2, value=element["value"],
                             even=element["even"])

        values = p.map(domain, shifted)
    elif kind == "fold":
        weights = computed

        # The carry is unchanged: its element 0 is computed[COUNT - 1] and weights[k] is computed[k].
        def iteration(b, old, k):
            first = old[b.index(b.domain(old), 0)]
            weight = weights[b.index(domain, k)]
            shift = first["exact"] - weight["exact"] - (COUNT - 1 - k)
            return b.map(b.domain(old), lambda c, i: c.product(
                exact=old[i]["exact"] + shift, value=old[i]["value"], even=old[i]["even"]))

        values = p.fold(values, FOLDS, iteration)
    total = p.reduce(values, p.product(k=p.int64(0), x=p.float64(0), position=p.int64(0)),
                     lambda b, s, x: b.product(
                         k=s["k"] + (x["exact"] - big) * (s["position"] + 1),
                         x=s["x"] + x["value"] * b.float64(s["position"] + 1),
                         position=s["position"] + 1), order="ascending")
    left = p.value("integer_result", p.float64(total["k"]) * q.n)
    right = p.value("real_result", total["x"] * q.n)
    selected = p.value("map_result", (left[0], right[1]))
    p.commit(q.next, p.value("accepted", selected, at=q.next.point))
    p.step_strategy(FixedDt(DT))
    case.program(p)
    order = list(reversed(range(COUNT)))
    factors = (sum((j + 1) * k for j, k in enumerate(order)),
               sum((j + 1) * (7. if k == 0 else 1. / k) for j, k in enumerate(order)))
    expected = initial.copy()
    for component, factor in enumerate(factors):
        expected[component] *= float(factor) ** STEPS

    def check(runtime):
        actual = state_array(runtime, "method", initial.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual - expected) / np.abs(expected)))
        return dict(passed=error < 2e-12, metrics_owner=True, max_relative_error=error)

    return CaseSpec(case, layout, _initial(case, block[state], initial), {"method": block[state]},
                    STEPS * DT, DT, check, "finite array residency " + kind)


def _counter(dso, name, restype=ctypes.c_uint64):
    function = getattr(dso, name)
    function.argtypes = []
    function.restype = restype
    return function()


def main(argv):
    if len(argv) != 2 or argv[0] not in KINDS:
        raise SystemExit("usage: python -m tests.python.support.finite_residency_probe "
                         "{device,host,fold} REPORT")
    from tests.python.support.api_v02_native import execute_case
    from tests.python.support.public_native_profile import native_profile

    kind, report = argv
    _, world = native_profile(dimension=2, scope="finite residency " + kind)
    if world is not None and int(world.size) != 1:
        raise SystemExit("the finite residency probe runs on one process")
    _, artifact, gathered = execute_case(lambda: build(kind), world, "finite residency " + kind,
                                         max_steps=STEPS)
    dso = ctypes.CDLL(str(Path(artifact.program.so_path).resolve()))
    counters = {name: _counter(dso, name) for name in (
        "pops_finite_kokkos_launch_count", "pops_finite_host_map_count",
        "pops_finite_device_to_host_bytes", "pops_finite_host_to_device_bytes",
        "pops_finite_host_transfer_count")}
    counters["execution_space"] = _counter(dso, "pops_finite_kokkos_execution_space",
                                           ctypes.c_char_p).decode()
    counters["steps"] = STEPS
    (metrics,) = gathered
    counters["metrics"] = {key: value for key, value in metrics.items()
                           if isinstance(value, (bool, int, float, str))}
    Path(report).write_text(json.dumps(counters, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
