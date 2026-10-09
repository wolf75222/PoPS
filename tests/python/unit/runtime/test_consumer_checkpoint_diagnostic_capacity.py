"""Finite Native diagnostic image bounds from declared consumer reductions; Source only."""
from dataclasses import replace
import json
import struct
from types import SimpleNamespace

import pytest

from pops.identity import make_identity
from pops.model import Handle, OwnerPath
from pops.output._consumer_contracts import DiagnosticQuantity
from pops.output.data import DiagnosticKey, DiagnosticPayload
from pops.runtime import _checkpoint_resource_budget as budget
from pops.runtime import _checkpoint_program_diagnostics as codec
from pops.runtime._runtime_consumers import (
    _diagnostic_record_name, _potential_diagnostic_record_names,
)


def _quantity(*reductions, conservation=False, levels=(0, 1), label="measure_λ"):
    operations = [{"name": reduction, "reduction": reduction, "transform": "identity",
                   "metric_weighted": False, "coefficient": (1.0).hex()}
                  for reduction in reductions]
    if reductions == ("accepted_balance",):
        operations[0]["balance_route"] = make_identity("balance-ledger-route", {}).token
    return DiagnosticQuantity(
        Handle(label, kind="diagnostic", owner=OwnerPath.consumer("declared-output")),
        Handle("U", kind="state", owner=OwnerPath.model("state-owner")),
        "state:U", "layout:mesh", levels,
        {"schema_version": 2, "role": None, "operations": operations,
         "conservation": {"tolerance": (1e-9).hex()} if conservation else None},
    )


def _artifact(quantities, source='ctx.record_scalar("program", 0);'):
    # Compiler/source authority substitute only, not a Native artifact/result.
    graph = SimpleNamespace(nodes=tuple(SimpleNamespace(diagnostic_quantities=tuple(rows))
                                        for rows in quantities))
    return SimpleNamespace(verify=lambda: None, program=SimpleNamespace(_generated_cpp=source),
                           plan=SimpleNamespace(consumer_graph=graph))


@pytest.mark.parametrize("reductions,conservation,expected", [
    (("min", "max", "sum", "abs_sum", "abs_max", "sum_sq"), False,
     ("min", "max", "sum", "abs_sum", "abs_max", "sum_sq")),
    (("step_change_l2",), False, ("step_change_l2",)),
    (("sum",), True, ("conservation:sum",)),
    (("accepted_balance",), False, ("discrete_balance",)),
])
def test_potential_names_are_exactly_the_actual_diagnostic_sink(reductions, conservation, expected):
    quantity = _quantity(*reductions, conservation=conservation)
    actual = []
    for reduction in expected:
        payload = DiagnosticPayload(DiagnosticKey(
            quantity.handle, make_identity("component-manifest", {}),
            make_identity("layout", {}), min(quantity.levels), quantity.identity.token, reduction),
            1.0, "kg", {})
        actual.append(_diagnostic_record_name(payload))
    assert _potential_diagnostic_record_names(quantity) == tuple(actual)
    assert len(actual) == len(expected)  # Levels are aggregated, never a per-cell multiplier.


def test_program_and_all_consumer_names_form_one_utf8_deduplicated_union():
    first = _quantity("sum", label="long_μ_" + "λ" * 128)
    other_levels = replace(first, levels=(1, 2, 3))
    conserved = _quantity("sum", conservation=True)
    balance = _quantity("accepted_balance")
    artifact = _artifact(((first, conserved), (first, other_levels), (balance,)))
    consumer_names = {name for q in (first, other_levels, conserved, balance)
                      for name in _potential_diagnostic_record_names(q)}
    names = budget._compiled_diagnostic_inventory(artifact)
    assert set(names) == {"program", "pops.frontier.duration"} | consumer_names
    assert len(names) == 6
    assert budget._diagnostic_inventory_capacity(names) == 40 + sum(
        8 + len(name.encode("utf-8")) + 8 for name in names)
    # An exact scalar record name can be shared by Program and ConsumerGraph.
    shared = next(iter(consumer_names))
    artifact.program._generated_cpp = 'ctx.record_scalar(' + json.dumps(shared) + ', 0);'
    assert budget._compiled_diagnostic_inventory(artifact).count(shared) == 1


@pytest.mark.parametrize("source", [None, "ctx.record_scalar(dynamic_name, 0);"])
def test_known_consumer_names_do_not_hide_unknown_program_inventory(source):
    artifact = _artifact(((_quantity("sum"),),), source=source)
    assert budget._compiled_diagnostic_inventory(artifact) is None
    assert budget._diagnostic_inventory_capacity(None) is None


@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_declared_image_fits_per_rank_and_larger_record_is_refused(ranks):
    names = budget._compiled_diagnostic_inventory(_artifact(((_quantity("sum"),),)))
    capacity = budget._diagnostic_inventory_capacity(names)
    def image(rank, records):
        return b"POPSDIA1" + struct.pack("<QQQQ", 64, rank, ranks, len(records)) + b"".join(
            struct.pack("<Q", len(name.encode("utf-8"))) + name.encode("utf-8")
            + struct.pack("<d", -0.0) for name in records)
    import numpy as np
    images = tuple(image(rank, names) for rank in range(ranks))
    assert all(len(row) == capacity for row in images)
    payload = {"program_diagnostics_state": np.frombuffer(b"".join(images), dtype=np.uint8).copy(),
               "program_diagnostics_offsets": np.asarray([capacity*i for i in range(ranks+1)],
                                                          dtype=np.int64)}
    assert codec.validate_checkpoint_program_diagnostic_arrays(
        payload, capacity=capacity, bound=capacity*ranks)
    with pytest.raises(ValueError, match="chosen resource capacity"):
        codec.validate_checkpoint_program_diagnostic_arrays(
            payload, capacity=capacity-1, bound=capacity*ranks)
    # An explicit old78-byte choice remains strict; default inference must not raise it.
    with pytest.raises(ValueError, match="chosen resource capacity"):
        codec.validate_checkpoint_program_diagnostic_arrays(payload, capacity=78,
                                                            bound=78*ranks)
    expanded = image(0, names + ("external-unplanned",))
    with pytest.raises(ValueError, match="chosen resource capacity"):
        codec.validate_checkpoint_program_diagnostic_arrays(
            {"program_diagnostics_state": np.frombuffer(expanded, dtype=np.uint8).copy(),
             "program_diagnostics_offsets": np.asarray([0, len(expanded)], dtype=np.int64)},
            capacity=capacity, bound=capacity)
