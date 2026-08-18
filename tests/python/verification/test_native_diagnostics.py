"""Public pops.diagnostics attached to ConsumerGraph (plan §2.3 / §7.0)."""
from __future__ import annotations

from pathlib import Path

import pytest

pops = pytest.importorskip("pops")

from pops.diagnostics import (  # noqa: E402
    Balance,
    BalanceLedger,
    ConservationCheck,
    Integral,
    MinMax,
    Norm,
    StepChangeNorm,
)
from pops.linalg.norms import L1, L2, LInf  # noqa: E402
from pops.model import Module  # noqa: E402
from pops.output import Checkpoint, ConsumerGraph  # noqa: E402
from pops.problem import Case  # noqa: E402
from pops.time import Clock, every, on_end  # noqa: E402

from verification.pops_verify.native_diagnostics import (  # noqa: E402
    attach_state_diagnostics,
    state_diagnostics,
)

_CASE = Case(name="verification-native-diagnostics")
_BLOCK = _CASE.block("state", Module("verification-model"))
_CADENCE = every(10, clock=Clock("verification", owner=_CASE.owner_path))


def test_state_diagnostics_returns_six_public_types_in_order():
    rows = state_diagnostics(_BLOCK, _CADENCE)
    assert tuple(type(item) for item in rows) == (
        Integral,
        Norm,
        Norm,
        Norm,
        MinMax,
        StepChangeNorm,
    )
    assert rows[0].block is _BLOCK and rows[0].cadence is _CADENCE
    assert type(rows[1].norm) is L1
    assert type(rows[2].norm) is L2
    assert type(rows[3].norm) is LInf
    assert rows[4].block is _BLOCK and rows[4].cadence is _CADENCE
    assert type(rows[5].norm) is L2
    assert all(item.block is _BLOCK and item.cadence is _CADENCE for item in rows)


def test_attach_state_diagnostics_returns_consumer_graph():
    graph = attach_state_diagnostics(block=_BLOCK, cadence=_CADENCE)
    assert type(graph) is ConsumerGraph
    assert graph.is_resolved is False
    (node,) = graph._authoring
    assert node.label == "console-monitor"
    assert tuple(type(item) for item in node.diagnostics) == (
        Integral,
        Norm,
        Norm,
        Norm,
        MinMax,
        StepChangeNorm,
    )


def test_non_handle_block_raises_type_error():
    with pytest.raises(TypeError, match="BlockHandle"):
        state_diagnostics("state", _CADENCE)
    with pytest.raises(TypeError, match="BlockHandle"):
        attach_state_diagnostics(block="state", cadence=_CADENCE)


def test_helper_does_not_import_reference_errors_or_define_norm_formulas():
    import verification.pops_verify.native_diagnostics as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "reference_errors" not in source
    assert "verification.pops_verify.reference_errors" not in module.__dict__
    assert "numpy" not in source
    assert "total_volume" not in source
    assert "np.sqrt" not in source
    assert "np.sum" not in source


def test_state_diagnostics_appends_conservation_check_and_balance_when_given():
    ledger = BalanceLedger("mass")
    quantity = Integral(block=_BLOCK, cadence=_CADENCE)
    check = ConservationCheck(quantity)
    rows = state_diagnostics(
        _BLOCK, _CADENCE, conservation_check=check, ledger=ledger
    )
    assert len(rows) == 8
    assert rows[6] is check
    assert type(rows[7]) is Balance
    assert rows[7].ledger is ledger
    assert rows[7].block is _BLOCK
    assert rows[7].cadence is _CADENCE


def test_invalid_conservation_check_raises_type_error():
    with pytest.raises(TypeError, match="ConservationCheck"):
        state_diagnostics(_BLOCK, _CADENCE, conservation_check="mass")
    with pytest.raises(TypeError, match="ConservationCheck"):
        attach_state_diagnostics(
            block=_BLOCK, cadence=_CADENCE, conservation_check="mass"
        )


def test_invalid_ledger_raises_type_error():
    with pytest.raises(TypeError, match="BalanceLedger"):
        state_diagnostics(_BLOCK, _CADENCE, ledger="mass")
    with pytest.raises(TypeError, match="BalanceLedger"):
        attach_state_diagnostics(block=_BLOCK, cadence=_CADENCE, ledger="mass")


def test_attach_state_diagnostics_appends_extra_consumers_after_monitor():
    extra = Checkpoint(
        schedule=on_end(clock=Clock("end", owner=_CASE.owner_path)),
        target="checkpoints/restart",
        bit_identical=True,
    )
    graph = attach_state_diagnostics(
        block=_BLOCK, cadence=_CADENCE, consumers=(extra,)
    )
    assert type(graph) is ConsumerGraph
    labels = [node.label for node in graph._authoring]
    assert labels == ["console-monitor", "checkpoint-checkpoints-restart"]
