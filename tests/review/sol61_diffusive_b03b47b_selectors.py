"""Independent selector/producer agreement probes against an exact b03b47b snapshot.

Run from repository root after extracting its python and public fixture support.
This probes metadata semantics and emitted producers; it executes no native solver.
"""
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "outputs/diffusive-sol61-b03b47b-source"
sys.path[:0] = [str(SNAPSHOT / "python"), str(SNAPSHOT)]
import pops
from pops.codegen.program_emit_diffusion import (
    _emit_diffusive_accepted, _resolved_diffusive_operation, _resolved_diffusive_trace_selection,
)
from tests.python.support.integral_diffusion_case import build_diffusion_integral_case

assert Path(pops.__file__).resolve() == SNAPSHOT / "python/pops/__init__.py"
case, _, _, _, _ = build_diffusion_integral_case()
real = next(value for value in case._time._values if value.op == "diffusive_rhs")
balance = real.attrs["physical_balance"].balance

def row(kind, ordinal):
    return SimpleNamespace(kind=kind, ordinal=ordinal, identity=(kind, ordinal), coefficient=1.)

def evaluation(rows, *, fitted=False):
    return SimpleNamespace(block=real.block, attrs={**real.attrs, "fitted": fitted,
        "physical_balance": SimpleNamespace(balance=balance, occurrences=rows)})

for rows, fitted, suffix in (
    ((row("source", 7), row("diffusion", 3)), False, "/occurrence:3"),
    ((row("drift", 2), row("source", 7), row("diffusion", 9)), True,
     "/joint-occurrences:2,9"),
):
    value = evaluation(rows, fitted=fitted)
    operation = _resolved_diffusive_operation(value)
    assert _resolved_diffusive_trace_selection(value) == (operation, operation + suffix)
    lines = []
    _emit_diffusive_accepted(value, "carrier", lines, "dt", "independent-stage")
    assert len(lines) == 1
    literals = re.findall(r'"([^"\\]*)"', lines[0])
    assert literals == [operation, operation + suffix, "independent-stage"]

for rows in (
    (row("diffusion", 3), row("diffusion", 11)),
    (row("diffusion", 3), row("flux", 8)),
    (row("source", 7),),
):
    try:
        _resolved_diffusive_trace_selection(evaluation(rows))
    except ValueError as error:
        assert "one exact conservative diffusive occurrence" in str(error)
    else:
        raise AssertionError("an ambiguous or source-only carrier was admitted")
print("b03b47b independent selector/producer probes: 5 passed (sources excluded; fitted joint exact)")
