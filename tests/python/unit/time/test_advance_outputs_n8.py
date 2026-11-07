"""N8 lot 3: the Outputs of an elaborated Advance, source side (spec 0.3 5.2, CC14, T33).

``Output(value, when=None, at=None, write=None)`` is traced once into a read-only region of the
builder. Its record (the accepted state, its cell reductions and scalar arithmetic on them) and
its schedule enter the Program IR (version 31) and its semantic identity; the compiled control
lists the schedule and the record shape, which the artifact manifest publishes; the generated
C++ exports one entry that evaluates the scalars of each Output. A Program without Outputs keeps
the IR, the control record and the C++ of lot 2. What this lot does not lower is refused by name.
``tests/python/integration/runtime/test_advance_outputs_n8_native.py`` runs it.
"""
import math

import pytest

import pops
import examples.api_v02.models_M01_M14 as early
from pops._ir.symbolic import SymbolicTruthValueError
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.identity.semantic import program_semantic_data
from pops.runtime._simulation import RunResult, _ahead
from pops.time._program.control import (
    ADVANCE_CONTROL_OUTPUTS_SCHEMA, ADVANCE_CONTROL_SCHEMA, AdvanceControl,
)
from pops.time._program.elaboration import Advance, Proposal, elaborate
from pops.time._program.outputs import (
    Output, Product, cell_max, cell_min, cell_sum, every, every_dt,
)
from pops.time._program.traced_method import ClosedBodyViolation, NotYetElaborated
from pops.time._schedule.api import _every_dt_lattice_time

ENTRIES = ('extern "C" int pops_program_advance_output_count()',
           'extern "C" int pops_program_advance_output_scalar_count(int output)',
           'extern "C" void pops_program_advance_output_scalars(')


def _snapshot(B, interval):
    return Product(u=B.state, mass=cell_sum(B.state), spread=cell_max(B.state) - cell_min(B.state))


def _mass(B, interval):
    return Product(mass=cell_sum(B.state))


def _elaborated(monkeypatch, outputs):
    elaborations = []

    def route(case, block, state, flow, dt, *, name):
        elaborations.append(elaborate(Advance(early.shu_osher_ssprk2(flow=flow)), case=case,
                                      state=block[state], proposal=Proposal(dt), name=name,
                                      outputs=outputs))
    with monkeypatch.context() as patch:
        patch.setattr(early, "_traced_ssprk2", route)
        spec = early.build_m01()
    (elaboration,) = elaborations
    return spec, elaboration


def _source(spec):
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    return emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(
        resolved.blocks))


def _refused(monkeypatch, value, error, match, **schedule):
    with pytest.raises(error, match=match):
        _elaborated(monkeypatch, (Output(value, **schedule),))


# --- the Output values -------------------------------------------------------------------------

def test_an_output_holds_its_body_and_its_schedule():
    output = Output(_mass, at=(0.5, 0.25, 0.5))
    assert output.at == (0.25, 0.5)  # sorted, and a time written twice is one time
    assert Output(_mass, when=every(3)).when == every(3) and every(3).steps == 3
    assert every_dt(0.1) == every_dt(0.1) and every_dt(0.1).dt == 0.1
    with pytest.raises(AttributeError):
        output.at = None
    for when in (lambda B, interval: True, 3, "every(3)"):
        with pytest.raises(NotYetElaborated, match="every"):
            Output(_mass, when=when)
    with pytest.raises(NotYetElaborated, match="writers of pops.lib.output"):
        Output(_mass, write=print)
    with pytest.raises(TypeError, match="tuple of physical times"):
        Output(_mass, at=0.5)
    with pytest.raises(ValueError, match="finite"):
        Output(_mass, at=(math.inf,))
    with pytest.raises(ValueError, match="at=None"):
        Output(_mass, at=())
    with pytest.raises(ValueError, match="k >= 1"):
        every(0)
    with pytest.raises(ValueError, match="positive"):
        every_dt(0.0)
    with pytest.raises(TypeError, match="closed body"):
        Output(None)
    with pytest.raises(ValueError, match="identifiers"):
        Product(**{"not a name": 1})


def test_every_dt_lands_on_the_lattice_of_b6():
    lattice = every_dt(0.1)
    # The decimal spelling of dt, as b6's EveryDt: 3 * 0.1 is one unit above 0.3, the lattice is 0.3.
    assert 3 * 0.1 != 0.3 and lattice.time(3) == 0.3
    assert all(lattice.time(k) == _every_dt_lattice_time(k, 0.1) for k in range(1, 200))
    assert lattice.times(0.0, 0.35) == (0.1, 0.2, 0.3)
    assert lattice.times(0.1, 0.3) == (0.2, 0.3)  # a time the run starts on is behind it
    assert lattice.times(0.15, 0.15) == ()
    with pytest.raises(ValueError, match="below the binary64 resolution"):
        every_dt(1e-300).times(1.0, 2.0)


def test_a_time_within_four_units_below_until_is_until():
    below = math.nextafter(math.nextafter(0.1, 0.0), 0.0)
    assert _ahead((below, 0.1), 0.0, 0.1) == [0.1]
    assert _ahead((0.0, 0.05, 0.1, 0.2), 0.0, 0.1) == [0.05, 0.1]
    assert _ahead((0.05,), 0.05, 0.1) == []
    far = 0.1 - 8 * math.ulp(0.1)
    assert _ahead((far,), 0.0, 0.1) == [far]


# --- elaboration -------------------------------------------------------------------------------

def test_outputs_enter_the_ir_the_identity_and_the_control(monkeypatch):
    outputs = (Output(_snapshot, at=(0.0123, 0.05)), Output(_mass, when=every(4)),
               Output(_mass, at=every_dt(0.01)))
    spec, elaboration = _elaborated(monkeypatch, outputs)
    program = elaboration.program
    assert [region.output for region in program._advance_outputs] == list(outputs)
    data = program._serialize(include_provenance=False)
    assert data["version"] == 31
    records = [row["record"] for row in data["advance_outputs"]]
    assert records[0][0] == {"name": "u", "array": "fluid"}
    assert [row["name"] for row in records[0]] == ["u", "mass", "spread"]
    assert {row["op"] for row in data["advance_outputs"][0]["nodes"]} == {"reduce", "scalar_op"}
    control = elaboration.control.to_data()
    assert control["schema"] == ADVANCE_CONTROL_OUTPUTS_SCHEMA
    assert control["outputs"] == [
        {"when": None, "at": {"times": [0.0123.hex(), 0.05.hex()]},
         "record": [{"name": "u", "array": "fluid"}, {"name": "mass", "scalar": 0},
                    {"name": "spread", "scalar": 1}]},
        {"when": {"every": 4}, "at": None, "record": [{"name": "mass", "scalar": 0}]},
        {"when": None, "at": {"every_dt": 0.01.hex()}, "record": [{"name": "mass", "scalar": 0}]},
    ]
    assert AdvanceControl.from_data(control) == elaboration.control
    assert "advance_outputs" in program_semantic_data(program)
    # Another record, or another schedule, is another Program.
    _, other = _elaborated(monkeypatch, (Output(_snapshot, at=(0.0123, 0.05)),
                                         Output(_mass, when=every(5)),
                                         Output(_mass, at=every_dt(0.01))))
    assert program_semantic_data(other.program) != program_semantic_data(program)
    source = _source(spec)
    for entry in ENTRIES:
        assert source.count(entry) == 1
    assert "extern \"C\" int pops_program_advance_output_count() { return 3; }" in source
    assert "ctx.sum_component(0, " in source and "ctx.max_component(0, " in source


def test_a_program_without_outputs_keeps_lot_2(monkeypatch):
    spec, elaboration = _elaborated(monkeypatch, ())
    data = elaboration.program._serialize(include_provenance=False)
    assert data["version"] == 30 and "advance_outputs" not in data
    assert elaboration.control.to_data()["schema"] == ADVANCE_CONTROL_SCHEMA
    assert "outputs" not in elaboration.control.to_data()
    source = _source(spec)
    assert not any(entry in source for entry in ENTRIES)


def test_a_record_of_arrays_only_exports_no_scalar(monkeypatch):
    spec, _ = _elaborated(monkeypatch, (Output(lambda B, interval: Product(u=B.state)),))
    source = _source(spec)
    assert "    case 0: return 0;\n" in source


# --- refusals (T33 and the later lots) ---------------------------------------------------------

def test_an_output_body_is_closed(monkeypatch):
    kept = []

    def leaks(B, interval):
        kept.append(B.state)
        return Product(mass=cell_sum(B.state))
    _elaborated(monkeypatch, (Output(leaks),))
    with pytest.raises(ClosedBodyViolation, match="outside the Output body"):
        cell_sum(kept[0])
    _refused(monkeypatch, lambda B, interval: Product(n=1.0), TypeError, "constant")
    _refused(monkeypatch, lambda B, interval: Product(x="u"), ClosedBodyViolation,
             "neither the accepted state")
    _refused(monkeypatch, lambda B, interval: {"mass": cell_sum(B.state)}, TypeError,
             "returns a Product")
    _refused(monkeypatch, lambda B, interval: Product(), ValueError, "at least one value")
    _refused(monkeypatch, lambda B, interval: Product(u=B.state * 2.0), NotYetElaborated,
             "definition of the model")
    _refused(monkeypatch, lambda B, interval: Product(t=B.time), NotYetElaborated,
             "achieved time")
    _refused(monkeypatch, lambda B, interval: Product(d=interval.increment), NotYetElaborated,
             "step_change")
    _refused(monkeypatch, lambda B, interval: Product(m=cell_sum(B.state, component=-1)),
             ValueError, "component")

    def branches(B, interval):
        if B.state:
            return Product(u=B.state)
        return Product(m=cell_sum(B.state))
    _refused(monkeypatch, branches, SymbolicTruthValueError, "undecided at elaboration")


def test_the_simulation_result_keeps_its_outputs():
    result = RunResult(state=None, time=0.0, steps=0, rejections=0, diagnostics={},
                       continuation=None, failure=None, outputs={"k": ()})
    assert dict(result.outputs) == {"k": ()}
    with pytest.raises(TypeError):
        result.outputs["j"] = ()
