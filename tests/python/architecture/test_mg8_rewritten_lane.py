"""The « rewritten » lane of MG8 (OD31, S1-c): the slice against master's tutorial 02.

``scripts/run_mg8_rewritten.py`` compares the case file of the slice, as written and with master's
initial array bound as data, with master's archive. Each rule of the comparison and of the
contract fails here on data written for it: a cell outside the bound, a gap beyond its ulps, a
final state beyond 1e-12, a companion that is not bit for
bit, a scalar or a step count that differs, and a contract that grows against its base (MG6); a
recorded cell that no longer differs is named so that the bound shrinks.
The test reads the source tree only and runs nothing native.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("run_mg8_rewritten",
                                              ROOT / "scripts" / "run_mg8_rewritten.py")
assert SPEC is not None and SPEC.loader is not None
lane = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lane)

CONTRACT = lane.read()


class _Archive(dict):
    @property
    def files(self):
        return list(self)


def _master():
    rng = np.random.default_rng(7)
    initial = rng.random((1, 4, 3)) + 0.05
    final = rng.random((1, 4, 3)) + 0.05
    archive = _Archive(initial=initial, final=final)
    archive.update({name: np.float64(index) for index, name in enumerate(CONTRACT["scalars"])})
    return archive


def _record(master, initial=None, final=None):
    first = master["initial"][0].T.copy() if initial is None else initial
    last = master["final"][0].T.copy() if final is None else final
    record = _Archive(u=np.stack([first, last]), time=np.array([0.0, 0.2]),
                      step=np.array([0, 29]))
    record.update({name: master[name] for name in CONTRACT["scalars"]})
    return record


def _contract(cells=((1, 2),), ulps=1, final_relative=1e-12):
    contract = dict(CONTRACT)
    contract["bound"] = dict(CONTRACT["bound"], cells=[list(cell) for cell in cells], ulps=ulps,
                             final_relative=final_relative)
    return contract


def _nudge(array, cell, ulps):
    value = np.float64(array[cell])
    for _ in range(ulps):
        value = np.nextafter(value, np.inf)
    array[cell] = value


def test_the_contract_of_the_tree_holds():
    assert lane.check(type("A", (), {"contract": str(lane.CONTRACT), "source": str(ROOT),
                                     "check_base": None})()) == 0
    assert CONTRACT["bound"]["issue"] == "G07"
    assert CONTRACT["bound"]["final_relative"] <= 1e-12


def test_the_file_within_its_bound_and_the_companion_bit_for_bit_pass():
    master = _master()
    initial = master["initial"][0].T.copy()
    _nudge(initial, (2, 1), 1)                      # master's cell (1, 2), transposed
    final = master["final"][0].T.copy()
    final[0, 0] *= 1 + 1e-15
    assert lane.compare(_contract(), master, _record(master, initial, final), (29, 0.2),
                        "as-written") == []
    assert lane.compare(_contract(), master, _record(master), (29, 0.2), "master-initial") == []


@pytest.mark.parametrize("change, message", [
    (lambda initial, final: _nudge(initial, (0, 0), 1), "outside the bound"),
    (lambda initial, final: _nudge(initial, (2, 1), 2), "differs by 2 ulp"),
    (lambda initial, final: (_nudge(initial, (2, 1), 1),
                             final.__setitem__((0, 0), final[0, 0] * (1 + 1e-9))),
     "relative, the bound is"),
], ids=["a-cell-outside", "two-ulp", "final-beyond-1e-12"])
def test_the_file_as_written_fails_outside_its_bound(change, message):
    master = _master()
    initial = master["initial"][0].T.copy()
    final = master["final"][0].T.copy()
    change(initial, final)
    problems = lane.compare(_contract(), master, _record(master, initial, final), (29, 0.2),
                            "as-written")
    assert any(message in problem for problem in problems), problems


def test_a_recorded_cell_that_no_longer_differs_is_named_to_shrink_the_bound():
    master = _master()
    initial = master["initial"][0].T.copy()
    _nudge(initial, (2, 1), 1)
    notes: list = []
    assert lane.compare(_contract(cells=((1, 2), (3, 0))), master, _record(master, initial),
                        (29, 0.2), "as-written", notes) == []
    assert notes == ["as-written: initial equals master's in the recorded cells [(3, 0)]: "
                     "shrink the bound"]


def test_the_companion_is_bit_for_bit():
    master = _master()
    initial = master["initial"][0].T.copy()
    _nudge(initial, (2, 1), 1)
    problems = lane.compare(_contract(), master, _record(master, initial), (29, 0.2),
                            "master-initial")
    assert problems == ["master-initial: initial differs from master's in 1 cells (bit for bit)"]


def test_the_scalars_and_the_steps_are_master_s():
    master = _master()
    record = _record(master)
    record["cfl"] = np.float64(0.5)
    problems = lane.compare(_contract(), master, record, (30, 0.2), "master-initial")
    assert "master-initial: the scalar cfl differs from master's" in problems
    assert "master-initial: (steps, final time) (29, 0.2), master (30, 0.2)" in problems


def test_the_companion_replaces_three_statements_of_the_case_file():
    source = (ROOT / CONTRACT["case"]).read_text(encoding="utf-8")
    companion = lane.companion_source(source)
    assert "initial={u: master_initial}," in companion
    assert "pops.project" not in companion.split("program = pops.Program(")[1].split(")")[0]
    with pytest.raises(lane.ContractError, match="holds 0 times"):
        lane.companion_source(source.replace("simulation = pops.bind(artifact)", ""))


def test_the_bound_only_shrinks():
    base = _contract(cells=((1, 2), (3, 0)))
    assert lane.grown(_contract(cells=((1, 2),)), base) == []
    without = dict(base)
    del without["bound"]
    assert lane.grown(without, base) == []
    assert lane.grown(_contract(cells=((1, 2), (0, 0))), base) == [
        "[bound] gains the cells [(0, 0)]"]
    assert lane.grown(_contract(cells=((1, 2),), ulps=2), base) == ["[bound] widens"]
    assert lane.grown(base, without) == ["[bound] is not in the base"]
    with pytest.raises(lane.ContractError, match="final_relative <= 1e-12"):
        lane.load((ROOT / "tests" / "gates" / "mg8_rewritten.toml").read_text("utf-8").replace(
            "final_relative = 1e-12", "final_relative = 1e-10"))
