"""Final metadata-only report for the installed ``Program`` subsystem.

The public runtime is ``RuntimeInstance`` and delegates ``program_report()`` to its authenticated
executor.  These unit checks exercise the single report owner directly: no legacy ``System`` is
constructed and no native state array is read.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pops.identity import make_identity
from pops.runtime._multi_layout_executor import _MultiLayoutUniformExecutor
from pops.runtime.program_report import ProgramRuntimeReport, build_program_report


class _Transaction:
    def to_data(self):
        return {"strategy": {"kind": "fixed"}, "rollback": "snapshot"}


class _Temporal:
    def to_data(self):
        return {"schema_version": 1, "accepted_step": 4}


class _AcceptedProgramAuthority:
    _step_transaction_plan = _Transaction()
    _temporal_restart_state = _Temporal()

    def installed_program_hash(self):
        return "sha256:accepted-program"

    def program_block_map(self):
        return ["fluid"]

    def program_params(self, block):
        assert block == 0
        return SimpleNamespace(count=2)

    def program_diagnostics(self):
        return {"mass": 3.5}

    def history_names(self):
        return ["u_prev"]

    def history_depth(self, name):
        assert name == "u_prev"
        return 2

    def history_ncomp(self, name):
        assert name == "u_prev"
        return 3

    def history_initialized(self, name):
        assert name == "u_prev"
        return True

    def history_fill_count(self, name):
        assert name == "u_prev"
        return 2

    def history_slot_dt(self, name, slot):
        assert name == "u_prev"
        return (0.125, 0.0625)[slot]

    def program_cache_slots(self):
        return [0, 1]

    def program_cache_valid(self, slot):
        return slot == 0

    def program_cache_cold(self, slot):
        return slot == 1

    def program_cache_name(self, slot):
        return ("stage_rhs", "lazy_flux")[slot]

    def program_cache_last_update_step(self, slot):
        return (4, -1)[slot]

    def program_cache_accumulated_dt(self, slot):
        return (0.125, 0.25)[slot]

    def is_profiling(self):
        return False

    def program_clock_manifest(self):
        return [("logical", "main", 4), ("level", 1, 4, 1, 2, 0.5)]

    def checkpoint_temporal_relations(self):
        return [(0, 1, 1, 2, "exact")]

    def program_flux_ledger_manifest(self):
        return [("fluid", "U", "rhs", "transport", 1, 4, 1, 2, 1, 2, "outward", 0.25, 0.125)]

    def program_temporal_partition_manifest(self):
        return [
            ("summary", "cell_local", "test.partition@1", 7, 16, 32, 5),
            ("rung", 0, 3),
            ("rung", 1, 2),
        ]

    def program_sync_manifest(self):
        return [(0, 1, 0, "reflux", 4, 1, 2)]


class _EmptyProgramAuthority:
    pass


def _layout_report(layout_id, marker):
    return ProgramRuntimeReport(
        installed=True,
        program_hash="sha256:%s" % marker,
        step_transaction={"strategy": {"kind": "fixed"}},
        block_map=[0],
        params=[{"program_block": 0, "count": 0, "limit": 8}],
        diagnostics={"marker": marker},
        histories=[],
        cache=[],
        profiler={"enabled": False},
        clocks=[],
        level_relations=[],
        flux_ledger=[],
        synchronization=[],
        temporal_partition={"kind": "global"},
        temporal={"schema_version": 1, "accepted_step": 0},
    )


class _GenerationChild:
    def __init__(
        self,
        layout_id,
        block,
        reports,
        *,
        events=None,
        generation=0,
        churn=False,
        bump_after_first_report=False,
        time_value=0.0,
        macro_value=0,
        bump_on_first_time=False,
    ):
        self.layout_id = layout_id
        self.block = block
        self._reports = tuple(reports)
        self._events = events if events is not None else []
        self._generation = generation
        self._churn = churn
        self._bump_after_first_report = bump_after_first_report
        self._time_value = time_value
        self._macro_value = macro_value
        self._bump_on_first_time = bump_on_first_time
        self.generation_calls = 0
        self.report_calls = 0

    def _accepted_transaction_generation_(self):
        self.generation_calls += 1
        self._events.append(("generation", self.layout_id, self._generation))
        generation = self._generation
        if self._churn:
            self._generation += 1
        return generation

    def program_report(self):
        self._events.append(("report", self.layout_id, self.report_calls))
        report = self._reports[min(self.report_calls, len(self._reports) - 1)]
        self.report_calls += 1
        if self._bump_after_first_report and self.report_calls == 1:
            self._generation += 1
        return report

    def block_names(self):
        return (self.block,)

    def time(self):
        self._events.append(("time", self.layout_id, self._time_value))
        if self._bump_on_first_time:
            self._generation += 1
            self._bump_on_first_time = False
        return self._time_value

    def macro_step(self):
        self._events.append(("macro_step", self.layout_id, self._macro_value))
        return self._macro_value


class _ChildWithoutGeneration:
    def __init__(self, layout_id, block, report):
        self.layout_id = layout_id
        self.block = block
        self._report = report

    def program_report(self):
        return self._report

    def block_names(self):
        return (self.block,)


def _multi_layout_executor(children):
    layout_programs = tuple(
        SimpleNamespace(
            layout_id=child.layout_id,
            block_names=(child.block,),
            identity=make_identity("layout-program", {"layout": child.layout_id}),
        )
        for child in children
    )
    executor = object.__new__(_MultiLayoutUniformExecutor)
    executor._plan = SimpleNamespace(
        artifact=SimpleNamespace(layout_programs=layout_programs)
    )
    executor._engines = {child.layout_id: child for child in children}
    executor._block_layouts = {child.block: child.layout_id for child in children}
    executor.block_names = lambda: tuple(child.block for child in children)
    return executor


def test_empty_authority_produces_an_honest_empty_report():
    report = build_program_report(_EmptyProgramAuthority())

    assert type(report) is ProgramRuntimeReport
    assert report.installed is False
    assert report.program_hash == ""
    assert report.step_transaction == {}
    assert report.block_map == []
    assert report.diagnostics == {}
    assert report.histories == []
    assert report.cache == []
    assert report.clocks == []
    assert report.level_relations == []
    assert report.flux_ledger == []
    assert report.synchronization == []
    assert report.temporal_partition == {}
    assert report.temporal == {}
    assert report.profiler == {"enabled": None}


def test_accepted_program_report_preserves_owned_metadata():
    report = build_program_report(_AcceptedProgramAuthority())

    assert report.installed is True
    assert report.program_hash == "sha256:accepted-program"
    assert report.step_transaction["strategy"] == {"kind": "fixed"}
    assert report.block_map == ["fluid"]
    assert report.params[0]["program_block"] == 0
    assert report.params[0]["count"] == 2
    assert report.params[0]["limit"] > 0
    assert report.diagnostics == {"mass": 3.5}
    assert report.histories == [
        {
            "name": "u_prev",
            "depth": 2,
            "ncomp": 3,
            "initialized": True,
            "fill_count": 2,
            "slot_dt": [0.125, 0.0625],
        }
    ]
    assert report.cache == [
        {
            "slot": 0,
            "valid": True,
            "cold": False,
            "name": "stage_rhs",
            "last_update_step": 4,
            "accumulated_dt": 0.125,
        },
        {
            "slot": 1,
            "valid": False,
            "cold": True,
            "name": "lazy_flux",
            "last_update_step": -1,
            "accumulated_dt": 0.25,
        },
    ]
    assert report.clocks == [
        {"kind": "logical", "clock": "main", "tick": 4},
        {
            "kind": "level",
            "level": 1,
            "macro_step": 4,
            "phase": {"numerator": 1, "denominator": 2},
            "physical_time": 0.5,
        },
    ]
    assert report.level_relations[0]["remainder_policy"] == "exact"
    assert report.flux_ledger[0]["flux"] == "transport"
    assert report.synchronization[0]["phase"] == "reflux"
    assert report.temporal_partition == {
        "kind": "cell_local",
        "provider_identity": "test.partition@1",
        "topology_epoch": 7,
        "synchronization_tick": 16,
        "tick_denominator": 32,
        "cell_count": 5,
        "rungs": [{"rung": 0, "cells": 3}, {"rung": 1, "cells": 2}],
    }
    assert report.temporal == {"schema_version": 1, "accepted_step": 4}


def test_report_serialization_is_array_free_and_detached():
    report = build_program_report(_AcceptedProgramAuthority())
    data = report.to_dict()

    assert data["schema_version"] == 5
    assert data["report_type"] == "program_runtime"
    assert json.loads(report.to_json()) == data
    assert "accepted-program" in str(report)
    assert "ProgramRuntimeReport" in repr(report)

    data["histories"].clear()
    assert report.histories


def test_multi_layout_report_preserves_the_common_temporal_partition():
    partition = {
        "kind": "global",
        "provider_identity": "pops.temporal-partition.global.v1",
    }

    def child(layout_id, block):
        report = ProgramRuntimeReport(
            installed=True,
            program_hash="sha256:%s" % layout_id,
            step_transaction={"strategy": {"kind": "fixed"}},
            block_map=[0],
            params=[{"program_block": 0, "count": 0, "limit": 8}],
            diagnostics={},
            histories=[],
            cache=[],
            profiler={"enabled": False},
            clocks=[],
            level_relations=[],
            flux_ledger=[],
            synchronization=[],
            temporal_partition=partition,
            temporal={"schema_version": 1, "accepted_step": 0},
        )
        return _GenerationChild(layout_id, block, [report])

    executor = _multi_layout_executor(
        (child("layout-a", "fluid"), child("layout-b", "field"))
    )

    report = executor.program_report()

    assert report.temporal_partition == partition


def test_multi_layout_report_retries_after_interleaved_publication():
    events = []
    first = _layout_report("layout-a", "first")
    second = _layout_report("layout-a", "second")
    child_a = _GenerationChild(
        "layout-a",
        "fluid",
        [first, second],
        events=events,
        bump_after_first_report=True,
    )
    child_b = _GenerationChild(
        "layout-b",
        "field",
        [_layout_report("layout-b", "stable")],
        events=events,
    )

    report = _multi_layout_executor((child_a, child_b)).program_report()

    assert report.diagnostics == {
        "layout-a::marker": "second",
        "layout-b::marker": "stable",
    }
    assert child_a.report_calls == 2
    assert child_b.report_calls == 2
    assert [event[0] for event in events] == [
        "generation",
        "generation",
        "report",
        "report",
        "generation",
        "generation",
        "generation",
        "generation",
        "report",
        "report",
        "generation",
        "generation",
    ]


def test_multi_layout_report_retries_a_transient_mixed_generation_exception():
    class _TransientMixedChild(_GenerationChild):
        def program_report(self):
            if self.report_calls == 0:
                self.report_calls += 1
                self._generation += 1
                raise RuntimeError("mixed accepted generation")
            return super().program_report()

    child_a = _TransientMixedChild(
        "layout-a",
        "fluid",
        [_layout_report("layout-a", "stable")],
    )
    child_b = _GenerationChild(
        "layout-b",
        "field",
        [_layout_report("layout-b", "stable")],
    )

    report = _multi_layout_executor((child_a, child_b)).program_report()

    assert report.diagnostics == {
        "layout-a::marker": "stable",
        "layout-b::marker": "stable",
    }
    assert child_a.report_calls == 2
    assert child_b.report_calls == 1


def test_multi_layout_report_refuses_bounded_generation_churn():
    from pops.runtime.program_report import _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS

    child_a = _GenerationChild(
        "layout-a",
        "fluid",
        [_layout_report("layout-a", "churn")],
        churn=True,
    )
    child_b = _GenerationChild(
        "layout-b",
        "field",
        [_layout_report("layout-b", "stable")],
        churn=True,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"multi-layout Program report changed during optimistic snapshot after "
            r"%d attempts" % _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS
        ),
    ):
        _multi_layout_executor((child_a, child_b)).program_report()

    assert child_a.generation_calls == 2 * _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS
    assert child_b.generation_calls == 2 * _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS
    assert child_a.report_calls == _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS
    assert child_b.report_calls == _MAX_OPTIMISTIC_SNAPSHOT_ATTEMPTS


def test_multi_layout_snapshot_requires_generation_witness():
    child_a = _ChildWithoutGeneration("layout-a", "fluid", _layout_report("layout-a", "a"))
    child_b = _GenerationChild(
        "layout-b",
        "field",
        [_layout_report("layout-b", "b")],
    )

    executor = _multi_layout_executor((child_a, child_b))
    with pytest.raises(
        RuntimeError,
        match=r"multi-layout Program report child 0 lacks _accepted_transaction_generation_\(\)",
    ):
        executor.program_report()


def test_multi_layout_snapshot_uses_stable_layout_order_for_reports():
    events = []
    child_b = _GenerationChild(
        "layout-b",
        "coarse",
        [_layout_report("layout-b", "b")],
        events=events,
    )
    child_a = _GenerationChild(
        "layout-a",
        "fine",
        [_layout_report("layout-a", "a")],
        events=events,
    )

    report = _multi_layout_executor((child_b, child_a)).program_report()

    assert report.block_map == [0, 1]
    assert [event[:2] for event in events] == [
        ("generation", "layout-b"),
        ("generation", "layout-a"),
        ("report", "layout-b"),
        ("report", "layout-a"),
        ("generation", "layout-b"),
        ("generation", "layout-a"),
    ]


def test_multi_layout_clock_reads_retry_and_refuse_missing_witness():
    events = []
    child_a = _GenerationChild(
        "layout-a",
        "fluid",
        [_layout_report("layout-a", "a")],
        events=events,
        time_value=1.25,
        macro_value=7,
        bump_on_first_time=True,
    )
    child_b = _GenerationChild(
        "layout-b",
        "field",
        [_layout_report("layout-b", "b")],
        events=events,
        time_value=1.25,
        macro_value=7,
    )
    executor = _multi_layout_executor((child_a, child_b))

    assert executor.time() == 1.25
    assert executor.macro_step() == 7

    missing = _ChildWithoutGeneration("layout-a", "fluid", _layout_report("layout-a", "a"))
    missing_executor = _multi_layout_executor((missing, child_b))
    with pytest.raises(
        RuntimeError,
        match=r"multi-layout native clock time child 0 lacks _accepted_transaction_generation_\(\)",
    ):
        missing_executor.time()


def test_system_and_amr_system_forward_native_generation_witness():
    repository = Path(__file__).resolve().parents[4]
    for relative, class_name in (
        ("python/pops/runtime/_system.py", "System"),
        ("python/pops/runtime/_amr_system.py", "AmrSystem"),
    ):
        tree = ast.parse((repository / relative).read_text(encoding="utf-8"))
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        method = next(
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_accepted_transaction_generation_"
        )
        returns = [node for node in method.body if isinstance(node, ast.Return)]
        assert len(returns) == 1
        call = returns[0].value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Attribute)
        assert call.func.attr == "accepted_transaction_generation_"
        assert isinstance(call.func.value, ast.Attribute)
        assert call.func.value.attr == "_s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
