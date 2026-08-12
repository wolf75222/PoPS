"""Final metadata-only report for the installed ``Program`` subsystem.

The public runtime is ``RuntimeInstance`` and delegates ``program_report()`` to its authenticated
executor.  These unit checks exercise the single report owner directly: no legacy ``System`` is
constructed and no native state array is read.
"""

from __future__ import annotations

import json
import sys
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

    def program_cache_nodes(self):
        return [7]

    def program_cache_name(self, node):
        assert node == 7
        return "stage_rhs"

    def program_cache_last_update_step(self, node):
        assert node == 7
        return 4

    def program_cache_accumulated_dt(self, node):
        assert node == 7
        return 0.125

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
            "node_id": 7,
            "name": "stage_rhs",
            "last_update_step": 4,
            "accumulated_dt": 0.125,
        }
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
        layout_program = SimpleNamespace(
            layout_id=layout_id,
            identity=make_identity("layout-program", {"layout": layout_id}),
        )
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
        return layout_program, (block,), report

    executor = object.__new__(_MultiLayoutUniformExecutor)
    executor._ordered_program_reports = lambda: (
        child("layout-a", "fluid"),
        child("layout-b", "field"),
    )
    executor.block_names = lambda: ("fluid", "field")

    report = executor.program_report()

    assert report.temporal_partition == partition


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
