"""Executable acceptance for the normative explicit-IMEX + AMR target."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_final_imex_amr", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_runs_and_every_scientific_format_reopens(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["POPS_INCLUDE"] = str(ROOT / "include")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--output-dir", str(tmp_path / "published")],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "HDF5:" in completed.stdout
    assert "ParaView:" in completed.stdout
    assert "checkpoint:" in completed.stdout
    assert "bit-identical restart: True" in completed.stdout
    assert "bit-identical continuation: True" in completed.stdout
    assert "manual/pops.lib.time.IMEX parity: True" in completed.stdout
    assert "rejected-attempt rollback: True" in completed.stdout
    assert "regrid count:" in completed.stdout
    report_line, = [line for line in completed.stdout.splitlines() if line.startswith("report: ")]
    report = json.loads(report_line.removeprefix("report: "))
    assert report["finite"] is True
    assert report["checkpoint_restart_bit_identical"] is True
    assert report["continuation_bit_identical"] is True
    assert report["manual_preset_bit_identical"] is True
    assert report["rejected_attempt_rollback"] is True
    assert report["rejected_attempt_error"].startswith(
        "step attempt rejected during "
    )
    assert report["levels"] == 2
    assert report["regrid_count"] >= 0
    assert report["topology_epoch"] >= 0
    assert report["regrid_count_after_continuation"] >= report["regrid_count"]
    assert report["topology_epoch_after_continuation"] >= report["topology_epoch"]
    assert report["program_accepted_state_bytes"] > 0
    assert report["tagging_hysteresis_min_cycles"] == 2
    assert report["flux_ledger_levels"] == [0, 1]
    assert report["synchronization_phases"] == ["reflux", "average_down"]
    assert report["runtime_steps"] == 1
    assert report["runtime_steps_after_continuation"] == 2

    from pops.output import read_hdf5, read_npz, read_paraview

    output = tmp_path / "published"
    assert not tuple((output / "rejected").rglob("*"))
    readers = {".h5": read_hdf5, ".npz": read_npz, ".vtu": read_paraview}
    for suffix, reader in readers.items():
        # Scientific writers use the stable ``consumer__clock__step`` stem. Checkpoints are also
        # NPZ containers but intentionally have a different schema and must not be opened as output.
        paths = tuple(output.rglob("*__*%s" % suffix))
        assert paths, "the accepted step did not publish %s" % suffix
        reopened = reader(paths[-1])
        assert reopened.arrays
    assert tuple(output.rglob("manual_restart*.npz"))


def test_resolved_amr_lowering_report_covers_every_executed_authority() -> None:
    import pops

    example = _load_example()
    target = example.build_final_case()
    resolved = pops.resolve(
        pops.validate(target.authoring.case),
        layout=target.layout,
    )
    coverage = resolved.lowering_coverage
    amr_rows = tuple(
        row for row in coverage.rows if row.source.startswith("amr-")
    )
    assert amr_rows
    assert all(row.disposition == "lowered" and row.targets for row in amr_rows)

    predicate_sources = {
        row.source.rsplit(":", 1)[-1]
        for row in amr_rows
        if row.source.startswith("amr-tagging-predicate:")
    }
    assert predicate_sources == {"above", "any_of", "below", "gradient_above"}
    source_families = {
        row.source.split(":", 1)[0]
        for row in amr_rows
    }
    assert {
        "amr-bootstrap",
        "amr-execution",
        "amr-hierarchy",
        "amr-regrid",
        "amr-subcycling",
        "amr-tagging-conflict-policy",
        "amr-tagging-graph",
        "amr-tagging-hysteresis",
        "amr-tagging-predicate",
        "amr-transfer-entry",
        "amr-transfer-plan",
    } <= source_families
    transfer_targets = {
        target
        for row in amr_rows
        for target in row.targets
        if target.startswith("amr-runtime-transfer-operation:")
    }
    assert transfer_targets == {
        "amr-runtime-transfer-operation:apply_transfer_provider:coarse_fine_fill",
        "amr-runtime-transfer-operation:apply_transfer_provider:prolongation",
        "amr-runtime-transfer-operation:apply_transfer_provider:restriction",
        "amr-runtime-transfer-operation:apply_transfer_provider:temporal_interpolation",
        "amr-runtime-transfer-operation:recompute:coarse_fine_fill",
    }
    assert any(
        target == "amr-runtime-clock-relation:0-1:2/1"
        for row in amr_rows
        for target in row.targets
    )
    hysteresis_row, = (
        row for row in amr_rows
        if row.source.startswith("amr-tagging-hysteresis:")
    )
    assert (
        "amr-runtime-program-accepted-state:tagging_hysteresis_state"
        in hysteresis_row.targets
    )

    tagging = resolved.bootstrap_plan.tagging.inspect()["graph"]
    assert tagging["refine"]["node_type"] == "any_of"
    assert {
        child["node_type"] for child in tagging["refine"]["children"]
    } == {"above", "gradient_above"}
    assert tagging["coarsen"]["node_type"] == "below"
    assert tagging["hysteresis"] == {
        "schema_version": 1,
        "hysteresis_type": "min_cycles",
        "min_cycles": 2,
        "equality": "hold",
    }
    assert tagging["conflict_policy"] == "refine_wins"


def test_normative_example_uses_only_the_final_root_lifecycle() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "pops.validate(" in source
    assert "pops.resolve(" in source
    assert "pops.compile(" in source
    assert "pops.bind(" in source
    assert "pops.run(simulation," in source
    assert ".run(**" not in source
    assert "BindInputs" not in source
    assert "simulation.program_accepted_state()" in source
    for forbidden in (
        "ProgramContext",
        "AmrProgramContext",
        "SystemStepper",
        "_executor",
        "_begin_step_transaction",
        "_commit_step_transaction",
        "_rollback_step_transaction",
    ):
        assert forbidden not in source
    assert source.count("case.program(") == 1
    assert source.count("case.consumers(") == 1

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    root_resolve = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pops"
        and node.func.attr == "resolve"
    ]
    assert len(root_resolve) == 1
    assert all(keyword.arg != "strict" for keyword in root_resolve[0].keywords)
    root_run = [
        node for node in calls
        if isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pops"
        and node.func.attr == "run"
    ]
    # Rejected proof, accepted manual step, both continuations and preset parity.
    assert len(root_run) == 5
