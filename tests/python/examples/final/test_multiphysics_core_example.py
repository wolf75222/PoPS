"""Executable acceptance for the canonical ADC-690 multi-block Program core."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

from pops.mesh import CartesianMesh
from pops.mesh.layouts import Uniform


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_final_multiphysics_core", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_script_exits_zero_without_mock_or_fallback() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PoPS multiphysics Program graph:" in completed.stdout


def test_program_has_exact_field_context_and_transactional_implicit_join() -> None:
    core = _load_example().build_multiphysics_core()
    values = tuple(core.program._values)
    operations = [value.op for value in values]

    assert operations.count("solve_fields_from_blocks") == 1
    assert operations.count("solve_coupled_implicit") == 1
    assert operations.count("solve_outcome") == 2
    assert operations.count("solve_outcome_component") == 3
    assert len(core.program.commits()) == 2

    field_token = next(value for value in values if value.op == "solve_fields_from_blocks")
    assert field_token.field_context.stage_sources == (
        (field_token.inputs[0].block, field_token.inputs[0].id),
        (field_token.inputs[1].block, field_token.inputs[1].id),
    )
    solve_actions = {
        value.inputs[0].op: value.attrs["action"].kind
        for value in values if value.op == "solve_outcome"
    }
    assert solve_actions == {
        "solve_fields_from_blocks": "fail_run",
        "solve_coupled_implicit": "reject_attempt",
    }
    assert {state.block_ref.local_id for state in core.program.commits()} == {
        "electrons", "ions"
    }
    core.program.to_graph()


def test_case_resolves_two_provider_field_and_two_state_spaces() -> None:
    import pops

    core = _load_example().build_multiphysics_core()
    manifest = core.module.manifest()
    field_entries = [
        row for row in manifest.provider_pack["entries"]
        if row["key"]["space_kind"] == "field"
    ]
    assert len(field_entries) == 3
    assert all(
        row["provider"]["producer"].startswith("field_provider_set:[")
        for row in field_entries
    )

    validated = pops.validate(core.case)
    resolved = pops.resolve(
        validated,
        layout=Uniform(CartesianMesh(n=8, periodic=True)),
    )

    assert tuple(block.name for block in resolved.blocks) == ("electrons", "ions")
    assert tuple(resolved.field_plans) == ("electrostatic",)
    provider_pack = resolved.field_plans["electrostatic"].native_options["provider_pack"]
    assert [row["owner_block"] for row in provider_pack] == ["electrons", "ions"]
    assert [row["key"] for row in provider_pack] == ["electron_charge", "ion_charge"]
