"""Executable acceptance for the final ADC-690 multiphysics lifecycle."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_final_multiphysics_core", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_example_script_runs_outputs_and_restart_without_mock_or_fallback(tmp_path) -> None:
    output = tmp_path / "complete"
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE), "--cells", "8", "--output-dir", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "PoPS final multiphysics acceptance:" in completed.stdout
    assert "bit-identical restart: step 2" in completed.stdout

    from pops.output import HDF5, ParaView

    hdf5 = HDF5().reopen_series(output / "accepted" / "state" / "two_fluid").latest
    paraview = ParaView().reopen_series(
        output / "accepted" / "visualization" / "two_fluid").latest
    assert hdf5.output_identity.token in completed.stdout
    assert paraview.output_identity.token in completed.stdout
    example = _load_example()
    target = example.build_final_case(
        cells=8,
        output_mode=example._native_output_mode(),
    )
    import pops

    resolved = pops.resolve(
        target.authoring.case,
        layout=target.layout_plan,
        layout_providers={target.layout_handle: target.layout_provider},
    )
    diagnostic_output = next(
        node
        for node in resolved.consumer_graph.nodes
        if node.target_uri == "state/two_fluid"
    )
    quantities = {
        quantity.identity.token: quantity
        for quantity in diagnostic_output.diagnostic_quantities
    }
    diagnostic_rows = hdf5.manifest["snapshot"]["diagnostics"]
    assert len(diagnostic_rows) == 6
    assert {row["key"]["state_id"] for row in diagnostic_rows} == set(quantities)
    from pops.identity import Identity

    for row in diagnostic_rows:
        quantity = quantities[row["key"]["state_id"]]
        assert row["key"]["reference"] == quantity.reference.canonical_identity()
        assert row["key"]["reduction"] == "integral"
        assert row["key"]["level"] == 0
        assert Identity.from_token(row["key"]["layout_identity"]).domain == "layout"
        block = quantity.reference.block_ref.local_id
        role = quantity.execution["role"]
        coefficient = quantity.execution["operations"][0]["coefficient"]
        expected_coefficient = -1.0 if (block, role) == ("electrons", "Density") else 1.0
        assert coefficient == expected_coefficient.hex()
        if role == "Density":
            value = float.fromhex(row["value"])
            assert value < 0.0 if block == "electrons" else value > 0.0
        # State-space units intentionally fail closed until PoPS has a typed unit protocol.
        assert row["units"] == "unspecified"

    checkpoint = output / "accepted_restart.npz"
    assert checkpoint.is_file()
    with np.load(checkpoint, allow_pickle=False) as stored:
        assert [str(name) for name in stored["blocks"]] == ["electrons", "ions"]
        assert float(stored["t"]) == 1.0e-3
        assert int(stored["macro_step"]) == 1
        assert sorted(str(name) for name in stored["history_names"]) == [
            "electrons.electrons", "ions.ions"]
        assert "runtime_consumer_graph" in stored
        assert "field_provider_slots" in stored


def test_program_has_exact_field_context_and_transactional_implicit_join() -> None:
    core = _load_example().build_authoring()
    values = tuple(core.program._values)
    operations = [value.op for value in values]

    assert operations.count("solve_fields_from_blocks") == 1
    assert operations.count("solve_coupled_implicit") == 1
    assert operations.count("solve_outcome") == 2
    assert operations.count("solve_outcome_component") == 3
    assert operations.count("store_history") == 2
    assert len(core.program.commits()) == 2

    field_token = next(value for value in values if value.op == "solve_fields_from_blocks")
    assert field_token.field_context.stage_sources == (
        (field_token.inputs[0].block, field_token.inputs[0].id),
        (field_token.inputs[1].block, field_token.inputs[1].id),
    )
    collision_token = next(
        value for value in values if value.op == "solve_coupled_implicit"
    )
    assert collision_token.attrs["operator"] == "implicit_collision"
    assert collision_token.attrs["problem_kind"] == "coupled_implicit_euler"
    assert collision_token.attrs["method"] == "newton"
    assert collision_token.attrs["max_iter"] == 12
    assert tuple(
        block.local_id for block in collision_token.attrs["blocks"]
    ) == ("electrons", "ions")
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


def test_multispecies_dependencies_refuse_name_selection() -> None:
    import pops

    with pytest.raises(TypeError, match="typed ComponentRole"):
        pops.Model("bad_roles").species(
            "electrons", state=("n",), roles={"n": "density"})
    core = _load_example().build_authoring()
    with pytest.raises(TypeError, match="exact StateHandle"):
        core.model.field_provider(
            "bad_named_provider", on="electrons", into=core.field_space, value=0.0)
    with pytest.raises(TypeError, match="exact StateHandle"):
        core.model.coupled_rate(
            "bad_named_collision",
            inputs=("electrons", core.ion_space),
            outputs={
                core.electron_space: (0.0, 0.0, 0.0),
                core.ion_space: (0.0, 0.0, 0.0),
            },
        )
    core.program.to_graph()


def test_case_resolves_explicit_layout_consumers_and_two_provider_field() -> None:
    import pops

    example = _load_example()
    target = example.build_final_case(cells=8)
    core = target.authoring
    assert type(core.model) is pops.Model
    manifest = core.model.module.manifest()
    field_entries = [
        row for row in manifest.provider_pack["entries"]
        if row["key"]["space_kind"] == "field"
    ]
    assert len(field_entries) == 3
    assert all(
        row["provider"]["producer"].startswith("field_provider_set:[")
        for row in field_entries
    )

    resolved = pops.resolve(
        core.case,
        layout=target.layout_plan,
        layout_providers={target.layout_handle: target.layout_provider},
    )

    assert tuple(block.name for block in resolved.blocks) == ("electrons", "ions")
    assert tuple(block.state_spaces for block in resolved.blocks) == (
        ("electrons",), ("ions",))
    assert tuple(resolved.field_plans) == ("electrostatic",)
    assert len(resolved.layout_plan.layouts) == 1
    subjects = core.case.layout_subjects()
    assert len(resolved.layout_plan.assignments) == sum(
        map(len, (subjects.blocks, subjects.states, subjects.fields)))
    assert resolved.consumer_graph.is_resolved
    assert sorted(node.kind.value for node in resolved.consumer_graph.nodes) == [
        "checkpoint", "scientific_output", "scientific_output"]
    diagnostic_output = next(
        node
        for node in resolved.consumer_graph.nodes
        if node.target_uri == "state/two_fluid"
    )
    assert len(diagnostic_output.diagnostics) == 6
    assert len(diagnostic_output.diagnostic_quantities) == 6
    expected_diagnostics = {
        ("electrons", "Density"),
        ("electrons", "MomentumX"),
        ("electrons", "MomentumY"),
        ("ions", "Density"),
        ("ions", "MomentumX"),
        ("ions", "MomentumY"),
    }
    actual_diagnostics = {
        (
            quantity.reference.block_ref.local_id,
            quantity.execution["role"],
        )
        for quantity in diagnostic_output.diagnostic_quantities
    }
    assert actual_diagnostics == expected_diagnostics
    assert {
        quantity.layout_id
        for quantity in diagnostic_output.diagnostic_quantities
    } == {target.layout_handle.qualified_id}
    assert all(
        quantity.levels == (0,)
        and quantity.execution["operations"] == (
            {
                "name": "integral",
                "reduction": "sum",
                "transform": "identity",
                "metric_weighted": True,
                "coefficient": (
                    -1.0
                    if quantity.reference.block_ref.local_id == "electrons"
                    and quantity.execution["role"] == "Density"
                    else 1.0
                ).hex(),
            },
        )
        for quantity in diagnostic_output.diagnostic_quantities
    )
    provider_pack = resolved.field_plans["electrostatic"].native_options["provider_pack"]
    assert [row["owner_block"] for row in provider_pack] == ["electrons", "ions"]
    assert [row["key"] for row in provider_pack] == ["electron_charge", "ion_charge"]
    field_plan = resolved.field_plans["electrostatic"]
    native_options = field_plan.native_options
    assert native_options["rhs"] == "composite"
    assert native_options["method"] == {
        "native_method": "cell_centered_second_order",
        "order": 2,
        "ghost_depth": 1,
    }
    assert native_options["bc"] == "explicit"
    solver_provider = native_options["solver_provider"]
    assert solver_provider["provider"]["provider_id"] == "pops.field-solver.geometric-mg"
    assert {
        face["type"]
        for face in solver_provider["facts"]["boundary"]["faces"]
    } == {"periodic"}
    nullspace_provider = native_options["nullspace_provider"]
    assert (
        nullspace_provider["provider"]["provider_id"]
        == "pops.field-nullspace.constant"
    )
    assert nullspace_provider["resolution"]["singular"] is True
    assert (
        nullspace_provider["resolution"]["native_contract"]["options"]["gauge.value"]
        == 0.0
    )
    equation = field_plan.operator.inspect()["physics"]["equation"]["equation"]
    assert (
        equation["lhs"]["field_expression"]["type"]
        == "pops._ir.expr.Laplacian"
    )
    assert equation["rhs"]["protocol"] == "pops.expr.dag.v1"
    output_route = field_plan.native_options["output_route"]
    assert output_route["owner_block"] == "electrons"
    assert output_route["key"] == "electrostatic"
    assert tuple(output_route["components"]) == (
        "potential", "electric_x", "electric_y")
    output_targets = field_plan.coverage.source_to_targets["field:electrostatic:output"]
    assert len(output_targets) == 1
    assert output_targets[0].startswith(
        "field-install:electrostatic:output:electrons:pops.handle.v1::")
