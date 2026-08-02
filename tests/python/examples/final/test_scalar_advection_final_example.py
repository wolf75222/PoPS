from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_SCALAIRE_COMPLET.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_final_scalar_advection", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_supported_authoring_core_is_genuine_and_inert():
    from pops.identity.semantic import program_semantic_data, semantic_identity_of
    from pops.time import Commit, FixedDt

    module = _load_example()
    core = module.build_authoring()

    assert core.domain.boundaries.x_min.name == "inlet_x"
    assert core.domain.boundaries.x_max.name == "outlet_x"
    assert core.grid.cells == (128, 128)
    assert core.state.space.representation == "conservative"
    assert core.model.rate_contract(core.rate) == {
        "state": core.state,
        "flux": core.flux,
        "sources": (),
    }
    assert core.numerics.validate_for(core.model)
    assert core.finite_volume.formal_order == 2
    assert core.finite_volume.reconstruction.options["ghost_depth"] == 2

    assert core.tracer_state.is_instance
    assert core.tracer_state.declaration_ref == core.state
    assert core.refine_threshold.owner_path == core.case.owner_path
    assert core.coarsen_threshold.owner_path == core.case.owner_path
    assert core.numerics.boundaries.values() == ()

    graph = core.program.to_graph()
    assert sum(isinstance(node, Commit) for node in graph.nodes) == 1
    assert core.program.transaction_plan().strategy.kind == "adaptive_cfl"
    assert set(core.run_controls) == {"t_end", "max_steps", "output_dir"}

    preset = module.build_authoring(program_builder=module.preset_ssprk2)
    assert preset.program.to_graph().to_data() == core.program.to_graph().to_data()
    assert preset.program.to_graph().graph_hash == core.program.to_graph().graph_hash
    assert program_semantic_data(preset.program) == program_semantic_data(core.program)
    assert semantic_identity_of(program=preset.program) == \
        semantic_identity_of(program=core.program)

    different_controller = module.explicit_ssprk2(core.tracer_state, core.rate)
    different_controller.step_strategy(FixedDt(1.0e-2))
    assert semantic_identity_of(program=different_controller) != \
        semantic_identity_of(program=core.program)


def test_target_has_one_authority_per_concern_and_no_legacy_path():
    source = EXAMPLE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert source.count("numerics.boundaries.add(") == 1
    assert source.count("case.numerics(") == 1
    assert source.count("case.program(") == 1
    assert source.count("case.consumers(") == 1
    assert source.count("transfer.state(") == 1

    forbidden = (
        "disc.transfer",
        "numerics.transfer",
        "case.boundaries",
        "case.output(",
        "case.outputs(",
        "case.runtime(",
        "Runtime" + "Policies",
        "Output" + "Policy",
        "Checkpoint" + "Policy",
        "bind_operators",
        "linear_combine",
        "RejectOldManifest",
        "add_block(",
        "pops." + "Pro" + "blem",
        "Bind" + "Inputs",
        "simulation." + "run(",
    )
    for spelling in forbidden:
        assert spelling not in source

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    resolve_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve"
    ]
    assert resolve_calls
    assert all(
        keyword.arg != "strict"
        for node in resolve_calls
        for keyword in node.keywords
    )
    state_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "state"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "program"
    ]
    assert len(state_calls) == 1
    assert isinstance(state_calls[0].args[0], ast.Name)
    assert state_calls[0].args[0].id == "state"
    assert "program_builder(tracer_state, rate)" in source

    assert sum(
        isinstance(node.func, ast.Attribute) and node.func.attr == "value"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "program"
        for node in calls
    ) == 4
    assert "StagePoint(" in source
    assert "StateTransfer()" in source
    assert "pops.run(simulation, **controls)" in source
    assert "pops.run(\n        simulation," in source
    assert "AMRTransfer.conservative(order=" not in source
    assert "ScientificOutput(" in source
    assert "Checkpoint(" in source
    assert "def explicit_ssprk2(" in source
    assert "def preset_ssprk2(" in source
    assert "read_hdf5(" in source
    assert "read_paraview(" in source
    assert "_scalar_error_norms(paraview)" in source
    assert "simulation.program_report()" in source
    assert "simulation.amr.explain_regrid()" in source
    assert "simulation.amr.explain_checkpoint()" in source
    assert "simulation.checkpoint(" in source
    assert "resumed.restart(" in source


def test_handle_reads_are_explicit_before_symbolic_parameter_algebra():
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "a_x = model.value(velocity_x_param)" in source
    assert "a_y = model.value(velocity_y_param)" in source
    assert "u_in_x = model.value(inlet_x_param)" in source
    assert "u_in_y = model.value(inlet_y_param)" in source
    assert "ValueExpr(core.tracer_state)" in source
    assert "core.case.value(core.refine_threshold)" in source
    assert "core.case.value(core.coarsen_threshold)" in source


def test_reopened_leaf_cell_error_uses_exact_characteristics_and_cell_volumes():
    module = _load_example()
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=np.float64,
    )
    exact = module._analytic_solution(
        np.asarray((0.5,)),
        np.asarray((0.5,)),
        time=0.0,
    )
    perturbation = 1.0e-3
    reopened = SimpleNamespace(
        manifest={
            "datasets": {
                "fields": {
                    "qualified-state": {
                        "name": "U",
                        "association": "cell",
                    },
                },
            },
        },
        arrays={
            "U": (exact + perturbation).reshape((1, 1)),
            "Points": points,
            "connectivity": np.asarray((0, 1, 2, 3), dtype=np.int64),
            "offsets": np.asarray((4,), dtype=np.int64),
            "pops_coverage": np.asarray((0,), dtype=np.uint8),
            "vtkGhostType": np.asarray((0,), dtype=np.uint8),
            "pops_cell_volume": np.asarray((1.0,), dtype=np.float64),
            "TimeValue": np.asarray((0.0,), dtype=np.float64),
        },
    )

    error = module._scalar_error_norms(reopened)

    assert error.time == 0.0
    assert error.active_cells == 1
    assert np.isclose(error.l1, perturbation)
    assert np.isclose(error.l2, perturbation)
    assert np.isclose(error.linf, perturbation)
    assert np.isclose(error.relative_l2, perturbation / exact[0])
    assert error.relative_l2 < module.RELATIVE_L2_TOLERANCE


def test_program_evidence_requires_every_level_and_ordered_amr_synchronization():
    module = _load_example()
    synchronization = []
    for parent, child in ((0, 1), (1, 2)):
        for phase in ("reflux", "average_down"):
            synchronization.append({
                "parent_level": parent,
                "child_level": child,
                "block": 0,
                "phase": phase,
                "macro_step": 4,
                "clock_phase": {"numerator": 1, "denominator": 1},
            })
    report = SimpleNamespace(
        installed=True,
        flux_ledger=[{"level": level} for level in (0, 1, 2)],
        synchronization=synchronization,
    )

    evidence = module._require_multilevel_program_evidence(
        report,
        expected_levels=(0, 1, 2),
    )

    assert evidence.flux_ledger_levels == (0, 1, 2)
    assert evidence.synchronization_relations == ((0, 1), (1, 2))
    assert evidence.synchronization_phases == ("reflux", "average_down")


def test_regrid_progress_requires_a_completed_topology_replacement():
    module = _load_example()
    before = SimpleNamespace(macro_step=5, regrid_count=2, topology_epoch=3)

    with pytest.raises(RuntimeError, match="did not complete a dynamic regrid"):
        module._require_regrid_progress(
            before,
            SimpleNamespace(macro_step=10, regrid_count=2, topology_epoch=3),
            where="unit continuation",
        )
    with pytest.raises(RuntimeError, match="did not replace the accepted topology"):
        module._require_regrid_progress(
            before,
            SimpleNamespace(macro_step=10, regrid_count=3, topology_epoch=3),
            where="unit continuation",
        )

    module._require_regrid_progress(
        before,
        SimpleNamespace(macro_step=10, regrid_count=3, topology_epoch=4),
        where="unit continuation",
    )
