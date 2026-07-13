from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


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
    from pops.time import Commit

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
        "RuntimePolicies",
        "OutputPolicy",
        "CheckpointPolicy",
        "bind_operators",
        "linear_combine",
        "strict=True",
        "RejectOldManifest",
        "add_block(",
        "pops." + "Pro" + "blem",
    )
    for spelling in forbidden:
        assert spelling not in source

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    state_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "state"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "program"
    ]
    assert len(state_calls) == 1
    assert isinstance(state_calls[0].args[0], ast.Name)
    assert state_calls[0].args[0].id == "tracer_state"

    assert sum(
        isinstance(node.func, ast.Attribute) and node.func.attr == "value"
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "program"
        for node in calls
    ) == 2
    assert "StagePoint(" in source
    assert "StateTransfer()" in source
    assert "AMRTransfer.conservative(order=" not in source
    assert "ScientificOutput(" in source
    assert "Checkpoint(" in source


def test_handle_reads_are_explicit_before_symbolic_parameter_algebra():
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "a_x = model.value(velocity_x_param)" in source
    assert "a_y = model.value(velocity_y_param)" in source
    assert "u_in_x = model.value(inlet_x_param)" in source
    assert "u_in_y = model.value(inlet_y_param)" in source
    assert "ValueExpr(core.tracer_state)" in source
    assert "core.case.value(core.refine_threshold)" in source
    assert "core.case.value(core.coarsen_threshold)" in source
