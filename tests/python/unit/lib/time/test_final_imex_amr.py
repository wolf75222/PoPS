from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pops
from pops.codegen import Production
from pops.identity.semantic import program_semantic_data, semantic_identity_of
from pops.time import ValueRef


ROOT = Path(__file__).resolve().parents[5]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_ADVECTION_IMEX_AMR.py"


def _example():
    spec = importlib.util.spec_from_file_location("pops_final_imex_amr", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cn_heun_tableau_retains_exact_coefficients_and_stage_coordinates():
    example = _example()
    method = example.IMEX_CN_HEUN

    assert method.explicit.A == ((), (Fraction(1),))
    assert method.explicit.b == (Fraction(1, 2), Fraction(1, 2))
    assert method.implicit_A == (
        (Fraction(0),),
        (Fraction(1, 2), Fraction(1, 2)),
    )
    assert method.implicit_b == (Fraction(1, 2), Fraction(1, 2))
    assert method.abscissae == (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1)),
    )


def test_manual_and_library_imex_are_the_same_graph_and_consume_every_solve():
    example = _example()
    manual_authoring = example.build_authoring(use_preset=False)
    preset_authoring = example.build_authoring(use_preset=True)
    manual = manual_authoring.program
    preset = preset_authoring.program
    manual_graph = manual.to_graph()
    preset_graph = preset.to_graph()

    assert manual_graph.graph_hash == preset_graph.graph_hash
    assert manual_graph.to_data() == preset_graph.to_data()
    assert program_semantic_data(manual) == program_semantic_data(preset)
    assert semantic_identity_of(program=manual) == semantic_identity_of(program=preset)

    for authored in (manual_authoring, preset_authoring):
        field_solves = [value for value in authored.program._values if value.op == "solve_fields"]
        assert len(field_solves) == example.IMEX_CN_HEUN.stages
        assert all(value.attrs["field"] is authored.diagnostic_field for value in field_solves)
        assert all(
            value.field_context.outputs == ("relaxation_potential",)
            for value in field_solves
        )

    solves = [
        node for node in manual_graph.nodes
        if getattr(node, "op", None) in {
            "solve_fields", "solve_fields_from_blocks", "solve_linear", "solve_local_linear",
            "solve_residual", "solve_coupled_implicit",
        }
    ]
    outcomes = [node for node in manual_graph.nodes if getattr(node, "op", None) == "solve_outcome"]
    implicit_solves = sum(
        coefficient != 0
        for index, row in enumerate(example.IMEX_CN_HEUN.implicit_A)
        for coefficient in row[index:index + 1]
    )
    assert len(solves) == len(outcomes) == example.IMEX_CN_HEUN.stages + implicit_solves
    for solve in solves:
        consumed = [
            node for node in outcomes
            if node.inputs == (ValueRef(solve.node_id),)
        ]
        assert len(consumed) == 1
        action = consumed[0].attrs.to_data()["attrs"]["action"]
        assert action["kind"] == "reject_attempt"


def test_stage_field_contexts_are_distinct_and_read_at_their_exact_stage():
    example = _example()
    program = example.build_authoring().program
    fields = [value for value in program._values if value.name.startswith("cn-heun-imex_fields_")]
    assert len(fields) == example.IMEX_CN_HEUN.stages
    assert fields[0].field_context != fields[1].field_context
    assert fields[0].point.time_for("explicit").offset.to_python() == Fraction(0)
    assert fields[1].point.time_for("explicit").offset.to_python() == Fraction(1)
    assert fields[0].point.time_for("implicit").offset.to_python() == Fraction(0)
    assert fields[1].point.time_for("implicit").offset.to_python() == Fraction(1)


def test_runtime_snapshot_compares_every_amr_level_by_exact_bits():
    example = _example()

    class TwoLevelRuntime:
        amr = SimpleNamespace(
            explain_regrid=lambda: SimpleNamespace(regrid_count=0, topology_epoch=0),
        )
        consumer_graph = SimpleNamespace(identity=SimpleNamespace(token="consumer:test"))
        consumer_cursors = SimpleNamespace(
            to_data=lambda: {"schema_version": 1, "rows": []},
        )

        @staticmethod
        def block_names():
            return ("tracer",)

        @staticmethod
        def n_levels():
            return 2

        @staticmethod
        def field_provider_slots():
            return ("case:test::field::fields",)

        @staticmethod
        def field_provider_levels(_slot):
            return 2

        @staticmethod
        def block_level_state_global(_block, level):
            return np.asarray([0.0, float(level)], dtype=np.float64)

        @staticmethod
        def field_potential_level_global(_slot, level):
            return np.asarray([0.0, float(level + 2)], dtype=np.float64)

        @staticmethod
        def patch_boxes():
            return ((0, 0, 0, 7, 7), (1, 2, 2, 5, 5))

        @staticmethod
        def installed_program_hash():
            return "program:test"

        @staticmethod
        def time():
            return 0.125

        @staticmethod
        def macro_step():
            return 3

    snapshot = example._snapshot(TwoLevelRuntime())
    assert tuple(map(len, snapshot.states.values())) == (2,)
    assert tuple(map(len, snapshot.fields.values())) == (2,)
    assert example._snapshots_bit_identical(snapshot, snapshot)

    state_levels = list(snapshot.states["tracer"])
    state_levels[1] = state_levels[1].copy()
    state_levels[1][0] = -0.0
    changed_state = replace(snapshot, states={"tracer": tuple(state_levels)})
    assert not example._snapshots_bit_identical(snapshot, changed_state)

    slot = next(iter(snapshot.fields))
    field_levels = list(snapshot.fields[slot])
    field_levels[1] = field_levels[1].copy()
    field_levels[1][0] = -0.0
    changed_field = replace(snapshot, fields={slot: tuple(field_levels)})
    assert not example._snapshots_bit_identical(snapshot, changed_field)


def test_field_install_consumes_the_public_amr_layout_contract():
    example = _example()
    from pops.amr import ConflictPolicy, EqualityPolicy
    from pops.mesh._amr import Above, AnyOf, Below, GradientAbove

    target = example.build_final_case()
    resolved = pops.resolve(
        pops.validate(target.authoring.case),
        layout=target.layout,
        backend=Production(),
    )

    assert resolved.target == "amr_system"
    assert resolved.resolved_hierarchy.plan.level_count == 2
    recipe = resolved.field_plans["fields"].native_options["topology_recipe"]
    assert recipe["connectivity"]["graph"] == "amr-composite-cell-graph"
    assert recipe["levels"] == 2
    assert recipe["transition_ratios"] == [2]
    assert recipe["level_refinements"] == [1, 2]
    graph = resolved.bootstrap_plan.tagging.graph
    assert type(graph.refine) is AnyOf
    assert tuple(type(child) for child in graph.refine.children) == (Above, GradientAbove)
    assert type(graph.coarsen) is Below
    assert graph.hysteresis.min_cycles == 0
    assert graph.hysteresis.equality is EqualityPolicy.HOLD
    assert graph.conflict_policy is ConflictPolicy.REFINE_WINS


def test_every_output_and_checkpoint_schedule_is_accepted_step_only():
    example = _example()
    target = example.build_final_case()
    graph = target.authoring.case._consumers
    data = graph.inspect()
    assert len(data["nodes"]) == 4
    assert {node["output_format"]["provider_id"] for node in data["nodes"]
            if node["output_format"] is not None} == {
        "pops.output.hdf5.v1",
        "pops.output.npz.v1",
        "pops.output.paraview-vtu.v1",
    }
    assert all(node["schedule"]["domain"]["type"] == "accepted_step"
               for node in data["nodes"])
    checkpoint, = [node for node in data["nodes"] if node["operation"] is not None]
    assert checkpoint["operation"]["provider_id"] == "pops.restart.accepted-state-v3"
    assert checkpoint["operation"]["bit_identical"] is True


def test_amr_aggregate_accepts_an_external_hierarchy_protocol_by_identity():
    example = _example()
    target = example.build_final_case()
    authored = target.layout

    class ExternalHierarchy:
        __pops_ir_immutable__ = True

        def __init__(self, inner):
            self.ratios = inner.ratios
            self.max_levels = inner.max_levels

        def to_data(self):
            return {
                "schema_version": 1,
                "authority_type": "external_hierarchy_proof",
                "max_levels": self.max_levels,
                "ratios": list(self.ratios),
                "provider": "tests.external.hierarchy.v1",
            }

    from pops.layouts import AMR

    extended = AMR(
        grid=authored.grid,
        hierarchy=ExternalHierarchy(authored.hierarchy),
        tagging=authored.tagging,
        regrid=authored.regrid,
        transfer=authored.transfer,
        execution=authored.execution,
    )
    assert extended.available().ok
    ratios = extended.capabilities().get("transition_ratios")
    assert ratios == [2]
    assert len(ratios) == extended.hierarchy.max_levels - 1


def test_transfer_registry_accepts_an_external_policy_protocol():
    example = _example()
    core = example.build_authoring()
    from pops.lib.amr import StateTransfer
    from pops.amr import AMRTransfer

    built_in = StateTransfer()

    class ExternalStateTransfer:
        __pops_ir_immutable__ = True
        prolongation = built_in.prolongation
        restriction = built_in.restriction
        coarse_fine = built_in.coarse_fine
        time_interpolation = built_in.time_interpolation

        def amr_transfer_policy_data(self):
            return {
                "schema_version": 1,
                "authority_type": "amr_transfer_policy",
                "policy_kind": "state",
                "provider": "tests.external.state-transfer.v1",
                "routes": {
                    name: getattr(self, name).amr_transfer_kernel_data()
                    for name in (
                        "prolongation", "restriction", "coarse_fine", "time_interpolation"
                    )
                },
            }

    registry = AMRTransfer()
    registry.state(core.tracer_state, ExternalStateTransfer())
    policy = registry.inspect()["states"][0]["policy"]
    assert policy["provider"] == "tests.external.state-transfer.v1"


def test_boolean_tagging_composition_lowers_through_node_protocols():
    example = _example()
    target = example.build_final_case()
    core, authored = target.authoring, target.layout
    from pops.amr import AMRTagging, Buffer, Coarsen, Tag
    from pops.math import ValueExpr
    from pops.layouts import AMR
    from pops.math import grad, norm
    from pops.mesh._amr import AnyOf

    value = ValueExpr(core.tracer_state)
    predicate = (
        (value > core.case.value(core.refine_value))
        | (norm(grad(value)) > core.case.value(core.refine_gradient))
    )
    tagging = AMRTagging(
        rules=(
            Tag(predicate),
            Coarsen(value < core.case.value(core.coarsen_value)),
            Buffer(cells=2),
        ),
        hysteresis=authored.tagging.hysteresis,
        conflict_policy=authored.tagging.conflict_policy,
    )
    layout = AMR(
        grid=authored.grid, hierarchy=authored.hierarchy, tagging=tagging,
        regrid=authored.regrid, transfer=authored.transfer, execution=authored.execution,
    )
    resolved = pops.resolve(pops.validate(core.case), layout=layout)
    assert type(resolved.bootstrap_plan.tagging.graph.refine) is AnyOf
