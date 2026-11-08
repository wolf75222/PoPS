"""Authored two-trace equations and both AMR towers must reach native codegen."""
import pops
import pytest
from examples.api_v02.method_relaxed_multiblock import build_t10_multiblock, transport_trace_flux
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize("ratio", (1, 2))
def test_two_trace_affine_program_covers_both_owned_towers(ratio):
    flux = transport_trace_flux()
    spec = build_t10_multiblock(flux, temporal_ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout, components=(flux,))
    cpp = emit_cpp_program(plan.time, target="amr_system",
                          model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "ctx.begin_affine_finalization(dt, {0, 1})" in cpp
    assert "ctx.accept_increment" in cpp
    assert len(plan.blocks) == 2
    assert "PyObject" not in cpp


def _t10_with_patch_layout(spec, patch_layout):
    authored = spec.layout
    return type(authored)(
        grid=authored.grid, hierarchy=authored.hierarchy, tagging=authored.tagging,
        regrid=authored.regrid, transfer=authored.transfer, execution=authored.execution,
        patch_layout=patch_layout, load_balance=authored.load_balance,
        tagger=authored.tagger, clustering=authored.clustering, reflux=authored.reflux)


def test_plan_partitions_the_coarse_level_of_a_cross_block_interface():
    # Spec 0.3 gap C50 (DM3, PF3): T10 writes no storage layout, and the plan gives its two-block
    # hierarchy one owner per coarse patch, which the native interface scheduler requires on two
    # ranks or more. An explicit True stays a satisfied constraint, with its tile size.
    from pops.amr import PatchLayout

    flux = transport_trace_flux()
    spec = build_t10_multiblock(flux)
    assert spec.layout.patch_layout == PatchLayout()
    for patch_layout, tile in ((None, None), (PatchLayout(distribute_coarse=True,
                                                          coarse_max_grid=16), 16)):
        layout = spec.layout if patch_layout is None else _t10_with_patch_layout(spec, patch_layout)
        plan = pops.resolve(pops.validate(spec.case), layout=layout, components=(flux,))
        assert plan.resolved_hierarchy.plan.patch_generation.options.to_data() == {
            "native_route": "box_array",
            "distribute_coarse": True,
            "coarse_max_grid": tile,
        }


def test_replicated_coarse_constraint_on_a_cross_block_interface_is_schedule_illegal():
    from pops._ir.refusals import ScheduleIllegal
    from pops.amr import PatchLayout

    flux = transport_trace_flux()
    spec = build_t10_multiblock(flux)
    layout = _t10_with_patch_layout(spec, PatchLayout(distribute_coarse=False))
    plan = pops.resolve(pops.validate(spec.case), layout=layout, components=(flux,))
    hierarchy = next(iter(plan.layout_amr_authorities.values())).authorities.hierarchy
    assert hierarchy.plan.patch_generation.options.to_data()["distribute_coarse"] is False
    # Refused by pops.compile before any native work, whatever the rank count (C50, P3).
    with pytest.raises(ScheduleIllegal, match="one owner per patch") as refusal:
        pops.compile(plan)
    assert refusal.value.rule == "IR22" and refusal.value.phase == "compile"
    assert refusal.value.constraint == "PatchLayout(distribute_coarse=False)"
