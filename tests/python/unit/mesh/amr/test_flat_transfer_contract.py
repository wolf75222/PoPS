"""A flat adaptive hierarchy retains bootstrap authority without fabricated transfers."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from pops.initial import InitialConditionPlanBuilder, InitialConditionSource
from pops.layouts import Uniform
from pops.lib.amr import ConservativeLinear, StateTransfer
from pops.mesh import LayoutPlanBuilder
from pops.mesh._amr import BootstrapOrdering, BootstrapSelection, ProlongFromParent, resolve_bootstrap
from pops.mesh._amr.bootstrap import _physical_initial_subjects
from pops.mesh._amr.transfer import AMRTransfer, AMRTransferBuilder, PROLONGATION, ResolvedAMRTransfer
from pops.model import Handle, OwnerPath
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout
from tests.python.unit.mesh.amr.test_transfer_bootstrap import OWNER, _hierarchy, _tagging


def _layout(*, levels=1, uniform=False, cells=8):
    states = tuple(Handle(name, kind="state", owner=OwnerPath.model("flat")) for name in ("U", "V"))
    builder = LayoutPlanBuilder(OWNER)
    grid = cartesian_grid(n=cells)
    descriptor = Uniform(grid) if uniform else final_amr_layout(grid, max_levels=levels)
    layout = builder.layout("adaptive", descriptor)
    for state in states:
        builder.assign_state(state, layout)
    return builder.resolve(states=states), layout, states


def _transfer(plan, states):
    authored = AMRTransfer()
    for state in states:
        authored.state(state, StateTransfer())
    return authored.resolve(plan)


def _bootstrap(plan, transfer, states, *, initial_states=None, refined=False):
    hierarchy = _hierarchy(transfer)
    if not refined:
        hierarchy = replace(hierarchy, plan=replace(hierarchy.plan, transitions=()))
    initial = InitialConditionPlanBuilder(plan, states if initial_states is None else initial_states)
    for state in states if initial_states is None else initial_states:
        initial.add(state, InitialConditionSource(Handle(
            "initial_" + state.local_id, kind="initial_condition_provider", owner=OWNER)))
    return resolve_bootstrap(
        layout_plan=plan, hierarchy=hierarchy, transfers=transfer,
        initial_conditions=initial.resolve(), tagging=_tagging(states[0]),
        selections=tuple(BootstrapSelection(state, ProlongFromParent()) for state in states),
        ordering=BootstrapOrdering(("transfer", "projection", "constraint")),
    )


def test_flat_adaptive_transfer_keeps_exact_subjects_and_only_level_zero_initialization():
    plan, layout, states = _layout()
    transfer = _transfer(plan, states)
    assert plan.normalized(layout).adaptive
    assert plan.normalized(layout).transition_ratios == ()
    assert transfer.entries == transfer.requirement_manifest == ()
    assert transfer.nesting_requirement.minimum_buffer == (0, 0)
    assert transfer.nesting_requirement.minimum_lookahead == 0
    assert _physical_initial_subjects(transfer) == states
    assert transfer.flat_layout_plan is plan
    assert transfer.canonical_identity()["schema_version"] == 2
    assert transfer.identity == _transfer(plan, states).identity
    with pytest.raises(KeyError):
        transfer.for_subject(states[0], PROLONGATION)
    bootstrap = _bootstrap(plan, transfer, states)
    assert {(row.level, row.operation, row.subject_id) for row in bootstrap.actions} == {
        (0, "initialize_level_zero", state.qualified_id) for state in states
    }
    assert bootstrap.transfer_identity == transfer.identity


def test_flat_bootstrap_still_requires_exact_initial_subject_coverage():
    plan, _, states = _layout()
    transfer = _transfer(plan, states)
    with pytest.raises(ValueError, match="exactly cover physical AMR transfer subjects"):
        _bootstrap(plan, transfer, states, initial_states=states[:1])
    with pytest.raises(ValueError, match="cannot introduce hierarchy transitions"):
        _bootstrap(plan, transfer, states, refined=True)


def test_uniform_layout_cannot_authorize_empty_amr_transfers():
    plan, _, states = _layout(uniform=True)
    with pytest.raises(ValueError, match="adaptive layout"):
        _transfer(plan, states)


@pytest.mark.parametrize("mutation", ["level_count", "missing_transition", "dimension"])
def test_flat_layout_normalization_rejects_inconsistent_capability_evidence(mutation):
    plan, layout, _ = _layout(levels=2 if mutation == "missing_transition" else 1)
    normalized = plan.normalized(layout)
    if mutation == "missing_transition":
        kwargs = {"transition_ratios": ()}
        message = "one exact-rank row per transition"
    else:
        capabilities = dict(normalized.capabilities)
        capabilities["max_levels" if mutation == "level_count" else "dim"] = 2 if mutation == "level_count" else 3
        kwargs = {"capabilities": capabilities}
        message = "level count differs|dimension differs"
    with pytest.raises(ValueError, match=message):
        replace(normalized, **kwargs)


def test_flat_explicit_layout_does_not_admit_a_foreign_physical_subject():
    plan, layout, _ = _layout()
    foreign = Handle("U", kind="state", owner=OwnerPath.model("foreign"))
    authored = AMRTransfer()
    authored.state(foreign, StateTransfer(), layout=layout)
    with pytest.raises(KeyError, match="no exact state layout assignment"):
        authored.resolve(plan)


def test_flat_plan_still_requires_a_matching_resolved_state_method():
    plan, _, states = _layout()
    authored = AMRTransfer()
    authored.state(states[0], StateTransfer())
    with pytest.raises(ValueError, match="no exact resolved spatial method"):
        authored.resolve(plan, (SimpleNamespace(rates=()),))


def test_flat_plan_rejects_incompatible_provider_dimension_and_unknown_policy():
    class ThreeDimensionalOnly(ConservativeLinear):
        dimensions = (3,)

    plan, _, states = _layout()
    authored = AMRTransfer()
    authored.state(states[0], StateTransfer(prolongation=ThreeDimensionalOnly()))
    with pytest.raises(ValueError, match="provider does not support the layout dimension"):
        authored.resolve(plan)
    with pytest.raises(TypeError, match="amr_transfer_policy_data"):
        AMRTransfer().state(states[0], object())


def test_empty_manifest_needs_explicit_flat_layout_and_physical_authority():
    plan, _, states = _layout()
    transfer = _transfer(plan, states)
    with pytest.raises(TypeError, match="non-empty exact requirement manifest"):
        ResolvedAMRTransfer(plan.qualified_id, (), (), transfer.nesting_requirement)
    with pytest.raises(ValueError, match="explicit layout authority"):
        replace(transfer, flat_layout_plan=None)
    with pytest.raises(ValueError, match="physical subjects"):
        replace(transfer, flat_physical_subjects=())
    with pytest.raises(ValueError, match="duplicate flat AMR physical subject"):
        replace(transfer, flat_physical_subjects=(states[0], states[0]))
    with pytest.raises(ValueError, match="exact LayoutPlan authority"):
        replace(transfer, layout_plan_id="foreign")
    with pytest.raises(ValueError, match="explicit requirement manifest"):
        AMRTransferBuilder(plan).resolve()


def test_flat_layout_authority_is_immutable_and_content_authenticates_geometry():
    plan, _, states = _layout()
    other, _, _ = _layout(cells=16)
    assert plan.qualified_id != other.qualified_id
    with pytest.raises(ValueError, match="does not authenticate its complete payload"):
        replace(other, canonical_id=plan.canonical_id)
    with pytest.raises(FrozenInstanceError):
        plan.canonical_id = other.canonical_id
    transfer = _transfer(plan, states)
    with pytest.raises(ValueError, match="exact LayoutPlan authority"):
        replace(transfer, flat_layout_plan=other)


def test_refined_transfer_canonical_schema_and_requirements_are_unchanged():
    plan, _, states = _layout(levels=2)
    transfer = _transfer(plan, states)
    assert transfer.entries and transfer.requirement_manifest
    assert transfer.flat_layout_plan is None
    assert transfer.flat_physical_subjects == ()
    data = transfer.canonical_identity()
    assert data["schema_version"] == 1
    assert set(data) == {"schema_version", "layout_plan_id", "requirement_manifest", "entries", "nesting_requirement"}
    assert all(row.accuracy.refinement_ratio == (2, 2)
               for entry in transfer.entries for row in entry.requirements)


def test_public_flat_hierarchy_resolves_without_coarse_fine_authority():
    import pops
    from pops.solvers import CompositeTensorFAC
    from tests.python.unit.codegen.test_composite_tensor_fac_provider import (
        _public_amr_hierarchy_case,
    )

    case, layout, _ = _public_amr_hierarchy_case(
        CompositeTensorFAC(), max_levels=1, temporal_ratios=())
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert plan.resolved_hierarchy.plan.transitions == ()
    assert plan.amr_transfer.entries == ()
    assert {subject.qualified_id for subject in plan.amr_transfer.flat_physical_subjects} == {
        state for block in plan.blocks for state in block.state_identities
    }
    from pops.runtime._runtime_executor import _adaptive_initial_location

    install = SimpleNamespace(amr_transfer=plan.amr_transfer,
                              artifact=SimpleNamespace(layout_plan=plan.layout_plan),
                              resolved_hierarchy=plan.resolved_hierarchy)
    for subject in plan.amr_transfer.flat_physical_subjects:
        assert _adaptive_initial_location(install, subject, {}) == ("cell", "cell")
    foreign = Handle("U", kind="state", owner=OwnerPath.model("foreign"))
    with pytest.raises(ValueError, match="outside its physical authority"):
        _adaptive_initial_location(install, foreign, {})
    with pytest.raises(ValueError, match="zero-transition layout"):
        _adaptive_initial_location(
            SimpleNamespace(amr_transfer=plan.amr_transfer, artifact=install.artifact,
                            resolved_hierarchy=SimpleNamespace(plan=SimpleNamespace(transitions=(1,)))),
            plan.amr_transfer.flat_physical_subjects[0], {},
        )
