"""ADC-673: LayoutPlan is the exact resolve authority and fails closed before artifacts."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

import pops
from pops.codegen._layout_resolution import LayoutCapabilityError
from pops.mesh import CartesianMesh, LayoutPlanBuilder
from pops.mesh.layout_plan import LayoutMappingRequirement
from pops.mesh.layouts import Uniform
from pops.model import Module


@dataclass(frozen=True)
class _MappingProvider:
    qualified_id: str
    routes: frozenset[tuple[str, str]]

    def canonical_identity(self):
        return {"qualified_id": self.qualified_id, "routes": sorted(self.routes)}

    def supports_layout_mapping(self, requirement: LayoutMappingRequirement) -> bool:
        return (requirement.source.qualified_id,
                requirement.target.qualified_id) in self.routes


def _problem(name="layout-pipeline"):
    first = Module("layout-first")
    first_space = first.state_space("U", ("u",))
    second = Module("layout-second")
    second_space = second.state_space("V", ("v",))
    problem = pops.Problem(name=name)
    problem.add_block("first", first)
    problem.add_block("second", second)
    problem.program(pops.Program("layout-time"))
    pops.validate(problem)
    return problem, first.state_handle(first_space), second.state_handle(second_space)


def _heterogeneous_plan(problem):
    subjects = problem._materialized_layout_subjects()
    by_block = {value.local_id: value for value in subjects["blocks"]}
    by_state = {value.block_ref.local_id: value for value in subjects["states"]}
    builder = LayoutPlanBuilder(problem.owner_path.canonical())
    coarse = builder.layout("coarse", Uniform(CartesianMesh(n=8)))
    fine = builder.layout("fine", Uniform(CartesianMesh(n=16)))
    builder.assign_block(by_block["first"], coarse)
    builder.assign_state(by_state["first"], coarse)
    builder.assign_block(by_block["second"], fine)
    builder.assign_state(by_state["second"], fine)
    forward, reverse = builder.require_mapping(
        coarse, fine, channel="state", reverse=True)
    providers = (
        _MappingProvider("provider/coarse-to-fine", frozenset((
            (forward.source.qualified_id, forward.target.qualified_id),))),
        _MappingProvider("provider/fine-to-coarse", frozenset((
            (reverse.source.qualified_id, reverse.target.qualified_id),))),
    )
    return builder.resolve(**subjects, providers=providers)


def test_single_descriptor_is_normalized_through_the_layout_plan_pipeline():
    problem, _, _ = _problem("single-normalization")
    resolved = pops.resolve(problem, layout=Uniform(CartesianMesh(n=8)))

    assert len(resolved.layout_plan.layouts) == 1
    assert len(resolved.layout_plan.assignments) == 4
    resolved.layout_plan.validate_subjects(**problem._materialized_layout_subjects())
    assert resolved.capabilities["layout_plan"]["layouts"][0]["capabilities"]["layout"] \
        == "uniform"
    assert resolved.lowering_coverage.to_data()["rows"]


def test_explicit_single_plan_uses_the_same_path_with_a_separate_authenticated_provider():
    problem, _, _ = _problem("single-plan-provider")
    subjects = problem._materialized_layout_subjects()
    descriptor = Uniform(CartesianMesh(n=8))
    builder = LayoutPlanBuilder(problem.owner_path.canonical())
    layout = builder.layout("only", descriptor)
    for block in subjects["blocks"]:
        builder.assign_block(block, layout)
    for state in subjects["states"]:
        builder.assign_state(state, layout)
    plan = builder.resolve(**subjects)

    resolved = pops.resolve(
        problem, layout=plan, layout_providers={layout: Uniform(CartesianMesh(n=8))})

    assert resolved.layout_plan == plan
    assert resolved.layout is not descriptor
    assert resolved.layout.options() == descriptor.options()


def test_pipeline_rejects_an_unassigned_materialized_state_before_artifact_creation():
    problem, _, _ = _problem("unassigned-pipeline")
    subjects = problem._materialized_layout_subjects()
    builder = LayoutPlanBuilder(problem.owner_path.canonical())
    layout = builder.layout("only", Uniform(CartesianMesh(n=8)))
    for block in subjects["blocks"]:
        builder.assign_block(block, layout)
    builder.assign_state(subjects["states"][0], layout)
    incomplete = builder.resolve(
        blocks=subjects["blocks"], states=subjects["states"][:1])

    with pytest.raises(ValueError, match="unassigned layout subjects"):
        pops.resolve(problem, layout=incomplete)


def test_heterogeneous_plan_is_proved_then_refused_with_capability_and_coverage_evidence():
    problem, _, _ = _problem("multi-runtime-gate")
    plan = _heterogeneous_plan(problem)

    with pytest.raises(LayoutCapabilityError) as caught:
        pops.resolve(problem, layout=plan)

    error = caught.value
    assert error.evidence["gate"] == "multi_layout_runtime_unavailable"
    assert len(error.evidence["capabilities"]["layouts"]) == 2
    assert len(error.evidence["resources"]) == 2
    rejected = [row for row in error.coverage_report.rows
                if row.disposition == "rejected"]
    assert [row.gate for row in rejected] == ["multi_layout_runtime_unavailable"]


def test_adding_an_independent_layout_does_not_change_existing_assignment_capabilities():
    problem, _, _ = _problem("independent-capabilities")
    multi = _heterogeneous_plan(problem)
    subjects = problem._materialized_layout_subjects()
    first_block = next(value for value in subjects["blocks"] if value.local_id == "first")
    first_state = next(value for value in subjects["states"]
                       if value.block_ref.local_id == "first")
    builder = LayoutPlanBuilder(problem.owner_path.canonical())
    layout = builder.layout("coarse", Uniform(CartesianMesh(n=8)))
    builder.assign_block(first_block, layout)
    builder.assign_state(first_state, layout)
    single = builder.resolve(blocks=(first_block,), states=(first_state,))

    def assignment_caps(plan, subject):
        return next(row["capabilities"] for row in plan.capability_evidence()["assignments"]
                    if row["subject"]["qualified_id"] == subject.qualified_id)

    assert assignment_caps(single, first_state) == assignment_caps(multi, first_state)
