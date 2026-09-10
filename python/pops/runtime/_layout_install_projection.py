"""Immutable child install projection under one exact aggregate InstallPlan authority."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class LayoutCompiledPlanProjection:
    parent: Any
    local: Any
    blocks: tuple
    field_plans: Any

    @property
    def layout_plan(self):
        return self.local.layout_plan

    def verify(self):
        self.parent.verify()
        if self.parent.layout_amr_authorities.get(self.local.layout_id) is not self.local:
            raise ValueError("child compiled plan lost its parent AMR authority")

    def __getattr__(self, name):
        return getattr(self.parent, name)


@dataclass(frozen=True, slots=True)
class LayoutCompiledArtifactProjection:
    parent: Any
    selected: Any
    plan: LayoutCompiledPlanProjection
    blocks: tuple

    @property
    def program(self):
        return self.selected.program

    @property
    def program_param_routes(self):
        return self.selected.program.program_param_routes

    @property
    def program_block_routes(self):
        return self.selected.program.program_block_routes

    @property
    def so_path(self):
        return self.selected.program.so_path

    @property
    def target(self):
        return self.selected.target

    @property
    def layout_plan(self):
        return self.plan.layout_plan

    @property
    def layout_programs(self):
        return (self.selected,)

    def arguments(self):
        from pops.codegen.inspect_compiled import build_layout_arguments
        return build_layout_arguments(self.parent, self.selected.layout_id)

    def __getattr__(self, name):
        return getattr(self.parent, name)


@dataclass(frozen=True, slots=True)
class LayoutInstallProjection:
    """Only a registered compiled slice and registered local authority can form a child."""
    parent: Any
    selected: Any
    local: Any
    artifact: LayoutCompiledArtifactProjection = field(init=False)
    instances: Any = field(init=False)
    initial_values: Any = field(init=False)
    layout: Any = field(init=False)

    def __post_init__(self):
        from pops.codegen._plans import require_install_plan
        from pops.codegen._compiled_artifact import CompiledLayoutProgram
        from pops.codegen._layout_amr_authorities import ResolvedLayoutAMRAuthorities
        parent = require_install_plan(self.parent)
        if type(self.selected) is not CompiledLayoutProgram or not any(
                row is self.selected for row in parent.artifact.layout_programs):
            raise TypeError("child install requires an exact registered CompiledLayoutProgram")
        if type(self.local) is not ResolvedLayoutAMRAuthorities \
                or parent.layout_amr_authorities.get(self.selected.layout_id) is not self.local:
            raise TypeError("child install requires its parent's exact registered AMR authority")
        names = self.selected.block_names
        if not names or len(set(names)) != len(names):
            raise ValueError("child install needs a nonempty unique compiled block set")
        assigned = {row.subject.local_id for row in self.local.layout_plan.assignments
                    if row.subject_kind == "block"}
        if set(names) != assigned:
            raise ValueError("child compiled block set differs from its exact layout assignments")
        if self.selected.target != "amr_system":
            raise ValueError("child AMR authority requires an AMR compiled Program")
        # FieldOperator projections need their own layout assignments. Program FieldProblem
        # resources are already part of the exact compiled slice and require no Python owner.
        if parent.artifact.plan.field_plans:
            raise NotImplementedError("child FieldOperator installation needs explicit layout ownership")
        fields = MappingProxyType({})
        plan_blocks = tuple(row for row in parent.artifact.plan.blocks if row.name in assigned)
        blocks = tuple(row for row in parent.artifact.blocks if row.name in assigned)
        artifact = LayoutCompiledArtifactProjection(parent.artifact, self.selected,
            LayoutCompiledPlanProjection(parent.artifact.plan, self.local, plan_blocks, fields), blocks)
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "instances", MappingProxyType(
            {name: parent.instances[name] for name in names}))
        subjects = {row.subject.qualified_id for row in self.initial_condition_plan.bindings}
        object.__setattr__(self, "initial_values", MappingProxyType(
            {handle: value for handle, value in parent.initial_values.items()
             if handle.qualified_id in subjects}))
        handle = self.local.layout_plan.layouts[0].handle
        object.__setattr__(self, "layout", parent.layout.descriptor(handle))
        self.verify()

    @property
    def target(self):
        return self.selected.target

    @property
    def layout_id(self):
        return self.selected.layout_id

    @property
    def resolved_hierarchy(self):
        return self.local.authorities.hierarchy

    @property
    def amr_transfer(self):
        return self.local.authorities.transfer

    @property
    def bootstrap_plan(self):
        return self.local.authorities.bootstrap

    @property
    def initial_condition_plan(self):
        return self.local.authorities.initial_conditions

    @property
    def amr_execution(self):
        return self.local.authorities.execution

    @property
    def amr_providers(self):
        return self.local.authorities.providers

    def verify(self):
        from pops.codegen._plans import require_install_plan
        parent = require_install_plan(self.parent)
        self.artifact.plan.verify()
        if not any(row is self.selected for row in parent.artifact.layout_programs) \
                or parent.layout_amr_authorities.get(self.layout_id) is not self.local:
            raise ValueError("child install authorities no longer belong to the exact parent")
        if tuple(self.instances) != self.selected.block_names or any(
                value is not parent.instances[name] for name, value in self.instances.items()):
            raise ValueError("child instances differ from the exact parent objects")
        if self.artifact.parent is not parent.artifact or self.artifact.selected is not self.selected:
            raise ValueError("child artifact projection changed its registered binary")

    def __getattr__(self, name):
        return getattr(self.parent, name)


def require_install_authority(value):
    if type(value) is LayoutInstallProjection:
        value.verify()
        return value
    from pops.codegen._plans import require_install_plan
    return require_install_plan(value)
