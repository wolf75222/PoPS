"""Resolved authority for field problems executed by explicit Program solve nodes.

These plans authenticate Program scratch storage and its physical equation mapping. They do not
install a named legacy field provider or perform a hidden bind-time solve.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pops.codegen.lowering_coverage import LoweringCoverageReport, LoweringCoverageRow
from pops.fields._identity import field_identity
from pops.identity import Identity, canonical_bytes


def _canonical(value: Any) -> bytes:
    from pops.time._graph.base import strict_data

    return canonical_bytes(strict_data(value, where="Program field source mapping"))


def _nodes(program: Any) -> tuple[Any, ...]:
    from pops.codegen.program_emit_field_routes import _walk_program_nodes

    return tuple(_walk_program_nodes(tuple(program._values)))


def _physical_metadata(node: Any) -> Mapping:
    request = node.attrs.get("solve_request", {})
    if not isinstance(request, Mapping):
        return {}
    metadata = request.get("physical_problem", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _solve_nodes(program: Any, handle: Any) -> tuple[Any, ...]:
    expected = _canonical(handle.canonical_identity())
    return tuple(node for node in _nodes(program)
                 if node.op in ("solve_linear", "solve_spatial_field")
                 and _canonical(_physical_metadata(node).get("field_handle")) == expected)


def _reachable(node: Any, all_nodes: tuple[Any, ...]) -> tuple[Any, ...]:
    """Follow actual data inputs and nested apply blocks, never unrelated metadata witnesses."""
    by_id = {item.id: item for item in all_nodes}
    result = {}

    def visit(value: Any) -> None:
        if not hasattr(value, "id") or value.id in result:
            return
        value = by_id.get(value.id, value)
        result[value.id] = value
        for source in value.inputs:
            visit(source)
        for key in ("apply_block", "residual_block", "body_block"):
            for child in value.attrs.get(key, ()):
                visit(child)

    visit(node)
    return tuple(result.values())


@dataclass(frozen=True, slots=True)
class ResolvedProgramFieldPlan:
    """One registered equation, field-owned layout, and authenticated native solve graph."""

    name: str
    handle: Any
    operator: Any
    discretization: Any
    storage: Any
    target: str
    solve_node_ids: tuple[int, ...]
    coverage: LoweringCoverageReport
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        from pops.fields.operator import FieldOperator
        from pops.fields.problem import FieldProblem, FieldStorageBinding
        from pops.fields.discretization import require_field_discretization
        from pops.model import Handle, OwnerKind

        if not isinstance(self.operator, FieldProblem) or isinstance(self.operator, FieldOperator):
            raise TypeError("Program field plan requires a generic physical FieldProblem")
        if self.name != self.operator.name or self.name != self.handle.local_id:
            raise ValueError("Program field plan name disagrees with its registered equation")
        if (not isinstance(self.handle, Handle) or not self.handle.is_resolved
                or self.handle.kind != "field" or len(self.handle.owner_path.nodes) != 1
                or self.handle.owner_path.nodes[0].kind is not OwnerKind.CASE):
            raise TypeError("Program field plan requires an exact canonical Case field identity")
        require_field_discretization(self.discretization, where="Program field discretization")
        if type(self.storage) is not FieldStorageBinding:
            raise TypeError("Program field plan requires exact field-owned storage")
        if self.storage.unknowns != self.operator.unknowns:
            raise ValueError("Program field storage differs from the physical unknown tuple")
        if not all(item.is_resolved and item.block_ref is None for item in self.storage.unknowns):
            raise ValueError("Program field storage cannot be owned by a contributing species block")
        if self.target not in ("system", "amr_system"):
            raise NotImplementedError("generic Program field storage has no native target realization")
        if not self.solve_node_ids or any(type(item) is not int or item < 0
                                         for item in self.solve_node_ids):
            raise ValueError("Program field plan requires actual native solve node identities")
        if len(set(self.solve_node_ids)) != len(self.solve_node_ids):
            raise ValueError("Program field plan repeats a native solve node")
        if type(self.coverage) is not LoweringCoverageReport:
            raise TypeError("Program field plan requires exact lowering coverage")
        expected = field_identity("resolved-program-field", self.to_data(False))
        if hasattr(self, "identity") and self.identity != expected:
            raise ValueError("Program field plan identity changed after resolution")
        object.__setattr__(self, "identity", expected)

    @property
    def storage_identity(self) -> Identity:
        return self.storage.identity

    def to_data(self, include_identity: bool = True) -> dict[str, Any]:
        from pops.fields.discretization import field_discretization_data

        data = {
            "schema_version": 1,
            "name": self.name,
            "field_handle": self.handle.canonical_identity(),
            "operator": self.operator.to_data(),
            "discretization": field_discretization_data(
                self.discretization, where="Program field discretization"),
            "storage": self.storage.to_data(),
            "target": self.target,
            "solve_node_ids": list(self.solve_node_ids),
            "coverage": self.coverage.to_data(),
        }
        if include_identity:
            data["identity"] = self.identity.token
        return data

    def validate_program(self, program: Any) -> None:
        self.__post_init__()
        solves = _solve_nodes(program, self.handle)
        if tuple(node.id for node in solves) != self.solve_node_ids:
            raise ValueError("Program field solve graph changed after resolution")
        expected_problem = _canonical(self.operator.to_data())
        expected_dependencies = {
            _canonical(item.canonical_identity()) for item in self.operator.dependencies()
        }
        all_nodes = _nodes(program)
        for solve in solves:
            if self.target == "amr_system" and solve.attrs.get("scope") != "hierarchy" and solve.op != "solve_spatial_field":
                raise ValueError("AMR field problems require an explicit synchronized hierarchy solver")
            if self.target == "system" and solve.attrs.get("scope") == "hierarchy":
                raise ValueError("a hierarchy field solver requires an AMR layout")
            metadata = _physical_metadata(solve)
            if _canonical(metadata.get("field_problem")) != expected_problem:
                raise ValueError("Program field solve changed its registered physical equations")
            if solve.op == "solve_spatial_field":
                from pops.fields._program_nonlinear_problem import (
                    compile_equations, validate_nonlinear_field_request,
                )
                from pops.fields.methods import CellCenteredNonlinearCoupled
                from pops.fields._program_problem import _physical_boundary
                from pops.identity.scalar import scalar_data
                validate_nonlinear_field_request(program, solve)
                if solve.attrs.get("right_preconditioner") is not None and self.target != "amr_system":
                    name = "FullResidualBasisLU@1" if solve.attrs["right_preconditioner"] == "pops.amr.full-residual-basis-lu@1" else "SpatialBasisJacobi@1"
                    raise ValueError(f"{name} requires the original composite AMR FieldProblem provider; Uniform is unsupported")
                if self.target not in ("system", "amr_system") or type(self.discretization.method) is not CellCenteredNonlinearCoupled:
                    raise ValueError("original mixed residual requires its explicit native nonlinear method")
                captures = solve.inputs[2:2 + solve.attrs["capture_count"]]
                method_data = self.discretization.method.options()
                diffusion, local = compile_equations(self.operator, captures, per_candidate=method_data.get("coefficient_evaluation") is not None)
                from pops.fields._original_field_interaction import compile_interactions
                interactions = compile_interactions(self.operator, method_data.get("interaction_realization"))
                if _canonical(interactions) != _canonical(solve.attrs["source_contract"].get("interactions")):
                    raise ValueError("original residual changed its registered interaction physics/realization")
                if solve.attrs["contract"] != method_data["contract"] or \
                        solve.attrs.get("coefficient_face_policy") != method_data.get("coefficient_face_policy") or \
                        solve.attrs.get("coefficient_evaluation") != method_data.get("coefficient_evaluation") or \
                        solve.attrs.get("linear_residual_verification") != method_data.get("linear_residual_verification"):
                    raise ValueError("original field coefficient realization differs from registered method")
                if _canonical(local) != _canonical(solve.attrs["local_expressions"]) or \
                        _canonical(diffusion) != _canonical(solve.attrs["source_contract"]["diffusion"] if method_data.get("coefficient_evaluation") else solve.inputs[1].attrs["expressions"]) or \
                        _canonical(solve.attrs["finite_difference_step"]) != _canonical(scalar_data(self.discretization.method.finite_difference_step)) or \
                        solve.attrs["physical_boundary"] != _physical_boundary(self.operator):
                    raise ValueError("native residual changed its registered equations/method/boundaries")
                from pops.fields._evolved_stage_contract import stage_projection, compile_accumulation
                projection = stage_projection(self.operator, program, solve.point, authoring=False)
                source = solve.attrs["source_contract"]
                if projection is not None:
                    if _canonical(source.get("evolved_stage")) != _canonical(projection.to_data()) or \
                            _canonical(source.get("temporal_tau")) != _canonical(projection.tau.to_data()) or \
                            _canonical(source.get("accumulation")) != _canonical(compile_accumulation(projection, captures, self.operator.unknowns)):
                        raise ValueError("original evolved accumulation changed its registered declaration")
                elif any(key in source for key in ("temporal_tau", "evolved_stage", "accumulation")):
                    raise ValueError("original field solve invents an undeclared evolved accumulation")
                from pops.time.references import canonical_handle
                if {_canonical(canonical_handle(value.state_ref).canonical_identity()) for value in captures} != expected_dependencies:
                    raise ValueError("native residual captures differ from the physical dependencies")
                continue
            reachable = _reachable(solve, all_nodes)
            operations = tuple(node for node in reachable if node.op in (
                "field_problem_load", "field_problem_coefficients", "field_problem_apply"))
            if not {"field_problem_load", "field_problem_apply"}.issubset(
                    {node.op for node in operations}):
                raise ValueError("Program field metadata has no executable load and apply graph")
            seen_dependencies = set()
            for node in operations:
                if node.attrs.get("field_problem_identity") != self.operator.identity.token:
                    raise ValueError("Program field native operation belongs to a different equation")
                dependencies = node.attrs.get("field_dependencies", ())
                for dependency in dependencies:
                    data = (dependency.canonical_identity()
                            if callable(getattr(dependency, "canonical_identity", None)) else dependency)
                    seen_dependencies.add(_canonical(data))
            if seen_dependencies != expected_dependencies:
                raise ValueError("Program field native input reads differ from its physical dependencies")


def capture_program_field_plans(problem: Any, detach: Any, *, target: str,
                                layout_plan: Any, program: Any) -> dict[str, ResolvedProgramFieldPlan]:
    """Capture only exact generic registrations with actual Program solve consumers."""
    from pops.fields.operator import FieldOperator
    from pops.fields.problem import FieldStorageBinding

    result = {}
    for name, registration in problem._field_registry.resolved_items(problem.resolve):
        if isinstance(registration.operator, FieldOperator):
            continue
        handle = problem.resolve(problem._field_registry.handle(name))
        solves = _solve_nodes(program, handle)
        targets = tuple("program:%s:%d" % (node.op, node.id) for node in solves)
        if not targets:
            raise ValueError("generic field %r requires an explicit Program solve" % name)
        storage = FieldStorageBinding(registration.operator.unknowns, layout_plan.layout_for(handle))
        if target == "amr_system":
            resolved_layout = next(row for row in layout_plan.layouts if row.handle == storage.layout)
            if resolved_layout.capabilities.get("execution") != "synchronous":
                raise ValueError("general field hierarchy coupling requires explicit synchronous AMR execution; asynchronous or subcycled policies have no declared field-time transfer")
        registration = detach(registration)
        coverage = LoweringCoverageReport((LoweringCoverageRow(
            "program-field:%s" % handle.qualified_id, "lowered", targets),))
        plan = ResolvedProgramFieldPlan(
            name, handle, registration.operator, registration.discretization, storage,
            target, tuple(node.id for node in solves), coverage)
        plan.validate_program(program)
        result[name] = plan
    return result


__all__ = ["ResolvedProgramFieldPlan", "capture_program_field_plans"]
