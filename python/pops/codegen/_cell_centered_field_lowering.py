"""Prepared lowering for the native 2-D cell-centred elliptic implementation.

All Cartesian, scalar-operator and System/AMR decisions live here, behind the same open provider
interface available to third parties.  ``field_install`` never branches on these semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, NoReturn

from pops.codegen.field_boundary_lowering import (
    boundary_dependency_pack,
    boundary_plan,
    field_layout_contract,
    topology_recipe,
)
from pops.codegen.lowering_coverage import (
    LoweringCoverageReport,
    LoweringCoverageRow,
    LoweringRejection,
)
from pops.fields._identity import field_identity, strict_field_data
from pops.fields._prepared_field_lowering_registry import (
    PreparedFieldLoweringBinding,
    PreparedFieldLoweringEvidence,
    PreparedFieldLoweringProvider,
    PreparedFieldLoweringRequest,
    PreparedFieldLoweringResolution,
    PreparedFieldRuntimeInstallContext,
    PreparedFieldRuntimePreflightContext,
    register_prepared_field_lowering_provider,
)
from pops.fields._prepared_field_nullspace_registry import PreparedFieldNullspaceFacts
from pops.fields._prepared_field_solver_registry import PreparedFieldSolverFacts
from pops.fields.discretization import CompositeHierarchySolve, LevelByLevelSolve
from pops.fields.operator import FieldOperator, _field_targets_unknown
from pops.fields.solve import ResolvedHierarchyPolicy
from pops.math import principal_kinds


def _reject(
    rows: list[LoweringCoverageRow], source: str, gate: str, message: str,
) -> NoReturn:
    report = LoweringCoverageReport((*rows, LoweringCoverageRow(
        source, "rejected", gate=gate)))
    raise LoweringRejection(
        message, coverage_report=report, source=source, gate=gate)


@dataclass(frozen=True, slots=True)
class _HierarchyPolicyContext:
    inferred: ResolvedHierarchyPolicy

    def inferred_hierarchy_policy(self) -> ResolvedHierarchyPolicy:
        return self.inferred

    def bind_hierarchy_policy(
        self, policy: ResolvedHierarchyPolicy
    ) -> ResolvedHierarchyPolicy:
        if not isinstance(policy, ResolvedHierarchyPolicy):
            raise TypeError("field hierarchy policy returned a foreign authority")
        return policy


def _validate_outputs(request: PreparedFieldLoweringRequest) -> tuple[dict[str, Any], int]:
    from pops.fields.outputs import FieldOutput, GradientOutput

    operator = request.operator
    outputs = tuple(operator.outputs)
    components = request.output_components
    if len(components) not in (1, 3):
        raise TypeError(
            "cell-centred output must be one potential component or potential plus two gradients"
        )
    if len(components) == 1:
        if len(outputs) != 1 or not isinstance(outputs[0], FieldOutput):
            raise TypeError("cell-centred scalar output must be exactly one FieldOutput")
        gradient_sign = 1
    else:
        if (
            len(outputs) != 2
            or not isinstance(outputs[0], FieldOutput)
            or not isinstance(outputs[1], GradientOutput)
        ):
            raise TypeError(
                "cell-centred gradient output must be FieldOutput + GradientOutput"
            )
        gradient_sign = outputs[1].sign
        if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
            raise ValueError("resolved GradientOutput sign must be exactly -1 or 1")
    potential_source = outputs[0].source
    if potential_source is not None and potential_source != operator.unknown:
        raise ValueError("FieldOutput source disagrees with the FieldOperator solved unknown")
    if len(outputs) == 2 and outputs[1].source != operator.unknown:
        raise ValueError("GradientOutput source disagrees with the FieldOperator solved unknown")
    output_block = operator.unknown.block_ref
    if output_block is None:
        raise RuntimeError("resolved field output lost its owner-qualified block")
    return {
        "owner_identity": output_block.canonical_identity(),
        "owner_block": output_block.local_id,
        "key": operator.name,
        "components": components,
        "gradient_sign": gradient_sign,
    }, gradient_sign


def _reaction(
    request: PreparedFieldLoweringRequest,
    rows: list[LoweringCoverageRow],
) -> dict[str, Any] | None:
    operator = request.operator
    name = request.name
    source = "field:%s:operator" % name
    kinds = principal_kinds(operator.equation.lhs)
    if "laplacian" not in kinds or kinds - {"laplacian", "reaction"}:
        _reject(
            rows, source, "field.operator.not_native",
            "field %r principal operator %s has no cell-centred native lowering"
            % (name, sorted(kinds)),
        )
    from pops._ir.elliptic import Reaction, constant_reaction_scalar, elliptic_terms
    from pops._ir.expr import Laplacian
    from pops._ir.values import RuntimeParamRef

    terms = elliptic_terms(operator.equation.lhs)
    laplacians = [term for term in terms if isinstance(term, Laplacian)]
    reactions = [term for term in terms if isinstance(term, Reaction)]
    if len(laplacians) != 1 or len(reactions) > 1 or len(terms) != 1 + len(reactions):
        _reject(
            rows, source, "field.operator.poisson_shape_not_native",
            "field %r requires exactly one Laplacian and at most one scalar reaction term"
            % name,
        )
    laplacian_term = laplacians[0]
    if not _field_targets_unknown(laplacian_term.field, operator.unknown):
        _reject(
            rows, source, "field.operator.unknown_mismatch",
            "field %r Laplacian does not act on its declared unknown" % name,
        )
    normalization = -float(laplacian_term.scale)
    if not math.isfinite(normalization) or normalization == 0.0:
        _reject(
            rows, source, "field.operator.invalid_laplacian_scale",
            "field %r Laplacian scale must be finite and non-zero" % name,
        )
    result = None
    if reactions:
        reaction = reactions[0]
        coefficient = reaction.coeff
        handle = getattr(coefficient, "handle", None)
        multiplier = float(reaction.scale) / normalization
        constant = constant_reaction_scalar(coefficient)
        bind_parameter = (
            isinstance(coefficient, RuntimeParamRef)
            and handle is not None
            and getattr(handle, "kind", None) == "parameter"
            and getattr(handle, "param_kind", None) in ("runtime", "derived")
            and coefficient.dtype == "Real"
        )
        if not _field_targets_unknown(reaction.field, operator.unknown):
            _reject(
                rows, source, "field.operator.reaction_unknown_mismatch",
                "field %r reaction does not act on its declared unknown" % name,
            )
        if constant is NotImplemented and not bind_parameter:
            _reject(
                rows, source, "field.operator.reaction_coefficient_not_native",
                "field %r reaction requires a finite real/ConstParam or one typed Real "
                "RuntimeParam/DerivedParam read" % name,
            )
        if not math.isfinite(multiplier):
            _reject(
                rows, source, "field.operator.reaction_sign_not_native",
                "field %r must normalize to -laplacian(phi) + kappa*phi with kappa > 0"
                % name,
            )
        if constant is NotImplemented:
            if multiplier <= 0.0:
                _reject(
                    rows, source, "field.operator.reaction_sign_not_native",
                    "field %r must normalize to -laplacian(phi) + kappa*phi with kappa > 0"
                    % name,
                )
            if handle is None:
                raise RuntimeError("validated bind-parameter reaction lost its handle")
            result = {
                "schema_version": 1,
                "kind": "scalar_bind_parameter",
                "parameter": handle.canonical_identity(),
                "multiplier": multiplier,
            }
            route = "scalar-bind"
        else:
            try:
                effective = float(constant) * multiplier
            except (TypeError, ValueError, OverflowError):
                effective = float("nan")
            if not math.isfinite(effective) or effective <= 0.0:
                _reject(
                    rows, source, "field.operator.reaction_sign_not_native",
                    "field %r must normalize to -laplacian(phi) + kappa*phi with kappa > 0"
                    % name,
                )
            result = {
                "schema_version": 1,
                "kind": "scalar_constant",
                "value": effective,
            }
            route = "scalar-constant"
        rows.append(LoweringCoverageRow(
            "field:%s:reaction" % name,
            "lowered",
            ("field-install:%s:reaction:%s" % (name, route),),
        ))
    rows.append(LoweringCoverageRow(
        source, "lowered", ("field-install:%s:residual" % name,)
    ))
    return result


def _resolve(
    options: Mapping[str, Any], request: PreparedFieldLoweringRequest, where: str,
) -> PreparedFieldLoweringResolution:
    if options:
        raise TypeError("%s cell-centred second-order method accepts no options" % where)
    if not isinstance(request.operator, FieldOperator):
        raise TypeError("%s requires a FieldOperator" % where)
    plan = request.discretization
    name = request.name
    target = request.target
    if target not in ("system", "amr_system"):
        raise ValueError(
            "%s cell-centred native provider does not implement target %r" % (where, target)
        )
    rows: list[LoweringCoverageRow] = []
    output_route, _ = _validate_outputs(request)
    reaction = _reaction(request, rows)
    rows.append(LoweringCoverageRow(
        "field:%s:method" % name,
        "lowered",
        ("field-install:%s:cell-centered-second-order" % name,),
    ))

    layout_contract = field_layout_contract(request.layout)
    recipe = topology_recipe(request.layout)
    if layout_contract.embedded_boundary is not None:
        _reject(
            rows, "field:%s:topology" % name,
            "field.topology.embedded_boundary_not_native",
            "field %r uses an embedded boundary, but this residual provider has no "
            "material-cell connectivity/mask lowering" % name,
        )
    rows.append(LoweringCoverageRow(
        "field:%s:topology" % name,
        "derived",
        rule="resolved full rectangular cell graph has one connected material component",
    ))

    bc, faces = boundary_plan(
        name, plan, rows, request.layout, request.operator.unknown
    )
    dependencies = boundary_dependency_pack(plan, request.operator.unknown)
    if target == "amr_system" and dependencies["fields"]:
        _reject(
            rows, "field:%s:boundaries" % name,
            "field.boundary.amr_field_dependency_not_native",
            "field %r has a boundary law depending on another solved field; the AMR "
            "provider has no exact composite materialization route" % name,
        )
    if (
        target == "amr_system"
        and layout_contract.levels > 1
        and dependencies["states"]
    ):
        _reject(
            rows, "field:%s:boundaries" % name,
            "field.boundary.amr_multilevel_state_dependency_not_native",
            "field %r has a state-dependent boundary law on a multilevel hierarchy" % name,
        )
    for kind in ("states", "fields"):
        for dependency in dependencies[kind]:
            rows.append(LoweringCoverageRow(
                "field:%s:boundary-dependency:%s:%d" % (
                    name, dependency["qualified_id"], dependency["component"]
                ),
                "lowered",
                ("field-install:%s:boundary-buffer:%s" % (name, kind),),
            ))
    for coordinate in dependencies["logical_time"]:
        rows.append(LoweringCoverageRow(
            "field:%s:boundary-time:%s" % (name, coordinate),
            "lowered",
            ("field-install:%s:logical-timepoint" % name,),
        ))
    boundary_dynamic = faces is not None and any(face["dynamic"] for face in faces)
    boundary_iterate = faces is not None and any(
        face["iterate_dependent"] for face in faces
    )

    inferred_hierarchy = (
        CompositeHierarchySolve().resolved_authority()
        if target == "amr_system"
        else LevelByLevelSolve().resolved_authority()
    )
    resolver = getattr(plan.hierarchy_policy, "resolve", None)
    if not callable(resolver):
        _reject(
            rows, "field:%s:hierarchy" % name, "field.hierarchy.invalid_policy",
            "field %r hierarchy policy does not implement resolve(capabilities)" % name,
        )
    try:
        hierarchy_resolution = resolver(_HierarchyPolicyContext(inferred_hierarchy))
    except (TypeError, ValueError) as exc:
        _reject(
            rows, "field:%s:hierarchy" % name, "field.hierarchy.unsupported", str(exc)
        )
    if type(hierarchy_resolution) is not ResolvedHierarchyPolicy:
        _reject(
            rows, "field:%s:hierarchy" % name,
            "field.hierarchy.invalid_resolution",
            "field %r hierarchy policy returned a foreign resolution" % name,
        )
    hierarchy_authority = hierarchy_resolution.authority()
    policy = hierarchy_resolution.policy_id
    rows.append(LoweringCoverageRow(
        "field:%s:hierarchy" % name,
        "derived",
        rule="%s + provider-target=%s" % (policy, target),
    ))

    if plan.preconditioner is not None:
        _reject(
            rows, "field:%s:preconditioner" % name,
            "field.preconditioner.not_native",
            "field %r declares a preconditioner this spatial provider cannot consume" % name,
        )
    rows.append(LoweringCoverageRow(
        "field:%s:preconditioner" % name, "documentary"
    ))
    if boundary_iterate and plan.nonlinear is None:
        _reject(
            rows, "field:%s:boundaries" % name,
            "field.boundary.nonlinear_outer_solver_required",
            "field %r has iterate-dependent boundary expressions and requires a prepared "
            "nonlinear outer solver" % name,
        )
    dynamic_alpha = faces is not None and any("alpha" in face["dynamic"] for face in faces)
    statically_anchored = faces is not None and any(
        face["type"] != "periodic"
        and "alpha" not in face["dynamic"]
        and face["alpha"] != 0.0
        for face in faces
    )
    if reaction is None and dynamic_alpha and not statically_anchored:
        _reject(
            rows, "field:%s:boundaries" % name,
            "field.boundary.dynamic_nullspace_topology",
            "field %r dynamic Robin alpha can change the nullspace dimension" % name,
        )

    topology_identity = field_identity("resolved-field-topology", recipe).token
    mesh_cells = getattr(layout_contract.mesh, "cells", ())
    if not isinstance(mesh_cells, tuple):
        mesh_cells = tuple(mesh_cells) if isinstance(mesh_cells, (list, tuple)) else ()
    solver_facts = PreparedFieldSolverFacts(
        target=target,
        operator={
            "principal": "scalar-laplacian",
            "screened": reaction is not None,
            "reaction": reaction,
        },
        layout={
            "kind": layout_contract.kind,
            "levels": layout_contract.levels,
            "transition_ratios": layout_contract.transition_ratios,
            "embedded_boundary": False,
            "adaptive": getattr(request.layout, "refine", None) is not None,
            "cells": mesh_cells,
            "topology_identity": topology_identity,
            "topology_recipe": recipe,
        },
        hierarchy=hierarchy_authority,
        boundary={
            "faces": () if faces is None else faces,
            "dynamic": boundary_dynamic,
            "dependent": any(
                dependencies[kind] for kind in ("states", "fields", "logical_time")
            ),
            "iterate_dependent": boundary_iterate,
        },
        nonlinear=plan.nonlinear is not None,
    )
    singular_faces = faces is not None and all(
        face["type"] == "periodic" or face["alpha"] == 0.0 for face in faces
    )
    kernel_components = int(reaction is None and singular_faces)
    nullspace_facts = PreparedFieldNullspaceFacts(
        topology_identity,
        kernel_components,
        {
            "principal": "scalar-laplacian",
            "screened": reaction is not None,
            "boundary_dynamic": boundary_dynamic,
        },
    )
    native_options = {
        "rhs": "composite",
        "rhs_identity": field_identity(
            "field-rhs",
            {"rhs": strict_field_data(request.operator.equation.rhs)},
        ).token,
        "output_route": output_route,
        "method": {
            "native_method": "cell_centered_second_order",
            "order": 2,
            "ghost_depth": 1,
        },
        "reaction": reaction,
        "bc": bc,
        "boundary_faces": faces,
        "boundary_kernel_required": boundary_dynamic,
        "boundary_iterate_dependent": boundary_iterate,
        "boundary_dependencies": dependencies,
        "hierarchy_policy": hierarchy_authority,
        "topology_recipe": recipe,
    }
    rows.append(LoweringCoverageRow(
        "field:%s:output" % name,
        "lowered",
        ("field-install:%s:output:%s:%s" % (
            name, output_route["owner_block"], request.operator.unknown.qualified_id
        ),),
    ))
    evidence = tuple(
        PreparedFieldLoweringEvidence.from_data(row.to_data()) for row in rows
    )
    return PreparedFieldLoweringResolution(
        native_options, solver_facts, nullspace_facts, evidence
    )


def _validate_reaction(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise TypeError("cell-centred reaction must be a schema-v1 mapping")
    from pops.solvers._numeric import native_float
    if value.get("kind") == "scalar_constant":
        if set(value) != {"schema_version", "kind", "value"}:
            raise ValueError("scalar_constant reaction has an invalid shape")
        if native_float(value["value"], where="constant field reaction") <= 0.0:
            raise ValueError("constant field reaction must be strictly positive")
        return
    if value.get("kind") == "scalar_bind_parameter":
        if set(value) != {"schema_version", "kind", "parameter", "multiplier"}:
            raise ValueError("scalar_bind_parameter reaction has an invalid shape")
        parameter = value["parameter"]
        if (
            not isinstance(parameter, Mapping)
            or parameter.get("kind") != "parameter"
            or parameter.get("param_kind") not in ("runtime", "derived")
            or type(parameter.get("qualified_id")) is not str
            or not parameter["qualified_id"]
        ):
            raise ValueError("scalar_bind_parameter reaction lost its parameter identity")
        if native_float(
            value["multiplier"], where="field reaction parameter multiplier"
        ) <= 0.0:
            raise ValueError("field reaction multiplier must be strictly positive")
        return
    raise ValueError("cell-centred reaction carries an unknown kind")


def _validate_binding(binding: PreparedFieldLoweringBinding, where: str) -> None:
    if binding.options:
        raise ValueError("%s cell-centred lowering options changed" % where)
    resolution = binding.resolution
    expected = {
        "rhs", "rhs_identity", "output_route", "method", "reaction", "bc", "boundary_faces",
        "boundary_kernel_required", "boundary_iterate_dependent",
        "boundary_dependencies", "hierarchy_policy", "topology_recipe",
    }
    if set(resolution.native_options) != expected:
        raise ValueError("%s cell-centred native contract changed shape" % where)
    method = resolution.native_options["method"]
    if not isinstance(method, Mapping) or dict(method) != {
        "native_method": "cell_centered_second_order", "order": 2, "ghost_depth": 1,
    }:
        raise ValueError("%s cell-centred method consequences changed" % where)
    _validate_reaction(resolution.native_options["reaction"])
    if resolution.solver_facts.target not in ("system", "amr_system"):
        raise ValueError("%s cell-centred target is unsupported" % where)
    if resolution.solver_facts.operator.get("principal") != "scalar-laplacian":
        raise ValueError("%s cell-centred operator facts changed" % where)
    if dict(resolution.solver_facts.hierarchy) != dict(
        resolution.native_options["hierarchy_policy"]
    ):
        raise ValueError("%s cell-centred hierarchy facts changed" % where)
    topology_identity = resolution.solver_facts.layout.get("topology_identity")
    if resolution.nullspace_facts.topology_identity != topology_identity:
        raise ValueError("%s cell-centred topology authority changed" % where)


def _operator_parameter_handles(
    binding: PreparedFieldLoweringBinding, operator: Any,
) -> tuple[Any, ...]:
    reaction = binding.resolution.native_options["reaction"]
    if reaction is None or reaction["kind"] == "scalar_constant":
        return ()
    qualified_id = reaction["parameter"]["qualified_id"]
    matches = tuple(
        reference for reference in operator.declaration_references()
        if getattr(reference, "kind", None) == "parameter"
        and reference.qualified_id == qualified_id
    )
    if len(matches) != 1:
        raise ValueError("screened field plan lost its exact reaction parameter identity")
    matches[0].canonical_identity()
    return matches


def _parameter_handles(
    binding: PreparedFieldLoweringBinding,
    operator: Any,
    discretization: Any,
) -> Mapping[str, tuple[Any, ...]]:
    boundary = {}
    for condition_binding in discretization.boundaries:
        for handle in condition_binding.condition.declaration_references():
            if getattr(handle, "kind", None) == "parameter":
                handle.canonical_identity()
                boundary.setdefault(handle.qualified_id, handle)
    return {
        "boundary-kernel": tuple(boundary[key] for key in sorted(boundary)),
        "native-install": _operator_parameter_handles(binding, operator),
    }


def _bind_native_options(
    binding: PreparedFieldLoweringBinding,
    operator: Any,
    discretization: Any,
    params: Mapping[Any, Any],
) -> Mapping[str, Any]:
    del discretization
    reaction = binding.resolution.native_options["reaction"]
    if reaction is None:
        return {}
    from pops.solvers._numeric import native_float
    if reaction["kind"] == "scalar_constant":
        return {
            "reaction": native_float(
                reaction["value"], where="constant field reaction"
            )
        }
    handles = _operator_parameter_handles(binding, operator)
    if len(handles) != 1 or handles[0] not in params:
        raise ValueError(
            "screened field reaction parameter is missing at bind: %s"
            % reaction["parameter"]["qualified_id"]
        )
    value = native_float(
        params[handles[0]],
        where="screened field reaction parameter %s" % handles[0].qualified_id,
    )
    multiplier = native_float(
        reaction["multiplier"], where="screened field reaction multiplier"
    )
    effective = value * multiplier
    if not math.isfinite(effective) or effective <= 0.0:
        raise ValueError(
            "screened field reaction coefficient must be strictly positive at bind"
        )
    return {"reaction": effective}


def _install_bound_options(
    binding: PreparedFieldLoweringBinding,
    context: PreparedFieldRuntimeInstallContext,
    bound_options: Mapping[str, Any],
) -> None:
    """Commit coefficients that the engine-free phase already bound and validated."""
    del binding
    if "reaction" in bound_options:
        context.engine.set_field_reaction(context.slot, bound_options["reaction"])


def _prepare_output(
    binding: PreparedFieldLoweringBinding,
    context: PreparedFieldRuntimePreflightContext,
    operator: Any,
    discretization: Any,
) -> Mapping[str, Any]:
    """Resolve the concrete storage route without access to the mutable native engine."""
    del discretization
    if context.target != binding.resolution.solver_facts.target:
        raise ValueError("cell-centred runtime target changed after field lowering")
    route = binding.resolution.native_options["output_route"]
    block = route["owner_block"]
    models = context.resources.get("models")
    if not isinstance(models, Mapping):
        raise TypeError("cell-centred runtime requires a model registry resource")
    model = models.get(block)
    if model is None:
        raise ValueError("field output route names unknown block %r" % block)
    from pops.physics.aux import aux_component_index

    declared = tuple(getattr(model, "aux_extra_names", ()) or ())
    components = tuple(route["components"])
    try:
        indices = [aux_component_index(component, declared) for component in components]
    except ValueError as error:
        raise ValueError(
            "field output route %r is absent from block %r native aux layout: %s"
            % (operator.name, block, ", ".join(components))
        ) from error
    indices.extend([-1] * (3 - len(indices)))
    gradient_sign = route.get("gradient_sign")
    if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
        raise ValueError("field output route has no valid GradientOutput sign")
    if indices[1] < 0 and gradient_sign != 1:
        raise ValueError("field output route carries a sign without gradient components")
    return {
        "block": block,
        "key": route["key"],
        "indices": indices,
        "gradient_sign": gradient_sign,
    }


def _install_output(
    binding: PreparedFieldLoweringBinding,
    context: PreparedFieldRuntimeInstallContext,
    output_payload: Mapping[str, Any],
) -> None:
    """Commit the provider-owned output payload after every input preflight succeeded."""
    del binding
    indices = output_payload["indices"]
    context.engine.register_elliptic_field(
        output_payload["block"],
        output_payload["key"],
        indices[0],
        indices[1],
        indices[2],
        output_payload["gradient_sign"],
    )


_PROVIDER = register_prepared_field_lowering_provider(PreparedFieldLoweringProvider(
    provider_id="pops.field-lowering.cell-centered-second-order",
    version=1,
    resolver_id="pops.field-lowering.cell-centered-second-order.resolve@1",
    resolution_validator_id=(
        "pops.field-lowering.cell-centered-second-order.validate@1"
    ),
    runtime_binder_id="pops.field-lowering.cell-centered-second-order.bind@1",
    output_preparer_id=(
        "pops.field-lowering.cell-centered-second-order.prepare-output@1"
    ),
    bound_options_installer_id=(
        "pops.field-lowering.cell-centered-second-order.install-options@1"
    ),
    output_installer_id=(
        "pops.field-lowering.cell-centered-second-order.install-output@1"
    ),
    capabilities={
        "targets": ("system", "amr_system"),
        "dimension": 2,
        "layout": ("uniform", "amr"),
        "principal": "scalar-laplacian",
        "reaction": "positive-scalar",
        "order": 2,
        "ghost_depth": 1,
        "runtime_installation": {
            "preflight": "engine-free-canonical@1",
            "commit": "provider-owned@1",
        },
    },
    resolver=_resolve,
    resolution_validator=_validate_binding,
    parameter_handles=_parameter_handles,
    bind_native_options=_bind_native_options,
    prepare_output=_prepare_output,
    install_bound_options=_install_bound_options,
    install_output=_install_output,
))


def cell_centered_second_order_field_lowering_provider() -> PreparedFieldLoweringProvider:
    return _PROVIDER


__all__ = ["cell_centered_second_order_field_lowering_provider"]
