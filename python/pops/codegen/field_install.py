"""Total resolve-time lowering from field declarations to runtime install plans."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

from pops.codegen.lowering_coverage import (
    LoweringCoverageReport,
    LoweringCoverageRow,
    LoweringRejection,
)
from pops.codegen.field_boundary_lowering import (
    boundary_dependency_pack,
    boundary_plan,
    field_layout_contract,
    topology_recipe,
)
from pops.identity import Identity
from pops.math import principal_kinds

from pops.fields._identity import field_identity, strict_field_data
from pops.fields.discretization import (
    FieldDiscretizationProtocol,
    field_discretization_data,
    require_field_discretization,
)
from pops.fields.gauges import MeanValueGauge
from pops.fields.nullspace import ConstantNullspace
from pops.fields.operator import FieldOperator, _field_targets_unknown


def _reject(rows: list[LoweringCoverageRow], source: str, gate: str, message: str) -> None:
    report = LoweringCoverageReport((*rows, LoweringCoverageRow(
        source, "rejected", gate=gate)))
    raise LoweringRejection(
        message, coverage_report=report, source=source, gate=gate)


def _zero(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _validate_reaction_manifest(reaction: Any) -> None:
    """Validate the closed scalar-reaction plan variants carried across compile/bind."""
    if reaction is None:
        return
    if not isinstance(reaction, Mapping) or reaction.get("schema_version") != 1:
        raise TypeError("field reaction plan must be a schema-v1 mapping")
    kind = reaction.get("kind")
    from pops.solvers._numeric import native_float
    if kind == "scalar_constant":
        if set(reaction) != {"schema_version", "kind", "value"}:
            raise ValueError("scalar_constant reaction plan has an invalid shape")
        value = native_float(reaction["value"], where="constant field reaction")
        if value <= 0.0:
            raise ValueError("constant field reaction must be strictly positive")
        return
    if kind == "scalar_bind_parameter":
        if set(reaction) != {"schema_version", "kind", "parameter", "multiplier"}:
            raise ValueError("scalar_bind_parameter reaction plan has an invalid shape")
        parameter = reaction["parameter"]
        if not isinstance(parameter, Mapping) \
                or parameter.get("kind") != "parameter" \
                or parameter.get("param_kind") not in ("runtime", "derived") \
                or not isinstance(parameter.get("qualified_id"), str) \
                or not parameter["qualified_id"]:
            raise ValueError(
                "scalar_bind_parameter reaction plan requires one canonical runtime/derived "
                "parameter identity")
        multiplier = native_float(
            reaction["multiplier"], where="field reaction parameter multiplier")
        if multiplier <= 0.0:
            raise ValueError("field reaction parameter multiplier must be strictly positive")
        return
    raise ValueError("field reaction plan carries unknown kind %r" % kind)


@dataclass(frozen=True, slots=True)
class ResolvedFieldInstallPlan:
    """Complete field semantics plus their exact native lowering."""

    name: str
    operator: FieldOperator
    discretization: FieldDiscretizationProtocol
    target: str
    rhs_providers: tuple[Any, ...]
    native_options: Any
    coverage: LoweringCoverageReport
    nonlinear_provider: Any
    identity: Identity

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("ResolvedFieldInstallPlan name must be non-empty")
        if self.operator.name != self.name:
            raise ValueError("field install name disagrees with FieldOperator")
        require_field_discretization(
            self.discretization, where="resolved field install discretization")
        if self.target not in ("system", "amr_system"):
            raise ValueError("field install target is unsupported")
        from pops.model import Handle
        if not self.rhs_providers or any(
                not isinstance(provider, Handle) or not provider.is_resolved
                or provider.kind != "field_operator" for provider in self.rhs_providers):
            raise TypeError("field install requires canonical field-operator RHS providers")
        if not isinstance(self.coverage, LoweringCoverageReport):
            raise TypeError("field install requires exact lowering coverage")
        nonlinear_manifest = self.native_options.get("nonlinear")
        if nonlinear_manifest is None:
            if self.nonlinear_provider is not None:
                raise ValueError("field install retains an undeclared nonlinear provider")
        else:
            validate = getattr(self.nonlinear_provider, "__post_init__", None)
            install = getattr(self.nonlinear_provider, "install", None)
            to_data = getattr(self.nonlinear_provider, "to_data", None)
            if not callable(validate) or not callable(install) or not callable(to_data):
                raise TypeError("field nonlinear provider does not implement the prepared protocol")
            validate()
            if to_data() != nonlinear_manifest:
                raise ValueError("field nonlinear provider disagrees with its canonical manifest")
        _validate_reaction_manifest(self.native_options.get("reaction"))
        object.__setattr__(self, "native_options", MappingProxyType(dict(self.native_options)))
        expected = field_identity("resolved-field-install", self.to_data(include_identity=False))
        if self.identity != expected:
            raise ValueError("resolved field install identity is not canonical")

    def to_data(self, *, include_identity: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": 1,
            "name": self.name,
            "operator": self.operator.to_data(),
            "discretization": field_discretization_data(
                self.discretization, where="resolved field install discretization"),
            "target": self.target,
            "rhs_providers": [provider.canonical_identity()
                              for provider in self.rhs_providers],
            "native_options": dict(self.native_options),
            "coverage": self.coverage.to_data(),
        }
        if include_identity:
            data["identity"] = self.identity.token
        return data

    def boundary_parameter_handles(self) -> tuple[Any, ...]:
        """Exact compact parameter pack consumed by the generated boundary launchers."""
        handles = {}
        for binding in self.discretization.boundaries:
            for handle in binding.condition.declaration_references():
                if getattr(handle, "kind", None) == "parameter":
                    handle.canonical_identity()
                    handles.setdefault(handle.qualified_id, handle)
        return tuple(handles[key] for key in sorted(handles))

    def reaction_parameter_handles(self) -> tuple[Any, ...]:
        """Exact owner-qualified scalar coefficient consumed by a screened solve."""
        reaction = self.native_options.get("reaction")
        if reaction is None or reaction["kind"] == "scalar_constant":
            return ()
        if reaction["kind"] != "scalar_bind_parameter":
            raise ValueError("screened field plan carries an unknown reaction kind")
        qualified_id = reaction["parameter"]["qualified_id"]
        matches = tuple(
            reference for reference in self.operator.declaration_references()
            if getattr(reference, "kind", None) == "parameter"
            and reference.qualified_id == qualified_id
        )
        if len(matches) != 1:
            raise ValueError(
                "screened field plan lost its exact reaction parameter identity")
        matches[0].canonical_identity()
        return matches

    def native_reaction_value(self, params: Mapping[Any, Any]) -> float | None:
        """Resolve either reaction variant to the one native scalar installed for this solve."""
        reaction = self.native_options.get("reaction")
        if reaction is None:
            return None
        from pops.solvers._numeric import native_float
        if reaction["kind"] == "scalar_constant":
            return native_float(reaction["value"], where="constant field reaction")
        handles = self.reaction_parameter_handles()
        if len(handles) != 1 or handles[0] not in params:
            qualified_id = reaction["parameter"]["qualified_id"]
            raise ValueError(
                "screened field reaction parameter is missing at bind: %s" % qualified_id)
        value = native_float(
            params[handles[0]], where="screened field reaction parameter %s" %
            handles[0].qualified_id)
        multiplier = native_float(
            reaction["multiplier"], where="screened field reaction multiplier")
        effective = value * multiplier
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError(
                "screened field reaction coefficient must be strictly positive at bind")
        return effective

    def component_bindings(self) -> tuple[dict[str, Any], ...]:
        """Exact external component pair required by this field provider, if any."""
        provider = self.native_options["solver_provider"]
        if provider["provider_kind"] == "builtin_v1":
            return ()
        if provider["provider_kind"] != "external_component_v1":
            raise ValueError("resolved field plan carries an unknown solver provider kind")
        return provider["topology"], provider["solver"]

    def require_component_inputs(self, components: tuple[Any, ...]) -> None:
        """Match every authored field component against the explicit resolve inputs."""
        from pops.external import CompiledComponentArtifact, ExternalComponent

        by_id = {}
        for component in components:
            if type(component) is ExternalComponent:
                component_id = component.component_manifest.component_id
                manifest = component.component_manifest.manifest_digest.token
                interface = component.component_type.interface.to_data()
                source_package = component.package_identity.token
                parameters = component.to_data()["parameters"]
            elif type(component) is CompiledComponentArtifact:
                component.verify()
                component_id = component.component_id
                manifest = component.component_manifest.token
                interface = component.interface.to_data()
                source_package = (
                    None if component.source_package is None
                    else component.source_package.token
                )
                parameters = None
            else:
                continue
            by_id[component_id] = (manifest, interface, source_package, parameters)
        for binding in self.component_bindings():
            component_id = binding["component_id"]
            actual = by_id.get(component_id)
            if actual is None:
                raise ValueError(
                    "field %r requires exact component %r in resolve(components=)"
                    % (self.name, component_id)
                )
            expected = (
                binding["component_manifest_identity"], binding["native_interface"],
                binding["source_package_identity"])
            if actual[:3] != expected or (
                    actual[3] is not None and actual[3] != binding["parameters"]):
                raise ValueError(
                    "field %r component %r changed source package, manifest, or native "
                    "interface identity"
                    % (self.name, component_id)
                )


def resolve_field_install_plan(
    name: str,
    registration: Any,
    *,
    target: str,
    rhs_providers: tuple[Any, ...],
    provider_route: tuple[dict[str, Any], ...],
    output_components: tuple[str, ...],
    layout: Any,
) -> ResolvedFieldInstallPlan:
    operator = getattr(registration, "operator", None)
    plan = getattr(registration, "discretization", None)
    if not isinstance(operator, FieldOperator):
        raise TypeError("field registration lost its FieldOperator")
    plan = require_field_discretization(plan, where="field registration discretization")
    output_components = tuple(output_components)
    if len(output_components) not in (1, 3) or any(
            not isinstance(component, str) or not component
            for component in output_components):
        raise TypeError(
            "field output space must resolve to one potential component or "
            "potential plus two gradient components"
        )
    from pops.fields.outputs import FieldOutput, GradientOutput
    outputs = tuple(operator.outputs)
    if len(output_components) == 1:
        if len(outputs) != 1 or not isinstance(outputs[0], FieldOutput):
            raise TypeError(
                "native scalar field output must be exactly one FieldOutput")
        gradient_sign = 1
    else:
        if (len(outputs) != 2 or not isinstance(outputs[0], FieldOutput)
                or not isinstance(outputs[1], GradientOutput)):
            raise TypeError(
                "native gradient field output must be FieldOutput + GradientOutput")
        gradient_sign = outputs[1].sign
        if type(gradient_sign) is not int or gradient_sign not in (-1, 1):
            raise ValueError("resolved GradientOutput sign must be exactly -1 or 1")
    potential_source = outputs[0].source
    if potential_source is not None and potential_source != operator.unknown:
        raise ValueError(
            "native FieldOutput source disagrees with the FieldOperator solved unknown")
    if len(outputs) == 2 and outputs[1].source != operator.unknown:
        raise ValueError(
            "native GradientOutput source disagrees with the FieldOperator solved unknown")
    rows: list[LoweringCoverageRow] = []
    layout_contract = field_layout_contract(layout)
    resolved_topology_recipe = topology_recipe(layout)
    source = "field:%s:operator" % name
    kinds = principal_kinds(operator.equation.lhs)
    if "laplacian" not in kinds or kinds - {"laplacian", "reaction"}:
        _reject(rows, source, "field.operator.not_native",
                "field %r is generic but its principal operator %s has no native lowering"
                % (name, sorted(kinds)))
    from pops._ir.elliptic import Reaction, constant_reaction_scalar, elliptic_terms
    from pops._ir.expr import Laplacian
    from pops._ir.values import RuntimeParamRef

    terms = elliptic_terms(operator.equation.lhs)
    laplacians = [term for term in terms if isinstance(term, Laplacian)]
    reactions = [term for term in terms if isinstance(term, Reaction)]
    if len(laplacians) != 1 or len(reactions) > 1 or len(terms) != 1 + len(reactions):
        _reject(rows, source, "field.operator.poisson_shape_not_native",
                "field %r requires exactly one Laplacian and at most one scalar reaction term"
                % name)

    laplacian_term = laplacians[0]
    if not _field_targets_unknown(laplacian_term.field, operator.unknown):
        _reject(rows, source, "field.operator.unknown_mismatch",
                "field %r Laplacian does not act on its declared unknown" % name)
    normalization = -float(laplacian_term.scale)
    if not math.isfinite(normalization) or normalization == 0.0:
        _reject(rows, source, "field.operator.invalid_laplacian_scale",
                "field %r Laplacian scale must be finite and non-zero" % name)
    reaction_options = None
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
            _reject(rows, source, "field.operator.reaction_unknown_mismatch",
                    "field %r reaction does not act on its declared unknown" % name)
        if constant is NotImplemented and not bind_parameter:
            _reject(rows, source, "field.operator.reaction_coefficient_not_native",
                    "field %r reaction requires an exact finite real/ConstParam or one typed "
                    "Real RuntimeParam/DerivedParam read" % name)
        if not math.isfinite(multiplier):
            _reject(rows, source, "field.operator.reaction_sign_not_native",
                    "field %r must normalize to -laplacian(phi) + kappa*phi with kappa > 0"
                    % name)
        if constant is NotImplemented:
            if multiplier <= 0.0:
                _reject(rows, source, "field.operator.reaction_sign_not_native",
                        "field %r must normalize to -laplacian(phi) + kappa*phi with kappa > 0"
                        % name)
            reaction_options = {
                "schema_version": 1,
                "kind": "scalar_bind_parameter",
                "parameter": handle.canonical_identity(),
                "multiplier": multiplier,
            }
            reaction_route = "scalar-bind"
        else:
            try:
                effective = float(constant) * multiplier
            except (TypeError, ValueError, OverflowError):
                effective = float("nan")
            if not math.isfinite(effective) or effective <= 0.0:
                _reject(rows, source, "field.operator.reaction_sign_not_native",
                        "field %r must normalize to -laplacian(phi) + kappa*phi with kappa > 0"
                        % name)
            reaction_options = {
                "schema_version": 1,
                "kind": "scalar_constant",
                "value": effective,
            }
            reaction_route = "scalar-constant"
        rows.append(LoweringCoverageRow(
            "field:%s:reaction" % name, "lowered",
            ("field-install:%s:reaction:%s" % (name, reaction_route),)))
    rows.append(LoweringCoverageRow(source, "lowered", (
        "field-install:%s:residual" % name,)))
    if not rhs_providers or len(rhs_providers) != len(provider_route):
        _reject(rows, "field:%s:provider" % name, "field.provider.pack_invalid",
                "field %r requires a non-empty, fully routed provider pack" % name)
    provider_identity = {
        "schema_version": 1,
        "contributions": [
            {
                "provider": provider.canonical_identity(),
                "owner_block": route["owner_block"],
                "native_key": route["key"],
                "coefficient": route["coefficient"],
                "measure": route["measure"],
            }
            for provider, route in zip(rhs_providers, provider_route, strict=True)
        ],
    }
    from pops.identity import canonical_bytes
    provider_slot = field_identity(
        "qualified-field-provider", provider_identity).token
    rows.append(LoweringCoverageRow(
        "field:%s:provider" % name, "lowered", (
            "field-install:%s:provider:%s" % (name, provider_slot),)))
    for reference in operator.declaration_references():
        rows.append(LoweringCoverageRow(
            "field:%s:dependency:%s" % (name, reference.qualified_id),
            "lowered", ("field-install:%s:qualified-dependency" % name,)))

    method_source = "field:%s:method" % name
    method_adapter = getattr(plan.method, "lower_field_method", None)
    if not callable(method_adapter):
        _reject(rows, method_source, "field.method.not_native",
                 "field %r method %s has no native lowering" %
                 (name, type(plan.method).__name__))
    method_options = method_adapter(target=target, layout=layout)
    if not isinstance(method_options, dict) or not isinstance(
            method_options.get("native_method"), str):
        _reject(rows, method_source, "field.method.invalid_adapter",
                "field %r method returned an invalid native lowering" % name)
    rows.append(LoweringCoverageRow(method_source, "lowered", (
        "field-install:%s:cell-centered-second-order" % name,)))

    # The current native field operator is the full rectangular Cartesian cell graph.  It does not
    # yet consume Uniform.embedded_boundary, so pretending to materialise connected components here
    # would silently solve a different domain and derive the wrong nullspace dimension.  Refuse that
    # topology explicitly; the accepted recipe below truthfully has exactly one material component.
    embedded = layout_contract.embedded_boundary
    if embedded is not None:
        _reject(rows, "field:%s:topology" % name,
                "field.topology.embedded_boundary_not_native",
                "field %r uses an embedded boundary, but the native field residual has no "
                "material-cell connectivity/mask lowering for that topology" % name)
    rows.append(LoweringCoverageRow(
        "field:%s:topology" % name, "derived",
        rule="resolved full rectangular cell graph has one connected material component"))

    bc, boundary_faces = boundary_plan(name, plan, rows, layout, operator.unknown)
    boundary_dependencies = boundary_dependency_pack(plan, operator.unknown)
    if target == "amr_system" and boundary_dependencies["fields"]:
        _reject(rows, "field:%s:boundaries" % name,
                "field.boundary.amr_field_dependency_not_native",
                "field %r has a boundary law depending on another solved field; the AMR "
                "backend has no exact same-level/composite materialization route for that "
                "dependency yet" % name)
    if target == "amr_system" and layout_contract.levels > 1 \
            and boundary_dependencies["states"]:
        _reject(rows, "field:%s:boundaries" % name,
                "field.boundary.amr_multilevel_state_dependency_not_native",
                "field %r has a state-dependent boundary law on a multilevel hierarchy; the "
                "generated physical-face provider does not yet receive one exact state buffer "
                "per level" % name)
    for kind in ("states", "fields"):
        for dependency in boundary_dependencies[kind]:
            rows.append(LoweringCoverageRow(
                "field:%s:boundary-dependency:%s:%d" % (
                    name, dependency["qualified_id"], dependency["component"]),
                "lowered", ("field-install:%s:boundary-buffer:%s" % (name, kind),)))
    for coordinate in boundary_dependencies["logical_time"]:
        rows.append(LoweringCoverageRow(
            "field:%s:boundary-time:%s" % (name, coordinate), "lowered",
            ("field-install:%s:logical-timepoint" % name,)))
    boundary_kernel_required = boundary_faces is not None and any(
        face["dynamic"] for face in boundary_faces)
    boundary_iterate_dependent = boundary_faces is not None and any(
        face["iterate_dependent"] for face in boundary_faces)
    solver_source = "field:%s:solver" % name
    solver_adapter = getattr(plan.solver, "lower_field_solver", None)
    if not callable(solver_adapter):
        _reject(rows, solver_source, "field.solver.not_native",
                 "field %r solver %s has no native lowering" %
                 (name, type(plan.solver).__name__))
    try:
        solver_options = solver_adapter(target=target, layout=layout)
    except (TypeError, ValueError) as exc:
        _reject(rows, solver_source, "field.solver.layout_incompatible", str(exc))
    if not isinstance(solver_options, dict):
        _reject(rows, solver_source, "field.solver.invalid_adapter",
                "field %r solver returned an invalid native lowering" % name)
    provider_kind = solver_options.get("provider_kind")
    topology_identity = field_identity(
        "resolved-field-topology", resolved_topology_recipe).token
    if provider_kind == "external_component_v1":
        expected_keys = {
            "provider_id", "provider_kind", "topology", "solver", "request"
        }
        if set(solver_options) != expected_keys:
            _reject(rows, solver_source, "field.solver.invalid_external_provider",
                    "field %r external solver returned an incomplete provider pack" % name)
        solver_provider = {
            **solver_options,
            "topology_recipe_identity": topology_identity,
        }
        solver_label = solver_options["solver"]["component_id"]
        if boundary_kernel_required or any(boundary_dependencies[kind] for kind in (
                "states", "fields", "logical_time")):
            _reject(
                rows, solver_source, "field.solver.external_boundary_context_unavailable",
                "field %r uses a dynamic/dependent boundary law, but FieldSolver ABI v2 "
                "carries only an immutable boundary contract" % name,
            )
    else:
        solver_token = solver_options.get("native_solver")
        if not isinstance(solver_token, str) or not solver_token:
            _reject(rows, solver_source, "field.solver.invalid_adapter",
                    "field %r solver returned an invalid builtin provider" % name)
        solver_provider = {
            "schema_version": 1,
            "provider_kind": "builtin_v1",
            "solver": {
                "route": solver_token,
                "capabilities": dict(solver_options),
            },
            "topology": {
                "provider_kind": "builtin_rectangular_cell_graph_v1",
                "provenance": "pops.builtin.rectangular-cell-graph.v1",
                "topology_digest": topology_identity,
            },
            "request": None,
            "topology_recipe_identity": topology_identity,
        }
        solver_label = solver_token
    if reaction_options is not None:
        if provider_kind == "external_component_v1":
            _reject(rows, solver_source, "field.solver.external_screened_protocol_unavailable",
                    "field %r is screened, but FieldSolver ABI v2 has no reaction coefficient "
                    "carrier" % name)
        if solver_label != "geometric_mg":
            _reject(rows, solver_source, "field.solver.screened_route_unavailable",
                    "field %r is screened and requires the native GeometricMG route" % name)
    rows.append(LoweringCoverageRow(solver_source, "lowered", (
        "field-install:%s:solver:%s" % (name, solver_label),)))
    nonlinear_options = None
    nonlinear_provider = None
    if plan.nonlinear is not None:
        if provider_kind == "external_component_v1":
            _reject(
                rows, "field:%s:nonlinear" % name,
                "field.solver.external_nonlinear_protocol_unavailable",
                "field %r combines ExternalFieldSolver v2 with a PoPS-owned nonlinear "
                "outer iteration; no shared iterate/JVP protocol exists" % name,
            )
        nonlinear_adapter = getattr(plan.nonlinear, "lower_field_nonlinear", None)
        if not callable(nonlinear_adapter):
            _reject(rows, "field:%s:nonlinear" % name, "field.nonlinear.not_native",
                    "field %r nonlinear solver has no lowering adapter" % name)
        try:
            nonlinear_provider = nonlinear_adapter(target=target, layout=layout)
        except (TypeError, ValueError) as exc:
            _reject(rows, "field:%s:nonlinear" % name,
                    "field.nonlinear.layout_incompatible", str(exc))
        manifest = getattr(nonlinear_provider, "to_data", None)
        install_adapter = getattr(nonlinear_provider, "install", None)
        capabilities = getattr(nonlinear_provider, "capabilities", frozenset())
        identity = getattr(nonlinear_provider, "identity", None)
        if not callable(manifest) or not callable(install_adapter) or \
                not isinstance(identity, Identity) or not {
                    "residual", "publication_atomic", "reject_attempt"
                }.issubset(set(capabilities)):
            _reject(rows, "field:%s:nonlinear" % name,
                    "field.nonlinear.invalid_adapter",
                    "field %r nonlinear solver returned an invalid prepared provider" % name)
        nonlinear_options = manifest()
        rows.append(LoweringCoverageRow(
            "field:%s:nonlinear" % name, "lowered",
            ("field-install:%s:nonlinear:%s" % (name, identity.token),)))
    if boundary_iterate_dependent and nonlinear_provider is None:
        _reject(rows, "field:%s:boundaries" % name,
                "field.boundary.nonlinear_outer_solver_required",
                "field %r has iterate-dependent alpha/beta/value expressions and requires a "
                "prepared nonlinear outer solver" % name)

    # Nullspace dimension is a topology fact and cannot silently change with a runtime Robin alpha.
    # A dynamic alpha is nevertheless fully supported when another face is statically Dirichlet-like,
    # because that face anchors the operator for every runtime value of the dynamic coefficient.
    dynamic_alpha = boundary_faces is not None and any(
        "alpha" in face["dynamic"] for face in boundary_faces)
    statically_anchored = boundary_faces is not None and any(
        face["type"] != "periodic" and "alpha" not in face["dynamic"]
        and face["alpha"] != 0.0 for face in boundary_faces)
    if reaction_options is None and dynamic_alpha and not statically_anchored:
        _reject(rows, "field:%s:boundaries" % name,
                "field.boundary.dynamic_nullspace_topology",
                "field %r has a dynamic Robin alpha that can change the nullspace dimension; "
                "add a statically Dirichlet-like anchor or use a boundary law with invariant alpha"
                % name)

    if plan.preconditioner is not None:
        _reject(rows, "field:%s:preconditioner" % name,
                "field.preconditioner.not_native",
                "field %r declares an external preconditioner that the native seam cannot consume"
                % name)
    rows.append(LoweringCoverageRow(
        "field:%s:preconditioner" % name, "documentary"))

    # The mathematical kernel is derived from operator + resolved topology + BC.  A user-supplied
    # ConstantNullspace is an assertion only; it never creates a kernel that the operator does not
    # have.  The representative-selection gauge remains an independent, explicit choice.
    singular_faces = boundary_faces is not None and all(
        face["type"] == "periodic" or face["alpha"] == 0.0
        for face in boundary_faces)
    derived_nullspace = (
        "constant" if reaction_options is None and singular_faces else "none")
    nullspace_assertion = "none"
    if plan.nullspace is not None:
        if not isinstance(plan.nullspace, ConstantNullspace):
            _reject(rows, "field:%s:nullspace" % name, "field.nullspace.not_native",
                    "field %r nullspace assertion has no native topology derivation" % name)
        nullspace_assertion = "constant"
        if boundary_faces is not None and derived_nullspace != "constant":
            _reject(rows, "field:%s:nullspace" % name, "field.nullspace.assertion_mismatch",
                    "field %r asserts ConstantNullspace but operator+topology+BC are invertible"
                    % name)
    if derived_nullspace == "constant":
        if not isinstance(plan.gauge, MeanValueGauge) or not _zero(plan.gauge.value):
            _reject(rows, "field:%s:gauge" % name, "field.gauge.required",
                    "field %r has a topology-derived constant kernel and requires an explicit "
                    "MeanValueGauge(0)" % name)
        gauge = "mean_zero"
    else:
        if plan.gauge is not None:
            _reject(rows, "field:%s:gauge" % name, "field.gauge.without_nullspace",
                    "field %r declares a gauge for an invertible operator" % name)
        gauge = "none"
    nullspace = derived_nullspace
    rows.append(LoweringCoverageRow(
        "field:%s:nullspace" % name, "lowered" if nullspace != "none" else "documentary",
        (() if nullspace == "none" else ("field-install:%s:nullspace:constant" % name,))))
    rows.append(LoweringCoverageRow(
        "field:%s:gauge" % name, "lowered" if gauge != "none" else "documentary",
        (() if gauge == "none" else ("field-install:%s:gauge:mean-zero" % name,))))

    policy = plan.hierarchy_policy.options()["policy"]
    if target == "system":
        if policy == "composite":
            _reject(rows, "field:%s:hierarchy" % name, "field.hierarchy.unsupported",
                    "uniform System cannot lower CompositeHierarchySolve")
        hierarchy = "level_local"
    else:
        hierarchy = "composite" if policy in ("infer_from_layout", "composite") else "level_local"
    if target == "amr_system" and hierarchy == "level_local" \
            and layout_contract.levels > 1:
        _reject(rows, "field:%s:hierarchy" % name,
                "field.hierarchy.level_local_partial_topology_not_native",
                "field %r requests level-local solves on a refined partial BoxArray; the native "
                "field residual has no coarse/fine boundary closure or per-patch connected-component "
                "basis for that route (use CompositeHierarchySolve)" % name)
    rows.append(LoweringCoverageRow(
        "field:%s:hierarchy" % name, "derived", rule=(
            "%s + target=%s resolves to %s" % (policy, target, hierarchy))))

    output_owner = operator.unknown.block_ref.local_id
    rows.append(LoweringCoverageRow(
        "field:%s:output" % name,
        "lowered",
        ("field-install:%s:output:%s:%s" % (
            name, output_owner, operator.unknown.qualified_id),),
    ))

    options = {
        "rhs": "composite",
        "provider_slot": provider_slot,
        "provider_identity": provider_identity,
        "provider_identity_text": canonical_bytes(
            strict_field_data(provider_identity)).hex(),
        "provider_pack": [dict(route) for route in provider_route],
        "output_route": {
            "owner_identity": operator.unknown.block_ref.canonical_identity(),
            "owner_block": operator.unknown.block_ref.local_id,
            "key": operator.name,
            "components": output_components,
            "gradient_sign": gradient_sign,
        },
        "rhs_identity": field_identity(
            "field-rhs", {
                "rhs": operator.to_data()["equation"]["equation"]["rhs"],
            }).token,
        "solver_provider": solver_provider,
        "method": method_options,
        "solver_capabilities": solver_options,
        "reaction": reaction_options,
        "nonlinear": nonlinear_options,
        "bc": bc,
        "boundary_faces": boundary_faces,
        "boundary_kernel_required": boundary_kernel_required,
        "boundary_iterate_dependent": boundary_iterate_dependent,
        "boundary_dependencies": boundary_dependencies,
        "nullspace": nullspace,
        "nullspace_assertion": nullspace_assertion,
        "gauge": gauge,
        "hierarchy": hierarchy,
        "topology_recipe": resolved_topology_recipe,
    }
    report = LoweringCoverageReport(rows)
    data = {
        "schema_version": 1,
        "name": name,
        "operator": operator.to_data(),
        "discretization": field_discretization_data(
            plan, where="resolved field install discretization"),
        "target": target,
        "rhs_providers": [provider.canonical_identity() for provider in rhs_providers],
        "native_options": options,
        "coverage": report.to_data(),
    }
    identity = field_identity("resolved-field-install", data)
    return ResolvedFieldInstallPlan(
        name, operator, plan, target, rhs_providers, options, report,
        nonlinear_provider, identity)


__all__ = ["ResolvedFieldInstallPlan", "resolve_field_install_plan"]
