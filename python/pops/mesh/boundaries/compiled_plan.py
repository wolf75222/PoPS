"""Detached executable boundary plans retained after compilation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pops._report import Report
from pops.identity import make_identity
from pops.identity.semantic import semantic_value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class _BoundaryOperatorReport(Report):
    """Detached exact contribution/component rows for one implicit boundary operation."""

    def __init__(self, ghost_plan_identity: str, terms: list[dict[str, Any]]) -> None:
        self.ghost_plan_identity = ghost_plan_identity
        self.terms = terms

    def to_dict(self) -> dict[str, Any]:
        return self._stamp({
            "ghost_plan_identity": self.ghost_plan_identity,
            "terms": _thaw(self.terms),
        })


class BoundaryResidualReport(_BoundaryOperatorReport):
    """Residual-side implicit boundary terms carried by one compiled ghost plan."""

    report_type = "boundary_residual"


class BoundaryLinearizationReport(_BoundaryOperatorReport):
    """JVP-side implicit boundary terms carried by one compiled ghost plan."""

    report_type = "boundary_linearization"


@dataclass(frozen=True, slots=True)
class CompiledBoundaryPlan:
    """Canonical boundary lowering data with no authoring authority or Python callback."""

    compile_data: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.compile_data, Mapping):
            raise TypeError("CompiledBoundaryPlan.compile_data must be a mapping")
        data = _thaw(self.compile_data)
        if data.get("schema_version") != 1 \
                or data.get("authority_type") != "prepared_boundary_plan_compile" \
                or not isinstance(data.get("ghost_plan_identity"), str) \
                or not data["ghost_plan_identity"]:
            raise ValueError("CompiledBoundaryPlan requires total prepared v1 lowering data")
        if not isinstance(data.get("faces"), list) \
                or not isinstance(data.get("component_region_templates"), list):
            raise TypeError("CompiledBoundaryPlan face/component tables must be lists")
        object.__setattr__(self, "compile_data", _freeze(data))

    @classmethod
    def from_resolved(cls, boundary: Any) -> CompiledBoundaryPlan:
        compile_data = getattr(boundary, "compile_boundary_data", None)
        if not callable(compile_data):
            raise TypeError("resolved boundary authority lacks compile_boundary_data()")
        first, second = compile_data(), compile_data()
        if type(first) is not dict or first != second:
            raise TypeError("boundary compile data must be a deterministic exact dict")
        return cls(first)

    @property
    def canonical_id(self) -> str:
        return str(self.compile_data["ghost_plan_identity"])

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "compiled_boundary_plan": self.canonical_id,
            "compile_data": _thaw(self.compile_data),
        }

    def _operator_report(
            self, *, contribution_table: str, target_name: str, operation: str,
            report_type: type[_BoundaryOperatorReport]) -> _BoundaryOperatorReport:
        data = _thaw(self.compile_data)
        contributions = data.get(contribution_table, [])
        templates = data.get("component_region_templates", [])
        if not isinstance(contributions, list) or not isinstance(templates, list):
            raise TypeError("compiled boundary contribution report requires canonical lists")

        components: dict[str, dict[str, Any]] = {}
        for template in templates:
            if not isinstance(template, dict) or template.get("operation") != operation:
                continue
            target = template.get("target")
            qualified_id = target.get("qualified_id") if isinstance(target, dict) else None
            if not isinstance(qualified_id, str) or not qualified_id:
                raise TypeError(
                    "compiled boundary %s component has no qualified target" % operation)
            if qualified_id in components:
                raise ValueError(
                    "compiled boundary %s report has duplicate component target %s"
                    % (operation, qualified_id))
            components[qualified_id] = template

        rows = []
        contribution_targets = set()
        for contribution in contributions:
            if not isinstance(contribution, dict):
                raise TypeError("compiled boundary contribution report rows must be mappings")
            target = contribution.get(target_name)
            qualified_id = target.get("qualified_id") if isinstance(target, dict) else None
            if not isinstance(qualified_id, str) or not qualified_id:
                raise TypeError(
                    "compiled boundary %s contribution has no qualified operator"
                    % target_name)
            if qualified_id in contribution_targets:
                raise ValueError(
                    "compiled boundary %s report has duplicate contribution target %s"
                    % (target_name, qualified_id))
            contribution_targets.add(qualified_id)
            try:
                component = components[qualified_id]
            except KeyError:
                raise ValueError(
                    "compiled boundary %s contribution %s has no exact component report row"
                    % (target_name, qualified_id)) from None
            rows.append({
                "contribution": contribution,
                "component_region": component,
            })
        extra = sorted(set(components) - contribution_targets)
        if extra:
            raise ValueError(
                "compiled boundary %s report has unclaimed component target(s) %s"
                % (target_name, extra))
        return report_type(self.canonical_id, rows)

    def residual_report(self) -> BoundaryResidualReport:
        """Return exact residual contributions and their authenticated component rows."""
        return cast(BoundaryResidualReport, self._operator_report(
            contribution_table="residual_contributions",
            target_name="residual",
            operation="residual",
            report_type=BoundaryResidualReport,
        ))

    def linearization_report(self) -> BoundaryLinearizationReport:
        """Return exact JVP contributions and their authenticated component rows."""
        return cast(BoundaryLinearizationReport, self._operator_report(
            contribution_table="linearization_contributions",
            target_name="linearization",
            operation="jvp",
            report_type=BoundaryLinearizationReport,
        ))

    def runtime_boundary_data(self, params: Any) -> dict[str, Any]:
        """Bind scalar values through one generic evaluator, never an authoring callback."""
        from pops.model import Handle, ParamHandle
        from pops.model._bind_expression import eval_expression_key
        from pops.runtime._analytic_expression_lowering import lower_analytic_components

        if not isinstance(params, Mapping):
            raise TypeError("compiled boundary binding requires resolved BindSchema values")
        values_by_qid = {}
        handles_by_qid = {}
        for handle, value in params.items():
            if not isinstance(handle, ParamHandle) or not handle.is_resolved:
                raise TypeError("compiled boundary parameters require canonical ParamHandle keys")
            values_by_qid[handle.qualified_id] = value
            handles_by_qid[handle.qualified_id] = handle
        environment = dict(values_by_qid)

        data = _thaw(self.compile_data)
        ncomp = data.get("ncomp")
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
            raise TypeError("compiled boundary plan has no authenticated positive ncomp")
        faces = []
        for face in data["faces"]:
            if not isinstance(face, dict) or face.get("type") not in {
                    "periodic", "foextrap", "dirichlet", "slip_wall", "external"}:
                raise ValueError("compiled boundary face has no executable producer type")
            representation = face.get("representation", "conservative")
            converter = face.get("converter")
            if representation not in {"conservative", "primitive"}:
                raise ValueError("compiled boundary face has no executable state representation")
            if representation == "conservative" and converter is not None:
                raise ValueError(
                    "compiled conservative boundary face must not invent a converter")
            if representation == "primitive" and (
                    face["type"] != "dirichlet" or not isinstance(converter, str)
                    or not converter):
                raise ValueError(
                    "compiled primitive boundary face requires one exact fixed-state converter")
            if face["type"] in {"periodic", "foextrap", "slip_wall", "external"}:
                values = [0.0] * ncomp
                analytic_programs = []
                analytic_clock = None
            else:
                expressions = face.get("values")
                if not isinstance(expressions, list) or len(expressions) != ncomp:
                    raise ValueError(
                        "compiled Dirichlet boundary must exactly cover every state component"
                    )
                protocols = {
                    expression.get("protocol")
                    for expression in expressions
                    if isinstance(expression, dict)
                }
                if protocols == {"pops.analytic.scalar.v1"}:
                    clocks = set()
                    from pops.analytic import ScalarExpr

                    analytic_expressions = []
                    for expression in expressions:
                        analytic = ScalarExpr.from_data(expression["value"])
                        analytic_expressions.append(analytic)
                        clocks.update(clock.qualified_id for clock in analytic.time_clocks())
                    if len(clocks) > 1:
                        raise ValueError("compiled analytic boundary face mixes logical Clocks")
                    analytic_clock = next(iter(clocks), None)
                    lowered = lower_analytic_components(
                        [expression.to_data() for expression in analytic_expressions],
                        frame_id=data["frame_id"],
                        bindings=params,
                        time_clock_id=analytic_clock,
                    )
                    analytic_programs = [
                        {"opcodes": list(opcodes), "literals": list(literals)}
                        for opcodes, literals in lowered
                    ]
                    values = [0.0] * ncomp
                elif protocols:
                    raise TypeError(
                        "compiled Dirichlet boundary mixes unsupported expression protocols"
                    )
                else:
                    analytic_programs = []
                    analytic_clock = None
                    values = []
                    for index, expression in enumerate(expressions):
                        value = eval_expression_key(
                            expression,
                            environment,
                            where="compiled boundary face %d component %d"
                            % (int(face["ordinal"]), index),
                        )
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            raise TypeError(
                                "compiled boundary expression did not bind to a real scalar")
                        values.append(float(value))
            faces.append({
                "ordinal": int(face["ordinal"]),
                "geometry": face.get("geometry"),
                "producer": face.get("producer"),
                "type": face["type"],
                "representation": representation,
                "converter": converter,
                "values": values,
                "analytic_programs": analytic_programs,
                "analytic_clock": analytic_clock,
            })
        faces.sort(key=lambda row: row["ordinal"])

        component_regions = []
        for template in data["component_region_templates"]:
            row = dict(template)
            parameters = []
            for reference in row.get("parameters", []):
                qid = reference.get("qualified_id")
                handle = handles_by_qid.get(qid)
                if handle is None:
                    raise ValueError(
                        "boundary component parameter %s is absent from BindSchema values" % qid
                    )
                expected = reference.get("handle")
                if not isinstance(handle, Handle) or handle.canonical_identity() != expected:
                    raise ValueError(
                        "boundary component parameter %s changed qualified Handle identity" % qid
                    )
                value = values_by_qid[qid]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        "boundary component parameter %s must bind to a real scalar" % qid
                    )
                parameters.append({"qualified_id": qid, "value": float(value)})
            row["parameters"] = parameters
            component_regions.append(row)

        evidence = {
            "schema_version": 1,
            "authority_type": "prepared_boundary_plan",
            "source_plan": data.get("source_plan"),
            "state": data.get("state"),
            "required_depth": int(data["required_depth"]),
            "faces": faces,
            "corner_required": bool(data.get("corner_policies")),
            "residual_contributions": data.get("residual_contributions", []),
            "linearization_contributions": data.get("linearization_contributions", []),
            "residual_report": self.residual_report().to_dict(),
            "linearization_report": self.linearization_report().to_dict(),
            "interfaces": data.get("interfaces", []),
            # The endpoint rows were proved from owner-qualified BoundaryHandles by the
            # resolved GhostProducerPlan.  They are executable topology evidence, not
            # authoring metadata, and must survive the detached compile -> bind boundary.
            "interface_endpoints": data.get("interface_endpoints", []),
            "interface_component_bindings": data.get("interface_component_bindings", []),
            "omitted_interface_faces": list(data.get("omitted_interface_faces", [])),
            "periodic_identifications": list(data.get("periodic_identifications", [])),
            "ghost_plan_identity": self.canonical_id,
            "producer_order": list(data["producer_order"]),
            "component_regions": component_regions,
        }
        evidence["identity"] = make_identity(
            "prepared-boundary-plan",
            semantic_value(evidence, where="compiled prepared boundary plan"),
        ).token
        return evidence


__all__ = [
    "BoundaryLinearizationReport", "BoundaryResidualReport", "CompiledBoundaryPlan",
]
