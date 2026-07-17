"""Detached executable boundary plans retained after compilation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

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

    def runtime_boundary_data(self, params: Any) -> dict[str, Any]:
        """Bind scalar values through one generic evaluator, never an authoring callback."""
        from pops.model import Handle, ParamHandle
        from pops.model._bind_expression import eval_expression_key

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
                    "periodic", "foextrap", "dirichlet", "external"}:
                raise ValueError("compiled boundary face has no executable producer type")
            if face["type"] in {"periodic", "foextrap", "external"}:
                values = [0.0] * ncomp
            else:
                expressions = face.get("values")
                if not isinstance(expressions, list) or len(expressions) != ncomp:
                    raise ValueError(
                        "compiled Dirichlet boundary must exactly cover every state component"
                    )
                values = []
                for index, expression in enumerate(expressions):
                    value = eval_expression_key(
                        expression,
                        environment,
                        where="compiled boundary face %d component %d"
                        % (int(face["ordinal"]), index),
                    )
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("compiled boundary expression did not bind to a real scalar")
                    values.append(float(value))
            faces.append({
                "ordinal": int(face["ordinal"]),
                "geometry": face.get("geometry"),
                "producer": face.get("producer"),
                "type": face["type"],
                "values": values,
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
            "interfaces": data.get("interfaces", []),
            # The endpoint rows were proved from owner-qualified BoundaryHandles by the
            # resolved GhostProducerPlan.  They are executable topology evidence, not
            # authoring metadata, and must survive the detached compile -> bind boundary.
            "interface_endpoints": data.get("interface_endpoints", []),
            "interface_component_bindings": data.get("interface_component_bindings", []),
            "omitted_interface_faces": list(data.get("omitted_interface_faces", [])),
            "ghost_plan_identity": self.canonical_id,
            "producer_order": list(data["producer_order"]),
            "component_regions": component_regions,
        }
        evidence["identity"] = make_identity(
            "prepared-boundary-plan",
            semantic_value(evidence, where="compiled prepared boundary plan"),
        ).token
        return evidence


__all__ = ["CompiledBoundaryPlan"]
