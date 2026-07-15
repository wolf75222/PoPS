"""Resolve field topology and physical-face laws into exact native records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from pops.codegen.lowering_coverage import (
    LoweringCoverageReport,
    LoweringCoverageRow,
    LoweringRejection,
)
from pops.fields._identity import strict_field_data
from pops.fields.bcs import (
    AllPhysicalBoundaries,
    AxisBoundary,
    Dirichlet,
    Mixed,
    Neumann,
    Periodic,
)
from pops.fields.discretization import FieldDiscretizationProtocol


@dataclass(frozen=True, slots=True)
class FieldLayoutContract:
    """Small topology interface consumed by field lowering, independent of layout classes."""

    kind: str
    mesh: Any
    embedded_boundary: Any
    levels: int
    transition_ratios: tuple[int, ...]
    level_refinements: tuple[int, ...]


def _exact_positive_int(value: Any, *, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer, never bool" % where)
    if value < minimum:
        raise ValueError("%s must be >= %d" % (where, minimum))
    return value


def field_layout_contract(layout: Any) -> FieldLayoutContract:
    """Resolve the open layout protocol needed by elliptic field installation.

    The contract is selected by typed capability evidence, never by a concrete layout class.  The
    mesh provider is structural so an extension can expose the same contract without inheriting a
    PoPS layout implementation.  Missing or contradictory evidence is rejected rather than filled
    with uniform/one-level defaults.
    """

    capabilities = getattr(layout, "capabilities", None)
    if not callable(capabilities):
        raise TypeError("field layout must implement capabilities()")
    evidence = capabilities()
    to_dict = getattr(evidence, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("field layout capabilities must implement to_dict()")
    data = to_dict()
    if not isinstance(data, dict):
        raise TypeError("field layout capability evidence must be a dict")
    kind = data.get("layout")
    if kind == "amr":
        levels = _exact_positive_int(
            data.get("max_levels"), where="AMR field layout max_levels", minimum=1)
        raw_ratios = data.get("transition_ratios")
        if not isinstance(raw_ratios, (list, tuple)):
            raise TypeError(
                "AMR field layout transition_ratios must be an ordered integer sequence")
        ratios = tuple(
            _exact_positive_int(
                value,
                where="AMR field layout transition_ratios[%d]" % index,
                minimum=2,
            )
            for index, value in enumerate(raw_ratios)
        )
        if len(ratios) != levels - 1:
            raise ValueError(
                "AMR field layout transition_ratios must contain exactly one ratio per "
                "coarse/fine transition")
        refinements = [1]
        for ratio in ratios:
            refinements.append(refinements[-1] * ratio)
        candidates = [
            value for value in (getattr(layout, "grid", None), getattr(layout, "base", None))
            if value is not None
        ]
        if len(candidates) != 1:
            raise TypeError(
                "AMR field layout must expose exactly one grid/base mesh provider")
        embedded = getattr(layout, "embedded_boundary", None)
        if embedded is not None:
            raise TypeError(
                "AMR field layout advertises an embedded boundary without a typed field "
                "topology provider")
        return FieldLayoutContract(
            kind, candidates[0], None, levels, ratios, tuple(refinements))
    if kind == "uniform":
        levels = _exact_positive_int(
            data.get("levels"), where="uniform field layout levels")
        if levels != 1:
            raise ValueError("uniform field layout must advertise exactly one level")
        mesh = getattr(layout, "mesh", None)
        if mesh is None:
            raise TypeError("uniform field layout must expose its mesh provider")
        return FieldLayoutContract(
            kind, mesh, getattr(layout, "embedded_boundary", None), levels, (), (1,))
    raise ValueError(
        "field layout capabilities require layout='uniform' or layout='amr'; got %r" % kind)


def _mesh_periodic(mesh: Any) -> bool:
    periodic = getattr(mesh, "periodic", None)
    if isinstance(periodic, bool):
        return periodic
    topology = getattr(mesh, "topology", None)
    periodic_axes = getattr(topology, "periodic_axes", None)
    axis_pairs = getattr(topology, "axis_pairs", None)
    if isinstance(periodic_axes, tuple) and isinstance(axis_pairs, tuple) and axis_pairs:
        if not periodic_axes:
            return False
        if len(periodic_axes) == len(axis_pairs):
            return True
        raise TypeError(
            "field topology cannot lower an all-boundaries condition onto a partially periodic "
            "CartesianGrid; select typed per-axis field boundaries")
    raise TypeError("field topology requires an exact mesh periodicity contract")


def _reject(
    rows: list[LoweringCoverageRow], source: str, gate: str, message: str,
) -> NoReturn:
    report = LoweringCoverageReport((*rows, LoweringCoverageRow(
        source, "rejected", gate=gate)))
    raise LoweringRejection(
        message, coverage_report=report, source=source, gate=gate)


def _native_scalar(value: Any, *, name: str, source: str,
                   rows: list[LoweringCoverageRow], unknown: Any,
                   placeholder: float = 0.0) -> tuple[float, Any | None, bool]:
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        from pops._ir.expr import Expr
        from pops.model import Handle
        if not isinstance(value, (Expr, Handle)):
            _reject(rows, source, "field.boundary.expression_not_lowerable",
                    "%s must be a finite scalar or a typed symbolic Expr/Handle" % name)
        from pops.fields._references import collect_references
        references = collect_references(value)
        unsupported = [reference for reference in references
                       if getattr(reference, "kind", None)
                       not in {"parameter", "state", "field"} and
                       reference.canonical_identity() != unknown.canonical_identity()]
        if unsupported:
            _reject(rows, source, "field.boundary.prepared_dependency_not_native",
                    "%s references unsupported boundary dependencies %s"
                    % (name, [reference.qualified_id for reference in unsupported]))

        from pops._ir.expr import Const, Var, _Bin, Neg, Sqrt, Abs, Sign, Pow
        from pops._ir.handle_expr import ValueExpr
        from pops._ir.values import RuntimeParamRef
        from pops.fields.boundary_values import BoundaryValue, LogicalTimeValue

        def validate_expression(node: Any) -> None:
            if isinstance(node, (Const, RuntimeParamRef, BoundaryValue, LogicalTimeValue)):
                return
            if isinstance(node, ValueExpr):
                if getattr(node.handle, "kind", None) in {"state", "field"} \
                        and node.handle != unknown:
                    _reject(rows, source, "field.boundary.ambiguous_value_handle",
                            "%s reads %s without an explicit component/sample contract; use "
                            "pops.fields.boundary_value(handle, component)"
                            % (name, node.handle.qualified_id))
                return
            if isinstance(node, Var):
                _reject(rows, source, "field.boundary.unqualified_variable",
                        "%s contains unqualified Var(%r); boundary dependencies require typed, "
                        "owner-qualified handles" % (name, node.name))
            if isinstance(node, (Neg, Sqrt, Abs, Sign)):
                validate_expression(node.a)
                return
            if isinstance(node, (Pow, _Bin)):
                validate_expression(node.a)
                validate_expression(node.b)
                return
            _reject(rows, source, "field.boundary.expression_node_not_native",
                    "%s contains unsupported symbolic node %s"
                    % (name, type(node).__name__))

        if isinstance(value, Expr):
            validate_expression(value)
        iterate_dependent = any(
            reference.canonical_identity() == unknown.canonical_identity()
            for reference in references)
        rows.append(LoweringCoverageRow(
            "%s:expression:%s" % (source, name.lower().replace(" ", "_")),
            "lowered", ("field-boundary-generated-expression",),
        ))
        return placeholder, value, iterate_dependent
    return float(value), None, False


def _condition_record(condition: Any, *, source: str,
                      rows: list[LoweringCoverageRow], unknown: Any) -> dict[str, Any]:
    if isinstance(condition, Dirichlet):
        value, dynamic, iterate = _native_scalar(
            condition.value, name="Dirichlet value", source=source, rows=rows,
            unknown=unknown)
        return {"type": "dirichlet", "alpha": 1.0, "beta": 0.0,
                "value": value,
                "dynamic": ({"value": strict_field_data(dynamic)}
                            if dynamic is not None else {}),
                "iterate_dependent": iterate}
    if isinstance(condition, Neumann):
        value, dynamic, iterate = _native_scalar(
            condition.flux, name="Neumann flux", source=source, rows=rows,
            unknown=unknown)
        return {"type": "neumann", "alpha": 0.0, "beta": 1.0,
                "value": value,
                "dynamic": ({"value": strict_field_data(dynamic)}
                            if dynamic is not None else {}),
                "iterate_dependent": iterate}
    if isinstance(condition, Mixed):
        alpha, dynamic_alpha, iterate_alpha = _native_scalar(
            condition.alpha, name="Mixed alpha", source=source, rows=rows,
            unknown=unknown, placeholder=1.0)
        beta, dynamic_beta, iterate_beta = _native_scalar(
            condition.beta, name="Mixed beta", source=source, rows=rows,
            unknown=unknown, placeholder=1.0)
        if dynamic_alpha is None and dynamic_beta is None and alpha == 0.0 and beta == 0.0:
            _reject(rows, source, "field.boundary.mixed_degenerate",
                    "Mixed boundary alpha and beta cannot both be zero")
        value, dynamic_value, iterate_value = _native_scalar(
            condition.value, name="Mixed value", source=source, rows=rows,
            unknown=unknown)
        dynamic = {key: strict_field_data(item) for key, item in (
            ("alpha", dynamic_alpha), ("beta", dynamic_beta), ("value", dynamic_value))
                   if item is not None}
        return {"type": "mixed", "alpha": alpha, "beta": beta,
                "value": value, "dynamic": dynamic,
                "iterate_dependent": iterate_alpha or iterate_beta or iterate_value}
    if isinstance(condition, Periodic):
        return {"type": "periodic", "alpha": 0.0, "beta": 0.0, "value": 0.0,
                "dynamic": {}, "iterate_dependent": False}
    _reject(rows, source, "field.boundary.condition_not_native",
            "field boundary condition %s has no native residual lowering"
            % type(condition).__name__)


def boundary_plan(
    name: str,
    plan: FieldDiscretizationProtocol,
    rows: list[LoweringCoverageRow],
    layout: Any,
    unknown: Any,
) -> tuple[str, tuple[dict[str, Any], ...] | None]:
    """Resolve all four Cartesian faces, retaining exact Robin coefficients."""
    if not plan.boundaries:
        contract = field_layout_contract(layout)
        periodic = _mesh_periodic(contract.mesh)
        kind = "periodic" if periodic else "dirichlet"
        record = ({"type": kind, "alpha": 0.0 if periodic else 1.0,
                   "beta": 0.0, "value": 0.0, "dynamic": {},
                   "iterate_dependent": False},) * 4
        rows.append(LoweringCoverageRow(
            "field:%s:boundaries" % name, "derived",
            rule="resolved layout topology selects %s BC" % kind))
        return "explicit", record
    face_index = {(0, "lo"): 0, (0, "hi"): 1, (1, "lo"): 2, (1, "hi"): 3}
    faces: list[dict[str, Any] | None] = [None, None, None, None]
    for index, binding in enumerate(plan.boundaries):
        source = "field:%s:boundary:%d" % (name, index)
        selector = binding.selector
        record = _condition_record(
            binding.condition, source=source, rows=rows, unknown=unknown)
        if isinstance(selector, AllPhysicalBoundaries):
            selected = range(4)
        elif isinstance(selector, AxisBoundary):
            if (selector.axis, selector.side) not in face_index:
                _reject(rows, source, "field.boundary.dimension_not_native",
                        "native field solver is 2-D; boundary axis %d is unsupported" % selector.axis)
            selected = (face_index[(selector.axis, selector.side)],)
        else:
            _reject(rows, source, "field.boundary.selector_not_native",
                    "field %r uses a boundary selector the native Cartesian solver cannot lower"
                    % name)
        for face in selected:
            if faces[face] is not None:
                _reject(rows, source, "field.boundary.duplicate_face",
                        "field %r assigns a physical face more than once" % name)
            faces[face] = dict(record)
        rows.append(LoweringCoverageRow(source, "lowered", (
            "field-install:%s:boundary-residual" % name,)))
    if any(face is None for face in faces):
        _reject(rows, "field:%s:boundaries" % name, "field.boundary.incomplete",
                "field %r boundary plan does not cover all 2-D physical faces" % name)
    complete_faces = tuple(face for face in faces if face is not None)
    if len(complete_faces) != 4:
        raise RuntimeError("validated field boundary plan lost a physical face")
    for lo, hi, axis in ((0, 1, "x"), (2, 3, "y")):
        if (complete_faces[lo]["type"] == "periodic") != (
                complete_faces[hi]["type"] == "periodic"):
            _reject(rows, "field:%s:boundaries" % name,
                    "field.boundary.periodic_pair_incomplete",
                    "field %r marks only one %s face periodic; periodic topology is paired"
                    % (name, axis))
    return "explicit", complete_faces


def topology_recipe(layout: Any) -> dict[str, Any]:
    contract = field_layout_contract(layout)
    mesh = contract.mesh
    embedded = contract.embedded_boundary
    periodic = _mesh_periodic(mesh)
    hierarchy = ("amr-composite-cell-graph" if contract.kind == "amr"
                 else "uniform-cell-graph")
    connectivity = {
        "graph": hierarchy,
        "stencil": "axis-neighbor",
        "periodic_identifications": "all-axes" if periodic else "none",
        "coarse_fine_identifications": (
            [
                {
                    "coarse_level": index,
                    "fine_level": index + 1,
                    "transition_ratio": ratio,
                }
                for index, ratio in enumerate(contract.transition_ratios)
            ]
            if contract.kind == "amr" else []),
        "material_predicate": "all-cells" if embedded is None else "embedded-boundary",
    }
    return {
        "layout_type": type(layout).__name__,
        "mesh_type": type(mesh).__name__,
        "mesh": strict_field_data(mesh.options()),
        "embedded_boundary": (
            None if embedded is None else {
                "type": type(embedded).__name__,
                "options": strict_field_data(embedded.options()),
            }
        ),
        "levels": contract.levels,
        "transition_ratios": list(contract.transition_ratios),
        "level_refinements": list(contract.level_refinements),
        "connectivity": connectivity,
        "component_derivation": "connected-components-of-resolved-cell-graph",
        "basis_derivation": "one-constant-mode-per-connected-material-component",
    }


def boundary_dependency_pack(
    plan: FieldDiscretizationProtocol, unknown: Any,
) -> dict[str, Any]:
    """Canonical direct-buffer pack required by all dynamic physical-face laws."""
    from pops.fields.boundary_values import BoundaryValue, LogicalTimeValue
    from pops._ir.expr import Expr
    from pops._ir.visitors import _children

    states: dict[tuple[str, int], dict[str, Any]] = {}
    fields: dict[tuple[str, int], dict[str, Any]] = {}
    times: set[str] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, Expr):
            return
        stack = [value]
        while stack:
            node = stack.pop()
            if isinstance(node, BoundaryValue):
                handle = node.handle
                handle.canonical_identity()
                if handle == unknown:
                    continue
                block = getattr(handle, "block_ref", None)
                if block is None:
                    raise TypeError(
                        "boundary dependency %s is not qualified through a Case block"
                        % handle.qualified_id)
                record = {
                    "handle": handle.canonical_identity(),
                    "qualified_id": handle.qualified_id,
                    "owner_block": block.local_id,
                    "component": node.component,
                }
                if handle.kind == "state":
                    states[(handle.qualified_id, node.component)] = record
                elif handle.kind == "field":
                    record["output_key"] = handle.local_id
                    fields[(handle.qualified_id, node.component)] = record
                else:
                    raise TypeError("BoundaryValue retained unsupported Handle.kind")
            elif isinstance(node, LogicalTimeValue):
                times.add(node.coordinate)
            stack.extend(_children(node))

    for binding in plan.boundaries:
        condition = binding.condition
        for name in ("value", "flux", "alpha", "beta"):
            visit(getattr(condition, name, None))
    return {
        "schema_version": 1,
        "states": [states[key] for key in sorted(states)],
        "fields": [fields[key] for key in sorted(fields)],
        "logical_time": sorted(times),
    }


__all__ = [
    "FieldLayoutContract",
    "boundary_dependency_pack",
    "boundary_plan",
    "field_layout_contract",
    "topology_recipe",
]
