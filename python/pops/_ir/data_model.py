"""The one array type of a resolved plan: placements, IR types and layouts (spec 0.3 DM1 to DM3).

N10 phase 1, lot 1a. Three facts of a resolved plan, computed here and read by nothing but the
manifest reporter and the tests (review of the N10 note, A2):

* the **support placement** of each block. κ assigns a layout to each support (§4.1), and the
  b6 forms do not say which blocks share a support: co-located blocks (M12, M22) and two domains
  glued by an interface (the T10 multi-block example) both state one frame and one layout. The
  plan resolves them: on one layout every block has the default placement, except the endpoints
  of a cross-block interface, which get one placement each. « Endpoint » is the one predicate
  ``carries_cross_block_interface`` that the plan's coarse distribution and the C50 refusal read
  (A3). This is the plan's resolution of b6 forms into supports, not an IR26 admission (driver's
  P5); N15 replaces it with the supports that the surface writes;
* the **IR type** of each Program value, from its value signature (``ir_type``). A field leaf of
  block b is ``Array(cells(κ_p(b)), T)``, or ``Array(levels(H_p(b)), T)`` on an adaptive layout;
  a finite array is ``Array(range_set, T)``; scalars, products and variants are their IR1 types;
  operators, effects and solve outcomes are ``NotData``. What the plan cannot type yet is
  ``Untyped`` with its reason, never an exception (A1);
* the **layout record** of each data value (DM3): its layout, distribution and origin, the plan's
  choice or a b6 author choice read as a constraint.

``state``, ``rhs`` and a plain field are roles, not types: a state and its rate have one type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops._ir import arrays as A
from pops._ir import index_sets as S

__all__ = [
    "LayoutRecord", "Placements", "data_layout_and_distribution", "ir_type", "support_placements",
]

_FIELD_ROLES = {"state": "state", "rhs": "rhs", "scalar_field": "field", "vector_field": "field"}
_NOT_DATA = frozenset((
    "operator", "matrix_free_op", "effect", "condensed_coeffs", "solve_outcome", "unit",
))


@dataclass(frozen=True)
class Placements:
    """The support placement of each block of a plan, with the layout facts that type it."""

    configurations: dict[str, S.Configuration]
    hierarchies: dict[str, S.HierarchyInstance]
    adaptive: dict[str, bool]
    layouts: dict[str, str]
    default: S.Configuration | None

    def domain(self, block: str) -> Any:
        """The index set of a field leaf of ``block``."""
        if self.adaptive[block]:
            return S.levels(self.hierarchies[block])
        return S.cells(self.configurations[block])


def _periodic(row: Any) -> tuple[bool, ...]:
    try:
        flags = tuple(bool(flag) for flag in row.native_spatial_layout.periodicity)
    except Exception:  # noqa: BLE001 - a layout without native periodicity reports none
        flags = ()
    rank = len(tuple(row.geometry.cells))
    return flags if len(flags) == rank else (False,) * rank


def support_placements(plan: Any) -> Placements:
    """Resolve the b6 forms of ``plan`` into support placements (§4.1, P5)."""
    from pops.amr._resolution import carries_cross_block_interface

    rows = {row.handle.qualified_id: row for row in plan.layout_plan.layouts}
    block_layouts = {row.subject.local_id: row.layout.qualified_id
                     for row in plan.layout_plan.assignments if row.subject_kind == "block"}
    configurations: dict[str, S.Configuration] = {}
    hierarchies: dict[str, S.HierarchyInstance] = {}
    adaptive: dict[str, bool] = {}
    shared: dict[str, S.Configuration] = {}
    for block in plan.blocks:
        layout_id = block_layouts[block.name]
        row = rows[layout_id]
        if carries_cross_block_interface(block.numerics):
            identity = "%s@block:%s" % (layout_id, block.instance_owner_qid or block.name)
        else:
            identity = "%s@support" % layout_id
        configuration = S.Configuration(identity, tuple(row.geometry.cells), _periodic(row))
        if identity.endswith("@support"):
            configuration = shared.setdefault(layout_id, configuration)
        configurations[block.name] = configuration
        adaptive[block.name] = bool(row.adaptive)
        if row.adaptive:
            hierarchies[block.name] = S.HierarchyInstance(
                identity + "#hierarchy", configuration, len(tuple(row.levels)),
                tuple(row.transition_ratios))
    default = None
    if len(rows) == 1 and len(shared) == 1 and all(
            value == next(iter(shared.values())) for value in configurations.values()):
        default = next(iter(shared.values()))
    return Placements(configurations, hierarchies, adaptive, dict(block_layouts), default)


def _components(space: Any) -> tuple[str, ...] | None:
    components = getattr(space, "components", None)
    if components is None:
        return None
    names = tuple(str(name) for name in components)
    return names or None


def _element(names: tuple[str, ...] | None, count: Any) -> Any:
    if names:
        return A.Float64 if len(names) == 1 else A.Product(*((name, A.Float64) for name in names))
    if type(count) is int and count >= 1:
        if count == 1:
            return A.Float64
        return A.Product(*(("c%d" % index, A.Float64) for index in range(count)))
    return None


def _block_name(block: Any) -> str | None:
    if block is None:
        return None
    if isinstance(block, str):
        return block
    return getattr(block, "local_id", None)


def _range_set(signature: Any, program: str) -> Any:
    family = signature.get("family")
    if type(family) is not int:
        return A.Untyped("an index set without a family identity")
    return S.range_set("%s/family:%d" % (program, family))


def _signature_type(signature: Any, placements: Placements, program: str,
                    count: Any = None) -> Any:
    kind = signature.get("vtype")
    if kind == "scalar":
        return A.Float64
    if kind == "int64":
        return A.Int64
    if kind == "bool":
        return A.Bool
    if kind in ("state", "rhs", "scalar_field"):
        block = _block_name(signature.get("block"))
        element = _element(_components(signature.get("space")), count)
        if element is None:
            return A.Untyped("a field leaf whose components the plan does not state")
        if block is None:
            if placements.default is None:
                return A.Untyped("a field without a block on a layout with several supports "
                                 "(N15 names its support)")
            return A.Array(S.cells(placements.default), element)
        if block not in placements.configurations:
            return A.Untyped("a field of block %r that the plan does not place" % block)
        return A.Array(placements.domain(block), element)
    if kind in ("product", "variant", "continuation_base", "continuation_candidate",
                "continuation_assessment", "continuation_accepted"):
        members = signature.get("members")
        if not members:
            return A.Untyped("a %s without members in its signature" % kind)
        rows = tuple((name, _signature_type(child, placements, program))
                     for name, child in members)
        return A.Variant(*rows) if kind == "variant" else A.Product(*rows)
    if kind == "index_set":
        domain = _range_set(signature, program)
        return domain if isinstance(domain, A.Untyped) else A.IndexSetType(domain)
    if kind == "index":
        domain = _range_set(signature, program)
        return domain if isinstance(domain, A.Untyped) else A.Index(domain)
    if kind == "array":
        domain = _range_set(signature, program)
        if isinstance(domain, A.Untyped):
            return domain
        element = signature.get("element")
        if element is None:
            return A.Untyped("an array without an element signature")
        return A.Array(domain, _signature_type(element, placements, program))
    if kind == "result":
        success = signature.get("success")
        if success is None:
            return A.Untyped("a result without a success signature")
        return A.Result(_signature_type(success, placements, program),
                        A.Untyped("the failure type of a result is not in its signature"))
    if kind in _NOT_DATA:
        return A.NotData(kind)
    return A.Untyped("value type %r is typed by a later lot" % (kind,))


def ir_type(value: Any, placements: Placements, *, program: str = "program") -> Any:
    """The IR type of one Program value; ``Untyped`` with a reason, never an exception (A1)."""
    kind = value.vtype
    if kind in _NOT_DATA:
        return A.NotData(kind)
    shape = value.logical_shape or {}
    count = shape.get("n_comp") if hasattr(shape, "get") else None
    try:
        from pops.time.value_signatures import value_signature

        signature = value_signature(value)
    except Exception:  # noqa: BLE001 - A1: what has no signature is reported, not refused
        signature = {"vtype": kind, "block": value.block, "space": value.space}
    return _signature_type(signature, placements, program, count)


@dataclass(frozen=True)
class LayoutRecord:
    """The layout of one data value of the plan (DM3), as the manifest reports it."""

    value: str
    spelling: str
    role: str | None
    type: str
    layout: str
    distribution: str
    origin: str

    def to_data(self) -> dict[str, Any]:
        return {"value": self.value, "spelling": self.spelling, "role": self.role,
                "type": self.type, "layout": self.layout, "distribution": self.distribution,
                "origin": self.origin}


def _coarse(plan: Any, layout_id: str) -> tuple[str, str]:
    authorities = getattr(plan, "layout_amr_authorities", None) or {}
    row = authorities.get(layout_id)
    if row is None:
        return "hierarchy distribution", "plan"
    options = row.authorities.hierarchy.plan.patch_generation.options.to_data()
    distribution = ("coarse level partitioned" if options["distribute_coarse"]
                    else "coarse level replicated")
    authored = None
    for layout in plan.layout_plan.layouts:
        if layout.handle.qualified_id == layout_id:
            patch = (layout.options or {}).get("patch_layout") or {}
            authored = patch.get("distribute_coarse")
    return distribution, ("plan" if authored is None else "constraint")


def _layout(value: Any, value_type: Any, placements: Placements, plan: Any) -> tuple[str, ...]:
    if isinstance(value_type, A.NotData):
        return "none", "none", "none"
    if isinstance(value_type, A.Array) and isinstance(value_type.domain, (S.Entities, S.Union,
                                                                          S.Level)):
        block = _block_name(value.block)
        if block is not None and placements.adaptive.get(block):
            distribution, origin = _coarse(plan, placements.layouts[block])
            return "LevelPatches", distribution, origin
        return "Patches", "plan box decomposition", "plan"
    if isinstance(value_type, A.Array):
        try:
            from pops.time.value_signatures import value_signature

            ownership = value_signature(value).get("ownership", "replicated")
        except Exception:  # noqa: BLE001 - A1
            ownership = "replicated"
        if ownership == "replicated":
            return "Replicated", "every rank", "constraint"
        return "Contiguous", "partitioned by the offsets the case wrote", "constraint"
    if isinstance(value_type, A.Untyped):
        return "untyped", "untyped", "untyped"
    if isinstance(value_type, (A.Product, A.Variant, A.Result, A.Iteration)):
        return "members", "per member", "plan"
    return "Local", "every rank", "plan"


def data_layout_and_distribution(plan: Any) -> dict[str, Any]:
    """The manifest field ``data_layout_and_distribution`` (CC10) of a resolved plan."""
    placements = support_placements(plan)
    program = str(getattr(plan.time, "name", None) or "program")
    records = []
    for value in plan.time._values:
        value_type = ir_type(value, placements, program=program)
        layout, distribution, origin = _layout(value, value_type, placements, plan)
        records.append(LayoutRecord(
            value=str(value.name if getattr(value, "name", None) else value.id),
            spelling=value.vtype, role=_FIELD_ROLES.get(value.vtype), type=repr(value_type),
            layout=layout, distribution=distribution, origin=origin).to_data())
    return {
        "schema_version": 1,
        "placements": {block: configuration.identity
                       for block, configuration in sorted(placements.configurations.items())},
        "values": records,
        "untyped": sum(1 for row in records if row["layout"] == "untyped"),
    }
