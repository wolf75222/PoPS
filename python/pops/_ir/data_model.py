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
* the **layout record** of each data value (DM3), at every depth of the Program (the bodies of
  its control blocks included): its layout, distribution and origin, the plan's choice or a b6
  author choice read as a constraint, and its residency as G13 stores it.

``state``, ``rhs`` and a plain field are roles, not types: a state and its rate have one type.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from pops._ir import arrays as A
from pops._ir import index_sets as S

__all__ = [
    "LayoutRecord", "Placements", "data_layout_and_distribution", "field_role", "ir_type",
    "is_field_leaf", "is_finite_array", "is_finite_index", "support_placements", "value_kind",
]

_FIELD_ROLES = {"state": "state", "rhs": "rhs", "scalar_field": "field", "vector_field": "field"}
# Spellings of the §3.1 table that a later lot types, with that lot (note §3.10, review of #749 R2).
_LATER = {
    "vector_field": ("no builder spells a value so: Local.vector_field returns a scalar_field of d "
                     "components, typed as a field leaf"),
    "state_geometry": ("G03 types it: the swept geometry of a moving state, held by the runtime "
                       "and checked against its coordinate law (ALE: R26, D18, D21, probe M28)"),
    "mask": "phase 2 types it as an array of Bool with the one codec (DM4, CN11)",
}
_CONTINUATION = frozenset((
    "continuation_base", "continuation_candidate", "continuation_assessment",
    "continuation_accepted",
))
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


def _any_block(placements: Placements) -> str:
    """A block of the one support; with a default placement, every block shares it."""
    return next(iter(placements.configurations))


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
            return A.Untyped("a field leaf with no component count in its space, its shape or "
                             "its attributes")
        if block is None:
            if placements.default is None:
                return A.Untyped("a field without a block on a layout with several supports "
                                 "(N15 names its support)")
            # The one support of the layout: its cells, or its levels on an adaptive layout.
            return A.Array(placements.domain(_any_block(placements)), element)
        if block not in placements.configurations:
            return A.Untyped("a field of block %r that the plan does not place" % block)
        return A.Array(placements.domain(block), element)
    if kind in _CONTINUATION:
        return A.Untyped("a %s record carries its members in its codec (CN11, Phase 2)" % kind)
    if kind in ("product", "variant"):
        members = signature.get("members")
        if not members:
            if kind == "product":
                return A.Product()
            return A.Untyped("a variant without cases, which IR1 does not type")
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
    if kind == "csr":
        return _csr_type(signature, placements, program)
    if kind == "result":
        success = signature.get("success")
        if success is None:
            return A.Untyped("a result without a success signature")
        return A.Result(_signature_type(success, placements, program),
                        A.Untyped("the failure type of a result is not in its signature"))
    if kind in _NOT_DATA:
        return A.NotData(kind)
    if kind in _LATER:
        return A.Untyped("value type %r: %s" % (kind, _LATER[kind]))
    return A.Untyped("value type %r is in no row of the §3.1 table" % (kind,))


def _csr_type(signature: Any, placements: Placements, program: str) -> Any:
    """DM4 (N10 lot 4, addendum §5): ``Product(offsets, indices)``, as the runtime stores it."""
    rows, columns = signature.get("rows"), signature.get("columns")
    if not hasattr(rows, "get") or not hasattr(columns, "get"):
        return A.Untyped("a csr without the signatures of its rows and columns")
    row_family = rows.get("family")
    if type(row_family) is not int:
        return A.Untyped("a csr whose rows have no family identity")
    element = columns.get("element")
    target = _signature_type(element, placements, program) if element is not None else None
    if not isinstance(target, A.Index):
        return A.Untyped("a csr whose columns are no typed index")
    # offsets: on each rank, the local rows plus one, from 0 (distributed_array.hpp, CSR); a
    # finite set of its own, stored Local. indices: the requests array the csr was built from.
    offsets = A.Array(S.range_set("%s/family:%d/local_rows+1" % (program, row_family)), A.Int64)
    indices = _signature_type(columns, placements, program)
    if not isinstance(indices, A.Array):
        return A.Untyped("a csr whose requests are no typed array")
    return A.Product(("offsets", offsets), ("indices", indices))


def _published_fields(value: Any, placements: Placements, program: str) -> Any:
    """``Product(component=Array(D, Float64), ...)``: one source component per binding (§3.1)."""
    attrs = getattr(value, "attrs", None) or {}
    bindings = tuple(attrs.get("bindings") or ()) if hasattr(attrs, "get") else ()
    inputs = tuple(getattr(value, "inputs", ()) or ())
    if not bindings or len(bindings) != len(inputs):
        return A.Untyped("a field publication whose bindings do not name its inputs one by one")
    members = []
    for binding, source in zip(bindings, inputs, strict=True):
        source_type = ir_type(source, placements, program=program)
        if isinstance(source_type, A.Untyped):
            return source_type
        if not isinstance(source_type, A.Array):
            return A.Untyped("a field publication of a %r input" % (source.vtype,))
        members.append((str(binding["component"]), A.Array(source_type.domain, A.Float64)))
    return A.Product(*members)


def ir_type(value: Any, placements: Placements, *, program: str = "program") -> Any:
    """The IR type of one Program value; ``Untyped`` with a reason, never an exception (A1)."""
    kind = value.vtype
    if kind in _NOT_DATA:
        return A.NotData(kind)
    shape = value.logical_shape or {}
    count = shape.get("n_comp") if hasattr(shape, "get") else None
    attrs = getattr(value, "attrs", None) or {}
    if count is None and hasattr(attrs, "get"):
        # A ``scalar_field()`` without a block carries its component count here (review T3).
        count = attrs.get("ncomp")
    if kind == "iteration_result":
        state = attrs.get("state_signature") if hasattr(attrs, "get") else None
        if state is None:
            return A.Untyped("an iteration without the signature of its state")
        return A.Iteration(_signature_type(state, placements, program))
    if kind == "fields":
        return _published_fields(value, placements, program)
    try:
        from pops.time.value_signatures import value_signature

        # A csr carries its signature in its attributes, as the distributed builder reads it.
        signature = attrs["signature"] if kind == "csr" else value_signature(value)
    except Exception:  # noqa: BLE001 - A1: what has no signature is reported, not refused
        signature = {"vtype": kind, "block": value.block, "space": value.space}
    value_type = _signature_type(signature, placements, program, count)
    inputs = tuple(getattr(value, "inputs", ()) or ())
    if isinstance(value_type, A.Untyped) and getattr(value, "op", None) == "store_history" \
            and len(inputs) == 1:
        # A history keeps the field it stores: it has the type of its input (review of #749).
        return ir_type(inputs[0], placements, program=program)
    return value_type


@dataclass(frozen=True)
class LayoutRecord:
    """The layout of one data value of the plan (DM3), as the manifest reports it."""

    value: str
    node: int
    spelling: str
    role: str | None
    type: str
    layout: str
    distribution: str
    origin: str
    residency: str
    instance: str | None = None

    def to_data(self) -> dict[str, Any]:
        return {"value": self.value, "node": self.node, "spelling": self.spelling,
                "role": self.role,
                "type": self.type, "layout": self.layout, "distribution": self.distribution,
                "origin": self.origin, "residency": self.residency, "instance": self.instance}


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


_FIELD_RESIDENCY = "device: views of the default memory space"


def _finite_residency(signature: Any) -> str:
    """G13's rules, read where they are written (``field_array``, then ``device_array``)."""
    from pops.codegen.program_emit_finite_arrays import field_array
    from pops.codegen.program_emit_finite_kokkos import device_array

    if field_array(signature):
        return "device: one column per box of the default memory space"
    if device_array(signature):
        return "device: columns of the default memory space"
    return "host: structural cells"


# DM3 (N10 lot 4, addendum §1): who chose the storage of a finite family, and how it is spread.
_REPLICATED = "every rank"
_ISSUERS = {
    "distributed_domain": ("constraint", "partitioned by the offsets the case wrote"),
    "distributed_balanced_domain": ("plan", "balanced over the execution lane"),
}


_ISSUING = frozenset(("finite_range_set", *_ISSUERS))


def _family_of(value: Any) -> Any:
    attrs: Any = getattr(value, "attrs", None) or {}
    signature: Any = attrs.get("signature")
    family = signature.get("family") if hasattr(signature, "get") else None
    return family if type(family) is int else None


class _Origins:
    """Who chose the storage of each finite set instance, read on the value graph (A3).

    A set is issued by ``range_set``, ``Distributed.domain`` or ``Distributed.balanced_domain``.
    Its origin is the case's (« constraint ») when the case wrote ``ownership=`` or offsets, the
    plan's otherwise; a set issued from a witness has the origin of the witness. A value reaches
    the issuing nodes of its family through its inputs; its origin is that of the instances it
    reaches, « constraint » when they differ. A value that reaches none falls back on every
    instance of its family in the plan, by the same rule.
    """

    def __init__(self, nodes: Any):
        self._reach: dict[int, frozenset[tuple[int, str, str]]] = {}
        self._issuers: dict[int, tuple[str, str]] = {}
        self._family: dict[int, set[tuple[str, str]]] = {}
        for value in nodes:
            if getattr(value, "op", None) in _ISSUING:
                row = self._issued(value)
                family = _family_of(value)
                if family is not None:
                    self._family.setdefault(family, set()).add(row)

    def _issued(self, value: Any) -> tuple[str, str]:
        key = id(value)
        if key in self._issuers:
            return self._issuers[key]
        op, attrs = value.op, getattr(value, "attrs", None) or {}
        inputs = tuple(getattr(value, "inputs", ()) or ())
        if op in _ISSUERS:
            row = _ISSUERS[op]
        elif not attrs.get("fresh_family") and len(inputs) == 2:
            witness = self.of(inputs[1], _family_of(inputs[1]))
            row = witness if witness is not None else ("constraint", _REPLICATED)
        else:
            row = ("plan" if attrs.get("storage_origin") == "plan" else "constraint", _REPLICATED)
        self._issuers[key] = row
        return row

    def _reached(self, value: Any) -> frozenset[tuple[int, str, str]]:
        stack = [(value, False)]
        while stack:
            node, ready = stack.pop()
            key = id(node)
            if key in self._reach:
                continue
            inputs = tuple(getattr(node, "inputs", ()) or ())
            issuing = getattr(node, "op", None) in _ISSUING
            if issuing:
                family = _family_of(node)
                origin, distribution = self._issued(node)
                self._reach[key] = frozenset(
                    () if family is None else ((family, origin, distribution),))
                continue
            if not ready:
                stack.append((node, True))
                stack.extend((item, False) for item in inputs if id(item) not in self._reach)
                continue
            rows: set[tuple[int, str, str]] = set()
            for item in inputs:
                rows |= self._reach.get(id(item), frozenset())
            self._reach[key] = frozenset(rows)
        return self._reach[id(value)]

    def of(self, value: Any, family: Any) -> tuple[str, str] | None:
        if type(family) is not int:
            return None
        rows = {(origin, distribution) for found, origin, distribution in self._reached(value)
                if found == family}
        if not rows:
            rows = self._family.get(family, set())
        if not rows:
            return None
        origins = {origin for origin, _ in rows}
        return ("constraint" if "constraint" in origins else "plan",
                "; ".join(sorted({distribution for _, distribution in rows})))


def _layout(value: Any, value_type: Any, placements: Placements, plan: Any,
            origins: Any = None) -> tuple[str, ...]:
    """Layout, distribution, origin and residency of one value."""
    if isinstance(value_type, A.NotData):
        return "none", "none", "none", "none"
    if isinstance(value_type, A.Array) and isinstance(value_type.domain, S.ProductIndexSet):
        # DM4: phase space is stored with the base on the plan's boxes and the fibre local.
        return ("ProductLayout(Patches, Local)", "plan box decomposition of the base, the fibre "
                "on every rank", "plan", _FIELD_RESIDENCY)
    if isinstance(value_type, A.Array) and isinstance(value_type.domain, (S.Entities, S.Union,
                                                                          S.Level)):
        block = _block_name(value.block)
        if block is None and placements.default is not None:
            block = _any_block(placements)
        if block is not None and placements.adaptive.get(block):
            distribution, origin = _coarse(plan, placements.layouts[block])
            return "LevelPatches", distribution, origin, _FIELD_RESIDENCY
        return "Patches", "plan box decomposition", "plan", _FIELD_RESIDENCY
    if isinstance(value_type, A.Array):
        try:
            from pops.time.value_signatures import value_signature

            signature = value_signature(value)
            ownership = signature.get("ownership", "replicated")
            residency = _finite_residency(signature)
        except Exception as exc:  # noqa: BLE001 - A1
            ownership = "replicated"
            residency = "untyped: %s: %s" % (type(exc).__name__, exc)
        family = signature.get("family") if hasattr(signature, "get") else None
        found = origins.of(value, family) if origins is not None else None
        if found is None:
            unknown = "untyped: no node of the plan issues family %r" % (family,)
            found = (unknown, _REPLICATED if ownership == "replicated" else unknown)
        if ownership == "replicated":
            return "Replicated", found[1], found[0], residency
        return "Contiguous", found[1], found[0], residency
    if isinstance(value_type, A.Untyped):
        return "untyped", "untyped", "untyped", "untyped"
    unstated = "untyped: G13 states the residency of arrays only"
    if getattr(value, "vtype", None) == "csr" and isinstance(value_type, A.Product):
        # DM4 (addendum §5): offsets and indices are Contiguous, spread like the rows.
        signature = (getattr(value, "attrs", None) or {}).get("signature") or {}
        inputs = tuple(getattr(value, "inputs", ()) or ())
        found = [origins.of(item, (signature.get(part) or {}).get("family"))
                 if origins is not None else None
                 for item, part in zip(inputs[:2], ("rows", "columns"), strict=False)]
        if len(found) < 2 or None in found:
            origin = "untyped: no node of the plan issues the rows or the requests"
        else:
            origins = {row[0] for row in found if row is not None}
            origin = "constraint" if "constraint" in origins else "plan"
        return ("members", "offsets Local, the local rows plus one on each rank; indices "
                "Contiguous, as the requests", origin, "per member")
    if isinstance(value_type, (A.Product, A.Variant, A.Result, A.Iteration)):
        return "members", "per member", "plan", "per member"
    return "Local", "every rank", "plan", unstated


# DM5 (N10 lot 3, addendum §3): the general accesses of phase 1, with the input that carries
# their index (``at`` and the local access take an index; the imports take requests or a CSR).
_GENERAL_ACCESSES = {
    "finite_at": 1, "distributed_local_at": 1, "distributed_import": 1,
    "distributed_import_csr": 1, "distributed_scatter_reduce": 1,
}
_ARGUMENTS = frozenset(("finite_index_argument", "distributed_index_argument"))
_ARRAY_READS = frozenset(_GENERAL_ACCESSES)
# Op prefixes whose value depends on the run's state: a continuation, a solve or an attempt.
_COMPUTED_PREFIXES = ("continuation_", "solve", "attempt_")


def _provenance(carrier: Any, nodes: tuple[Any, ...]) -> str:
    """IR8 provenance of an index: ``computed``, ``data`` or ``arithmetic``.

    ``computed`` when the index depends on a field leaf, a continuation or a solve (the run's
    state); ``data`` when it reads an array and none of those; ``arithmetic`` when it reads only
    the argument, constants and integer operations.
    """
    from pops.time._graph.value_traversal import reachable_values

    reached = reachable_values(carrier, nodes)
    if any(is_field_leaf(value) or str(value.op).startswith(_COMPUTED_PREFIXES)
           for value in reached):
        return "computed"
    if any(value.op in _ARRAY_READS for value in reached):
        return "data"
    return "arithmetic"


def _general_accesses(values: tuple[Any, ...], layouts: dict[int, str]) -> list[dict[str, Any]]:
    """One row per general access of the plan (DM5): its node, op, array, layout and provenance.

    The array is the one the access reaches: the array read, or the array a scatter writes.
    """
    rows = []
    for value in values:
        position = _GENERAL_ACCESSES.get(value.op)
        if position is None:
            continue
        carrier = value.inputs[position]
        if carrier.op in _ARGUMENTS:
            continue  # the index of the map itself: the identity, a typed access
        array = value if value.op == "distributed_scatter_reduce" else value.inputs[0]
        rows.append({"node": int(value.id), "op": value.op, "array": int(array.id),
                     "layout": layouts.get(int(array.id), "untyped"),
                     "provenance": _provenance(carrier, values)})
    return rows


def _hierarchy(value: Any, placements: Placements) -> str | None:
    """The identity of the hierarchy H a field leaf of an adaptive block lives on."""
    block = _block_name(value.block)
    if block is None and len(placements.hierarchies) == 1:
        block = next(iter(placements.hierarchies))
    hierarchy = placements.hierarchies.get(block) if block is not None else None
    return None if hierarchy is None else hierarchy.identity


def _instance(value: Any, layout: str, placements: Placements, memo: dict[int, Any]) -> str | None:
    """DM2 (N10 lot 3, addendum §5): H, or H' = H + "#regrid" on the target side of a regrid."""
    if layout != "LevelPatches":
        return None
    from pops.time._program.continuation_regrid import (
        MIXED_INSTANCES, TARGET_INSTANCE, hierarchy_instance)

    identity = _hierarchy(value, placements)
    if identity is None:
        return None
    side = hierarchy_instance(value, memo)
    if side == TARGET_INSTANCE:
        return identity + "#regrid"
    return "mixed" if side == MIXED_INSTANCES else identity


def _relayouts(values: tuple[Any, ...], placements: Placements) -> list[dict[str, Any]]:
    """DM6: one row per relayout of the plan. Phase 1 has one, the copy inside ``regrid_read``."""
    rows = []
    for value in values:
        if value.op != "continuation_regrid_read":
            continue
        identity = _hierarchy(value, placements)
        rows.append({
            "node": int(value.id),
            "from": {"instance": identity, "level": "node %d" % int(value.inputs[1].id),
                     "layout": "LevelPatches"},
            "to": {"instance": None if identity is None else identity + "#regrid",
                   "level": "child", "layout": "LevelPatches"},
            "map": "Parent(r)",
            "ratio": "bound per child: source level to child level, the identity when equal",
        })
    return rows


def data_layout_and_distribution(plan: Any) -> dict[str, Any]:
    """The manifest field ``data_layout_and_distribution`` (CC10) of a resolved plan."""
    try:
        placements = support_placements(plan)
        unplaced = None
    except Exception as exc:  # noqa: BLE001 - A1: a plan the placement cannot read is reported
        placements = Placements({}, {}, {}, {}, None)
        unplaced = "%s: %s" % (type(exc).__name__, exc)
    from pops.time._graph.value_traversal import walk_program_nodes

    program = str(getattr(plan.time, "name", None) or "program")
    records = []
    # Every depth of the Program: the bodies of its control blocks hold field leaves too (T2).
    nodes = tuple(walk_program_nodes(plan.time._values))
    instances: dict[int, Any] = {}
    try:
        origins: Any = _Origins(nodes)
    except Exception:  # noqa: BLE001 - A1: each row then says it found no issuer
        origins = None
    for value in nodes:
        try:
            value_type = ir_type(value, placements, program=program)
            layout, distribution, origin, residency = _layout(value, value_type, placements, plan,
                                                              origins)
        except Exception as exc:  # noqa: BLE001 - A1: reported, never refused
            value_type = A.Untyped("%s: %s" % (type(exc).__name__, exc))
            layout = distribution = origin = residency = "untyped"
        try:
            instance = _instance(value, layout, placements, instances)
        except Exception as exc:  # noqa: BLE001 - A1: reported, never refused
            instance = "untyped: %s: %s" % (type(exc).__name__, exc)
        records.append(LayoutRecord(
            value=str(value.name if getattr(value, "name", None) else value.id),
            node=int(value.id), spelling=value.vtype, role=_FIELD_ROLES.get(value.vtype), type=repr(value_type),
            layout=layout, distribution=distribution, origin=origin,
            residency=residency, instance=instance).to_data())
    try:
        accesses = _general_accesses(nodes, {row["node"]: row["layout"] for row in records})
    except Exception as exc:  # noqa: BLE001 - A1: reported, never refused
        accesses = [{"untyped": "%s: %s" % (type(exc).__name__, exc)}]
    try:
        relayouts = _relayouts(nodes, placements)
    except Exception as exc:  # noqa: BLE001 - A1: reported, never refused
        relayouts = [{"untyped": "%s: %s" % (type(exc).__name__, exc)}]
    return {
        "schema_version": 6,
        "placements": {block: configuration.identity
                       for block, configuration in sorted(placements.configurations.items())},
        "values": records,
        # DM5: no row is on Patches or LevelPatches; the builder refuses a field leaf there.
        "general_accesses": accesses,
        # DM6: the copy of regrid_read from level l_s of H onto the coarsening of the child of H'.
        "relayouts": relayouts,
        "untyped": sum(1 for row in records if row["layout"] == "untyped"),
        # DM3, CF16: the storage choices a case wrote, which the plan checks.
        "constraints": sum(1 for row in records if row["origin"] == "constraint"),
        "unplaced": unplaced,
    }


# ------------------------------------------------------------------------------------------------
# Lot 1b: the one function that tells a field leaf from a finite array (note §3.4).
#
# The serialized spelling of a value (``vtype``) is the canonical encoding of its IR type and its
# role: ``state``, ``rhs`` and ``scalar_field`` encode ``(Array(cells(κ_σ), T), role)``, ``array``
# encodes ``Array(range_set, T)``. The builder and the emitters ask the kind of a value here, and
# the kind is read from the head of the IR type that ``_signature_type`` gives the spelling, on a
# symbolic support: which support a field lives on, and its element type, never change its kind.

class _EverySupport(dict):
    """A placement table that places every block on one symbolic support."""

    def __init__(self, configuration: S.Configuration):
        super().__init__()
        self._configuration = configuration

    def __contains__(self, block: object) -> bool:
        return True

    def __getitem__(self, block: str) -> S.Configuration:
        return self._configuration


def _symbolic_placements() -> Placements:
    configuration = S.Configuration("pops.ir.kind@support", (1,), (False,))
    return Placements(_EverySupport(configuration), {}, _EverySupport(False),  # type: ignore[arg-type]
                      {}, configuration)


# The smallest signature of each spelling whose type head is its kind: a field leaf of one
# component, an array of scalars over one family, an index set and an index of that family.
_KIND_PROBES = {
    "state": {"vtype": "state", "block": "b"},
    "rhs": {"vtype": "rhs", "block": "b"},
    "scalar_field": {"vtype": "scalar_field", "block": "b"},
    "array": {"vtype": "array", "family": 0, "element": {"vtype": "scalar"}},
    "index_set": {"vtype": "index_set", "family": 0},
    "index": {"vtype": "index", "family": 0},
}
_KINDS: dict[str, str] = {}


def _kind_of_type(value_type: Any) -> str:
    if isinstance(value_type, A.Array):
        if isinstance(value_type.domain, (S.Entities, S.Union, S.Level)):
            return "field"
        return "finite"
    if isinstance(value_type, A.IndexSetType):
        return "index_set"
    if isinstance(value_type, A.Index):
        return "index"
    return "other"


def _kinds() -> dict[str, str]:
    if not _KINDS:
        placements = _symbolic_placements()
        for spelling, probe in _KIND_PROBES.items():
            _KINDS[spelling] = _kind_of_type(_signature_type(probe, placements, "kind", 1))
    return _KINDS


def _spelling(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("vtype")
    return getattr(value, "vtype", None)


def value_kind(value: Any) -> str:
    """``field``, ``finite``, ``index_set``, ``index`` or ``other``: the head of its IR type.

    ``value`` is a live Program value, its serialized signature, or its spelling. A field leaf is
    an ``Array`` over entities or levels (stored on ``Patches`` or ``LevelPatches``); any other
    ``Array`` is ``finite``; an index set and an index of the finite world are ``index_set`` and
    ``index``.
    """
    return _kinds().get(_spelling(value), "other")


def is_field_leaf(value: Any) -> bool:
    """Whether ``value`` is a field leaf: an ``Array`` over the cells or levels of a support."""
    return value_kind(value) == "field"


def is_finite_array(value: Any) -> bool:
    """Whether ``value`` is an ``Array`` over a range set (a finite array)."""
    return value_kind(value) == "finite"


def is_finite_index(value: Any) -> bool:
    """Whether ``value`` is a finite array, or an index set or index of the finite world."""
    return value_kind(value) in ("finite", "index_set", "index")


_ROLES = {"state": "state", "rhs": "rhs", "scalar_field": "field"}


def field_role(value: Any) -> str | None:
    """The role of a field leaf (``state``, ``rhs`` or ``field``); ``None`` for any other value.

    A state and its rate have one type; the role is what tells them apart, and a plain field (a
    scratch or a solve operand) from both.
    """
    if not is_field_leaf(value):
        return None
    return _ROLES[_spelling(value)]
