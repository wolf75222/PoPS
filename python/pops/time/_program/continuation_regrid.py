"""Closed authoring operations for an explicitly assessed AMR topology change.

The runtime supplies immutable source images. A target Field is read from a source Field
along the map the regrid issues (``regrid_read``); which source supplies a target remains a
Python DAG, and the runtime only copies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .structured_control import new_typed, value_signature
from .value_validation import require_owned
from .versioned_values import _cap
from pops._ir.data_model import field_role

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


REGRID_OPS = frozenset((
    "continuation_regular_phase", "continuation_regrid_source_count",
    "continuation_regrid_source", "continuation_regrid_base",
    "continuation_regrid_templates", "continuation_regrid_child_publish",
    "continuation_regrid_parent_open", "continuation_regrid_parent_publish",
    "continuation_regrid_read", "continuation_regrid_integral",
))


def _context(program):
    context = program.__dict__.get("_continuation_regrid_authoring")
    if context is None:
        raise ValueError("regrid views require an explicitly authored Continuation regrid region")
    return context


def _level(program, value):
    require_owned(program, value, "regrid source or target level", vtype="int64")
    return value


def source(program, level):
    """Read an immutable source level. This operation cannot select a target Base."""
    context = _context(program)
    schema = context["level_schema"]
    return new_typed(program, schema["read"], "continuation_regrid_source",
                     (_level(program, level),), {"schema": schema})


def base(program, source_level):
    """Select explicitly the scientific Base whose descendant a target becomes."""
    context = _context(program)
    if context["scope"] != "child":
        raise ValueError("regrid source Base selection belongs to a target child region")
    schema = context["level_schema"]
    return new_typed(program, _cap("continuation_base", schema=schema, view="read"),
                     "continuation_regrid_base", (_level(program, source_level),),
                     {"schema": schema})


def _source_path(field):
    """The path of ``field`` inside one ``regrid_source`` image, and that image."""
    path, cursor = [], field
    while cursor.op == "product_get":
        path.append(cursor.attrs["field"])
        cursor = cursor.inputs[0]
    if cursor.op != "continuation_regrid_source":
        raise ValueError("regrid_read reads a Field of a regrid_source image of this child")
    return tuple(reversed(path)), cursor


def read(program, field):
    """The value of a source Field at each target cell, along the map the regrid issues.

    ``target(c) = field(Parent(r)(c))`` with r the ratio from the source level to the child's
    level (the identity when they are equal): the phase 1 form of ``region.gather(field,
    Parent(ratio=r))`` over the target cells (N10 lot 3, DM13). The runtime allocates the target
    storage with the source Field's components and ghosts; nothing is interpolated.
    """
    context = _context(program)
    if context["scope"] != "child":
        raise ValueError("regrid_read belongs to a target child region")
    require_owned(program, field, "regrid_read Field")
    if field_role(field) not in ("state", "field"):
        raise TypeError("regrid_read requires a Field leaf of a regrid source image")
    path, image = _source_path(field)
    signature = value_signature(field)
    if signature.get("field_authority") is None and signature.get("block") not in context["blocks"]:
        raise ValueError("regrid Field has no declared scientific storage owner")
    try:
        from .continuation_forest import _field_at
        written = _field_at(context["level_schema"]["write"], path)
    except (KeyError, TypeError):
        written = None
    result = written if written is not None and field_role(written) == field_role(signature) \
        else signature
    return new_typed(program, result, "continuation_regrid_read", (field, image.inputs[0]),
                     {"path": path, "map": "parent", "dimension": context["dimension"]},
                     point=result.get("point", field.point))


def _read_origin(value):
    """The ``regrid_read`` a pointwise field value of the child is computed from, if any."""
    from pops.time._graph.value_traversal import reachable_values

    reads = [item for item in reachable_values(value, (value,)) if item.op == "continuation_regrid_read"]
    return reads


def integral(program, field):
    """The target level's share of the composite integral of one component (N10 lot 3, §2.2).

    Sum of field(c)·|c|·κ(c) over the valid, active cells of the child's level that the next
    level does not cover and that this rank owns, then one collective sum (DM9).
    """
    from pops.time.expressions import ProgramComponent

    context = _context(program)
    if context["scope"] != "child":
        raise ValueError("regrid_integral belongs to a target child region")
    component = 0
    if isinstance(field, ProgramComponent):
        field, component = field.value, field.component
        selected = True
    else:
        selected = False
    require_owned(program, field, "regrid_integral Field")
    if field_role(field) not in ("state", "field"):
        raise TypeError("regrid_integral integrates a Field leaf of the child")
    from pops.time.expressions import component_names
    if not selected and len(component_names(field)) != 1:
        raise ValueError("regrid_integral integrates one component; select it with field[k]")
    if not _read_origin(field):
        raise ValueError("regrid_integral integrates a Field computed from regrid_read in this child")
    return new_typed(program, {"vtype": "scalar"}, "continuation_regrid_integral", (field,),
                     {"component": component})


# DM2 (N10 lot 3, addendum §5): the hierarchy instance of a field leaf of a regrid child. A child
# reads its sources over H and writes its targets over H' = H.identity + "#regrid". The instance
# is a plan fact, not a kind: the spelling of a value does not carry it.
SOURCE_INSTANCE, TARGET_INSTANCE, MIXED_INSTANCES = "H", "H'", "mixed"
_TARGET_ROOTS = frozenset(("continuation_regrid_read", "continuation_regrid_templates"))


def _member(value):
    """The value a ``product_get`` chain selects, through the products the child built."""
    while value.op == "product_get":
        root = _member(value.inputs[0])
        if root.op != "product":
            return value
        names = tuple(root.attrs.get("fields", ()))
        field = value.attrs["field"]
        if field not in names:
            return value
        value = root.inputs[names.index(field)]
    return value


def hierarchy_instance(value, memo=None):
    """``H'``, ``H``, ``mixed`` or ``None`` (not a field leaf) for one value of a regrid child."""
    from pops._ir.data_model import is_field_leaf

    memo = {} if memo is None else memo
    if value.id in memo:
        return memo[value.id]
    memo[value.id] = None  # a cycle reads as no field
    result = None
    if is_field_leaf(value):
        selected = _member(value)
        if selected is not value:
            result = hierarchy_instance(selected, memo)
        elif value.op == "product_get":
            cursor = value
            while cursor.op == "product_get":
                cursor = cursor.inputs[0]
            result = TARGET_INSTANCE if cursor.op in _TARGET_ROOTS else SOURCE_INSTANCE
        elif value.op in _TARGET_ROOTS:
            result = TARGET_INSTANCE
        else:
            sides = {hierarchy_instance(item, memo) for item in value.inputs} - {None}
            if MIXED_INSTANCES in sides or sides >= {SOURCE_INSTANCE, TARGET_INSTANCE}:
                result = MIXED_INSTANCES
            elif sides == {TARGET_INSTANCE}:
                result = TARGET_INSTANCE
            else:
                result = SOURCE_INSTANCE
    memo[value.id] = result
    return result


def require_one_instance(program, approved):
    """Refuse the first operation of a child that mixes a field over H and a field over H'."""
    from pops._ir.refusals import HierarchyInstancesMixed
    from pops.time._graph.value_traversal import reachable_values

    memo = {}
    for value in sorted(reachable_values(approved, ()), key=lambda item: item.id):
        if hierarchy_instance(value, memo) != MIXED_INSTANCES:
            continue
        sides = {hierarchy_instance(item, memo) for item in value.inputs}
        if MIXED_INSTANCES not in sides:
            raise HierarchyInstancesMixed(
                "%s mixes a field over the hierarchy the regrid replaces (H) with a field over "
                "the new one (H'); a source field reaches the target only through regrid_read"
                % value.op, region=program._current_region(), op=value.op)


def author_phase(program, metadata, child_author, parent_author, observe, dimension):
    from .continuation_forest import _field_at
    from .versioned_values import _fields, _require, _same

    child_schema, root_schema = metadata["level_schema"], metadata["root_schema"]
    contract = {"dimension": dimension}
    metadata["regrid"] = contract
    marker = {"level_schema": child_schema, "dimension": dimension,
              "blocks": frozenset(state.block_ref for state, _ in child_schema["fields"])}
    selected = new_typed(program, {"vtype": "bool"}, "continuation_forest_child_phase", (), {})

    def proposal_parts(p):
        old_count = new_typed(p, {"vtype": "int64"}, "continuation_regrid_source_count", (), {})
        new_count = new_typed(p, {"vtype": "int64"}, "continuation_forest_level_count", (), {})
        level = new_typed(p, {"vtype": "int64"}, "continuation_forest_level", (), {})
        return old_count, new_count, level

    def child(p):
        fields = tuple(("field_%d" % i, _field_at(child_schema["write"], path))
                       for i, (_, path) in enumerate(child_schema["fields"]))
        templates = new_typed(p, {"vtype": "product", "members": fields},
                              "continuation_regrid_templates", (), {"schema": child_schema})
        old_count, new_count, level = proposal_parts(p)
        proposal = p.product(templates=templates, source_count=old_count,
                             target_count=new_count, level=level)
        prior = p.__dict__.get("_continuation_regrid_authoring")
        p.__dict__["_continuation_regrid_authoring"] = dict(marker, scope="child")
        try:
            approved = child_author(p, proposal)
        finally:
            if prior is None:
                p.__dict__.pop("_continuation_regrid_authoring", None)
            else:
                p.__dict__["_continuation_regrid_authoring"] = prior
        require_one_instance(p, approved)
        signature = _require(p, approved, "continuation_accepted", "complete child regrid")
        _same(child_schema, signature["candidate"]["base"]["schema"], "regrid child schema")
        report = dict(signature["candidate"]["envelope"]["members"])["observations"]
        if tuple(_fields(report)):
            raise TypeError("regrid reports must contain explicit nonfield observations")
        contract["report"] = report
        new_typed(p, signature, "continuation_regrid_child_publish", (approved,),
                  {"schema": child_schema})
        return p.boolean(True)

    def parent(p):
        root = new_typed(p, _cap("continuation_base", schema=root_schema, view="read"),
                         "continuation_regrid_parent_open", (), {"schema": root_schema})
        old_count, new_count, _ = proposal_parts(p)
        domain = p.range_set(new_count, capacity=metadata["level_capacity"], ownership="replicated")
        reports = new_typed(p, dict(value_signature(domain), vtype="array", element=contract["report"]),
                            "continuation_child_reports", (domain,), {})
        prior = p.__dict__.get("_continuation_regrid_authoring")
        p.__dict__["_continuation_regrid_authoring"] = dict(marker, scope="parent")
        try:
            observations = p.product() if observe is None else observe(p)
            require_owned(p, observations, "regrid parent observations", vtype="product")
            proposal = p.product(children=reports, observations=observations,
                                 source_count=old_count, target_count=new_count)
            approved = parent_author(p, root, proposal)
        finally:
            if prior is None:
                p.__dict__.pop("_continuation_regrid_authoring", None)
            else:
                p.__dict__["_continuation_regrid_authoring"] = prior
        signature = _require(p, approved, "continuation_accepted", "complete parent regrid")
        _same(root_schema, signature["candidate"]["base"]["schema"], "regrid parent schema")
        new_typed(p, signature, "continuation_regrid_parent_publish", (approved,),
                  {"schema": root_schema})
        return p.boolean(True)

    return program.branch(selected, child, parent)


class _ProgramRegrid(_ProgramBase):
    """The builder operations of a regrid region (N10 lot 3: they replace the bridge)."""

    def regrid_source(self, level):
        """Read one immutable source image inside the explicit regrid transaction."""
        return source(self, level)

    def regrid_read(self, field):
        """Read a source Field at the child's target cells along the map the regrid issues."""
        return read(self, field)

    def regrid_integral(self, field):
        """The child's level share of the composite integral of one Field component."""
        return integral(self, field)

    def regrid_base(self, source_level):
        """Choose explicitly the source whose assessed descendant this target becomes."""
        return base(self, source_level)
