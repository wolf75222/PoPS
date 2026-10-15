"""Closed authoring operations for an explicitly assessed AMR topology change.

The runtime supplies immutable source images and target storage representations.
Which source supplies a target, and every numerical transfer, remain Python DAGs.
"""
from __future__ import annotations

from .structured_control import new_typed, value_signature
from .value_validation import require_owned
from .versioned_values import _cap


REGRID_OPS = frozenset((
    "continuation_regular_phase", "continuation_regrid_source_count",
    "continuation_regrid_source", "continuation_regrid_base",
    "continuation_regrid_templates", "continuation_regrid_child_publish",
    "continuation_regrid_parent_open", "continuation_regrid_parent_publish",
    "continuation_source_topology", "continuation_target_topology",
))


def _context(program):
    context = program.__dict__.get("_continuation_regrid_authoring")
    if context is None:
        raise ValueError("regrid views require an explicitly authored Continuation regrid region")
    return context


def _level(program, value):
    require_owned(program, value, "regrid source or target level", vtype="int64")
    return value


def topology(program, field, *, level, side, dimension):
    """Borrow one exact old/new layout; no geometry or interpolation is inferred."""
    from pops.time.field_topology import validate_topology_signature

    context = _context(program)
    require_owned(program, field, "regrid topology Field")
    signature = value_signature(field)
    if signature["vtype"] not in ("state", "scalar_field"):
        raise TypeError("regrid topology requires an immutable native Field template")
    if signature.get("block") not in context["blocks"]:
        raise ValueError("regrid Field has no declared State storage owner")
    if side not in ("source", "target"):
        raise ValueError("regrid topology side must be source or target")
    if type(dimension) is not int or not 1 <= dimension <= 3:
        raise ValueError("regrid topology dimension must match a native dimension in 1..3")
    if dimension != context["dimension"]:
        raise ValueError("regrid topology differs from the declared native dimension")
    result = {"vtype": "field_topology", "field_signature": signature,
              "side": side, "dimension": dimension}
    validate_topology_signature(result)
    return new_typed(program, result, "continuation_%s_topology" % side,
                     (field, _level(program, level)), {"dimension": dimension, "side": side})


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
