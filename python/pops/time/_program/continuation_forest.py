"""Author the complete AMR acceptance forest as ordinary closed Program regions.

The three callables execute once while building the graph. They contain Python
algebra (including Advance/Assess); none is retained as a runtime callback.
"""
from __future__ import annotations

from .structured_control import new_typed, value_signature
from .value_validation import require_owned, require_top_level
from .versioned_values import ContinuationSchema, _cap, _fields, _require, _same


FOREST_OPS = frozenset((
    "continuation_forest_child_phase", "continuation_forest_level",
    "continuation_forest_level_count", "continuation_forest_interval",
    "continuation_sync_open", "continuation_working_value",
    "continuation_synchronized_fields", "continuation_sync_publish",
    "continuation_parent_open", "continuation_child_reports",
    "continuation_parent_publish",
))


def forest_node(program):
    nodes = [v for v in program._values
             if v.op == "post_synchronization" and v.attrs.get("continuation_forest")]
    if len(nodes) > 1:
        raise ValueError("one complete AMR continuation forest is allowed")
    return nodes[0] if nodes else None


def author_forest(program, accepted, *, synchronize, initialize, parent,
                  observe=None, level_capacity, regrid=None, regrid_parent=None,
                  regrid_observe=None, dimension=None):
    """Publish local trials, then synchronize and assess one complete macro forest.

    ``initialize(p)`` returns ``(initial, parent_schema)`` for the unique global
    continuation. ``synchronize(p, base, proposal)`` receives the immutable
    macro-entry child Base and ``candidate, synchronized, interval, level``.
    ``parent(p, base, proposal)`` receives ``children, observations, interval``.
    Both return exact Accepted capabilities, usually composed with ``advance``.
    Every child report is its accepted envelope's observations Product. The
    parent is natively dependent on all those exact accepted children.
    """
    require_top_level(program, accepted, "AMR continuation forest")
    signature = _require(program, accepted, "continuation_accepted", "AMR child trial")
    if not all(callable(body) for body in (synchronize, initialize, parent)):
        raise TypeError("AMR forest authors must be callable Python graph builders")
    if observe is not None and not callable(observe):
        raise TypeError("AMR forest observe must author a Product")
    if type(level_capacity) is not int or level_capacity < 1 or level_capacity >= 2**63:
        raise ValueError("AMR forest level_capacity requires a positive Int64 bound")
    regrid_enabled = any(body is not None for body in (regrid, regrid_parent, regrid_observe))
    if regrid_enabled:
        if not callable(regrid) or not callable(regrid_parent):
            raise TypeError("a complete regrid requires explicit child and parent graph builders")
        if regrid_observe is not None and not callable(regrid_observe):
            raise TypeError("regrid observations must be an explicit graph builder")
        if type(dimension) is not int or not 1 <= dimension <= 3:
            raise ValueError("regrid requires its exact native dimension in 1..3")
    elif dimension is not None:
        raise ValueError("regrid dimension was supplied without a regrid graph")
    child_schema = signature["candidate"]["base"]["schema"]
    if not child_schema["fields"]:
        raise ValueError("AMR level continuation must bind its native State fields")
    publication = program.continuation_publish(accepted)
    metadata = {"version": 1, "level_capacity": level_capacity,
                "level_schema": child_schema}

    def ordinary_phase(p):
        condition = new_typed(p, {"vtype": "bool"}, "continuation_forest_child_phase", (), {})

        def child(q):
            base = new_typed(q, _cap("continuation_base", schema=child_schema, view="read"),
                             "continuation_sync_open", (), {"schema": child_schema})
            previous = new_typed(q, child_schema["write"], "continuation_working_value", (),
                                 {"schema": child_schema})
            field_members = tuple(("field_%d" % i, _field_at(child_schema["write"], path))
                                  for i, (_, path) in enumerate(child_schema["fields"]))
            fields = new_typed(q, {"vtype": "product", "members": field_members},
                               "continuation_synchronized_fields", (), {"schema": child_schema})
            level = new_typed(q, {"vtype": "int64"}, "continuation_forest_level", (), {})
            proposal = q.product(candidate=previous, synchronized=fields,
                                 interval=_interval(q), level=level)
            approved = synchronize(q, base, proposal)
            approved_type = _require(q, approved, "continuation_accepted", "AMR synchronization")
            _same(child_schema, approved_type["candidate"]["base"]["schema"], "AMR synchronized schema")
            report = dict(approved_type["candidate"]["envelope"]["members"])["observations"]
            if tuple(_fields(report)):
                raise TypeError("child reports are nonfield observations; composite fields use explicit integrals")
            metadata["report"] = report
            new_typed(q, approved_type, "continuation_sync_publish", (approved,), {"schema": child_schema})
            return q.boolean(True)

        def root(q):
            initial, schema = initialize(q)
            if type(schema) is not ContinuationSchema or schema.fields:
                raise TypeError("AMR parent requires an explicit field-free global ContinuationSchema")
            require_owned(q, initial, "AMR parent initialization", vtype="product")
            _same(schema.read_type.signature, value_signature(initial), "AMR parent initialization")
            root_schema = schema.to_data()
            metadata["root_schema"] = root_schema
            base = new_typed(q, _cap("continuation_base", schema=root_schema, view="read"),
                             "continuation_parent_open", (initial,), {"schema": root_schema})
            count = new_typed(q, {"vtype": "int64"}, "continuation_forest_level_count", (), {})
            domain = q.range_set(count, capacity=level_capacity, ownership="replicated")
            report_type = dict(value_signature(domain), vtype="array", element=metadata["report"])
            reports = new_typed(q, report_type, "continuation_child_reports", (domain,), {})
            observations = q.product() if observe is None else observe(q)
            require_owned(q, observations, "AMR parent observations", vtype="product")
            proposal = q.product(children=reports, observations=observations, interval=_interval(q))
            approved = parent(q, base, proposal)
            approved_type = _require(q, approved, "continuation_accepted", "AMR macro parent")
            _same(root_schema, approved_type["candidate"]["base"]["schema"], "AMR parent schema")
            new_typed(q, approved_type, "continuation_parent_publish", (approved,), {"schema": root_schema})
            return q.boolean(True)

        return p.branch(condition, child, root)

    def regrid_phase(p):
        from .continuation_regrid import author_phase
        return author_phase(p, metadata, regrid, regrid_parent, regrid_observe, dimension)

    def phase(p):
        if not regrid_enabled:
            ordinary_phase(p)
        else:
            regular = new_typed(p, {"vtype": "bool"}, "continuation_regular_phase", (), {})
            p.branch(regular, ordinary_phase, regrid_phase)

    program.after_synchronization(phase, name="continuation_forest", _continuation_forest=metadata)
    return publication


def _field_at(signature, path):
    for component in path:
        signature = dict(signature["members"])[component]
    return signature


def _interval(program):
    return new_typed(program, {"vtype": "product", "members": tuple(
        (name, {"vtype": "scalar"}) for name in ("start", "end", "duration"))},
        "continuation_forest_interval", (), {})
