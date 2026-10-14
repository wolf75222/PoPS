"""Pure structural value signatures shared by finite types and Program authoring.

This layer only describes existing SSA values. It does not construct Program nodes,
validate control flow, or import the private authoring mixins.
"""
from __future__ import annotations


_LEAVES = frozenset(("state", "rhs", "scalar_field", "scalar", "bool", "int64"))


def value_signature(value):
    if value.vtype in ("product", "variant", "index_set", "index", "array", "continuation_base", "continuation_candidate", "continuation_assessment", "continuation_accepted"):
        return value.attrs["signature"]
    if value.vtype == "result":
        return {"vtype": "result", "success": value.attrs["success_signature"],
                "point": value.point}
    if value.vtype not in _LEAVES:
        raise TypeError("structured control cannot carry value type %r" % value.vtype)
    return dict(vtype=value.vtype, block=value.block, space=value.space,
                state_ref=value.state_ref, field_context=value.field_context, point=value.point)


def same_signature(left, right):
    from pops.time.canonical_data import _json_ready
    return _json_ready(left) == _json_ready(right)
