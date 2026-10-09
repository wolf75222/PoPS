"""Generic SSA value traversal; no compiler, numerical or execution authority."""
from collections.abc import Mapping
from typing import Any


def walk_program_nodes(values: Any) -> Any:
    for node in values:
        yield node
        attrs = getattr(node, "attrs", {})
        if not isinstance(attrs, Mapping):
            continue
        for key in (
            "cond_block", "body_block", "true_block", "false_block",
            "apply_block", "residual_block",
        ):
            nested = attrs.get(key)
            if isinstance(nested, (list, tuple)):
                yield from walk_program_nodes(nested)



def reachable_values(node: Any, all_nodes: tuple[Any, ...]) -> tuple[Any, ...]:
    """Follow actual data inputs and nested apply blocks, never unrelated metadata witnesses."""
    by_id = {item.id: item for item in all_nodes}
    result = {}

    def visit(value: Any) -> None:
        if not hasattr(value, "id") or value.id in result:
            return
        value = by_id.get(value.id, value)
        result[value.id] = value
        for source in value.inputs:
            visit(source)
        for key in ("apply_block", "residual_block", "body_block"):
            for child in value.attrs.get(key, ()):
                visit(child)

    visit(node)
    return tuple(result.values())
