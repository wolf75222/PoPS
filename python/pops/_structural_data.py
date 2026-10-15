"""Stack-safe transformations of finite metadata, preserving shared substructures.

The work stack grows with the data. Cycles are rejected; no process recursion
limit is changed and no scientific operation is evaluated here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def transform(value, describe) -> Any:
    """Apply ``describe(node) -> (children, build)`` postorder without recursion.

    Return ``(None, leaf)`` for an opaque leaf. ``build`` receives transformed
    children in declaration order. Object identities remain live in each frame.
    """
    done, active, live = {}, set(), []
    stack: list[tuple[Any, Any, Any]] = [(value, None, None)]
    while stack:
        node, children, build = stack.pop()
        identity = id(node)
        if children is not None:
            done[identity] = build([done[id(child)] for child in children])
            active.remove(identity)
            continue
        if identity in active:
            raise ValueError("structural data cannot contain a cycle")
        if identity in done:
            continue
        live.append(node)  # protocol-produced nodes must keep their object identities alive
        children, build = describe(node)
        if children is None:
            done[identity] = build
            continue
        children = tuple(children)
        active.add(identity)
        stack.append((node, children, build))
        stack.extend((child, None, None) for child in reversed(children))
    return done[id(value)]


def equal(left, right):
    """Container equality without Python's recursive container comparison."""
    stack, checked = [(left, right)], set()
    while stack:
        a, b = stack.pop()
        if a is b:
            continue
        pair = id(a), id(b)
        if pair in checked:
            continue
        checked.add(pair)
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if a.keys() != b.keys():
                return False
            stack.extend((a[key], b[key]) for key in a)
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            stack.extend(zip(a, b, strict=True))
        elif a != b:
            return False
    return True


def transform_tree(value, describe) -> Any:
    """Occurrence-preserving traversal for protocols whose calls are observable."""
    result, active = [None], set()
    stack: list[tuple[Any, Any, int, Any]] = [(value, result, 0, None)]
    while stack:
        node, destination, index, finish = stack.pop()
        if finish is not None:
            build, children = finish
            destination[index] = build(children)
            active.remove(id(node))
            continue
        if id(node) in active:
            raise ValueError("structural data cannot contain a cycle")
        children, build = describe(node)
        if children is None:
            destination[index] = build
            continue
        children = tuple(children)
        outputs = [None] * len(children)
        active.add(id(node))
        stack.append((node, destination, index, (build, outputs)))
        stack.extend((child, outputs, i, None)
                     for i, child in reversed(tuple(enumerate(children))))
    return result[0]


def transform_tree_context(value, describe, context) -> Any:
    """Occurrence-preserving traversal with a caller-defined diagnostic context.

    Children are ``(value, context)`` pairs. Cycle checks concern the actual
    source objects, not the temporary context pairs or protocol-produced data.
    """
    result, active = [None], set()
    stack: list[tuple[Any, Any, Any, int, Any]] = [(value, context, result, 0, None)]
    while stack:
        node, context, destination, index, finish = stack.pop()
        if finish is not None:
            build, children = finish
            destination[index] = build(children)
            active.remove(id(node))
            continue
        if id(node) in active:
            raise ValueError("structural data cannot contain a cycle at %s" % context)
        children, build = describe(node, context)
        if children is None:
            destination[index] = build
            continue
        children = tuple(children)
        outputs = [None] * len(children)
        active.add(id(node))
        stack.append((node, context, destination, index, (build, outputs)))
        stack.extend((child, child_context, outputs, i, None)
                     for i, (child, child_context) in reversed(tuple(enumerate(children))))
    return result[0]


def canonical_json(value, *, allow_nan=True, sort_keys=True):
    """The standard compact JSON bytes, without recursive containers."""
    # Use the C encoder only for inert builtin data. The iterative fallback
    # remains authoritative for arbitrary depth and observable Mapping hooks.
    # No process recursion limit is changed and no type/key is coerced.
    if _plain_json_data(value):
        try:
            return json.dumps(value, allow_nan=allow_nan, sort_keys=sort_keys,
                              separators=(",", ":"))
        except RecursionError:
            pass
    return "".join(canonical_json_chunks(value, allow_nan=allow_nan, sort_keys=sort_keys))


def _plain_json_data(value, *, nonempty_keys=False, finite_floats=False):
    """Check the closed builtin codec language without executing user hooks."""
    seen, stack = set(), [value]
    while stack:
        node = stack.pop()
        kind = type(node)
        if kind is float and finite_floats:
            import math
            if not math.isfinite(node):
                return False
        if node is None or kind is str or kind is int or kind is float or kind is bool:
            continue
        if kind not in (dict, list, tuple):
            return False
        identity = id(node)
        if identity in seen:
            continue
        seen.add(identity)
        if kind is dict:
            if any(type(key) is not str or (nonempty_keys and not key) for key in node):
                return False
            stack.extend(node.values())
        else:
            stack.extend(node)
    return True


def canonical_json_chunks(value, *, allow_nan=True, sort_keys=True):
    """Stream exact JSON tokens; scalar spellings use stdlib's C string codec."""
    import math
    encode_string = json.encoder.encode_basestring_ascii
    active, stack = set(), [(0, value)]
    while stack:
        operation, node = stack.pop()
        if operation == 1:
            yield node
            continue
        if operation == 2:
            active.remove(node)
            continue
        if isinstance(node, (Mapping, list, tuple)):
            identity = id(node)
            if identity in active:
                raise ValueError("Circular reference detected")
            active.add(identity)
            stack.append((2, identity))
            if isinstance(node, Mapping):
                if any(not isinstance(key, str) for key in node):
                    raise TypeError("canonical JSON requires string mapping keys")
                entries = sorted(node.items()) if sort_keys else list(node.items())
                yield "{"
                stack.append((1, "}"))
                for i in range(len(entries) - 1, -1, -1):
                    key, item = entries[i]
                    stack.append((0, item))
                    stack.append((1, encode_string(key) + ":"))
                    if i:
                        stack.append((1, ","))
            else:
                yield "["
                stack.append((1, "]"))
                for i in range(len(node) - 1, -1, -1):
                    stack.append((0, node[i]))
                    if i:
                        stack.append((1, ","))
        elif isinstance(node, str):
            yield encode_string(node)
        elif node is None:
            yield "null"
        elif node is True:
            yield "true"
        elif node is False:
            yield "false"
        elif isinstance(node, int):
            yield int.__repr__(node)
        elif isinstance(node, float):
            if math.isfinite(node):
                yield float.__repr__(node)
            elif not allow_nan:
                raise ValueError("Out of range float values are not JSON compliant")
            else:
                yield "NaN" if math.isnan(node) else "Infinity" if node > 0 else "-Infinity"
        else:
            # Preserve the standard error for unsupported scalar objects.
            yield json.dumps(node, allow_nan=allow_nan, sort_keys=sort_keys, separators=(",", ":"))


def canonical_json_sha256(value, *, allow_nan=True, sort_keys=True):
    """Hash identical JSON bytes with bounded output buffering (not a value limit)."""
    import hashlib
    digest, pieces, size = hashlib.sha256(), [], 0
    for token in canonical_json_chunks(value, allow_nan=allow_nan, sort_keys=sort_keys):
        pieces.append(token)
        size += len(token)
        if size >= 65536:
            digest.update("".join(pieces).encode())
            pieces.clear()
            size = 0
    if pieces:
        digest.update("".join(pieces).encode())
    return digest.hexdigest()


def json_containers(text) -> Any:
    """Decode JSON containers iteratively; stdlib still parses every scalar.

    This preserves JSON's normal key/number/string behavior without using the
    recursive stdlib container scanner. It is a metadata codec, not an evaluator.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise TypeError("JSON metadata must be text or UTF-8 bytes")
    if type(text) is str:
        try:
            return json.loads(text)
        except RecursionError:
            # The C decoder is an optimization, never a structural depth limit.
            pass
    decoder = json.JSONDecoder()
    stack, result, complete, position = [], None, False, 0

    def error(message):
        raise json.JSONDecodeError(message, text, position)

    while True:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        if not stack and complete:
            if position != len(text):
                error("Extra data")
            return result
        if position == len(text):
            error("Expecting value")
        if stack:
            frame = stack[-1]
            container, phase, key = frame
            closing = "}" if isinstance(container, dict) else "]"
            if phase == "comma":
                if text[position] == closing:
                    stack.pop()
                    position += 1
                    continue
                if text[position] != ",":
                    error("Expecting ',' delimiter")
                frame[1] = "required"
                position += 1
                continue
            if phase == "first" and text[position] == closing:
                stack.pop()
                position += 1
                continue
            if isinstance(container, dict):
                if text[position] != '"':
                    error("Expecting property name enclosed in double quotes")
                key, position = decoder.raw_decode(text, position)
                while position < len(text) and text[position] in " \t\r\n":
                    position += 1
                if position == len(text) or text[position] != ":":
                    error("Expecting ':' delimiter")
                position += 1
                while position < len(text) and text[position] in " \t\r\n":
                    position += 1
                if position == len(text):
                    error("Expecting value")
                frame[2] = key
        token = text[position]
        nested = token in "[{"
        if nested:
            value = {} if token == "{" else []
            position += 1
        else:
            value, position = decoder.raw_decode(text, position)
        if stack:
            frame = stack[-1]
            if isinstance(frame[0], dict):
                frame[0][frame[2]] = value
            else:
                frame[0].append(value)
            frame[1] = "comma"
        else:
            result, complete = value, True
        if nested:
            stack.append([value, "first", None])
