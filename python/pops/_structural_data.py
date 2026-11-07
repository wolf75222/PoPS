"""Stack-safe transformations of finite metadata, preserving shared substructures.

The work stack grows with the data. Cycles are rejected; no process recursion
limit is changed and no scientific operation is evaluated here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class StructuralPath:
    """Diagnostic spine; successful traversals do not copy ancestor strings."""

    __slots__ = ("parent", "suffix")

    def __init__(self, parent, suffix=""):
        self.parent, self.suffix = parent, suffix

    def child(self, suffix):
        return StructuralPath(self, suffix)

    def __str__(self):
        parts, node = [], self
        while isinstance(node, StructuralPath):
            parts.append(node.suffix)
            node = node.parent
        return str(node) + "".join(reversed(parts))


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
        if kind is not dict and kind is not list and kind is not tuple:
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


# One streamed C-codec chunk: bounded in height (stdlib recursion) and in characters.
_JSON_CHUNK_HEIGHT = 64
_JSON_CHUNK_CHARACTERS = 65536


def _plain_json_scalar_length(node):
    """Compact JSON length of one inert builtin scalar."""
    import math
    kind = type(node)
    if kind is str:
        return len(json.encoder.encode_basestring_ascii(node))
    if node is None or node is True:
        return 4
    if node is False:
        return 5
    if kind is int:
        return len(int.__repr__(node))
    if math.isfinite(node):
        return len(float.__repr__(node))
    return 3 if math.isnan(node) else 8 if node > 0 else 9


def _plain_json_depths(value):
    """Measure inert builtin containers without running any extension protocol.

    Each container maps to its height and compact JSON length. A length counts every
    occurrence of a shared substructure; it saturates just above one chunk, since occurrences
    may expand far beyond the memory that holds them.
    """
    encode_string = json.encoder.encode_basestring_ascii
    extents, active, pending = {}, set(), [(False, value)]
    while pending:
        leaving, node = pending.pop()
        kind = type(node)
        if leaving:
            height, length = 0, 1 + max(len(node), 1)
            for key, child in (node.items() if kind is dict else ((None, child) for child in node)):
                extent = extents.get(id(child))
                child_height, child_length = (0, _plain_json_scalar_length(child)) if extent is None else extent
                if key is not None:
                    child_length += len(encode_string(key)) + 1
                height = max(height, child_height)
                length = min(length + child_length, _JSON_CHUNK_CHARACTERS + 1)
            extents[id(node)] = (height + 1, length)
            active.remove(id(node))
            continue
        if node is None or kind is str or kind is int or kind is bool or kind is float:
            continue
        if kind is not dict and kind is not list and kind is not tuple:
            return None
        marker = id(node)
        if marker in active:
            return None
        if marker in extents:
            continue
        if kind is dict and any(type(key) is not str for key in node):
            return None
        active.add(marker)
        pending.append((True, node))
        pending.extend((False, child) for child in (node.values() if kind is dict else node))
    return extents


def canonical_json_chunks(value, *, allow_nan=True, sort_keys=True):
    """Stream exact JSON in C-codec chunks of bounded height and length for inert builtin trees.

    A chunk encodes a whole subtree, or a run of consecutive members of one container, whose
    compact JSON fits one chunk; only a scalar longer than a chunk stands alone. Arbitrary
    finite depth, shared occurrences and extension protocols keep the iterative encoder below.
    """
    extents = _plain_json_depths(value)
    if extents is None:
        yield from _canonical_json_protocol_chunks(value, allow_nan=allow_nan, sort_keys=sort_keys)
        return
    encode_string = json.encoder.encode_basestring_ascii

    def extent(node):
        found = extents.get(id(node))
        return (0, _plain_json_scalar_length(node)) if found is None else found

    # Operations: 0 encodes or opens a node, 1 yields a literal, 2 and 3 encode a run of
    # consecutive list members or (key, value) items as the inside of their container.
    pending = [(0, value)]
    while pending:
        operation, node = pending.pop()
        if operation == 1:
            yield node
            continue
        if operation:
            run = node if operation == 2 else dict(node)
            try:
                encoded = json.dumps(run, allow_nan=allow_nan, sort_keys=sort_keys,
                                     separators=(",", ":"))[1:-1]
            except (RecursionError, ValueError):
                for i, member in enumerate(node):
                    if i:
                        yield ","
                    if operation == 3:
                        yield encode_string(member[0]) + ":"
                        member = member[1]
                    yield from _canonical_json_protocol_chunks(
                        member, allow_nan=allow_nan, sort_keys=sort_keys)
            else:
                yield encoded
            continue
        height, length = extent(node)
        if height == 0 or (height <= _JSON_CHUNK_HEIGHT and length <= _JSON_CHUNK_CHARACTERS):
            try:
                encoded = json.dumps(node, allow_nan=allow_nan, sort_keys=sort_keys,
                                     separators=(",", ":"))
            except (RecursionError, ValueError):
                yield from _canonical_json_protocol_chunks(
                    node, allow_nan=allow_nan, sort_keys=sort_keys)
            else:
                yield encoded
            continue
        mapping = type(node) is dict
        members = (sorted(node.items()) if sort_keys else list(node.items())) if mapping else node
        run_operation = 3 if mapping else 2
        # Each item is the operations of one comma-separated piece: a run, or one member that
        # is too high or too long to share a chunk and is encoded or opened by itself.
        items, run, run_length = [], [], 0
        for member in members:
            child_height, child_length = extent(member[1] if mapping else member)
            if mapping:
                child_length += len(encode_string(member[0])) + 1
            if child_height >= _JSON_CHUNK_HEIGHT or child_length > _JSON_CHUNK_CHARACTERS:
                if run:
                    items.append([(run_operation, tuple(run))])
                    run, run_length = [], 0
                items.append([(1, encode_string(member[0]) + ":"), (0, member[1])] if mapping
                             else [(0, member)])
                continue
            if run and run_length + 1 + child_length > _JSON_CHUNK_CHARACTERS:
                items.append([(run_operation, tuple(run))])
                run, run_length = [], 0
            run_length += child_length + (1 if run else 0)
            run.append(member)
        if run:
            items.append([(run_operation, tuple(run))])
        yield "{" if mapping else "["
        pending.append((1, "}" if mapping else "]"))
        for i in range(len(items) - 1, -1, -1):
            pending.extend(reversed(items[i]))
            if i:
                pending.append((1, ","))


def _canonical_json_protocol_chunks(value, *, allow_nan=True, sort_keys=True):
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
