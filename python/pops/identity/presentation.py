"""Presentation names out of an identity preimage (CC2, C47; N11 lot 2b).

Spec 0.3 CC2: the artifact identity hashes the normalized IR, the choices, the target and the
toolchain; presentation names do not enter it. The owner names (the case, its blocks, the model
definitions and the layouts) are presentation names: two cases that differ only by them have one
identity. :class:`PresentationMap` maps each of them to its position (``$case``, ``$block<k>`` in
block registration order, ``$model<k>`` in first use over the blocks, ``$layout<k>`` in layout
order), and :meth:`PresentationMap.project` rewrites an already canonical JSON preimage:

- an ``OwnerPath`` record (``schema_version``, ``nodes``, ``definition_fingerprint``) has the nodes
  of the four kinds renamed, and its definition fingerprint replaced by the name-free digest of the
  same model definition;
- a handle record (``qualified_id``, ``kind``, ``local_id``) of kind ``block`` or ``layout`` has its
  ``local_id`` renamed;
- a qualified id, owner path, or reference in a serialized schema is rebuilt from its renamed
  parts. Ordinary strings, including labels that resemble references, keep their bytes;
- a bare name is renamed only at a path the caller declares (``keys`` for a mapping keyed by block
  names, ``values`` for one exact field).

Every other name (quantities, components, parameters, fields, terms, the Program and its nodes,
frames, realizations, Python bodies) is kept: those are the kinds that gap C52 lists. A name of a
projected kind that the map does not know is refused, except in three kept categories (K1-K3 below,
the residuals of N11 lot 2b-i): such a name keeps itself, still enters the identities as before
this lot, and is never replaced by the label of another owner.
"""
from __future__ import annotations

import json
import re
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from heapq import merge
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, unquote


PRESENTATION_SCHEMA_VERSION = 1

_OWNER_KINDS = frozenset(
    ("case", "model_definition", "block", "layout", "descriptor", "consumer", "shared"))
_OWNER_RECORD_KEYS = frozenset(("schema_version", "nodes", "definition_fingerprint"))
_COMPONENT_KEY_FIELDS = frozenset(("owner_qid", "space_kind", "space_name", "component"))
_BOUNDARY_PROVIDER_FIELDS = frozenset(("schema_version", "plan_type", "topology", "needs", "bindings"))
_GHOST_PLAN_SETS = ("regions", "productions", "corner_policies", "interfaces",
                    "residual_contributions", "linearization_contributions", "component_bindings")
_GHOST_PLAN_FIELDS = frozenset((
    "schema_version", "plan_type", "topology", "coverage", "execution_authority", *_GHOST_PLAN_SETS,
))
_HANDLE_PREFIX = "pops.handle.v"
_AUTHORING = "#authoring="
# pops.model.registry._operator_registry_fingerprint: the content of a standalone registry, no name.
_REGISTRY_FINGERPRINT = "pops.operator-registry:sha256:"
# A projected definition fingerprint keeps the shape the owner records accept.
_PROJECTED_FINGERPRINT = "pops.module.projected:sha256:%s"
_DIGEST = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_LABEL_SEGMENT = re.compile(r"^[a-z_]+:[^:/]+$")
_LIST_ITEM = object()


# A label is ``$`` and its kind; an owner name cannot start with ``$`` (``ReservedOwnerNameError``),
# so a name kept in a preimage can never equal a label (review A2 of #797).
LABEL_PREFIX = "$"
_PRESENTATION_KINDS = frozenset(("case", "block", "model_definition", "layout"))


class ReservedOwnerNameError(ValueError):
    """An owner name of the reserved label form (``$...``), refused where the owner is named."""


def require_owner_name(kind: str, name: Any) -> None:
    """Refuse a case, block, model definition or layout name that starts with ``$``."""
    if kind in _PRESENTATION_KINDS and isinstance(name, str) and name.startswith(LABEL_PREFIX):
        raise ReservedOwnerNameError(
            "%s name %r is reserved: a name starting with %r is a presentation label "
            "(spec 0.3 CC2, N11 lot 2b)" % (kind.replace("_", " "), name, LABEL_PREFIX))


class PresentationProjectionError(ValueError):
    """A preimage holds a name of a projected kind that the map does not know."""


@dataclass(frozen=True, slots=True)
class PresentationMap:
    """Positions of the owner names of one case, in registration order.

    ``models`` holds ``(name, definition_fingerprint, name_free_fingerprint)`` per model definition
    in first use over the blocks; ``block_models`` gives, per block, the index of its model.
    ``digests`` pairs the sha256 hex digest of a named record (an initial condition, the layout
    plan) with the digest of the same record projected; such a digest is replaced wherever it
    appears, since a 64-digit digest cannot be confused with anything else.
    """

    case: str | None
    blocks: tuple[str, ...]
    models: tuple[tuple[str, str | None, str], ...]
    block_models: tuple[int, ...]
    layouts: tuple[str, ...] = ()
    digests: tuple[tuple[str, str], ...] = ()
    layouts_open: bool = False
    derived_handle_locals: tuple[tuple[str, str], ...] = ()
    _derived_handle_index: Mapping[str, str] = field(init=False, repr=False, compare=False)
    field_slots: tuple[tuple[str, str], ...] = ()
    field_coverage: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "_derived_handle_index",
                           MappingProxyType(dict(self.derived_handle_locals)))
        for kind, names in (("block", self.blocks), ("layout", self.layouts)):
            if len(set(names)) != len(names):
                raise ValueError("PresentationMap %s names must be distinct" % kind)
        if len(self.block_models) != len(self.blocks):
            raise ValueError("PresentationMap needs one model index per block")
        if any(not 0 <= index < len(self.models) for index in self.block_models):
            raise ValueError("PresentationMap block model index out of range")
        if len({(name, fingerprint) for name, fingerprint, _ in self.models}) != len(self.models):
            raise ValueError("PresentationMap model definitions must be distinct")

    # -- labels -----------------------------------------------------------------------------
    # A name of a projected kind is refused when the map does not know it, except in the kept
    # categories below (the residuals of N11 lot 2b-i, each pinned by a test of
    # tests/python/unit/problem/test_identity_names.py). A kept name still enters the identities,
    # as before this lot, and is never given another owner's label.
    #
    # K1  a layout handle the Program captured before resolve, in a validated snapshot, which has no
    #     compile layout yet (``layouts_open``); the resolved snapshot numbers it.
    # K2  the model definition of an authoring path (``#authoring=<serial>``, no fingerprint) that is
    #     not exactly one block model: a model outside the case, which validation then refuses with
    #     its own reason, or a name two block models share.
    # K3  a model definition fingerprinted by its standalone operator registry
    #     (``pops.operator-registry:``), which the Program reads without a block.
    def case_label(self, name: str) -> str:
        if self.case is None or name != self.case:
            raise PresentationProjectionError(
                "case name %r is not the case of this presentation map" % (name,))
        return "$case"

    def block_label(self, name: str) -> str:
        try:
            return "$block%d" % self.blocks.index(name)
        except ValueError:
            raise PresentationProjectionError(
                "block name %r is not a block of this case" % (name,)) from None

    def layout_label(self, name: str) -> str:
        if name in self.layouts:
            return "$layout%d" % self.layouts.index(name)
        if self.layouts_open:
            return name  # K1
        raise PresentationProjectionError(
            "layout name %r is not a layout of this compile" % (name,))

    def model_index(self, name: str, fingerprint: str | None) -> int | None:
        """The position of the block model ``name`` (``fingerprint``), or None if none is."""
        matches = [
            index for index, (model, recorded, _) in enumerate(self.models)
            if model == name and (fingerprint is None or recorded == fingerprint)
        ]
        return matches[0] if len(matches) == 1 else None

    def model_label(self, name: str, fingerprint: str | None, *, authoring: bool = False) -> str:
        index = self.model_index(name, fingerprint)
        if index is not None:
            return "$model%d" % index
        if authoring and fingerprint is None:
            return name  # K2
        if _is_registry_fingerprint(fingerprint):
            return name  # K3
        raise PresentationProjectionError(
            "model definition %r (fingerprint %r) is not one model of this case"
            % (name, fingerprint))

    def block_model_label(self, block: str) -> str:
        return "$model%d" % self.block_models[self.blocks.index(block)]

    def with_derived_handles(self, rows: Any) -> PresentationMap:
        """Extend the projection with producer-verified handle/local-id pairs."""
        handles = dict(self.derived_handle_locals)
        for qualified_id, local_id in rows:
            if handles.setdefault(qualified_id, local_id) != local_id:
                raise PresentationProjectionError(
                    "conflicting derived identities for handle %r" % qualified_id)
        return replace(self, derived_handle_locals=tuple(handles.items()))

    def projected_fingerprint(self, fingerprint: str | None) -> str | None:
        if fingerprint is None:
            return None
        matches = {free for _, recorded, free in self.models if recorded == fingerprint}
        if len(matches) == 1:
            return next(iter(matches))
        if _is_registry_fingerprint(fingerprint):
            return fingerprint  # K3: the registry fingerprint holds no owner name
        raise PresentationProjectionError(
            "definition fingerprint %r belongs to no model of this case" % (fingerprint,))

    def _digest_table(self) -> dict[str, str]:
        table = {original: projected for original, projected in self.digests}
        for _, recorded, free in self.models:
            if recorded is not None:
                # The bare module digest of a model definition (``scientific_hash``).
                table[recorded.rpartition(":")[2]] = free.rpartition(":")[2]
        return table

    def to_data(self) -> dict[str, Any]:
        """The map itself: presentation data, recorded beside an identity, never inside one."""
        return {
            "schema_version": PRESENTATION_SCHEMA_VERSION,
            "case": self.case,
            "blocks": list(self.blocks),
            "models": [
                {"name": name, "definition_fingerprint": fingerprint,
                 "projected_fingerprint": free}
                for name, fingerprint, free in self.models
            ],
            "block_models": list(self.block_models),
            "layouts": list(self.layouts),
            "digests": [list(row) for row in self.digests],
            "layouts_open": self.layouts_open,
        }

    # -- structured records -----------------------------------------------------------------
    def _node_name(self, kind: str, name: str, fingerprint: str | None, *,
                   authoring: bool = False) -> str:
        if kind == "case":
            return self.case_label(name)
        if kind == "block":
            return self.block_label(name)
        if kind == "model_definition":
            return self.model_label(name, fingerprint, authoring=authoring)
        if kind == "layout":
            return self.layout_label(name)
        return name

    def project_owner_record(self, record: Any) -> dict[str, Any]:
        fingerprint = record["definition_fingerprint"]
        return {
            "schema_version": record["schema_version"],
            "nodes": [
                {"kind": node["kind"],
                 "name": self._node_name(node["kind"], node["name"], fingerprint)}
                for node in record["nodes"]
            ],
            "definition_fingerprint": self.projected_fingerprint(fingerprint),
        }

    def project_owner_string(self, text: str) -> str:
        """Project ``kind:name/...``, canonical or an authoring path (``...#authoring=<serial>``).

        An authoring path names its model definition without a fingerprint: the name is projected
        when exactly one block model has it (see :meth:`model_label`). The authoring serial is not
        a name and stays.
        """
        text, authoring = _split_authoring(text)
        parsed = _parse_owner_string(text)
        if parsed is None:
            raise PresentationProjectionError("%r is not a canonical owner string" % (text,))
        nodes, fingerprint = parsed
        projected = self.projected_fingerprint(fingerprint)
        segments = []
        for kind, name in nodes:
            label = self._node_name(kind, name, fingerprint, authoring=bool(authoring))
            segment = "%s:%s" % (kind, quote(label, safe=""))
            if kind == "model_definition" and projected is not None:
                segment += "@%s" % quote(projected, safe="")
            segments.append(segment)
        return "/".join(segments) + authoring

    def project_qualified_id(self, text: str) -> str:
        parts = text.split("::")
        if len(parts) < 4 or not parts[0].startswith(_HANDLE_PREFIX):
            raise PresentationProjectionError("%r is not a qualified handle id" % (text,))
        kind = unquote(parts[2])
        owner = self.project_owner_string(parts[1])
        local = parts[3]
        derived = self.derived_local_id(text)
        if derived is not None:
            local = quote(derived, safe="")
        elif kind in ("block", "layout"):
            # The local id is quoted: a raw ``:`` starts a suffix of the label that holds the
            # qualified id (``<layout>:amr-bootstrap:<digest>`` in an AMR lowering coverage row).
            local, separator, suffix = local.partition(":")
            local = quote(self._node_name(kind, unquote(local), None), safe="") + separator + suffix
        return "::".join((parts[0], owner, parts[2], local, *parts[4:]))

    def derived_local_id(self, qualified_id: str) -> str | None:
        return self._derived_handle_index.get(qualified_id)

    def project_string(self, text: str, digests: Any = None) -> str:
        """Project a qualified id, an owner string or a label holding them; else keep ``text``.

        A label is a ``/``-separated row of ``kind:rest`` segments, which may end with a qualified
        id after a ``word:`` prefix (``block:<name>/resolved:operation:pops.handle.v1::...``): its
        ``block`` and ``case`` segments are owner names.
        """
        head, separator, rest = text.partition("/")
        if separator and head.startswith("block:") and rest.startswith(("operator:", "resolved:")):
            block = self.block_label(unquote(head.removeprefix("block:")))
            if rest.startswith("resolved:"):
                rest = self.project_string(rest, digests)
            return "block:%s/%s" % (block, rest)
        if _HANDLE_PREFIX in text:
            text = self._project_handles(text)
        elif _parse_owner_string(_split_authoring(text)[0]) is not None:
            text = self.project_owner_string(text)
        else:
            text = self._project_label(text)
        return _replace_digests(text, self._digest_table() if digests is None else digests)

    def _project_handles(self, text: str) -> str:
        """Project every qualified id of a label, and the label parts between them.

        A qualified id ends at the first ``/`` after its owner: its kind, local id and suffixes
        are quoted, so a ``/`` there starts the next part of the label.
        """
        parts = []
        position = 0
        while True:
            start = text.find(_HANDLE_PREFIX, position)
            owner = -1 if start < 0 else text.find("::", start)
            after = -1 if owner < 0 else text.find("::", owner + 2)
            if after < 0:
                parts.append(self._project_label(text[position:]))
                return "".join(parts)
            end = text.find("/", after + 2)
            end = len(text) if end < 0 else end
            parts.append(self._project_label(text[position:start]))
            parts.append(self.project_qualified_id(text[start:end]))
            position = end

    def _project_label(self, text: str) -> str:
        if ":" not in text:
            return text
        segments = text.split("/")
        if any(not _LABEL_SEGMENT.match(segment) for segment in segments[:-1]):
            return text
        result = []
        for segment in segments:
            kind, separator, name = segment.partition(":")
            if separator and kind in ("block", "case") and name and ":" not in name:
                label = self._node_name(kind, unquote(name), None)
                segment = "%s:%s" % (kind, quote(label, safe="") if "%" in name else label)
            result.append(segment)
        return "/".join(result)

    # -- whole preimage ---------------------------------------------------------------------
    def project(self, data: Any, *, keys: Any = (), values: Any = None, sets: Any = (),
                derived: Any = (), references: Any = (), reference_keys: Any = ()) -> Any:
        """Return ``data`` with every owner name replaced by its position.

        Paths are tuples of keys, with ``"*"`` for any key, ``"[]"`` for any list item and a
        leading ``"..."`` for any prefix.
        ``keys`` lists the mappings keyed by block names. ``values`` maps a path to the kind of the
        bare name held there: ``"case"``, ``"block"`` or ``"model_of_block"`` (the model of the
        block named by the path's block key), ``continuation_name`` (a block name in a
        block-owned row), ``continuation_identity`` (a qualified handle in a state row), or
        ``qualified`` (a complete qualified handle id).
        ``references`` and ``reference_keys`` identify fields
        and mapping keys that carry serialized references. ``sets`` lists the lists whose order
        is a sort by name (a set): they are sorted again by the canonical JSON of their projected
        items.
        ``derived`` lists the digests of named data that the preimage already holds in full: they
        are dropped (``None``), since a digest cannot be projected. The data must be canonical
        identity data: dicts, lists, tuples, sets, strings, bytes, numbers, booleans, None.
        """
        return _Projection(
            self, tuple(tuple(path) for path in keys), dict(values or {}),
            tuple(tuple(path) for path in sets), tuple(tuple(path) for path in derived),
            tuple(tuple(path) for path in references), tuple(tuple(path) for path in reference_keys),
        ).walk(data, ())


class _PathRules:
    """Select an ordered subsequence of rules for an exact built-in path."""

    __slots__ = ("rules", "by_last", "widths", "empty", "any_key", "any_item")

    def __init__(self, rules: tuple[tuple[tuple[Any, ...], Any], ...]) -> None:
        self.rules = rules
        self.by_last: dict[Any, list[int]] | None = None
        self.widths: list[tuple[int, bool]] = []
        self.empty: list[int] = []
        self.any_key: list[int] = []
        self.any_item: list[int] = []
        # A non-exact rule may observe even comparisons that do not match.
        if any(type(part) is not str for pattern, _ in rules for part in pattern):
            return
        self.by_last = {}
        for index, (pattern, _) in enumerate(rules):
            suffix = bool(pattern) and pattern[0] == "..."
            width = len(pattern) - suffix
            self.widths.append((width, suffix))
            if width == 0:
                self.empty.append(index)
                if suffix:
                    self.any_key.append(index)
                    self.any_item.append(index)
            elif pattern[-1] == "*":
                self.any_key.append(index)
            else:
                last = _LIST_ITEM if pattern[-1] == "[]" else pattern[-1]
                self.by_last.setdefault(last, []).append(index)

    def first(self, path: Sequence[Any], exact_path: bool) -> Any:
        if self.by_last is None or not exact_path:
            for pattern, result in self.rules:
                if _matches(pattern, path):
                    return result
            return None
        if path:
            last = path[-1]
            specific = self.by_last.get(last)
            general = self.any_item if last is _LIST_ITEM else self.any_key
            candidates = merge(specific, general) if specific and general else specific or general
        else:
            candidates = self.empty
        depth = len(path)
        for index in candidates:
            width, suffix = self.widths[index]
            if depth < width or (not suffix and depth != width):
                continue
            pattern, result = self.rules[index]
            if _matches_exact(pattern, path, width):
                return result
        return None


class _Projection:
    __slots__ = ("map", "keys", "values", "sets", "derived", "digests",
                 "references", "reference_keys", "field_slots", "field_coverage")

    def __init__(self, presentation: PresentationMap, keys: Any, values: Any, sets: Any,
                 derived: Any, references: Any, reference_keys: Any) -> None:
        self.map = presentation
        self.digests = presentation._digest_table()
        self.keys = _PathRules(tuple((pattern, True) for pattern in keys))
        normalized_values = {tuple(path): kind for path, kind in values.items()}
        self.values = _PathRules(tuple(normalized_values.items()))
        self.sets = _PathRules(tuple((pattern, True) for pattern in sets))
        self.derived = _PathRules(tuple((pattern, True) for pattern in derived))
        reference_rules: dict[Any, list[tuple[tuple[Any, ...], bool]]] = {}
        for pattern in references:
            reference_rules.setdefault(pattern[-1], []).append((pattern, True))
        self.references = {last: _PathRules(tuple(rules)) for last, rules in reference_rules.items()}
        self.reference_keys = _PathRules(tuple((pattern, True) for pattern in reference_keys))
        self.field_slots = dict(presentation.field_slots)
        self.field_coverage = dict(presentation.field_coverage)

    def walk(self, value: Any, prefix: tuple[Any, ...]) -> Any:
        # Resume one parent entry only after its child has been projected. The same
        # node can appear in several positions with different projection rules.
        identity = id(value)
        active = {identity}
        # Only the current occurrence needs its ancestor path. Suspended frames resume
        # with that same path restored, without copying it for each child.
        path = list(prefix)
        exact_path = all(type(part) is str or part is _LIST_ITEM for part in prefix)
        stack = [(identity, self._visit(value, path, None, False, exact_path))]
        result = None
        while stack:
            identity, visit = stack[-1]
            try:
                child, part, block, expression, exact_path = visit.send(result)
            except StopIteration as completed:
                stack.pop()
                active.remove(identity)
                if stack:
                    path.pop()
                result = completed.value
                continue
            path.append(part)
            if type(child) in (str, bytes, int, float, bool, type(None)):
                result = self._scalar(child, path, exact_path)
                path.pop()
                continue
            child_identity = id(child)
            if child_identity in active:
                raise PresentationProjectionError("projection contains a cycle at %r" % (tuple(path),))
            active.add(child_identity)
            stack.append((child_identity, self._visit(child, path, block, expression, exact_path)))
            result = None
        return result

    def _scalar(self, value: Any, path: list[Any], exact_path: bool) -> Any:
        if self.derived.first(path, exact_path):
            return None
        return self._string_reference(value, path, exact_path) if type(value) is str else value

    def _string_reference(self, value: str, path: list[Any], exact_path: bool) -> str:
        if path:
            last = path[-1]
            rules = self.references.get("[]" if last is _LIST_ITEM else last)
            if rules is not None and rules.first(path, exact_path):
                return self._reference(value)
        return value

    def _visit(
        self, value: Any, path: list[Any], block: str | None, expression: bool, exact_path: bool,
    ) -> Generator[tuple[Any, Any, str | None, bool, bool], Any, Any]:
        if self.derived.first(path, exact_path):
            return None
        if isinstance(value, dict):
            return (yield from self._mapping(value, path, block, expression=expression,
                                             exact_path=exact_path))
        if isinstance(value, (list, tuple)):
            items = []
            for item in value:
                items.append((yield item, _LIST_ITEM, block, expression, exact_path))
            # These two forms encode expression_handle_key and ValueExpr in the Expr DAG.
            if expression and len(value) == 2 and value[0] in ("qualified", "handle_value") \
                    and isinstance(value[1], str):
                items[1] = self.map.project_string(value[1], self.digests)
            if self.sets.first(path, exact_path):
                items.sort(key=_order_key)
            return items if isinstance(value, list) else tuple(items)
        if isinstance(value, (set, frozenset)):
            projected = set()
            for item in value:
                projected.add((yield item, _LIST_ITEM, block, expression, exact_path))
            return frozenset(projected)
        if isinstance(value, str):
            return self._string_reference(value, path, exact_path)
        return value

    def _reference(self, value: str) -> str:
        target = self.field_coverage.get(value)
        return self.map.project_string(value, self.digests) if target is None else target

    def _bare(self, value: str, kind: str, block: str | None, record: dict[str, Any]) -> str:
        if kind == "qualified":
            return self.map.project_qualified_id(value)
        if kind == "case":
            return self.map.case_label(value)
        if kind == "block":
            return self.map.block_label(value)
        if kind == "continuation_name":
            if record["kind"] in ("state", "auxiliary"):
                return self.map.block_label(value)
            if record["kind"] in ("field_value", "field_observation", "solver_cache"):
                return self.field_slots.get(value, value)
            return value
        if kind == "continuation_identity":
            if record["kind"] == "state":
                return self.map.project_qualified_id(value)
            if record["kind"] in ("field_value", "field_observation", "solver_cache"):
                slot = self.field_slots.get(record["name"])
                if slot is not None:
                    return record["kind"] + ":" + slot
            return value
        if kind == "model_of_block":
            if block is None:
                raise PresentationProjectionError("model name has no enclosing block record")
            return self.map.block_model_label(block)
        raise ValueError("unknown presentation kind %r" % (kind,))

    def _mapping(
        self, value: dict[str, Any], path: list[Any], block: str | None, *,
        expression: bool, exact_path: bool,
    ) -> Generator[tuple[Any, Any, str | None, bool, bool], Any, dict[str, Any]]:
        if set(value) == _OWNER_RECORD_KEYS and isinstance(value.get("nodes"), list):
            return self.map.project_owner_record(value)
        if value.keys() == _COMPONENT_KEY_FIELDS:
            return {**value, "owner_qid": self.map.project_owner_string(value["owner_qid"])}
        keyed = self.keys.first(path, exact_path)
        referenced = not keyed and self.reference_keys.first(path, exact_path)
        handle = {"qualified_id", "local_id", "kind", "owner_path"} <= value.keys()
        expression = expression or value.get("protocol") == "pops.expr.dag.v1"
        result = {}
        for key, item in value.items():
            exact_item_path = exact_path and type(key) is str
            projected_key = (self.map.block_label(key) if keyed
                             else self._reference(key)
                             if referenced and isinstance(key, str) else key)
            if projected_key in result:
                raise PresentationProjectionError("projection merged two keys at %r" % (tuple(path),))
            kind = None
            if isinstance(item, str):
                path.append(key)
                kind = self.values.first(path, exact_item_path)
                path.pop()
            if kind is not None:
                result[projected_key] = self._bare(item, kind, block, value)
            elif handle and key == "qualified_id":
                projected = self.map.project_qualified_id(item)
                result[projected_key] = (
                    _replace_digests(projected, self.digests)
                    if value["kind"] == "initial_condition_provider" else projected)
            elif handle and key == "local_id" and value["kind"] in ("block", "layout"):
                result[projected_key] = self.map._node_name(value["kind"], item, None)
            elif handle and key == "local_id" and value["kind"] == "initial_condition_provider":
                result[projected_key] = _replace_digests(item, self.digests)
            elif handle and key == "local_id":
                derived = self.map.derived_local_id(value["qualified_id"])
                result[projected_key] = item if derived is None else derived
            else:
                result[projected_key] = yield (
                    item, key, key if keyed else block, expression, exact_item_path)
        if value.get("op") == "state" and isinstance(value.get("block"), dict) \
                and value.get("name") == value["block"].get("local_id") \
                and isinstance(value.get("name"), str):
            # The Program names a block state after its block (a derived label, SU8).
            result["name"] = self.map.block_label(value["name"])
        if value.keys() == _BOUNDARY_PROVIDER_FIELDS and value["plan_type"] == "boundary_providers":
            result["needs"].sort(key=_order_key)
            result["bindings"].sort(key=lambda row: _order_key(row["need"]))
        if value.keys() == _GHOST_PLAN_FIELDS and value["plan_type"] == "ghost_producers":
            # This plan sorts these collections by generated region/handle identity. The
            # dependency order itself is carried by each producer's edges.
            for name in _GHOST_PLAN_SETS:
                result[name].sort(key=_order_key)
            result["coverage"]["regions"].sort(key=_order_key)
        if keyed:
            result = dict(sorted(result.items(), key=lambda row: _label_order(row[0])))
        return result


def _replace_digests(text: str, table: dict[str, str]) -> str:
    if not table or len(text) < 64:
        return text
    return _DIGEST.sub(lambda match: table.get(match.group(0), match.group(0)), text)


def _order_key(item: Any) -> str:
    return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      default=lambda value: value.hex() if isinstance(value, bytes)
                      else sorted(value, key=_order_key))


def _label_order(label: str) -> tuple[int, str]:
    digits = label.removeprefix("$block")
    return (int(digits), label) if digits.isdigit() else (-1, label)


def _matches(pattern: tuple[Any, ...], path: Sequence[Any]) -> bool:
    """Match a path; ``"*"`` is any key, and a leading ``"..."`` matches any prefix."""
    if pattern and pattern[0] == "...":
        pattern = pattern[1:]
        if len(path) < len(pattern):
            return False
        path = path[len(path) - len(pattern):]
    if len(pattern) != len(path):
        return False
    for expected, actual in zip(pattern, path):
        if not (actual is _LIST_ITEM if expected == "[]" else
                actual is not _LIST_ITEM if expected == "*" else expected == actual):
            return False
    return True


def _matches_exact(pattern: tuple[Any, ...], path: Sequence[Any], width: int) -> bool:
    # Exact strings have no comparison hooks. Check the nearest ancestors first without
    # slicing the path. Most suffix rules differ on one of these components.
    for index in range(-1, -width - 1, -1):
        expected, actual = pattern[index], path[index]
        if not (actual is _LIST_ITEM if expected == "[]" else
                actual is not _LIST_ITEM if expected == "*" else expected == actual):
            return False
    return True


def _is_registry_fingerprint(fingerprint: str | None) -> bool:
    return fingerprint is not None and fingerprint.startswith(_REGISTRY_FINGERPRINT)


def _split_authoring(text: str) -> tuple[str, str]:
    """Split an authoring path into its owner string and its ``#authoring=<serial>`` suffix."""
    base, separator, serial = text.rpartition(_AUTHORING)
    if separator and serial.isdigit() and "#" not in base:
        return base, separator + serial
    return text, ""


def _parse_owner_string(text: str) -> tuple[list[tuple[str, str]], str | None] | None:
    """Parse ``kind:name/...`` (with ``@fingerprint`` on a model definition), or return None."""
    if not text or "#" in text or "::" in text:
        return None
    nodes = []
    fingerprint = None
    for raw in text.split("/"):
        kind, separator, encoded = raw.partition(":")
        if not separator or kind not in _OWNER_KINDS or not encoded:
            return None
        if kind == "model_definition" and "@" in encoded:
            encoded, _, encoded_fingerprint = encoded.partition("@")
            if fingerprint is not None or not encoded or not encoded_fingerprint:
                return None
            fingerprint = unquote(encoded_fingerprint)
        elif "@" in encoded:
            return None
        nodes.append((kind, unquote(encoded)))
    return nodes, fingerprint


def projected_fingerprint(digest: str) -> str:
    """Spell the name-free digest of one model definition as a definition fingerprint."""
    return _PROJECTED_FINGERPRINT % digest


__all__ = [
    "LABEL_PREFIX",
    "PRESENTATION_SCHEMA_VERSION",
    "PresentationMap",
    "PresentationProjectionError",
    "ReservedOwnerNameError",
    "require_owner_name",
    "projected_fingerprint",
]
