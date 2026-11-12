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
- a qualified id (``pops.handle.v<k>::<owner>::<kind>::<local>[::...]``) or an owner string
  (``kind:name/...``) is parsed and rebuilt from its renamed parts: a name is never replaced
  inside a longer string;
- a bare name is renamed only at a path the caller declares (``keys`` for a mapping keyed by block
  names, ``values`` for one exact field).

Every other name (quantities, components, parameters, fields, terms, the Program and its nodes,
frames, realizations, Python bodies) is kept: those are the kinds that gap C52 lists. A name of a
projected kind that the map does not know (an owner of another case, a layout no block uses, a model
that is not a block model) keeps itself: it is never replaced by the label of another owner. Such a
name still enters the identities, as before this lot (the residuals of N11 lot 2b-i).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote


PRESENTATION_SCHEMA_VERSION = 1

_OWNER_KINDS = frozenset(
    ("case", "model_definition", "block", "layout", "descriptor", "consumer", "shared"))
_OWNER_RECORD_KEYS = frozenset(("schema_version", "nodes", "definition_fingerprint"))
_HANDLE_PREFIX = "pops.handle.v"
_AUTHORING = "#authoring="
# A projected definition fingerprint keeps the shape the owner records accept.
_PROJECTED_FINGERPRINT = "pops.module.projected:sha256:%s"
_DIGEST = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_LABEL_SEGMENT = re.compile(r"^[a-z_]+:[^:/]+$")


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

    def __post_init__(self) -> None:
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
    # A name the map does not know keeps itself: an owner outside this case (a layout record a
    # test builds under its own case), a layout no block uses (a synthetic AMR bootstrap storage),
    # a layout the Program captured before resolve (the validated snapshot has no compile layout).
    # Keeping it is the residual of N11 lot 2b-i: such a name still enters the identities, as
    # before, and is never replaced by the label of another owner.
    def case_label(self, name: str) -> str:
        return "$case" if self.case is not None and name == self.case else name

    def block_label(self, name: str) -> str:
        return "$block%d" % self.blocks.index(name) if name in self.blocks else name

    def layout_label(self, name: str) -> str:
        return "$layout%d" % self.layouts.index(name) if name in self.layouts else name

    def model_index(self, name: str, fingerprint: str | None) -> int | None:
        """The position of the block model ``name`` (``fingerprint``), or None.

        None for a model definition that is not one block model of the case: a model the Program
        reads without a block (its definition fingerprint is then its operator registry's, or
        an earlier one), a model outside the case (refused by validation with its own reason),
        or a name two block models share when no fingerprint tells them apart (an authoring path).
        """
        matches = [
            index for index, (model, recorded, _) in enumerate(self.models)
            if model == name and (fingerprint is None or recorded == fingerprint)
        ]
        return matches[0] if len(matches) == 1 else None

    def model_label(self, name: str, fingerprint: str | None) -> str:
        """``$model<k>`` for a block model; any other model definition keeps its name.

        Keeping it is the residual of N11 lot 2b-i: such a name still enters the identities, as
        before; it is never replaced by the label of another model.
        """
        index = self.model_index(name, fingerprint)
        return name if index is None else "$model%d" % index

    def block_model_label(self, block: str) -> str:
        return "$model%d" % self.block_models[self.blocks.index(block)]

    def projected_fingerprint(self, fingerprint: str | None) -> str | None:
        if fingerprint is None:
            return None
        matches = {free for _, recorded, free in self.models if recorded == fingerprint}
        # The fingerprint of a model definition outside the block models is kept with its name.
        return next(iter(matches)) if len(matches) == 1 else fingerprint

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
        }

    # -- structured records -----------------------------------------------------------------
    def _node_name(self, kind: str, name: str, fingerprint: str | None) -> str:
        if kind == "case":
            return self.case_label(name)
        if kind == "block":
            return self.block_label(name)
        if kind == "model_definition":
            return self.model_label(name, fingerprint)
        if kind == "layout":
            return self.layout_label(name)
        return name

    def project_owner_record(self, record: Any) -> dict[str, Any]:
        if self._foreign([(node["kind"], node["name"]) for node in record["nodes"]]):
            return dict(record)
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
        if self._foreign(nodes):
            return text + authoring
        projected = self.projected_fingerprint(fingerprint)
        segments = []
        for kind, name in nodes:
            label = self._node_name(kind, name, fingerprint)
            segment = "%s:%s" % (kind, quote(label, safe=""))
            if kind == "model_definition" and projected is not None:
                segment += "@%s" % quote(projected, safe="")
            segments.append(segment)
        return "/".join(segments) + authoring

    def _foreign(self, nodes: Any) -> bool:
        """Whether an owner path belongs to another case: none of its names is then projected."""
        return any(kind == "case" and name != self.case for kind, name in nodes)

    def project_qualified_id(self, text: str) -> str:
        parts = text.split("::")
        if len(parts) < 4 or not parts[0].startswith(_HANDLE_PREFIX):
            raise PresentationProjectionError("%r is not a qualified handle id" % (text,))
        kind = unquote(parts[2])
        owner = self.project_owner_string(parts[1])
        local = parts[3]
        parsed = _parse_owner_string(_split_authoring(parts[1])[0])
        if kind in ("block", "layout") and parsed is not None and not self._foreign(parsed[0]):
            local = quote(self._node_name(kind, unquote(local), None), safe="")
        return "::".join((parts[0], owner, parts[2], local, *parts[4:]))

    def project_string(self, text: str, digests: Any = None) -> str:
        """Project a qualified id, an owner string or a label holding them; else keep ``text``.

        A label is a ``/``-separated row of ``kind:rest`` segments, which may end with a qualified
        id after a ``word:`` prefix (``block:<name>/resolved:operation:pops.handle.v1::...``): its
        ``block`` and ``case`` segments are owner names.
        """
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
        if ":" not in text or " " in text:
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
                derived: Any = ()) -> Any:
        """Return ``data`` with every owner name replaced by its position.

        Paths are tuples of keys, with ``"*"`` for any key, ``"[]"`` for any list item and a
        leading ``"..."`` for any prefix.
        ``keys`` lists the mappings keyed by block names. ``values`` maps a path to the kind of the
        bare name held there: ``"case"``, ``"block"`` or ``"model_of_block"`` (the model of the
        block named by the path's block key). ``sets`` lists the lists whose order is a sort by
        name (a set): they are sorted again by the canonical JSON of their projected items.
        ``derived`` lists the digests of named data that the preimage already holds in full: they
        are dropped (``None``), since a digest cannot be projected. The data must be canonical
        identity data: dicts, lists, tuples, sets, strings, bytes, numbers, booleans, None.
        """
        return _Projection(
            self, tuple(tuple(path) for path in keys), dict(values or {}),
            tuple(tuple(path) for path in sets), tuple(tuple(path) for path in derived),
        ).walk(data, ())


class _Projection:
    __slots__ = ("map", "keys", "values", "sets", "derived", "digests")

    def __init__(self, presentation: PresentationMap, keys: Any, values: Any, sets: Any,
                 derived: Any) -> None:
        self.map = presentation
        self.digests = presentation._digest_table()
        self.keys = keys
        self.values = {tuple(path): kind for path, kind in values.items()}
        self.sets = sets
        self.derived = derived

    def walk(self, value: Any, path: tuple[Any, ...]) -> Any:
        if any(_matches(pattern, path) for pattern in self.derived):
            return None
        if isinstance(value, dict):
            return self._mapping(value, path)
        if isinstance(value, (list, tuple)):
            items = [self.walk(item, path + ("[]",)) for item in value]
            if any(_matches(pattern, path) for pattern in self.sets):
                items.sort(key=_order_key)
            return items if isinstance(value, list) else tuple(items)
        if isinstance(value, (set, frozenset)):
            return frozenset(self.walk(item, path + ("[]",)) for item in value)
        if isinstance(value, str):
            kind = _lookup(self.values, path)
            if kind is not None:
                return self._bare(value, kind, path)
            return self.map.project_string(value, self.digests)
        return value

    def _bare(self, value: str, kind: str, path: tuple[Any, ...]) -> str:
        if kind == "case":
            return self.map.case_label(value)
        if kind == "block":
            return self.map.block_label(value)
        if kind == "block_if_listed":
            # A row naming either a block or a PoPS object (``accepted Program exchange mailbox``).
            return self.map.block_label(value) if value in self.map.blocks else value
        if kind == "model_of_block":
            block = next((item for item in path if item in self.map.blocks), None)
            if block is None:
                raise PresentationProjectionError(
                    "model name at %r has no block key on its path" % (path,))
            return self.map.block_model_label(block)
        raise ValueError("unknown presentation kind %r" % (kind,))

    def _foreign_handle(self, qualified_id: Any) -> bool:
        parts = qualified_id.split("::") if isinstance(qualified_id, str) else ()
        parsed = _parse_owner_string(_split_authoring(parts[1])[0]) if len(parts) > 1 else None
        return parsed is not None and self.map._foreign(parsed[0])

    def _mapping(self, value: dict[str, Any], path: tuple[Any, ...]) -> dict[str, Any]:
        if set(value) == _OWNER_RECORD_KEYS and isinstance(value.get("nodes"), list):
            return self.map.project_owner_record(value)
        keyed = any(_matches(pattern, path) for pattern in self.keys)
        result = {}
        for key, item in value.items():
            projected_key = (self.map.block_label(key) if keyed
                             else self.map.project_string(key, self.digests)
                             if isinstance(key, str) else key)
            if projected_key in result:
                raise PresentationProjectionError("projection merged two keys at %r" % (path,))
            # The original key stays on the path, so a rule below a block key finds its block.
            result[projected_key] = self.walk(item, path + (key,))
        if "qualified_id" in value and "local_id" in value \
                and value.get("kind") in ("block", "layout") and isinstance(value["local_id"], str) \
                and not self._foreign_handle(value["qualified_id"]):
            result["local_id"] = self.map._node_name(value["kind"], value["local_id"], None)
        if value.get("op") == "state" and isinstance(value.get("block"), dict) \
                and value.get("name") == value["block"].get("local_id") \
                and isinstance(value.get("name"), str):
            # The Program names a block state after its block (a derived label, SU8).
            result["name"] = self.map.block_label(value["name"])
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


def _matches(pattern: tuple[Any, ...], path: tuple[Any, ...]) -> bool:
    """Match a path; ``"*"`` is any key, and a leading ``"..."`` matches any prefix."""
    if pattern and pattern[0] == "...":
        pattern = pattern[1:]
        if len(path) < len(pattern):
            return False
        path = path[len(path) - len(pattern):]
    if len(pattern) != len(path):
        return False
    return all(expected == "*" and actual != "[]" or expected == actual
               for expected, actual in zip(pattern, path))


def _lookup(table: dict[tuple[Any, ...], str], path: tuple[Any, ...]) -> str | None:
    for pattern, kind in table.items():
        if _matches(pattern, path):
            return kind
    return None


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
    "PRESENTATION_SCHEMA_VERSION",
    "PresentationMap",
    "PresentationProjectionError",
    "projected_fingerprint",
]
