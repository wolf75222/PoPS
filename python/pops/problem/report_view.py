"""Typed, inert inspection report for a :class:`pops.Case`.

``Case.inspect()`` returns a :class:`CaseReport`: a typed immutable report carrying the
assembly's name / blocks / fields / params / consumers / requirements /
capabilities as ATTRIBUTES, with :meth:`to_dict` the JSON bridge. It is inert -- built from the
registries' metadata, it triggers no validation and no compilation. This is distinct from the
per-registry ``inspect()`` dicts it composes and from ``Case.to_dict()`` (the array-free
serialisation the snapshot / codegen consume, a superset that adds the stable handle ids).
"""
from __future__ import annotations

from typing import Any

from pops._report import Report


class CaseReport(Report):
    """The typed inspection report of a :class:`~pops.problem.problem.Case`.

    Attributes (all read directly): ``name`` / ``category`` / ``native_id`` / ``options`` /
    ``requirements`` / ``capabilities`` / ``blocks`` / ``fields`` / ``params`` /
    ``consumers`` / ``time``. :meth:`to_dict` is the JSON bridge (its keys
    match the historical ``inspect()`` dict byte-for-byte so a ``to_dict()`` consumer is unchanged).
    """

    report_type = "case"
    schema_version = 1

    def __init__(self, payload: Any) -> None:
        # Store each field as an attribute AND keep the ordered payload for the byte-identical
        # to_dict (the historical inspect() dict shape).
        self._payload = dict(payload)
        for key, value in payload.items():
            setattr(self, key, value)

    def to_dict(self) -> Any:
        # Byte-identical to the historical Case.inspect() dict, stamped with the report identity.
        return self._stamp(dict(self._payload))

    def __str__(self) -> str:
        return ("case %r [%s]: blocks=%s fields=%s params=%s consumers=%s time=%s"
                % (self._payload.get("name"), self._payload.get("category"),
                   list(self._payload.get("blocks", {})), list(self._payload.get("fields", {})),
                   list(self._payload.get("params", {})), self._payload.get("consumers"),
                   self._payload.get("time")))


__all__ = ["CaseReport"]
