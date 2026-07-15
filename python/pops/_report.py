"""Typed, inert reports shared by every PoPS lifecycle phase.

``ReportTree`` is the single diagnostic/explanation value.  It is deeply immutable, contains only
detached JSON evidence, and composes recursively without a mutable accumulator.  ``Report`` remains
the small base used by aggregate inspection views whose domain-specific attributes are their public
API; both families use ``to_dict`` as their one explicit mapping bridge.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import json
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar, cast


class ReportPhase(str, Enum):
    """Closed lifecycle vocabulary for a :class:`ReportTree` node."""

    AUTHORING = "authoring"
    VALIDATION = "validation"
    COMPILE = "compile"
    BIND = "bind"
    RUNTIME = "runtime"
    INSPECTION = "inspection"


class ReportSeverity(str, Enum):
    """Closed severity vocabulary for a :class:`ReportTree` node."""

    TRACE = "trace"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")


@dataclass(frozen=True, slots=True)
class ReportTree:
    """One immutable diagnostic, explanation, or inspection node.

    ``code`` is a stable dotted identifier (for example ``validation.field.invalid``), never a
    rendered message.  ``owner`` and ``evidence`` are detached on construction; mutating their input
    containers later cannot alter the report.  Composition is functional via :meth:`with_child` and
    :meth:`with_children`.
    """

    phase: ReportPhase | str
    severity: ReportSeverity | str
    code: str
    message: str = ""
    source: str | None = None
    notes: tuple[str, ...] | Sequence[str] = ()
    owner: Any = field(default=None, hash=False)
    evidence: Mapping[str, Any] = field(default_factory=dict, hash=False)
    actions: tuple[str, ...] | Sequence[str] = ()
    children: tuple[ReportTree, ...] | Sequence[ReportTree] = ()

    report_type: ClassVar[str] = "report_tree"
    schema_version: ClassVar[int] = 1

    def __post_init__(self) -> None:
        try:
            phase = self.phase if isinstance(self.phase, ReportPhase) else ReportPhase(self.phase)
        except ValueError as exc:
            raise ValueError("unknown report phase %r; expected one of %s" % (
                self.phase, ", ".join(item.value for item in ReportPhase))) from exc
        try:
            severity = (self.severity if isinstance(self.severity, ReportSeverity)
                        else ReportSeverity(self.severity))
        except ValueError as exc:
            raise ValueError("unknown report severity %r; expected one of %s" % (
                self.severity, ", ".join(item.value for item in ReportSeverity))) from exc
        code = str(self.code)
        if not _CODE_RE.fullmatch(code):
            raise ValueError(
                "report code %r must be a stable namespaced identifier such as "
                "'validation.field.invalid'" % code)
        children = tuple(self.children)
        if not all(isinstance(child, ReportTree) for child in children):
            raise TypeError("ReportTree children must contain only ReportTree nodes")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("ReportTree evidence must be a JSON mapping")

        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "source", None if self.source is None else str(self.source))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(self, "owner", _freeze_json(_owner_payload(self.owner), "owner"))
        object.__setattr__(self, "evidence", _freeze_json(dict(self.evidence), "evidence"))
        object.__setattr__(self, "actions", tuple(str(action) for action in self.actions))
        object.__setattr__(self, "children", children)

    @property
    def ok(self) -> bool:
        """Whether neither this node nor any descendant has error severity."""
        return self.severity is not ReportSeverity.ERROR and all(child.ok for child in self.children)

    @property
    def issues(self) -> tuple[ReportTree, ...]:
        """Warning/error descendants, including this node when applicable, in tree order."""
        return tuple(node for node in self.walk()
                     if node.severity in (ReportSeverity.WARNING, ReportSeverity.ERROR))

    def walk(self) -> Iterator[ReportTree]:
        """Yield this node followed by descendants in deterministic pre-order."""
        yield self
        for child in self.children:
            yield from child.walk()

    def with_child(self, child: ReportTree) -> ReportTree:
        """Return a new tree with one appended child."""
        if not isinstance(child, ReportTree):
            raise TypeError("ReportTree child must be a ReportTree")
        return replace(self, children=tuple(self.children) + (child,))

    def with_children(self, children: Sequence[ReportTree]) -> ReportTree:
        """Return a new tree with @p children appended in their supplied order."""
        additions = tuple(children)
        if not all(isinstance(child, ReportTree) for child in additions):
            raise TypeError("ReportTree children must contain only ReportTree nodes")
        return replace(self, children=tuple(self.children) + additions)

    def error(self, source: str, code: str, message: str, *,
              context: Mapping[str, Any] | None = None,
              evidence: Mapping[str, Any] | None = None,
              alternatives: Sequence[str] = (), actions: Sequence[str] = (),
              notes: Sequence[str] = (), owner: Any = None) -> ReportTree:
        """Return a new tree with one error child built from diagnostic parts.

        A local ``code`` is qualified as ``<source>.<code>``; an already namespaced code is kept.
        ``context`` and ``alternatives`` are accepted as producer-facing vocabulary and are folded
        into the node's canonical ``evidence`` and ``actions`` fields.  This method never mutates the
        receiver, so callers must assign its return value.
        """
        source_text = str(source)
        code_text = str(code)
        qualified_code = code_text if _CODE_RE.fullmatch(code_text) else "%s.%s" % (
            source_text, code_text)
        details = dict(context or {})
        details.update(dict(evidence or {}))
        child = ReportTree(
            phase=self.phase, severity=ReportSeverity.ERROR, code=qualified_code,
            message=message, source=source_text, notes=notes,
            owner=self.owner if owner is None else owner, evidence=details,
            actions=tuple(alternatives) + tuple(actions),
        )
        return self.with_child(child)

    def extend(self, other: ReportTree) -> ReportTree:
        """Return a new tree containing @p other as an appended diagnostic subtree."""
        if not isinstance(other, ReportTree):
            raise TypeError("ReportTree.extend expects a ReportTree")
        return self.with_child(other)

    def by_source(self) -> dict[str, tuple[ReportTree, ...]]:
        """Group warning/error nodes by their stable source label."""
        grouped: dict[str, list[ReportTree]] = {}
        for issue in self.issues:
            grouped.setdefault(issue.source or "report", []).append(issue)
        return {source: tuple(nodes) for source, nodes in grouped.items()}

    def raise_if_error(self) -> None:
        """Raise :class:`DiagnosticError` carrying this exact tree when it is not OK."""
        if not self.ok:
            raise DiagnosticError(self)

    def inspect(self) -> ReportTree:
        """Typed inspection is the value itself; :func:`pops.inspect` is the dict bridge."""
        return self

    def explain(self) -> ReportTree:
        """A report already is its complete typed explanation."""
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-ready representation with a stable field order."""
        return {
            "report_type": self.report_type,
            "schema_version": self.schema_version,
            "phase": cast(ReportPhase, self.phase).value,
            "severity": cast(ReportSeverity, self.severity).value,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "notes": list(self.notes),
            "owner": _thaw_json(self.owner),
            "evidence": _thaw_json(self.evidence),
            "actions": list(self.actions),
            "children": [child.to_dict() for child in self.children],
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReportTree:
        """Rebuild a tree from :meth:`to_dict`, rejecting schema or verdict drift."""
        if not isinstance(payload, Mapping):
            raise TypeError("ReportTree.from_dict expects a mapping")
        if payload.get("report_type", cls.report_type) != cls.report_type:
            raise ValueError("not a ReportTree payload: %r" % payload.get("report_type"))
        if payload.get("schema_version", cls.schema_version) != cls.schema_version:
            raise ValueError("unsupported ReportTree schema version %r" %
                             payload.get("schema_version"))
        required = ("phase", "severity", "code")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError("ReportTree payload missing required field(s): %s" % ", ".join(missing))
        tree = cls(
            phase=payload["phase"], severity=payload["severity"], code=payload["code"],
            message=payload.get("message", ""), source=payload.get("source"),
            notes=payload.get("notes", ()), owner=payload.get("owner"),
            evidence=payload.get("evidence", {}), actions=payload.get("actions", ()),
            children=tuple(cls.from_dict(child) for child in payload.get("children", ())),
        )
        if "ok" in payload and payload["ok"] is not tree.ok:
            raise ValueError("ReportTree payload ok verdict disagrees with its severities")
        return tree

    def to_json(self, path: Any = None, *, indent: int | None = 2) -> Any:
        """Serialize deterministically; write to @p path when supplied, else return text."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True,
                          separators=(",", ":") if indent is None else None)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @classmethod
    def from_json(cls, source: Any) -> ReportTree:
        """Read JSON text, bytes, or a path and rebuild the typed tree."""
        if hasattr(source, "read"):
            payload = json.load(source)
        elif isinstance(source, bytes):
            payload = json.loads(source.decode("utf-8"))
        else:
            text = str(source)
            try:
                if not text.lstrip().startswith(("{", "[")):
                    with open(text, encoding="utf-8") as handle:
                        payload = json.load(handle)
                else:
                    payload = json.loads(text)
            except OSError:
                payload = json.loads(text)
        return cls.from_dict(payload)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        lines: list[str] = []
        for depth, node in _walk_with_depth(self):
            message = ": %s" % node.message if node.message else ""
            lines.append("%s[%s] %s%s" % (
                "  " * depth, cast(ReportSeverity, node.severity).value, node.code, message))
        return "\n".join(lines)


class DiagnosticError(ValueError):
    """Strict gate failure carrying the immutable report that explains it."""

    def __init__(self, report: ReportTree) -> None:
        if not isinstance(report, ReportTree):
            raise TypeError("DiagnosticError requires a ReportTree")
        if report.ok:
            raise ValueError("DiagnosticError requires a ReportTree containing an error")
        self.report = report
        super().__init__(str(report))


def _walk_with_depth(root: ReportTree) -> Iterator[tuple[int, ReportTree]]:
    stack = [(0, root)]
    while stack:
        depth, node = stack.pop()
        yield depth, node
        stack.extend((depth + 1, child) for child in reversed(node.children))


def _owner_payload(owner: Any) -> Any:
    """Project a domain owner to detached JSON identity without retaining the live object."""
    if owner is None or isinstance(owner, (str, int, float, bool, Mapping, list, tuple)):
        return owner
    canonical = getattr(owner, "canonical", None)
    if callable(canonical):
        owner = canonical()
    to_data = getattr(owner, "to_data", None)
    if callable(to_data):
        return to_data()
    qualified_id = getattr(owner, "qualified_id", None)
    if qualified_id is not None:
        return str(qualified_id)
    name = getattr(owner, "name", None)
    if name is not None:
        return str(name)
    raise TypeError("ReportTree owner must expose detached JSON identity (to_data/qualified_id/name)")


def _freeze_json(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ReportTree %s contains non-finite JSON number" % path)
        return value
    if isinstance(value, Enum):
        return _freeze_json(value.value, path)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise TypeError("ReportTree %s mapping keys must be strings" % path)
            frozen[key] = _freeze_json(value[key], "%s.%s" % (path, key))
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, "%s[%d]" % (path, index))
                     for index, item in enumerate(value))
    raise TypeError("ReportTree %s contains non-JSON value %s.%s" % (
        path, type(value).__module__, type(value).__qualname__))


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class Report:
    """Base of domain-specific typed inspection reports; never a ``dict`` subclass."""

    report_type = "report"
    schema_version = 1

    def _stamp(self, payload: dict) -> dict:
        stamped = {"report_type": self.report_type, "schema_version": self.schema_version}
        stamped.update(payload)
        return stamped

    def to_dict(self) -> dict:  # pragma: no cover - overridden by concrete reports
        return self._stamp({})

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self) -> str:
        lines = ["%s (schema v%d):" % (self.report_type, self.schema_version)]
        for key, value in self.to_dict().items():
            if key not in ("report_type", "schema_version"):
                lines.append("  %-14s %s" % (key, _short(value)))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "%s(report_type=%r)" % (type(self).__name__, self.report_type)


def _short(value: Any) -> str:
    if isinstance(value, dict):
        return "{%d key(s)}" % len(value)
    if isinstance(value, (list, tuple)):
        return "[%d item(s)]" % len(value)
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


__all__ = [
    "DiagnosticError", "Report", "ReportPhase", "ReportSeverity", "ReportTree",
]
