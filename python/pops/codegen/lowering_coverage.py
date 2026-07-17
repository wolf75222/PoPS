"""Immutable, machine-readable coverage of one authoring-to-codegen lowering.

The report is deliberately small and strict.  It is an audit boundary, not a
free-form log: every source declaration occurs once and has one of four
exhaustive dispositions.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


LOWERING_DISPOSITIONS = frozenset({"lowered", "derived", "documentary", "rejected"})


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("%s must be a non-empty string" % where)
    return value


@dataclass(frozen=True, slots=True)
class LoweringCoverageRow:
    """The single lowering disposition of one stable source identifier."""

    source: str
    disposition: str
    targets: tuple[str, ...] = ()
    rule: str | None = None
    gate: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _identifier(self.source, "row source"))
        if self.disposition not in LOWERING_DISPOSITIONS:
            raise ValueError(
                "row disposition must be exactly one of %s; got %r"
                % (sorted(LOWERING_DISPOSITIONS), self.disposition))
        if isinstance(self.targets, str):
            raise TypeError("row targets must be an iterable of target identifiers, not a string")
        targets = tuple(_identifier(value, "row target") for value in self.targets)
        if len(set(targets)) != len(targets):
            raise ValueError("row %r contains duplicate targets" % self.source)
        object.__setattr__(self, "targets", targets)
        if self.rule is not None:
            _identifier(self.rule, "row rule")
        if self.gate is not None:
            _identifier(self.gate, "row gate")

        if self.disposition == "lowered" and not targets:
            raise ValueError("lowered row %r must name at least one target" % self.source)
        if self.disposition == "derived" and self.rule is None:
            raise ValueError("derived row %r must name its derivation rule" % self.source)
        if self.disposition == "documentary" and targets:
            raise ValueError("documentary row %r cannot name behavior targets" % self.source)
        if self.disposition == "rejected":
            if self.gate is None:
                raise ValueError("rejected row %r must name its rejection gate" % self.source)
            if targets:
                raise ValueError("rejected row %r cannot name targets" % self.source)
        if self.disposition != "derived" and self.rule is not None:
            raise ValueError("only derived rows may name a derivation rule")
        if self.disposition != "rejected" and self.gate is not None:
            raise ValueError("only rejected rows may name a rejection gate")

    @property
    def source_id(self) -> str:
        """Explicit alias used by graph-oriented consumers."""
        return self.source

    def to_data(self) -> dict[str, Any]:
        return {"source": self.source, "disposition": self.disposition,
                "targets": list(self.targets), "rule": self.rule, "gate": self.gate}

    @classmethod
    def from_data(cls, data: Any) -> "LoweringCoverageRow":
        if not isinstance(data, Mapping) or set(data) != {
            "source", "disposition", "targets", "rule", "gate",
        }:
            raise TypeError("LoweringCoverageRow data has an invalid schema")
        if not isinstance(data["targets"], list):
            raise TypeError("LoweringCoverageRow targets must be a list")
        return cls(data["source"], data["disposition"], tuple(data["targets"]),
                   data["rule"], data["gate"])


class LoweringCoverageReport:
    """Canonical immutable rows plus both directions of their coverage graph."""

    __slots__ = ("rows", "source_to_targets", "target_to_sources", "_sealed")

    def __init__(self, rows: Iterable[LoweringCoverageRow] = ()) -> None:
        if isinstance(rows, (str, bytes)):
            raise TypeError("LoweringCoverageReport rows must be coverage rows")
        supplied = tuple(rows)
        if any(not isinstance(row, LoweringCoverageRow) for row in supplied):
            raise TypeError("LoweringCoverageReport rows must be LoweringCoverageRow values")
        canonical = tuple(sorted(supplied, key=lambda row: row.source))
        sources = [row.source for row in canonical]
        if len(set(sources)) != len(sources):
            duplicate = next(source for source in sources if sources.count(source) > 1)
            raise ValueError("duplicate lowering source %r" % duplicate)
        forward = {row.source: row.targets for row in canonical}
        reverse: dict[str, list[str]] = {}
        for row in canonical:
            for target in row.targets:
                reverse.setdefault(target, []).append(row.source)
        object.__setattr__(self, "rows", canonical)
        object.__setattr__(self, "source_to_targets", MappingProxyType(forward))
        object.__setattr__(self, "target_to_sources", MappingProxyType({
            target: tuple(sorted(source_ids)) for target, source_ids in sorted(reverse.items())
        }))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("LoweringCoverageReport is immutable")
        object.__setattr__(self, name, value)

    def __iter__(self) -> Any:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "rows": [row.to_data() for row in self.rows],
            "source_to_targets": {
                source: list(targets) for source, targets in self.source_to_targets.items()
            },
            "target_to_sources": {
                target: list(sources) for target, sources in self.target_to_sources.items()
            },
        }

    @classmethod
    def from_data(cls, data: Any) -> "LoweringCoverageReport":
        if not isinstance(data, Mapping) or set(data) != {
            "schema_version", "rows", "source_to_targets", "target_to_sources",
        }:
            raise TypeError("LoweringCoverageReport data has an invalid schema")
        if data["schema_version"] != 1:
            raise ValueError("unsupported LoweringCoverageReport schema_version %r"
                             % data["schema_version"])
        if not isinstance(data["rows"], list):
            raise TypeError("LoweringCoverageReport rows must be a list")
        report = cls(LoweringCoverageRow.from_data(row) for row in data["rows"])
        if report.to_data() != dict(data):
            raise ValueError("LoweringCoverageReport data is not in canonical order")
        return report


class LoweringRejection(ValueError):
    """A deterministic lowering gate failure carrying all coverage known so far."""

    def __init__(self, message: str, *, coverage_report: LoweringCoverageReport,
                 source: str, gate: str) -> None:
        super().__init__(message)
        if not isinstance(coverage_report, LoweringCoverageReport):
            raise TypeError("coverage_report must be a LoweringCoverageReport")
        self.coverage_report = coverage_report
        self.source = _identifier(source, "rejection source")
        self.gate = _identifier(gate, "rejection gate")

    @property
    def report(self) -> LoweringCoverageReport:
        return self.coverage_report


__all__ = ["LOWERING_DISPOSITIONS", "LoweringCoverageRow", "LoweringCoverageReport",
           "LoweringRejection"]
