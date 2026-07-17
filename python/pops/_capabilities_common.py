"""Capability value objects and shared route primitives (ADC-619 split).

The inert building blocks shared across the capability layer: the message /
status helpers (``_unsupported_error``, ``_route_status_from_availability``,
``_availability_status``, ``_axis_for_route``, ``_flag_value``,
``_status_from_flag``) and the four printable value objects
(:class:`CapabilityEntry`, :class:`CapabilityMatrix`, :class:`CapabilityRouteRow`,
:class:`CapabilityRouteMatrix`). Split out of ``_capabilities`` for the 500-line
cap; ``pops._capabilities`` re-exports every name here so the public import paths
stay unchanged. Everything in this module is PURE: it never imports ``_pops``.
"""
from __future__ import annotations

from typing import Any


def _availability_status(descriptor: Any) -> str:
    """The Availability status string of a descriptor (always defined; no context needed)."""
    try:
        return descriptor.available().status
    except Exception:  # a descriptor whose availability needs a context is reported as unknown.
        return "unknown"


def _route_status_from_availability(available: Any) -> str:
    """Map the legacy availability token to the route-matrix status vocabulary."""
    if available == "yes":
        return "available"
    if available == "no":
        return "unavailable"
    if available == "partial":
        return "partial"
    return "unknown"


def _unsupported_error(*, requested: Any, available: Any, alternative: Any = None) -> str:
    """Uniform ADC-549 unsupported-route message fragment."""
    msg = "unsupported route: requested %s; available route: %s" % (requested, available)
    if alternative:
        msg += "; alternative: %s" % alternative
    return msg


def _axis_for_route(layout: Any, backend: Any, platform: Any) -> str:
    if layout not in ("any", "uniform|amr", "context"):
        return "layout"
    if platform in ("mpi", "gpu"):
        return "backend"
    if backend not in ("any", "module", "context"):
        return "backend"
    return "transport"


def _flag_value(flags: Any, name: str) -> Any:
    if flags is None:
        return None
    return flags.get(name)


def _status_from_flag(flags: Any, name: str) -> str:
    value = _flag_value(flags, name)
    if value is None:
        return "unknown"
    return "available" if bool(value) else "unavailable"


class CapabilityEntry:
    """One row of the capability matrix: a catalogued descriptor's declared metadata.

    A plain value -- name / category / native_id / available (an ``Availability`` status string)
    / requirements / source -- read from an inert descriptor. It computes nothing. ``source`` is
    ``"descriptor"`` for a row read from the Python catalog and ``"native"`` for a row sourced
    from the C++ ``_pops.module_capabilities()`` authoritative facts (Spec 5 sec.13.12).
    """

    def __init__(self, name: str, category: str, native_id: Any, available: Any,
                 requirements: Any, source: str = "descriptor", *, feature: Any = None,
                 layout: str = "context", backend: str = "context", platform: str = "context",
                 mpi: Any = None, gpu: Any = None, status: Any = None, limitation: str = "",
                 error_message: str = "") -> None:
        self.name = name
        self.category = category
        self.native_id = native_id
        self.available = available
        self.requirements = dict(requirements or {})
        self.source = source
        # ADC-549 route-matrix columns. Descriptor-sourced rows keep the old identity fields above
        # and add a route view so tooling can inspect unsupported routes without prose scraping.
        self.feature = feature or ("%s:%s" % (category, name))
        self.layout = layout
        self.backend = backend
        self.platform = platform
        self.mpi = mpi
        self.gpu = gpu
        self.status = status or _route_status_from_availability(available)
        self.limitation = limitation
        self.error_message = error_message

    def to_dict(self) -> dict:
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "available": self.available, "requirements": self.requirements,
                "source": self.source, "feature": self.feature, "layout": self.layout,
                "backend": self.backend, "platform": self.platform, "mpi": self.mpi,
                "gpu": self.gpu, "status": self.status, "limitation": self.limitation,
                "error_message": self.error_message}

    def __repr__(self) -> str:
        return ("CapabilityEntry(name=%r, category=%r, native_id=%r, available=%r, source=%r)"
                % (self.name, self.category, self.native_id, self.available, self.source))


class CapabilityMatrix:
    """Internal structured descriptor-catalog report.

    Holds the :class:`CapabilityEntry` rows grouped by category; :meth:`to_dict` returns a
    plain nested dict and :meth:`__str__` a short, deterministic table. It is inert.
    """

    def __init__(self, entries: Any) -> None:
        self.entries = list(entries)

    def categories(self) -> list:
        return sorted({e.category for e in self.entries})

    def by_category(self, category: str) -> list:
        return [e for e in self.entries if e.category == category]

    def to_dict(self) -> dict:
        out: dict = {}
        for entry in self.entries:
            out.setdefault(entry.category, []).append(entry.to_dict())
        return out

    def __iter__(self) -> Any:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return "CapabilityMatrix(%d entries, %d categories)" % (
            len(self.entries), len(self.categories()))

    def __str__(self) -> str:
        lines = ["capability matrix (%d entries):" % len(self.entries)]
        for category in self.categories():
            lines.append("  [%s]" % category)
            for entry in self.by_category(category):
                native = entry.native_id or "-"
                lines.append("    %-18s available=%-7s source=%-10s native_id=%s"
                             % (entry.name, entry.available, entry.source, native))
        return "\n".join(lines)


class CapabilityRouteRow:
    """One ADC-549 route row.

    The row shape is intentionally flat and JSON-ready:
    ``feature, layout, backend, platform, mpi, gpu, status, limitation, error_message``.
    ``axis`` and ``source`` are kept for compatibility with the earlier ``Case.explain_routes``
    route matrix tests.
    """

    def __init__(self, feature: str, *, layout: str = "any", backend: str = "any",
                 platform: str = "host", mpi: Any = False, gpu: Any = False,
                 status: str = "unknown", limitation: str = "", error_message: str = "",
                 source: str = "native", axis: Any = None, available_route: str = "",
                 alternative: str = "") -> None:
        self.feature = feature
        self.layout = layout
        self.backend = backend
        self.platform = platform
        self.mpi = mpi
        self.gpu = gpu
        self.status = status
        self.limitation = limitation
        self.error_message = error_message
        self.source = source
        self.axis = axis or _axis_for_route(layout, backend, platform)
        self.available_route = available_route
        self.alternative = alternative

    def to_dict(self) -> dict:
        return {
            "route_id": self.feature,
            "feature": self.feature,
            "layout": self.layout,
            "backend": self.backend,
            "platform": self.platform,
            "mpi": self.mpi,
            "gpu": self.gpu,
            "status": self.status,
            "limitation": self.limitation,
            "reason": self.limitation,
            "error_message": self.error_message,
            "source": self.source,
            "axis": self.axis,
            "available_route": self.available_route,
            "alternative": self.alternative,
        }

    def __repr__(self) -> str:
        return ("CapabilityRouteRow(feature=%r, layout=%r, backend=%r, status=%r, source=%r)"
                % (self.feature, self.layout, self.backend, self.status, self.source))


class CapabilityRouteMatrix:
    """Printable ADC-549 matrix of feature x layout/backend/platform support."""

    def __init__(self, owner: Any, layout: Any, rows: Any, *, schema_version: Any = None,
                 abi_version: Any = None, target: Any = None, abi_key: Any = None,
                 platform: Any = None) -> None:
        self.owner = owner
        self.case_name = owner  # compatibility with the old Case route matrix object.
        self.layout = layout
        self.layout_name = layout
        self.rows = list(rows)
        self.schema_version = schema_version
        self.abi_version = abi_version
        self.target = target
        self.abi_key = abi_key
        self.platform = platform

    def to_dict(self) -> dict:
        return {"case": self.owner, "owner": self.owner, "layout": self.layout,
                "schema_version": self.schema_version, "abi_version": self.abi_version,
                "target": self.target, "abi_key": self.abi_key, "platform": self.platform,
                "rows": [r.to_dict() for r in self.rows]}

    def __iter__(self) -> Any:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return "CapabilityRouteMatrix(owner=%r, layout=%r, %d rows)" % (
            self.owner, self.layout, len(self.rows))

    def __str__(self) -> str:
        lines = ["route matrix for %r (layout=%s, ADC-549):" % (self.owner, self.layout)]
        for row in self.rows:
            note = ("  -- %s" % row.limitation) if row.limitation else ""
            lines.append(
                "  %-30s layout=%-12s backend=%-11s platform=%-5s mpi=%-5s gpu=%-5s %-11s%s"
                % (row.feature, row.layout, row.backend, row.platform, row.mpi, row.gpu,
                   row.status, note))
        return "\n".join(lines)
