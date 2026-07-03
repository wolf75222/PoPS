"""pops._report -- the ONE typed-report base (ADC-564).

Every ``inspect()`` on an AGGREGATE surface (a Problem, a Program, a layout, a compiled artifact, a
bound simulation) returns a typed :class:`Report`: ATTRIBUTES the caller reads directly plus a
:meth:`to_dict` bridge for JSON. A :class:`Report` is NEVER a ``dict`` subclass -- the dict-emulation
crutch (``x[key]`` / iteration) cannot creep back; :meth:`to_dict` is the ONE mapping bridge, exactly
as the descriptor-side result objects (:mod:`pops.descriptors_report`) already are.

The base factors the shared boilerplate the hand-rolled reports each re-implemented: ``report_type``
/ ``schema_version`` identity, :meth:`to_json` over :meth:`to_dict`, and a ``__str__`` pretty
printer. A subclass sets ``report_type`` / ``schema_version`` and defines its attributes + a
:meth:`to_dict`; it inherits the rest. A report is INERT -- building one runs no numeric loop, opens
no extension and triggers no validation / compilation (it reads carried metadata only).
"""
import json


class Report:
    """Base of the typed inspection reports (ADC-564). Attributes + :meth:`to_dict`; NOT a dict.

    A subclass sets :attr:`report_type` (a short stable string) and :attr:`schema_version` (an int,
    bumped only on a breaking shape change), declares its fields as attributes, and defines
    :meth:`to_dict` returning a JSON-ready dict (stamped with the type + version via
    :meth:`_stamp`). It inherits :meth:`to_json` and a default ``__str__``. The report never
    subclasses ``dict``; :meth:`to_dict` is the ONE mapping bridge.
    """

    #: A short, stable identifier of the report kind (overridden per subclass).
    report_type = "report"
    #: The report's schema version; bump only on a breaking shape change (additive keeps it).
    schema_version = 1

    def _stamp(self, payload):
        """Prepend the ``report_type`` / ``schema_version`` identity to a subclass ``to_dict`` body."""
        stamped = {"report_type": self.report_type, "schema_version": self.schema_version}
        stamped.update(payload)
        return stamped

    def to_dict(self):  # pragma: no cover - overridden by every concrete report
        """A JSON-ready dict view (subclasses override; must call :meth:`_stamp`)."""
        return self._stamp({})

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to ``path`` if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    def __str__(self):
        """A short, deterministic pretty print (a subclass may override for a richer table)."""
        lines = ["%s (schema v%d):" % (self.report_type, self.schema_version)]
        for key, value in self.to_dict().items():
            if key in ("report_type", "schema_version"):
                continue
            lines.append("  %-14s %s" % (key, _short(value)))
        return "\n".join(lines)

    def __repr__(self):
        return "%s(report_type=%r)" % (type(self).__name__, self.report_type)


def _short(value):
    """A compact, single-line rendering of a value for the ``__str__`` pretty printer."""
    if isinstance(value, dict):
        return "{%d key(s)}" % len(value)
    if isinstance(value, (list, tuple)):
        return "[%d item(s)]" % len(value)
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


__all__ = ["Report"]
