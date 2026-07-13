"""pops._inspect -- the stable ``pops.inspect(obj)`` dispatcher (ADC-527).

``pops.inspect(obj)`` is the ONE stable, serialisable structured view of any descriptor / Problem /
report. It dispatches to ``obj.inspect()`` when the object exposes it (every DescriptorProtocol
member and the Problem assembly do); otherwise it builds the base dict from the protocol members it
can read. It runs NO numerics and touches no runtime -- it is metadata only. Layout inspection
embeds its adaptive hierarchy declaration and relevant native capability rows in the same result;
there is no competing layout-specific public inspector.
"""
from __future__ import annotations

from typing import Any

from pops._report import ReportTree


def inspect(obj: Any) -> Any:
    """Return a stable, serialisable ``dict`` view of @p obj (descriptor / Problem / report).

    The ONE explicit dict bridge (ADC-564): it dispatches to ``obj.inspect()`` and, when that returns
    a typed :class:`pops.Report` (``Problem.inspect()`` / ``Program.inspect()`` / a compiled or
    runtime report), returns its ``to_dict()`` -- so a caller wanting JSON gets a plain dict while
    structure-wanting callers read ``obj.inspect().<attr>``. A per-descriptor ``inspect()`` that
    already returns a dict passes through unchanged. Falls back to the protocol members / a
    ``to_dict()`` / repr view when the object exposes no ``inspect``. Never runs numerics.
    """
    own: Any = getattr(obj, "inspect", None)
    if callable(own):
        result: Any = own()
        to_dict = getattr(result, "to_dict", None)
        return to_dict() if callable(to_dict) else result
    to_dict: Any = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    record: dict = {}
    for member in ("name", "category", "native_id"):
        if hasattr(obj, member):
            record[member] = getattr(obj, member)
    for method in ("options", "requirements", "capabilities"):
        fn: Any = getattr(obj, method, None)
        if callable(fn):
            value: Any = fn()
            record[method] = dict(value) if hasattr(value, "keys") else value
    if not record:
        return {"repr": repr(obj)}
    return record


def explain(obj: Any) -> ReportTree:
    """Return the object's typed explanation without crossing the dict bridge.

    Objects with a domain-specific ``explain()`` own their explanation.  A ``ReportTree`` returned
    by ``inspect()`` is also accepted directly.  Legacy/domain inspection values are captured as
    detached JSON evidence under one generic inspection node; no live object is retained.
    """
    own_explain: Any = getattr(obj, "explain", None)
    if callable(own_explain):
        report = own_explain()
        if not isinstance(report, ReportTree):
            raise TypeError("%s.explain() must return ReportTree, got %s" % (
                type(obj).__qualname__, type(report).__qualname__))
        return report

    own_inspect: Any = getattr(obj, "inspect", None)
    if callable(own_inspect):
        inspected = own_inspect()
        if isinstance(inspected, ReportTree):
            return inspected
        to_dict: Any = getattr(inspected, "to_dict", None)
        payload = to_dict() if callable(to_dict) else inspected
    else:
        payload = inspect(obj)

    owner = getattr(obj, "owner_path", None)
    if owner is None:
        owner = getattr(obj, "name", None)
    source = "%s.%s" % (type(obj).__module__, type(obj).__qualname__)
    return ReportTree(
        phase="inspection", severity="info", code="inspection.object.summary",
        message="structured inspection of %s" % type(obj).__qualname__, source=source,
        owner=owner, evidence={"inspection": payload},
    )


__all__ = ["explain", "inspect"]
