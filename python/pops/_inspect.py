"""pops._inspect -- the stable ``pops.inspect(obj)`` dispatcher (ADC-527).

``pops.inspect(obj)`` is the ONE stable, serialisable structured view of any descriptor / Problem /
report. It dispatches to ``obj.inspect()`` when the object exposes it (every DescriptorProtocol
member and the Problem assembly do); otherwise it builds the base dict from the protocol members it
can read. It runs NO numerics and touches no runtime -- it is metadata only. It is DISTINCT from
``pops.inspect_capabilities()`` / ``pops.inspect_amr()`` (those build the native capability matrix
from the C++ core); this one is the per-object introspection entry point.
"""


def inspect(obj):
    """Return a stable, serialisable ``dict`` view of @p obj (descriptor / Problem / report).

    The ONE explicit dict bridge (ADC-564): it dispatches to ``obj.inspect()`` and, when that returns
    a typed :class:`pops.Report` (``Problem.inspect()`` / ``Program.inspect()`` / a compiled or
    runtime report), returns its ``to_dict()`` -- so a caller wanting JSON gets a plain dict while
    structure-wanting callers read ``obj.inspect().<attr>``. A per-descriptor ``inspect()`` that
    already returns a dict passes through unchanged. Falls back to the protocol members / a
    ``to_dict()`` / repr view when the object exposes no ``inspect``. Never runs numerics.
    """
    own = getattr(obj, "inspect", None)
    if callable(own):
        result = own()
        to_dict = getattr(result, "to_dict", None)
        return to_dict() if callable(to_dict) else result
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    record = {}
    for member in ("name", "category", "native_id"):
        if hasattr(obj, member):
            record[member] = getattr(obj, member)
    for method in ("options", "requirements", "capabilities"):
        fn = getattr(obj, method, None)
        if callable(fn):
            value = fn()
            record[method] = dict(value) if hasattr(value, "keys") else value
    if not record:
        return {"repr": repr(obj)}
    return record


__all__ = ["inspect"]
