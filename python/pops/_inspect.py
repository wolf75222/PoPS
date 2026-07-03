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

    Dispatches to ``obj.inspect()`` when present; else assembles the base record from the protocol
    members (``name`` / ``category`` / ``native_id`` / ``options`` / ``requirements`` /
    ``capabilities``) it can read, or falls back to a ``to_dict()`` / repr view. Never runs numerics.
    """
    own = getattr(obj, "inspect", None)
    if callable(own):
        return own()
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
