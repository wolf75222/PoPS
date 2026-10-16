"""One invocation-local set of diagnostic tables; never a runtime authority cache."""

from __future__ import annotations

_NAMES = frozenset({
    "program_accepted_state_manifest",
    "program_clock_manifest",
    "program_temporal_partition_manifest",
    "program_flux_ledger_manifest",
    "program_interface_flux_ledger_manifest",
    "program_sync_manifest",
})


def program_manifest_reader(sim, *, fallback=None):
    """Read one newly authenticated native image, or the legacy accessor protocol.

    A native validation failure propagates. Only absence of the batch operation
    selects the legacy protocol; no result survives this invocation.
    """
    try:
        method = sim.program_manifest_snapshot
    except AttributeError:
        # A present descriptor failing its lookup is not a missing capability.
        # Static inspection does not evaluate that descriptor a second time.
        from inspect import getattr_static
        missing = object()
        if getattr_static(sim, "program_manifest_snapshot", missing) is not missing:
            raise
        return fallback if fallback is not None else lambda name: getattr(sim, name)()
    if not callable(method):
        raise TypeError("native AMR Program manifest snapshot is not callable")
    snapshot = method()
    if (type(snapshot) is not dict or any(type(name) is not str for name in snapshot)
            or snapshot.keys() != _NAMES):
        raise ValueError("native AMR Program manifest snapshot has invalid tables")
    tables = {}
    for name, rows in snapshot.items():
        if type(rows) is not list or any(
            type(row) is not list or any(type(item) is not str for item in row)
            for row in rows
        ):
            raise ValueError("native AMR Program manifest snapshot has an invalid row")
        tables[name] = tuple(tuple(row) for row in rows)
    return tables.__getitem__
