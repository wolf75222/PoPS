"""pops._capabilities -- the descriptor-sourced capability matrix (Spec 5 sec.6 / sec.13.12.1).

:func:`inspect_capabilities` walks the inert descriptor catalogs (the Riemann / reconstruction
/ limiter / projection bricks, the mesh layouts, the solver / field catalogs) and reports, per
entry, its name / category / native id / availability / requirements. It is PURE: it imports
only the pure-stdlib authoring packages, never ``_pops``, and runs nothing -- it instantiates
each catalogued descriptor and reads its declared metadata.

This is the introspectable counterpart of the hand-written ``pops.capabilities()`` (the runtime
doctor's dispatch table): that one mirrors what the compiled runtime can dispatch, this one is
sourced straight from the typed descriptors, so the two cannot silently disagree about which
bricks exist.
"""


class CapabilityEntry:
    """One row of the capability matrix: a catalogued descriptor's declared metadata.

    A plain value -- name / category / native_id / available (an ``Availability`` status string)
    / requirements -- read from an inert descriptor. It computes nothing.
    """

    def __init__(self, name, category, native_id, available, requirements):
        self.name = name
        self.category = category
        self.native_id = native_id
        self.available = available
        self.requirements = dict(requirements or {})

    def to_dict(self):
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "available": self.available, "requirements": self.requirements}

    def __repr__(self):
        return ("CapabilityEntry(name=%r, category=%r, native_id=%r, available=%r)"
                % (self.name, self.category, self.native_id, self.available))


class CapabilityMatrix:
    """The structured, printable result of :func:`inspect_capabilities`.

    Holds the :class:`CapabilityEntry` rows grouped by category; :meth:`to_dict` returns a
    plain nested dict and :meth:`__str__` a short, deterministic table. It is inert.
    """

    def __init__(self, entries):
        self.entries = list(entries)

    def categories(self):
        return sorted({e.category for e in self.entries})

    def by_category(self, category):
        return [e for e in self.entries if e.category == category]

    def to_dict(self):
        out = {}
        for entry in self.entries:
            out.setdefault(entry.category, []).append(entry.to_dict())
        return out

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return "CapabilityMatrix(%d entries, %d categories)" % (
            len(self.entries), len(self.categories()))

    def __str__(self):
        lines = ["capability matrix (%d entries):" % len(self.entries)]
        for category in self.categories():
            lines.append("  [%s]" % category)
            for entry in self.by_category(category):
                native = entry.native_id or "-"
                lines.append("    %-18s available=%-7s native_id=%s"
                             % (entry.name, entry.available, native))
        return "\n".join(lines)


def _availability_status(descriptor):
    """The Availability status string of a descriptor (always defined; no context needed)."""
    try:
        return descriptor.available().status
    except Exception:  # a descriptor whose availability needs a context is reported as unknown.
        return "unknown"


def _entry_from_brick(descriptor):
    """A :class:`CapabilityEntry` from a :class:`pops.descriptors.BrickDescriptor`."""
    status = "yes" if descriptor.available else "no"
    return CapabilityEntry(descriptor.name, descriptor.category,
                           descriptor.native_id or None, status, descriptor.requirements)


def _walk_brick_catalog(namespace):
    """Yield brick-catalog entries from a SimpleNamespace of zero-arg descriptor factories.

    A factory that requires an argument (e.g. ``User(brick_id)``) is skipped: it names a slot
    that is only realisable with user input, not a standing catalog entry.
    """
    for attr_name in sorted(vars(namespace)):
        factory = getattr(namespace, attr_name)
        if not callable(factory):
            continue
        try:
            descriptor = factory()
        except TypeError:
            continue  # needs an argument (User selectors); not a standing entry.
        if hasattr(descriptor, "brick_type"):  # a BrickDescriptor
            yield _entry_from_brick(descriptor)


def _walk_class_catalog(category, classes):
    """Yield entries for descriptor CLASSES that need constructor args (e.g. mesh layouts).

    These cannot be instantiated without a mesh; we report the slot from the class itself
    (name / category) with an ``unknown`` availability that depends on the route context.
    """
    for cls in classes:
        native = getattr(cls, "native_id", None)
        yield CapabilityEntry(cls.__name__, category, native, "context", {})


def inspect_capabilities():
    """Return the descriptor-sourced :class:`CapabilityMatrix` (Spec 5 sec.6).

    Walks the available descriptor catalogs and reports one row per catalogued entry. PURE: it
    imports only the inert authoring packages and instantiates each descriptor to read its
    declared metadata; it NEVER imports ``_pops`` or runs a numeric loop.
    """
    from pops.numerics.riemann import riemann
    from pops.numerics.reconstruction import reconstruction
    from pops.numerics.reconstruction.limiters import limiters
    from pops.numerics.projections import projections
    from pops.mesh.layouts import Uniform, AMR

    entries = []
    for namespace in (riemann, reconstruction, limiters, projections):
        entries.extend(_walk_brick_catalog(namespace))
    entries.extend(_walk_class_catalog("layout", (Uniform, AMR)))

    # The solver / field catalogs live under pops.lib when present (optional layer).
    try:
        from pops.lib.solvers import solvers
        entries.extend(_walk_brick_catalog(solvers))
    except ImportError:
        pass
    try:
        from pops.lib.fields import fields
        entries.extend(_walk_brick_catalog(fields))
    except ImportError:
        pass

    return CapabilityMatrix(entries)


__all__ = ["inspect_capabilities", "CapabilityMatrix", "CapabilityEntry"]
