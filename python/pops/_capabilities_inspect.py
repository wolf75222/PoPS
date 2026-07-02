"""Descriptor-catalog walk, native cross-check and AMR report (ADC-619 split).

The introspection side of the capability layer: the descriptor-catalog walk
(``_walk_brick_catalog`` / ``_walk_class_catalog`` / ``_entry_from_brick``), the
native cross-check (``_cross_check`` / :class:`CapabilityMismatchError` /
``_native_rows``) that adjudicates descriptor availability against the C++ facts,
the public :func:`inspect_capabilities`, and the AMR hierarchy report
(:class:`AmrReport` / :func:`inspect_amr` / ``_amr_policy_rows``). Split out of
``_capabilities`` for the 500-line cap; ``pops._capabilities`` re-exports every
public name. The descriptor walk is PURE; ``_pops`` is only reached lazily through
``_module_capabilities`` for the optional cross-check.
"""

from pops._capabilities_common import (
    CapabilityEntry,
    CapabilityMatrix,
    _route_status_from_availability,
    _unsupported_error,
)
from pops._capabilities_report import (
    _feature_backend,
    _feature_layout,
    _feature_platform,
    _flag_error_message,
    _module_capabilities,
)


def _entry_from_brick(descriptor):
    """A :class:`CapabilityEntry` from a :class:`pops.descriptors.BrickDescriptor`."""
    status = "yes" if descriptor.available else "no"
    feature = "%s:%s" % (descriptor.category, descriptor.name)
    limitation = "" if descriptor.available else "catalogued descriptor has no native C++ symbol"
    error = "" if descriptor.available else _unsupported_error(
        requested=feature,
        available="native %s descriptors with a non-empty native_id" % descriptor.category,
        alternative="choose an available descriptor from pops.inspect_capabilities()")
    return CapabilityEntry(descriptor.name, descriptor.category,
                           descriptor.native_id or None, status, descriptor.requirements,
                           feature=feature, backend="native" if descriptor.native_id else "none",
                           status=_route_status_from_availability(status),
                           limitation=limitation, error_message=error)


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
        yield CapabilityEntry(cls.__name__, category, native, "context", {},
                              feature="%s:%s" % (category, cls.__name__),
                              layout=cls.__name__.lower(), status="unknown",
                              limitation="availability depends on the requested route context")


# Spec 5 sec.13.12: the descriptor layout entries whose availability the C++ source ADJUDICATES.
# A native ``supports_uniform`` / ``supports_amr`` of ``False`` would make a layout descriptor that
# reports itself available a SILENT lie; the cross-check forbids that. Keyed by descriptor name.
_LAYOUT_NATIVE_FLAG = {"Uniform": "supports_uniform", "AMR": "supports_amr"}


class CapabilityMismatchError(RuntimeError):
    """A descriptor's declared availability disagrees with the C++ authoritative source (#36/#37).

    Raised by :func:`inspect_capabilities` when the native ``_pops.module_capabilities()`` reports a
    transport as UNAVAILABLE while the Python descriptor catalog still advertises it available. It
    closes the Spec 5 sec.13.12 "Python-derived, not authoritative" gap: a descriptor can no longer
    silently claim a capability the built module does not provide.
    """


def _native_rows(native_caps):
    """Native-sourced :class:`CapabilityEntry` rows from the C++ ``module_capabilities()`` dict.

    One ``transport`` row per ``supports_*`` flag, ``source="native"``, ``available`` = ``"yes"`` /
    ``"no"`` straight from the C++ boolean (no Python computation). These are the AUTHORITATIVE facts
    the descriptor walk is checked against.
    """
    rows = []
    for key in sorted(native_caps):
        if not key.startswith("supports_"):
            continue
        status = "yes" if native_caps[key] else "no"
        rows.append(CapabilityEntry(
            key, "transport", None, status, {}, source="native", feature=key,
            layout=_feature_layout(key), backend=_feature_backend(key),
            platform=_feature_platform(key), mpi=(key == "supports_mpi" and native_caps[key]),
            gpu=(key == "supports_gpu" and native_caps[key]),
            status=_route_status_from_availability(status),
            limitation="" if native_caps[key] else "%s is not provided by this build" % key,
            error_message="" if native_caps[key] else _flag_error_message(key)))
    return rows


def _cross_check(entries, native_caps):
    """Raise :class:`CapabilityMismatchError` if a descriptor disagrees with the C++ source (#36).

    For each descriptor whose capability the native facts adjudicate (the layout entries, via
    :data:`_LAYOUT_NATIVE_FLAG`), a descriptor that reports itself available while the C++ flag is
    ``False`` is a silent lie -- FAIL LOUD. A descriptor reported unavailable / context-dependent is
    never escalated (the no-false-positive discipline): the C++ source can only DEMOTE, never promote.
    """
    for entry in entries:
        flag = _LAYOUT_NATIVE_FLAG.get(entry.name)
        if flag is None or flag not in native_caps:
            continue
        descriptor_claims_available = entry.available in ("yes", "context")
        if descriptor_claims_available and not native_caps[flag]:
            raise CapabilityMismatchError(
                "descriptor %r (category %r) reports available=%r but the C++ source "
                "%s=False; the built module does not provide it (Spec 5 sec.13.12)"
                % (entry.name, entry.category, entry.available, flag))


def inspect_capabilities():
    """Return the capability :class:`CapabilityMatrix`, cross-checked against the C++ source (sec.6).

    Walks the available descriptor catalogs and reports one row per catalogued entry (``source =
    "descriptor"``), THEN cross-checks the route-deciding entries against the authoritative C++
    ``_pops.module_capabilities()`` facts and APPENDS those as ``source="native"`` rows (Spec 5
    sec.13.12, #36): the capability VALUES now come from the C++ core, not a Python walk. A descriptor
    that declares itself available while the C++ source reports the transport unavailable raises a
    :class:`CapabilityMismatchError` (closing the silent-default-fallback gap).

    The descriptor walk is PURE (only the inert authoring packages, no numeric loop). ``_pops`` is
    imported LAZILY (inside the function) and ONLY for the cross-check, so the module import graph
    stays acyclic; when ``_pops`` is absent or predates ``module_capabilities`` the walk proceeds
    WITHOUT the native rows / cross-check rather than failing (graceful degradation).
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

    # The solver / field brick catalogs (Spec 5 criterion 7: solvers under pops.solvers, the
    # field brick catalog under pops.fields.catalog) -- optional layers walked when present.
    try:
        from pops.solvers import solvers
        entries.extend(_walk_brick_catalog(solvers))
    except ImportError:
        pass
    try:
        from pops.fields import catalog as fields
        entries.extend(_walk_brick_catalog(fields))
    except ImportError:
        pass

    native_caps = _module_capabilities()
    if native_caps is not None:
        _cross_check(entries, native_caps)
        entries.extend(_native_rows(native_caps))

    return CapabilityMatrix(entries)


class AmrReport:
    """The structured, printable result of :func:`inspect_amr` (Spec 5 sec.5.11 / sec.8).

    A plain record of an AMR hierarchy's declared metadata -- the level / ratio envelope, the
    regrid / patch / nesting / refinement / checkpoint / output policies, the runtime
    requirements (reflux, tag reduction), and the explainable route limitations. :meth:`to_dict`
    returns a plain nested dict and :meth:`__str__` a short, deterministic report. It is inert:
    it holds metadata read from the descriptors, it computes nothing.
    """

    def __init__(self, *, layout, max_levels, ratio, native_max_levels, native_ratios,
                 available, limitations, requirements, policies):
        self.layout = layout
        self.max_levels = max_levels
        self.ratio = ratio
        self.native_max_levels = native_max_levels
        self.native_ratios = tuple(native_ratios)
        self.available = available
        self.limitations = list(limitations)
        self.requirements = dict(requirements or {})
        # policies: ordered list of (slot, name, options-dict) for the attached policies.
        self.policies = list(policies)

    def to_dict(self):
        return {
            "layout": self.layout,
            "max_levels": self.max_levels,
            "ratio": self.ratio,
            "native_max_levels": self.native_max_levels,
            "native_ratios": list(self.native_ratios),
            "available": self.available,
            "limitations": list(self.limitations),
            "requirements": dict(self.requirements),
            "policies": [{"slot": slot, "name": name, "options": dict(options)}
                         for slot, name, options in self.policies],
        }

    def __repr__(self):
        return ("AmrReport(layout=%r, max_levels=%r, ratio=%r, available=%r)"
                % (self.layout, self.max_levels, self.ratio, self.available))

    def __str__(self):
        lines = ["AMR hierarchy report (%s):" % self.layout]
        lines.append("  levels: max_levels=%s ratio=%s (native envelope: max_levels<=%s, "
                     "ratios=%s)" % (self.max_levels, self.ratio, self.native_max_levels,
                                     ", ".join(map(str, self.native_ratios))))
        lines.append("  available: %s" % self.available)
        if self.requirements:
            req = ", ".join("%s=%s" % (k, v) for k, v in sorted(self.requirements.items()))
            lines.append("  requirements: %s" % req)
        if self.policies:
            lines.append("  policies:")
            for slot, name, options in self.policies:
                body = ", ".join("%s=%r" % (k, v) for k, v in options.items())
                lines.append("    %-11s %s(%s)" % (slot + ":", name, body))
        if self.limitations:
            lines.append("  limitations:")
            for note in self.limitations:
                lines.append("    - %s" % note)
        return "\n".join(lines)


def _amr_policy_rows(layout):
    """Ordered (slot, name, options) rows for the policies attached to an AMR layout.

    A deterministic, stable walk of the descriptor chain (refine / regrid / patches / nesting /
    checkpoint / output); a slot left as ``None`` on the layout is skipped. The refinement
    criterion is expanded into its sub-criteria when it is a ``TagUnion`` so the report names
    each tagged subject / predicate / threshold, not just the union count.
    """
    rows = []
    for slot in ("refine", "regrid", "patches", "nesting", "checkpoint", "output"):
        policy = getattr(layout, slot, None)
        if policy is None:
            continue
        rows.append((slot, policy.name, policy.options()))
        criteria = getattr(policy, "criteria", None)
        if criteria is not None:
            for sub in criteria:
                rows.append((slot + ".criterion", sub.name, sub.options()))
    return rows


def inspect_amr(layout_or_context=None):
    """Return a printable :class:`AmrReport` of an AMR hierarchy (Spec 5 sec.5.11 / sec.8).

    The introspectable counterpart of :func:`inspect_capabilities` for the adaptive-mesh
    route. PURE: it imports only the inert :mod:`pops.mesh` authoring descriptors and reads
    their declared metadata (levels / ratio, the regrid / patch / nesting / refine / checkpoint
    / output policies, the runtime requirements such as reflux / tag reduction, and the
    explainable route limitations); it NEVER imports ``_pops`` / the runtime / codegen and runs
    no numeric loop.

    Args:
        layout_or_context: an :class:`pops.mesh.layouts.AMR` (or :class:`Uniform`) descriptor to
            report on, or ``None`` to report the current native AMR envelope (the
            :data:`pops.mesh.amr.NATIVE_MAX_LEVELS` / ``NATIVE_RATIOS`` capability limits).
    """
    from pops.mesh.amr import NATIVE_MAX_LEVELS, NATIVE_RATIOS
    from pops.mesh.layouts import AMR, Uniform

    native_note = ("the current native AMR route supports max_levels<=%d at ratio %s; a request "
                   "beyond that is refused before the runtime, not silently clamped"
                   % (NATIVE_MAX_LEVELS, ", ".join(map(str, NATIVE_RATIOS))))

    if layout_or_context is None:
        return AmrReport(
            layout="native-envelope", max_levels=NATIVE_MAX_LEVELS, ratio=NATIVE_RATIOS[0],
            native_max_levels=NATIVE_MAX_LEVELS, native_ratios=NATIVE_RATIOS,
            available="yes", limitations=[native_note], requirements={}, policies=[])

    if isinstance(layout_or_context, Uniform):
        caps = layout_or_context.capabilities()
        return AmrReport(
            layout="uniform", max_levels=caps.get("levels", 1), ratio=1,
            native_max_levels=NATIVE_MAX_LEVELS, native_ratios=NATIVE_RATIOS,
            available="yes",
            limitations=["a Uniform layout is single-level: no refinement, regrid or reflux"],
            requirements={}, policies=[])

    if not isinstance(layout_or_context, AMR):
        raise TypeError(
            "inspect_amr expects a pops.mesh.layouts.AMR / Uniform descriptor (or None for the "
            "native envelope); got %r" % (type(layout_or_context).__name__,))

    layout = layout_or_context
    status = layout.available()
    limitations = [native_note]
    if not status.ok and status.reason:
        limitations.append(status.reason)
    return AmrReport(
        layout="amr", max_levels=layout.max_levels, ratio=layout.ratio,
        native_max_levels=NATIVE_MAX_LEVELS, native_ratios=NATIVE_RATIOS,
        available=status.status, limitations=limitations,
        requirements=layout.requirements(), policies=_amr_policy_rows(layout))
