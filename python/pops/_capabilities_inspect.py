"""Descriptor-catalog walk, native cross-check and AMR report (ADC-619 split).

The introspection side of the capability layer: the descriptor-catalog walk
(``_walk_brick_catalog`` / ``_walk_class_catalog`` / ``_entry_from_brick``), the
native cross-check (``_cross_check`` / :class:`CapabilityMismatchError` /
``_native_rows``) that adjudicates descriptor availability against the C++ facts,
the internal descriptor-catalog report, and the AMR hierarchy report
(:class:`AmrReport` / the private layout-adaptivity protocol / ``_amr_policy_rows``). Split out of
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
    # ADC-625: availability is the explained route; read it once through available().ok.
    ok = descriptor.available().ok
    status = "yes" if ok else "no"
    feature = "%s:%s" % (descriptor.category, descriptor.name)
    limitation = "" if ok else "catalogued descriptor has no native C++ symbol"
    error = "" if ok else _unsupported_error(
        requested=feature,
        available="native %s descriptors with a non-empty native_id" % descriptor.category,
        alternative="choose a typed descriptor from its pops.lib catalog and inspect it with pops.inspect()")
    return CapabilityEntry(descriptor.name, descriptor.category,
                           descriptor.native_id or None, status, descriptor.requirements,
                           feature=feature, backend="native" if descriptor.native_id else "none",
                           status=_route_status_from_availability(status),
                           limitation=limitation, error_message=error)


def _walk_brick_catalog(namespace):
    """Yield brick-catalog entries from a SimpleNamespace of zero-arg descriptor factories.

    A factory that requires an argument (e.g. ``User(brick_id)``) is skipped: it names a slot
    that is only realisable with user input, not a standing catalog entry. A Krylov solver factory
    that now requires a mandatory ``max_iter`` (ADC-535) IS a standing catalog entry, though -- the
    budget is per-use and does NOT change the entry identity (native_id / category / capabilities),
    so it is built with a nominal budget for the listing rather than dropped.
    """
    for attr_name in sorted(vars(namespace)):
        factory = getattr(namespace, attr_name)
        if not callable(factory):
            continue
        try:
            descriptor = factory()
        except TypeError:
            continue  # needs an argument (User selectors); not a standing entry.
        except ValueError:
            # A mandatory-budget solver (ADC-535): the entry identity is budget-independent, so a
            # nominal max_iter lets it appear in the report. Still skipped if it needs more.
            try:
                descriptor = factory(max_iter=1)
            except (TypeError, ValueError):
                continue
        if hasattr(descriptor, "brick_type"):  # a BrickDescriptor
            yield _entry_from_brick(descriptor)


def _external_brick_rows():
    """CapabilityEntry rows for the registered EXTERNAL C++ bricks (ADC-611 / ADC-544): one
    ``source="external"`` row per brick registered via ``pops.load_cpp_library`` /
    ``pops.external.register`` (the in-process catalog ``pops.descriptors._EXTERNAL_BRICKS``). Empty
    when none are registered. This surfaces external bricks in the internal catalog report so a
    third-party brick loaded at runtime appears in the capability report instead of being invisible.

    ADC-544 enriches the row with the brick's declared route surface: ``supported_layouts`` becomes the
    row ``layout`` (``uniform|amr`` when several, ``context`` when unconstrained), ``supported_platforms``
    the ``platform`` (with the mpi/gpu flags derived from it), and the ``native_id`` (distinct from the
    selector id) the row native id. The ``limitation`` names the declared layouts / platforms so a reader
    sees WHY a brick would be unavailable under an unlisted route, not just that it exists."""
    from pops.descriptors import _EXTERNAL_BRICKS
    rows = []
    for brick_id in sorted(_EXTERNAL_BRICKS):
        record = _EXTERNAL_BRICKS[brick_id]
        category = record.get("category", "brick")
        reqs = {"capabilities": list(record.get("requirements", []))} if record.get("requirements") \
            else {}
        layouts = list(record.get("supported_layouts", []))
        platforms = list(record.get("supported_platforms", []))
        native_id = record.get("native_id") or brick_id
        layout = "|".join(sorted(layouts)) if layouts else "context"
        platform = "|".join(sorted(platforms)) if platforms else "context"
        limitation = ""
        if layouts or platforms:
            parts = []
            if layouts:
                parts.append("layouts=%s" % ",".join(sorted(layouts)))
            if platforms:
                parts.append("platforms=%s" % ",".join(sorted(platforms)))
            limitation = "declared route surface: %s" % "; ".join(parts)
        rows.append(CapabilityEntry(
            brick_id, category, native_id, "yes", reqs, source="external",
            feature="%s:%s" % (category, brick_id), backend="external_cpp",
            layout=layout, platform=platform,
            mpi=("mpi" in platforms) or None, gpu=("gpu" in platforms) or None,
            status="available", limitation=limitation))
    return rows


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

    Raised by :func:`_descriptor_catalog_report` when native ``_pops.module_capabilities()`` reports a
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


def _descriptor_catalog_report():
    """Build the internal descriptor catalog, cross-checked against native facts.

    Walks the available descriptor catalogs and reports one row per catalogued entry (``source =
    "descriptor"``), THEN cross-checks the route-deciding entries against the authoritative C++
    ``_pops.module_capabilities()`` facts and APPENDS those as ``source="native"`` rows (Spec 5
    sec.13.12, #36): the capability VALUES now come from the C++ core, not a Python walk. A descriptor
    that declares itself available while the C++ source reports the transport unavailable raises a
    :class:`CapabilityMismatchError` (closing the silent-default-fallback gap).

    The descriptor walk is PURE (only the inert authoring packages, no numeric loop). ``_pops`` is
    imported LAZILY (inside this function) and ONLY for the cross-check, so the module import graph
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
    # External C++ bricks registered at runtime (ADC-611): surfaced so a loaded third-party brick
    # appears in the report. Empty when none registered -> bit-identical to before.
    entries.extend(_external_brick_rows())

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
    """The structured AMR section embedded by ``pops.inspect(layout)``.

    A plain record of an AMR hierarchy's declared metadata -- the level / ratio envelope, the
    regrid / patch / nesting / refinement policies, the runtime requirements (reflux, tag
    reduction), and the explainable route limitations. Checkpoint and output declarations live in
    the Case's ConsumerGraph and are deliberately absent. :meth:`to_dict` returns a plain nested
    dict and :meth:`__str__` a short, deterministic report. It is inert: it holds metadata read from
    the descriptors, it computes nothing.
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
        lines.append("  levels: max_levels=%s ratio=%s (native depth: %s; ratios=%s)" % (
            self.max_levels, self.ratio, self.native_max_levels,
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

    A deterministic, stable walk of the descriptor chain (refine / regrid / patches / nesting); a
    slot left as ``None`` on the layout is skipped. The refinement criterion is expanded into its
    sub-criteria when it is a ``TagUnion`` so the report names each tagged subject / predicate /
    threshold, not just the union count.
    """
    rows = []
    for slot in ("refine", "regrid", "patches", "nesting"):
        policy = getattr(layout, slot, None)
        if policy is None:
            continue
        rows.append((slot, policy.name, policy.options()))
        criteria = getattr(policy, "criteria", None)
        if criteria is not None:
            for sub in criteria:
                rows.append((slot + ".criterion", sub.name, sub.options()))
    return rows


def _native_amr_context():
    """Return the immutable native facts shared by layout-owned AMR reports."""
    from pops.mesh.amr import NATIVE_RATIOS

    native_depth = "resource_policy"
    native_note = (
        "resolved hierarchy depth is resource-policy controlled; native transfer ratios: %s"
        % ", ".join(map(str, NATIVE_RATIOS))
    )
    return native_depth, tuple(NATIVE_RATIOS), native_note


def _native_amr_envelope():
    """Build the runtime's descriptor-free native AMR capability envelope."""
    native_depth, native_ratios, native_note = _native_amr_context()
    return AmrReport(
        layout="native-envelope", max_levels=native_depth, ratio=native_ratios[0],
        native_max_levels=native_depth, native_ratios=native_ratios,
        available="yes", limitations=[native_note], requirements={}, policies=[])


def _layout_amr_report(layout):
    """Ask a layout for its AMR report through the open, branch-free layout protocol.

    This helper deliberately knows no concrete layout class. New layout kinds participate by
    implementing ``_amr_report()``; the public inspection path remains ``pops.inspect(layout)``.
    """
    provider = getattr(layout, "_amr_report", None)
    if not callable(provider):
        raise TypeError(
            "%s must implement the layout adaptivity protocol _amr_report()"
            % type(layout).__qualname__)
    report = provider()
    if not isinstance(report, AmrReport):
        raise TypeError(
            "%s._amr_report() must return AmrReport, got %s"
            % (type(layout).__qualname__, type(report).__qualname__))
    return report
