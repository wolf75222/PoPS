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
    / requirements / source -- read from an inert descriptor. It computes nothing. ``source`` is
    ``"descriptor"`` for a row read from the Python catalog and ``"native"`` for a row sourced
    from the C++ ``_pops.module_capabilities()`` authoritative facts (Spec 5 sec.13.12).
    """

    def __init__(self, name, category, native_id, available, requirements, source="descriptor",
                 *, feature=None, layout="context", backend="context", platform="context",
                 mpi=None, gpu=None, status=None, limitation="", error_message=""):
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

    def to_dict(self):
        return {"name": self.name, "category": self.category, "native_id": self.native_id,
                "available": self.available, "requirements": self.requirements,
                "source": self.source, "feature": self.feature, "layout": self.layout,
                "backend": self.backend, "platform": self.platform, "mpi": self.mpi,
                "gpu": self.gpu, "status": self.status, "limitation": self.limitation,
                "error_message": self.error_message}

    def __repr__(self):
        return ("CapabilityEntry(name=%r, category=%r, native_id=%r, available=%r, source=%r)"
                % (self.name, self.category, self.native_id, self.available, self.source))


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
                lines.append("    %-18s available=%-7s source=%-10s native_id=%s"
                             % (entry.name, entry.available, entry.source, native))
        return "\n".join(lines)


def _availability_status(descriptor):
    """The Availability status string of a descriptor (always defined; no context needed)."""
    try:
        return descriptor.available().status
    except Exception:  # a descriptor whose availability needs a context is reported as unknown.
        return "unknown"


def _route_status_from_availability(available):
    """Map the legacy availability token to the route-matrix status vocabulary."""
    if available == "yes":
        return "available"
    if available == "no":
        return "unavailable"
    if available == "partial":
        return "partial"
    return "unknown"


def _unsupported_error(*, requested, available, alternative=None):
    """Uniform ADC-549 unsupported-route message fragment."""
    msg = "unsupported route: requested %s; available route: %s" % (requested, available)
    if alternative:
        msg += "; alternative: %s" % alternative
    return msg


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


def _module_capabilities(target="module"):
    """The C++ authoritative capability dict (``_pops.module_capabilities()``) or ``None``.

    Lazily imports ``_pops`` (top-level then ``pops._pops``, mirroring the codegen toolchain) so the
    module import graph stays acyclic and the catalog walk works with no compiled module present.
    Returns ``None`` when ``_pops`` is unavailable or predates ``module_capabilities`` (old build):
    the descriptor walk then proceeds WITHOUT the cross-check rather than failing -- a missing native
    source is a graceful degradation, never a fabricated capability.
    """
    try:
        import _pops as mod  # noqa: PLC0415 -- lazy: keeps this module's import graph acyclic
    except Exception:
        try:
            from pops import _pops as mod  # noqa: PLC0415
        except Exception:
            return None
    fn = getattr(mod, "module_capabilities", None)
    if fn is None:  # an _pops built before this work: no authoritative source to cross-check against.
        return None
    try:
        return dict(fn(target))
    except Exception:
        return None


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


def _feature_layout(feature):
    if feature == "supports_uniform":
        return "uniform"
    if feature == "supports_amr":
        return "amr"
    return "uniform|amr"


def _feature_backend(feature):
    if feature in ("supports_stride", "supports_amr"):
        return "production"
    return "module"


def _feature_platform(feature):
    if feature == "supports_mpi":
        return "mpi"
    if feature == "supports_gpu":
        return "gpu"
    return "host"


def _flag_error_message(feature):
    requests = {
        "supports_amr": ("layout=AMR", "layout=Uniform or backend='production' target='amr_system'",
                         "use layout=Uniform or compile with backend='production' target='amr_system'"),
        "supports_mpi": ("platform=MPI", "serial/OpenMP build", "rebuild _pops with POPS_USE_MPI=ON"),
        "supports_gpu": ("platform=GPU", "host CPU platform", "use KokkosOpenMP/KokkosSerial or a CUDA/HIP build"),
        "supports_stride": ("strided cell access", "backend='production'",
                            "compile with backend='production'"),
        "supports_partial_imex_mask": ("partial IMEX mask", "full source implicit / split routes",
                                       "use IMEX/IMEXRK/Split without partial masks"),
    }
    requested, available, alternative = requests.get(
        feature, (feature, "no route in this build", None))
    return _unsupported_error(requested=requested, available=available, alternative=alternative)


class CapabilityRouteRow:
    """One ADC-549 route row.

    The row shape is intentionally flat and JSON-ready:
    ``feature, layout, backend, platform, mpi, gpu, status, limitation, error_message``.
    ``axis`` and ``source`` are kept for compatibility with the earlier ``Case.explain_routes``
    route matrix tests.
    """

    def __init__(self, feature, *, layout="any", backend="any", platform="host",
                 mpi=False, gpu=False, status="unknown", limitation="", error_message="",
                 source="native", axis=None, available_route="", alternative=""):
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

    def to_dict(self):
        return {
            "feature": self.feature,
            "layout": self.layout,
            "backend": self.backend,
            "platform": self.platform,
            "mpi": self.mpi,
            "gpu": self.gpu,
            "status": self.status,
            "limitation": self.limitation,
            "error_message": self.error_message,
            "source": self.source,
            "axis": self.axis,
            "available_route": self.available_route,
            "alternative": self.alternative,
        }

    def __repr__(self):
        return ("CapabilityRouteRow(feature=%r, layout=%r, backend=%r, status=%r, source=%r)"
                % (self.feature, self.layout, self.backend, self.status, self.source))


class CapabilityRouteMatrix:
    """Printable ADC-549 matrix of feature x layout/backend/platform support."""

    def __init__(self, owner, layout, rows):
        self.owner = owner
        self.case_name = owner  # compatibility with the old Case route matrix object.
        self.layout = layout
        self.layout_name = layout
        self.rows = list(rows)

    def to_dict(self):
        return {"case": self.owner, "owner": self.owner, "layout": self.layout,
                "rows": [r.to_dict() for r in self.rows]}

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return "CapabilityRouteMatrix(owner=%r, layout=%r, %d rows)" % (
            self.owner, self.layout, len(self.rows))

    def __str__(self):
        lines = ["route matrix for %r (layout=%s, ADC-549):" % (self.owner, self.layout)]
        for row in self.rows:
            note = ("  -- %s" % row.limitation) if row.limitation else ""
            lines.append(
                "  %-30s layout=%-12s backend=%-11s platform=%-5s mpi=%-5s gpu=%-5s %-11s%s"
                % (row.feature, row.layout, row.backend, row.platform, row.mpi, row.gpu,
                   row.status, note))
        return "\n".join(lines)


def _axis_for_route(layout, backend, platform):
    if layout not in ("any", "uniform|amr", "context"):
        return "layout"
    if platform in ("mpi", "gpu"):
        return "backend"
    if backend not in ("any", "module", "context"):
        return "backend"
    return "transport"


def _flag_value(flags, name):
    if flags is None:
        return None
    return flags.get(name)


def _status_from_flag(flags, name):
    value = _flag_value(flags, name)
    if value is None:
        return "unknown"
    return "available" if bool(value) else "unavailable"


def _row(feature, *, layout="any", backend="any", platform="host", flags=None,
         flag=None, mpi=False, gpu=False, limitation="", requested=None,
         available_route=None, alternative=None, source="native", status=None):
    if status is None:
        status = _status_from_flag(flags, flag) if flag else "available"
    err = ""
    if status == "unavailable":
        err = _unsupported_error(
            requested=requested or feature,
            available=available_route or "no native route",
            alternative=alternative)
    return CapabilityRouteRow(
        feature, layout=layout, backend=backend, platform=platform, mpi=mpi, gpu=gpu,
        status=status, limitation=limitation, error_message=err, source=source,
        available_route=available_route or "", alternative=alternative or "")


def _support_rows(flags, source):
    return [
        _row("supports_uniform", layout="uniform", backend="module", platform="host",
             flags=flags, flag="supports_uniform", limitation="single-level Uniform layout",
             requested="layout=Uniform", available_route="layout=Uniform", source=source),
        _row("supports_amr", layout="amr", backend="production", platform="host",
             flags=flags, flag="supports_amr",
             limitation="native AMR envelope: max_levels<=2, ratio=2",
             requested="layout=AMR", available_route="backend='production' target='amr_system'",
             alternative="use Uniform or AMR(max_levels<=2, ratio=2)", source=source),
        _row("supports_mpi", layout="uniform|amr", backend="production", platform="mpi",
             flags=flags, flag="supports_mpi", mpi=bool(_flag_value(flags, "supports_mpi")),
             limitation="MPI is available only when _pops is built with POPS_USE_MPI=ON",
             requested="platform=MPI", available_route="serial/OpenMP build",
             alternative="rebuild with -DPOPS_USE_MPI=ON", source=source),
        _row("supports_gpu", layout="uniform|amr", backend="production", platform="gpu",
             flags=flags, flag="supports_gpu", gpu=bool(_flag_value(flags, "supports_gpu")),
             limitation="GPU is available only for a Kokkos CUDA/HIP device build",
             requested="platform=GPU", available_route="host CPU platform",
             alternative="use KokkosOpenMP/KokkosSerial or a CUDA/HIP build", source=source),
        _row("supports_stride", layout="uniform|amr", backend="production", platform="host",
             flags=flags, flag="supports_stride",
             limitation="real cell stride is carried only by the production/native route",
             requested="strided cell access", available_route="backend='production'",
             alternative="compile with backend='production'", source=source),
        _row("supports_named_fields", layout="uniform|amr", backend="production", platform="host",
             flags=flags, flag="supports_named_fields",
             limitation="named aux-field transport", requested="named aux fields",
             available_route="native named-field transport", source=source),
        _row("supports_partial_imex_mask", layout="uniform|amr", backend="production",
             platform="host", flags=flags, flag="supports_partial_imex_mask",
             limitation="no C++ route backs a partial IMEX mask",
             requested="partial IMEX mask", available_route="full source implicit / split routes",
             alternative="use IMEX/IMEXRK/Split without partial masks", source=source),
    ]


def _inventory_rows(flags, source):
    mpi = bool(_flag_value(flags, "supports_mpi"))
    gpu = bool(_flag_value(flags, "supports_gpu"))
    return [
        _row("layout:Uniform", layout="uniform", backend="module", platform="host",
             mpi=mpi, gpu=gpu, limitation="2D single-level Cartesian/Polar layout", source=source),
        _row("layout:AMR", layout="amr", backend="production", platform="host",
             flags=flags, flag="supports_amr", mpi=mpi, gpu=gpu,
             limitation="max_levels<=2 and ratio=2; unsupported ratios/levels validate before bind",
             requested="AMR(max_levels>2 or ratio!=2)", available_route="AMR(max_levels<=2, ratio=2)",
             alternative="use Uniform or the native AMR envelope", source=source),
        _row("spatial:finite_volume", layout="uniform|amr", backend="production|aot|prototype",
             platform="host", mpi=mpi, gpu=gpu,
             limitation="2D finite-volume route; prototype backend is host-only", source=source),
        _row("riemann:rusanov", layout="uniform|amr", backend="production|aot|prototype",
             platform="host", mpi=mpi, gpu=gpu,
             limitation="requires model max_wave_speed", source=source),
        _row("riemann:hll", layout="uniform|amr", backend="production|aot", platform="host",
             mpi=mpi, gpu=gpu, limitation="requires physical_flux and wave_speeds", source=source),
        _row("riemann:hllc", layout="uniform|amr", backend="production|aot", platform="host",
             mpi=mpi, gpu=gpu,
             limitation="requires Euler/HLLC model capabilities; polar route is unavailable",
             source=source),
        _row("riemann:roe", layout="uniform|amr", backend="production|aot", platform="host",
             mpi=mpi, gpu=gpu,
             limitation="requires Roe dissipation capability; polar route is unavailable",
             source=source),
        _row("reconstruction:firstorder", layout="uniform|amr", backend="production|aot|prototype",
             limitation="ghost_depth=1", source=source),
        _row("reconstruction:muscl", layout="uniform|amr", backend="production|aot|prototype",
             limitation="ghost_depth=2; native limiters minmod/vanleer", source=source),
        _row("reconstruction:weno5", layout="uniform|amr", backend="production|aot",
             limitation="ghost_depth=3; high-order route is native", source=source),
        _row("limiter:mc", layout="uniform|amr", backend="none", status="unavailable",
             limitation="catalogued but no native C++ limiter symbol exists",
             requested="limiter=MC()", available_route="Minmod() or VanLeer()",
             alternative="use pops.numerics.reconstruction.limiters.Minmod()", source=source),
        _row("limiter:superbee", layout="uniform|amr", backend="none", status="unavailable",
             limitation="catalogued but no native C++ limiter symbol exists",
             requested="limiter=Superbee()", available_route="Minmod() or VanLeer()",
             alternative="use pops.numerics.reconstruction.limiters.VanLeer()", source=source),
        _row("elliptic:geometric_mg", layout="uniform|amr", backend="production", platform="host",
             mpi=mpi, gpu=gpu, limitation="native multigrid route; supports variable epsilon",
             source=source),
        _row("elliptic:fft", layout="uniform", backend="production", platform="host", mpi=mpi,
             gpu=gpu, limitation="periodic, constant coefficient, power-of-two uniform grid only",
             source=source),
        _row("elliptic:fft_amr", layout="amr", backend="none", status="unavailable",
             limitation="FFT requires a single uniform periodic mesh, not AMR",
             requested="solver=FFT() with layout=AMR", available_route="GeometricMG() on AMR",
             alternative="use pops.solvers.elliptic.GeometricMG()", source=source),
        _row("krylov:cg_bicgstab_gmres_richardson", layout="uniform|amr", backend="production",
             platform="host", mpi=mpi, gpu=gpu,
             limitation="matrix-free Krylov over native MultiFab primitives", source=source),
        _row("program_context:system", layout="uniform", backend="production", platform="host",
             mpi=mpi, gpu=gpu, limitation="compiled ProgramContext install on System",
             source=source),
        _row("program_context:amr", layout="amr", backend="production", platform="host",
             flags=flags, flag="supports_amr", mpi=mpi, gpu=gpu,
             limitation="AMR program install requires target='amr_system'", source=source),
        _row("output:npz_vtk_hdf5", layout="uniform|amr", backend="runtime", platform="host",
             mpi=mpi, limitation="runtime output writers; AMR VTK is coarse + patch metadata",
             source=source),
        _row("output:plotfile_uniform", layout="uniform", backend="none", status="unavailable",
             limitation="Plotfile is an AMR per-level format; Uniform System has no writer",
             requested="OutputPolicy(format=Plotfile()) on Uniform",
             available_route="HDF5() or npz on Uniform",
             alternative="use HDF5() or bind an AMR output route", source=source),
        _row("checkpoint:system_v1", layout="uniform", backend="runtime", platform="host",
             mpi=mpi, limitation="npz rank-0 gather checkpoint/restart v1", source=source),
        _row("checkpoint:parallel_hdf5", layout="uniform|amr", backend="none",
             platform="mpi", status="unavailable",
             limitation="parallel HDF5 checkpoint is not a native checkpoint route",
             requested="checkpoint(parallel=True)",
             available_route="checkpoint(parallel=False) or write(format='hdf5', parallel=True)",
             alternative="use checkpoint(parallel=False)", source=source),
        _row("checkpoint:amr_dynamic_regrid", layout="amr", backend="none", status="unavailable",
             limitation="bit-identical AMR checkpoint requires a frozen hierarchy (regrid_every=0)",
             requested="AMR checkpoint with dynamic regrid",
             available_route="AMR checkpoint with regrid_every=0",
             alternative="use AmrSystem.write for visualization or freeze regrid", source=source),
    ]


def native_capability_matrix(*, owner="module", layout="module", target="module",
                             flags=None, source=None):
    """Return the ADC-549 native route matrix.

    ``flags`` can be supplied by a compiled artifact manifest. When absent, the built module's
    C++ ``module_capabilities(target)`` is used. The returned rows always expose:
    feature, layout, backend, platform, MPI, GPU, status, limitation and error_message.
    """
    if flags is None:
        flags = _module_capabilities(target)
        source = source or ("native" if flags is not None else "unknown")
    else:
        source = source or "manifest"
    rows = _support_rows(flags, source) + _inventory_rows(flags, source)
    return CapabilityRouteMatrix(owner, layout, rows)


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


__all__ = ["inspect_capabilities", "CapabilityMatrix", "CapabilityEntry",
           "CapabilityMismatchError", "inspect_amr", "AmrReport",
           "CapabilityRouteRow", "CapabilityRouteMatrix", "native_capability_matrix"]
