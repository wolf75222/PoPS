"""pops.codegen.backends -- typed compile-backend descriptors (Spec 5 sec.8.15 / criterion 22).

Spec 5 stabilises "every object that chooses a route is a typed descriptor", and the compile
backend is one such route. Today the compile drivers select the engine by a bare string
(``backend=pops.codegen.Production()`` / ``"aot"`` / ``"prototype"``); this module adds the typed counterparts
:class:`Production` / :class:`AOT` / :class:`JIT` so a caller can pass an object instead of a
string. Each descriptor is INERT: it carries the chosen options + an optional :class:`platform`
and :meth:`lower`\\ s to the existing backend string the drivers already understand. Nothing here
compiles or runs -- :mod:`pops.codegen.compile_drivers` consumes the lowered string.

The three descriptors map onto the existing ``_BACKENDS`` table (``compile_emit``):

* :class:`Production` -> ``"production"`` (the native zero-copy loader; MPI + AMR capable);
* :class:`AOT`        -> ``"aot"``        (ahead-of-time host-marshalled production path);
* :class:`JIT`        -> ``"prototype"``  (JIT virtual-dispatch host prototyping).

``Production(platform=KokkosOpenMP())`` records the platform on the descriptor for later wiring;
the platform does not change the lowered backend string (the device is selected at build /
runtime, not by the backend token). The :mod:`platform <pops.runtime.platforms>` descriptors live
in the runtime layer (they introspect the compiled ``_pops`` build flags).
"""
from pops.descriptors import Availability, Descriptor

# The canonical backend string each typed descriptor lowers to (the token the compile drivers'
# ``_BACKENDS`` table -- ``compile_emit`` -- already keys on). Kept here as the single mapping so
# the lowering is one obvious line per class.
_PRODUCTION = "production"
_AOT = "aot"
_JIT = "prototype"


class _Backend(Descriptor):
    """Base of the typed compile-backend descriptors (category ``"backend"``).

    A backend descriptor names the compile engine (its lowered string) and optionally carries a
    :class:`platform <pops.runtime.platforms>` descriptor recording the device the produced code
    should target. It is inert -- :meth:`lower` returns the backend string + the platform's
    lowering, it never invokes a compiler.

    Subclasses set :attr:`_string` (the canonical backend token). The constructor takes an
    optional ``platform=`` descriptor; passing a string for it is refused (Spec 5 sec.7: a route
    is a typed object, never a free string).
    """

    category = "backend"
    #: The canonical backend string this descriptor lowers to. Subclasses override.
    _string = None

    def __init__(self, platform=None):
        if isinstance(platform, str):
            raise TypeError(
                "%s(platform=%r): platform must be a typed pops.runtime.platforms descriptor "
                "(e.g. KokkosOpenMP()), not a string" % (type(self).__name__, platform))
        self.platform = platform

    @property
    def scheme(self):
        """The canonical backend token (alias of :meth:`lower`; mirrors the brick ``scheme``)."""
        return self._string

    def lower(self, context=None):
        """The backend string the compile drivers consume (Spec 5 sec.6/7: inert metadata).

        Returns the canonical token (``"production"`` / ``"aot"`` / ``"prototype"``) so a caller
        can pass either the string or this descriptor to ``compile_problem`` / ``compile_model``.
        The recorded platform (if any) is reported by :meth:`inspect`; it does NOT change the
        backend token (the device is chosen at build / run time).
        """
        return self._string

    def options(self):
        return {"backend": self._string,
                "platform": self.platform.name if self.platform is not None else None}

    def capabilities(self):
        """The honest backend characteristics (cpu / mpi / amr / gpu) from the native table."""
        # Imported lazily to keep this module dependency-light and avoid an import cycle with the
        # compile pipeline (which imports the backend descriptors back).
        from pops.codegen.compile_emit import _BACKEND_CAPS
        return dict(_BACKEND_CAPS.get(self._string, {}))

    def available(self, context=None):
        """Available when the recorded platform (if any) is available; else explain why not."""
        if self.platform is not None and hasattr(self.platform, "available"):
            status = self.platform.available(context)
            if not status.ok:
                return Availability.no(
                    "%s targets platform %s which is not available: %s"
                    % (self.name, self.platform.name, status.reason),
                    missing=getattr(status, "missing", None),
                    alternatives=getattr(status, "alternatives", None))
        return Availability.yes()

    def inspect(self):
        record = super().inspect()
        record["platform"] = self.platform.inspect() if self.platform is not None else None
        return record


class Production(_Backend):
    """The native production backend (lowers to ``"production"``; MPI + AMR capable).

    ``Production()`` or ``Production(platform=KokkosOpenMP())``. The native zero-copy loader path
    (``add_native_block``); the only backend that supports ``target="amr_system"`` and the
    compiled time-program path (``compile_problem``).
    """

    _string = _PRODUCTION


class AOT(_Backend):
    """The ahead-of-time backend (lowers to ``"aot"``; host-marshalled production path).

    ``AOT()`` or ``AOT(platform=KokkosSerial())``. Numerically identical to the native path; the
    one backend that materialises runtime block params (``add_compiled_block``).
    """

    _string = _AOT


class JIT(_Backend):
    """The JIT prototyping backend (lowers to ``"prototype"``; virtual-dispatch host path).

    ``JIT()``. Fast host prototyping (``IModel`` virtual dispatch); no MPI / AMR / GPU.
    """

    _string = _JIT


# The typed-string mapping the compile drivers reuse to lower a typed backend back to its token
# without importing the classes (keeps the consumer's import light): {token: class}.
BACKEND_DESCRIPTORS = {_PRODUCTION: Production, _AOT: AOT, _JIT: JIT}


def lower_backend(backend):
    """Strictly lower a public backend selector to its canonical string.

    Public compile surfaces are Spec-5 descriptors, not string selectors. ``None`` means the
    canonical default ``Production()``. Internal codegen seams that already operate on the native
    backend tokens must call :func:`lower_internal_backend` instead.
    """
    if backend is None:
        backend = Production()
    if isinstance(backend, str):
        raise TypeError(
            "backend must be a typed pops.codegen backend descriptor "
            "(Production(), AOT() or JIT()), not %r" % backend)
    if not isinstance(backend, _Backend):
        raise TypeError(
            "backend must be Production()/AOT()/JIT(); got %r"
            % type(backend).__name__)
    return backend.lower()


def lower_internal_backend(backend):
    """Permissive lowering for private codegen seams that already receive native tokens.

    This is deliberately named as an internal escape hatch. It is used only where the caller is
    not a public route selector, for example a cached runtime install path that already derived
    ``"production"`` from a layout/backend descriptor.
    """
    if isinstance(backend, _Backend):
        return backend.lower()
    return backend


def lower_problem_backend(backend):
    """Strict lowering for ``compile_problem``.

    ``compile_problem`` is the Spec-5/6 public problem route; it must not accept string selectors.
    ``None`` means the canonical default ``Production()``. Anything else must be a typed backend
    descriptor.
    """
    try:
        return lower_backend(backend)
    except TypeError as exc:
        raise TypeError("compile_problem: %s" % exc) from None


__all__ = ["Production", "AOT", "JIT", "lower_backend", "lower_problem_backend",
           "lower_internal_backend", "BACKEND_DESCRIPTORS"]
