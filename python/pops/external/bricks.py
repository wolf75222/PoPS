"""pops.external.bricks -- typed references to compiled-out-of-core bricks (Spec 5 sec.5.17).

A :class:`CompiledBrickRef` names a brick that lives in a ``.so`` / manifest outside the
standard PoPS core but is compatible with its ABI manifests. Spec 5 forbids a free string to
a native brick: a reference carries a manifest + a native id, and resolves to the typed
``external_cpp`` :class:`pops.descriptors.BrickDescriptor` (with the manifest's requirements /
capabilities). A missing manifest or an unregistered id is a clear error before runtime.

ADC-544 threads the four compile-time validation gates through :meth:`CompiledBrickRef.resolve`:
G1 ABI mismatch, G2 missing capability, G3 unsupported layout and G4 a missing exported symbol
(dlsym probe on the loaded ``.so`` handle) are run BEFORE any use and ALL RAISE (never warn). A
``.json``-only manifest has no ``.so`` to probe, so G4 is honestly SKIPPED there.
"""
from pops.descriptors import Availability, Descriptor, _external_descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet
from ._brick_gates import validate_ref
from .manifests import register_and_capture


class CompiledBrickRef(Descriptor):
    """A reference to a compiled brick: a manifest (``.json`` or ``.so``) + a native id.

    ``CompiledBrickRef(manifest="build/my_riemann.json", native_id="my_hll_variant")``.
    :meth:`resolve` registers the manifest, runs the ADC-544 validation gates and returns the
    validated external descriptor; :meth:`available` reports whether it resolves, with the reason
    if not.

    ``expect_layouts`` / ``expect_platforms`` (optional) declare the route context the brick must
    support, feeding the G3 layout gate: a Problem passing this ref among its bricks can pin the
    layout it will bind under so an incompatible brick is refused at compile, not at runtime.
    """

    category = "external_brick"

    def __init__(self, manifest, native_id, *, expect_category=None,
                 expect_layouts=None, expect_platforms=None):
        self.manifest = str(manifest)
        self.native_id = str(native_id)
        self.expect_category = expect_category
        self.expect_layouts = list(expect_layouts) if expect_layouts else []
        self.expect_platforms = list(expect_platforms) if expect_platforms else []
        self._registered = False
        # Captured at registration for the ADC-544 gates: the parsed per-brick record, the manifest
        # abi_key (G1) and the loaded .so ctypes handle (G4; None for a .json manifest -> G4 skipped).
        self._record = None
        self._manifest_abi_key = None
        self._handle = None

    def options(self):
        return {"manifest": self.manifest, "native_id": self.native_id,
                "expect_category": self.expect_category,
                "expect_layouts": list(self.expect_layouts),
                "expect_platforms": list(self.expect_platforms)}

    def _ensure_registered(self):
        if not self._registered:
            records, abi_key, handle = register_and_capture(self.manifest)
            self._manifest_abi_key = abi_key
            self._handle = handle
            self._record = next((r for r in records if r["native_id"] == self.native_id
                                 or r["id"] == self.native_id), None)
            self._registered = True

    def _gate_context(self, context=None):
        """The gate context for this ref: the caller's @p context merged with the ref's route pins.

        The G3 layout gate reads a requested ``layout`` and the G2 capability gate reads the model's
        provided ``capabilities`` from the context. The ref's own ``expect_layouts`` supplies a
        layout when the caller passed none, so a Problem can pin the route on the ref itself."""
        merged = dict(context) if isinstance(context, dict) else {}
        if context is not None and not isinstance(context, dict):
            merged.setdefault("model", context)
        if "layout" not in merged and self.expect_layouts:
            merged["layout"] = self.expect_layouts[0]
        return merged

    def validate(self, context=None):
        """Run the four ADC-544 gates on this ref, raising on the first failure (never warns).

        Registers the manifest (capturing the abi_key + ``.so`` handle) then runs G1 ABI mismatch
        (:class:`RuntimeError`), G2 missing capability, G3 unsupported layout and G4 missing exported
        symbol (:class:`ValueError`). A ``.json``-only manifest skips G4 (no ``.so`` to probe). A ref
        whose ``native_id`` is not in the manifest is left to :meth:`resolve`'s clear ``LookupError``.
        Returns ``True`` when every checkable gate passes."""
        self._ensure_registered()
        if self._record is not None:
            validate_ref(self._record, manifest_abi_key=self._manifest_abi_key,
                         context=self._gate_context(context), handle=self._handle)
        return True

    def resolve(self, context=None):
        """Register the manifest, run the ADC-544 gates and return the typed ``external_cpp`` descriptor.

        The gates (:meth:`validate`) fire BEFORE the descriptor is surfaced, so an ABI-mismatched /
        missing-capability / unsupported-layout / missing-symbol brick is refused before any use. A
        gate failure RAISES (never warns). @p context (optional) carries the model's provided
        capabilities (``"capabilities"`` or a ``"model"``) and the requested ``"layout"``."""
        self._ensure_registered()
        self.validate(context)
        return _external_descriptor(self.native_id, expect_category=self.expect_category)

    def manifest_record(self):
        """The parsed per-brick manifest dict (native_id / category / requirements / capabilities /
        supported_layouts / supported_platforms / params / options / exported_symbols), or ``None``
        when the ``native_id`` is not in the manifest. Registers the manifest on first call. Drives
        the ``compiled.manifest()`` external-bricks integration (ADC-544)."""
        self._ensure_registered()
        return dict(self._record) if self._record is not None else None

    def requirements(self):
        # Inert metadata accessor: an unresolved brick (not loaded / bad manifest) has no
        # requirements to report; the loud signal is surfaced by available()/resolve(), so this
        # degrades to an empty typed set rather than crashing an introspection walk.
        try:
            return RequirementSet(dict(self.resolve().requirements))
        except (LookupError, ValueError, OSError, RuntimeError):
            return RequirementSet()

    def capabilities(self):
        try:
            return CapabilitySet(dict(self.resolve().capabilities))
        except (LookupError, ValueError, OSError, RuntimeError):
            return CapabilitySet()

    def available(self, context=None):
        try:
            self.resolve(context)
            return Availability.yes()
        except Exception as err:
            return Availability.no(
                "compiled brick %r could not be resolved: %s" % (self.native_id, err),
                alternatives=["check the manifest path and native_id"])


# Spec 5 uses both names; ExternalBrick is the same typed reference.
ExternalBrick = CompiledBrickRef

__all__ = ["CompiledBrickRef", "ExternalBrick"]
