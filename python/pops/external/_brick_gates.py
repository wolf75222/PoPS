"""pops.external._brick_gates -- the compile-time validation gates for a CompiledBrickRef (ADC-544).

A :class:`pops.external.bricks.CompiledBrickRef` names a brick that lives in a ``.so`` / manifest
outside the PoPS core. ADC-544 requires that such a brick be VALIDATED before any use, at compile /
resolve time, and that every refusal RAISE (never warn). This module owns the four gates
:func:`validate_ref` runs, each with a verbatim-testable error string:

  - G1 ABI mismatch (:class:`RuntimeError`): the brick's ``abi_key`` differs from the loaded ``_pops``
    module key. Only a PRESENT mismatched key raises; a manifest with no key (a ``.json`` that never
    compiled a ``.so``) is not checkable and is SKIPPED, mirroring the library-``.so`` guard's
    ``module_abi in ("", "abi_key=unavailable")`` skip.
  - G2 missing capability (:class:`ValueError`): a brick ``requirement`` names a model capability the
    bind context does not provide. A context with no capability information cannot know, so it does
    NOT refuse (no false positive, mirroring the riemann-availability "no model in scope" rule).
  - G3 unsupported layout (:class:`ValueError`): the requested layout is excluded by the brick's
    ``supported_layouts``. An empty / unknown ``supported_layouts`` is unconstrained -- NOT a
    rejection (the no-false-positive discipline the artifact-manifest layout check follows).
  - G4 missing symbol (:class:`ValueError`): a ``.so`` manifest lists ``exported_symbols`` the ``.so``
    does not export. Probed by ``getattr(handle, sym)`` (ctypes dlsym / GetProcAddress) on THIS
    brick ``.so``'s OWN handle, never a process-global lookup. A ``.json``-only manifest has no
    ``.so`` to probe, so G4 is honestly SKIPPED.

The gates read the parsed per-brick record (:func:`pops.descriptors.parse_brick_manifest` output),
so they are numerics-free and pull no ``_pops`` / numpy at module scope; the ABI key and the ``.so``
handle are reached function-locally, keeping ``pops.external`` at the bottom of the import graph.
"""


def _module_abi_key():
    """The loaded ``_pops`` module ABI key, or a stable placeholder when ``_pops`` is unavailable.

    Mirrors :func:`pops.codegen.library._abi_key`: the key namespaces a brick to the exact toolchain
    that will dlopen it. When ``_pops`` is absent (a numpy-free / module-free interpreter exercising
    the pure-Python gate layer), the placeholder ``"abi_key=unavailable"`` makes G1 SKIP -- an
    un-loadable module cannot adjudicate an ABI mismatch."""
    try:
        from pops import abi_key as _key  # pops.abi_key delegates to _pops.abi_key()
        return _key()
    except Exception:
        return "abi_key=unavailable"


def check_abi(record, manifest_abi_key, *, module_abi_key=None):
    """G1: refuse a brick whose manifest ``abi_key`` differs from the loaded ``_pops`` module (ADC-544).

    Raises :class:`RuntimeError` ONLY when BOTH keys are present AND differ. A manifest with no
    ``abi_key`` (a hand-written ``.json`` that never compiled a ``.so``) is not checkable and is
    SKIPPED -- a missing key is not a mismatch, mirroring the library-``.so`` guard
    (``pops.codegen.library._read_so_manifest``). Likewise an unavailable module key
    (``""`` / ``"abi_key=unavailable"``, e.g. ``_pops`` absent) SKIPS: nothing to compare against.
    """
    if module_abi_key is None:
        module_abi_key = _module_abi_key()
    if module_abi_key in ("", "abi_key=unavailable"):
        return  # module key not loadable -> not checkable (skip, never a false reject)
    if not manifest_abi_key:
        return  # manifest carries no key (a .json without a .so) -> not checkable (skip)
    if manifest_abi_key != module_abi_key:
        raise RuntimeError(
            "external brick %r was compiled with an ABI key DIFFERENT from the loaded module "
            "(%r vs %r); recompile the brick .so with the SAME toolchain (POPS_KOKKOS_ROOT, "
            "compiler, C++ standard, header tree) that built _pops -- dlopen-ing an ABI-mismatched "
            "brick would fail with a cryptic symbol error or undefined behavior."
            % (record["native_id"], manifest_abi_key, module_abi_key))


def _provided_capabilities(context):
    """The set of capability names the bind @p context provides, or ``None`` when it carries none.

    Accepts a plain ``dict`` carrying an explicit ``"capabilities"`` (a collection of capability
    names the model provides) or a ``"model"`` / ``"compiled"`` object; a model itself is also
    accepted. ``None`` means "no capability information in scope" and the caller then does NOT refuse
    (no false positive, mirroring pops.numerics.riemann.availability._model_of).
    """
    if context is None:
        return None
    provided = None
    if isinstance(context, dict):
        if "capabilities" in context and context["capabilities"] is not None:
            provided = context["capabilities"]
        else:
            model = context.get("model", context.get("compiled"))
            provided = _model_capabilities(model)
    else:
        provided = _model_capabilities(context)
    if provided is None:
        return None
    return {str(name) for name in provided}


def _model_capabilities(model):
    """The capability-name collection a model advertises, or ``None`` when it advertises none.

    A model may expose a plain ``provided_capabilities`` collection (the sanctioned brick-context
    surface) -- read it when present. Anything else is treated as "no capability information" (None),
    so the gate does not fabricate a provided set from an unrelated object (no false positive)."""
    if model is None:
        return None
    caps = getattr(model, "provided_capabilities", None)
    if caps is None:
        return None
    return list(caps)


def check_capabilities(record, context):
    """G2: refuse a brick whose ``requirements`` name a capability the model does not provide (ADC-544).

    Raises :class:`ValueError` when the bind @p context provides a KNOWN capability set that is
    missing one of the brick's ``requirements``. A context that carries no capability information
    (no model, no explicit set) cannot know, so it is NOT refused -- the install-time guard still
    fires when a real model is present (no false positive)."""
    required = record.get("requirements") or []
    if not required:
        return
    provided = _provided_capabilities(context)
    if provided is None:
        return  # no capability info in scope -> not checkable (skip, never a false reject)
    missing = [cap for cap in required if cap not in provided]
    if missing:
        raise ValueError(
            "external brick %r requires capability %r not provided by the model; available "
            "capabilities: %s"
            % (record["native_id"], missing[0], sorted(provided) or "(none)"))


def _requested_layout(context):
    """The requested layout token in @p context, or ``None`` when none is declared."""
    if context is None:
        return None
    if isinstance(context, dict):
        layout = context.get("layout")
    else:
        layout = getattr(context, "layout", None)
    if layout is None:
        return None
    return str(layout).lower()


def check_layout(record, context):
    """G3: refuse a brick that does not support the requested layout (ADC-544).

    Raises :class:`ValueError` ONLY when the brick declares a NON-EMPTY ``supported_layouts`` that
    EXCLUDES the requested layout. An empty ``supported_layouts`` (unconstrained / unknown) is NOT a
    rejection, and a context with no requested layout is not checkable -- the no-false-positive
    discipline mirrored from :func:`pops.external.artifact_manifest.check_layout_supported`."""
    supported = record.get("supported_layouts") or []
    if not supported:
        return  # unconstrained / unknown -> not a rejection
    requested = _requested_layout(context)
    if requested is None:
        return  # no layout in scope -> not checkable (skip)
    if requested not in {s.lower() for s in supported}:
        raise ValueError(
            "external brick %r does not support layout=%s : supported layouts are %s"
            % (record["native_id"], requested, sorted(supported)))


def check_symbols(record, handle):
    """G4: refuse a ``.so`` brick that does not export a symbol its manifest lists (ADC-544).

    For each name in the brick's ``exported_symbols``, probe ``getattr(@p handle, name)`` -- the
    ctypes ``CDLL`` handle wraps ``dlsym`` / ``GetProcAddress`` portably. A missing symbol
    (``AttributeError``) raises :class:`ValueError`.

    STB_GNU_UNIQUE caveat (ADC-622): gcc/Linux unifies the header-only ``BrickRegistry`` static
    across every dlopen'd brick ``.so``, so a process-global symbol count is unreliable. This probe
    is therefore keyed on THIS ``.so``'s OWN @p handle (the ctypes ``CDLL`` returned by the dlopen of
    this specific brick), never a process-global lookup, and it asserts only that each expected symbol
    is PRESENT on this handle -- never that a symbol is absent process-wide.

    @p handle of ``None`` (a ``.json``-only manifest -- there is no ``.so`` to probe) SKIPS the gate:
    a manifest may list ``exported_symbols`` for documentation, but only a loaded ``.so`` can be
    probed honestly. The caller documents that ``.json`` manifests skip G4."""
    symbols = record.get("exported_symbols") or []
    if not symbols or handle is None:
        return  # no symbols declared, or a .json manifest with no .so to probe (honest skip)
    for symbol in symbols:
        try:
            getattr(handle, symbol)
        except AttributeError as err:
            raise ValueError(
                "external brick %r does not export symbol %s(); rebuild the .so with the expected "
                "entry point" % (record["native_id"], symbol)) from err


def validate_ref(record, *, manifest_abi_key=None, context=None, handle=None,
                 module_abi_key=None):
    """Run the four ADC-544 compile-time gates on a parsed brick @p record, raising on the first fail.

    @p record is the per-brick dict :func:`pops.descriptors.parse_brick_manifest` returns (it carries
    native_id / requirements / supported_layouts / exported_symbols). Runs, in order:

      - G1 :func:`check_abi` (@p manifest_abi_key vs @p module_abi_key / the loaded module);
      - G2 :func:`check_capabilities` (@p context's provided capabilities);
      - G3 :func:`check_layout` (@p context's requested layout);
      - G4 :func:`check_symbols` (@p handle -- the loaded ``.so``'s ctypes handle, or ``None`` for a
        ``.json`` manifest, which honestly SKIPS G4).

    Every gate RAISES (never warns): G1 :class:`RuntimeError`, G2 / G3 / G4 :class:`ValueError`. A
    successful return means the brick passed every checkable gate; a gate that is not checkable in the
    given scope (no key, no capability info, no layout, no ``.so``) is SKIPPED, never a false reject."""
    check_abi(record, manifest_abi_key, module_abi_key=module_abi_key)
    check_capabilities(record, context)
    check_layout(record, context)
    check_symbols(record, handle)


__all__ = ["validate_ref", "check_abi", "check_capabilities", "check_layout", "check_symbols"]
