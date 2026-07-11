"""Rich, inert self-description of a compiled artifact (Spec 5 sec.13.12).

Metadata comes from the compiled handle's public surface. Missing native facts remain ``None``
(unknown), and imports stay lazy so this module remains at the bottom of the import graph.
"""

# Capability flags Spec 5 sec.13.12 enumerates. The first four are GENUINELY derivable from the
# backend capability dict the compiled model carries (CompiledModel.caps = {cpu, mpi, amr, gpu});
# the last three have NO emitted source in today's C++ codegen, so they are reported UNKNOWN
# (None) and listed by needs_cpp_followup() rather than fabricated.
_SUPPORTS_FROM_CAPS = ("supports_uniform", "supports_amr", "supports_mpi", "supports_gpu")
_SUPPORTS_UNKNOWN = ("supports_stride", "supports_partial_imex_mask", "supports_named_fields")
_SUPPORTS_RUNTIME = ("supports_custom_communicator",)
_SUPPORTS_FLAGS = _SUPPORTS_FROM_CAPS + _SUPPORTS_UNKNOWN + _SUPPORTS_RUNTIME

# STRICT versioned schema of the rich compiled-artifact manifest (ADC-611). to_dict() stamps it;
# from_dict() refuses a dict without it (or a wrong one) so any future read-back path is strict by
# construction. Bump when a field is renamed/removed (an additive field keeps version 1).
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


class CompiledArtifactManifest:
    """Self-describing value needed to bind a compiled artifact safely.

    It records ABI identity, blocks, state roles, parameter classes, halo requirements, outputs,
    external bricks and backend capabilities. A capability of ``None`` is unknown, not false;
    validators must not reject it as unsupported. ``to_dict`` is the versioned wire form.
    """

    def __init__(self, *, model_name=None, abi_key=None, abi_version=None,
                 required_headers_sig=None, blocks=None, variables=None, roles=None,
                 aux_required=None, params_const=None, params_runtime=None, params_derived=None,
                 bind_schema=None, bind_schema_hash=None, bind_schema_artifact_hash=None,
                 ghost_depth=None,
                 ghost_depth_by_block=None,
                 field_outputs=None, supports_uniform=None, supports_amr=None, supports_mpi=None,
                 supports_gpu=None, supports_stride=None, supports_partial_imex_mask=None,
                 supports_named_fields=None, native_entrypoints=None, dimension=2,
                 amr_refinement_ratio=2, precision="double", real_bytes=8,
                 communicator="unknown", supports_custom_communicator=False,
                 external_bricks=None):
        self.model_name = model_name
        self.abi_key = abi_key
        self.abi_version = abi_version
        self.required_headers_sig = required_headers_sig
        self.blocks = list(blocks or [])
        self.variables = list(variables or [])
        self.roles = list(roles) if roles is not None else None
        self.aux_required = list(aux_required or [])
        self.params_const = list(params_const or [])
        self.params_runtime = list(params_runtime or [])
        self.params_derived = list(params_derived or [])
        if bind_schema is None:
            if bind_schema_hash is not None or bind_schema_artifact_hash is not None:
                raise ValueError(
                    "bind schema hashes cannot be present without a bind_schema payload"
                )
            self.bind_schema = None
            self.bind_schema_hash = None
            self.bind_schema_artifact_hash = None
        else:
            from pops.model.bind_schema import BindSchema

            schema = BindSchema.from_dict(bind_schema)
            if bind_schema_hash is not None and bind_schema_hash != schema.hash:
                raise ValueError("bind_schema_hash does not match the BindSchema payload")
            if (
                bind_schema_artifact_hash is not None
                and bind_schema_artifact_hash != schema.artifact_hash
            ):
                raise ValueError(
                    "bind_schema_artifact_hash does not match the BindSchema artifact projection"
                )
            self.bind_schema = schema.to_dict()
            self.bind_schema_hash = schema.hash
            self.bind_schema_artifact_hash = schema.artifact_hash
            # Qualified summaries are projections of the schema, never a competing declaration
            # table supplied by the caller.
            self.params_const = sorted(slot.qid for slot in schema.const_slots)
            self.params_runtime = sorted(slot.qid for slot in schema.runtime_slots)
            self.params_derived = sorted(slot.qid for slot in schema.derived_slots)
        self.ghost_depth = ghost_depth
        # Per-block halo depth (ADC-536 / CONTRACTS6 decision 4): the bind stream validates each
        # block's initial-state ghosts against this map. A plain {name: depth} dict, serializable
        # so the ADC-564 typed-report conversion wraps it unchanged.
        self.ghost_depth_by_block = dict(ghost_depth_by_block or {})
        self.field_outputs = list(field_outputs or [])
        # supports_* flags: True / False when GENUINELY known, None when the C++ does not emit it.
        self.supports_uniform = supports_uniform
        self.supports_amr = supports_amr
        self.supports_mpi = supports_mpi
        self.supports_gpu = supports_gpu
        self.supports_stride = supports_stride
        self.supports_partial_imex_mask = supports_partial_imex_mask
        self.supports_named_fields = supports_named_fields
        self.native_entrypoints = list(native_entrypoints or [])
        # ADC-544: the external compiled bricks bound into this artifact (via CompiledBrickRef entries
        # in libraries=). Each entry is the brick's manifest record (native_id / category /
        # requirements / capabilities / supported_layouts / supported_platforms / exported_symbols), so
        # the artifact self-describes its external dependencies. Additive field -> no schema bump; []
        # for an artifact with no external bricks (byte-identical serialization to before).
        self.external_bricks = [dict(b) for b in external_bricks] if external_bricks else []
        self.dimension = dimension
        self.amr_refinement_ratio = amr_refinement_ratio
        self.precision = precision
        self.real_bytes = real_bytes
        self.communicator = communicator
        self.supports_custom_communicator = bool(supports_custom_communicator)

    def supports(self):
        """The ``{flag: True/False/None}`` capability map (``None`` = honestly unknown)."""
        return {name: getattr(self, name) for name in _SUPPORTS_FLAGS}

    def capability_matrix(self):
        """The ADC-549 route matrix generated from this manifest's capability flags."""
        from pops._capabilities import native_capability_matrix
        return native_capability_matrix(
            owner=self.model_name or "compiled-artifact", layout="manifest",
            flags=self.supports(), source="manifest")

    def needs_cpp_followup(self):
        """Return unknown capability flags plus absent native entrypoints."""
        pending = [name for name in _SUPPORTS_FLAGS if getattr(self, name) is None]
        if not self.native_entrypoints:
            pending.append("native_entrypoints")
        return pending

    def to_dict(self):
        """A plain-dict view of every manifest field (JSON-ready; ``None`` flags stay ``None``). Stamped
        with ``schema_version`` (ADC-611) so a strict :meth:`from_dict` read-back can reject a legacy or
        incompatible dict by construction."""
        out = {"schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
               "model_name": self.model_name, "abi_key": self.abi_key,
               "abi_version": self.abi_version, "required_headers_sig": self.required_headers_sig,
               "blocks": list(self.blocks), "variables": list(self.variables),
               "roles": list(self.roles) if self.roles is not None else None,
               "aux_required": list(self.aux_required),
               "params_const": list(self.params_const),
               "params_runtime": list(self.params_runtime),
               "params_derived": list(self.params_derived),
               "bind_schema": self.bind_schema,
               "bind_schema_hash": self.bind_schema_hash,
               "bind_schema_artifact_hash": self.bind_schema_artifact_hash,
               "ghost_depth": self.ghost_depth,
               "ghost_depth_by_block": dict(self.ghost_depth_by_block),
               "field_outputs": list(self.field_outputs),
               "dimension": self.dimension,
               "amr_refinement_ratio": self.amr_refinement_ratio,
               "precision": self.precision,
               "real_bytes": self.real_bytes,
               "communicator": self.communicator,
               "supports_custom_communicator": self.supports_custom_communicator,
               "native_entrypoints": list(self.native_entrypoints),
               "external_bricks": [dict(b) for b in self.external_bricks],
               "capability_matrix": [row.to_dict() for row in self.capability_matrix().rows]}
        out.update(self.supports())
        return out

    @classmethod
    def from_dict(cls, data):
        """Strict read-back of :meth:`to_dict` (ADC-611): reconstruct a manifest from its dict, refusing a
        legacy/incompatible payload BY CONSTRUCTION. Policy -- the error NAMES the offending field:
          - not a dict -> ValueError;
          - missing ``schema_version`` -> ValueError (legacy dict; re-serialize with the current build);
          - ``schema_version`` != ARTIFACT_MANIFEST_SCHEMA_VERSION -> ValueError (naming got vs expected);
          - an UNKNOWN key -> ValueError (naming it; no permissive silent-ignore).
        ``capability_matrix`` is a DERIVED view (rebuilt from the flags), so it is accepted and ignored on
        read-back; the constructor recomputes it. Round-trip: ``from_dict(m.to_dict())`` equals ``m``."""
        if not isinstance(data, dict):
            raise ValueError("compiled-artifact manifest must be a dict; got %r" % (data,))
        if "schema_version" not in data:
            raise ValueError("compiled-artifact manifest is missing the required 'schema_version' "
                             "field (expected %d); it predates the versioned schema -- re-serialize it "
                             "with the current build" % (ARTIFACT_MANIFEST_SCHEMA_VERSION,))
        version = data["schema_version"]
        if version != ARTIFACT_MANIFEST_SCHEMA_VERSION:
            raise ValueError("compiled-artifact manifest 'schema_version' is %r, incompatible with the "
                             "supported version %d" % (version, ARTIFACT_MANIFEST_SCHEMA_VERSION))
        # DERIVED keys are recomputed by the constructor, not passed to it: accept-and-ignore on read-back.
        derived = {"schema_version", "capability_matrix"}
        # The constructor keyword arguments (every stored field, including the supports_* flags).
        ctor_keys = set(_SUPPORTS_FLAGS) | {
            "model_name", "abi_key", "abi_version", "required_headers_sig", "blocks", "variables",
            "roles", "aux_required", "params_const", "params_runtime", "params_derived",
            "bind_schema", "bind_schema_hash", "bind_schema_artifact_hash", "ghost_depth",
            "ghost_depth_by_block", "field_outputs",
            "native_entrypoints", "external_bricks", "dimension", "amr_refinement_ratio", "precision",
            "real_bytes", "communicator"}
        unknown = sorted(set(data) - ctor_keys - derived)
        if unknown:
            raise ValueError("compiled-artifact manifest has unknown field(s) %s; the strict schema does "
                             "not accept them" % (unknown,))
        kwargs = {k: data[k] for k in data if k in ctor_keys}
        result = cls(**kwargs)
        if result.to_dict() != data:
            raise ValueError(
                "compiled-artifact manifest is not canonical; parameter summaries and BindSchema "
                "hashes must be derived from the embedded schema"
            )
        return result

    def __str__(self):
        def _flag(value):
            return "unknown" if value is None else ("yes" if value else "no")

        lines = ["compiled-artifact manifest %r (Spec 5 sec.13.12)"
                 % (self.model_name or "problem")]
        lines.append("  abi          : key=%s version=%s headers=%s"
                     % (_short(self.abi_key), self.abi_version,
                        _short(self.required_headers_sig)))
        lines.append("  blocks       : %s" % (", ".join(self.blocks) or "(none)"))
        lines.append("  variables    : %s" % (", ".join(self.variables) or "(none)"))
        lines.append("  roles        : %s"
                     % ("(unknown)" if self.roles is None else
                        (", ".join(self.roles) or "(none)")))
        lines.append("  aux_required : %s" % (", ".join(self.aux_required) or "(none)"))
        lines.append("  params       : const=[%s] runtime=[%s] derived=[%s]"
                     % (", ".join(self.params_const), ", ".join(self.params_runtime),
                        ", ".join(self.params_derived)))
        if self.bind_schema is not None:
            lines.append("  bind_schema  : hash=%s artifact_hash=%s slots=%d"
                         % (_short(self.bind_schema_hash),
                            _short(self.bind_schema_artifact_hash),
                            len(self.bind_schema.get("slots", []))))
        by_block = ("; ".join("%s=%s" % (b, d) for b, d in sorted(self.ghost_depth_by_block.items()))
                    if self.ghost_depth_by_block else "(none)")
        lines.append("  ghost_depth  : %s (by block: %s)" % (self.ghost_depth, by_block))
        lines.append("  field_outputs: %s" % (", ".join(self.field_outputs) or "(none)"))
        lines.append("  runtime      : dimension=%s amr_refinement_ratio=%s precision=%s "
                     "real_bytes=%s communicator=%s custom_communicator=%s"
                     % (self.dimension, self.amr_refinement_ratio, self.precision,
                        self.real_bytes, self.communicator,
                        "yes" if self.supports_custom_communicator else "no"))
        lines.append("  supports     :")
        for name in _SUPPORTS_FLAGS:
            lines.append("    %-26s %s" % (name, _flag(getattr(self, name))))
        lines.append("  native_entrypoints: %s"
                     % (", ".join(self.native_entrypoints) or "(none)"))
        if self.external_bricks:
            lines.append("  external_bricks:")
            for brick in self.external_bricks:
                layouts = ",".join(brick.get("supported_layouts") or []) or "(any)"
                lines.append("    - %s [%s] native_id=%s layouts=%s"
                             % (brick.get("id"), brick.get("category", "brick"),
                                brick.get("native_id") or brick.get("id"), layouts))
        pending = self.needs_cpp_followup()
        if pending:
            lines.append("  needs C++ follow-up (UNKNOWN, not fabricated): %s"
                         % ", ".join(pending))
        unsupported = [row for row in self.capability_matrix().rows
                       if row.status == "unavailable"]
        if unsupported:
            lines.append("  explicit limitations:")
            for row in unsupported[:8]:
                lines.append("    - %s: %s" % (row.feature, row.error_message or row.limitation))
        return "\n".join(lines)

    def __repr__(self):
        return ("CompiledArtifactManifest(model_name=%r, abi_key=%r, blocks=%r)"
                % (self.model_name, _short(self.abi_key), self.blocks))


def _short(value):
    """A short (12-char) head of an abi key / signature for printing (``None`` stays ``None``)."""
    if not value:
        return value
    text = str(value)
    return text if len(text) <= 12 else text[:12] + "..."


def _headers_sig(abi_key):
    """The header-signature token of an ``<headers>|<cxx>|<std>`` abi key (``None`` if absent).

    Mirrors :func:`pops.codegen.abi.module_header_signature`'s ``|`` split: the abi key's first
    pipe-delimited field is the header signature. ``None`` when there is no key or no signature."""
    if not abi_key:
        return None
    head = str(abi_key).split("|", 1)[0].strip()
    return head or None


def _caps_flags(model):
    """The ``{supports_uniform/amr/mpi/gpu: bool}`` flags from a model's backend caps (else None).

    The compiled model carries ``caps = {cpu, mpi, amr, gpu}`` (CompiledModel, the backend
    capability dict). ``supports_uniform`` mirrors ``cpu`` (a CPU-capable artifact runs on a
    single uniform grid). When the carried model records NO caps (e.g. a bare facade model), every
    flag is ``None`` -- honestly unknown, never fabricated as ``True``/``False``."""
    caps = getattr(model, "caps", None)
    if not caps:
        return {name: None for name in _SUPPORTS_FROM_CAPS}
    return {"supports_uniform": bool(caps.get("cpu", False)),
            "supports_amr": bool(caps.get("amr", False)),
            "supports_mpi": bool(caps.get("mpi", False)),
            "supports_gpu": bool(caps.get("gpu", False))}


def build_compiled_manifest(compiled):
    """Build a manifest from a compiled handle's inert public metadata.

    A degraded handle without ``arguments()`` keeps its ABI/model facts and empty bind groups.
    """
    model = getattr(compiled, "model", None)
    abi_key = getattr(compiled, "abi_key", None)
    model_name = (getattr(compiled, "program_name", None)
                  or getattr(model, "name", None))

    args = compiled.arguments() if hasattr(compiled, "arguments") else None
    if args is not None:
        blocks = sorted(args.instances)
        aux_required = sorted(args.aux)
        params_const = sorted(n for n, s in args.params.items() if s.get("kind") == "const")
        params_runtime = sorted(n for n, s in args.params.items() if s.get("kind") == "runtime")
        params_derived = sorted(n for n, s in args.params.items() if s.get("kind") == "derived")
        field_outputs = sorted(set(args.solvers) | set(args.outputs))
        ghost_depth = args.layout_runtime.get("ghost_depth")
        ghost_depth_by_block = dict(args.layout_runtime.get("ghost_depth_by_block") or {})
    else:
        blocks = aux_required = params_const = params_runtime = params_derived = field_outputs = []
        ghost_depth = None
        ghost_depth_by_block = {}

    variables = list(getattr(model, "cons_names", []) or [])
    raw_roles = getattr(model, "cons_roles", None)
    roles = list(raw_roles) if raw_roles else None

    # ADC-544: the external compiled bricks bound into the artifact (the CompiledBrickRef records
    # compile_problem validated + captured on the handle). [] for an artifact with no external bricks.
    external_bricks = list(getattr(compiled, "external_bricks", []) or [])

    bind_schema = getattr(compiled, "bind_schema", None)
    bind_schema_data = bind_schema.to_dict() if bind_schema is not None else None
    bind_schema_hash = bind_schema.hash if bind_schema is not None else None
    bind_schema_artifact_hash = bind_schema.artifact_hash if bind_schema is not None else None

    caps_flags = _caps_flags(model)
    from pops.runtime_environment import compiled_runtime_facts
    runtime_facts = compiled_runtime_facts(supports_mpi=caps_flags.get("supports_mpi"))

    return CompiledArtifactManifest(
        model_name=model_name, abi_key=abi_key, abi_version=None,
        required_headers_sig=_headers_sig(abi_key), blocks=blocks, variables=variables,
        roles=roles, aux_required=aux_required, params_const=params_const,
        params_runtime=params_runtime, params_derived=params_derived,
        bind_schema=bind_schema_data, bind_schema_hash=bind_schema_hash,
        bind_schema_artifact_hash=bind_schema_artifact_hash, ghost_depth=ghost_depth,
        ghost_depth_by_block=ghost_depth_by_block, field_outputs=field_outputs,
        supports_stride=None, supports_partial_imex_mask=None, supports_named_fields=None,
        native_entrypoints=[], external_bricks=external_bricks,
        dimension=runtime_facts["dimension"],
        amr_refinement_ratio=runtime_facts["amr_refinement_ratio"],
        precision=runtime_facts["precision"], real_bytes=runtime_facts["real_bytes"],
        communicator=runtime_facts["communicator"],
        supports_custom_communicator=runtime_facts["supports_custom_communicator"],
        **caps_flags)


# Fields emitted authoritatively by the native artifact supersede Python-derived values.
_NATIVE_BOOL_FIELDS = ("supports_stride", "supports_partial_imex_mask", "supports_named_fields",
                       "supports_custom_communicator")

# Layout/platform facts also come from the artifact itself, not model potential.
_NATIVE_LAYOUT_PLATFORM_FIELDS = ("supports_uniform", "supports_amr", "supports_mpi", "supports_gpu")


def apply_native_manifest(manifest, native):
    """Overlay authoritative native facts; absent fields leave carried metadata untouched."""
    if not native:
        return manifest
    if "abi_version" in native:
        manifest.abi_version = native["abi_version"]
    if "ghost_depth" in native and manifest.ghost_depth is None:
        manifest.ghost_depth = native["ghost_depth"]
    if native.get("ghost_depth_by_block") and not manifest.ghost_depth_by_block:
        manifest.ghost_depth_by_block = dict(native["ghost_depth_by_block"])
    for field in _NATIVE_BOOL_FIELDS:
        if field in native:
            setattr(manifest, field, bool(native[field]))
    for field in _NATIVE_LAYOUT_PLATFORM_FIELDS:
        if field in native:
            setattr(manifest, field, bool(native[field]))
    for field in ("dimension", "amr_refinement_ratio", "real_bytes"):
        if field in native:
            setattr(manifest, field, int(native[field]))
    for field in ("precision", "communicator"):
        if field in native:
            setattr(manifest, field, str(native[field]))
    roles = native.get("roles")
    if roles and manifest.roles is None:
        manifest.roles = list(roles)
    entrypoints = native.get("native_entrypoints")
    if entrypoints:
        manifest.native_entrypoints = list(entrypoints)
    return manifest


def load_native_manifest(so_path):
    """The authoritative ``pops_compiled_manifest()`` dict of the .so at @p so_path (or ``None``).

    A thin wrapper over :func:`pops.descriptors.load_compiled_manifest` (dlopens the ``.so``, reads
    the exported C symbol). Imported function-locally so the ``external`` layer stays import-graph
    clean and pulls in ``ctypes`` only on demand. Returns ``None`` for an old ``.so`` that does not
    export the symbol (graceful fallback)."""
    from pops.descriptors import load_compiled_manifest  # lazy: keep external's module scope clean
    return load_compiled_manifest(so_path)


def build_compiled_manifest_from_so(compiled, so_path):
    """Build the rich manifest of @p compiled, then overlay the .so's authoritative facts (sec.13.12).

    Combines :func:`build_compiled_manifest` (the metadata the handle carries) with
    :func:`apply_native_manifest` (the artifact's own ``pops_compiled_manifest()``), so the fields the
    C++ codegen now emits become REAL rather than honest-None. When @p so_path is an old ``.so``
    without the symbol the manifest is identical to :func:`build_compiled_manifest` (graceful
    fallback)."""
    manifest = build_compiled_manifest(compiled)
    return apply_native_manifest(manifest, load_native_manifest(so_path))


# Spec 5 sec.13.12 maps each layout kind to the capability flag a compiled artifact must support
# to bind under it. ``"amr"`` -> supports_amr, ``"uniform"`` -> supports_uniform.
_LAYOUT_SUPPORT_FLAG = {"amr": "supports_amr", "uniform": "supports_uniform",
                        "system": "supports_uniform"}


def check_layout_supported(manifest, layout_kind):
    """Validate a compiled artifact against a target layout via its capability flags (sec.13.12).

    Spec 5 sec.13.12 error shape: a manifest whose layout flag is GENUINELY ``False`` rejects with
    "Compiled artifact cannot be used with layout=AMR(...): supports_amr=false". This raises
    :class:`ValueError` ONLY when the flag is known-``False``; a flag that is ``None`` (UNKNOWN --
    the C++ has not emitted it) is NOT a rejection -- it returns the (truthy) "unknown" status so a
    working route is never broken by a missing check (the no-false-positive discipline: a false
    rejection is worse than a missing one). An unrecognised @p layout_kind returns "unknown" too.

    Returns an :class:`pops.descriptors.Availability` -- ``yes`` when the flag is ``True``,
    ``partial`` (truthy is ``False``, but carries the reason) when ``None``/unrecognised, and
    raises on a known ``False``. The caller wires it on the layout-bind branch."""
    from pops.descriptors import Availability  # lazy: keeps the external layer's module scope clean
    flag_name = _LAYOUT_SUPPORT_FLAG.get(str(layout_kind).lower())
    if flag_name is None:
        return Availability.partial(
            "layout %r has no known capability flag; not validated" % (layout_kind,))
    value = getattr(manifest, flag_name, None)
    if value is None:
        return Availability.partial(
            "%s is unknown (the C++ codegen does not emit it yet); layout %r not validated -- "
            "not rejecting on an unknown flag" % (flag_name, layout_kind))
    if value is False:
        raise ValueError(
            "Compiled artifact cannot be used with layout=%s(...): %s=false; unsupported route: "
            "requested layout=%s; available route: %s; alternative: %s"
            % (layout_kind, flag_name, layout_kind,
               "layout=Uniform" if flag_name == "supports_amr" else "layout=AMR",
               "compile a backend/target that emits %s=true or choose a supported layout"
               % flag_name))
    return Availability.yes("%s=true" % flag_name)


__all__ = ["CompiledArtifactManifest", "build_compiled_manifest", "check_layout_supported",
           "apply_native_manifest", "load_native_manifest", "build_compiled_manifest_from_so",
           "ARTIFACT_MANIFEST_SCHEMA_VERSION"]
