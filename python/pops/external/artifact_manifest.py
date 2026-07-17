"""Rich, inert self-description of a compiled artifact (Spec 5 sec.13.12).

Metadata comes from the compiled handle's public surface. Missing native facts remain ``None``
(unknown), and imports stay lazy so this module remains at the bottom of the import graph.
"""
from __future__ import annotations

from typing import Any

from pops._manifest_protocol import manifest_envelope, parse_manifest_envelope
from pops._manifest_immutability import freeze_manifest_json, thaw_manifest_json

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
ARTIFACT_MANIFEST_SCHEMA_VERSION = 2
_MANIFEST_KIND = "compiled-artifact"


def _strict_optional_bool(value, *, where):
    if value is not None and not isinstance(value, bool):
        raise TypeError("%s must be bool or None" % where)
    return value


def _strict_optional_int(value, *, where, minimum=0):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError("%s must be an integer >= %d or None" % (where, minimum))
    return value


def _strict_optional_text(value, *, where):
    if value is not None and (not isinstance(value, str) or not value):
        raise TypeError("%s must be a non-empty string or None" % where)
    return value


class CompiledArtifactManifest:
    """Self-describing value needed to bind a compiled artifact safely.

    It records ABI identity, blocks, state roles, parameter classes, halo requirements, outputs,
    external bricks and backend capabilities. A capability of ``None`` is unknown, not false;
    validators must not reject it as unsupported. ``to_dict`` is the versioned wire form.
    """

    __slots__ = (
        "model_name", "abi_key", "abi_version", "required_headers_sig", "blocks",
        "variables", "roles", "aux_required", "params_const", "params_runtime",
        "params_derived", "bind_schema", "bind_schema_hash", "bind_schema_artifact_hash",
        "ghost_depth", "ghost_depth_by_block", "field_outputs", "supports_uniform",
        "supports_amr", "supports_mpi", "supports_gpu", "supports_stride",
        "supports_partial_imex_mask", "supports_named_fields", "native_entrypoints",
        "external_bricks", "dimension", "amr_refinement_ratio", "precision", "real_bytes",
        "communicator", "supports_custom_communicator",
    )

    def __init__(self, *, model_name=None, abi_key=None, abi_version=None,
                 required_headers_sig=None, blocks=None, variables=None, roles=None,
                 aux_required=None, params_const=None, params_runtime=None, params_derived=None,
                 bind_schema=None, bind_schema_hash=None, bind_schema_artifact_hash=None,
                 ghost_depth=None,
                 ghost_depth_by_block=None,
                 field_outputs=None, supports_uniform=None, supports_amr=None, supports_mpi=None,
                 supports_gpu=None, supports_stride=None, supports_partial_imex_mask=None,
                 supports_named_fields=None, native_entrypoints=None, dimension=None,
                 amr_refinement_ratio=None, precision=None, real_bytes=None,
                 communicator=None, supports_custom_communicator=None,
                 external_bricks=None):
        put = object.__setattr__
        put(self, "model_name", freeze_manifest_json(model_name, where="artifact.model_name"))
        put(self, "abi_key", freeze_manifest_json(abi_key, where="artifact.abi_key"))
        put(self, "abi_version", freeze_manifest_json(abi_version, where="artifact.abi_version"))
        put(
            self,
            "required_headers_sig",
            freeze_manifest_json(required_headers_sig, where="artifact.required_headers_sig"),
        )
        put(self, "blocks", freeze_manifest_json(list(blocks or []), where="artifact.blocks"))
        put(
            self,
            "variables",
            freeze_manifest_json(list(variables or []), where="artifact.variables"),
        )
        put(
            self,
            "roles",
            None if roles is None else freeze_manifest_json(list(roles), where="artifact.roles"),
        )
        put(
            self,
            "aux_required",
            freeze_manifest_json(list(aux_required or []), where="artifact.aux_required"),
        )
        put(
            self,
            "params_const",
            freeze_manifest_json(list(params_const or []), where="artifact.params_const"),
        )
        put(
            self,
            "params_runtime",
            freeze_manifest_json(list(params_runtime or []), where="artifact.params_runtime"),
        )
        put(
            self,
            "params_derived",
            freeze_manifest_json(list(params_derived or []), where="artifact.params_derived"),
        )
        if bind_schema is None:
            if bind_schema_hash is not None or bind_schema_artifact_hash is not None:
                raise ValueError(
                    "bind schema hashes cannot be present without a bind_schema payload"
                )
            put(self, "bind_schema", None)
            put(self, "bind_schema_hash", None)
            put(self, "bind_schema_artifact_hash", None)
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
            put(
                self,
                "bind_schema",
                freeze_manifest_json(schema.to_dict(), where="artifact.bind_schema"),
            )
            put(self, "bind_schema_hash", schema.hash)
            put(self, "bind_schema_artifact_hash", schema.artifact_hash)
            # Qualified summaries are projections of the schema, never a competing declaration
            # table supplied by the caller.
            put(self, "params_const", tuple(sorted(slot.qid for slot in schema.const_slots)))
            put(self, "params_runtime", tuple(sorted(slot.qid for slot in schema.runtime_slots)))
            put(self, "params_derived", tuple(sorted(slot.qid for slot in schema.derived_slots)))
        put(self, "ghost_depth", freeze_manifest_json(ghost_depth, where="artifact.ghost_depth"))
        # Per-block halo depth (ADC-536 / CONTRACTS6 decision 4): held as a read-only mapping;
        # to_dict() returns the detached plain {name: depth} JSON form consumed by reports.
        put(
            self,
            "ghost_depth_by_block",
            freeze_manifest_json(
                dict(ghost_depth_by_block or {}), where="artifact.ghost_depth_by_block"
            ),
        )
        put(
            self,
            "field_outputs",
            freeze_manifest_json(list(field_outputs or []), where="artifact.field_outputs"),
        )
        # supports_* flags: True / False when GENUINELY known, None when the C++ does not emit it.
        capability_values = {
            "supports_uniform": supports_uniform,
            "supports_amr": supports_amr,
            "supports_mpi": supports_mpi,
            "supports_gpu": supports_gpu,
            "supports_stride": supports_stride,
            "supports_partial_imex_mask": supports_partial_imex_mask,
            "supports_named_fields": supports_named_fields,
        }
        for name, value in capability_values.items():
            if value is not None and not isinstance(value, bool):
                raise TypeError("artifact.%s must be bool or None" % name)
            put(self, name, value)
        put(
            self,
            "native_entrypoints",
            freeze_manifest_json(
                list(native_entrypoints or []), where="artifact.native_entrypoints"
            ),
        )
        # ADC-544: the external compiled bricks bound into this artifact (via CompiledBrickRef entries
        # in libraries=). Each entry is the brick's manifest record (native_id / category /
        # requirements / capabilities / supported_layouts / supported_platforms / exported_symbols), so
        # the artifact self-describes its external dependencies. Additive field -> no schema bump; []
        # for an artifact with no external bricks (byte-identical serialization to before).
        put(
            self,
            "external_bricks",
            freeze_manifest_json(
                list(external_bricks or []), where="artifact.external_bricks"
            ),
        )
        put(self, "dimension", _strict_optional_int(
            dimension, where="artifact.dimension", minimum=1))
        put(
            self,
            "amr_refinement_ratio",
            _strict_optional_int(
                amr_refinement_ratio, where="artifact.amr_refinement_ratio", minimum=2),
        )
        put(self, "precision", _strict_optional_text(precision, where="artifact.precision"))
        put(self, "real_bytes", _strict_optional_int(
            real_bytes, where="artifact.real_bytes", minimum=1))
        put(
            self,
            "communicator",
            _strict_optional_text(communicator, where="artifact.communicator"),
        )
        put(self, "supports_custom_communicator", _strict_optional_bool(
            supports_custom_communicator, where="artifact.supports_custom_communicator"))

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
        payload = {"model_name": self.model_name, "abi_key": self.abi_key,
               "abi_version": self.abi_version, "required_headers_sig": self.required_headers_sig,
               "blocks": thaw_manifest_json(self.blocks),
               "variables": thaw_manifest_json(self.variables),
               "roles": thaw_manifest_json(self.roles),
               "aux_required": thaw_manifest_json(self.aux_required),
               "params_const": thaw_manifest_json(self.params_const),
               "params_runtime": thaw_manifest_json(self.params_runtime),
               "params_derived": thaw_manifest_json(self.params_derived),
               "bind_schema": thaw_manifest_json(self.bind_schema),
               "bind_schema_hash": self.bind_schema_hash,
               "bind_schema_artifact_hash": self.bind_schema_artifact_hash,
               "ghost_depth": self.ghost_depth,
               "ghost_depth_by_block": thaw_manifest_json(self.ghost_depth_by_block),
               "field_outputs": thaw_manifest_json(self.field_outputs),
               "dimension": self.dimension,
               "amr_refinement_ratio": self.amr_refinement_ratio,
               "precision": self.precision,
               "real_bytes": self.real_bytes,
               "communicator": self.communicator,
               "supports_custom_communicator": self.supports_custom_communicator,
               "native_entrypoints": thaw_manifest_json(self.native_entrypoints),
               "external_bricks": thaw_manifest_json(self.external_bricks),
               "capability_matrix": [row.to_dict() for row in self.capability_matrix().rows]}
        payload.update(self.supports())
        return manifest_envelope(
            kind=_MANIFEST_KIND,
            schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
            payload=payload,
        )

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
        # DERIVED keys are recomputed by the constructor and compared exactly on read-back.
        derived = {"capability_matrix"}
        # The constructor keyword arguments (every stored field, including the supports_* flags).
        ctor_keys = set(_SUPPORTS_FLAGS) | {
            "model_name", "abi_key", "abi_version", "required_headers_sig", "blocks", "variables",
            "roles", "aux_required", "params_const", "params_runtime", "params_derived",
            "bind_schema", "bind_schema_hash", "bind_schema_artifact_hash", "ghost_depth",
            "ghost_depth_by_block", "field_outputs",
            "native_entrypoints", "external_bricks", "dimension", "amr_refinement_ratio", "precision",
            "real_bytes", "communicator"}
        payload = parse_manifest_envelope(
            data,
            kind=_MANIFEST_KIND,
            schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
            where="compiled-artifact manifest",
        )
        unknown = sorted(set(payload) - ctor_keys - derived)
        if unknown:
            raise ValueError("compiled-artifact manifest has unknown field(s) %s; the strict schema does "
                             "not accept them" % (unknown,))
        missing = sorted((ctor_keys | derived) - set(payload))
        if missing:
            raise ValueError(
                "compiled-artifact manifest is missing required field(s) %s" % missing
            )
        kwargs = {k: payload[k] for k in ctor_keys}
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
                            len(self.bind_schema["payload"]["slots"])))
        by_block = ("; ".join("%s=%s" % (b, d) for b, d in sorted(self.ghost_depth_by_block.items()))
                    if self.ghost_depth_by_block else "(none)")
        lines.append("  ghost_depth  : %s (by block: %s)" % (self.ghost_depth, by_block))
        lines.append("  field_outputs: %s" % (", ".join(self.field_outputs) or "(none)"))
        lines.append("  runtime      : dimension=%s amr_refinement_ratio=%s precision=%s "
                     "real_bytes=%s communicator=%s custom_communicator=%s"
                     % (self.dimension, self.amr_refinement_ratio, self.precision,
                        self.real_bytes, self.communicator,
                        _flag(self.supports_custom_communicator)))
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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CompiledArtifactManifest) and self.to_dict() == other.to_dict()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CompiledArtifactManifest is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CompiledArtifactManifest is immutable")


def _short(value):
    """A short (12-char) head of an abi key / signature for printing (``None`` stays ``None``)."""
    if not value:
        return value
    text = str(value)
    return text if len(text) <= 12 else text[:12] + "..."


def _headers_sig(abi_key):
    """Return the header-signature token of an ABI key."""
    if not abi_key:
        return None
    head = str(abi_key).split("|", 1)[0].strip()
    return head or None


def _caps_flags(compiled):
    """Intersect layout/platform flags across every InstallPlan block model."""
    from pops.codegen._artifact_models import aggregate_capability

    return {
        "supports_uniform": aggregate_capability(compiled, "cpu"),
        "supports_amr": aggregate_capability(compiled, "amr"),
        "supports_mpi": aggregate_capability(compiled, "mpi"),
        "supports_gpu": aggregate_capability(compiled, "gpu"),
    }


def build_compiled_manifest(compiled):
    """Build a rich manifest from a compiled handle's inert metadata."""
    from pops.codegen._artifact_models import primary_artifact_model

    model = primary_artifact_model(compiled)
    abi_key = getattr(compiled, "abi_key", None)
    model_name = getattr(compiled, "program_name", None) or getattr(model, "name", None)

    args = compiled.arguments() if hasattr(compiled, "arguments") else None
    if args is not None:
        blocks = sorted(args.instances)
        aux_required = sorted(args.aux)
        params_const = sorted(
            name for name, slot in args.params.items() if slot.get("kind") == "const")
        params_runtime = sorted(
            name for name, slot in args.params.items() if slot.get("kind") == "runtime")
        params_derived = sorted(
            name for name, slot in args.params.items() if slot.get("kind") == "derived")
        field_plans = getattr(getattr(compiled, "plan", None), "field_plans", {}) or {}
        field_outputs = sorted(set(field_plans) | set(args.outputs))
        ghost_depth = args.layout_runtime.get("ghost_depth")
        ghost_depth_by_block = dict(
            args.layout_runtime.get("ghost_depth_by_block") or {})
    else:
        blocks = aux_required = params_const = params_runtime = params_derived = field_outputs = []
        ghost_depth = None
        ghost_depth_by_block = {}

    variables = list(getattr(model, "cons_names", []) or [])
    raw_roles = getattr(model, "cons_roles", None)
    roles = list(raw_roles) if raw_roles else None
    external_bricks = list(getattr(compiled, "external_bricks", []) or [])

    bind_schema = getattr(compiled, "bind_schema", None)
    bind_schema_data = bind_schema.to_dict() if bind_schema is not None else None
    bind_schema_hash = bind_schema.hash if bind_schema is not None else None
    bind_schema_artifact_hash = (
        bind_schema.artifact_hash if bind_schema is not None else None)

    caps_flags = _caps_flags(compiled)
    return CompiledArtifactManifest(
        model_name=model_name,
        abi_key=abi_key,
        abi_version=None,
        required_headers_sig=_headers_sig(abi_key),
        blocks=blocks,
        variables=variables,
        roles=roles,
        aux_required=aux_required,
        params_const=params_const,
        params_runtime=params_runtime,
        params_derived=params_derived,
        bind_schema=bind_schema_data,
        bind_schema_hash=bind_schema_hash,
        bind_schema_artifact_hash=bind_schema_artifact_hash,
        ghost_depth=ghost_depth,
        ghost_depth_by_block=ghost_depth_by_block,
        field_outputs=field_outputs,
        supports_stride=None,
        supports_partial_imex_mask=None,
        supports_named_fields=None,
        native_entrypoints=[],
        external_bricks=external_bricks,
        dimension=None,
        amr_refinement_ratio=None,
        precision=None,
        real_bytes=None,
        communicator=None,
        supports_custom_communicator=None,
        **caps_flags,
)


_NATIVE_BOOL_FIELDS = (
    "supports_stride", "supports_partial_imex_mask", "supports_named_fields",
    "supports_custom_communicator",
)
_NATIVE_LAYOUT_PLATFORM_FIELDS = (
    "supports_uniform", "supports_amr", "supports_mpi", "supports_gpu",
)


def _validate_native_manifest(native):
    """Validate and return the exact current compiled-block native contract."""
    if not isinstance(native, dict):
        raise ValueError("compiled native manifest must be a JSON object")
    expected = {
        "schema_version", "kind", "abi_version", "n_vars", "n_aux", "n_params",
        "ghost_depth", "supports_uniform", "supports_amr", "supports_mpi",
        "supports_gpu", "supports_stride", "supports_partial_imex_mask",
        "supports_named_fields", "roles", "native_entrypoints",
    }
    missing = sorted(expected - set(native))
    unknown = sorted(set(native) - expected)
    if missing or unknown:
        raise ValueError(
            "compiled native manifest fields must be exact; missing=%s unknown=%s"
            % (missing, unknown)
        )
    version = native["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 2:
        raise ValueError("compiled native manifest schema_version must be integer 2")
    if native["kind"] != "pops.compiled-block":
        raise ValueError("compiled native manifest kind must be 'pops.compiled-block'")
    for field in ("abi_version", "n_vars", "n_aux", "n_params", "ghost_depth"):
        value = native[field]
        minimum = 1 if field in ("abi_version", "n_vars") else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                "compiled native manifest %s must be integer >= %d" % (field, minimum))
    for field in _NATIVE_BOOL_FIELDS[:-1] + _NATIVE_LAYOUT_PLATFORM_FIELDS:
        if not isinstance(native[field], bool):
            raise ValueError("compiled native manifest %s must be bool" % field)
    for field in ("roles", "native_entrypoints"):
        value = native[field]
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item for item in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(
                "compiled native manifest %s must contain unique non-empty strings" % field)
    if native["roles"] and len(native["roles"]) != native["n_vars"]:
        raise ValueError("compiled native manifest roles cardinality must equal n_vars")
    return native


def apply_native_manifest(manifest, native):
    """Return a manifest completed by one exact authoritative native contract.

    Loading is deliberately not a migration seam: every current field must be emitted by the
    shared object, unknown fields are refused, and values are copied without Python coercion.
    """
    if not isinstance(manifest, CompiledArtifactManifest):
        raise TypeError("apply_native_manifest requires a CompiledArtifactManifest")
    native = _validate_native_manifest(native)
    if manifest.variables and len(manifest.variables) != native["n_vars"]:
        raise ValueError("compiled native manifest n_vars disagrees with carried variables")
    if manifest.params_runtime and len(manifest.params_runtime) != native["n_params"]:
        raise ValueError("compiled native manifest n_params disagrees with BindSchema")
    values = dict(manifest.to_dict()["payload"])
    values.pop("capability_matrix", None)
    values["abi_version"] = native["abi_version"]
    if manifest.ghost_depth is not None and manifest.ghost_depth != native["ghost_depth"]:
        raise ValueError("compiled native manifest ghost_depth disagrees with carried metadata")
    values["ghost_depth"] = native["ghost_depth"]
    for field in _NATIVE_BOOL_FIELDS[:-1]:
        values[field] = native[field]
    for field in _NATIVE_LAYOUT_PLATFORM_FIELDS:
        values[field] = native[field]
    if manifest.roles is not None and list(manifest.roles) != native["roles"]:
        raise ValueError("compiled native manifest roles disagree with carried metadata")
    values["roles"] = list(native["roles"])
    values["native_entrypoints"] = list(native["native_entrypoints"])
    return CompiledArtifactManifest(**values)


_LAYOUT_SUPPORT_FLAG = {
    "amr": "supports_amr",
    "uniform": "supports_uniform",
    "system": "supports_uniform",
}


def check_layout_supported(manifest, layout_kind):
    """Validate known-false layout support without rejecting unknown facts."""
    from pops.descriptors import Availability

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
            % (
                layout_kind,
                flag_name,
                layout_kind,
                "layout=Uniform" if flag_name == "supports_amr" else "layout=AMR",
                "compile a backend/target that emits %s=true or choose a supported layout"
                % flag_name,
            )
        )
    return Availability.yes("%s=true" % flag_name)


__all__ = ["CompiledArtifactManifest", "build_compiled_manifest", "check_layout_supported",
           "apply_native_manifest", "ARTIFACT_MANIFEST_SCHEMA_VERSION"]
