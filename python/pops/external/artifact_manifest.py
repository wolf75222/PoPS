"""Rich, inert self-description of a compiled artifact (Spec 5 sec.13.12).

Metadata comes from the compiled handle's public surface. Missing native facts remain ``None``
(unknown), and imports stay lazy so this module remains at the bottom of the import graph.
"""
from __future__ import annotations

from typing import Any

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
ARTIFACT_MANIFEST_SCHEMA_VERSION = 1


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
                 supports_named_fields=None, native_entrypoints=None, dimension=2,
                 amr_refinement_ratio=2, precision="double", real_bytes=8,
                 communicator="unknown", supports_custom_communicator=False,
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
        put(self, "dimension", freeze_manifest_json(dimension, where="artifact.dimension"))
        put(
            self,
            "amr_refinement_ratio",
            freeze_manifest_json(amr_refinement_ratio, where="artifact.amr_refinement_ratio"),
        )
        put(self, "precision", freeze_manifest_json(precision, where="artifact.precision"))
        put(self, "real_bytes", freeze_manifest_json(real_bytes, where="artifact.real_bytes"))
        put(
            self,
            "communicator",
            freeze_manifest_json(communicator, where="artifact.communicator"),
        )
        put(self, "supports_custom_communicator", bool(supports_custom_communicator))

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


from pops.external._artifact_manifest_ops import (  # noqa: E402
    apply_native_manifest,
    build_compiled_manifest,
    build_compiled_manifest_from_so,
    check_layout_supported,
    load_native_manifest,
)


__all__ = ["CompiledArtifactManifest", "build_compiled_manifest", "check_layout_supported",
           "apply_native_manifest", "load_native_manifest", "build_compiled_manifest_from_so",
           "ARTIFACT_MANIFEST_SCHEMA_VERSION"]
