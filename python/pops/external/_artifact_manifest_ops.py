"""Construction, native overlay and layout checks for artifact manifests."""
from __future__ import annotations


_SUPPORTS_FROM_CAPS = ("supports_uniform", "supports_amr", "supports_mpi", "supports_gpu")


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
    from pops.external.artifact_manifest import CompiledArtifactManifest

    from pops.codegen._artifact_models import primary_artifact_model

    model = primary_artifact_model(compiled)
    abi_key = getattr(compiled, "abi_key", None)
    model_name = getattr(compiled, "program_name", None) or getattr(model, "name", None)

    args = compiled.arguments() if hasattr(compiled, "arguments") else None
    if args is not None:
        blocks = sorted(args.instances)
        aux_required = sorted(args.aux)
        params_const = sorted(n for n, slot in args.params.items()
                              if slot.get("kind") == "const")
        params_runtime = sorted(n for n, slot in args.params.items()
                                if slot.get("kind") == "runtime")
        params_derived = sorted(n for n, slot in args.params.items()
                                if slot.get("kind") == "derived")
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
    external_bricks = list(getattr(compiled, "external_bricks", []) or [])

    bind_schema = getattr(compiled, "bind_schema", None)
    bind_schema_data = bind_schema.to_dict() if bind_schema is not None else None
    bind_schema_hash = bind_schema.hash if bind_schema is not None else None
    bind_schema_artifact_hash = bind_schema.artifact_hash if bind_schema is not None else None

    caps_flags = _caps_flags(compiled)
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
        dimension=None, amr_refinement_ratio=None, precision=None, real_bytes=None,
        communicator=None, supports_custom_communicator=None,
        **caps_flags)


_NATIVE_BOOL_FIELDS = (
    "supports_stride", "supports_partial_imex_mask", "supports_named_fields",
    "supports_custom_communicator",
)
_NATIVE_LAYOUT_PLATFORM_FIELDS = (
    "supports_uniform", "supports_amr", "supports_mpi", "supports_gpu",
)


def validate_native_manifest(native):
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
            raise ValueError("compiled native manifest %s must be integer >= %d"
                             % (field, minimum))
    for field in _NATIVE_BOOL_FIELDS[:-1] + _NATIVE_LAYOUT_PLATFORM_FIELDS:
        if not isinstance(native[field], bool):
            raise ValueError("compiled native manifest %s must be bool" % field)
    for field in ("roles", "native_entrypoints"):
        value = native[field]
        if (not isinstance(value, list)
                or any(not isinstance(item, str) or not item for item in value)
                or len(value) != len(set(value))):
            raise ValueError(
                "compiled native manifest %s must contain unique non-empty strings" % field
            )
    if native["roles"] and len(native["roles"]) != native["n_vars"]:
        raise ValueError("compiled native manifest roles cardinality must equal n_vars")
    return native


def apply_native_manifest(manifest, native):
    """Return a manifest completed by one exact authoritative native contract.

    Loading is deliberately not a migration seam: every current field must be
    emitted by the shared object, unknown fields are refused, and values are
    copied without Python coercion.
    """
    from pops.external.artifact_manifest import CompiledArtifactManifest

    if not isinstance(manifest, CompiledArtifactManifest):
        raise TypeError("apply_native_manifest requires a CompiledArtifactManifest")
    native = validate_native_manifest(native)
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
    "amr": "supports_amr", "uniform": "supports_uniform", "system": "supports_uniform",
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
            % (layout_kind, flag_name, layout_kind,
               "layout=Uniform" if flag_name == "supports_amr" else "layout=AMR",
               "compile a backend/target that emits %s=true or choose a supported layout"
               % flag_name))
    return Availability.yes("%s=true" % flag_name)


__all__ = [
    "apply_native_manifest", "build_compiled_manifest", "check_layout_supported",
    "validate_native_manifest",
]
