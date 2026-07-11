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
    from pops.runtime_environment import runtime_environment_report

    runtime_facts = runtime_environment_report()
    # Manifest compatibility fields describe what these bytes REQUIRE, not every capability the
    # abstract production route could provide under another build.  A serial native artifact must
    # therefore remain bindable to the serial runtime that produced it.
    if runtime_facts.get("mpi_compiled") is not None:
        caps_flags["supports_mpi"] = bool(runtime_facts["mpi_compiled"])
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


_NATIVE_BOOL_FIELDS = (
    "supports_stride", "supports_partial_imex_mask", "supports_named_fields",
    "supports_custom_communicator",
)
_NATIVE_LAYOUT_PLATFORM_FIELDS = (
    "supports_uniform", "supports_amr", "supports_mpi", "supports_gpu",
)


def apply_native_manifest(manifest, native):
    """Return a new immutable manifest with authoritative native facts overlaid."""
    from pops.external.artifact_manifest import CompiledArtifactManifest

    if not isinstance(manifest, CompiledArtifactManifest):
        raise TypeError("apply_native_manifest requires a CompiledArtifactManifest")
    if not native:
        return manifest
    values = manifest.to_dict()
    values.pop("schema_version", None)
    values.pop("capability_matrix", None)
    if "abi_version" in native:
        values["abi_version"] = native["abi_version"]
    if "ghost_depth" in native and manifest.ghost_depth is None:
        values["ghost_depth"] = native["ghost_depth"]
    if native.get("ghost_depth_by_block") and not manifest.ghost_depth_by_block:
        values["ghost_depth_by_block"] = dict(native["ghost_depth_by_block"])
    for field in _NATIVE_BOOL_FIELDS:
        if field in native:
            values[field] = bool(native[field])
    for field in _NATIVE_LAYOUT_PLATFORM_FIELDS:
        if field in native:
            values[field] = bool(native[field])
    for field in ("dimension", "amr_refinement_ratio", "real_bytes"):
        if field in native:
            values[field] = int(native[field])
    for field in ("precision", "communicator"):
        if field in native:
            values[field] = str(native[field])
    roles = native.get("roles")
    if roles and manifest.roles is None:
        values["roles"] = list(roles)
    entrypoints = native.get("native_entrypoints")
    if entrypoints:
        values["native_entrypoints"] = list(entrypoints)
    return CompiledArtifactManifest(**values)


def load_native_manifest(so_path):
    """Read the authoritative native manifest exported by a shared object."""
    from pops.descriptors import load_compiled_manifest

    return load_compiled_manifest(so_path)


def build_compiled_manifest_from_so(compiled, so_path):
    """Build carried metadata, then overlay the shared object's native facts."""
    return apply_native_manifest(
        build_compiled_manifest(compiled), load_native_manifest(so_path))


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
    "apply_native_manifest", "build_compiled_manifest", "build_compiled_manifest_from_so",
    "check_layout_supported", "load_native_manifest",
]
