"""Structured identity helpers for compiled problem artifacts."""

import hashlib
import json

from pops.codegen.toolchain import _native_feature_key

GENERATED_SOURCE_IDENTITY_VERSION = "pops-generated-source-v1"


def problem_target_from_layout(layout):
    """Return the native problem ABI target selected by a typed mesh layout."""
    if layout is None:
        return "system"
    from pops.mesh.layouts import AMR, Uniform
    if isinstance(layout, AMR):
        return "amr_system"
    if isinstance(layout, Uniform):
        return "system"
    raise TypeError(
        "compile_problem: layout must be a typed pops.mesh.layouts.Uniform(...) or AMR(...) "
        "descriptor; got %r" % type(layout).__name__)


def stable_identity_value(value):
    """JSON-stable, side-effect-free representation for problem identity records."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [stable_identity_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): stable_identity_value(value[k]) for k in sorted(value, key=str)}
    if hasattr(value, "inspect") and callable(value.inspect):
        return stable_identity_value(value.inspect())
    if hasattr(value, "options") and callable(value.options):
        return {
            "type": type(value).__name__,
            "category": getattr(value, "category", None),
            "options": stable_identity_value(value.options()),
        }
    raise TypeError(
        "compiled problem identity cannot serialize %s: route descriptors must expose "
        "inspect() or options(), and identity values must be JSON primitives, lists or dicts; "
        "repr() is intentionally rejected because it is not a stable identity"
        % type(value).__name__)


def library_identity(manifests):
    out = []
    for manifest in manifests or []:
        if hasattr(manifest, "to_dict") and callable(manifest.to_dict):
            out.append(stable_identity_value(manifest.to_dict()))
        elif hasattr(manifest, "as_dict") and callable(manifest.as_dict):
            out.append(stable_identity_value(manifest.as_dict()))
        else:
            out.append(stable_identity_value(manifest))
    return out


def semantic_problem_hash(record):
    """Digest only the semantic part of a compiled problem identity."""
    blob = json.dumps(record["semantic"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compiled_problem_cache_key(record):
    """Digest the full binary cache identity: semantic + provenance + generated-source guard."""
    blob = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compiled_problem_identity(*, source, model, program, layout, backend, target,
                              include, compiler, std, abi_key, optflags,
                              library_manifests,
                              codegen_version=GENERATED_SOURCE_IDENTITY_VERSION):
    """Structured identity of the combined problem artifact."""
    source_identity = {"version": str(codegen_version), "source": source}
    source_hash = hashlib.sha256(
        json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    module_hash = model.module_hash() if hasattr(model, "module_hash") else None
    program_hash = program._ir_hash() if hasattr(program, "_ir_hash") else None
    record = {
        "schema": "pops-compiled-problem-v1",
        "semantic": {
            "name": getattr(program, "name", "problem"),
            "module": {"name": getattr(model, "name", None), "hash": module_hash},
            "program": {"name": getattr(program, "name", None), "hash": program_hash},
            "descriptors": {
                "layout": stable_identity_value(layout),
                "backend": stable_identity_value(backend),
                "libraries": library_identity(library_manifests),
            },
            "toolchain": {
                "compiler": compiler,
                "std": std,
                "abi_key": abi_key,
                "native_features": _native_feature_key(),
                "optflags": list(optflags),
            },
            "runtime_route": {"target": target},
        },
        "provenance": {
            "include": include,
            "compiler": compiler,
            "std": std,
            "abi_key": abi_key,
            "native_features": _native_feature_key(),
            "optflags": list(optflags),
        },
        "generated_source": {
            "version": source_identity["version"],
            "hash": source_hash,
            "language": "c++",
        },
    }
    problem_hash = semantic_problem_hash(record)
    return record, problem_hash, module_hash, program_hash, source_hash


__all__ = [
    "GENERATED_SOURCE_IDENTITY_VERSION",
    "compiled_problem_cache_key",
    "compiled_problem_identity",
    "library_identity",
    "problem_target_from_layout",
    "semantic_problem_hash",
    "stable_identity_value",
]
