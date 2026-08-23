"""Canonical artifact-spec construction at the codegen boundary."""
from __future__ import annotations

import hashlib
from typing import Any


def model_artifact_spec(
    model: Any, *, backend: str, target: str, name: Any, compiler: str, standard: str,
    abi_key: str, hoist_reciprocals: bool, consumer_owner_qid: Any = None,
    declare_auxiliary_providers: bool = True, sealed_system_routes: Any = None,
) -> tuple[Any, Any]:
    """Return semantic and artifact-spec identities for one formula model."""
    from pops.codegen.cache import (
        _dsl_optflags, _platform_cache_key, _precision_cache_key, _registry_cache_key,
    )
    from pops.codegen._compile_emit import (
        NATIVE_SYSTEM_PACKAGE_ABI_EXPORT,
        NATIVE_SYSTEM_PACKAGE_ABI_VERSION,
        model_hash,
    )
    from pops.codegen.toolchain import _native_feature_key
    from pops.identity import artifact_spec_identity, make_identity

    digest = str(model_hash(model))
    wave_speeds = getattr(model, "_ws_jacobian", {}) or {}
    semantic = make_identity(
        "semantic", {"kind": "model", "model_digest": digest})
    spec = artifact_spec_identity(
        semantic,
        target=target,
        backend=backend,
        precision=_precision_cache_key(),
        abi=abi_key,
        toolchain="%s|%s" % (compiler, standard),
        routes={
            "registry": _registry_cache_key(),
            "features": _native_feature_key(),
        },
        components={
            "model_hash": digest,
            "emitted_name": str(name or ""),
            "consumer_owner_qid": str(consumer_owner_qid or ""),
            "declares_auxiliary_providers": bool(declare_auxiliary_providers),
            "sealed_system_routes": (
                None
                if sealed_system_routes is None
                else tuple(str(route) for route in sealed_system_routes)
            ),
            # Host add_native_block looks up this exact export.  Changing the
            # emitted loader surface without this component reuses a stale .so.
            "native_system_package_abi_version": NATIVE_SYSTEM_PACKAGE_ABI_VERSION,
            "native_system_package_abi_export": NATIVE_SYSTEM_PACKAGE_ABI_EXPORT,
            "eig_max_iter": wave_speeds.get("eig_max_iter"),
            "im_tol": (
                None if wave_speeds.get("im_tol") is None
                else float(wave_speeds["im_tol"]).hex()
            ),
        },
        flags=[_platform_cache_key(), *_dsl_optflags(),
               "hoist_reciprocals=%d" % bool(hoist_reciprocals)],
        libraries=(),
    )
    return semantic, spec


def program_artifact_spec(
    *, snapshot: Any, model_authority: Any, program: Any, program_graph: Any,
    target: str, abi_key: str,
    compiler: str, standard: str, source: str, cflags: Any, lflags: Any, optflags: Any,
    libraries: Any, native_components: Any = (), native_dimension: int,
) -> tuple[Any, Any]:
    """Return semantic and artifact-spec identities for one compiled Program."""
    from pops.codegen.cache import _precision_cache_key, _registry_cache_key
    from pops.codegen.toolchain import _native_feature_key
    from pops.identity import artifact_spec_identity
    from pops.identity.semantic import (
        model_semantic_data, program_semantic_data, semantic_identity, semantic_identity_of,
    )

    if snapshot is not None:
        semantic = semantic_identity_of(snapshot=snapshot)
    else:
        from pops.codegen.program_models import ProgramModelGraph

        if type(model_authority) is ProgramModelGraph:
            semantic_model = {
                "kind": "program-model-graph",
                "models": [
                    {
                        "owner": str(owner),
                        "model": model_semantic_data(source),
                    }
                    for owner, source in sorted(
                        model_authority.source_modules_by_owner.items(),
                        key=lambda item: str(item[0]),
                    )
                ],
                "blocks": [
                    {"block": block, "owner": str(owner)}
                    for block, owner in sorted(model_authority.owners_by_block.items())
                ],
            }
        else:
            semantic_model = (
                {"kind": "program-only"}
                if model_authority is None else model_semantic_data(model_authority)
            )
        semantic = semantic_identity({
            "model": semantic_model,
            "program": program_semantic_data(program),
        })
    spec = artifact_spec_identity(
        semantic,
        target=target,
        backend="production",
        precision=_precision_cache_key(),
        abi=abi_key,
        toolchain="%s|%s" % (compiler, standard),
        routes={"registry": _registry_cache_key(), "features": _native_feature_key()},
        components={
            "generated_source": hashlib.sha256(source.encode("utf-8")).digest(),
            "native_dependency_contract": "compiler-observed-v1",
            "program_entry": str(getattr(program, "name", "problem")),
            "program_graph_hash": program_graph.graph_hash,
            "prepared_native_components": list(native_components),
            "native_dimension": native_dimension,
        },
        flags=[*optflags, *cflags, *lflags],
        libraries=[manifest.content_hash for manifest in libraries],
    )
    return semantic, spec


__all__ = ["model_artifact_spec", "program_artifact_spec"]
