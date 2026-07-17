"""Exact model metadata for compiled artifacts, obtained through one small provider protocol."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


_METADATA_KEYS = frozenset({
    "schema_version",
    "state_spaces",
    "cons_names",
    "cons_roles",
    "n_vars",
    "params",
    "aux_names",
    "n_aux",
    "capabilities",
    "wave_speed_provider",
})

_WAVE_SPEED_PROVIDERS = frozenset({"explicit_pair", "jacobian", "pressure_derived"})


@runtime_checkable
class ArtifactModelMetadataProvider(Protocol):
    """Structural report interface implemented by compiled and low-level model providers."""

    def __pops_artifact_model_metadata__(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ArtifactModelMetadata:
    """One block's compiled-model metadata, detached from authoring registries."""

    block_name: str | None
    model: Any
    cons_names: tuple[str, ...]
    cons_roles: tuple[str, ...]
    n_vars: int
    params: dict[str, Any]
    aux_names: tuple[str, ...]
    n_aux: int
    state_space: str
    capabilities: dict[str, bool]
    wave_speed_provider: str | None


def artifact_model_metadata(compiled: Any) -> tuple[ArtifactModelMetadata, ...]:
    """Return every exact compiled block in declaration order; no historical fallback."""
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact

    if type(compiled) is not CompiledSimulationArtifact:
        raise TypeError("artifact_model_metadata requires a CompiledSimulationArtifact")
    return tuple(
        _metadata(block.name, block.model, expected_state_spaces=block.state_spaces)
        for block in compiled.blocks
    )


def component_model_metadata(compiled: Any) -> tuple[ArtifactModelMetadata, ...]:
    """Metadata for one explicitly low-level compiled component."""
    from pops.codegen.loader import CompiledModel, CompiledProblem

    if type(compiled) is CompiledModel:
        return (_metadata(None, compiled),)
    if type(compiled) is not CompiledProblem:
        raise TypeError("component metadata requires exact CompiledModel/CompiledProblem")
    model = compiled.model
    if model is None:
        return ()
    routes = compiled.program_block_routes
    if type(routes) is not tuple or len(routes) != 1:
        raise ValueError(
            "component metadata for a CompiledProblem requires exactly one program block route"
        )
    route = routes[0]
    if (
        type(route) is not tuple
        or len(route) != 2
        or type(route[0]) is not int
        or type(route[1]) is not str
        or not route[1]
    ):
        raise ValueError(
            "component metadata requires an unambiguous (index, block_name) program route"
        )
    return (_metadata(route[1], model),)


def primary_artifact_model(compiled: Any) -> Any:
    """Return the sole model, never a representative of an aggregate artifact."""
    metadata = artifact_model_metadata(compiled)
    return metadata[0].model if len(metadata) == 1 else None


def aggregate_model_metadata(compiled: Any) -> tuple[Any, ...]:
    """Return aggregate counts used by whole-artifact memory formulas."""
    rows = artifact_model_metadata(compiled)
    if not rows:
        return [], 0, {}, [], 0, "U"
    cons_names = [name for row in rows for name in row.cons_names]
    params = {}
    owners = {}
    for row in rows:
        for name, value in row.params.items():
            if name in params and params[name] != value:
                raise ValueError(
                    "compiled blocks %r and %r declare conflicting parameter metadata for %r"
                    % (owners[name], row.block_name, name))
            params[name] = value
            owners.setdefault(name, row.block_name)
    aux_names = list(dict.fromkeys(name for row in rows for name in row.aux_names))
    state_spaces = {row.state_space for row in rows}
    return (
        cons_names,
        sum(row.n_vars for row in rows),
        params,
        aux_names,
        sum(row.n_aux for row in rows),
        next(iter(state_spaces)) if len(state_spaces) == 1 else None,
    )


def aggregate_capability(compiled: Any, name: str) -> bool | None:
    """Return the proven intersection of one capability across every installed model."""
    rows = artifact_model_metadata(compiled)
    if not rows or any(name not in row.capabilities for row in rows):
        return None
    return all(row.capabilities[name] for row in rows)


def _metadata(
    block_name: str | None,
    model: Any,
    *,
    expected_state_spaces: tuple[str, ...] | None = None,
) -> ArtifactModelMetadata:
    if not isinstance(model, ArtifactModelMetadataProvider):
        raise TypeError(
            "compiled model metadata requires __pops_artifact_model_metadata__(); got %s.%s"
            % (type(model).__module__, type(model).__qualname__)
        )
    data = model.__pops_artifact_model_metadata__()
    if not isinstance(data, dict) or set(data) != _METADATA_KEYS:
        raise TypeError("artifact model metadata provider returned an unknown schema")
    if data["schema_version"] != 3:
        raise ValueError("artifact model metadata provider uses an unsupported schema")
    state_spaces = _strings(data["state_spaces"], where="state_spaces")
    if len(state_spaces) != 1:
        raise ValueError("compiled runtime model metadata requires exactly one state space")
    if expected_state_spaces is not None and state_spaces != tuple(expected_state_spaces):
        raise ValueError("compiled model metadata disagrees with the resolved state-space route")
    cons_names = _strings(data["cons_names"], where="cons_names")
    cons_roles = _strings(data["cons_roles"], where="cons_roles")
    if len(cons_roles) != len(cons_names):
        raise ValueError("compiled model cons_roles must exactly match cons_names")
    n_vars = data["n_vars"]
    if not isinstance(n_vars, int) or isinstance(n_vars, bool) or n_vars != len(cons_names):
        raise ValueError("compiled model n_vars must exactly match cons_names")
    if not isinstance(data["params"], Mapping):
        raise TypeError("compiled model params metadata must be a mapping")
    params = dict(data["params"])
    if any(not isinstance(name, str) or not name for name in params):
        raise TypeError("compiled model parameter names must be non-empty strings")
    aux_names = _strings(data["aux_names"], where="aux_names")
    n_aux = data["n_aux"]
    if not isinstance(n_aux, int) or isinstance(n_aux, bool) or n_aux < len(aux_names):
        raise ValueError("compiled model n_aux cannot be smaller than its named aux metadata")
    if not isinstance(data["capabilities"], Mapping):
        raise TypeError("compiled model capabilities metadata must be a mapping")
    capabilities = dict(data["capabilities"])
    if any(not isinstance(key, str) or not key or not isinstance(value, bool)
           for key, value in capabilities.items()):
        raise TypeError("compiled model capabilities must map non-empty strings to bool")
    wave_speed_provider = data["wave_speed_provider"]
    if wave_speed_provider is not None and wave_speed_provider not in _WAVE_SPEED_PROVIDERS:
        raise ValueError(
            "compiled model wave_speed_provider %r must be None or one of %s"
            % (wave_speed_provider, ", ".join(sorted(_WAVE_SPEED_PROVIDERS)))
        )
    return ArtifactModelMetadata(
        block_name=block_name,
        model=model,
        cons_names=cons_names,
        cons_roles=cons_roles,
        n_vars=n_vars,
        params=params,
        aux_names=aux_names,
        n_aux=n_aux,
        state_space=state_spaces[0],
        capabilities=capabilities,
        wave_speed_provider=wave_speed_provider,
    )


def _strings(value: Any, *, where: str) -> tuple[str, ...]:
    try:
        result = tuple(value)
    except TypeError:
        raise TypeError("compiled model %s must be a sequence of strings" % where) from None
    if any(not isinstance(item, str) or not item for item in result):
        raise TypeError("compiled model %s must contain non-empty strings" % where)
    return result


__all__ = [
    "ArtifactModelMetadata",
    "ArtifactModelMetadataProvider",
    "aggregate_capability",
    "aggregate_model_metadata",
    "artifact_model_metadata",
    "component_model_metadata",
    "primary_artifact_model",
]
