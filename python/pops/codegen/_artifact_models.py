"""Read-only model metadata for one compiled artifact and every installed block."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactModelMetadata:
    """One block's compiled-model metadata, detached from authoring registries."""

    block_name: str | None
    model: Any
    cons_names: tuple[str, ...]
    n_vars: int
    params: dict[str, Any]
    aux_names: tuple[str, ...]
    n_aux: int
    state_space: str


def artifact_model_metadata(compiled: Any) -> tuple[ArtifactModelMetadata, ...]:
    """Return every exact compiled block in declaration order; no historical fallback."""
    from pops.codegen.compiled_artifact import CompiledSimulationArtifact

    if type(compiled) is not CompiledSimulationArtifact:
        raise TypeError("artifact_model_metadata requires a CompiledSimulationArtifact")
    return tuple(_metadata(block.name, block.model) for block in compiled.blocks)


def component_model_metadata(compiled: Any) -> tuple[ArtifactModelMetadata, ...]:
    """Metadata for an explicitly low-level compiled component, never a public artifact."""
    from pops.codegen.loader import CompiledModel, CompiledProblem

    if type(compiled) is CompiledModel:
        return (_metadata(getattr(compiled, "name", None), compiled),)
    if type(compiled) is not CompiledProblem:
        raise TypeError("component metadata requires exact CompiledModel/CompiledProblem")
    model = compiled.model
    if model is None:
        return ()
    name = compiled.program_name or getattr(model, "name", None)
    return (_metadata(name, model),)


def primary_artifact_model(compiled: Any) -> Any:
    """Return the first install-plan model, or ``None`` for a model-free artifact."""
    metadata = artifact_model_metadata(compiled)
    return metadata[0].model if metadata else None


def aggregate_model_metadata(compiled: Any) -> tuple[Any, ...]:
    """Return aggregate counts used by whole-artifact memory formulas.

    Component and anonymous-aux counts are summed across blocks. Names are retained in block order;
    the per-instance authoring surface uses :func:`artifact_model_metadata` directly.
    """
    rows = artifact_model_metadata(compiled)
    if not rows:
        return [], 0, {}, [], 0, "U"
    cons_names = [name for row in rows for name in row.cons_names]
    params = {name: value for row in rows for name, value in row.params.items()}
    aux_names = list(dict.fromkeys(name for row in rows for name in row.aux_names))
    return (
        cons_names,
        sum(row.n_vars for row in rows),
        params,
        aux_names,
        sum(row.n_aux for row in rows),
        rows[0].state_space,
    )


def aggregate_capability(compiled: Any, name: str) -> bool | None:
    """Return the intersection of one capability across all installed models.

    A multi-block artifact supports a runtime route only when every block loader does. Missing
    capability metadata remains honestly unknown instead of being fabricated as supported.
    """
    rows = artifact_model_metadata(compiled)
    if not rows:
        return None
    values = []
    for row in rows:
        caps = getattr(row.model, "caps", None)
        if not caps or name not in caps:
            return None
        values.append(bool(caps[name]))
    return all(values)


def _metadata(block_name: str | None, model: Any) -> ArtifactModelMetadata:
    cons_names = tuple(getattr(model, "cons_names", ()) or ())
    n_vars = int(getattr(model, "n_vars", len(cons_names)) or len(cons_names))
    params = dict(getattr(model, "params", {}) or {})
    aux_names = tuple(getattr(model, "aux_extra_names", ()) or ())
    n_aux = int(getattr(model, "n_aux", len(aux_names)) or len(aux_names))
    state_space = "U"
    spaces = getattr(model, "list_state_spaces", None)
    if callable(spaces):
        names = spaces()
        if names:
            state_space = names[0]
    return ArtifactModelMetadata(
        block_name=block_name,
        model=model,
        cons_names=cons_names,
        n_vars=n_vars,
        params=params,
        aux_names=aux_names,
        n_aux=n_aux,
        state_space=state_space,
    )


__all__ = [
    "ArtifactModelMetadata", "aggregate_capability", "aggregate_model_metadata",
    "artifact_model_metadata", "component_model_metadata", "primary_artifact_model",
]
