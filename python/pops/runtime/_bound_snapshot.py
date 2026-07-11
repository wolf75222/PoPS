"""BoundSnapshot -- the manifest of WHAT a pops.bind froze (ADC-592).

When ``pops.bind`` completes, the runtime is FROZEN: the composition it lowered onto the engine is
recorded here as a self-describing, JSON-ready, INERT snapshot manifest (ModuleManifest-style,
:mod:`pops.model.manifest`). It answers "what got bound" -- the layout, the blocks (with their
model hash + spatial tokens + time kind), the field solvers, the installed Program hash / ABI key /
cache key, cadence, aux inputs, effective typed parameter rows, BindSchema identities and outputs -- and
carries a STABLE :attr:`snapshot_hash` (sha256 over the canonical JSON) that ties the bound identity
to the compiled artifact's cache/ABI key. ``sim.inspect()`` surfaces it (lifecycle + snapshot hash +
block/solver summary) so a bound simulation states its identity.

The snapshot is built INSIDE ``_install_compiled`` where every piece is already in hand (the install
steps touch each), then handed to ``_finalize_bind`` as the LAST act of the install. Stdlib-only
imports so this module stays import-light and buildable without the compiled ``_pops`` extension.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

from typing import Any

SCHEMA_VERSION = 2


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError("BoundSnapshot contains non-JSON value %r" % type(value).__name__)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class BoundSnapshot:
    """The inert, JSON-ready manifest of a bound composition (ADC-592).

    Deeply frozen after construction and serialised through detached plain containers.

      - ``layout``: ``"system"`` (Uniform) or ``"amr_system"`` (AMR).
      - ``blocks``: ordered list of ``{name, model_hash, limiter, flux, recon, time, evolve}`` --
        the per-block identity (model hash or ``None`` when the model records none; the spatial
        tokens as plain strings; the time policy kind).
      - ``solvers``: ``{field: solver_token}`` -- the field-solver routes bound.
      - ``program_hash`` / ``abi_key`` / ``cache_key``: compiled Program identity when installed.
      - ``cadence``: ``{substeps, stride, cfl}`` or ``None`` (no compiled cadence).
      - ``aux``: sorted auxiliary inputs.
      - ``params``: typed effective rows with QID, value, provenance and materialization source.
      - ``bind_schema_hash`` / ``bind_schema_artifact_hash``: full and reusable-plan identities.
      - ``outputs``: the output / checkpoint policy kind names (empty when none).

    :meth:`to_dict` is the canonical view; :attr:`snapshot_hash` is a stable sha256 over it (so a
    Python mutation that changed the composition would change the hash -- but the compiled artifact
    is captured at compile time, so it is unaffected: cf. orchestration.compile snapshot authority).
    """

    __slots__ = (
        "schema_version", "layout", "blocks", "solvers", "program_hash", "abi_key",
        "cache_key", "cadence", "aux", "params", "bind_schema_hash",
        "bind_schema_artifact_hash", "outputs",
    )

    def __init__(self, *, layout: Any, blocks: Any = None, solvers: Any = None,
                 program_hash: Any = None, abi_key: Any = None, cache_key: Any = None,
                 cadence: Any = None, aux: Any = None, params: Any = None,
                 bind_schema_hash: Any = None, bind_schema_artifact_hash: Any = None,
                 outputs: Any = None) -> None:
        rows = list(params or ())
        if any(not isinstance(row, Mapping) or not isinstance(row.get("qid"), str)
               for row in rows):
            raise TypeError("BoundSnapshot params must be resolved binding rows with qid")
        if len({row["qid"] for row in rows}) != len(rows):
            raise ValueError("BoundSnapshot params contain duplicate qualified IDs")
        for name, value in (("bind_schema_hash", bind_schema_hash),
                            ("bind_schema_artifact_hash", bind_schema_artifact_hash)):
            if value is not None and (
                not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("BoundSnapshot %s must be a sha256 hex string or None" % name)
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "blocks", _freeze(list(blocks or ())))
        object.__setattr__(self, "solvers", _freeze(dict(solvers or {})))
        object.__setattr__(self, "program_hash", program_hash)
        object.__setattr__(self, "abi_key", abi_key)
        object.__setattr__(self, "cache_key", cache_key)
        object.__setattr__(self, "cadence", _freeze(cadence))
        object.__setattr__(self, "aux", tuple(sorted(aux or ())))
        object.__setattr__(self, "params", _freeze(rows))
        object.__setattr__(self, "bind_schema_hash", bind_schema_hash)
        object.__setattr__(self, "bind_schema_artifact_hash", bind_schema_artifact_hash)
        object.__setattr__(self, "outputs", _freeze(list(outputs or ())))

    def to_dict(self) -> Any:
        """A plain-dict view of the whole snapshot (JSON-ready)."""
        return {
            "schema_version": self.schema_version,
            "layout": self.layout,
            "blocks": _thaw(self.blocks),
            "solvers": _thaw(self.solvers),
            "program_hash": self.program_hash,
            "abi_key": self.abi_key,
            "cache_key": self.cache_key,
            "cadence": _thaw(self.cadence),
            "aux": list(self.aux),
            "params": _thaw(self.params),
            "bind_schema_hash": self.bind_schema_hash,
            "bind_schema_artifact_hash": self.bind_schema_artifact_hash,
            "outputs": _thaw(self.outputs),
        }

    def to_json(self, path: Any = None, *, indent: int = 2) -> Any:
        """Serialise :meth:`to_dict` to JSON; write to @p path if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @property
    def snapshot_hash(self) -> Any:
        """A stable sha256 (64-hex) over the canonically serialised snapshot.

        Ties the bound identity to WHAT was frozen: the layout, the per-block model hashes + spatial
        tokens, the solver routes, the Program hash / abi_key / cache_key, the cadence, aux, params
        and output kinds. Integrated into inspection so a bound simulation states its identity.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def block_names(self) -> Any:
        """The bound block names in order (a convenience for inspection summaries)."""
        return [b.get("name") for b in self.blocks]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("BoundSnapshot is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("BoundSnapshot is immutable")

    def __repr__(self) -> Any:
        return ("BoundSnapshot(layout=%r, blocks=[%s], hash=%s)"
                % (self.layout, ", ".join(str(n) for n in self.block_names()),
                   self.snapshot_hash[:12]))


def _model_hash(model: Any) -> Any:
    """The stable hash of an instance's block model, or ``None`` when it records none.

    A CompiledModel exposes ``model_hash`` / ``hash``; a native ModelSpec / physics Model records
    none, so the field is honestly ``None`` (never fabricated). Read-only introspection.
    """
    for attr in ("model_hash", "hash"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
        if callable(value):
            try:
                out = value()
                if isinstance(out, str) and out:
                    return out
            except Exception:  # noqa: BLE001 -- a hash read must never break the bind
                pass
    return None


def _spatial_tokens(spatial: Any) -> Any:
    """The ``{limiter, flux, recon}`` plain-string tokens of a lowered spatial brick.

    Reads the canonical token attributes an ``pops.Spatial`` carries (``limiter`` / ``flux`` /
    ``recon``); a missing one is reported ``None`` rather than guessed.
    """
    return {
        "limiter": getattr(spatial, "limiter", None),
        "flux": getattr(spatial, "flux", None),
        "recon": getattr(spatial, "recon", None),
    }


def block_snapshot_entry(name: Any, model: Any, spatial: Any, time: Any,
                         evolve: Any = True) -> Any:
    """One :class:`BoundSnapshot` block row from the install-time pieces (helper for the seam).

    @p model / @p spatial are the RESOLVED objects the install already built; @p time is the block's
    time policy (its ``kind`` token is recorded). Returns a plain dict (inert).
    """
    tokens = _spatial_tokens(spatial)
    return {
        "name": name,
        "model_hash": _model_hash(model),
        "limiter": tokens["limiter"],
        "flux": tokens["flux"],
        "recon": tokens["recon"],
        "time": getattr(time, "kind", None) if time is not None else None,
        "evolve": bool(evolve),
    }


def _program_hash(compiled: Any) -> Any:
    """The compiled Program's IR hash for the snapshot (None on a native install)."""
    if compiled is None:
        return None
    fn = getattr(compiled, "program_hash", None)
    if callable(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 -- a hash read must never break the bind
            return None
    return fn if isinstance(fn, str) else getattr(compiled, "hash", None)


def _cadence_row(cadence: Any) -> Any:
    """The ``{substeps, stride, cfl}`` snapshot row of a CompiledTime cadence (None when absent)."""
    if cadence is None:
        return None
    cfl = getattr(cadence, "cfl", None)
    if cfl not in (None, "default"):
        from pops.solvers._numeric import native_float
        cfl = native_float(cfl, where="CompiledTime cfl snapshot")
    else:
        cfl = None
    return {"substeps": getattr(cadence, "substeps", None),
            "stride": getattr(cadence, "stride", None),
            "cfl": cfl}


def _binding_snapshot(params: Any) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Effective typed rows plus the full/artifact BindSchema identities."""
    rows = getattr(params, "rows", None)
    schema = getattr(params, "schema", None)
    if params is None or (not params and (not callable(rows) or schema is None)):
        return [], None, None
    if not callable(rows) or schema is None:
        raise TypeError("BoundSnapshot parameters must be a ResolvedBindings value")
    return list(rows()), schema.hash, schema.artifact_hash


def build_uniform_snapshot(engine: Any, compiled: Any, resolved_models: Any, instances: Any,
                           solvers: Any, cadence: Any, aux: Any, params: Any) -> Any:
    """Assemble the Uniform :class:`BoundSnapshot` from the lowered install pieces (ADC-592).

    Reads only what ``System._install_compiled`` already resolved (block models / spatial / time /
    evolve, the solver tokens, the compiled handle's program hash / abi_key / cache_key, the cadence
    and the aux / effective parameter rows) -- inert. The Uniform route carries the whole-system compiled time
    Program, so program_hash / abi_key / cache_key come from the handle; a native install
    (compiled=None) leaves them None. @p engine supplies the spatial lowering + solver-token helpers
    (already on the install mixin) and the stored output policies.
    """
    blocks = []
    for name, spec in (instances or {}).items():
        model = resolved_models.get(name, spec.get("model"))
        spatial = engine._lower_spatial(spec.get("spatial"))
        blocks.append(block_snapshot_entry(name, model, spatial, spec.get("time"),
                                           spec.get("evolve", True)))
    solver_tokens = {field: engine._solver_token(brick)
                     for field, brick in (solvers or {}).items()}
    param_rows, schema_hash, schema_artifact_hash = _binding_snapshot(params)
    return BoundSnapshot(
        layout="system", blocks=blocks, solvers=solver_tokens, program_hash=_program_hash(compiled),
        abi_key=getattr(compiled, "abi_key", None), cache_key=getattr(compiled, "cache_key", None),
        cadence=_cadence_row(cadence), aux=list(aux or {}), params=param_rows,
        bind_schema_hash=schema_hash, bind_schema_artifact_hash=schema_artifact_hash,
        outputs=[_output_policy_row(p) for p in getattr(engine, "_output_policies", []) or []])


def build_amr_snapshot(engine: Any, compiled: Any, instances: Any, solvers: Any,
                       cadence: Any, aux: Any, params: Any) -> Any:
    """Assemble the AMR :class:`BoundSnapshot` from the lowered install pieces (ADC-592).

    When AMR installs a whole-system Program its program/cache/ABI identity and cadence are retained;
    otherwise those fields are absent while each block's CompiledModel hash remains in its row.
    """
    from pops.runtime.bricks import Spatial
    blocks = []
    for name, spec in (instances or {}).items():
        spatial = spec.get("spatial")
        spatial = spatial if isinstance(spatial, Spatial) else Spatial()
        blocks.append(block_snapshot_entry(name, spec.get("model"), spatial, spec.get("time"),
                                           spec.get("evolve", True)))
    solver_tokens = {}
    for field, brick in (solvers or {}).items():
        solver_tokens[field] = brick if isinstance(brick, str) else (
            getattr(brick, "scheme", None) or getattr(brick, "name", None))
    param_rows, schema_hash, schema_artifact_hash = _binding_snapshot(params)
    return BoundSnapshot(layout="amr_system", blocks=blocks, solvers=solver_tokens,
                         program_hash=_program_hash(compiled),
                         abi_key=getattr(compiled, "abi_key", None),
                         cache_key=getattr(compiled, "cache_key", None),
                         cadence=_cadence_row(cadence), aux=list(aux or {}), params=param_rows,
                         bind_schema_hash=schema_hash,
                         bind_schema_artifact_hash=schema_artifact_hash,
                         outputs=[_output_policy_row(p) for p in
                                  getattr(engine, "_output_policies", []) or []])


def _output_policy_row(policy: Any) -> Any:
    """The typed ``{name, category, options}`` row of an output / checkpoint policy (ADC-562 / G9).

    Enriches the snapshot's ``outputs`` from a bare type name to the policy's typed options
    (format / cadence / levels / prefix). Each option value is coerced JSON-ready (a non-scalar such
    as a Schedule cadence becomes its name token) so the snapshot round-trips through JSON. A policy
    exposing no ``options`` degrades to its class name so an unusual entry never breaks it."""
    opts = getattr(policy, "options", None)
    raw: Any = opts() if callable(opts) else {}
    return {"name": getattr(policy, "name", type(policy).__name__),
            "category": getattr(policy, "category", None),
            "options": {k: _jsonable_option(v) for k, v in raw.items()}}


def _jsonable_option(value: Any) -> Any:
    """Coerce an option value to a JSON-ready form (a non-scalar, e.g. a Schedule, becomes a token)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_option(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_option(v) for k, v in value.items()}
    return getattr(value, "name", None) or repr(value)


__all__ = ["BoundSnapshot", "block_snapshot_entry", "build_uniform_snapshot", "build_amr_snapshot",
           "SCHEMA_VERSION"]
