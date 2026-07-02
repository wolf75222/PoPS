"""BoundSnapshot -- the manifest of WHAT a pops.bind froze (ADC-592).

When ``pops.bind`` completes, the runtime is FROZEN: the composition it lowered onto the engine is
recorded here as a self-describing, JSON-ready, INERT snapshot manifest (ModuleManifest-style,
:mod:`pops.model.manifest`). It answers "what got bound" -- the layout, the blocks (with their
model hash + spatial tokens + time kind), the field solvers, the installed Program hash / ABI key /
cache key, the macro-step cadence, the aux + runtime-param names, and the output policy kinds -- and
carries a STABLE :attr:`snapshot_hash` (sha256 over the canonical JSON) that ties the bound identity
to the compiled artifact's cache/ABI key. ``sim.inspect()`` surfaces it (lifecycle + snapshot hash +
block/solver summary) so a bound simulation states its identity.

The snapshot is built INSIDE ``_install_compiled`` where every piece is already in hand (the install
steps touch each), then handed to ``_finalize_bind`` as the LAST act of the install. Stdlib-only
imports so this module stays import-light and buildable without the compiled ``_pops`` extension.
"""
import hashlib
import json

SCHEMA_VERSION = 1


class BoundSnapshot:
    """The inert, JSON-ready manifest of a bound composition (ADC-592).

    Frozen after construction: every field is a plain value / list / dict, so it can be serialised,
    hashed and compared without touching the live engine. The fields:

      - ``layout``: ``"system"`` (Uniform) or ``"amr_system"`` (AMR).
      - ``blocks``: ordered list of ``{name, model_hash, limiter, flux, recon, time, evolve}`` --
        the per-block identity (model hash or ``None`` when the model records none; the spatial
        tokens as plain strings; the time policy kind).
      - ``solvers``: ``{field: solver_token}`` -- the field-solver routes bound.
      - ``program_hash`` / ``abi_key`` / ``cache_key``: the compiled time Program identity (all
        ``None`` on the AMR route, which installs no whole-system Program).
      - ``cadence``: ``{substeps, stride, cfl}`` or ``None`` (no compiled cadence).
      - ``aux`` / ``params``: sorted aux field names and runtime-param names supplied at bind.
      - ``outputs``: the output / checkpoint policy kind names (empty when none).

    :meth:`to_dict` is the canonical view; :attr:`snapshot_hash` is a stable sha256 over it (so a
    Python mutation that changed the composition would change the hash -- but the compiled artifact
    is captured at compile time, so it is unaffected: cf. orchestration.compile snapshot authority).
    """

    def __init__(self, *, layout, blocks=None, solvers=None, program_hash=None, abi_key=None,
                 cache_key=None, cadence=None, aux=None, params=None, outputs=None):
        self.schema_version = SCHEMA_VERSION
        self.layout = layout
        self.blocks = [dict(b) for b in (blocks or [])]
        self.solvers = dict(solvers or {})
        self.program_hash = program_hash
        self.abi_key = abi_key
        self.cache_key = cache_key
        self.cadence = dict(cadence) if cadence is not None else None
        self.aux = sorted(aux or [])
        self.params = sorted(params or [])
        self.outputs = list(outputs or [])

    def to_dict(self):
        """A plain-dict view of the whole snapshot (JSON-ready)."""
        return {
            "schema_version": self.schema_version,
            "layout": self.layout,
            "blocks": [dict(b) for b in self.blocks],
            "solvers": dict(self.solvers),
            "program_hash": self.program_hash,
            "abi_key": self.abi_key,
            "cache_key": self.cache_key,
            "cadence": dict(self.cadence) if self.cadence is not None else None,
            "aux": list(self.aux),
            "params": list(self.params),
            "outputs": list(self.outputs),
        }

    def to_json(self, path=None, *, indent=2):
        """Serialise :meth:`to_dict` to JSON; write to @p path if given, else return the string."""
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(str(path), "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        return text

    @property
    def snapshot_hash(self):
        """A stable sha256 (64-hex) over the canonically serialised snapshot.

        Ties the bound identity to WHAT was frozen: the layout, the per-block model hashes + spatial
        tokens, the solver routes, the Program hash / abi_key / cache_key, the cadence, aux, params
        and output kinds. Integrated into inspection so a bound simulation states its identity.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def block_names(self):
        """The bound block names in order (a convenience for inspection summaries)."""
        return [b.get("name") for b in self.blocks]

    def __repr__(self):
        return ("BoundSnapshot(layout=%r, blocks=[%s], hash=%s)"
                % (self.layout, ", ".join(str(n) for n in self.block_names()),
                   self.snapshot_hash[:12]))


def _model_hash(model):
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


def _spatial_tokens(spatial):
    """The ``{limiter, flux, recon}`` plain-string tokens of a lowered spatial brick.

    Reads the canonical token attributes an ``pops.Spatial`` carries (``limiter`` / ``flux`` /
    ``recon``); a missing one is reported ``None`` rather than guessed.
    """
    return {
        "limiter": getattr(spatial, "limiter", None),
        "flux": getattr(spatial, "flux", None),
        "recon": getattr(spatial, "recon", None),
    }


def block_snapshot_entry(name, model, spatial, time, evolve=True):
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


def _program_hash(compiled):
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


def _cadence_row(cadence):
    """The ``{substeps, stride, cfl}`` snapshot row of a CompiledTime cadence (None when absent)."""
    if cadence is None:
        return None
    cfl = getattr(cadence, "cfl", None)
    return {"substeps": getattr(cadence, "substeps", None),
            "stride": getattr(cadence, "stride", None),
            "cfl": cfl if isinstance(cfl, (int, float)) else None}


def build_uniform_snapshot(engine, compiled, resolved_models, instances, solvers, cadence, aux,
                           params):
    """Assemble the Uniform :class:`BoundSnapshot` from the lowered install pieces (ADC-592).

    Reads only what ``System._install_compiled`` already resolved (block models / spatial / time /
    evolve, the solver tokens, the compiled handle's program hash / abi_key / cache_key, the cadence
    and the aux / param names) -- inert. The Uniform route carries the whole-system compiled time
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
    return BoundSnapshot(
        layout="system", blocks=blocks, solvers=solver_tokens, program_hash=_program_hash(compiled),
        abi_key=getattr(compiled, "abi_key", None), cache_key=getattr(compiled, "cache_key", None),
        cadence=_cadence_row(cadence), aux=list(aux or {}), params=list(params or {}),
        outputs=[type(p).__name__ for p in getattr(engine, "_output_policies", []) or []])


def build_amr_snapshot(instances, solvers, aux, params):
    """Assemble the AMR :class:`BoundSnapshot` from the lowered install pieces (ADC-592).

    The AMR route installs NO whole-system compiled time Program, so program_hash / abi_key /
    cache_key stay None; each block carries its OWN target='amr_system' CompiledModel, whose hash
    lands in the per-block row. @p instances / @p solvers come straight from the AMR install seam.
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
    return BoundSnapshot(layout="amr_system", blocks=blocks, solvers=solver_tokens,
                         program_hash=None, abi_key=None, cache_key=None, cadence=None,
                         aux=list(aux or {}), params=list(params or {}), outputs=[])


__all__ = ["BoundSnapshot", "block_snapshot_entry", "build_uniform_snapshot", "build_amr_snapshot",
           "SCHEMA_VERSION"]
