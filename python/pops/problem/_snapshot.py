"""pops.problem._snapshot -- the frozen ProblemSnapshot the compile cache keys on (ADC-563).

``Problem.freeze()`` returns a :class:`ProblemSnapshot`: an inert, JSON-ready, array-free capture of
the whole assembly (blocks / fields / params / aux / outputs / constraints / time / layout) with a
stable ``.hash`` (sha256 over the canonical ``to_dict``). ``pops.compile`` freezes the Problem it
compiles and folds ``snapshot.hash`` into the compile cache key, so a post-compile Problem mutation
cannot change a bound artifact -- the snapshot is the FROZEN identity the compile stream keys on.

The snapshot holds PLAIN values only: no runtime object, no numpy array, no live descriptor. A value
that is not a JSON scalar / list / dict is coerced to a stable string token (its ``name`` or repr),
so the canonical serialisation is deterministic and there is no shallow-copy escape (the snapshot is
a deep, inert copy). It is stdlib-only (``hashlib`` / ``json``); it imports no ``_pops`` / runtime.
"""
import hashlib
import json
import re

# A default object repr carries a memory address ("<pops._pops.ModelSpec object at 0x10384f0f0>");
# the address is per-instance and would make the hash non-reproducible. We strip the "at 0x..." tail
# so a native model with no stable Python name still canonicalises to an address-free, type-stable
# token (two instances of the same ModelSpec type hash identically).
_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")

#: Bumped only when the snapshot's canonical shape changes (a field rename / removal); an additive
#: field keeps version 1 so an old hash and a new hash of the SAME assembly stay comparable.
SNAPSHOT_SCHEMA_VERSION = 1


def _canonical(value):
    """Coerce @p value to a JSON-ready, DETERMINISTIC form (a non-scalar becomes a stable token).

    A JSON scalar / ``None`` passes through; a list / tuple / dict is canonicalised element-wise
    (dict keys stringified, sorted at serialisation). A non-scalar object becomes a STABLE structural
    token -- its ``name``, else its ``options()`` / ``to_dict()`` view, else its class name -- NEVER a
    bare ``repr`` (which leaks a memory address and would make the hash non-reproducible). So two
    structurally identical assemblies produce the same hash regardless of object identity."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _ADDRESS_RE.sub("", value)  # a stringified repr may already carry an address
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    for accessor in ("options", "to_dict"):
        fn = getattr(value, accessor, None)
        if callable(fn):
            try:
                return _canonical(fn())
            except Exception:  # noqa: BLE001 -- an accessor that raises is not a stable token
                pass
    return type(value).__name__  # stable, address-free (never a bare repr with a 0x... address)


class ProblemSnapshot:
    """The frozen, JSON-ready capture of a :class:`~pops.problem.problem.Problem` (ADC-563).

    A plain inert value: :attr:`to_dict` is the canonical dict (deep, array-free, no runtime object)
    and :attr:`hash` is its stable sha256. Two snapshots of the same assembly have the same hash; a
    mutation before freeze changes it, a mutation after freeze is impossible (the Problem raises).
    ``pops.compile`` attaches it to the compiled handle (``compiled._problem_snapshot``) and folds
    :attr:`hash` into the cache key.
    """

    schema_version = SNAPSHOT_SCHEMA_VERSION

    def __init__(self, payload):
        # A deep, canonical, JSON-ready copy: no shared reference to a live registry, no runtime
        # object -- so there is no shallow-copy escape from the frozen identity.
        self._payload = _canonical(payload)
        self._hash = None

    def to_dict(self):
        """The canonical, JSON-ready dict of the frozen assembly (stamped with the schema version)."""
        out = {"schema_version": self.schema_version}
        out.update(self._payload)
        return out

    @property
    def hash(self):
        """The stable sha256 (64-hex) over the canonical ``to_dict`` (computed once, then cached)."""
        if self._hash is None:
            canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
            self._hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self._hash

    def __eq__(self, other):
        return isinstance(other, ProblemSnapshot) and self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    def __repr__(self):
        return "ProblemSnapshot(hash=%s...)" % self.hash[:12]


def build_problem_snapshot(problem):
    """Build the :class:`ProblemSnapshot` of @p problem (the frozen input to the compile cache key).

    Reads the Problem's ``to_dict`` (the array-free, registry-sourced serialisation of the whole
    assembly) and canonicalises it into a JSON-ready, deep, inert payload. It computes nothing on a
    grid and imports no ``_pops``; the resulting ``.hash`` is the stable identity the compile stream
    folds into the cache key so a mutated Problem cannot silently rebind a compiled artifact."""
    payload = problem.to_dict() if hasattr(problem, "to_dict") else {}
    return ProblemSnapshot(payload)


def freeze_compiled(problem, time, compiled):
    """Freeze the compiled Problem + time Program and fold the snapshot hash into the cache key.

    ``pops.compile``'s LAST authoring act (ADC-563): freeze the ``Problem`` (-> a stable
    ``ProblemSnapshot`` whose ``.hash`` is the frozen identity), attach it on the handle
    (``compiled._problem_snapshot``), freeze the time ``Program`` so a post-compile IR edit RAISES,
    and FOLD ``snapshot.hash`` into the handle's cache key so a mutated Problem cannot silently rebind
    a compiled artifact. A degraded / externally built handle with no writable cache-key slot keeps
    its base key (never a crash)."""
    snapshot = problem.freeze() if hasattr(problem, "freeze") else None
    compiled._problem_snapshot = snapshot
    if time is not None and hasattr(time, "freeze"):
        time.freeze()
    if snapshot is not None:
        fold_snapshot_hash(compiled, snapshot.hash)


def fold_snapshot_hash(compiled, snapshot_hash):
    """Compose @p snapshot_hash into the handle's cache key (APPEND, never replace; ADC-563).

    The compile stream (536) owns the base cache key (model / program IR / registry / platform +
    kokkos/mpi/precision tokens); this fold APPENDS the frozen ProblemSnapshot hash so the identity
    also covers the whole assembly. It composes with whatever the compile stream produced -- the
    ladder rebase keeps both. A handle exposing no writable ``_cache_key`` slot is left untouched."""
    base = getattr(compiled, "_cache_key", None)
    folded = "%s|problem_snapshot=%s" % (base, snapshot_hash) if base else \
        "problem_snapshot=%s" % snapshot_hash
    try:
        compiled._cache_key = folded
    except Exception:  # noqa: BLE001 -- a sealed / read-only handle keeps its base key (never a crash)
        pass


__all__ = ["ProblemSnapshot", "build_problem_snapshot", "freeze_compiled", "fold_snapshot_hash",
           "SNAPSHOT_SCHEMA_VERSION"]
