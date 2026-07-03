"""pops.time history-persistence policies (ADC-626).

How a depth-d history ring (the buffer behind ``T.prev(lag)``) is PERSISTED inside a
checkpoint: which of the d slots are STORED verbatim and which are RECOMPUTED at restart by
deterministic replay of the installed compiled Program. This is a DIFFERENT axis from
:class:`pops.output.CheckpointPolicy` (which governs WHEN a checkpoint is written); to avoid
the category collision at :mod:`pops.problem.registries` the descriptors here declare the
category ``"history_persistence"``.

The three policies subclass :class:`pops.descriptors.Descriptor` (they inspect / freeze /
serialize like every route object) and share ONE generic primitive,
:meth:`HistoryPersistence.stored_slots`, that the checkpoint writer, the manifest reader and
the schedule all consume. Slot 0 is the NEWEST value, slot d-1 the OLDEST.

  - :class:`Dense`         stores every slot (today's implicit behaviour; the default).
  - :class:`Interval` (k)  stores slot 0 + every k-th older slot, recomputing the gaps.
  - :class:`Revolve` (s)   stores s slots placed to minimise the worst-case replay latency.

A policy is INERT: it declares which slots are stored and computes nothing. The replay that
reconstructs the recomputed slots is the native ``System::rebuild_history_slots`` seam
(ADC-626), driven by the checkpoint reader at restart.

SCHEDULE (see :func:`_optimal_placement`): ring reconstruction fills each gap between adjacent
stored anchors ONCE by re-stepping the Program forward, so the TOTAL replay work is
placement-invariant (``d - s`` steps). The only objective is the worst-case latency = the
largest gap, minimised by EQUISPACED anchors with BOTH endpoints forced. Because replay
re-steps forward in time (toward slot 0) the OLDEST slot d-1 has nothing older to replay from
and MUST be a stored anchor -- so ``Revolve`` forces ``{0, d-1}`` and ``Interval`` is refused
when its stride misses the oldest slot. This is the EXACT min-max-gap optimum, not the classic
Griewank-Walsh adjoint Revolve (which optimises a different objective); the name is kept
because the policy is budget-bounded storage, but the schedule is documented as equispaced.
"""
from pops.descriptors import Descriptor

#: The category every history-persistence descriptor declares. Distinct from the
#: ``"checkpoint_policy"`` category of :class:`pops.output.CheckpointPolicy` (a different axis),
#: so a runtime-policy registry never confuses the two (ADC-626).
HISTORY_PERSISTENCE_CATEGORY = "history_persistence"


class HistoryPersistence(Descriptor):
    """Base of the history-persistence policies (ADC-626): how a depth-d ring is checkpointed.

    Subclasses declare which of the d ring slots are STORED verbatim via
    :meth:`stored_slots`; the remaining slots are RECOMPUTED at restart by deterministic replay
    of the installed Program. The base is inert -- it computes nothing and holds no runtime data.
    """

    category = HISTORY_PERSISTENCE_CATEGORY
    #: The manifest tag that identifies this policy in a checkpoint. Subclasses set it; it is the
    #: reader's dispatch key (:meth:`from_manifest`), NOT a free string sniffed from other fields.
    kind = "history_persistence"

    def stored_slots(self, depth):
        """The sorted tuple of slot indices STORED verbatim for a ring of @p depth (a subset of
        ``range(depth)``; slot 0 = newest). The generic primitive the writer / reader / schedule
        consume. Subclasses override; the base refuses (an abstract policy stores nothing)."""
        raise NotImplementedError(
            "%s.stored_slots is abstract; use Dense / Interval / Revolve" % (self.name,))

    def recomputed_slots(self, depth):
        """The slots NOT stored (reconstructed by replay), sorted ascending. Derived generically
        from :meth:`stored_slots` so a subclass never re-derives it."""
        stored = set(self.stored_slots(int(depth)))
        return tuple(s for s in range(int(depth)) if s not in stored)

    def _check_depth(self, depth):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError(
                "%s: ring depth must be a Python int >= 1 (got %r)" % (self.name, depth))
        return depth

    def validate_for(self, depth):
        """Refuse a policy incoherent with the ring @p depth, LOUD (a ``ValueError``), at author /
        compile time -- never at restart. A depth-1 ring degenerates to Dense for every policy
        (its single slot is always stored). Subclasses add their own coherence rules and MUST call
        this base check first. Returns @p depth (validated) so callers can chain."""
        return self._check_depth(depth)

    def degenerate_to_dense(self, depth):
        """True when the policy stores EVERY slot at this @p depth (no recomputation): a depth-1
        ring, or a budget >= depth. Reported by :meth:`inspect` for transparency."""
        depth = self._check_depth(depth)
        return len(self.stored_slots(depth)) == depth

    # --- serialization: the tagged manifest dict + the reader's dispatch ------------------------
    def to_manifest(self):
        """The small tagged dict the checkpoint carries VERBATIM for this policy (schema-driven).
        The ``"kind"`` tag is the policy identity; subclasses add their scalar knobs."""
        return {"kind": self.kind}

    @staticmethod
    def from_manifest(manifest):
        """Rebuild the policy a checkpoint recorded, dispatching on the ``"kind"`` tag (ADC-626).

        The reader NEVER sniffs other fields: an unknown kind fails loud (a checkpoint written by
        a newer pops), never a silent Dense fallback. @p manifest is the dict :meth:`to_manifest`
        produced (already parsed from the npz JSON string)."""
        if not isinstance(manifest, dict) or "kind" not in manifest:
            raise ValueError(
                "history persistence manifest must be a dict with a 'kind' tag (got %r)"
                % (manifest,))
        kind = manifest["kind"]
        factory = _KIND_REGISTRY.get(kind)
        if factory is None:
            raise ValueError(
                "history persistence kind %r unknown -- this checkpoint was written by a newer "
                "pops (known kinds: %s)" % (kind, ", ".join(sorted(_KIND_REGISTRY))))
        return factory._from_manifest(manifest)

    @classmethod
    def _from_manifest(cls, manifest):
        """Default reader: a no-knob policy (Dense). Subclasses with knobs override."""
        return cls()

    def options(self):
        return {}

    def inspect(self):
        info = super().inspect()
        info["kind"] = self.kind
        return info


class Dense(HistoryPersistence):
    """Store EVERY ring slot (today's implicit behaviour, made explicit and the default).

    A depth-d ring persists all d slots -- zero recomputation, byte-compatible with a pre-ADC-626
    (v1) checkpoint. ``Dense`` needs no replay and is never refused: it is the historical
    behaviour and the resolved default when ``checkpoint_policy`` is omitted."""

    kind = "dense"

    def stored_slots(self, depth):
        return tuple(range(self._check_depth(depth)))


class Interval(HistoryPersistence):
    """Store slot 0 (newest, always) plus every k-th older slot; recompute the gaps.

    ``Interval(1)`` is exactly :class:`Dense`. VALID only when the stride lands on the OLDEST slot
    (``(depth - 1) % k == 0``): the oldest lag has nothing older to replay from, so if the stride
    skips it the ring is unreconstructable and the policy is refused at :meth:`validate_for`."""

    kind = "interval"

    def __init__(self, k):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("Interval(k): k must be a Python int >= 1 (got %r)" % (k,))
        self.k = int(k)

    def options(self):
        return {"k": self.k}

    def stored_slots(self, depth):
        depth = self._check_depth(depth)
        return tuple(sorted({0} | {s for s in range(depth) if s % self.k == 0}))

    def validate_for(self, depth):
        depth = super().validate_for(depth)
        if depth == 1:
            return depth  # a single-slot ring is always fully stored (Interval == Dense)
        if self.k >= depth:
            raise ValueError(
                "Interval(k=%d) invalid for ring depth %d: k >= depth stores only the newest slot "
                "-- use Dense() to keep the whole ring" % (self.k, depth))
        if (depth - 1) % self.k != 0:
            raise ValueError(
                "Interval(k=%d) does not store the oldest slot %d ((depth-1)=%d %% k=%d != 0); the "
                "oldest lag cannot be replayed -- choose k dividing depth-1, or Dense()"
                % (self.k, depth - 1, depth - 1, self.k))
        return depth

    def to_manifest(self):
        return {"kind": self.kind, "k": self.k}

    @classmethod
    def _from_manifest(cls, manifest):
        return cls(int(manifest["k"]))


class Revolve(HistoryPersistence):
    """Budget-bounded persistence: store ``snapshots`` slots placed to MINIMISE the worst-case
    replay latency (the largest gap between adjacent stored anchors).

    The placement is the EXACT min-max-gap optimum -- EQUISPACED anchors with both endpoints
    ``{0, depth-1}`` forced (see :func:`_optimal_placement`) -- NOT the classic Griewank-Walsh
    adjoint Revolve, because ring reconstruction fills each gap once by forward re-stepping so the
    total work is placement-invariant (``depth - snapshots``) and only the latency varies.
    ``snapshots >= 2`` is required (a single anchor cannot bound a depth>1 replay)."""

    kind = "revolve"

    def __init__(self, snapshots):
        if isinstance(snapshots, bool) or not isinstance(snapshots, int) or snapshots < 2:
            raise ValueError(
                "Revolve(snapshots): snapshots must be a Python int >= 2 (got %r); a single stored "
                "slot cannot bound the replay of a depth>1 ring -- use Dense() for a full ring"
                % (snapshots,))
        self.snapshots = int(snapshots)

    def options(self):
        return {"snapshots": self.snapshots}

    def stored_slots(self, depth):
        return _optimal_placement(self._check_depth(depth), self.snapshots)

    def validate_for(self, depth):
        depth = super().validate_for(depth)
        if depth == 1:
            return depth  # a single-slot ring is always fully stored (Revolve == Dense)
        if self.snapshots > depth:
            raise ValueError(
                "Revolve(snapshots=%d) exceeds ring depth %d: more budget than slots -- use Dense() "
                "to store the whole ring" % (self.snapshots, depth))
        return depth

    def to_manifest(self):
        return {"kind": self.kind, "snapshots": self.snapshots}

    @classmethod
    def _from_manifest(cls, manifest):
        return cls(int(manifest["snapshots"]))


#: Reader dispatch table: the ``"kind"`` tag -> the policy class (ADC-626). An unknown kind at
#: restart fails loud via :meth:`HistoryPersistence.from_manifest`.
_KIND_REGISTRY = {Dense.kind: Dense, Interval.kind: Interval, Revolve.kind: Revolve}

#: The default persistence when ``keep_history(checkpoint_policy=...)`` is omitted: a named
#: constant so the default is not a magic ``None`` -- it resolves to :class:`Dense`, the whole-ring
#: historical behaviour (:func:`resolve_history_persistence`).
DEFAULT_HISTORY_PERSISTENCE = None


def resolve_history_persistence(policy):
    """Resolve a ``keep_history(checkpoint_policy=...)`` argument to a concrete policy (ADC-626).

    ``None`` (the default) becomes :class:`Dense` -- the historical whole-ring behaviour, named via
    :data:`DEFAULT_HISTORY_PERSISTENCE` so the default is explicit. A :class:`HistoryPersistence`
    passes through. Any other type is refused (a free string / bare object is not a typed policy)."""
    if policy is DEFAULT_HISTORY_PERSISTENCE:
        return Dense()
    if isinstance(policy, HistoryPersistence):
        return policy
    raise TypeError(
        "keep_history(checkpoint_policy=%r): expected a pops.time history-persistence policy "
        "(Dense / Interval(k) / Revolve(snapshots)) or None; a bare object / string is not a typed "
        "policy" % (policy,))


def _optimal_placement(depth, snapshots):
    """The min-max-gap stored-slot placement for @p snapshots anchors on a depth-@p depth ring.

    EQUISPACED anchors with BOTH endpoints ``{0, depth-1}`` forced (the oldest slot must be an
    anchor -- nothing older to replay it from -- and the newest is stored for free). The largest
    gap is then ``ceil((depth-1)/(snapshots-1))``, the exact optimum for placing s points with
    fixed endpoints. Rounding collisions for small (depth, snapshots) are resolved by greedily
    splitting the LARGEST remaining gap at its midpoint (lower index wins ties) until exactly
    ``min(snapshots, depth)`` distinct anchors exist. Pure host logic, computed once at author /
    compile time and STORED in the manifest so the native reader never recomputes it.

    @return the sorted tuple of stored slot indices.
    """
    if snapshots >= depth:
        return tuple(range(depth))  # budget exceeds slots -> Dense
    if snapshots < 2:
        raise ValueError(
            "_optimal_placement requires snapshots >= 2 (got %d) to bound a depth-%d ring"
            % (snapshots, depth))
    # Equispaced base placement including both endpoints (i=0 -> 0, i=snapshots-1 -> depth-1).
    anchors = sorted({round(i * (depth - 1) / (snapshots - 1)) for i in range(snapshots)})
    # Deterministic gap-fill for rounding collisions: split the largest gap at its midpoint until
    # the count is met. anchors already contains 0 and depth-1, so every split lands strictly
    # inside an existing gap and the endpoints stay forced.
    while len(anchors) < snapshots:
        widest_lo = anchors[0]
        widest_gap = -1
        for lo, hi in zip(anchors, anchors[1:]):
            gap = hi - lo
            if gap > widest_gap:
                widest_gap = gap
                widest_lo = lo
        mid = widest_lo + widest_gap // 2
        if mid in anchors or mid == widest_lo:
            mid = widest_lo + 1  # a unit gap has no interior; step one in (still strictly inside)
        anchors = sorted(set(anchors) | {mid})
    return tuple(anchors)


__all__ = ["HistoryPersistence", "Dense", "Interval", "Revolve",
           "HISTORY_PERSISTENCE_CATEGORY", "DEFAULT_HISTORY_PERSISTENCE",
           "resolve_history_persistence"]
