"""pops.time Program IR passes + serialization (authoring mixin).

Optimization passes (dead-node / CSE / redundant-solve elimination), IR serialization (``_serialize``
/ ``_ir_hash``), ``validate``, ``_block_indices``; cost inspection is ``_ProgramInspect``. No _pops."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.program_serialization import _ProgramSerialization
from pops.time.values import ProgramValue, _Affine

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramPasses(_ProgramSerialization, _ProgramConstants, _ProgramBase):
    """IR optimization passes, serialization and validation for the Program authoring class."""

    @staticmethod
    def _subblock_value_refs(v: Any) -> Any:
        """Yield every ProgramValue an op references THROUGH its attrs (sub-block result pointers + the ops
        nested in its recorded sub-blocks). Used to keep alive anything a control-flow / matrix-free
        node closes over from the enclosing scope -- v1 never rewrites a sub-block, so it is all live.
        The flat ``v.inputs`` already covers the directly-passed values; this adds the attr-borne ones."""
        attrs = v.attrs
        for key in ("cond", "body", "true_result", "false_result",
                    "residual", "iterate", "guess",
                    "apply_result", "apply_in", "apply_out"):
            ref = attrs.get(key)
            if isinstance(ref, ProgramValue):
                yield ref
            elif isinstance(ref, _Affine):
                for term, _ in ref.terms:
                    yield term
        sched = attrs.get("schedule")
        if sched is not None:
            cond = getattr(sched, "params", {}).get("cond")
            if isinstance(cond, ProgramValue):
                yield cond  # a when(cond) predicate is live (kept off the dead-node / CSE-drop path)
        for key in ("cond_block", "body_block", "true_block", "false_block",
                    "apply_block", "residual_block"):
            block = attrs.get(key)
            if block:
                for w in block:
                    yield w
                    yield from w.inputs
                    yield from _ProgramPasses._subblock_value_refs(w)

    @staticmethod
    def _cse_key(v: Any, canon: Any) -> Any:
        """A canonical, alias-aware fingerprint of a PURE node: its op, vtype, block, the attrs the IR
        hash uses, and its inputs MAPPED THROUGH @p canon (each input id replaced by the id of the
        representative node it was deduplicated to). Two pure nodes with the same key compute the SAME
        value, so the later one can alias the earlier. Built on the same ``_serialize_node`` the IR hash
        uses (so attr equality is exactly the hash's notion of equality), with the node id stripped (it
        is position, not identity) and the input ids canonicalized."""
        node = _ProgramPasses._serialize_node(v, include_provenance=False)
        node.pop("id", None)
        node["inputs"] = tuple(canon.get(i, i) for i in node["inputs"])
        # JSON-serialize the attrs dict to a stable string so the whole key is hashable / comparable
        # exactly as the IR hash compares it.
        # Typed block/state identities are canonical JSON mappings, not scalar labels.  Encode both
        # with the same stable JSON projection as attrs/space before using them in the dict key.  The
        # state identity is semantically required too: two pure reads from distinct state families in
        # one block must never become a common subexpression merely because their local IR matches.
        return (node["op"], node["vtype"],
                json.dumps(node.get("block"), sort_keys=True, separators=(",", ":")),
                json.dumps(node.get("state"), sort_keys=True, separators=(",", ":")),
                node["inputs"],
                json.dumps(node.get("space"), sort_keys=True, separators=(",", ":")),
                json.dumps(node.get("field_context"), sort_keys=True, separators=(",", ":")),
                json.dumps(node["attrs"], sort_keys=True, separators=(",", ":")))

    def _live_value_ids(self) -> Any:
        """The set of value ids reverse-reachable from the live roots: the commits plus every flat node
        whose op is NOT on the ``_REMOVABLE_OPS`` allow-list (safe-by-default -- a buffer-writing,
        side-effecting, sub-block-owning, or unknown op is a live root). A flat node is DEAD only if its
        op IS allow-listed AND no live op consumes its result. Sub-block-internal values are included so
        a dead-node clone keeps a self-consistent IR."""
        by_id = {v.id: v for v in self._values}
        # Sub-block ops are not in self._values (they belong to their owning op); index them too so a
        # reference from one sub-block op to another resolves during the walk.
        for v in self._values:
            for w in self._subblock_value_refs(v):
                by_id.setdefault(w.id, w)
        roots = [s.id for s in self._commits.values()]
        for v in self._values:
            if v.op not in self._REMOVABLE_OPS:
                roots.append(v.id)
        live, stack = set(), list(roots)
        while stack:
            vid = stack.pop()
            if vid in live or vid not in by_id:
                continue
            live.add(vid)
            v = by_id[vid]
            for inp in v.inputs:
                stack.append(inp.id)
            for w in self._subblock_value_refs(v):
                stack.append(w.id)
        return live

    def eliminate_dead_nodes(self) -> Any:
        """Return a NEW Program with the dead flat-list nodes removed (Spec 3 s28 dead-node
        elimination, ADC-465). An OPT-IN pass: call it explicitly to optimize a copy -- it NEVER runs
        on the default ``emit_cpp_program`` path, so it cannot change an existing compiled program.

        The pass is SAFE-BY-DEFAULT: a flat node is DEAD only if its op is on the ``_REMOVABLE_OPS``
        allow-list (ops verified to allocate a FRESH result scratch and have no other side effect: rhs,
        source, apply, linear_combine, linear_source, solve_local_linear, cell_compare, where, reduce,
        scalar_op, compare) AND no live op consumes its result. EVERY other op -- the buffer-writers
        that alias a caller-allocated input buffer (schur_rhs, laplacian, gradient, divergence,
        schur_*), the side-effecting ops (solve_fields, project, fill_boundary, store_history,
        record_scalar), solve_linear, and the sub-block-owning ops (while/if/range,
        matrix_free_operator, solve_local_nonlinear) -- is treated as LIVE even when its result looks
        unconsumed, so an unknown/new op is NEVER wrongly dropped. The live set is reverse-reachability
        from the commits plus those non-removable nodes. The surviving nodes are renumbered to
        contiguous ids in their original order, so a program with no dead node round-trips byte-for-byte
        (same ``_ir_hash`` and emitted C++) and one with a dead node matches the same program written
        without it. The histories, optional dt bound and bound operator registry carry over unchanged."""
        live = self._live_value_ids()
        return self._rebuild(lambda v: v.id in live)

    def _rebuild(
        self,
        keep: Any,
        alias: Any = None,
        space_of: Any = None,
        *,
        reference_of: Any = None,
        retain_operator_registries: bool = True,
        canonical_owner: bool = False,
        transformation: str = "normalize",
    ) -> Any:
        """Clone and remap this Program through the shared lossless rebuild engine.

        The method remains the optimization and compiled-boundary hook. ``reference_of`` replaces
        semantic handles, ``retain_operator_registries=False`` drops authoring registries, and
        ``canonical_owner=True`` detaches the rebuilt Program from its authoring owner.
        """
        from pops.time.program_rebuild import rebuild_program
        return rebuild_program(
            self,
            keep,
            alias,
            space_of,
            reference_of=reference_of,
            retain_operator_registries=retain_operator_registries,
            canonical_owner=canonical_owner,
            transformation=transformation,
        )

    # --- common-subexpression elimination (Spec 3 s28, ADC-465) ---
    def eliminate_common_subexpressions(self) -> Any:
        """Return a NEW Program with duplicated PURE sub-IR computed once and aliased (Spec 3 s28
        common-subexpression elimination, ADC-465).

        PROVEN SOUND, not heuristic: a node is a CSE candidate ONLY if its op is on the
        ``_PURE_OPS`` allow-list -- ops that allocate a fresh result from their inputs alone, read no
        buffer through a side channel, and have no side effect. For each such node, a canonical key (op,
        vtype, block, attrs, and inputs mapped to their already-chosen representatives) is computed in
        creation order; the FIRST node with a given key is the representative, and every later node with
        the same key is dropped and its uses rewired to the representative. Because the key is exactly
        the IR hash's notion of equality (same ``_serialize_node`` attrs) over identical
        (canonicalized) inputs, the representative computes a bit-identical result -- so aliasing the
        duplicate CANNOT change the emitted numerics. Every NON-pure op (a reduce, a solve, a
        buffer-writer, a side-effecting op, an unknown op) is NEVER a representative target and is never
        dropped, so it is always recomputed -- safe-by-default.

        The pass is OPT-IN: it never runs on the default ``emit_cpp_program`` path. Sub-blocks are not
        descended into (v1), so a value consumed only inside a control-flow body is left untouched. A
        program with no duplicated pure sub-IR rebuilds BYTE-FOR-BYTE identically (same ``_ir_hash`` and
        emitted C++); a program with a duplicate emits, after the pass, C++ identical to the same
        program written with the value computed once."""
        canon = {}        # duplicate node id -> representative node id (the survivor it aliases)
        reps = {}         # cse key -> representative node id
        drop = set()      # ids of dropped duplicates
        for v in self._values:
            if v.op not in self._PURE_OPS:
                continue
            # An input that was itself dropped resolves to its representative (so chained duplicates
            # collapse onto one representative chain, not a dangling id).
            if any(i.id in drop for i in v.inputs):
                # canon already maps every dropped input to its survivor; the key uses canon, so this
                # node keys against the survivors -- no special handling needed beyond canon lookup.
                pass
            key = self._cse_key(v, canon)
            if key in reps:
                canon[v.id] = reps[key]
                drop.add(v.id)
            else:
                reps[key] = v.id
                canon[v.id] = v.id
        if not drop:
            return self._rebuild(lambda v: True, transformation="cse")
        return self._rebuild(
            lambda v: v.id not in drop, alias=canon, transformation="cse")

    # --- redundant field-solve elimination (Spec 3 s28, ADC-465) ---
    def eliminate_redundant_field_solves(self) -> Any:
        """Return a NEW Program with a provably-redundant second ``solve_fields`` removed and aliased
        (Spec 3 s28 redundant-solve elimination, ADC-465).

        ``solve_fields`` is side-effecting (it fills the shared phi/aux and returns a FieldContext), so
        it is NEVER touched by CSE or dead-node elimination. But two ``solve_fields`` over the SAME
        state input with NO intervening STATE OR AUX MUTATION recompute the identical fields: the
        second is redundant and its FieldContext can alias the first. This is the ONLY field solve this
        pass removes, and ONLY when redundancy is PROVABLE.

        CONSERVATIVE soundness rule: walking the flat list in order, a ``solve_fields(state=U,
        field=f)`` is redundant iff an EARLIER ``solve_fields`` with the SAME state input AND the same
        ``field`` attr exists AND, between the two, NO op on the ``_STATE_BARRIER_OPS`` set (a commit
        target write, ``project``, ``fill_boundary``, ``store_history``, or ANY other field solve --
        which would re-fill the shared aux) appears. The Poisson RHS reads every block's LIVE state, so
        a write to ANY block's state -- not just U's -- is a barrier; a ``linear_combine`` that is a
        commit target is therefore a barrier too. If anything between the two solves could have changed
        what the elliptic solve sees, the second is kept. The single-block, no-mutation case (e.g. a
        macro accidentally solving twice from U^n before the first stage) is the one provably-redundant
        shape eliminated; everything else is left as written.

        OPT-IN: never on the default emit path. Byte-identical no-op when no redundant solve exists."""
        commit_ids = {s.id for s in self._commits.values()}
        active = {}       # (state input id, field attr) -> the live representative solve_fields id
        canon = {}
        drop = set()
        for v in self._values:
            if v.op == "solve_fields":
                (state_in,) = v.inputs
                sig = (state_in.id, v.attrs.get("field"))
                prior = active.get(sig)
                if prior is not None:
                    # A redundant re-solve over the same state with no barrier since `prior`.
                    canon[v.id] = prior
                    drop.add(v.id)
                    continue
                active[sig] = v.id
                # This solve_fields is itself a barrier for OTHER signatures (it re-fills the shared
                # aux), so any pending solve over a different state is no longer safe to reuse.
                active = {sig: v.id}
                continue
            # A barrier op invalidates every pending reuse: a state write (commit target, project,
            # fill_boundary), a history store, or anything that mutates what the elliptic solve reads.
            if v.op in self._STATE_BARRIER_OPS or v.id in commit_ids:
                active = {}
        if not drop:
            return self._rebuild(lambda v: True)
        return self._rebuild(lambda v: v.id not in drop, alias=canon)

    # --- proven-safe optimization pipeline (Spec 3 s28, ADC-465) ---
    # The TRANSFORM passes, in the order ``optimize`` runs them. Each is PROVEN to preserve the emitted
    # numerics (see its docstring) and is a byte-identical no-op when it finds nothing to do, so the
    # whole pipeline is a no-op on an already-optimal Program. Analysis passes (liveness / estimate /
    # GPU detector) are reports, NOT in this list -- they never rewrite the IR.

    def optimize(self) -> Any:
        """Return a NEW Program with the proven-safe Spec 3 s28 transform passes applied in sequence
        (ADC-465): dead-node elimination, common-subexpression elimination, redundant field-solve
        elimination. OPT-IN -- the default ``emit_cpp_program`` path never optimizes. Each pass is
        proven to preserve the emitted numerics and is byte-identical when it changes nothing, so a
        Program with no optimizable structure round-trips byte-for-byte (same ``_ir_hash`` and C++)
        with the pipeline on or off (the spec's hard requirement: optimization must not change
        results). Use :meth:`dump_passes` to inspect what each pass did."""
        prog = self
        for name, _ in self._OPTIMIZE_PASSES:
            prog = getattr(prog, name)()
        return prog

    def dump_passes(self) -> Any:
        """Inspect the optimization pipeline: for each proven-safe transform pass, the number of flat
        nodes before / after and whether it changed the IR hash. A report only -- it RUNS the pipeline
        on a copy (``self`` is never mutated) and returns a human-readable trace, so a reviewer can see
        which pass fired and that an all-no-op pipeline leaves the hash unchanged (Spec 3 s28,
        ADC-465)."""
        lines = ["# optimization pipeline for Program %r" % self.name]
        prog = self
        for name, label in self._OPTIMIZE_PASSES:
            before_n, before_h = len(prog._values), prog._ir_hash()
            nxt = getattr(prog, name)()
            after_n, after_h = len(nxt._values), nxt._ir_hash()
            changed = after_h != before_h
            lines.append("  %-34s %3d -> %3d nodes  %s"
                         % (label, before_n, after_n, "CHANGED" if changed else "no-op"))
            prog = nxt
        lines.append("  final: %d nodes, hash %s" % (len(prog._values), prog._ir_hash()[:12]))
        return "\n".join(lines)

    # --- analysis passes: liveness / buffer reuse / cost estimate / GPU detectors (Spec 3 s28) ---
    # Ops that allocate ONE step-body scratch buffer (a MultiFab the size of the block state / a
    # scalar field). The number of buffers each op writes per "kernel" + how many for_each_cell kernels
    # it launches drive the memory-traffic and kernel-count estimates and the per-scratch live ranges.
    # These counts are STRUCTURAL (read off the IR / the lowering above), not a measured profile -- the
    # measured GPU kernel count is a ROMEO run; this is the host-side static estimate.
    # Ops that launch at least one per-cell (for_each_cell) kernel when lowered -- the small-kernel
    # count a GPU launch-overhead detector flags. A linear_combine lowers to axpy/lincomb (vectorized,
    # counted as one kernel); a rhs to a divergence/flux kernel; a source/apply/where/cell_compare/
    # coupled_rate to an explicit for_each_cell. solve_fields / solve_linear launch many internal
    # kernels (an elliptic / Krylov solve) -- counted as a HEAVY kernel, not a small one.


    def validate(self) -> Any:
        """Structural validation of the IR. Raises ValueError on a malformed program."""
        from pops.time.program_region_validation import validate_program_regions
        validate_program_regions(self)
        if not self._commits:
            raise ValueError("a time Program must commit each advanced block exactly once "
                             "(no block was committed)")
        seen = set()
        for v in self._values:
            for inp in v.inputs:
                if inp.id not in seen:
                    raise ValueError("IR value '%s' used before definition" % inp.name)
            seen.add(v.id)
            if v.op == "while":
                self._validate_block(v.attrs["cond_block"], seen)
                self._validate_block(v.attrs["body_block"], seen)
            elif v.op == "range":
                self._validate_block(v.attrs["body_block"], seen)
            elif v.op == "branch":
                self._validate_block(v.attrs["true_block"], seen)
                self._validate_block(v.attrs["false_block"], seen)
            elif v.op == "matrix_free_operator" and v.attrs.get("apply_block"):
                # The apply sub-block is self-contained (its in/out placeholders + scratch are defined
                # inside it); it reads nothing from the enclosing scope.
                self._validate_block(v.attrs["apply_block"], seen)
            elif v.op == "solve_local_nonlinear":
                # The residual sub-block is self-contained: the iterate / guess State placeholders are
                # defined inside it (first ops) and every op reads only the placeholders or earlier
                # sub-block ops. Validate against an EMPTY outer scope so a residual that closes over an
                # enclosing value (which the per-cell kernel cannot evaluate) fails loud here, not as a
                # codegen KeyError.
                self._validate_block(v.attrs["residual_block"], set())
        # ADC-626 compile-time gate: per keep_history ring, policy coherence + program-determinism of
        # the replay that recomputes the non-stored slots. Loud (never a silent degrade to Dense).
        from pops.time.history_persistence_validate import check_program
        check_program(self)
        return True

    def _validate_block(self, block: Any, outer_seen: Any) -> Any:
        """Validate a control-flow sub-block: each op may read values defined earlier in the SAME block
        or in the enclosing scope (the loop variable / anything defined before the while). @p outer_seen
        is the enclosing scope's def set (copied, not mutated -- the sub-block ops are not visible
        outside)."""
        seen = set(outer_seen)
        for v in block:
            for inp in v.inputs:
                imported = inp.region in self._region_imports.get(v.region, ())
                if inp.id not in seen and not imported:
                    raise ValueError("IR value '%s' used before definition" % inp.name)
            seen.add(v.id)
