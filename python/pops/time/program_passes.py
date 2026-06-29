"""pops.time Program IR passes + serialization (authoring mixin).

Optimization passes (dead-node / CSE / redundant-solve elimination, ``optimize`` /
``dump_passes``), IR serialization (``_serialize`` / ``_serialize_node`` / ``_ir_hash``),
``validate`` and ``_block_indices``. The static cost inspection surface lives in
``_ProgramInspect`` (``pops.time.program_inspect``). All pure IR analysis: no codegen, no
_pops.
"""
import hashlib
import json

from pops.time.program_base import _ProgramConstants
from pops.time.schedule import Schedule
from pops.time.values import Value, _Affine, _affine_ids  # noqa: F401


class _ProgramPasses(_ProgramConstants):
    """IR optimization passes, serialization and validation for the Program authoring class."""

    @staticmethod
    def _is_removable_node(v):
        """Whether dead-node elimination may drop an unconsumed flat node.

        Typed ``call`` nodes need output-sensitive handling: rate/operator calls allocate derived
        scratch or metadata and are removable when dead; field and state calls mutate shared runtime
        state/aux and are live roots.
        """
        if v.op == "call":
            return v.vtype in ("rhs", "operator")
        return v.op in _ProgramPasses._REMOVABLE_OPS

    @staticmethod
    def _subblock_value_refs(v):
        """Yield attr-borne Value refs, including nested sub-block values."""
        attrs = v.attrs
        for key in ("cond", "body", "residual", "iterate", "guess",
                    "apply_result", "apply_in", "apply_out"):
            ref = attrs.get(key)
            if isinstance(ref, Value):
                yield ref
            elif isinstance(ref, _Affine):
                for term, _ in ref.terms:
                    yield term
        sched = attrs.get("schedule")
        if sched is not None:
            cond = getattr(sched, "params", {}).get("cond")
            if isinstance(cond, Value):
                yield cond  # a when(cond) predicate is live (kept off the dead-node / CSE-drop path)
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            block = attrs.get(key)
            if block:
                for w in block:
                    yield w
                    yield from w.inputs
                    yield from _ProgramPasses._subblock_value_refs(w)

    @staticmethod
    def _cse_key(v, canon):
        """Canonical, alias-aware fingerprint of a pure node."""
        node = _ProgramPasses._serialize_node(v)
        node.pop("id", None)
        node["inputs"] = tuple(canon.get(i, i) for i in node["inputs"])
        # JSON-serialize the attrs dict to a stable string so the whole key is hashable / comparable
        # exactly as the IR hash compares it.
        return (node["op"], node["vtype"], node["block"], node["inputs"],
                json.dumps(node["attrs"], sort_keys=True, separators=(",", ":")))

    @staticmethod
    def _serialize_node(v):
        attrs = dict(v.attrs)
        if "schedule" in attrs:  # an authoring annotation: serialize its repr so the IR hash is
            attrs["schedule"] = repr(attrs["schedule"])  # schedule-sensitive yet JSON-safe
        if "coeffs" in attrs:  # dict keys (powers) -> sorted [power, value] for stable JSON
            attrs["coeffs"] = [sorted((int(p), c) for p, c in d.items()) for d in attrs["coeffs"]]
        if "a_coeff" in attrs:  # solve_local_linear: the dt-polynomial a in (I - a*L)
            attrs["a_coeff"] = sorted((int(p), c) for p, c in attrs["a_coeff"].items())
        if "c_dt" in attrs:  # rhs_jacvec: the dt-polynomial BDF coefficient c*dt
            attrs["c_dt"] = sorted((int(p), c) for p, c in attrs["c_dt"].items())
        if v.op == "while":  # the cond/body sub-blocks are nested node lists; the results are ids
            attrs["cond_block"] = [_ProgramPasses._serialize_node(w) for w in attrs["cond_block"]]
            attrs["body_block"] = [_ProgramPasses._serialize_node(w) for w in attrs["body_block"]]
            attrs["cond"] = attrs["cond"].id
            attrs["body"] = attrs["body"].id
        elif v.op in ("range", "if"):  # body sub-block (range carries its int count in attrs too)
            attrs["body_block"] = [_ProgramPasses._serialize_node(w) for w in attrs["body_block"]]
            attrs["body"] = attrs["body"].id
        elif v.op == "matrix_free_operator":  # the apply sub-block is a nested node list; refs are ids
            attrs["apply_block"] = ([_ProgramPasses._serialize_node(w) for w in attrs["apply_block"]]
                                    if attrs.get("apply_block") else None)
            for k in ("apply_result", "apply_in", "apply_out"):
                ref = attrs.get(k)
                attrs[k] = (_affine_ids(ref) if isinstance(ref, _Affine)
                            else (ref.id if isinstance(ref, Value) else None))
        elif v.op == "solve_local_nonlinear":  # the residual sub-block is a nested node list; refs ids
            attrs["residual_block"] = [_ProgramPasses._serialize_node(w) for w in attrs["residual_block"]]
            for k in ("residual", "iterate", "guess"):
                attrs[k] = attrs[k].id
        return {"id": v.id, "vtype": v.vtype, "op": v.op, "block": v.block,
                "inputs": [i.id for i in v.inputs], "attrs": attrs}


    def _live_value_ids(self):
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
        roots.extend(f.id for f in getattr(self, "_commit_fields", {}).values())
        for v in self._values:
            if not self._is_removable_node(v):
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

    def eliminate_dead_nodes(self):
        """Return a NEW Program with the dead flat-list nodes removed (Spec 3 s28 dead-node
        elimination, ADC-465). An OPT-IN pass: call it explicitly to optimize a copy -- it NEVER runs
        on the default ``emit_cpp_program`` path, so it cannot change an existing compiled program.

        The pass is SAFE-BY-DEFAULT: a flat node is DEAD only if it is on the removable allow-list, or
        if it is a typed rate/operator ``call`` that allocates a FRESH result scratch/metadata and no
        live op consumes it. EVERY other op -- the buffer-writers that alias a caller-allocated input
        buffer (schur_rhs, laplacian, gradient, divergence, schur_*), the side-effecting ops
        (field/state calls, solve_fields, project, fill_boundary, store_history, record_scalar),
        solve_linear, and the sub-block-owning ops (while/if/range,
        matrix_free_operator, solve_local_nonlinear) -- is treated as LIVE even when its result looks
        unconsumed, so an unknown/new op is NEVER wrongly dropped. The live set is reverse-reachability
        from the commits plus those non-removable nodes. The surviving nodes are renumbered to
        contiguous ids in their original order, so a program with no dead node round-trips byte-for-byte
        (same ``_ir_hash`` and emitted C++) and one with a dead node matches the same program written
        without it. The histories, optional dt bound and bound operator registry carry over unchanged."""
        live = self._live_value_ids()
        return self._rebuild(lambda v: v.id in live)

    def _rebuild(self, keep, alias=None):
        """Clone this Program into a fresh one keeping the flat nodes for which ``keep(v)`` is true,
        renumbering surviving ids to a contiguous 0.. range in original order. Sub-blocks are cloned
        wholesale (never filtered). The clone reproduces the IR identity of an equivalent hand-built
        Program (same serialization), so it is byte-for-byte identical when nothing was dropped.

        @p alias (optional) maps a DROPPED node id -> the kept representative node id it should be
        replaced by (the CSE / redundant-solve passes use it to rewire every use of a duplicate onto its
        survivor). Every reference -- a flat input, an attr-borne Value / affine ref, a commit target --
        is resolved THROUGH this alias before id lookup, so a dropped node never leaves a dangling
        reference. A dropped node MUST have an alias entry (the passes guarantee a representative whose
        id < the duplicate's, hence already cloned); a kept node maps to itself. Without an alias map
        the behavior is the plain drop-only rebuild."""
        out = type(self)(self.name)
        out.dt = self.dt
        out._histories = dict(self._histories)
        out._registry = self._registry
        idmap = {}  # old Value -> new Value
        by_id = {v.id: v for v in self._values}
        for v in self._values:  # sub-block ops too, so an alias to a sub-block-internal id resolves
            for w in self._subblock_value_refs(v):
                by_id.setdefault(w.id, w)
        alias = alias or {}

        def rep(v):
            """Follow @p v through the alias chain to the surviving representative Value (identity for a
            kept node). The passes only alias onto an EARLIER, kept node, so the chain terminates."""
            seen = set()
            while v.id in alias and alias[v.id] != v.id:
                if v.id in seen:  # defensive: never loop on a malformed alias map
                    break
                seen.add(v.id)
                v = by_id[alias[v.id]]
            return v

        def clone_block(block):
            return [clone(w) for w in block]

        def remap(ref):
            if isinstance(ref, Value):
                return idmap[rep(ref)]
            if isinstance(ref, _Affine):
                return _Affine([(idmap[rep(v)], c) for v, c in ref.terms])
            return ref

        def clone_attrs(v):
            attrs = {}
            for key, val in v.attrs.items():
                if key in ("cond_block", "body_block", "apply_block", "residual_block"):
                    attrs[key] = clone_block(val) if val else val
                elif key in ("cond", "body", "residual", "iterate", "guess",
                             "apply_result", "apply_in", "apply_out"):
                    attrs[key] = remap(val)
                elif key == "schedule" and val is not None and isinstance(
                        getattr(val, "params", {}).get("cond"), Value):
                    # a when(cond) schedule embeds a predicate Value in params["cond"]; remap it onto
                    # the survivor so a CSE-collapsed/renumbered predicate is not left dangling.
                    attrs[key] = Schedule(val.kind, val.policy,
                                          **{**val.params, "cond": remap(val.params["cond"])})
                else:
                    attrs[key] = val
            return attrs

        def deps(v):
            """The values v depends on that must be cloned (hence id-assigned) BEFORE v, in their
            ORIGINAL creation order. A fresh build records the inputs and most sub-blocks before the
            owning node, BUT a matrix_free_operator is created (its node id assigned) BEFORE
            ``set_apply`` records its apply sub-block -- the node id precedes the sub-block ids. Ordering
            every dependency by its original id (ascending) reproduces the build order verbatim for both
            shapes, so a no-drop clone is byte-for-byte stable (same renumbering) rather than reordering the
            matrix_free_operator node after its own sub-block. Each input is resolved THROUGH the alias
            map, so a dropped duplicate is replaced by its (already-earlier) representative."""
            seen = []
            for inp in v.inputs:
                seen.append(rep(inp))
            for key in ("cond_block", "body_block", "apply_block", "residual_block"):
                block = v.attrs.get(key)
                if block:
                    seen.extend(block)
            # A matrix_free_operator's sub-block ops are created AFTER the node, so they must NOT be
            # forced ahead of it; an input / control-flow body is created BEFORE. Keep only the deps
            # whose original id precedes v's (the genuine predecessors) and visit them id-ascending.
            return sorted((w for w in seen if w.id < v.id), key=lambda w: w.id)

        def clone(v):
            if v in idmap:
                return idmap[v]
            # Assign new ids in ORIGINAL creation order: clone every predecessor (id < v.id) first,
            # id-ascending, then v, then any sub-block op created AFTER v (e.g. a matrix_free_operator's
            # apply ops, whose original ids exceed the operator node's). Inputs / attr refs are remapped
            # through idmap after alias resolution (every referenced surviving value is mappable on its
            # own clone).
            for w in deps(v):
                clone(w)
            vid = out._next_id
            out._next_id += 1
            nv = Value(out, vid, v.vtype, v.op, [idmap[rep(i)] for i in v.inputs],
                       clone_attrs(v), v.name, v.block)
            nv.space = v.space
            idmap[v] = nv
            return nv

        # Clone all surviving flat nodes (and, transitively, their sub-block ops and any later-created
        # sub-block ops) in ascending original id, so the contiguous renumbering matches the original
        # build order exactly -- a no-op clone is byte-for-byte identical.
        kept = sorted((v for v in self._values if keep(v)), key=lambda v: v.id)
        for v in kept:
            clone(v)
        out._values = [idmap[v] for v in kept]
        out._commits = {b: idmap[rep(s)] for b, s in self._commits.items()}
        out._commit_fields = {
            b: idmap[rep(f)] for b, f in getattr(self, "_commit_fields", {}).items()
        }
        if self._dt_bound is not None:
            sub, result = self._dt_bound
            cloned_sub = [clone(w) for w in sub]
            out._dt_bound = (cloned_sub, idmap[result])
        return out

    # --- common-subexpression elimination (Spec 3 s28, ADC-465) ---
    def eliminate_common_subexpressions(self):
        """Return a new Program with duplicate pure nodes computed once and aliased."""
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
            return self._rebuild(lambda v: True)  # byte-for-byte no-op clone
        return self._rebuild(lambda v: v.id not in drop, alias=canon)

    # --- redundant field-solve elimination (Spec 3 s28, ADC-465) ---
    @staticmethod
    def _is_field_solve_node(v):
        """True for the legacy field-solve op and the canonical typed field-operator call."""
        return (
            v.op in ("solve_fields", "solve_fields_from_blocks")
            or (v.op == "call" and v.vtype == "fields")
        )

    @staticmethod
    def _field_solve_signature(v):
        """Stable redundancy signature for one field-solve node."""
        if v.op == "call":
            return (
                "call",
                v.attrs.get("operator_id"),
                v.attrs.get("operator"),
                tuple(inp.id for inp in v.inputs),
            )
        return (
            v.op,
            v.attrs.get("field"),
            tuple(inp.id for inp in v.inputs),
        )

    @staticmethod
    def _is_field_solve_barrier(v, commit_ids):
        """Whether a node invalidates a pending field-solve reuse."""
        if v.id in commit_ids:
            return True
        if v.op in ("project", "fill_boundary", "store_history"):
            return True
        if v.op == "call" and v.vtype == "state":
            return True
        return False

    def eliminate_redundant_field_solves(self):
        """Return a NEW Program with a provably-redundant second field solve removed and aliased
        (Spec 3 s28 redundant-solve elimination, ADC-465).

        Field solves are side-effecting (they fill the shared phi/aux and return a FieldContext), so
        they are NEVER touched by CSE or dead-node elimination. But two field solves over the SAME
        typed operator and SAME input states with NO intervening STATE OR AUX MUTATION recompute the
        identical fields: the second is redundant and its FieldContext can alias the first. This pass
        handles both the old internal ``solve_fields`` op and the canonical field-operator
        ``P.call(...)`` node; it does not make all ``call`` nodes barriers.

        CONSERVATIVE soundness rule: walking the flat list in order, a field solve is redundant iff an
        EARLIER field solve with the SAME signature exists AND, between the two, no commit target write,
        ``project``, ``fill_boundary``, ``store_history``, state-returning call, or different field
        solve appears. The Poisson RHS reads every block's LIVE state, so a write to ANY block's state
        is a barrier. If anything between the two solves could have changed what the elliptic solve
        sees, the second is kept.

        OPT-IN: never on the default emit path. Byte-for-byte no-op when no redundant solve exists."""
        commit_ids = {s.id for s in self._commits.values()}
        active = {}       # field-solve signature -> live representative id
        canon = {}
        drop = set()
        for v in self._values:
            if self._is_field_solve_node(v):
                sig = self._field_solve_signature(v)
                prior = active.get(sig)
                if prior is not None:
                    # A redundant re-solve over the same state with no barrier since `prior`.
                    canon[v.id] = prior
                    drop.add(v.id)
                    continue
                active[sig] = v.id
                # This field solve is itself a barrier for OTHER signatures (it re-fills the shared
                # aux), so any pending solve over a different signature is no longer safe to reuse.
                active = {sig: v.id}
                continue
            # A barrier op invalidates every pending reuse: a state write (commit target, project,
            # fill_boundary), a history store, or anything that mutates what the elliptic solve reads.
            if self._is_field_solve_barrier(v, commit_ids):
                active = {}
        if not drop:
            return self._rebuild(lambda v: True)
        return self._rebuild(lambda v: v.id not in drop, alias=canon)

    # --- proven-safe optimization pipeline (Spec 3 s28, ADC-465) ---
    # The TRANSFORM passes, in the order ``optimize`` runs them. Each is PROVEN to preserve the emitted
    # numerics (see its docstring) and is a byte-for-byte no-op when it finds nothing to do, so the
    # whole pipeline is a no-op on an already-optimal Program. Analysis passes (liveness / estimate /
    # GPU detector) are reports, NOT in this list -- they never rewrite the IR.

    def optimize(self):
        """Return a NEW Program with the proven-safe Spec 3 s28 transform passes applied in sequence
        (ADC-465): dead-node elimination, common-subexpression elimination, redundant field-solve
        elimination. OPT-IN -- the default ``emit_cpp_program`` path never optimizes. Each pass is
        proven to preserve the emitted numerics and is byte-for-byte stable when it changes nothing, so a
        Program with no optimizable structure round-trips byte-for-byte (same ``_ir_hash`` and C++)
        with the pipeline on or off (the spec's hard requirement: optimization must not change
        results). Use :meth:`dump_passes` to inspect what each pass did."""
        prog = self
        for name, _ in self._OPTIMIZE_PASSES:
            prog = getattr(prog, name)()
        return prog

    def dump_passes(self):
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
    # Structural host-side estimates: scratch buffers, kernel counts and live ranges by op kind.


    def validate(self):
        """Structural validation of the IR. Raises ValueError on a malformed program."""
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
            elif v.op in ("range", "if"):
                self._validate_block(v.attrs["body_block"], seen)
            elif v.op == "matrix_free_operator" and v.attrs.get("apply_block"):
                # The apply sub-block is self-contained (its in/out placeholders + scratch are defined
                # inside it); it reads nothing from the enclosing scope.
                self._validate_block(v.attrs["apply_block"], seen)
            elif v.op == "solve_local_nonlinear":
                # Residual sub-blocks are self-contained; fail loud on captured outer values.
                self._validate_block(v.attrs["residual_block"], set())
        for block, state in self._commits.items():
            if state.id not in seen:
                raise ValueError("commit for block '%s' uses undefined state '%s'"
                                 % (block, state.name))
        for block, fields in getattr(self, "_commit_fields", {}).items():
            if fields.id not in seen:
                raise ValueError("commit for block '%s' uses undefined fields '%s'"
                                 % (block, fields.name))
        return True

    def _validate_block(self, block, outer_seen):
        """Validate a control-flow sub-block: each op may read values defined earlier in the SAME block
        or in the enclosing scope (the loop variable / anything defined before the while). @p outer_seen
        is the enclosing scope's def set (copied, not mutated -- the sub-block ops are not visible
        outside)."""
        seen = set(outer_seen)
        for v in block:
            for inp in v.inputs:
                if inp.id not in seen:
                    raise ValueError("IR value '%s' used before definition" % inp.name)
            seen.add(v.id)

    # --- serialization / hash ---
    def _serialize(self):
        nodes = [self._serialize_node(v) for v in self._values]
        commits = sorted((b, s.id) for b, s in self._commits.items())
        commit_fields = sorted(
            (b, f.id) for b, f in getattr(self, "_commit_fields", {}).items()
        )
        # Program block declaration order is part of the IR identity and cache key.
        _order = self._block_indices()
        block_order = sorted(_order, key=_order.get)
        out = {"name": self.name, "version": 1, "nodes": nodes, "commits": commits,
               "block_order": block_order}
        if commit_fields:
            out["commit_fields"] = commit_fields
        # Optional dt-bound sub-program participates in the Program IR hash.
        if self._dt_bound is not None:
            sub, result = self._dt_bound
            out["dt_bound"] = {"nodes": [self._serialize_node(w) for w in sub],
                               "result": result.id}
        return out

    def _ir_hash(self):
        """Stable SHA-256 of the IR (feeds the compiled-problem cache key in a later phase)."""
        blob = json.dumps(self._serialize(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


    def _block_indices(self):
        """Map each block name to a stable runtime index in first ``P.state`` declaration order."""
        order = {}
        for v in self._values:
            if v.op == "state" and v.block not in order:
                order[v.block] = len(order)
        return order
