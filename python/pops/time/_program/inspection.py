"""pops.time Program static cost inspection (authoring mixin).

Scratch-buffer liveness / reuse (``scratch_liveness`` / ``buffer_reuse_report``), the static
memory-traffic + kernel-count ``estimate``, the GPU anti-pattern detectors (``gpu_detectors``)
and the human-readable ``estimate_report`` (Spec 3 s28, ADC-465). A REPORT surface: pure IR
analysis, never mutates the Program, no codegen / _pops.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._report import Report
from pops.time._program.constants import _ProgramConstants
from pops.time.references import block_name, handle_data
from pops.time.values import ProgramValue, _Affine  # noqa: F401

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class ProgramReport(Report):
    """The typed inspection report of a :class:`pops.time.Program` (ADC-564).

    Attributes: ``name`` / ``ops`` (node count) / ``commits`` (qualified committed state handles) /
    ``hash`` (the IR hash) / ``histories`` / ``dt_bound`` / ``scratch``. The JSON bridge serializes
    every commit with its complete Case, block, model and state identity.
    """

    report_type = "program"
    schema_version = 1

    def __init__(self, *, name: Any, ops: Any, commits: Any, hash: Any, histories: Any,
                 dt_bound: Any, scratch: Any) -> None:
        self.name = name
        self.ops = ops
        self.commits = list(commits)
        self.hash = hash
        self.histories = dict(histories)
        self.dt_bound = dt_bound
        self.scratch = dict(scratch)

    def to_dict(self) -> Any:
        commits = [handle_data(item) if hasattr(item, "canonical_identity") else item
                   for item in self.commits]
        return self._stamp({"name": self.name, "ops": self.ops, "commits": commits,
                            "hash": self.hash, "histories": dict(self.histories),
                            "dt_bound": self.dt_bound, "scratch": dict(self.scratch)})


class _ProgramInspect(_ProgramConstants, _ProgramBase):
    """Static cost / buffer inspection for the Program authoring class."""

    def inspect(self) -> Any:
        """A typed :class:`ProgramReport` of this Program (ADC-564). Inert: no codegen, no mutation.

        Aggregates the IR facts the Program already carries -- its name, node count, committed
        blocks, IR hash, declared histories, the optional dt bound and the static scratch estimate
        -- into attributes + a ``to_dict`` bridge. ``pops.inspect(program)`` returns its ``to_dict``.
        """
        ir_hash = self._ir_hash() if hasattr(self, "_ir_hash") else None
        estimate = self.estimate() if hasattr(self, "estimate") else {}
        dt_bound = self._dt_bound[0] if getattr(self, "_dt_bound", None) else None
        return ProgramReport(
            name=getattr(self, "name", None), ops=len(getattr(self, "_values", [])),
            commits=sorted(getattr(self, "_commits", {}), key=lambda item: item.qualified_id),
            hash=ir_hash,
            histories=dict(getattr(self, "_histories", {})), dt_bound=dt_bound,
            scratch=dict(estimate))

    def ir_nodes(self) -> Any:
        """The generated IR nodes as a structured, inert list (ADC-554 inspection surface).

        One dict per SSA node in build order -- ``{name, op, vtype, block, inputs, attrs}`` -- plus the
        committed blocks appended as ``{op: "commit", block, inputs: [state], ...}`` entries. This is
        the machine-readable counterpart of :meth:`dump_operator_ir`: a macro-built Program and the
        equivalent manual Program expose the SAME node list, so a caller can inspect what a
        ``pops.lib.time`` macro generated without reaching into ``self._values``. Read-only: it copies
        each node's ``attrs`` (never the live dict) and mutates nothing.
        """
        def _attr(val: Any) -> Any:
            # Keep the report array-free and JSON-friendly: reference Values / sub-blocks by name/id,
            # pass scalars through, summarize anything else by its type name.
            if isinstance(val, ProgramValue):
                return "#%d" % val.id
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            if isinstance(val, (list, tuple)):
                return [_attr(x) for x in val]
            return type(val).__name__

        from pops.time._program.serialization import _serialize_field_context
        nodes = []
        for v in self._values:
            # logical_shape / source_location are INSPECTION-ONLY (ADC-530): derived on demand and
            # never serialized into the IR / the hash, so surfacing them here cannot change a .so cache
            # key. source_location is None unless the Program enabled capture_source_locations().
            nodes.append({
                "name": v.name, "op": v.op, "vtype": v.vtype,
                "block": handle_data(v.block) if v.block is not None else None,
                "state": handle_data(v.state_ref) if v.state_ref is not None else None,
                "inputs": [self._canonical_value(i).name for i in v.inputs],
                "attrs": {k: _attr(val) for k, val in v.attrs.items()},
                "field_context": (_serialize_field_context(v.field_context)
                                  if v.field_context is not None else None),
                "logical_shape": v.logical_shape,
                "source_location": v.source_location,
            })
        for state_ref, st in self._commits.items():
            nodes.append({"name": st.name, "op": "commit", "vtype": st.vtype,
                          "block": handle_data(state_ref.block_ref),
                          "state": handle_data(state_ref),
                          "inputs": [st.name], "attrs": {},
                          "field_context": (_serialize_field_context(st.field_context)
                                            if st.field_context is not None else None),
                          "logical_shape": st.logical_shape, "source_location": st.source_location})
        return nodes

    def scratch_liveness(self) -> Any:
        """Per-scratch LIVE RANGES over the linear step-body order (Spec 3 s28 scratch-liveness
        analysis, ADC-465). A REPORT, not a transform: it never rewrites the IR.

        Each scratch-allocating flat node (rhs / source / apply / linear_combine / ... -- the
        ``_SCRATCH_OPS`` set, the ops that allocate a step-body MultiFab) owns one buffer. Its live
        range is ``[def_index, last_use_index]``: the node's own position in the flat list, to the
        position of the LAST flat node that reads it (through a flat input OR an affine combine term).
        A scratch read only inside a control-flow / matrix-free sub-block (v1 does not descend) is
        conservatively live to the END of the body. Returns a list of dicts (one per scratch) with the
        node name, op, def/last-use indices and the live span -- the raw material for buffer reuse."""
        order = {v.id: i for i, v in enumerate(self._values)}
        last_use = {}  # scratch node id -> last flat index that reads it
        end = len(self._values) - 1
        for i, v in enumerate(self._values):
            # Flat inputs + affine-term refs + sub-block closed-over refs are all "reads".
            reads = set(inp.id for inp in v.inputs)
            for key in ("cond", "body", "residual", "iterate", "guess",
                        "apply_result", "apply_in", "apply_out"):
                ref = v.attrs.get(key)
                if isinstance(ref, ProgramValue):
                    reads.add(ref.id)
                elif isinstance(ref, _Affine):
                    reads.update(term.id for term, _ in ref.terms)
            for rid in reads:
                if rid in order:
                    last_use[rid] = max(last_use.get(rid, order[rid]), i)
            # A sub-block-owning op may read a scratch inside its body (v1 never inspects it): keep every
            # such closed-over scratch live to the END (conservative, never under-estimates the span).
            if v.op in self._SUBBLOCK_OPS:
                for w in self._subblock_value_refs(v):
                    if w.id in order:
                        last_use[w.id] = end
        # A committed scratch is read by the final commit copy (after the last flat node).
        for state in self._commits.values():
            if state.id in order:
                last_use[state.id] = max(last_use.get(state.id, order[state.id]), end + 1)
        out = []
        for v in self._values:
            if v.op not in self._SCRATCH_OPS:
                continue
            d = order[v.id]
            lu = last_use.get(v.id, d)  # an unused scratch is live only at its own def
            out.append({"name": v.name, "op": v.op,
                        "block": block_name(v.block) if v.block is not None else None,
                        "def_index": d, "last_use_index": lu, "live_span": lu - d})
        return out

    def buffer_reuse_report(self) -> Any:
        """Buffer-reuse opportunities from the scratch live ranges (Spec 3 s28 buffer reuse, ADC-465).
        A REPORT, not a transform: the codegen may keep separate buffers, but the memory ESTIMATE
        reflects reuse.

        Greedy left-to-right register-allocation over the live ranges from :meth:`scratch_liveness`:
        scratches sorted by def index; a free buffer (one whose last use precedes the current
        scratch's def) is reused, else a new buffer is allocated. Two scratches share a buffer only
        when their live ranges are DISJOINT (the earlier one's last read strictly precedes the later
        one's def), so reuse can never alias two simultaneously-live values -- this is why it is a
        sound ESTIMATE of the minimum buffer count, independent of whether the codegen actually
        reuses. Returns ``{"scratch_count", "buffer_count", "reused", "assignment"}`` where
        ``reused`` is how many scratches landed on a recycled buffer and ``assignment`` maps each
        scratch name to its buffer index."""
        ranges = sorted(self.scratch_liveness(), key=lambda r: r["def_index"])
        free = []          # buffer index -> last_use_index of its current occupant
        assignment = {}
        reused = 0
        n_buffers = 0
        for r in ranges:
            slot = None
            for b in range(n_buffers):
                if free[b] < r["def_index"]:  # this buffer's occupant is dead before r is defined
                    slot = b
                    break
            if slot is None:
                slot = n_buffers
                n_buffers += 1
                free.append(r["last_use_index"])
            else:
                free[slot] = r["last_use_index"]
                reused += 1
            assignment[r["name"]] = slot
        return {"scratch_count": len(ranges), "buffer_count": n_buffers,
                "reused": reused, "assignment": assignment}

    def estimate(self) -> Any:
        """Static memory-traffic + kernel-count estimate over the lowered IR (Spec 3 s28, ADC-465).
        A REPORT, not a transform. All figures are STRUCTURAL (counted off the IR / the lowering),
        in UNITS of one block-state field traversal (n_cons * n_cells * 8 bytes) -- the absolute byte
        count needs the runtime grid, so the estimate is grid-relative: a "field" is one full
        state-sized buffer pass. The measured GPU kernel count / wall time is a ROMEO profile; this is
        the host-side static prediction the GPU detectors threshold on.

        Returns a dict:
          - ``kernel_count``: total per-cell + heavy kernel launches (one per scratch-writing op, plus
            the elliptic / Krylov solves);
          - ``small_kernels``: per-cell kernels that touch only a handful of buffers (launch-overhead
            bound on a GPU);
          - ``heavy_kernels``: elliptic / Krylov solves (each many internal kernels);
          - ``scratch_count`` / ``buffer_count`` / ``buffers_saved``: from the buffer-reuse report;
          - ``field_reads`` / ``field_writes`` / ``traffic_fields``: field-sized buffer passes
            (reads + writes), the proxy for memory traffic."""
        reuse = self.buffer_reuse_report()
        kernel_count = small = heavy = 0
        reads = writes = 0
        for v in self._values:
            if v.op in self._HEAVY_KERNEL_OPS:
                heavy += 1
                kernel_count += 1
                # An elliptic / Krylov solve reads the state and writes the shared aux/field: count one
                # read + one write as a coarse field-traffic proxy (the internal V-cycle traffic is
                # solver-dependent and out of scope for a structural estimate).
                reads += 1
                writes += 1
                continue
            if v.op in self._PERCELL_KERNEL_OPS:
                kernel_count += 1
                # One write (its result scratch) + one read per distinct field input it consumes.
                in_fields = len({i.id for i in v.inputs if i.is_field()})
                # An affine combine also reads each of its terms.
                aff = v.attrs.get("coeffs")
                if aff is not None:
                    in_fields = max(in_fields, len(aff))
                writes += 1 if v.op in self._SCRATCH_OPS else 0
                reads += in_fields
                # A "small" kernel touches few buffers (the GPU launch-overhead regime): a per-cell
                # source / where / cell_compare / a 1-2 term combine.
                if in_fields <= 2:
                    small += 1
        traffic = reads + writes
        return {"kernel_count": kernel_count, "small_kernels": small, "heavy_kernels": heavy,
                "scratch_count": reuse["scratch_count"], "buffer_count": reuse["buffer_count"],
                "buffers_saved": reuse["scratch_count"] - reuse["buffer_count"],
                "field_reads": reads, "field_writes": writes, "traffic_fields": traffic}


    def gpu_detectors(self) -> Any:
        """Flag GPU anti-patterns from the static :meth:`estimate` (Spec 3 s28, ADC-465). Returns a
        list of warning dicts ``{"detector", "value", "threshold", "message"}`` -- NEVER raises (an
        analysis report, not a hard error). Detects too-many-small-kernels (launch overhead),
        too-many-scratches (buffer pressure / allocator churn) and excessive memory traffic
        (bandwidth bound). An empty list means the IR trips no host-side GPU heuristic; the measured
        kernel count / occupancy is validated on ROMEO."""
        est = self.estimate()
        warnings = []
        checks = (
            ("too_many_small_kernels", est["small_kernels"], self._GPU_MAX_SMALL_KERNELS,
             "many small per-cell kernels: launch overhead may dominate on a GPU; consider fusing"),
            ("too_many_scratches", est["buffer_count"], self._GPU_MAX_SCRATCHES,
             "many live scratch buffers: GPU memory pressure / allocator churn; consider reuse"),
            ("excessive_memory_traffic", est["traffic_fields"], self._GPU_MAX_TRAFFIC_FIELDS,
             "high field-sized memory traffic: likely bandwidth bound on a GPU"),
        )
        for name, value, thresh, msg in checks:
            if value > thresh:
                warnings.append({"detector": name, "value": value, "threshold": thresh,
                                 "message": msg})
        return warnings

    def estimate_report(self) -> Any:
        """A human-readable cost report: the static memory-traffic / kernel-count :meth:`estimate`,
        the buffer-reuse summary and any GPU detector warnings (Spec 3 s28 inspection surface,
        ADC-465). A report only -- ``self`` is never mutated."""
        est = self.estimate()
        lines = ["# cost estimate for Program %r (static, grid-relative)" % self.name,
                 "  kernels        : %d (%d small, %d heavy elliptic/Krylov)"
                 % (est["kernel_count"], est["small_kernels"], est["heavy_kernels"]),
                 "  scratch buffers: %d allocated, %d after reuse (%d saved)"
                 % (est["scratch_count"], est["buffer_count"], est["buffers_saved"]),
                 "  memory traffic : %d field-passes (%d read, %d write)"
                 % (est["traffic_fields"], est["field_reads"], est["field_writes"])]
        warnings = self.gpu_detectors()
        if warnings:
            lines.append("  GPU detectors  :")
            for w in warnings:
                lines.append("    [warn] %s (%d > %d): %s"
                             % (w["detector"], w["value"], w["threshold"], w["message"]))
        else:
            lines.append("  GPU detectors  : none tripped (host-side heuristic)")
        return "\n".join(lines)
