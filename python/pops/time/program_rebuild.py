"""Lossless Program cloning, alias remapping and compiled-boundary detachment."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Any

from pops.time.schedule import Schedule
from pops.time.values import ProgramValue, _Affine
from pops.time.points import Clock, StagePoint, TimePoint
from pops.provenance import ProvenanceRecord


def rebuild_program(
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
    """Clone this Program into a fresh one keeping the flat nodes for which ``keep(v)`` is true,
    renumbering surviving ids to a contiguous 0.. range in original order. Sub-blocks are cloned
    wholesale (never filtered). The clone reproduces the IR identity of an equivalent hand-built
    Program (same serialization), so it is byte-identical when nothing was dropped.

    @p alias (optional) maps a DROPPED node id -> the kept representative node id it should be
    replaced by (the CSE / redundant-solve passes use it to rewire every use of a duplicate onto its
    survivor). Every reference -- a flat input, an attr-borne ProgramValue / affine ref, a commit target --
    is resolved THROUGH this alias before id lookup, so a dropped node never leaves a dangling
    reference. A dropped node MUST have an alias entry (the passes guarantee a representative whose
    id < the duplicate's, hence already cloned); a kept node maps to itself. Without an alias map
    the behavior is the historical drop-only rebuild.

    ``reference_of`` is the compiled-boundary hook: when supplied, every semantic Handle in
    nodes, metadata, field provenance, histories, commits and temporal tables is replaced by
    that canonical detached value. ``retain_operator_registries=False`` then removes the
    authoring-only registry graph. Ordinary optimization passes use the defaults unchanged.
    """
    if reference_of is None:
        def _identity(value: Any) -> Any:
            return value
        reference_of = _identity
    elif not callable(reference_of):
        raise TypeError("Program._rebuild reference_of must be callable or None")
    if not isinstance(retain_operator_registries, bool):
        raise TypeError("Program._rebuild retain_operator_registries must be bool")
    if not isinstance(canonical_owner, bool):
        raise TypeError("Program._rebuild canonical_owner must be bool")
    out = type(self)(self.name)
    if canonical_owner:
        object.__setattr__(out, "_owner_path", out.owner_path.canonical())
        case_owner = getattr(self, "_case_owner_path", None)
        object.__setattr__(
            out,
            "_case_owner_path",
            None if case_owner is None else case_owner.canonical(),
        )
        object.__setattr__(out, "clock", Clock("macro", owner=out.owner_path))
    out.dt = self.dt
    out._step_strategy = getattr(self, "_step_strategy", None)
    out._transaction_stores = tuple(getattr(self, "_transaction_stores", ()))
    out._acceptance_guards = tuple(getattr(self, "_acceptance_guards", ()))
    clock_map = {self.clock: out.clock}

    def remap_clock(clock: Clock) -> Clock:
        mapped = clock_map.get(clock)
        if mapped is None:
            mapped = Clock(clock.name, owner=out.owner_path)
            clock_map[clock] = mapped
        return mapped

    def remap_point(point: Any) -> Any:
        if type(point) is TimePoint:
            return TimePoint(remap_clock(point.clock), point.offset, step=point.step)
        if type(point) is StagePoint:
            return StagePoint(point.name, {
                partition: remap_point(coordinate)
                for partition, coordinate in point.partitions.items()
            })
        raise TypeError("Program rebuild encountered a value without an exact evaluation point")
    out._state_spaces = {
        reference_of(state_ref): space
        for state_ref, space in getattr(self, "_state_spaces", {}).items()
    }
    out._histories = dict(self._histories)
    out._histories_ncomp = dict(getattr(self, "_histories_ncomp", {}))
    out._history_spaces = dict(getattr(self, "_history_spaces", {}))
    out._history_blocks = {
        name: reference_of(block)
        for name, block in getattr(self, "_history_blocks", {}).items()
    }
    out._history_state_refs = {
        name: reference_of(state)
        for name, state in getattr(self, "_history_state_refs", {}).items()
    }
    from pops.time.history_persistence import HistoryPersistence
    out._history_persistence = {}
    for name, (depth, policy) in getattr(self, "_history_persistence", {}).items():
        copied = HistoryPersistence.from_manifest(policy.to_manifest())
        if hasattr(copied, "freeze"):
            copied.freeze()
        out._history_persistence[name] = (depth, copied)
    out._operator_registries = (
        dict(self._operator_registries) if retain_operator_registries else {}
    )
    out._default_state_spaces = (
        dict(self._default_state_spaces) if retain_operator_registries else {}
    )
    out._default_field_spaces = (
        dict(self._default_field_spaces) if retain_operator_registries else {}
    )
    # ProgramValue deliberately has no hash: equality authors an Equation.  Every rewrite map is
    # therefore indexed by the stable SSA id, never by a symbolic object (which would otherwise
    # make dict membership invoke forbidden symbolic equality/truth semantics).
    idmap = {}  # old SSA id -> new ProgramValue
    by_id = {v.id: v for v in self._values}
    for v in self._values:  # sub-block ops too, so an alias to a sub-block-internal id resolves
        for w in self._subblock_value_refs(v):
            by_id.setdefault(w.id, w)
    alias = alias or {}
    region_map = {0: 0}

    def mapped_region(region: int) -> int:
        if region not in region_map:
            # Region ids are semantic authoring tokens, not traversal-order ordinals. Preserving
            # them makes a no-drop detach byte-identical even when deps(v) visits a nested branch
            # before its owning node; optimization remains deterministic because removed regions
            # may leave harmless gaps.
            region_map[region] = region
            out._next_region = max(out._next_region, region + 1)
        return region_map[region]

    def rep(v: Any) -> Any:
        """Follow @p v through the alias chain to the surviving representative ProgramValue (identity for a
        kept node). The passes only alias onto an EARLIER, kept node, so the chain terminates."""
        seen = set()
        while v.id in alias and alias[v.id] != v.id:
            if v.id in seen:  # defensive: never loop on a malformed alias map
                break
            seen.add(v.id)
            v = by_id[alias[v.id]]
        return v

    def provenance_inputs(v: Any) -> tuple[ProvenanceRecord, ...]:
        """Ordered source lineage for a rebuilt representative, including aliased duplicates."""
        representative = rep(v).id
        source_values = [v]
        for candidate in sorted(by_id.values(), key=lambda item: item.id):
            if candidate.id != v.id and rep(candidate).id == representative:
                source_values.append(candidate)
        return tuple(item.provenance for item in source_values)

    def clone_block(block: Any, region_hint: Any = None) -> Any:
        copied = [clone(w) for w in block]
        regions = {mapped_region(w.region) for w in block}
        if not regions and region_hint is not None:
            regions.add(mapped_region(region_hint))
        if not regions:
            entry = getattr(self, "_recording_regions", {}).get(id(block))
            if entry is not None and entry[0] is block:
                regions.add(mapped_region(entry[1]))
        if len(regions) == 1:
            region = next(iter(regions))
            if region != 0:
                out._recording_regions[id(copied)] = (copied, region)
        return copied

    def remap(ref: Any) -> Any:
        if isinstance(ref, ProgramValue):
            return idmap[rep(ref).id]
        if isinstance(ref, _Affine):
            return _Affine([(idmap[rep(v).id], c) for v, c in ref.terms])
        return ref

    def remap_source(source: Any) -> Any:
        if isinstance(source, int) and source in by_id:
            value = rep(by_id[source])
            if value.id not in idmap:
                clone(value)
            return idmap[value.id].id
        return source

    def remap_provenance(provenance: Any) -> Any:
        from pops.time.field_context import remap_field_provenance
        return remap_field_provenance(
            provenance,
            remap_source,
            remap_reference=reference_of,
        )

    def remap_metadata(value: Any) -> Any:
        """Recursively reown ProgramValue and semantic Handle leaves in IR metadata."""
        if isinstance(value, (ProgramValue, _Affine)):
            return remap(value)
        from pops.model.handles import Handle
        if isinstance(value, Handle):
            return reference_of(value)
        from pops.time.field_context import FieldContext, FieldReadProvenance
        if isinstance(value, (FieldContext, FieldReadProvenance)):
            return remap_provenance(value)
        if isinstance(value, Schedule):
            from dataclasses import replace
            domain = replace(
                value.domain,
                clock=remap_clock(value.clock),
                at=remap_point(value.domain.at) if value.domain.at is not None else None,
            )
            trigger = replace(value.trigger, domain=domain)
            from pops.time.schedule import When
            if type(trigger) is When:
                trigger = replace(trigger, condition=remap_metadata(trigger.condition))
            return Schedule(trigger, off=value.off)
        if getattr(value, "__pops_ir_immutable__", False) is True:
            return value
        if isinstance(value, Mapping):
            return {
                remap_metadata(key): remap_metadata(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return tuple(remap_metadata(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return frozenset(remap_metadata(item) for item in value)
        if value is None or isinstance(
                value, (bool, int, float, complex, str, bytes, Decimal, Fraction, Enum)):
            return value
        if retain_operator_registries:
            # Optimization/authoring rebuilds preserve extension metadata. The compiled-boundary
            # rebuild below is intentionally stricter because preserving a foreign record there
            # could retain its mutable builder or registry graph in the public artifact.
            return value
        raise TypeError(
            "compiled Program metadata contains unsupported mutable value %s.%s; lower it to "
            "immutable scalar/container/Handle data or declare __pops_ir_immutable__ = True"
            % (type(value).__module__, type(value).__qualname__)
        )

    def clone_attrs(v: Any) -> Any:
        attrs = {}
        for key, val in v.attrs.items():
            if key in ("cond_block", "body_block", "apply_block", "residual_block",
                       "true_block", "false_block"):
                region_key = key.replace("_block", "_region")
                attrs[key] = (clone_block(val, v.attrs.get(region_key))
                              if val is not None else None)
            elif key in ("cond_region", "body_region", "apply_region", "residual_region",
                         "true_region", "false_region"):
                attrs[key] = mapped_region(val)
            elif key in ("cond", "body", "residual", "iterate", "guess",
                         "apply_result", "apply_in", "apply_out",
                         "true_result", "false_result"):
                attrs[key] = remap(val)
            else:
                attrs[key] = remap_metadata(val)
        return attrs

    def deps(v: Any) -> Any:
        """The values v depends on that must be cloned (hence id-assigned) BEFORE v, in their
        ORIGINAL creation order. A fresh build records the inputs and most sub-blocks before the
        owning node, BUT a matrix_free_operator is created (its node id assigned) BEFORE
        ``set_apply`` records its apply sub-block -- the node id precedes the sub-block ids. Ordering
        every dependency by its original id (ascending) reproduces the build order verbatim for both
        shapes, so a no-drop clone is byte-identical (same renumbering) rather than reordering the
        matrix_free_operator node after its own sub-block. Each input is resolved THROUGH the alias
        map, so a dropped duplicate is replaced by its (already-earlier) representative."""
        seen = []
        for inp in v.inputs:
            seen.append(rep(inp))
        for key in ("cond_block", "body_block", "apply_block", "residual_block",
                    "true_block", "false_block"):
            block = v.attrs.get(key)
            if block:
                seen.extend(block)
        # A matrix_free_operator's sub-block ops are created AFTER the node, so they must NOT be
        # forced ahead of it; an input / control-flow body is created BEFORE. Keep only the deps
        # whose original id precedes v's (the genuine predecessors) and visit them id-ascending.
        return sorted((w for w in seen if w.id < v.id), key=lambda w: w.id)

    def clone(v: Any) -> Any:
        if v.id in idmap:
            return idmap[v.id]
        # Assign new ids in ORIGINAL creation order: clone every predecessor (id < v.id) first,
        # id-ascending, then v, then any sub-block op created AFTER v (e.g. a matrix_free_operator's
        # apply ops, whose original ids exceed the operator node's). Inputs / attr refs are remapped
        # through idmap after alias resolution (every referenced surviving value is mappable on its
        # own clone).
        for w in deps(v):
            clone(w)
        vid = out._next_id
        out._next_id += 1
        # Reserve the owning node's region before clone_attrs recursively maps branch/sub-block
        # regions. Parent-first allocation is part of exact rebuild identity for nested branches.
        node_region = mapped_region(v.region)
        new_inputs = [idmap[rep(i).id] for i in v.inputs]
        field_context = v.field_context
        if field_context is not None:
            field_context = remap_provenance(field_context)
        nv = ProgramValue(
            out,
            vid,
            v.vtype,
            v.op,
            new_inputs,
            clone_attrs(v),
            v.name,
            reference_of(v.block),
            space=v.space if space_of is None else space_of(v),
            source_location=v.source_location,
            field_context=field_context,
            region=node_region,
            state_ref=reference_of(v.state_ref),
            point=remap_point(v.point),
            provenance=ProvenanceRecord.derive(
                provenance_inputs(v), transformation=transformation, owner=out.owner_path),
        )
        out._issued_values[id(nv)] = nv
        idmap[v.id] = nv
        return nv

    # Clone all surviving flat nodes (and, transitively, their sub-block ops and any later-created
    # sub-block ops) in ascending original id, so the contiguous renumbering matches the original
    # build order exactly -- a no-op clone is byte-for-byte identical.
    kept = sorted((v for v in self._values if keep(v)), key=lambda v: v.id)
    for v in kept:
        clone(v)
    out._values = [idmap[v.id] for v in kept]
    out._commits = {reference_of(state_ref): idmap[rep(value).id]
                    for state_ref, value in self._commits.items()}
    out._region_imports = {
        mapped_region(destination): {mapped_region(source) for source in sources}
        for destination, sources in getattr(self, "_region_imports", {}).items()
    }
    if self._dt_bound is not None:
        sub, result = self._dt_bound
        cloned_sub = clone_block(sub)
        out._dt_bound = (cloned_sub, idmap[rep(result).id])
    self._rebuild_time_handle_tables(
        out, idmap, rep, reference_of=reference_of)
    return out


__all__ = ["rebuild_program"]
