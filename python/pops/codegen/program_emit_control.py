"""pops.codegen.program_emit_control : the body walk + control-flow op emitters.

Extracted verbatim from ``pops.codegen.program_codegen`` so the Program -> C++ lowering
fits the Spec-4 file-size budget.  ``_emit_body`` is the two-phase body walk;
``_emit_while`` / ``_emit_range`` / ``_emit_branch`` lower the control-flow ops (they re-run
the per-op lowering on their sub-blocks); ``_coupled_rate_components`` / ``_walk_expr``
resolve and scan a coupled_rate node.  The op dispatcher ``_emit_op`` lives in
``program_emit_ops`` and is imported LAZILY inside the functions below to break the
``ops`` <-> ``control`` recursion cycle at import time (it resolves fine at call time).
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

import json
from pops.time.references import block_name


def _coupled_rate_components(program: Any, v: Any, authority: Any = None) -> dict:
    """Resolve a ``coupled_rate`` node @p v to its per-block component formulas (Spec 3 criterion
    27, ADC-457), validated for the cons-only MVP. Returns ``{block: [Expr, ...]}`` (one formula
    per component of that block's StateSpace).

    The component formulas live in the BOUND operator's body (``op.body`` = the ``expr=`` dict
    passed to ``Module.operator``), reachable through the registry the node's ``operator`` attr
    names; the input states' cons names come from each input value's StateSpace (set by
    ``T.state(block, U)``). Raises a clear NotImplementedError naming ADC-457 when a coupled_rate
    cannot lower in this MVP: no bound registry, no operator body, a block whose component count
    does not match its StateSpace, or a formula referencing a non-cons (prim / aux) Var."""
    from pops._ir.expr import Var
    op_name = v.attrs["operator"]
    from pops.time.operator_resolution import resolve_operator_handle
    operator_handle = v.attrs.get("operator_handle")
    if operator_handle is not None and getattr(program, "_operator_registries", None):
        op = resolve_operator_handle(
            program, operator_handle, where="coupled_rate codegen", values=v.inputs)
    elif operator_handle is not None:
        # Compiled Programs deliberately detach mutable authoring registries.  Whole-system
        # compilation retains the immutable source Module in ProgramModelGraph, so resolve the
        # canonical owner-qualified handle against that authority instead of reattaching a registry.
        from pops.codegen.program_models import ProgramModelGraph
        if type(authority) is not ProgramModelGraph:
            raise ValueError(
                "coupled_rate codegen: detached Program requires a ProgramModelGraph authority")
        source = authority.source_module_for_owner(operator_handle.owner_path)
        if source.owner_path.canonical() != operator_handle.owner_path.canonical():
            raise ValueError("coupled_rate codegen: source Module owner identity drift")
        matches = tuple(
            candidate for candidate in source.operator_registry()
            if candidate.name == operator_handle.local_id)
        if len(matches) != 1:
            raise ValueError(
                "coupled_rate codegen: operator %r is absent or ambiguous in its source Module"
                % operator_handle.local_id)
        op = matches[0]
        if op.kind != operator_handle.kind or op.kind != "coupled_rate":
            raise ValueError(
                "coupled_rate codegen: operator handle kind differs from its source Module")
    else:
        raise ValueError(
            "coupled_rate codegen: node %r lacks its owner-qualified OperatorHandle" % v.name)
    expr = op.body
    if not isinstance(expr, Mapping):
        raise NotImplementedError(
            "the coupled_rate kernel codegen (ADC-457) needs operator %r to carry its per-block "
            "component formulas as an expr={block: [Expr, ...]} dict (got %r); a decorator-body "
            "coupled_rate is a later phase (node %r)" % (op_name, type(expr).__name__, v.name))
    # Each coupled_rate_out block must own one input state (its rate scratch is shaped like that
    # block's state) whose StateSpace gives the component count + cons names.
    by_block = {block_name(state.block): state for state in v.inputs}
    components = {}
    for blk, comps in expr.items():
        state_in = by_block.get(blk)
        if state_in is None or getattr(state_in, "space", None) is None:
            raise NotImplementedError(
                "the coupled_rate kernel codegen (ADC-457) needs every output block to map to an "
                "input State declared through T.state(block[U]) with typed space metadata; operator %r "
                "block %r has none (node %r)" % (op_name, blk, v.name))
        ncons = len(state_in.space.components)
        if len(comps) != ncons:
            raise NotImplementedError(
                "coupled_rate operator %r block %r emits %d component formulas but its StateSpace "
                "has %d components; the rate must be full-rank over the block state (ADC-457, "
                "node %r)" % (op_name, blk, len(comps), ncons, v.name))
        for e in comps:
            for node in _walk_expr(e):
                if isinstance(node, Var) and node.kind != "cons":
                    raise NotImplementedError(
                        "coupled_rate formulas referencing prim/aux vars are deferred (ADC-457): "
                        "operator %r block %r references %s var %r; the MVP per-cell binding is "
                        "cons-only (node %r)" % (op_name, blk, node.kind, node.name, v.name))
        components[state_in.block] = list(comps)
    # Every cons var a formula references must be a coordinate of SOME input state (an output block
    # OR a read-only catalyst input). Qualified coordinates are always available. A bare component
    # name remains accepted only when it identifies exactly one input state; otherwise it would
    # silently select one of several physical quantities with the same display name.
    from collections import Counter
    from pops.model.state_symbols import state_component_symbol

    spaces = [s.space for s in v.inputs if getattr(s, "space", None) is not None]
    counts = Counter(component for space in spaces for component in space.components)
    all_cons = {
        state_component_symbol(space, component)
        for space in spaces for component in space.components
    }
    all_cons.update(component for component, count in counts.items() if count == 1)
    referenced = set()
    for comps in components.values():
        for e in comps:
            referenced |= e.deps()
    ambiguous = sorted(
        component for component, count in counts.items()
        if count > 1 and component in referenced)
    if ambiguous:
        raise ValueError(
            "coupled_rate operator %r references ambiguous bare component(s) %s; obtain exact "
            "coordinates with module.state_symbols(state_space) (node %r)"
            % (op_name, ambiguous, v.name))
    missing = referenced - all_cons
    if missing:
        raise NotImplementedError(
            "coupled_rate operator %r references cons var(s) %s that are a component of no input "
            "state; declare them via T.state(block[U]) or fix the formula (ADC-457, node %r)"
            % (op_name, sorted(missing), v.name))
    return components

def _walk_expr(e: Any) -> Any:
    """Yield every node of a dsl Expr tree (used to scan a coupled_rate formula for non-cons Vars)."""
    from pops._ir.visitors import _children
    stack = [e]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(_children(node))


def _groupable_default_rhs(value: Any) -> bool:
    """Whether one RHS can enter the native simultaneous-interface batch.

    Named flux kernels and named source additions remain independent SSA work.  The default full or
    flux-only finite-volume residual is the exact route for which the runtime owns an interface-
    omitting closure and a shared NumericalFlux insertion.
    """
    if value.op != "rhs" or not value.attrs.get("flux", True):
        return False
    fluxes = value.attrs.get("fluxes")
    if fluxes and tuple(fluxes) != ("default",):
        return False
    requested = value.attrs.get("sources")
    return not any(source != "default" for source in (requested or ()))


def _stage_fraction(value: Any) -> Fraction:
    point = value.point
    if not hasattr(point, "offset"):
        try:
            point = point.time
        except ValueError:
            point = point.time_for("explicit")
    return Fraction(point.step) + Fraction(point.offset.to_python())


def _emit_contiguous_rhs_group(
        values: list[Any], block_idx: Mapping[Any, int], var: dict[Any, str],
        lines: list[str], group_identity: int) -> None:
    """Emit one complete same-StagePoint residual group before any result is consumable."""
    from pops.codegen.program_emit_ops import _required_block_index

    stage = _stage_fraction(values[0])
    lines.append("ctx.set_stage_time(%d, %d);" % (stage.numerator, stage.denominator))
    requests = []
    for value in values:
        state = value.inputs[0]
        var[value.id] = "r%d" % value.id
        lines.append("pops::MultiFab %s = ctx.rhs_scratch_like(%s);"
                     % (var[value.id], var[state.id]))
        index = _required_block_index(
            block_idx, value.block, "emit simultaneous rhs %r" % value.name)
        requested = value.attrs.get("sources")
        default_source = requested is None or "default" in requested
        requests.append("{%d, &%s, &%s, %d, %d}" % (
            index, var[state.id], var[value.id], int(value.id), 0 if default_source else 1))
    lines.append("ctx.rhs_group(%d, {%s});" % (group_identity, ", ".join(requests)))

def _emit_body(program: Any, model: Any = None, target: Any = "system",
               field_plans: Any = None) -> tuple:
    """Generate the C++ of the install function in TWO phases (each list indented uniformly by the
    template). Assumes `_check_lowerable` has passed. @p model supplies the symbolic coefficients of
    the Phase-4b source / apply / solve_local_linear ops. Returns ``(prelude, body)``:

      - ``prelude``: INSTALL-TIME C++ (before ``ctx.install``) -- persistent scratch fields (held
        via ``std::shared_ptr`` so they outlive the install call and are reused across every step
        and every Krylov iteration) and the matrix-free apply lambdas. Captured by value into the
        step closure (shared_ptr / lambda / ctx all copy cheaply).
      - ``body``: the STEP closure body (one macro-step over dt).

    Multi-block (ADC-426): the SSA walk allocates a per-block base (``ctx.state(idx)`` for each
    declared block) and routes every op to ITS block's index via ``_block_indices`` / ``v.block``.
    Each committed block's final value is copied into that block's state (a scratch commit) or was
    written in place (a linear_combine commit). A single block reduces to the historical lowering."""
    from pops.codegen.program_emit_ops import _emit_op
    block_idx = program._block_indices()
    # The first-declared state Value per block: the "base" any op of that block clones / commits into.
    bases = {}
    for v in program._values:
        if v.op == "state" and v.block not in bases:
            bases[v.block] = v
    # IR value id -> C++ token: a MultiFab variable name (states / RHS scratches), a scalar variable
    # name (reductions, ``s{id}``) or a parenthesized boolean expression (compares).
    var = {}
    prelude = []
    lines = []
    # ``var`` also carries emission-local schedule/coupled scratch tokens under tuple keys. Nothing
    # is written back into Program authoring state, so codegen remains pure after deep freeze.
    # Every final value remains provisional scratch until the single tail commit group.  In
    # particular a committed linear_combine must never overwrite live state while later operators
    # are still executing.
    committed_ids = frozenset()
    # Multistep histories (ADC-406a): register each declared history at its MAX lag FIRST (a
    # registration-only call, NOT a read -- a read before the first store fails loud), so the ring
    # depth is locked before any store. The first ctx.store_history then cold-start-fills every
    # (already-allocated) slot -- step 0 reads the same value at every lag and the scheme degenerates
    # to a one-step method. register_history is idempotent (no-op once registered).
    # A NARROW ring (ADC-427: the 1-component condensed-Schur phi^n carry) declares its slot ncomp here
    # before any read. AMR always emits the six-argument identity-qualified form; owner-less rings are
    # rejected during lowering instead of silently binding block zero.  AMR registration also
    # carries the logical state and field-space identities used by clock-qualified history slots.
    histories_ncomp = getattr(program, "_histories_ncomp", {})
    temporal = program.temporal_manifest()
    prelude.append("ctx.configure_primary_clock(%s);" % json.dumps(temporal["primary_clock"]))
    for relation in temporal["subcycles"]:
        prelude.append(
            "ctx.declare_clock_relation(%s, %s, %d);"
            % (json.dumps(relation["parent_clock"]), json.dumps(relation["child_clock"]),
               int(relation["count"])))
    history_manifest = {row["name"]: row for row in temporal["histories"]}
    for name, lag in sorted(program._histories.items()):
        ncomp = histories_ncomp.get(name)
        owner = getattr(program, "_history_blocks", {}).get(name)
        owner_index = block_idx.get(owner) if owner is not None else None
        if target == "amr_system" and owner_index is None:
            raise ValueError(
                "AMR history %r requires explicit block owner provenance" % name)
        state_ref = getattr(program, "_history_state_refs", {}).get(name)
        state_identity = (state_ref.qualified_id if state_ref is not None
                          else "scalar-history:" + name)
        space = getattr(program, "_history_spaces", {}).get(name)
        space_identity = (json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
                          if space is not None else "scalar-field")
        row = history_manifest[name]
        interpolation = json.dumps(
            row["interpolation"], sort_keys=True, separators=(",", ":"))
        lines.append("ctx.register_history(%s, %d, %d, %d, %s, %s, %s, %s);"
                     % (json.dumps(name), int(lag), -1 if ncomp is None else int(ncomp),
                        -1 if owner_index is None else int(owner_index),
                        json.dumps(state_identity), json.dumps(space_identity),
                        json.dumps(row["clock"]), json.dumps(interpolation)))
    values = list(program._values)
    index = 0
    # Group identities occupy compiler-reserved slots after the authored SSA namespace.  They are
    # deterministic, cannot alias a rate node, and keep BoundaryEvaluationPoint.stage faithful to
    # the atomic group while every RhsGroupRequest retains its own exact rate identity.
    next_group_identity = int(program._next_id)
    while index < len(values):
        v = values[index]
        if _groupable_default_rhs(v):
            end = index + 1
            while end < len(values) and _groupable_default_rhs(values[end]) \
                    and values[end].point == v.point:
                end += 1
            group = values[index:end]
            if len(group) > 1 and len({row.block for row in group}) == len(group) \
                    and all(row.inputs[0].id in var for row in group):
                _emit_contiguous_rhs_group(
                    group, block_idx, var, lines, next_group_identity)
                next_group_identity += 1
                index = end
                continue
        base = bases.get(v.block)  # the block-state value of THIS op's block (None: a scalar op)
        _emit_op(program, v, base, committed_ids, var, model, lines, prelude, block_idx,
                 target=target, field_plans=field_plans)
        index += 1
    # Each committed block: a scratch commit (solve_local_linear / solve_linear / a non-base
    # linear_combine wrote a scratch) is copied into the block state; a linear_combine commit already
    # wrote ctx.state(idx) in place (var == base), so its copy is a no-op (skipped).
    commit_pairs = []
    for state_ref, committed in program._commits.items():
        base = bases[state_ref.block_ref]
        commit_pairs.append("{&%s, &%s}" % (var[base.id], var[committed.id]))
    if commit_pairs:
        lines.append("ctx.commit_many({%s});" % ", ".join(commit_pairs))
    # Rotate the history rings ONCE at the very end of the step (after the commit), so the next step
    # reads lag k as the value k stores ago. Only emitted when the Program uses histories.
    if any(row["clock"] == program.clock.qualified_id for row in temporal["histories"]):
        lines.append("ctx.rotate_histories(%s);" % json.dumps(program.clock.qualified_id))
    prelude_src = "\n".join("  " + ln for ln in prelude)
    body_src = "\n".join("    " + ln for ln in lines)
    return prelude_src, body_src

def _emit_amr_hierarchy_bodies(program: Any, model: Any = None,
                               field_plans: Any = None) -> tuple | None:
    """Emit gather / solve-once / publish regions for one hierarchy-scoped linear solve.

    The transform keys only on the generic solve scope.  It does not recognize a physical scheme.
    Multiple hierarchy barriers are rejected until the region scheduler can represent them explicitly.
    """
    from pops.codegen.program_emit_ops import _emit_op
    solves = [v for v in program._values if v.op == "solve_linear"]
    scoped = [v for v in solves if v.attrs.get("scope") == "hierarchy"]
    if not scoped:
        return None
    if len(scoped) != 1 or len(solves) != 1:
        raise NotImplementedError(
            "AMR hierarchy-scoped lowering supports exactly one top-level solve_linear; multiple "
            "hierarchy barriers require an explicit region schedule")
    solve = scoped[0]
    if solve.attrs.get("hierarchy_solver") != "composite_tensor_fac":
        raise NotImplementedError(
            "AMR hierarchy-scoped solver %r is not lowerable; supported solver: "
            "CompositeTensorFAC()" % solve.attrs.get("hierarchy_solver"))
    split = next(index for index, value in enumerate(program._values) if value is solve)
    control = {"while", "range", "branch"}
    nested = [v.name for v in program._values if v.op in control]
    if nested:
        raise NotImplementedError(
            "a hierarchy-scoped solve must be a top-level barrier; control-flow regions %r cannot "
            "cross the gather/solve/publish boundary" % nested)

    # Values crossing from gather into the solve/publish regions must name storage whose lifetime is
    # wider than one per-level loop iteration.  This is the load-bearing refusal that prevents a local
    # C++ temporary from being referenced after the loop that declared it.  Alias ops are admitted only
    # when their storage input is itself portable.
    portable_ops = {"state", "history", "scalar_field", "matrix_free_operator", "condensed_coeffs"}
    portable = {v.id for v in program._values[:split] if v.op in portable_ops}
    changed = True
    aliases = {"solve_fields": 0, "condensed_rhs": 0, "laplacian": 0,
               "gradient": 0, "divergence": 0, "fill_boundary": 0}
    while changed:
        changed = False
        for value in program._values[:split]:
            source_index = aliases.get(value.op)
            if (source_index is not None and len(value.inputs) > source_index
                    and value.inputs[source_index].id in portable and value.id not in portable):
                portable.add(value.id)
                changed = True
    solve_inputs = [item.id for item in solve.inputs]
    missing_solve = [item for item in solve_inputs if item not in portable]
    if missing_solve:
        raise NotImplementedError(
            "hierarchy-scoped solve inputs must use persistent/state/history storage across the "
            "level barrier; non-portable value ids %r" % missing_solve)
    available = set(portable)
    available.add(solve.id)
    for value in program._values[split + 1:]:
        missing = [item.id for item in value.inputs if item.id not in available]
        if missing:
            raise NotImplementedError(
                "hierarchy publish node %r depends on gather-local value ids %r; materialize the "
                "value in persistent storage or move it after the hierarchy solve"
                % (value.name, missing))
        available.add(value.id)
    block_idx = program._block_indices()
    bases = {}
    for value in program._values:
        if value.op == "state" and value.block not in bases:
            bases[value.block] = value
    committed_ids = frozenset()
    binding_ops = frozenset({"state", "history", "scalar_field", "matrix_free_operator"})

    def registrations() -> list[str]:
        lines = []
        ncomps = getattr(program, "_histories_ncomp", {})
        manifests = {
            row["name"]: row for row in program.temporal_manifest()["histories"]}
        for name, lag in sorted(program._histories.items()):
            ncomp = ncomps.get(name)
            owner = program._history_blocks.get(name)
            if owner is None or owner not in block_idx:
                raise ValueError("AMR history %r requires explicit block owner provenance" % name)
            state_ref = program._history_state_refs.get(name)
            state_identity = (state_ref.qualified_id if state_ref is not None
                              else "scalar-history:" + name)
            space = program._history_spaces.get(name)
            space_identity = (json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
                              if space is not None else "scalar-field")
            row = manifests[name]
            interpolation = json.dumps(
                row["interpolation"], sort_keys=True, separators=(",", ":"))
            lines.append("ctx.register_history(%s, %d, %d, %d, %s, %s, %s, %s);"
                         % (json.dumps(name), int(lag), -1 if ncomp is None else int(ncomp),
                            int(block_idx[owner]), json.dumps(state_identity),
                            json.dumps(space_identity), json.dumps(row["clock"]),
                            json.dumps(interpolation)))
        return lines

    def emit_phase(phase: str) -> str:
        var = {}
        if phase == "solve":
            # The normal AMR body is the flat-topology branch and executes the authenticated apply
            # through BiCGStab. Only this gathered solve phase owns the refined direct-FAC call.
            var[("direct_hierarchy_solve", solve.id)] = True
        lines = registrations() if phase == "gather" else []
        for index, value in enumerate(program._values):
            emitted = []
            ignored_prelude = []
            _emit_op(program, value, bases.get(value.block), committed_ids, var, model, emitted,
                     ignored_prelude, block_idx, target="amr_system",
                     field_plans=field_plans or {})
            if phase == "gather":
                keep = index < split
            elif phase == "solve":
                keep = index == split or (index < split and value.op in binding_ops)
            else:
                keep = index > split or (index < split and value.op in binding_ops)
            if keep:
                lines.extend(emitted)
            if phase == "gather" and index == split:
                # The ordinary solve emitter seeds one level-local iterate immediately before the
                # solve.  A hierarchy solve instead needs one initial guess per level, gathered at the
                # same barrier as its coefficients/RHS.  Stage it in context-owned hierarchy storage;
                # the solve pass later consumes the complete tower exactly once.  This is also what
                # makes the ADC-427 scalar phi^n history carry compose on refined AMR.
                if value.attrs.get("has_guess"):
                    guess = value.inputs[2]
                    lines.append("ctx.stage_linear_initial_guess(%s);" % var[guess.id])
                else:
                    lines.append("ctx.stage_linear_initial_guess();")
        if phase == "publish":
            commit_pairs = []
            for state_ref, committed in program._commits.items():
                base = bases[state_ref.block_ref]
                commit_pairs.append("{&%s, &%s}" % (var[base.id], var[committed.id]))
            if commit_pairs:
                lines.append("ctx.commit_many({%s});" % ", ".join(commit_pairs))
            histories = program.temporal_manifest()["histories"]
            if any(row["clock"] == program.clock.qualified_id for row in histories):
                lines.append(
                    "ctx.rotate_histories(%s);" % json.dumps(program.clock.qualified_id))
        return "\n".join("    " + line for line in lines)

    return emit_phase("gather"), emit_phase("solve"), emit_phase("publish")


def _emit_while(program: Any, v: Any, base: Any, var: Any, model: Any, lines: Any,
                block_idx: Any = None, field_plans: Any = None,
                *, target: str = "system") -> None:
    """Lower a while op to an infinite C++ loop with a break (the condition re-evaluates each pass).
    The loop variable is a single MultiFab mutated IN PLACE across iterations; the cond / body sub-
    blocks re-run the per-op lowering each pass, with the loop-variable value id seeded to the loop
    var so their references resolve to it."""
    from pops.codegen.program_emit_ops import _emit_op
    loop_in = v.inputs[0]  # the initial loop-variable state
    x = "x%d" % v.id
    var[v.id] = x
    # Hoist + initialize the loop variable from the entry state (x <- loop_in).
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[loop_in.id]))
    lines.append("for (;;) {")
    # The sub-blocks see the loop variable in place of the entry-state value id (the body / cond were
    # built reading the loop-var State; they resolve to x here). A fresh sub-var map keeps the inner
    # scratch names from leaking out, but inherits the outer bindings (the loop var, target, ...).
    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["cond_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx,
                 field_plans=field_plans, target=target)
    cond_expr = sub[v.attrs["cond"].id]
    body_lines.append("if (!(%s)) break;" % cond_expr)
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx,
                 field_plans=field_plans, target=target)
    # Write the next state into the loop variable in place (x <- body result).
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")

def _emit_range(program: Any, v: Any, base: Any, var: Any, model: Any, lines: Any,
                block_idx: Any = None, field_plans: Any = None,
                *, target: str = "system") -> None:
    """Lower a range op to a C++ ``for`` over a fixed count. Like a while, the loop variable is one
    MultiFab mutated in place and the body sub-block is emitted ONCE inside the loop (re-run each
    pass at runtime); the loop-variable value id is seeded to the loop var for the sub-block."""
    from pops.codegen.program_emit_ops import _emit_op
    loop_in = v.inputs[0]
    x = "x%d" % v.id
    i = "i%d" % v.id
    var[v.id] = x
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                 % (x, x, var[loop_in.id]))
    lines.append("for (int %s = 0; %s < %d; ++%s) {" % (i, i, int(v.attrs["count"]), i))
    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["body_block"]:
        _emit_op(program, w, base, frozenset(), sub, model, body_lines, block_idx=block_idx,
                 field_plans=field_plans, target=target)
    body_lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, static_cast<pops::Real>(1), %s);"
                      % (x, x, sub[v.attrs["body"].id]))
    lines += ["  " + ln for ln in body_lines]
    lines.append("}")


def _emit_subcycle(program: Any, v: Any, base: Any, var: Any, model: Any, lines: Any,
                   block_idx: Any = None, field_plans: Any = None,
                   *, target: str = "system") -> None:
    """Lower one exact parent/child clock relation with an exception-safe native cursor scope."""
    from pops.codegen.program_emit_ops import _emit_op
    from pops.time.references import block_name, state_name

    loop_in = v.inputs[0]
    child = v.clock
    parent = v.attrs["parent_clock"]
    count = int(v.attrs["count"])
    child_histories = [
        row for row in program.temporal_manifest()["histories"]
        if row["clock"] == child.qualified_id]
    if target == "amr_system" and child_histories:
        raise NotImplementedError(
            "AMR logical-clock subcycling with child-clock histories requires a composed "
            "AMR-level/logical-clock dense-output provider; refusing an incorrect ring cadence")
    x = "x%d" % v.id
    i = "i%d" % v.id
    scope = "subcycle_scope_%d" % v.id
    parent_dt = "parent_dt_%d" % v.id
    var[v.id] = x
    lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    lines.append("ctx.lincomb(%s, static_cast<pops::Real>(0), %s, "
                 "static_cast<pops::Real>(1), %s);" % (x, x, var[loop_in.id]))
    lines.append("{")
    lines.append("  auto %s = ctx.subcycle_scope(%s, %s, %d);"
                 % (scope, json.dumps(parent.qualified_id),
                    json.dumps(child.qualified_id), count))
    lines.append("  const pops::Real %s = dt;" % parent_dt)
    lines.append("  for (int %s = 0; %s < %d; ++%s) {" % (i, i, count, i))
    lines.append("    %s.iteration(%s);" % (scope, i))
    lines.append("    const pops::Real dt = %s / static_cast<pops::Real>(%d);"
                 % (parent_dt, count))

    # ``keep_history`` owns a state history. Store the loop input at every child tick so lag one in
    # the body means the immediately preceding child-clock value, not the enclosing macro value.
    for state, _config in sorted(
            getattr(program, "_time_history_configs", {}).items(),
            key=lambda item: item[0].qualified_id):
        if (state.clock != child or state.block != loop_in.block
                or state.state != loop_in.state_ref):
            continue
        history_name = "%s.%s" % (block_name(state.block), state_name(state.state))
        owner = block_idx[state.block]
        if target == "amr_system":
            lines.append("    ctx.store_history(%s, %s, %d);"
                         % (json.dumps(history_name), x, owner))
        else:
            lines.append("    ctx.store_history(%s, %s);" % (json.dumps(history_name), x))

    sub = dict(var)
    sub[loop_in.id] = x
    body_lines = []
    for w in v.attrs["body_block"]:
        _emit_op(
            program, w, base, frozenset(), sub, model, body_lines,
            block_idx=block_idx, field_plans=field_plans, target=target)
    body_lines.append(
        "ctx.lincomb(%s, static_cast<pops::Real>(0), %s, "
        "static_cast<pops::Real>(1), %s);" % (x, x, sub[v.attrs["body"].id]))
    if child_histories:
        body_lines.append("ctx.rotate_histories(%s);" % json.dumps(child.qualified_id))
    lines += ["    " + line for line in body_lines]
    lines.append("  }")
    lines.append("  %s.finish();" % scope)
    lines.append("}")

def _emit_branch(program: Any, v: Any, base: Any, var: Any, model: Any, lines: Any,
                 block_idx: Any = None, field_plans: Any = None,
                 *, target: str = "system") -> None:
    """Lower two captured regions to a genuinely lazy C++ ``if``/``else`` value branch."""
    (cond,) = v.inputs
    x = ("b%d" if v.vtype == "bool" else "s%d") % v.id
    is_field = v.vtype in ("state", "rhs", "scalar_field")
    if is_field:
        x = "x%d" % v.id
        if base is None:
            raise NotImplementedError(
                "branch codegen for a block-free scalar_field result requires an explicit "
                "layout template")
        lines.append("pops::MultiFab %s = ctx.scratch_state_like(%s);" % (x, var[base.id]))
    else:
        cpp_type = "bool" if v.vtype == "bool" else "pops::Real"
        lines.append("%s %s;" % (cpp_type, x))
    var[v.id] = x
    lines.append("if (%s) {" % var[cond.id])
    lines += ["  " + line for line in _emit_branch_arm(
        program, v.attrs["true_block"], v.attrs["true_result"], x, is_field,
        base, var, model, block_idx, field_plans, target=target)]
    lines.append("} else {")
    lines += ["  " + line for line in _emit_branch_arm(
        program, v.attrs["false_block"], v.attrs["false_result"], x, is_field,
        base, var, model, block_idx, field_plans, target=target)]
    lines.append("}")


def _emit_branch_arm(program: Any, block: Any, result: Any, output: str, is_field: bool,
                     base: Any, outer_var: Any, model: Any, block_idx: Any,
                     field_plans: Any = None, *, target: str = "system") -> list[str]:
    from pops.codegen.program_emit_ops import _emit_op

    sub = dict(outer_var)
    arm_lines = []
    for value in block:
        _emit_op(
            program, value, base, frozenset(), sub, model, arm_lines,
            block_idx=block_idx, field_plans=field_plans, target=target)
    token = sub[result.id]
    if is_field:
        arm_lines.append(
            "ctx.lincomb(%s, static_cast<pops::Real>(0), %s, "
            "static_cast<pops::Real>(1), %s);" % (output, output, token))
    else:
        arm_lines.append("%s = %s;" % (output, token))
    return arm_lines
