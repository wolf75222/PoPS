"""Program control flow, field ops, reductions and decorator authoring."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
from pops._ir.literals import scalar_literal
from pops.time._program.dump import _ProgramDump
from pops.time._program.constants import _ProgramConstants
from pops.time._authoring import atomic_authoring
from pops.time.references import block_name
from pops.time._program.value_validation import (
    require_compatible_spaces, require_region, require_top_level,
    validate_input_regions,
)
from pops.time.values import ProgramValue, _is_field_value, _resolve_handle

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object
class _ProgramAuthoring(_ProgramDump, _ProgramConstants, _ProgramBase):
    """Decorator, field-op, reduction and control-flow Program authoring."""

    # --- decorator mode (ADC-423): record the step body from a function ---
    @atomic_authoring
    def step(self, fn: Any) -> Any:
        """Record this Program's IR by calling @p fn(self) ONCE, at build time (decorator mode).

        ``@P.step`` is sugar for an inline builder body: the decorated function receives the Program
        and builds the IR exactly as if its statements had been written at module scope. It is a
        BUILD-TIME callback -- it runs once, here, to populate the SSA value list; it is NEVER executed
        numerically during ``sim.step`` (the compiled ``.so`` owns the runtime step). So

            P = pops.time.Program("fe")

            @P.step
            def _(P):
                pops.lib.time.forward_euler(P, "plasma")

        produces byte-identical IR (same ``_ir_hash``) to calling the macro inline. Returns the
        Program (one-liner friendly); @p fn is invoked with the Program as its single argument."""
        if not callable(fn):
            raise TypeError("Program.step expects a callable build_fn(P); got %r" % (fn,))
        fn(self)
        return self

    # --- ghost fill / positivity projection (spec ops 22 / 21) ---
    def fill_boundary(self, x: Any) -> Any:
        """Fill the ghost cells of a State/scalar_field @p x in place (spec op 22): the transport
        BC exchange laplacian/gradient/divergence run internally. Returns @p x (side effect on
        ghosts; valid cells untouched). Lowered to ``ctx.fill_boundary(x)``."""
        if not _is_field_value(x):
            raise ValueError("fill_boundary: a State/RHS/scalar_field value is required (got %r)"
                             % (x,))
        return self._new(
            x.vtype, "fill_boundary", (x,), {}, x.name, x.block, space=x.space)

    def project(self, name: Any = None, state: Any = None, projection: Any = None) -> Any:
        """Apply the block's post-step positivity projection to @p state in place (spec op 21):
        ``U <- project(U, aux)`` over the valid cells, the SAME Zhang-Shu / floor projection the native
        per-step path runs (ADC-177). Returns a State value (the projected state). @p projection selects
        the projection primitive.  The final surface accepts only the typed ``BlockProjection``
        descriptor; omitting it selects that exact descriptor, never a string token. Lowered to
        ``ctx.apply_projection(idx, state)`` for the state's own block."""
        from pops.time._step.transaction import BlockProjection
        name, state = _resolve_handle(name), _resolve_handle(state)
        if isinstance(name, ProgramValue) and state is None:
            name, state = None, name
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("project: a State value is required (state=...)")
        projection = BlockProjection() if projection is None else projection
        if type(projection) is not BlockProjection:
            raise TypeError("project: projection must be BlockProjection()")
        return self._new("state", "project", (state,), {"projection": projection}, name,
                         state.block, space=state.space, point=state.point,
                         field_context=state.field_context, state_ref=state.state_ref)

    def _acceptance_guard_value(
        self, name: str, value: ProgramValue, condition: ProgramValue, action: Any,
    ) -> ProgramValue:
        from pops.time.solve_outcome import FailRun, RejectAttempt
        if not isinstance(action, (RejectAttempt, FailRun)):
            raise TypeError("acceptance guard terminal action must be RejectAttempt() or FailRun()")
        return self._new(
            value.vtype, "acceptance_guard", (value, condition),
            {"guard": name, "action": action}, value.name, value.block,
            space=value.space, point=value.point, field_context=value.field_context,
            state_ref=value.state_ref,
        )

    @atomic_authoring
    def guard(
        self,
        name: Any,
        value: Any,
        condition: Any,
        *,
        action: Any,
        role: Any = None,
        recheck: Any = None,
    ) -> Any:
        """Gate one provisional value before commit.

        ``condition`` is a compiled scalar Bool.  ``RejectAttempt`` and ``FailRun`` lower directly.
        ``ProjectAndRecheck`` authors a lazy failure arm: the native block projection runs only after
        the first failure, then ``recheck(P, projected)`` builds the second Bool and the configured
        terminal action is applied if it still fails.  The callable is authoring-only; numerical
        evaluation remains entirely in generated C++.
        """
        from pops.time.solve_outcome import FailRun, RejectAttempt
        from pops.time._step.transaction import (
            AcceptanceGuard, GuardRole, ProjectAndRecheck,
        )
        if not isinstance(name, str) or not name:
            raise ValueError("guard: name must be a non-empty string")
        if not isinstance(value, ProgramValue) or not value.is_field():
            raise TypeError("guard: value must be a State/RHS/scalar_field ProgramValue")
        if not isinstance(condition, ProgramValue) or condition.vtype != "bool":
            raise TypeError("guard: condition must be a scalar Bool ProgramValue")
        role = GuardRole.INVARIANT if role is None else role
        if type(role) is not GuardRole:
            raise TypeError("guard: role must be a GuardRole")
        validate_input_regions(
            self, (value, condition), self._current_region(), "acceptance guard")

        if isinstance(action, (RejectAttempt, FailRun)):
            if recheck is not None:
                raise ValueError("guard: recheck is valid only with ProjectAndRecheck")
            guarded = self._acceptance_guard_value(name, value, condition, action)
        elif type(action) is ProjectAndRecheck:
            if not callable(recheck):
                raise TypeError(
                    "guard: ProjectAndRecheck requires an authoring recheck(P, projected) callable")

            def project_and_recheck(program: Any) -> ProgramValue:
                projected = program.project(
                    name="%s_projection" % name, state=value,
                    projection=action.projection)
                predicate = recheck(program, projected)
                if not isinstance(predicate, ProgramValue) or predicate.vtype != "bool":
                    raise TypeError("guard: recheck must return a scalar Bool ProgramValue")
                return program._acceptance_guard_value(
                    name, projected, predicate, action.on_failure)

            guarded = self.branch(
                condition,
                when_true=lambda _program: value,
                when_false=project_and_recheck,
                name="%s_guarded" % name,
            )
        else:
            raise TypeError(
                "guard: action must be RejectAttempt(), FailRun(), or ProjectAndRecheck()")
        self._register_acceptance_guard(AcceptanceGuard(name, role, action))
        return guarded

    # --- per-cell conditional select (spec op 17, ADC-418) ---
    def cell_compare(self, field: Any, value: Any, cmp: Any, name: Any = None) -> Any:
        """A PER-CELL comparison ``field <cmp> value`` -> a fresh 1-component 0/1 mask scalar_field (1.0
        where the comparison holds, 0.0 otherwise), evaluated cell by cell on component 0 of @p field
        (its sole / first conserved component). @p field is a State/RHS/scalar_field value; @p value is a
        Python float threshold (a per-cell field threshold is a later phase); @p cmp is one of
        ``'>' '>=' '<' '<='``. The mask is the input the per-cell `where` selects on. Lowered to a
        for_each_cell kernel ``maskA(i,j,0) = fieldA(i,j,0) <cmp> value ? 1 : 0``. Convenience wrappers:
        `cell_gt` / `cell_ge` / `cell_lt` / `cell_le`."""
        if not _is_field_value(field):
            raise ValueError("cell_compare: a State/RHS/scalar_field value is required (got %r)"
                             % (field,))
        try:
            threshold = scalar_literal(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("cell_compare: value must be an exact scalar threshold (a per-cell field "
                            "threshold is a later phase); got %r" % (value,)) from exc
        if cmp not in self._CELL_CMPS:
            raise ValueError("cell_compare: cmp must be one of %s; got %r"
                             % (sorted(self._CELL_CMPS), cmp))
        return self._new("scalar_field", "cell_compare", (field,),
                         {"cmp": cmp, "value": threshold}, name, field.block)

    def cell_gt(self, field: Any, value: Any, name: Any = None) -> Any:
        """Per-cell ``field > value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, ">", name)

    def cell_ge(self, field: Any, value: Any, name: Any = None) -> Any:
        """Per-cell ``field >= value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, ">=", name)

    def cell_lt(self, field: Any, value: Any, name: Any = None) -> Any:
        """Per-cell ``field < value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, "<", name)

    def cell_le(self, field: Any, value: Any, name: Any = None) -> Any:
        """Per-cell ``field <= value`` mask (1.0 / 0.0). See `cell_compare`."""
        return self.cell_compare(field, value, "<=", name)

    def where(self, mask: Any, a: Any, b: Any, name: Any = None) -> Any:
        """A PER-CELL conditional select (spec op 17): ``out(i,j,c) = mask(i,j,*) != 0 ? a(i,j,c) :
        b(i,j,c)`` COMPONENT-WISE over the field. This is NOT the scalar runtime `branch` -- the
        condition is decided per cell INSIDE a Kokkos kernel.

          - @p mask: a 0/1 (or any nonzero/zero) mask field. Either 1-component (one mask shared by all
            components -- read at component 0) or with the SAME ncomp as @p a / @p b (a per-component
            mask). Built with `cell_ge` / `cell_gt` / ... (a threshold) or any scalar_field.
          - @p a, @p b: the two field values to choose between, on the SAME grid and ncomp (a State or a
            scalar_field). The result has @p a's vtype / block / ncomp.

        Lowered to a for_each_cell select kernel (a ternary, no branch divergence concern at MVP)."""
        for nm, fv in (("mask", mask), ("a", a), ("b", b)):
            if not _is_field_value(fv):
                raise ValueError("where: %s must be a State/RHS/scalar_field value (got %r)"
                                 % (nm, fv))
        if a.vtype != b.vtype:
            raise ValueError("where: a and b must have the same value type (a is %s, b is %s)"
                             % (a.vtype, b.vtype))
        if a.block != b.block:
            raise ValueError("where: a and b must belong to the same block (a is %r, b is %r)"
                             % (a.block, b.block))
        if mask.block not in (None, a.block):
            raise ValueError("where: mask and selected values must belong to the same block")
        require_compatible_spaces(a.space, b.space, "where", typed_pair=True)
        if mask.vtype in ("state", "rhs"):
            require_compatible_spaces(a.space, mask.space, "where mask", typed_pair=True)
        na, nb, nm_ = self._ncomp(a), self._ncomp(b), self._ncomp(mask)
        if na is not None and nb is not None and na != nb:
            raise ValueError("where: a and b must have the same ncomp (a has %d, b has %d)" % (na, nb))
        ncomp = na if na is not None else nb
        if nm_ is not None and ncomp is not None and nm_ not in (1, ncomp):
            raise ValueError("where: mask must be 1-component or match a/b's ncomp (mask has %d, "
                             "a/b have %d)" % (nm_, ncomp))
        attrs = {"ncomp": ncomp} if ncomp is not None else {}
        return self._new(
            a.vtype, "where", (mask, a, b), attrs, name, a.block, space=a.space)

    @staticmethod
    def _ncomp(value: Any) -> Any:
        """The statically-known component count of a field value, or None when it is not pinned in the
        IR (a State / RHS ncomp is the model's n_cons, known only at codegen): a scalar_field carries its
        own ``ncomp`` attr. Used by `where` for the static a/b/mask ncomp consistency check."""
        if value.vtype == "scalar_field":
            return int(value.attrs.get("ncomp", 1))
        return None

    # --- reductions / comparisons / control flow (ADC-404a) ---
    def norm2(self, state: Any) -> Any:
        """The Euclidean norm ``||u||_2`` of a State (collective all_reduce; Scalar). Lowered as
        ``sqrt(pops::dot(u, u))``. NOTE (holds for every component-0 reduction below): it reduces
        COMPONENT 0 only and MUST run on every rank; use the ``*_component`` forms for a role."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("norm2: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "norm2"}, None, state.block)

    def dot(self, a: Any, b: Any) -> Any:
        """The inner product ``<a, b>`` of two States (collective, Scalar): ``pops::dot(a, b)``."""
        if not (isinstance(a, ProgramValue) and a.is_field() and isinstance(b, ProgramValue) and b.is_field()):
            raise ValueError("dot: two State/RHS values are required")
        if a.block != b.block:
            raise ValueError("dot: both fields must belong to the same block")
        require_compatible_spaces(a.space, b.space, "dot", typed_pair=True)
        if self._ncomp(a) != self._ncomp(b):
            raise ValueError("dot: both fields must have the same known component count")
        return self._new("scalar", "reduce", (a, b), {"kind": "dot"}, None, a.block)

    def norm_inf(self, state: Any) -> Any:
        """The infinity norm ``max|u|`` (collective, component 0, Scalar): ``pops::norm_inf(u)``."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("norm_inf: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "norm_inf"}, None, state.block)

    def norm1(self, state: Any) -> Any:
        """The 1-norm ``sum|u|`` (collective, component 0, Scalar): ``pops::reduce_abs_sum(u, 0)``."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("norm1: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "abs_sum", "comp": 0}, None,
                         state.block)

    def sum(self, state: Any) -> Any:
        """The sum over component 0 (collective, Scalar): ``pops::reduce_sum(u, 0)``; cf sum_component."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("sum: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "sum", "comp": 0}, None, state.block)

    def max(self, state: Any) -> Any:
        """The maximum ``max_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``pops::reduce_max(u, 0)`` (the SIGNED max, not the magnitude -- use
        `norm_inf` for max|u|). COLLECTIVE: called on every rank. Component 0 only."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("max: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "max", "comp": 0}, None, state.block)

    def min(self, state: Any) -> Any:
        """The minimum ``min_cells u`` of a State over component 0 (a collective all_reduce). Returns a
        Scalar value. Lowered as ``pops::reduce_min(u, 0)``. COLLECTIVE: called on every rank.
        Component 0 only."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("min: a State/RHS value is required")
        return self._new("scalar", "reduce", (state,), {"kind": "min", "comp": 0}, None, state.block)

    def sum_component(self, state: Any, comp: Any) -> Any:
        """The sum ``sum_cells u(.,comp)`` of a State over conservative component @p comp (a collective
        all_reduce). Returns a Scalar value. Lowered as ``pops::reduce_sum(u, comp)``. COLLECTIVE:
        called on every rank. @p comp must be a Python int >= 0 (a runtime component is meaningless)."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("sum_component: a State/RHS value is required")
        if isinstance(comp, bool) or not isinstance(comp, int) or comp < 0:
            raise ValueError("sum_component: comp must be a Python int >= 0 (got %r)" % (comp,))
        return self._new("scalar", "reduce", (state,), {"kind": "sum", "comp": int(comp)}, None,
                         state.block)

    def abs_sum_component(self, state: Any, comp: Any) -> Any:
        """The role-selected L1 ``sum_cells |u(.,comp)|`` (collective all_reduce). Lowered as
        ``pops::reduce_abs_sum(u, comp)``. @p comp must be a Python int >= 0."""
        if not (isinstance(state, ProgramValue) and state.is_field()):
            raise ValueError("abs_sum_component: a State/RHS value is required")
        if isinstance(comp, bool) or not isinstance(comp, int) or comp < 0:
            raise ValueError("abs_sum_component: comp must be a Python int >= 0 (got %r)" % (comp,))
        return self._new("scalar", "reduce", (state,), {"kind": "abs_sum", "comp": int(comp)}, None,
                         state.block)

    def record_scalar(self, name: Any, value: Any) -> Any:
        """Record a runtime Scalar @p value (e.g. ``P.norm2(R)``) into the System diagnostics map under
        @p name, retrievable after the step via ``sim.program_diagnostic(name)`` /
        ``sim.program_diagnostics()`` (spec op 23). A side-effecting op (no value): it stores the scalar
        for inspection / logging, it does not feed the numerics. @p name must be a non-empty string;
        @p value must be a Scalar value (a P.norm2 / P.dot / P.sum / ... result), not a field. Lowered
        to ``ctx.record_scalar("<name>", <scalar>)``."""
        if not isinstance(name, str) or not name:
            raise ValueError("record_scalar: name must be a non-empty string")
        if not (isinstance(value, ProgramValue) and value.vtype == "scalar"):
            raise ValueError("record_scalar: value must be a Scalar value (e.g. P.norm2(R)); got %r"
                             % (value,))
        return self._new("scalar", "record_scalar", (value,), {"diagnostic": name}, name,
                         value.block)

    def _scalar_binop(self, a: Any, b: Any, fn: Any) -> Any:
        """Build a Scalar arithmetic node ``a <fn> b`` (fn in add/sub/mul/div). Each operand is a Scalar
        ProgramValue or a Python number (a literal constant, stored in attrs). Used by the ProgramValue scalar dunders
        so a dt_bound can express cfl * hmin / max_wave_speed (spec s18); never evaluated in Python."""
        inputs = []
        operands = []  # per operand: ("v", index-into-inputs) or ("c", exact scalar literal)
        for x in (a, b):
            if isinstance(x, ProgramValue):
                if x.vtype != "scalar":
                    raise TypeError("scalar arithmetic operands must be Scalar values or numbers; got "
                                    "a %s value %r" % (x.vtype, x.name))
                operands.append(("v", len(inputs)))
                inputs.append(x)
            else:
                try:
                    operands.append(("c", scalar_literal(x)))
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "scalar arithmetic operands must be Scalar values or numbers; got %r" % (x,)
                    ) from exc
        blocks = {
            x.block for x in (a, b)
            if isinstance(x, ProgramValue) and x.block is not None
        }
        if len(blocks) > 1:
            raise ValueError(
                "scalar arithmetic cannot combine values owned by different blocks %s"
                % sorted(block_name(item) for item in blocks))
        block = next(iter(blocks), None)
        return self._new("scalar", "scalar_op", tuple(inputs), {"fn": fn, "operands": operands}, None,
                         block)

    def max_wave_speed(self, state: Any) -> Any:
        """The maximum |wave speed| of @p state's block (a collective reduction): the SAME per-block
        wave speed the native CFL uses (BlockState::max_speed). Returns a Scalar value. Lowered to
        ``ctx.max_wave_speed(idx, u)`` for @p state's own block (ADC-426). The denominator of a
        CFL-style dt bound cfl * hmin / w (spec s18). REUSES the block's wave-speed closure -- it does
        not recompute the speed."""
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("max_wave_speed: a State value is required")
        # A dt-bound is emitted as an isolated sub-program.  Materialize an explicit state read in
        # that region when its builder refers to an already-authored top-level TimeState; otherwise
        # the bound would retain a cross-region SSA input that its emitter cannot name.
        if self._recording and state.region != self._current_region():
            state = self._new(
                "state", "state", (), {"state": state.state_ref}, block_name(state.block),
                state.block, space=state.space, state_ref=state.state_ref)
        return self._new("scalar", "max_wave_speed", (state,), {}, None, state.block)

    def hmin(self) -> Any:
        """The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)):
        the SAME hmin the native CFL uses. Returns a Scalar value. Lowered to ``ctx.hmin()``. The
        numerator factor of a CFL-style dt bound cfl * hmin / max_wave_speed (spec s18)."""
        return self._new("scalar", "hmin", (), {}, None, None)

    def _compare(self, lhs: Any, rhs: Any, cmp: Any) -> Any:
        """Build a Bool predicate ``s_lhs <cmp> rhs`` (re-evaluated each loop pass). @p rhs is a Python
        float tolerance (stored as a literal) or another Scalar value (compared at runtime). Inputs are
        the Scalar operand(s); the float bound lives in attrs['rhs']."""
        if isinstance(rhs, ProgramValue) and rhs.vtype == "scalar":
            return self._new("bool", "compare", (lhs, rhs), {"cmp": cmp}, None, lhs.block)
        try:
            tolerance = scalar_literal(rhs)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "compare: the right-hand side must be a scalar tolerance or a Scalar value, got %r"
                % (rhs,)
            ) from exc
        return self._new("bool", "compare", (lhs,), {"cmp": cmp, "rhs": tolerance}, None,
                         lhs.block)

    @atomic_authoring
    def while_(self, state: Any, cond_fn: Any, body_fn: Any) -> Any:
        """A convergence loop: starting from @p state, while ``cond_fn(self, x)`` holds, replace x by
        ``body_fn(self, x)``; return the final State.

        The condition and body are RE-EVALUATED each iteration, so the ops they build are captured into
        a separate recording sub-block (NOT the flat SSA list) and re-emitted inside a C++ loop. The
        loop variable is the SAME C++ MultiFab across passes (mutated in place).

          - ``cond_fn(self, x)`` must return a Bool value (e.g. ``self.norm2(diff) > tol``);
          - ``body_fn(self, x)`` must return the next-iteration State (e.g. a linear_combine)."""
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("while_: the loop variable must be a State value")
        require_top_level(self, state, "while_")
        if self._recording:
            raise NotImplementedError(
                "while_: nested control flow is a later phase; a while_ body cannot itself open a "
                "while_ yet")
        cond_block, cond_val = self._record(cond_fn, state)
        cond_region = self._region_for_block(cond_block)
        require_region(
            self, cond_val, cond_region, "while_ cond", vtype="bool")
        body_block, next_state = self._record(body_fn, state)
        body_region = self._region_for_block(body_block)
        if not (isinstance(next_state, ProgramValue) and next_state.vtype == "state"):
            raise ValueError("while_: body_fn must return the next-iteration State value")
        if next_state.block != state.block:
            raise ValueError("while_: body_fn must return a State of the same block as the loop "
                             "variable")
        require_region(
            self, next_state, body_region, "while_ body", vtype="state",
            allow=(state,))
        require_compatible_spaces(state.space, next_state.space, "while_ body", typed_pair=True)
        return self._new("state", "while", (state,),
                         {"cond_block": cond_block, "cond_region": cond_region, "cond": cond_val,
                          "body_block": body_block, "body_region": body_region, "body": next_state},
                         None, state.block, space=state.space)

    @atomic_authoring
    def static_range(self, state: Any, count: Any, body_fn: Any) -> Any:
        """A COMPILE-TIME (unrolled) loop: apply ``body_fn(self, x)`` to the State @p count times,
        threading the result, and return the final State. @p count must be a Python int known at IR
        build time -- the loop is unrolled HERE (no IR control-flow op, no C++ loop): it simply builds
        @p count copies of the body ops in order. Use `range` for a C++ ``for`` over a fixed count."""
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("static_range count must be a Python int (compile-time); use P.range for "
                            "a runtime / C++-loop count")
        if count < 0:
            raise ValueError("static_range count must be non-negative")
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("static_range: the loop variable must be a State value")
        region = self._current_region()
        validate_input_regions(self, (state,), region, "static_range")
        x = state
        for _ in range(count):
            x = body_fn(self, x)
            if not (isinstance(x, ProgramValue) and x.vtype == "state" and x.block == state.block):
                raise ValueError("static_range: body_fn must return a State of the loop variable's "
                                 "block")
            require_region(self, x, region, "static_range body", vtype="state", allow=(state,))
            require_compatible_spaces(state.space, x.space, "static_range body", typed_pair=True)
        return x

    @atomic_authoring
    def range(self, state: Any, count: Any, body_fn: Any) -> Any:
        """A C++ ``for`` loop over a FIXED count: from @p state, apply ``body_fn(self, x)`` @p count
        times, threading the loop-variable State in place, and return the final State. @p count must be
        a Python int (a runtime/Scalar count is a later phase). The body is RE-EXECUTED each pass, so
        its ops are captured into a recording sub-block (NOT the flat SSA list) and emitted ONCE inside
        the loop. Use `static_range` to unroll instead."""
        if isinstance(count, ProgramValue):
            if count.vtype == "scalar":
                raise NotImplementedError("range with a runtime Scalar count is deferred; use a "
                                          "Python int")
            raise TypeError("range count must be a Python int")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("range count must be a Python int")
        if count < 0:
            raise ValueError("range count must be non-negative")
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("range: the loop variable must be a State value")
        require_top_level(self, state, "range")
        if self._recording:
            raise NotImplementedError("range: nested control flow is a later phase; a control-flow "
                                      "body cannot itself open a range yet")
        body_block, next_state = self._record(body_fn, state)
        body_region = self._region_for_block(body_block)
        if not (isinstance(next_state, ProgramValue) and next_state.vtype == "state"
                and next_state.block == state.block):
            raise ValueError("range: body_fn must return the next-iteration State of the same block")
        require_region(
            self, next_state, body_region, "range body", vtype="state",
            allow=(state,))
        require_compatible_spaces(state.space, next_state.space, "range body", typed_pair=True)
        return self._new("state", "range", (state,),
                         {"count": int(count), "body_block": body_block,
                          "body_region": body_region, "body": next_state},
                         None, state.block, space=state.space)

    @atomic_authoring
    def subcycle(self, state: Any, *, clock: Any, within: Any,
                 count: Any, body_fn: Any, name: Any = None) -> Any:
        """Advance ``state`` on ``clock`` exactly ``count`` ticks inside one ``within`` tick.

        ``state`` must already belong to the child clock, normally through an explicit
        :meth:`Program.synchronize`.  The recorded body sees ``P.dt`` as the active child-tick
        duration at lowering time; nested subcycles therefore divide the enclosing duration without
        changing authoring coefficients.  Returning to the parent clock is another explicit
        synchronization, never an implicit cast.
        """
        from pops.time.points import Clock, TimePoint

        state = _resolve_handle(state)
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise TypeError("subcycle: state must be a State ProgramValue")
        if type(clock) is not Clock or type(within) is not Clock:
            raise TypeError("subcycle: clock= and within= must be exact Clock values")
        if clock == within:
            raise ValueError("subcycle: child and parent clocks must be distinct")
        if state.clock != clock:
            raise ValueError(
                "subcycle: state belongs to clock %r, not child clock %r; synchronize it first"
                % (state.clock.name, clock.name))
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("subcycle: count must be a positive Python int")
        if not callable(body_fn):
            raise TypeError("subcycle: body_fn must be callable as body_fn(P, state)")
        parent_region = self._current_region()
        body_block, next_state = self._record(body_fn, state)
        body_region = self._region_for_block(body_block)
        if parent_region:
            self._allow_region_capture(parent_region, body_region)
        if not (isinstance(next_state, ProgramValue) and next_state.vtype == "state"):
            raise TypeError("subcycle: body_fn must return a State ProgramValue")
        if next_state.block != state.block or next_state.state_ref != state.state_ref:
            raise ValueError("subcycle: body_fn must preserve the exact block/state identity")
        if next_state.clock != clock:
            raise ValueError("subcycle: body_fn result must remain on the child clock")
        require_region(
            self, next_state, body_region, "subcycle body", vtype="state", allow=(state,))
        require_compatible_spaces(state.space, next_state.space, "subcycle body", typed_pair=True)
        return self._new(
            "state", "subcycle", (state,),
            {"count": int(count), "parent_clock": within,
             "child_clock": clock, "body_block": body_block,
             "body_region": body_region, "body": next_state},
            name, state.block, space=state.space, state_ref=state.state_ref,
            point=TimePoint(clock, step=int(count)))

    @atomic_authoring
    def branch(self, condition: Any, when_true: Any, when_false: Any,
               name: Any = None) -> Any:
        """Select one lazily-authored typed value at runtime.

        ``when_true`` and ``when_false`` are build callbacks taking this Program.  Their nodes are
        captured in disjoint regions and are emitted inside the corresponding runtime ``if`` arm;
        neither arm is evaluated eagerly by the generated program.  This graph control-flow
        operation is deliberately distinct from :meth:`where`, which selects per cell.
        """
        if not (isinstance(condition, ProgramValue) and condition.vtype == "bool"):
            raise ValueError("branch: condition must be a scalar Bool ProgramValue")
        if not callable(when_true) or not callable(when_false):
            raise TypeError("branch: when_true and when_false must be callables accepting Program")
        validate_input_regions(self, (condition,), self._current_region(), "branch condition")
        parent_region = self._current_region()
        true_block, true_result = self._record_branch_arm(when_true)
        false_block, false_result = self._record_branch_arm(when_false)
        true_region = self._region_for_block(true_block)
        false_region = self._region_for_block(false_block)
        if parent_region:
            self._allow_region_capture(parent_region, true_region)
            self._allow_region_capture(parent_region, false_region)
        for label, result, region in (
                ("when_true", true_result, true_region),
                ("when_false", false_result, false_region)):
            if not isinstance(result, ProgramValue):
                raise TypeError("branch: %s must return a ProgramValue" % label)
            if result.region not in (0, region, parent_region):
                raise ValueError(
                    "branch: %s result escapes an unrelated authoring region" % label)
            validate_input_regions(self, (result,), region, "branch %s" % label)
        self._require_branch_compatible(true_result, false_result)
        if condition.clock != true_result.clock:
            raise ValueError("branch: condition and result arms must share one clock")
        return self._new(
            true_result.vtype, "branch", (condition,),
            {"true_block": true_block, "true_region": true_region,
             "true_result": true_result, "false_block": false_block,
             "false_region": false_region, "false_result": false_result},
            name, true_result.block, space=true_result.space, point=true_result.point,
            field_context=true_result.field_context)

    @staticmethod
    def _require_branch_compatible(left: Any, right: Any) -> None:
        if left.vtype != right.vtype:
            raise ValueError("branch: both arms must return the same value type")
        if left.block != right.block:
            raise ValueError("branch: both arms must return values from the same owned block")
        if left.clock != right.clock or left.point != right.point:
            raise ValueError("branch: both arms must share one clock and exact point")
        if left.space != right.space:
            raise ValueError("branch: both arms must return the same typed space")
        if left.field_context != right.field_context:
            raise ValueError("branch: both arms must return the same field context")

    def _record_branch_arm(self, fn: Any) -> Any:
        sub = []
        self._recording.append(sub)
        try:
            result = fn(self)
        finally:
            self._recording.pop()
        return sub, result

    @atomic_authoring
    def _record(self, fn: Any, x: Any) -> Any:
        """Run a control-flow callable ``fn(self, x)`` with a fresh recording scope active, capturing the
        ops it builds into a sub-block (returned with the value fn produced). The sub-block ops are NOT
        appended to self._values (they belong to the owning control-flow op)."""
        sub = []
        destination = self._region_for_block(sub)
        source = getattr(x, "region", 0)
        if source not in (0, destination):
            self._allow_region_capture(source, destination)
        self._recording.append(sub)
        try:
            out = fn(self, x)
        finally:
            self._recording.pop()
        return sub, out
