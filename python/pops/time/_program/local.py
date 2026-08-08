"""pops.time Program authoring mixin -- local + matrix-free ops.

Local solves, matrix-free operators, laplacian/gradient/divergence and the coefficiented apply.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pops.time._program.constants import _ProgramConstants
from pops.time._authoring import atomic_authoring
from pops.time._program.value_validation import (
    rate_space_for, require_affine_region, require_compatible_spaces, require_region,
    require_owned, require_top_level,
)
from pops.time.operator_resolution import resolve_operator_handle
from pops.time.references import block_name
from pops.time.stencil import StencilAccess
from pops.time.value_metadata import positive_scalar_literal
from pops.time.values import (
    ProgramValue, _Coeff, _Operator, _exact_number, _residual_wants_guess,
    _resolve_handle)

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


def _prepared_local_nonlinear_controls(prepared: Any, *, where: str) -> dict[str, Any]:
    """Validate the small immutable provider protocol and return canonical IR controls."""
    def positive(name: str) -> Any:
        return positive_scalar_literal(getattr(prepared, name, None), where=where + " " + name)

    def nonnegative(name: str) -> Any:
        value = getattr(prepared, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("%s %s must be a non-negative scalar" % (where, name))
        return value if value == 0 else positive_scalar_literal(
            value, where=where + " " + name)

    def integer(name: str, *, positive_only: bool) -> int:
        value = getattr(prepared, name, None)
        minimum = 1 if positive_only else 0
        if isinstance(value, bool) or not isinstance(value, int) \
                or value < minimum or value > (1 << 31) - 1:
            qualifier = "positive" if positive_only else "non-negative"
            raise ValueError("%s %s must be a %s int" % (where, name, qualifier))
        return value

    safeguard = getattr(prepared, "safeguard", None)
    if safeguard not in ("exact", "damped", "backtracking"):
        raise ValueError("%s safeguard is not a supported prepared policy" % where)
    damping = getattr(prepared, "damping", None)
    minimum_step = getattr(prepared, "minimum_step", None)
    armijo = getattr(prepared, "armijo", None)
    if not isinstance(damping, (int, float)) or isinstance(damping, bool) \
            or damping <= 0 or damping > 1:
        raise ValueError("%s damping must be in (0, 1]" % where)
    if safeguard == "exact" and damping != 1:
        raise ValueError("%s safeguard='exact' requires damping=1" % where)
    if not isinstance(minimum_step, (int, float)) or isinstance(minimum_step, bool) \
            or minimum_step <= 0 or minimum_step > damping:
        raise ValueError("%s minimum_step must be in (0, damping]" % where)
    if not isinstance(armijo, (int, float)) or isinstance(armijo, bool) \
            or armijo <= 0 or armijo >= 1:
        raise ValueError("%s armijo must be in (0, 1)" % where)
    return {
        "tol": positive("tolerance"),
        "relative_tol": nonnegative("relative_tolerance"),
        "step_tol": nonnegative("step_tolerance"),
        "max_iter": integer("max_iterations", positive_only=True),
        "max_evaluations": integer("max_evaluations", positive_only=False),
        "fd_eps": positive("finite_difference_step"),
        "safeguard": safeguard,
        "damping": positive("damping"),
        "max_backtracks": integer("max_backtracks", positive_only=False),
        "minimum_step": positive("minimum_step"),
        "armijo": positive("armijo"),
    }


class _ProgramLocal(_ProgramConstants, _ProgramBase):
    """Local solves, matrix-free operators, laplacian/gradient/divergence and the coefficiented apply."""

    def _solve_local_linear(self, *, operator: Any, rhs: Any, prepared: Any,
                            fields: Any = None, name: Any = None) -> Any:
        """Solve a LOCAL linear system ``operator U = rhs`` cell by cell, where
        ``operator = self.I +/- a*L`` for a single model linear source ``L`` (``a`` may depend on dt
        / constants). Returns the solution State. A non-local or non-linear operator is rejected;
        dense storage is specialized to the model manifest's exact component count."""
        if not isinstance(operator, _Operator) or operator.identity.as_dict() != {0: 1}:
            raise ValueError("solve_local_linear currently supports local linear operators only")
        if len(operator.terms) != 1:
            raise NotImplementedError(
                "solve_local_linear currently supports a single linear source (I +/- a*L); got %d "
                "term(s)" % len(operator.terms))
        if not (isinstance(rhs, ProgramValue) and rhs.vtype == "state"):
            raise ValueError("solve_local_linear: rhs must be a State value (rhs=...)")
        if fields is not None and not (isinstance(fields, ProgramValue) and fields.vtype == "fields"):
            raise ValueError("solve_local_linear: fields must be a FieldContext from solve_fields")
        field_context = getattr(rhs, "field_context", None)
        if fields is not None:
            from pops.time.field_context import (
                field_provenance_contains, merge_field_provenance, require_field_read,
            )
            required_context = require_field_read(
                fields, rhs, "solve_local_linear", allow_derived=True)
            field_context = merge_field_provenance(field_context, required_context)
        op_value, l_coeff = operator.terms[0]
        require_owned(self, op_value, "solve_local_linear operator")
        if fields is not None and op_value.field_context is not None and not (
                field_provenance_contains(op_value.field_context, required_context)):
            raise ValueError(
                "solve_local_linear: operator was authored for field provenance %r, not the "
                "explicit fields context %r" % (op_value.field_context, required_context))
        self._check_operator_state(op_value, rhs, "solve_local_linear")
        lname = op_value.attrs["linear_source"]
        a = (-l_coeff).to_polynomial()  # operator = I - a*L, so L carries coefficient -a
        inputs = (rhs, op_value, fields) if fields is not None else (rhs, op_value)
        identity = getattr(prepared, "identity", None)
        identity_token = getattr(identity, "token", None)
        if not isinstance(identity_token, str):
            raise TypeError("solve: DenseLU provider identity is not canonical")
        attrs = {
            "linear_source": lname, "a_coeff": a,
            "solver_identity": identity_token,
            "problem_kind": "local_linear",
        }
        if "operator_handle" in op_value.attrs:
            attrs["operator_handle"] = op_value.attrs["operator_handle"]
        token = self._new(
            "state", "solve_local_linear", inputs, attrs, name, rhs.block, space=rhs.space,
            point=rhs.point, field_context=field_context)
        from pops.time.solve_outcome import SolveOutcome

        outcome_name = name or token.name

        def project(outcome: Any) -> Any:
            return self._new(
                "state", "solve_outcome_component", (outcome,), {"index": 0},
                outcome_name, rhs.block, space=rhs.space, point=rhs.point,
                field_context=field_context)

        return SolveOutcome(self, token, project, outcome_name)

    @atomic_authoring
    def solve(self, problem: Any, *, solver: Any, name: Any = None) -> Any:
        """Build one typed solve through the solver's small Program provider interface.

        The Program does not select a PDE family or algorithm.  A solver descriptor prepares an
        immutable provider and that provider builds the normalized IR through private primitives.
        Strings, option bags and parallel ``solve_*`` public verbs are deliberately absent.
        """
        if isinstance(solver, str):
            raise TypeError("solve: solver must be a typed descriptor, not %r" % solver)
        prepare = getattr(solver, "prepare_program_solve", None)
        if not callable(prepare):
            raise TypeError(
                "solve: solver must implement prepare_program_solve(); got %r"
                % type(solver).__name__)
        prepared = prepare()
        build = getattr(prepared, "build_program_solve", None)
        if not callable(build):
            raise TypeError(
                "solve: prepared solver must implement build_program_solve(); got %r"
                % type(prepared).__name__)
        return build(program=self, problem=problem, name=name)

    def _solve_coupled_implicit(self, operator: Any, states: Any, *, prepared: Any,
                                name: Any = None, at: Any = None, coefficient: Any,
                                ) -> Any:
        """Solve ``U - U0 - dt * operator(U) = 0`` over owner-qualified blocks.

        The typed ``coupled_rate`` signature is the join contract for every input and output.  The
        returned solve token stays unreadable until an explicit ``FailRun`` or ``RejectAttempt`` is
        consumed; no partially converged state is ever published.
        """
        from pops.model import OperatorHandle
        from pops.time.solve_outcome import SolveOutcome
        from pops.time.value_collections import _CoupledResult

        if not isinstance(operator, OperatorHandle):
            raise TypeError("solve requires a typed coupled_rate OperatorHandle")
        if isinstance(states, Mapping):
            raw = tuple(states.values())
            for block, value in states.items():
                if getattr(_resolve_handle(value), "block", None) != block:
                    raise ValueError(
                        "solve state mapping keys must match each State value's BlockHandle")
        elif isinstance(states, Sequence) and not isinstance(states, (str, bytes)):
            raw = tuple(states)
        else:
            raise TypeError("solve inputs must be a non-empty sequence or block mapping")
        values = tuple(_resolve_handle(value) for value in raw)
        if not values or any(not isinstance(value, ProgramValue) or value.vtype != "state"
                             for value in values):
            raise ValueError("solve inputs must contain one or more State values")
        for value in values:
            require_top_level(self, value, "solve")
        if len({value.point for value in values}) != 1:
            raise ValueError("solve inputs must share one exact evaluation point")
        op = resolve_operator_handle(
            self, operator, where="solve", expected_kinds="coupled_rate", values=values)
        self._check_call_args(op, values)
        controls = _prepared_local_nonlinear_controls(
            prepared, where="solve: solver")
        bundle = op.signature.output
        by_name = {block_name(value.block): value for value in values}
        missing = tuple(output for output in bundle.keys() if output not in by_name)
        if missing:
            raise ValueError(
                "solve operator outputs %s without matching input block names"
                % (missing,))
        blocks = tuple(by_name[output].block for output in bundle.keys())
        token_name = name or operator.name
        if at is None:
            from pops.time.points import TimePoint
            result_points = (TimePoint(self.clock, step=1),) * len(blocks)
        elif isinstance(at, Mapping):
            if set(at) != set(blocks):
                raise ValueError("solve at mapping must name every output BlockHandle")
            result_points = tuple(at[block] for block in blocks)
        elif isinstance(at, Sequence) and not isinstance(at, (str, bytes)):
            result_points = tuple(at)
            if len(result_points) != len(blocks):
                raise ValueError("solve at sequence must match output arity")
        else:
            result_points = (at,) * len(blocks)
        token = self._new(
            "coupled_solution", "solve_coupled_implicit", values,
            {"operator": op.name, "operator_handle": operator, "blocks": blocks,
             "method": "newton", "solver_identity": prepared.identity.token,
             "problem_kind": "coupled_implicit_euler",
             "coefficient": coefficient,
             **controls, "output_count": len(blocks)},
            token_name, blocks[0], point=result_points[0])

        def project(outcome: Any) -> Any:
            projected = {}
            for index, (output_name, block) in enumerate(
                    zip(bundle.keys(), blocks, strict=True)):
                initial = by_name[output_name]
                projected[block] = self._new(
                    "state", "solve_outcome_component", (outcome,),
                    {"index": index, "out_block": block},
                    "%s_%s" % (token_name, output_name), block,
                    space=initial.space, point=result_points[index])
            return _CoupledResult(projected)

        return SolveOutcome(self, token, project, token_name)

    # The LOCAL per-cell ops a solve_local_nonlinear residual sub-block may use: the iterate / guess
    # State placeholders, named per-cell sources / linear-source applies, and the affine combine of
    # them. All lower to a per-cell scalar expression in the cell-local conservative stack -- NO
    # non-local op (rhs / divergence / solve_fields / a nested solve) is allowed (it would need a halo
    # / global solve, which a per-cell Newton kernel cannot evaluate at a perturbed stack state).

    def _solve_local_nonlinear(self, *, residual: Any, initial_guess: Any, prepared: Any,
                               name: Any = None) -> Any:
        """Solve a LOCAL non-linear system ``residual(U) = 0`` cell by cell with a per-cell Newton
        iteration (spec op 10). Returns the converged solution State.

        @p residual is an IR-building callable ``residual_fn(P, U, U0) -> State``: given the Newton
        iterate State @p U and the frozen initial-guess State @p U0 it BUILDS the residual ``r(U)`` (a
        State value) from LOCAL per-cell ops only -- ``P.source`` (a named ``m.source_term``),
        ``P.apply`` (a named ``m.linear_source``), the iterate / initial-guess States, and the affine
        algebra over them (e.g. an implicit reaction ``r(U) = U - U0 - dt*S(U)``). A non-local op
        (a grid operator / callable field operator / nested solve) is rejected: the residual
        must be re-evaluable at a PERTURBED cell-local stack state, which a halo / global solve cannot.
        The sub-block (like a ``set_apply`` body) lowers to a device-inlinable per-cell residual the
        kernel re-evaluates at ``U`` and at the finite-difference perturbations ``U + eps*e_j``. A
        two-argument ``residual_fn(P, U)`` (ignoring the guess) is also accepted.

        @p initial_guess is the start State ``U0`` (typically ``U^n``); it seeds the immutable
        prepared problem and the residual reads it as a frozen per-cell constant.  The solver
        descriptor contributes explicit tolerances, evaluation/iteration budgets, scaling-free
        finite-difference controls and a safeguard policy.  Those controls are hashed on the IR.

        Generated code contributes only the residual functor.  The shared native provider owns the
        iterations, finite-difference/analytic Jacobian contract, pivoted linear solve and outcome.
        It returns a transaction-local candidate plus diagnostics; no candidate is publishable until
        the collective ``SolveReport`` has been consumed."""
        if not callable(residual):
            raise ValueError(
                "solve_local_nonlinear: residual must be an IR-building callable "
                "residual_fn(P, U, U0) returning the residual State r(U)")
        if not (isinstance(initial_guess, ProgramValue) and initial_guess.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: initial_guess must be a State value (initial_guess=...)")
        require_top_level(self, initial_guess, "solve_local_nonlinear")
        controls = _prepared_local_nonlinear_controls(
            prepared, where="LocalNewton")
        if self._recording:
            raise NotImplementedError(
                "solve_local_nonlinear: recording a residual inside another sub-block (apply / while "
                "body) is a later phase")
        block = initial_guess.block
        # Record the residual sub-block (like set_apply / a while body): the iterate U and the frozen
        # initial-guess U0 are State placeholders local to the sub-block; residual_fn builds r(U) from
        # them with LOCAL per-cell ops. The placeholders are NOT appended to self._values (they belong
        # to this op) -- the kernel binds the iterate to the cell stack and U0 to the frozen guess.
        wants_guess = _residual_wants_guess(residual)
        sub = []
        self._recording.append(sub)
        try:
            iterate = self._new(
                "state", "state", (), {}, "newton_iterate", block, space=initial_guess.space)
            guess_ph = self._new(
                "state", "state", (), {}, "newton_guess", block, space=initial_guess.space)
            # residual_fn(P, U, U0); a two-arg residual_fn(P, U) (ignoring the guess) is also accepted.
            r = residual(self, iterate, guess_ph) if wants_guess else residual(self, iterate)
        finally:
            self._recording.pop()
        if not (isinstance(r, ProgramValue) and r.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: residual_fn must return the residual State r(U) (got %r)" % (r,))
        residual_region = self._region_for_block(sub)
        require_region(self, r, residual_region, "solve_local_nonlinear residual",
                       vtype="state")
        require_compatible_spaces(
            initial_guess.space, r.space, "solve_local_nonlinear residual", typed_pair=True)
        for w in sub:
            if w.op not in self._RESIDUAL_LOCAL_OPS:
                raise ValueError(
                    "solve_local_nonlinear: residual op '%s' is not LOCAL; a per-cell Newton residual "
                    "may use only %s (the iterate / guess State, P.source, P.apply, affine combines). "
                    "Use non-local grid and field operators outside the local residual."
                    % (w.op, sorted(self._RESIDUAL_LOCAL_OPS)))
        token = self._new(
            "state", "solve_local_nonlinear", (initial_guess,),
            {"residual_block": sub, "residual_region": residual_region,
             "residual": r, "iterate": iterate, "guess": guess_ph,
             **controls, "method": "newton",
             "problem_kind": "local_residual",
             "solver_identity": prepared.identity.token,
            }, name, block,
            space=initial_guess.space)
        from pops.time.solve_outcome import SolveOutcome

        outcome_name = name or "local_residual"

        def project(outcome: Any) -> Any:
            return self._new(
                "state", "solve_outcome_component", (outcome,), {"index": 0},
                outcome_name + "_value", block, space=initial_guess.space,
                point=initial_guess.point)

        return SolveOutcome(self, token, project, outcome_name)

    def _linear_source_name(self, operator: Any, where: Any, values: Any = ()) -> Any:
        """Resolve `operator` to the linear-source name.

        Accepts a typed :class:`pops.model.OperatorHandle` (resolved against the exact bound registry),
        a validated `linear_source` ProgramValue, a single unit-coefficient ``_Operator`` term, or a
        bare name string on this private internal seam."""
        from pops.model import OperatorHandle
        if isinstance(operator, OperatorHandle):
            return resolve_operator_handle(
                self, operator, where=where,
                expected_kinds="local_linear_operator", values=values).name
        if isinstance(operator, str) and operator:
            return operator
        if isinstance(operator, ProgramValue) and operator.op == "linear_source":
            require_owned(self, operator, "%s operator" % where)
            return self._canonical_value(operator).attrs["linear_source"]
        if (isinstance(operator, _Operator) and not operator.identity.as_dict()
                and len(operator.terms) == 1 and operator.terms[0][1].as_dict() == {0: 1}):
            value = operator.terms[0][0]
            require_owned(self, value, "%s operator" % where)
            return self._canonical_value(value).attrs["linear_source"]
        raise ValueError(
            "%s: operator must be a linear source (P.linear_source(handle) or its OperatorHandle)"
            % where)

    def _linear_source(self, name: Any, operator_handle: Any = None) -> Any:
        """Internal seam: reference a linear source by its bare NAME (an internal selector).

        NOT a public surface -- it is the byte-identical lowering the public typed
        :meth:`linear_source` delegates to (after unwrapping its handle), and the path the internal
        lowering (``_lower_call``) and the ``pops.lib.time`` macros use directly with a bare name."""
        if not isinstance(name, str) or not name:
            raise ValueError("_linear_source: a non-empty operator name is required")
        attrs = {"linear_source": name}
        if operator_handle is not None:
            attrs["operator_handle"] = operator_handle
        return self._new("operator", "linear_source", (), attrs, name, None)

    def _apply(self, operator: Any = None, state: Any = None, fields: Any = None,
               name: Any = None) -> Any:
        """Internal seam: apply a linear source given as a typed value / handle OR a bare name.

        NOT a public surface -- the public :meth:`apply` refuses a bare-name string and delegates
        here; the solver-DSL and other internal callers pass the name selector directly."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("apply: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, ProgramValue) and fields.vtype == "fields"):
            raise ValueError("apply: fields must be a FieldContext from solve_fields")
        lname = self._linear_source_name(
            operator, "apply", tuple(value for value in (state, fields) if value is not None))
        field_context = None
        if fields is not None:
            from pops.time.field_context import require_field_read
            field_context = require_field_read(fields, state, "apply")
        self._check_operator_state(operator, state, "apply")
        inputs = (state, fields) if fields is not None else (state,)
        attrs = {"linear_source": lname}
        from pops.model import OperatorHandle
        if isinstance(operator, OperatorHandle):
            attrs["operator_handle"] = operator
        elif isinstance(operator, ProgramValue) and "operator_handle" in operator.attrs:
            attrs["operator_handle"] = operator.attrs["operator_handle"]
        return self._new(
            "rhs", "apply", inputs, attrs,
            name or ("apply_" + lname), state.block, space=rate_space_for(state.space),
            field_context=field_context)

    # --- matrix-free operators / dynamic linear solve (ADC-405 Phase 6b) ----------------------------
    # A ``matrix_free_op`` supplies an exact apply callback. Mathematical shape and hierarchy use
    # are validated by the selected prepared solver provider, not by this generic authoring core.

    def scalar_field(self, name: Any = None, ncomp: Any = 1) -> Any:
        """A fresh, zero-initialized scalar field: scratch the apply sub-block uses (e.g. the Laplacian
        output, or a 2-component gradient buffer). @p ncomp is the component count (1 by default; 2 for a
        gradient field consumed by ``P.divergence``). Lowered to ``ctx.alloc_scalar_field(ncomp, 1)``."""
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
            raise ValueError("scalar_field: ncomp must be a positive integer (got %r)" % (ncomp,))
        return self._new(
            "scalar_field", "scalar_field", (),
            {"ncomp": int(ncomp), "stencil_access": StencilAccess.pointwise()}, name, None)

    def vector_field(self, *components: Any, name: Any = None) -> Any:
        """Compose one exact spatial vector from scalar-field components.

        The component count is the authored spatial rank and is matched against the domain rank
        during resolve.  Native lowering allocates one field and packs each scalar into its matching
        axis component; no missing axis is padded and no 2D convention survives in the runtime.
        """
        if len(components) not in (1, 2, 3):
            raise ValueError(
                "vector_field requires exactly 1, 2, or 3 scalar components")
        for axis, component in enumerate(components):
            if not (isinstance(component, ProgramValue)
                    and component.vtype == "scalar_field"):
                raise ValueError(
                    "vector_field component %d must be a scalar_field value" % axis)
            if int(component.attrs.get("ncomp", 1)) != 1:
                raise ValueError(
                    "vector_field component %d must have exactly one component" % axis)
            require_owned(self, component, "vector_field")
        dimension = len(components)
        return self._new(
            "scalar_field", "vector_field", tuple(components),
            {"ncomp": dimension, "spatial_dimension": dimension,
             "stencil_access": StencilAccess.pointwise()}, name, None)

    def matrix_free_operator(self, name: Any, domain: Any = "scalar", range_: Any = "scalar",
                             ncomp: Any = None, *, scope: Any = None,
                             stencil_depth: Any = None) -> Any:
        """Declare a matrix-free operator ``A : domain -> range_``. @p domain / @p range_ are the field
        kind on each side and MUST match (a square operator: the Krylov iterate, residual and solution
        share one layout): ``"scalar"`` (a 1-component scalar field, the default), or ``"vector"`` /
        ``"state"`` (a multi-component field, e.g. a coupled block unknown). For a
        ``vector`` / ``state`` operator @p ncomp (an int >= 1) is REQUIRED -- the component count of the
        apply's in/out buffers and of the solution; for a ``scalar`` operator @p ncomp must be omitted
        (or 1). ``stencil_depth`` is the exact non-negative halo width required by a custom apply.
        When omitted, :meth:`set_apply` derives the minimum from PoPS' typed stencil operations;
        an explicit value may be larger (for an external/custom stencil) but never smaller than
        the derived minimum. Supply the apply via ``P.set_apply(A, body_fn)`` before using it in
        ``P.solve(LinearProblem(A, rhs, nullspace=None), solver=...)``."""
        if domain not in self._OPERATOR_KINDS or range_ not in self._OPERATOR_KINDS:
            raise ValueError(
                "matrix_free_operator: domain / range_ must be one of %s; got domain=%r range_=%r"
                % (sorted(self._OPERATOR_KINDS), domain, range_))
        if domain != range_:
            raise ValueError(
                "matrix_free_operator: domain and range_ must match (a square operator); got "
                "domain=%r range_=%r" % (domain, range_))
        if domain == "scalar":
            if ncomp not in (None, 1):
                raise ValueError(
                    "matrix_free_operator: a scalar operator has ncomp=1 (omit ncomp); got ncomp=%r"
                    % (ncomp,))
            ncomp = 1
        else:  # vector / state: an explicit positive component count is required
            if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
                raise ValueError(
                    "matrix_free_operator: a %r operator requires ncomp (an int >= 1); got ncomp=%r"
                    % (domain, ncomp))
        if (stencil_depth is not None
                and (isinstance(stencil_depth, bool) or not isinstance(stencil_depth, int)
                     or stencil_depth < 0)):
            raise ValueError(
                "matrix_free_operator: stencil_depth must be a non-negative integer or None; "
                "got %r" % (stencil_depth,))
        from pops.solvers.scopes import solve_scope_id
        scope_id = solve_scope_id(scope)
        attrs = {"domain": domain, "range": range_, "ncomp": int(ncomp), "apply_block": None,
                 "apply_result": None, "apply_in": None, "apply_out": None,
                 "declared_stencil_access": (
                     None if stencil_depth is None else StencilAccess(stencil_depth)),
                 "stencil_access": None}
        if scope_id != "level":
            attrs["scope"] = scope_id
        return self._new("matrix_free_op", "matrix_free_operator", (), attrs, name, None)

    @atomic_authoring
    def set_apply(self, operator: Any, body_fn: Any) -> Any:
        """Record the apply ``out <- A(in)`` of a ``matrix_free_operator``. @p body_fn(P, out, in) is an
        IR-building callable: @p in and @p out are scalar_field values (the operator's argument and
        result); the body builds @p out from @p in (e.g. ``P.laplacian(tmp, in); ...``) using
        ``P.laplacian`` + the affine algebra and RETURNS the result scalar_field (the value written into
        @p out). The ops are captured into a separate sub-block (like a while body) and re-emitted as a
        C++ lambda the Krylov loop calls."""
        operator = self._canonical_value(operator)
        if not (isinstance(operator, ProgramValue) and operator.vtype == "matrix_free_op"):
            raise ValueError("set_apply: operator must be a matrix_free_operator value")
        require_top_level(self, operator, "set_apply")
        if operator.attrs["apply_block"] is not None:
            raise ValueError("set_apply: operator '%s' already has an apply" % operator.name)
        if self._recording:
            raise NotImplementedError(
                "set_apply: recording an apply inside another sub-block (apply / while body) is a "
                "later phase")
        # The apply ops (the in/out placeholders + the body) live in the operator's OWN sub-block, NOT
        # the flat SSA list: they are re-emitted as the C++ apply lambda, never walked at the top level.
        sub = []
        self._recording.append(sub)
        # The in/out buffers carry the operator's component count: a vector / state operator applies on
        # an ncomp buffer (scalar -> ncomp == 1). The apply body sees ncomp-component in / out fields.
        op_ncomp = int(operator.attrs["ncomp"])
        try:
            pointwise = StencilAccess.pointwise()
            out_sf = self._new(
                "scalar_field", "apply_out", (),
                {"ncomp": op_ncomp, "stencil_access": pointwise}, "apply_out", None)
            in_sf = self._new(
                "scalar_field", "apply_in", (),
                {"ncomp": op_ncomp, "stencil_access": pointwise}, "apply_in", None)
            result = body_fn(self, out_sf, in_sf)
        finally:
            self._recording.pop()
        block = sub
        result = result if result is not None else out_sf
        if isinstance(result, ProgramValue) and result.vtype != "scalar_field":
            raise ValueError("set_apply: body_fn must return the result scalar_field (out <- A(in))")
        apply_region = self._region_for_block(sub)
        require_affine_region(self, result, apply_region, "set_apply")
        attrs = dict(operator.attrs)
        inferred_stencil_access = StencilAccess.compose(
            (node.attrs.get("stencil_access") for node in block),
            where="set_apply(%s)" % operator.name,
        )
        declared_stencil_access = attrs.get("declared_stencil_access")
        if (declared_stencil_access is not None
                and declared_stencil_access.required_ghost_depth
                < inferred_stencil_access.required_ghost_depth):
            raise ValueError(
                "set_apply: matrix_free_operator stencil_depth=%d is smaller than the typed "
                "apply's required depth %d" %
                (declared_stencil_access.required_ghost_depth,
                 inferred_stencil_access.required_ghost_depth))
        attrs["stencil_access"] = (
            inferred_stencil_access if declared_stencil_access is None
            else StencilAccess.compose(
                (inferred_stencil_access, declared_stencil_access),
                where="set_apply(%s) declaration" % operator.name,
            ))
        captured_scopes = {
            item.attrs.get("scope", "level")
            for node in block for item in node.inputs
            if isinstance(item, ProgramValue)
        }
        if "hierarchy" in captured_scopes:
            attrs["scope"] = "hierarchy"
        attrs.update({
            "apply_block": block,
            "apply_region": apply_region,
            "apply_result": result,
            "apply_in": in_sf,
            "apply_out": out_sf,
        })
        return self._replace_value(operator, attrs=attrs)

    def laplacian(self, out: Any, in_: Any) -> Any:
        """Record ``out = Lap(in_)`` (the shared exact-ranked Cartesian Laplacian). @p out and @p in_ are
        scalar_field values. Lowered to ``ctx.laplacian(out, in_)``. Used inside an apply sub-block to
        form a Helmholtz operator ``A(in) = in - alpha*Lap(in)`` via the affine algebra."""
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("laplacian: out must be a scalar_field value")
        if not (isinstance(in_, ProgramValue) and in_.vtype == "scalar_field"):
            raise ValueError("laplacian: in must be a scalar_field value")
        return self._new(
            "scalar_field", "laplacian", (out, in_),
            {"stencil_access": StencilAccess.nearest_neighbour()}, out.name, None)

    def gradient(self, out: Any, phi: Any) -> Any:
        """Record ``out = grad(phi)`` (centered differences; @p out has one component per native axis). @p out and
        @p phi are scalar_field values. Lowered to ``ctx.gradient(out, phi)``."""
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("gradient: out must be a scalar_field value")
        if not (isinstance(phi, ProgramValue) and phi.vtype == "scalar_field"):
            raise ValueError("gradient: phi must be a scalar_field value")
        output_components = int(out.attrs.get("ncomp", 1))
        if output_components not in (1, 2, 3):
            raise ValueError("gradient: out must carry one component per spatial axis")
        if int(phi.attrs.get("ncomp", 1)) != 1:
            raise ValueError("gradient: phi must have exactly one component")
        return self._new(
            "scalar_field", "gradient", (out, phi),
            {"ncomp": output_components, "spatial_dimension": output_components,
             "stencil_access": StencilAccess.nearest_neighbour()}, out.name, None)

    def divergence(self, out: Any, flux: Any) -> Any:
        """Record ``out = div(flux)`` for one exact-ranked vector field.

        ``flux`` carries one component per native axis, so it is directly composable with
        :meth:`gradient`; native binding authenticates its width against the resolved domain.
        """
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("divergence: out must be a scalar_field value")
        if not (isinstance(flux, ProgramValue) and flux.vtype == "scalar_field"):
            raise ValueError("divergence: flux must be a scalar_field value")
        if int(out.attrs.get("ncomp", 1)) != 1:
            raise ValueError("divergence: out must have exactly one component")
        flux_components = int(flux.attrs.get("ncomp", 1))
        if flux_components not in (1, 2, 3):
            raise ValueError("divergence: flux must carry one component per spatial axis")
        return self._new(
            "scalar_field", "divergence", (out, flux),
            {"ncomp": 1, "spatial_dimension": flux_components,
             "stencil_access": StencilAccess.nearest_neighbour()}, out.name, None)

    # --- finite-difference Jacobian-vector product (ADC-431: implicit-flux BDF Newton-Krylov) --------
    def rhs_jacvec(self, out: Any, in_: Any, *, iterate: Any, r0: Any, c_dt: Any, eps: Any = 1e-7,
                   flux: Any = True, sources: Any = ("default",),
                   field_coupled: Any) -> Any:
        """Record the finite-difference Jacobian-vector product of an implicit-flux residual, INSIDE a
        matrix_free_operator apply sub-block (ADC-431). It lowers to ``out <- J(@p iterate) @p in`` where
        the Newton-system Jacobian is ``J = I - c*dt * d(rhs)/dU`` and the matvec is formed matrix-free by
        a directional finite difference::

            out = in - (c*dt/eps) * (rhs(U^k + eps*in) - rhs(U^k))

        @p out / @p in_ are the apply sub-block's out / in scalar_field buffers (carrying the operator's
        component count). @p iterate is the FROZEN Newton iterate ``U^k`` (a State, defined OUTSIDE the
        apply, captured into the apply lambda); @p r0 is the precomputed ``rhs(U^k)`` (a State/RHS value,
        also captured) so the perturbation cost is one ``rhs`` per matvec. @p c_dt is the BDF coefficient
        ``c*dt`` (a number or a dt-polynomial: ``c == 1`` for BDF1, ``c == 2/3`` for BDF2). @p eps is the
        relative FD step (scaled by ``||U^k|| / ||in||`` inside the kernel). The implemented codegen
        linearizes the default flux with either its default/composite source (``sources=None`` or
        ``["default"]``) or no source (``sources=[]``). ``flux=False`` and named source terms are
        rejected rather than silently emitting a different Jacobian. The op may ONLY appear inside
        ``set_apply`` (it captures the apply's in/out buffers). ``field_coupled`` states explicitly
        whether both the base residual and every perturbed residual solve fields from their own state.

        Unlike the cell-local FD Jacobian of `solve_local_nonlinear` (a per-cell dense inverse), this is a
        GLOBAL operator: ``rhs`` couples the cells through the flux stencil, so the matvec is dense over
        the coupled stencil and the Newton step ``J dU = -F`` is solved by `solve_linear` (GMRES)."""
        if not self._recording:
            raise ValueError("rhs_jacvec may only be recorded inside a matrix_free_operator apply "
                             "(call it from the set_apply body_fn)")
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("rhs_jacvec: out must be the apply sub-block's out scalar_field value")
        if not (isinstance(in_, ProgramValue) and in_.vtype == "scalar_field"):
            raise ValueError("rhs_jacvec: in_ must be the apply sub-block's in scalar_field value")
        if not (isinstance(iterate, ProgramValue) and iterate.vtype == "state"):
            raise ValueError("rhs_jacvec: iterate must be the frozen Newton-iterate State (iterate=...)")
        if not (isinstance(r0, ProgramValue) and r0.op == "rhs" and len(r0.inputs) >= 1):
            raise ValueError("rhs_jacvec: r0 must be an exact precomputed rhs(iterate) value")
        if r0.inputs[0] is not iterate:
            raise ValueError("rhs_jacvec: r0 must be computed from the exact frozen iterate")
        if r0.block != iterate.block or r0.point != iterate.point:
            raise ValueError(
                "rhs_jacvec: r0 and iterate must share one exact block and temporal point")
        if not isinstance(field_coupled, bool):
            raise TypeError("rhs_jacvec: field_coupled must be a bool")
        context = getattr(r0, "field_context", None)
        field = getattr(context, "field", None)
        stage_sources = tuple(getattr(context, "stage_sources", ()))
        context_matches = (
            field is not None
            and stage_sources == ((iterate.block, iterate.id),)
            and len(r0.inputs) == 2
            and getattr(r0.inputs[1], "vtype", None) == "fields"
            and getattr(r0.inputs[1], "field_context", None) == context
        )
        if field_coupled and not context_matches:
            raise ValueError(
                "rhs_jacvec: field_coupled=True requires r0 computed with one unambiguous field "
                "context solved only from the frozen iterate")
        if not field_coupled and context is not None:
            raise ValueError(
                "rhs_jacvec: field_coupled=False requires an r0 with no field-solve provenance")
        if not field_coupled and len(r0.inputs) != 1:
            raise ValueError(
                "rhs_jacvec: field_coupled=False requires r0 to consume only the frozen iterate")
        if flux is not True:
            raise NotImplementedError(
                "rhs_jacvec cannot linearize flux=False: the matrix-free kernel currently requires "
                "the default flux divergence")
        # ``None`` is the public shorthand for the model's default source.  Public
        # ``Program.rhs(terms=[DefaultSource()])`` stores the same choice canonically as
        # ``("default",)``; normalize both before authenticating the frozen residual.
        src = ["default"] if sources is None else list(sources)
        named_sources = [source for source in src if source != "default"]
        if named_sources:
            raise NotImplementedError(
                "rhs_jacvec cannot linearize named sources %r yet; use sources=[] (flux-only) or "
                "sources=['default']" % named_sources)
        r0_sources = r0.attrs.get("sources")
        if r0_sources is not None:
            r0_sources = list(r0_sources)
        if (r0.attrs.get("flux") is not True or r0_sources != src
                or r0.attrs.get("fluxes") not in (None, (), [])):
            raise ValueError(
                "rhs_jacvec: r0 must use the exact same default-flux/default-source selection "
                "and may not use a named flux")
        if not isinstance(c_dt, _Coeff):
            try:
                c_dt = _Coeff({0: _exact_number(c_dt)})
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "rhs_jacvec: c_dt must be a number or a dt-polynomial (got %r)" % (c_dt,)
                ) from exc
        eps_literal = positive_scalar_literal(eps, where="rhs_jacvec: eps")
        c_d = c_dt.to_polynomial()
        return self._new("scalar_field", "rhs_jacvec", (out, in_, iterate, r0),
                         {"c_dt": c_d, "eps": eps_literal, "flux": True, "sources": src,
                          "field_coupled": field_coupled,
                          "stencil_access": StencilAccess.nearest_neighbour()},
                         out.name, iterate.block, state_ref=iterate.state_ref)

    # --- tensor-coefficient matrix-free apply of the generic condensed-implicit route (ADC-637) --------
    def apply_laplacian_coeff(self, out: Any, in_: Any, coeffs: Any) -> Any:
        """Record ``out = div(A grad in_)`` with the tensor ``A`` of a @ref condensed_coeffs bundle (the
        coefficiented matrix-free matvec of the condensed-implicit elliptic operator,
        ``pops::apply_laplacian``'s coefficient path). @p out and @p in_ are scalar_field values; @p
        coeffs is a ``condensed_coeffs`` value. Used inside a matrix-free apply: the condensed operator
        ``L(phi) = -div(A grad phi) = -out``, so build it as ``-1 * P.apply_laplacian_coeff(out, in_,
        A)`` via the affine algebra. Emitted INLINE (ctx.fill_boundary + pops::apply_laplacian), no
        coupling/schur call."""
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("apply_laplacian_coeff: out must be a scalar_field value")
        if not (isinstance(in_, ProgramValue) and in_.vtype == "scalar_field"):
            raise ValueError("apply_laplacian_coeff: in_ must be a scalar_field value")
        # The exact-ranked condensed bundle carries one row-major Dim*Dim tensor field.
        if not (isinstance(coeffs, ProgramValue) and coeffs.vtype == "condensed_coeffs"):
            raise ValueError("apply_laplacian_coeff: coeffs must be a coefficient bundle "
                             "(P.condensed_coeffs(...))")
        # ``out`` / ``in_`` are matrix-free scratch buffers and therefore intentionally have no
        # block of their own.  The coefficient bundle is different: it is assembled from one
        # concrete state and determines the physical block on which this apply is valid.  Preserve
        # that provenance on the result instead of letting ``_new`` infer only ``state_ref`` while
        # leaving ``block=None`` (an impossible half-qualified ProgramValue).
        return self._new(
            "scalar_field", "apply_laplacian_coeff", (out, in_, coeffs),
            {"stencil_access": StencilAccess.nearest_neighbour()}, out.name,
            coeffs.block, state_ref=coeffs.state_ref)
