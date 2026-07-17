"""Program solve, history, value, commit, and record operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.handles import (
    HistoryHandle,
    StageHandle,
    StateEndpointHandle,
    TimeState,
)
from pops.time._program.constants import _ProgramConstants
from pops.time._program.commit_validation import validate_commit_many
from pops.time._program.diagnostics import _ProgramDiagnostics
from pops.time.references import block_name
from pops.time._program.value_validation import (
    require_compatible_spaces,
    require_top_level,
    structural_state_space,
)
from pops.time.solve_outcome import SolveOutcome
from pops.time.value_metadata import positive_scalar_literal
from pops.time.values import ProgramValue, _Affine, _is_field_value, _resolve_handle

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


# Preconditioner schemes that lower to REAL C++ in the matrix-free Krylov path (Spec 5 sec.7, ADC-516):
#   - "identity":     the empty pops::ApplyFn{} (unpreconditioned; the historical default);
#   - "geometric_mg": one V-cycle of the wired pops::GeometricMG, emitted as a real ApplyFn callback.
_WIRED_PRECOND_SCHEMES = frozenset({"identity", "geometric_mg"})


def _lower_preconditioner(preconditioner: Any) -> Any:
    """Lower a typed preconditioner descriptor to ``(scheme, precond_options|None)`` (Spec 5 sec.7).

    ``preconditioner`` is a :mod:`pops.solvers.preconditioners` descriptor
    (``preconditioners.Identity()`` / ``preconditioners.GeometricMG()`` ...); its ``scheme`` is the
    C++ token. A bare string is REJECTED; ``None`` defaults to ``Identity()`` (the unpreconditioned
    default). The geometric-multigrid preconditioner lowers to a real V-cycle ApplyFn.

    ADC-644: a ``GeometricMG(...)`` with validated V-cycle-shape knobs returns its option dict; a
    default one returns ``None`` (the IR omits ``precond_options`` -> emitted V-cycle byte-identical).
    """
    if preconditioner is None:
        preconditioner = _preconditioners().Identity()
    if isinstance(preconditioner, str):
        raise TypeError(
            "solve_linear: preconditioner must be a typed pops.solvers.preconditioners "
            "descriptor (e.g. pops.solvers.preconditioners.Identity() / GeometricMG()), not the "
            "string %r" % (preconditioner,)
        )
    scheme = getattr(preconditioner, "scheme", None)
    if getattr(preconditioner, "category", None) != "preconditioner" or not isinstance(scheme, str):
        raise TypeError(
            "solve_linear: preconditioner must be a pops.solvers.preconditioners descriptor "
            "(e.g. Identity() / GeometricMG()); got %r" % (preconditioner,)
        )
    if scheme not in _WIRED_PRECOND_SCHEMES:
        raise NotImplementedError(
            "solve: the %r preconditioner has no executable Program route; use "
            "preconditioners.Identity() or preconditioners.GeometricMG()" % (scheme,)
        )
    options = getattr(preconditioner, "options", None)
    precond_options = dict(options) if options else None
    return scheme, precond_options


def _preconditioners() -> Any:
    """The pops.solvers.preconditioners catalog (imported lazily)."""
    from pops.solvers import preconditioners

    return preconditioners


class _ProgramSolve(_ProgramDiagnostics, _ProgramConstants, _ProgramBase):
    """Private Krylov lowering plus histories, commits and records."""

    def _solve_linear(
        self,
        *,
        operator: Any,
        rhs: Any,
        prepared: Any,
        properties: Any,
        nullspace_contract: Any,
        gauge_contract: Any,
        initial_guess: Any = None,
        name: Any = None,
        at: Any = None,
        scope: Any = None,
    ) -> Any:
        """Solve the matrix-free linear system ``operator x = rhs`` with the runtime's Krylov loop and
        return the solution as a field. A state-domain operator with a State rhs returns a State;
        scratch/vector solves return a scalar_field. The iteration is DYNAMIC (C++-side, inside the loop):
        the IR only carries the operator (its apply lambda), the rhs, the initial guess, and the
        method / tolerance / iteration budget.

          - @p operator: a ``matrix_free_operator`` value (with a ``set_apply`` body);
          - @p rhs: the right-hand side -- a scalar_field or State value. A typed StateSpace must have
            exactly the operator's ``ncomp``; an untyped State is checked against the physical model
            component count during lowering;
          - @p initial_guess: warm start (defaults to zero);
          - @p method: a TYPED Krylov descriptor (``pops.solvers.krylov.CG()`` (SPD),
            ``BiCGStab()`` (general), ``Richardson()``, or ``GMRES()`` -- restarted GMRES(m), the
            robust choice for a NON-symmetric operator). A bare string is REJECTED (Spec 5
            sec.7); ``None`` defaults to ``CG()``;
          - @p preconditioner: a typed ``pops.solvers.preconditioners`` descriptor.
            ``Identity()`` (the unpreconditioned default) and ``GeometricMG()`` (one V-cycle of the
            wired geometric multigrid, for ``GMRES()`` / ``BiCGStab()`` only) lower to real C++; the
            planned ``Jacobi()`` / ``BlockJacobi()`` are rejected (no native kernel yet). A non-identity
            preconditioner with ``CG()`` / ``Richardson()`` is rejected (those loops have no
            preconditioner slot). A bare string is REJECTED; ``None`` defaults to ``Identity()``;
          - @p tol: relative L2 residual stop (>= 0); zero selects absolute-only stopping and
            requires a positive ``abs_tol`` on the prepared solver;
          - @p max_iter: iteration budget (REQUIRED, > 0: a dynamic solver loop with no budget is a
            configuration error refused before the prepared native route is entered);
          - @p restart: GMRES restart length m (a positive int; defaults to 30). Ignored by the other
            methods; passing it to a non-gmres solve is rejected."""
        operator = self._canonical_value(operator)
        from pops.solvers.scopes import solve_scope_id

        solve_scope = (
            operator.attrs.get("scope", "level") if scope is None else solve_scope_id(scope)
        )
        if operator.attrs.get("scope") == "hierarchy" and solve_scope != "hierarchy":
            raise ValueError("solve: a hierarchy-scoped operator cannot be downgraded to Level()")
        if solve_scope == "hierarchy":
            raise TypeError(
                "solve: Hierarchy() is a direct native solve and requires "
                "CompositeTensorFAC(max_iter=..., rel_tol=...); Krylov descriptors solve Level() "
                "operators only"
            )
        method = getattr(prepared, "method", None)
        tol = getattr(prepared, "tolerance", None)
        abs_tol = getattr(prepared, "absolute_tolerance", None)
        max_iter = getattr(prepared, "max_iterations", None)
        restart = getattr(prepared, "restart", None)
        preconditioner_data = getattr(prepared, "preconditioner", None)
        if not (isinstance(preconditioner_data, tuple) and len(preconditioner_data) == 2):
            raise TypeError("solve: Krylov provider has an invalid preconditioner contract")
        preconditioner, precond_options = preconditioner_data
        omega = getattr(prepared, "omega", None)
        solver_identity = getattr(prepared, "identity", None)
        solver_identity_token = getattr(solver_identity, "token", None)
        if not isinstance(solver_identity_token, str):
            raise TypeError("solve: Krylov provider identity is not canonical")
        if not (isinstance(operator, ProgramValue) and operator.vtype == "matrix_free_op"):
            raise ValueError("solve_linear: operator must be a matrix_free_operator value")
        if operator.attrs["apply_block"] is None:
            raise ValueError(
                "solve_linear: operator '%s' has no apply; call P.set_apply first" % operator.name
            )
        if not _is_field_value(rhs):
            raise ValueError("solve_linear: rhs must be a scalar_field or State value (rhs=...)")
        if initial_guess is not None and not _is_field_value(initial_guess):
            raise ValueError("solve_linear: initial_guess must be a scalar_field or State value")
        if initial_guess is not None:
            unqualified_initial = (
                initial_guess.vtype == "scalar_field"
                and initial_guess.block is None
                and initial_guess.space is None
            )
            unqualified_rhs = (
                rhs.vtype == "scalar_field" and rhs.block is None and rhs.space is None
            )
            # A fresh scalar scratch has no independent owner/layout identity: the qualified
            # operand supplies it.  This is the generic condensed-solve case (an owner-qualified
            # persistent warm start against a freshly allocated RHS).  Two explicit owners must
            # still agree; no owner is guessed when both operands are qualified.
            if initial_guess.block != rhs.block and not (unqualified_initial or unqualified_rhs):
                raise ValueError(
                    "solve_linear: rhs and initial_guess must belong to the same block"
                )
            if not (unqualified_initial or unqualified_rhs):
                require_compatible_spaces(
                    rhs.space, initial_guess.space, "solve_linear initial_guess", typed_pair=True
                )
        op_ncomp = int(operator.attrs["ncomp"])
        # The rhs and initial guess must inhabit exactly the operator's vector space.  The native
        # prepared problem intentionally has no implicit component slicing; accepting a wider field
        # here would only fail later when the exact ncomp solution buffer is bound.
        for label, fld in (("rhs", rhs), ("initial_guess", initial_guess)):
            if fld is None:
                continue
            state_space = structural_state_space(fld.space)
            if fld.vtype == "state" and state_space is not None:
                fld_ncomp = len(state_space.components)
                if fld_ncomp != op_ncomp:
                    raise ValueError(
                        "solve_linear: %s StateSpace has %d component(s) but the operator declares "
                        "ncomp=%d" % (label, fld_ncomp, op_ncomp)
                    )
                continue
            if fld.vtype != "scalar_field":
                continue
            fld_ncomp = int(fld.attrs.get("ncomp", 1))
            if fld_ncomp != op_ncomp:
                raise ValueError(
                    "solve_linear: %s has %d component(s) but the operator declares ncomp=%d; "
                    "select an explicit component view before solving"
                    % (label, fld_ncomp, op_ncomp)
                )
        if method not in self._KRYLOV_METHODS:
            raise ValueError(
                "solve_linear: method must be one of %s; got %r"
                % (sorted(self._KRYLOV_METHODS), method)
            )
        from pops.linalg import LinearOperatorProperties

        if not isinstance(properties, LinearOperatorProperties):
            raise TypeError("solve_linear: properties must be pops.linalg.LinearOperatorProperties")
        from pops._ir.literals import ScalarLiteral

        if (
            type(nullspace_contract) is not dict
            or set(nullspace_contract) != {"schema_version", "kind"}
            or type(nullspace_contract.get("schema_version")) is not int
            or nullspace_contract.get("schema_version") != 1
        ):
            raise TypeError("solve_linear: nullspace contract is not canonical")
        nullspace_kind = nullspace_contract.get("kind")
        if nullspace_kind == "none":
            if type(gauge_contract) is not dict or gauge_contract != {
                "schema_version": 1,
                "kind": "none",
            }:
                raise TypeError(
                    "solve_linear: a nonsingular problem requires the canonical no-gauge contract"
                )
        elif nullspace_kind == "constant":
            if (
                type(gauge_contract) is not dict
                or set(gauge_contract) != {"schema_version", "kind", "value"}
                or type(gauge_contract.get("schema_version")) is not int
                or gauge_contract.get("schema_version") != 1
                or gauge_contract.get("kind") != "mean_value"
                or type(gauge_contract.get("value")) is not ScalarLiteral
            ):
                raise TypeError(
                    "solve_linear: ConstantNullspace requires an authenticated MeanValueGauge "
                    "snapshot"
                )
            if op_ncomp != 1:
                raise ValueError(
                    "solve_linear: ConstantNullspace is scalar-only (ncomp=1); no vector nullspace "
                    "basis is inferred"
                )
        else:
            raise TypeError(
                "solve_linear: nullspace must be the explicit canonical none/constant contract"
            )
        declared_nullspace = nullspace_kind == "constant"
        if declared_nullspace and preconditioner != "identity":
            raise NotImplementedError(
                "solve_linear: ConstantNullspace currently requires preconditioners.Identity(); "
                "GeometricMG has no explicit public certificate that its prepared V-cycle "
                "preserves the mean-zero complement"
            )
        if method == "cg" and not properties.certifies_cg(declared_nullspace=declared_nullspace):
            required = (
                "LinearOperatorProperties.symmetric_positive_definite_on_nullspace_complement()"
                if declared_nullspace
                else "LinearOperatorProperties.symmetric_positive_definite()"
            )
            raise ValueError("solve_linear: CG requires %s; no property is inferred" % required)
        # A non-identity preconditioner needs the runtime ApplyFn slot, which only the Krylov methods
        # that take one (BiCGStab / GMRES) expose. CG / Richardson have no preconditioner slot. This is
        # an honest capability limit of the matrix-free path,
        # not a transitional reject.
        if preconditioner != "identity" and method not in ("gmres", "bicgstab"):
            raise ValueError(
                "solve_linear: preconditioning is not available for CG/Richardson in the matrix-free "
                "Krylov path; use GMRES() or BiCGStab()"
            )
        if preconditioner == "geometric_mg" and op_ncomp != 1:
            raise ValueError(
                "solve_linear: preconditioners.GeometricMG() is scalar-only (ncomp=1); "
                "a component-coupled multigrid operator is not implemented, so a multi-component "
                "Krylov solve must use Identity() or another genuinely block-aware provider"
            )
        from pops._ir.literals import PREPARED_GMRES_MAX_RESTART, exact_cpp_int, scalar_literal

        try:
            tol_literal = scalar_literal(tol)
            tol_value = tol_literal.to_python()
            tol_valid = 0 <= tol_value < 1
        except (OverflowError, TypeError, ValueError):
            tol_valid = False
        if not tol_valid:
            raise ValueError("solve_linear: rel_tol must be a finite scalar literal in [0, 1)")
        try:
            abs_tol_literal = scalar_literal(abs_tol)
            abs_tol_value = abs_tol_literal.to_python()
            abs_tol_valid = abs_tol_value >= 0
        except (OverflowError, TypeError, ValueError):
            abs_tol_valid = False
        if not abs_tol_valid:
            raise ValueError("solve_linear: abs_tol must be a finite scalar literal >= 0")
        if tol_value == 0 and abs_tol_value == 0:
            raise ValueError(
                "solve_linear: rel_tol and abs_tol cannot both be zero; at least one stopping "
                "threshold must be positive"
            )
        try:
            max_iter_int = exact_cpp_int(max_iter, where="solve_linear: max_iter", minimum=1)
        except ValueError as exc:
            raise ValueError(
                "dynamic solver loops require max_iter as a positive signed C++ int"
            ) from exc
        # restart is a gmres-only knob; the GMRES(m) basis size. Other methods have no restart concept,
        # so passing one to them is a config error (fail loud rather than silently ignore it).
        if method == "gmres":
            if restart is None:
                restart = self._GMRES_RESTART_DEFAULT
            try:
                restart_int = exact_cpp_int(
                    restart,
                    where=(
                        "solve_linear: GMRES restart "
                        "(MPI Arnoldi reduction count requires restart + 1)"
                    ),
                    minimum=1,
                    maximum=PREPARED_GMRES_MAX_RESTART,
                )
            except ValueError as exc:
                raise ValueError(
                    "solve_linear: restart exceeds the native batched robust-dot collective "
                    "capacity (got %r)" % (restart,)
                ) from exc
        elif restart is not None:
            raise ValueError(
                "solve_linear: restart only applies to method='gmres' (got method=%r)" % (method,)
            )
        else:
            restart_int = None
        inputs = (operator, rhs) if initial_guess is None else (operator, rhs, initial_guess)
        from pops.time.stencil import StencilAccess

        stencil_access = operator.attrs.get("stencil_access")
        if type(stencil_access) is not StencilAccess:
            raise ValueError(
                "solve_linear: matrix-free operator has no authenticated StencilAccess; "
                "call set_apply on a current operator declaration"
            )
        input_ghosts = stencil_access.required_ghost_depth
        attrs = {
            "method": method,
            "preconditioner": preconditioner,
            "tol": tol_literal,
            "abs_tol": abs_tol_literal,
            "max_iter": max_iter_int,
            "has_guess": initial_guess is not None,
            "ncomp": op_ncomp,
            "restart": restart_int,
            "operator_properties": properties.canonical_data(),
            "nullspace_contract": dict(nullspace_contract),
            "gauge_contract": dict(gauge_contract),
            "krylov_footprint": {
                "components": op_ncomp,
                "input_ghosts": input_ghosts,
                "restart": restart_int or 0,
                "preconditioned": preconditioner != "identity",
            },
        }
        # ADC-644: the resolved V-cycle-shape options of a configured GeometricMG preconditioner. Added
        # ONLY when non-None (a default GeometricMG() lowers to None), so an unconfigured program's IR
        # hash / emitted source stays byte-identical (the attr is JSON-dumped into _serialize_node).
        if precond_options is not None:
            attrs["precond_options"] = precond_options
        # ADC-645: Richardson relaxation factor, added ONLY when the descriptor set it (a default
        # Richardson() program's IR hash / emitted source stays byte-identical: omega = 1 literal).
        if omega is not None:
            attrs["omega"] = positive_scalar_literal(omega, where="solve_linear: Richardson omega")
        attrs["solver_identity"] = solver_identity_token
        attrs["problem_kind"] = "matrix_free_linear"
        # A state-domain solve over a State rhs returns a State, preserving the mathematical unknown's
        # block and StateSpace. Scalar/vector scratch solves remain scalar_field values. This keeps a
        # Newton update ``U + dU`` typed without an implicit scalar-field-to-State conversion.
        result_type = (
            "state"
            if operator.attrs["domain"] == "state" and rhs.vtype == "state"
            else "scalar_field"
        )
        token = self._new(
            result_type,
            "solve_linear",
            inputs,
            attrs,
            name,
            rhs.block,
            space=rhs.space,
            point=rhs.point if at is None else at,
        )
        outcome_name = name or token.name

        def project(outcome: Any) -> Any:
            return self._new(
                result_type,
                "solve_outcome_component",
                (outcome,),
                {"index": 0, "ncomp": op_ncomp},
                outcome_name,
                rhs.block,
                space=rhs.space,
                point=token.point,
            )

        return SolveOutcome(self, token, project, outcome_name)

    def commit(self, endpoint: StateEndpointHandle, state: ProgramValue) -> None:
        """Commit ``state`` to ``endpoint``, at most once for that qualified state.

        The public form is exactly ``P.commit(U.next, U_next)``. ``U.next`` is a
        commit-only :class:`StateEndpointHandle`; a block-name string is not a public target and a
        stage handle is not implicitly resolved. ``state`` must already be a ProgramValue owned by
        this Program.

        A 1-component model's conservative state may be represented by a ``scalar_field`` (for
        example a ``solve_linear`` result); it is accepted and copied back into the block state by
        the final runtime ``ctx.lincomb``."""
        self._guard_mutable("commit a state")
        if not isinstance(endpoint, StateEndpointHandle):
            raise TypeError(
                "commit: target must be U.next (a StateEndpointHandle); block-name strings and "
                "stage handles are not public commit targets"
            )
        endpoint = self._require_endpoint(endpoint, "commit")
        if isinstance(state, ProgramValue) and state.block != endpoint.block:
            raise ValueError(
                "commit: cross-block write: endpoint for block %r cannot receive a value owned "
                "by block %r" % (block_name(endpoint.block), block_name(state.block))
            )
        require_top_level(self, state, "commit")
        if state.clock != endpoint.clock:
            raise ValueError(
                "commit: endpoint clock %r cannot receive value %r on clock %r; "
                "insert Program.synchronize(..., at=TimePoint(endpoint.clock)) first"
                % (endpoint.clock.name, state.name, state.clock.name)
            )
        if state.point != endpoint.point:
            raise ValueError(
                "commit: value %r is at %r, but the endpoint is at %r; construct the final "
                "value with at=U.next.point" % (state.name, state.point, endpoint.point)
            )
        require_compatible_spaces(endpoint.space, state.space, "commit", typed_pair=True)
        return self._commit_state(endpoint.state, state)

    def _commit_state(self, state_ref: Any, state: ProgramValue) -> None:
        """Record one validated qualified-state commit."""
        self._guard_mutable("commit a state")
        from pops.model.handles import Handle

        if (
            not isinstance(state_ref, Handle)
            or state_ref.kind != "state"
            or not state_ref.is_instance
        ):
            raise TypeError("_commit_state: target must be a block-qualified state Handle")
        block = state_ref.block_ref
        if not (isinstance(state, ProgramValue) and state.vtype in ("state", "scalar_field")):
            raise TypeError("_commit_state: a State (or scalar_field) ProgramValue is required")
        if state.prog is not self:
            raise ValueError("_commit_state: the State value belongs to a different Program")
        require_top_level(self, state, "_commit_state")
        if state.block != block:
            raise ValueError(
                "_commit_state: block %r cannot receive a value owned by block %r"
                % (block_name(block), block_name(state.block))
            )
        if state.state_ref is not None and state.state_ref != state_ref:
            raise ValueError(
                "_commit_state: state %s cannot receive a value derived from %s"
                % (state_ref.qualified_id, state.state_ref.qualified_id)
            )
        if state_ref not in self._state_spaces:
            raise ValueError(
                "_commit_state: state %s has no declared StateSpace" % state_ref.qualified_id
            )
        require_compatible_spaces(
            self._state_spaces[state_ref], state.space, "_commit_state", typed_pair=True
        )
        if state_ref in self._commits:
            raise ValueError("state %s committed more than once" % state_ref.qualified_id)
        self._commits[state_ref] = state

    def commits(self) -> Any:
        """Map of qualified state Handle -> committed State value (copy)."""
        return dict(self._commits)

    def value(self, name: Any, expr: Any, *, at: Any = None) -> Any:
        """Materialize one named SSA value or one exact temporal stage.

        ``T.value("U1", expression, at=point)`` is the sole free-value form. A
        :class:`StageHandle` may replace the name; its exact point and generated
        qualified name are then authoritative. Endpoints remain commit-only.
        """
        if isinstance(name, StateEndpointHandle):
            self._require_endpoint(name, "T.value")
            raise TypeError(
                "T.value: U.next is a commit-only StateEndpointHandle; use T.commit(U.next, value)"
            )
        if isinstance(name, StageHandle):
            if at is not None:
                raise TypeError("T.value(stage, expr) gets its point from the StageHandle")
            return self._define_stage(name, expr)
        if isinstance(name, HistoryHandle):
            self._require_history(name, "T.value")
            raise ValueError("history is produced by the history policy")
        if isinstance(name, TimeState):
            self._require_time_state(name, "T.value")
            raise TypeError("T.value requires a string name or a TimeState.stage(...) handle")
        if isinstance(name, ProgramValue):
            raise ValueError("current state is read-only in Program")
        if not isinstance(name, str) or not name:
            raise ValueError("T.value: name must be a non-empty string")
        value = _resolve_handle(expr)
        from pops import math as _bm

        if isinstance(value, _bm.Equation):
            if not isinstance(value.lhs, _bm.TimeDerivative):
                raise ValueError(
                    "value(%r): an equation must read 'rate(U) == <rate expression>'" % (name,)
                )
            value = value.rhs
        if isinstance(value, _Affine):
            return self._linear_combine(name, value, at=at)
        if isinstance(value, ProgramValue):
            return self._replace_value(value, name=name, point=value.point if at is None else at)
        raise TypeError(
            "value(%r): expected a ProgramValue, an affine combination, or a rate equation; got %r"
            % (name, value)
        )

    def commit_many(self, mapping: Any) -> None:
        """Commit ``{Ua.next: Ua_next, Ub.next: Ub_next}`` as one atomic group.

        Every endpoint/value owner and block is checked before ``_commits`` changes.
        """
        self._guard_mutable("commit a state group")
        self._commits.update(validate_commit_many(self, mapping))

    # --- inspection / debug (Spec 3 section 33): show the lowering ---
