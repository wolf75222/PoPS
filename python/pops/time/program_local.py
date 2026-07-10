"""pops.time Program authoring mixin -- local + matrix-free ops.

Local solves, matrix-free operators, laplacian/gradient/divergence and the coefficiented apply.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.program_transaction import atomic_authoring
from pops.time.program_value_validation import (
    rate_space_for, require_affine_region, require_compatible_spaces, require_region,
    require_owned, require_top_level,
)
from pops.time.operator_resolution import resolve_operator_handle
from pops.time.value_metadata import positive_scalar_literal
from pops.time.values import (
    ProgramValue, _Affine, _Coeff, _Operator, _exact_number, _is_field_value,
    _residual_wants_guess, _resolve_handle)

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramLocal(_ProgramConstants, _ProgramBase):
    """Local solves, matrix-free operators, laplacian/gradient/divergence and the coefficiented apply."""

    def solve_local_linear(self, name: Any = None, operator: Any = None, rhs: Any = None,
                           fields: Any = None) -> Any:
        """Solve a LOCAL linear system ``operator U = rhs`` cell by cell, where
        ``operator = self.I +/- a*L`` for a single model linear source ``L`` (``a`` may depend on dt
        / constants). Returns the solution State. A non-local or non-linear operator is rejected; the
        per-cell dense fallback bound (n_cons <= 8) is enforced by the codegen (a later phase)."""
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
        return self._new(
            "state", "solve_local_linear", inputs,
            {"linear_source": lname, "a_coeff": a}, name, rhs.block, space=rhs.space,
            field_context=field_context)

    # The LOCAL per-cell ops a solve_local_nonlinear residual sub-block may use: the iterate / guess
    # State placeholders, named per-cell sources / linear-source applies, and the affine combine of
    # them. All lower to a per-cell scalar expression in the cell-local conservative stack -- NO
    # non-local op (rhs / divergence / solve_fields / a nested solve) is allowed (it would need a halo
    # / global solve, which a per-cell Newton kernel cannot evaluate at a perturbed stack state).

    @atomic_authoring
    def solve_local_nonlinear(self, name: Any = None, residual: Any = None,
                              initial_guess: Any = None, method: Any = "newton",
                              tol: Any = 1e-12, max_iter: Any = 20, fd_eps: Any = None) -> Any:
        """Solve a LOCAL non-linear system ``residual(U) = 0`` cell by cell with a per-cell Newton
        iteration (spec op 10). Returns the converged solution State.

        @p residual is an IR-building callable ``residual_fn(P, U, U0) -> State``: given the Newton
        iterate State @p U and the frozen initial-guess State @p U0 it BUILDS the residual ``r(U)`` (a
        State value) from LOCAL per-cell ops only -- ``P.source`` (a named ``m.source_term``),
        ``P.apply`` (a named ``m.linear_source``), the iterate / initial-guess States, and the affine
        algebra over them (e.g. an implicit reaction ``r(U) = U - U0 - dt*S(U)``). A non-local op
        (``P.rhs`` / ``P.divergence`` / ``P.solve_fields`` / a nested solve) is rejected: the residual
        must be re-evaluable at a PERTURBED cell-local stack state, which a halo / global solve cannot.
        The sub-block (like a ``set_apply`` body) lowers to a device-inlinable per-cell residual the
        kernel re-evaluates at ``U`` and at the finite-difference perturbations ``U + eps*e_j``. A
        two-argument ``residual_fn(P, U)`` (ignoring the guess) is also accepted.

        @p initial_guess is the start State ``U0`` (typically ``U^n``); it seeds the Newton iterate and
        the residual reads it as a frozen per-cell constant. @p method is ``"newton"`` (the only
        method). @p tol is the convergence threshold on ``max_c |r_c|`` (per cell) and @p max_iter the
        iteration budget (the kernel runs a fixed C++ ``for`` bounded by @p max_iter, breaking early
        once ``|r| < tol``).

        @p fd_eps (ADC-617) is the RELATIVE finite-difference step of the in-kernel Jacobian columns:
        the perturbation is ``fd_eps * max(|U_j|, 1)``. ``None`` keeps the historical ``1e-7``. Because
        the value is EMITTED into the C++ kernel, it is stored on the IR node and so participates in
        the program hash / compile cache key -- two programs differing only in ``fd_eps`` never share a
        cached ``.so``. Must be a positive number when given.

        The Jacobian is formed in-kernel by finite differences (``J_ij = (r_i(U+eps e_j) - r_i(U))/eps``)
        and the Newton step ``J dU = -r`` is solved with the SAME stack-only dense inverse
        (``pops::detail::mat_inverse<N>``) `solve_local_linear` uses -- so the kernel is heap-free
        / allocation-free / dispatch-free (no ``std::function`` / Eigen / ``std::vector``). The dense
        fallback bound ``n_cons <= 8`` is enforced by the codegen (same as `solve_local_linear`)."""
        if not callable(residual):
            raise ValueError(
                "solve_local_nonlinear: residual must be an IR-building callable "
                "residual_fn(P, U, U0) returning the residual State r(U)")
        if not (isinstance(initial_guess, ProgramValue) and initial_guess.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: initial_guess must be a State value (initial_guess=...)")
        require_top_level(self, initial_guess, "solve_local_nonlinear")
        if method != "newton":
            raise NotImplementedError(
                "solve_local_nonlinear: only method='newton' is supported (got %r)" % (method,))
        tol_literal = positive_scalar_literal(tol, where="solve_local_nonlinear: tol")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(
                "solve_local_nonlinear: max_iter must be a positive int (got %r)" % (max_iter,))
        fd_eps_literal = (None if fd_eps is None else positive_scalar_literal(
            fd_eps, where="solve_local_nonlinear: fd_eps"))
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
                    "Use a non-local op (P.rhs / P.divergence / P.solve_fields) outside the residual."
                    % (w.op, sorted(self._RESIDUAL_LOCAL_OPS)))
        return self._new(
            "state", "solve_local_nonlinear", (initial_guess,),
            {"residual_block": sub, "residual_region": residual_region,
             "residual": r, "iterate": iterate, "guess": guess_ph,
             "tol": tol_literal, "max_iter": int(max_iter), "method": method,
             # ADC-617: the FD Jacobian relative step. None -> the historical 1e-7 literal. Stored on
             # the node so the generic attrs hash (_ir_hash) busts the compile cache when it changes.
             "fd_eps": fd_eps_literal}, name, block,
            space=initial_guess.space)

    def _linear_source_name(self, operator: Any, where: Any) -> Any:
        """Resolve `operator` to the linear-source name.

        Accepts a typed :class:`pops.model.OperatorHandle` (resolved against the exact bound registry),
        a validated `linear_source` ProgramValue, a single unit-coefficient ``_Operator`` term, or a
        bare name string on this private internal seam."""
        from pops.model import OperatorHandle
        if isinstance(operator, OperatorHandle):
            return resolve_operator_handle(
                self, operator, where=where,
                expected_kinds="local_linear_operator").name
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

    def _linear_source(self, name: Any) -> Any:
        """Internal seam: reference a linear source by its bare NAME (an internal selector).

        NOT a public surface -- it is the byte-identical lowering the public typed
        :meth:`linear_source` delegates to (after unwrapping its handle), and the path the internal
        lowering (``_lower_call``) and the ``pops.lib.time`` macros use directly with a bare name."""
        if not isinstance(name, str) or not name:
            raise ValueError("_linear_source: a non-empty operator name is required")
        return self._new("operator", "linear_source", (), {"linear_source": name}, name, None)

    def _apply(self, operator: Any = None, state: Any = None, fields: Any = None,
               name: Any = None) -> Any:
        """Internal seam: apply a linear source given as a typed value / handle OR a bare name.

        NOT a public surface -- the public :meth:`apply` refuses a bare-name string and delegates
        here; the solver-DSL and other internal callers pass the name selector directly."""
        state, fields = _resolve_handle(state), _resolve_handle(fields)
        lname = self._linear_source_name(operator, "apply")
        if not (isinstance(state, ProgramValue) and state.vtype == "state"):
            raise ValueError("apply: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, ProgramValue) and fields.vtype == "fields"):
            raise ValueError("apply: fields must be a FieldContext from solve_fields")
        field_context = None
        if fields is not None:
            from pops.time.field_context import require_field_read
            field_context = require_field_read(fields, state, "apply")
        self._check_operator_state(operator, state, "apply")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new(
            "rhs", "apply", inputs, {"linear_source": lname},
            name or ("apply_" + lname), state.block, space=rate_space_for(state.space),
            field_context=field_context)

    # --- matrix-free operators / dynamic linear solve (ADC-405 Phase 6b) ----------------------------
    # A ``matrix_free_op`` names a GLOBAL matrix-free operator A : scalar_field -> scalar_field whose
    # apply ``out <- A(in)`` is an IR sub-block recorded by ``set_apply``. ``solve_linear`` lowers to a
    # call into the runtime's Krylov loop (pops::cg_solve / bicgstab_solve / richardson_solve /
    # gmres_solve): the iteration is DYNAMIC and lives C++-side (inside the loop), invisible to the IR --
    # the Program only supplies the apply (a C++ lambda) + the rhs / tolerance / iteration budget.

    def scalar_field(self, name: Any = None, ncomp: Any = 1) -> Any:
        """A fresh, zero-initialized scalar field: scratch the apply sub-block uses (e.g. the Laplacian
        output, or a 2-component gradient buffer). @p ncomp is the component count (1 by default; 2 for a
        gradient field consumed by ``P.divergence``). Lowered to ``ctx.alloc_scalar_field(ncomp, 1)``."""
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
            raise ValueError("scalar_field: ncomp must be a positive integer (got %r)" % (ncomp,))
        return self._new("scalar_field", "scalar_field", (), {"ncomp": int(ncomp)}, name, None)


    def matrix_free_operator(self, name: Any, domain: Any = "scalar", range_: Any = "scalar",
                             ncomp: Any = None) -> Any:
        """Declare a matrix-free operator ``A : domain -> range_``. @p domain / @p range_ are the field
        kind on each side and MUST match (a square operator: the Krylov iterate, residual and solution
        share one layout): ``"scalar"`` (a 1-component scalar field, the default), or ``"vector"`` /
        ``"state"`` (a multi-component field, e.g. the condensed-Schur block unknown). For a
        ``vector`` / ``state`` operator @p ncomp (an int >= 1) is REQUIRED -- the component count of the
        apply's in/out buffers and of the solution; for a ``scalar`` operator @p ncomp must be omitted
        (or 1). Supply the apply via ``P.set_apply(A, body_fn)`` before using it in ``P.solve_linear``."""
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
        return self._new("matrix_free_op", "matrix_free_operator", (),
                         {"domain": domain, "range": range_, "ncomp": int(ncomp), "apply_block": None,
                          "apply_result": None, "apply_in": None, "apply_out": None}, name, None)

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
            out_sf = self._new("scalar_field", "apply_out", (), {"ncomp": op_ncomp}, "apply_out", None)
            in_sf = self._new("scalar_field", "apply_in", (), {"ncomp": op_ncomp}, "apply_in", None)
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
        attrs.update({
            "apply_block": block,
            "apply_region": apply_region,
            "apply_result": result,
            "apply_in": in_sf,
            "apply_out": out_sf,
        })
        return self._replace_value(operator, attrs=attrs)

    def laplacian(self, out: Any, in_: Any) -> Any:
        """Record ``out = Lap(in_)`` (the shared discrete 5-point Laplacian). @p out and @p in_ are
        scalar_field values. Lowered to ``ctx.laplacian(out, in_)``. Used inside an apply sub-block to
        form a Helmholtz operator ``A(in) = in - alpha*Lap(in)`` via the affine algebra."""
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("laplacian: out must be a scalar_field value")
        if not (isinstance(in_, ProgramValue) and in_.vtype == "scalar_field"):
            raise ValueError("laplacian: in must be a scalar_field value")
        return self._new("scalar_field", "laplacian", (out, in_), {}, out.name, None)

    def gradient(self, out: Any, phi: Any) -> Any:
        """Record ``out = grad(phi)`` (centered differences; @p out has >= 2 components). @p out and
        @p phi are scalar_field values. Lowered to ``ctx.gradient(out, phi)``."""
        if not (isinstance(out, ProgramValue) and out.vtype == "scalar_field"):
            raise ValueError("gradient: out must be a scalar_field value")
        if not (isinstance(phi, ProgramValue) and phi.vtype == "scalar_field"):
            raise ValueError("gradient: phi must be a scalar_field value")
        return self._new("scalar_field", "gradient", (out, phi), {}, out.name, None)

    def divergence(self, out: Any, fx: Any, fy: Any) -> Any:
        """Record ``out = div(fx, fy)`` (centered FV divergence d fx/dx + d fy/dy, component 0). @p out,
        @p fx and @p fy are scalar_field values. Lowered to ``ctx.divergence(out, fx, fy)``. The exact
        inverse of @ref gradient: chaining ``P.gradient(g, phi); P.divergence(d, gx, gy)`` recovers the
        5-point Laplacian, so a matrix-free apply ``phi - alpha*div(grad phi)`` is the Schur-like flux
        operator ``phi - alpha*Lap(phi)``."""
        for nm, val in (("out", out), ("fx", fx), ("fy", fy)):
            if not (isinstance(val, ProgramValue) and val.vtype == "scalar_field"):
                raise ValueError("divergence: %s must be a scalar_field value" % nm)
        return self._new("scalar_field", "divergence", (out, fx, fy), {}, out.name, None)

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
        if not (isinstance(r0, ProgramValue) and r0.is_field()):
            raise ValueError("rhs_jacvec: r0 must be the precomputed rhs(U^k) State/RHS value (r0=...)")
        if not isinstance(field_coupled, bool):
            raise TypeError("rhs_jacvec: field_coupled must be a bool")
        context = getattr(r0, "field_context", None)
        context_matches = (context is not None
                           and context.matches(None, iterate.block, iterate.id))
        if field_coupled and not context_matches:
            raise ValueError(
                "rhs_jacvec: field_coupled=True requires r0 computed with fields solved from "
                "the exact Newton iterate")
        if not field_coupled and context is not None:
            raise ValueError(
                "rhs_jacvec: field_coupled=False requires an r0 with no field-solve provenance")
        if flux is not True:
            raise NotImplementedError(
                "rhs_jacvec cannot linearize flux=False: the matrix-free kernel currently requires "
                "the default flux divergence")
        src = list(sources) if sources is not None else None
        named_sources = [source for source in (src or ()) if source != "default"]
        if named_sources:
            raise NotImplementedError(
                "rhs_jacvec cannot linearize named sources %r yet; use sources=[] (flux-only) or "
                "sources=['default']" % named_sources)
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
                          "field_coupled": field_coupled},
                         out.name, None)

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
        # The GENERIC condensed bundle (P.condensed_coeffs, ADC-637) carries the four tensor-coefficient
        # fields (eps_x, eps_y, a_xy, a_yx) the coefficiented apply consumes.
        if not (isinstance(coeffs, ProgramValue) and coeffs.vtype == "condensed_coeffs"):
            raise ValueError("apply_laplacian_coeff: coeffs must be a coefficient bundle "
                             "(P.condensed_coeffs(...))")
        return self._new("scalar_field", "apply_laplacian_coeff", (out, in_, coeffs), {}, out.name,
                         None)
