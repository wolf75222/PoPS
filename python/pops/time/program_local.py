"""pops.time Program authoring mixin -- local + matrix-free ops.

Local solves, matrix-free operators, laplacian/gradient/divergence and the Schur helpers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.values import (
    Value, _Affine, _Coeff, _Operator, _is_field_value, _residual_wants_guess, _resolve_handle)

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramLocal(_ProgramConstants, _ProgramBase):
    """Local solves, matrix-free operators, laplacian/gradient/divergence and the Schur helpers."""

    def solve_local_linear(self, name: Any = None, operator: Any = None, rhs: Any = None,
                           fields: Any = None) -> Any:
        """Solve a LOCAL linear system ``operator U = rhs`` cell by cell, where
        ``operator = self.I +/- a*L`` for a single model linear source ``L`` (``a`` may depend on dt
        / constants). Returns the solution State. A non-local or non-linear operator is rejected; the
        per-cell dense fallback bound (n_cons <= 8) is enforced by the codegen (a later phase)."""
        if not isinstance(operator, _Operator) or operator.identity.as_dict() != {0: 1.0}:
            raise ValueError("solve_local_linear currently supports local linear operators only")
        if len(operator.terms) != 1:
            raise NotImplementedError(
                "solve_local_linear currently supports a single linear source (I +/- a*L); got %d "
                "term(s)" % len(operator.terms))
        if not (isinstance(rhs, Value) and rhs.vtype == "state"):
            raise ValueError("solve_local_linear: rhs must be a State value (rhs=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("solve_local_linear: fields must be a FieldContext from solve_fields")
        op_value, l_coeff = operator.terms[0]
        self._check_operator_state(op_value, rhs, "solve_local_linear")
        lname = op_value.attrs["linear_source"]
        a = (-l_coeff).as_dict()  # operator = I - a*L, so the L term carries the coefficient -a
        inputs = (rhs, op_value, fields) if fields is not None else (rhs, op_value)
        out = self._new("state", "solve_local_linear", inputs,
                        {"linear_source": lname, "a_coeff": a}, name, rhs.block)
        out.space = rhs.space  # the solution is a State over the same space as the rhs
        return out

    # The LOCAL per-cell ops a solve_local_nonlinear residual sub-block may use: the iterate / guess
    # State placeholders, named per-cell sources / linear-source applies, and the affine combine of
    # them. All lower to a per-cell scalar expression in the cell-local conservative stack -- NO
    # non-local op (rhs / divergence / solve_fields / a nested solve) is allowed (it would need a halo
    # / global solve, which a per-cell Newton kernel cannot evaluate at a perturbed stack state).

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
        if not (isinstance(initial_guess, Value) and initial_guess.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: initial_guess must be a State value (initial_guess=...)")
        if method != "newton":
            raise NotImplementedError(
                "solve_local_nonlinear: only method='newton' is supported (got %r)" % (method,))
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError("solve_local_nonlinear: tol must be a positive number (got %r)" % (tol,))
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError(
                "solve_local_nonlinear: max_iter must be a positive int (got %r)" % (max_iter,))
        if fd_eps is not None and (isinstance(fd_eps, bool) or not isinstance(fd_eps, (int, float))
                                   or fd_eps <= 0):
            raise ValueError(
                "solve_local_nonlinear: fd_eps must be a positive number or None (got %r)" % (fd_eps,))
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
            iterate = self._new("state", "state", (), {}, "newton_iterate", block)
            guess_ph = self._new("state", "state", (), {}, "newton_guess", block)
            # residual_fn(P, U, U0); a two-arg residual_fn(P, U) (ignoring the guess) is also accepted.
            r = residual(self, iterate, guess_ph) if wants_guess else residual(self, iterate)
        finally:
            self._recording.pop()
        if not (isinstance(r, Value) and r.vtype == "state"):
            raise ValueError(
                "solve_local_nonlinear: residual_fn must return the residual State r(U) (got %r)" % (r,))
        for w in sub:
            if w.op not in self._RESIDUAL_LOCAL_OPS:
                raise ValueError(
                    "solve_local_nonlinear: residual op '%s' is not LOCAL; a per-cell Newton residual "
                    "may use only %s (the iterate / guess State, P.source, P.apply, affine combines). "
                    "Use a non-local op (P.rhs / P.divergence / P.solve_fields) outside the residual."
                    % (w.op, sorted(self._RESIDUAL_LOCAL_OPS)))
        return self._new(
            "state", "solve_local_nonlinear", (initial_guess,),
            {"residual_block": sub, "residual": r, "iterate": iterate, "guess": guess_ph,
             "tol": float(tol), "max_iter": int(max_iter), "method": method,
             # ADC-617: the FD Jacobian relative step. None -> the historical 1e-7 literal. Stored on
             # the node so the generic attrs hash (_ir_hash) busts the compile cache when it changes.
             "fd_eps": (None if fd_eps is None else float(fd_eps))}, name, block)

    def _linear_source_name(self, operator: Any, where: Any) -> Any:
        """Resolve `operator` to the linear-source name.

        Accepts a typed :class:`pops.model.OperatorHandle` (ADC-532; unwrapped to its ``.name``, so the
        IR is byte-identical to the historical string form), a `linear_source` Value, a single
        unit-coefficient ``_Operator`` term, or a bare name string (an internal selector)."""
        from pops.model import OperatorHandle
        if isinstance(operator, OperatorHandle):
            return operator.name
        if isinstance(operator, str) and operator:
            return operator
        if isinstance(operator, Value) and operator.op == "linear_source":
            return operator.attrs["linear_source"]
        if (isinstance(operator, _Operator) and not operator.identity.as_dict()
                and len(operator.terms) == 1 and operator.terms[0][1].as_dict() == {0: 1.0}):
            return operator.terms[0][0].attrs["linear_source"]
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
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("apply: a State value is required (state=...)")
        if fields is not None and not (isinstance(fields, Value) and fields.vtype == "fields"):
            raise ValueError("apply: fields must be a FieldContext from solve_fields")
        self._check_operator_state(operator, state, "apply")
        inputs = (state, fields) if fields is not None else (state,)
        return self._new("rhs", "apply", inputs, {"linear_source": lname},
                         name or ("apply_" + lname), state.block)

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
        if not isinstance(ncomp, int) or ncomp < 1:
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

    def set_apply(self, operator: Any, body_fn: Any) -> Any:
        """Record the apply ``out <- A(in)`` of a ``matrix_free_operator``. @p body_fn(P, out, in) is an
        IR-building callable: @p in and @p out are scalar_field values (the operator's argument and
        result); the body builds @p out from @p in (e.g. ``P.laplacian(tmp, in); ...``) using
        ``P.laplacian`` + the affine algebra and RETURNS the result scalar_field (the value written into
        @p out). The ops are captured into a separate sub-block (like a while body) and re-emitted as a
        C++ lambda the Krylov loop calls."""
        if not (isinstance(operator, Value) and operator.vtype == "matrix_free_op"):
            raise ValueError("set_apply: operator must be a matrix_free_operator value")
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
        if not (isinstance(result, (Value, _Affine)) or _is_field_value(result)):
            raise ValueError("set_apply: body_fn must return the result scalar_field (out <- A(in))")
        operator.attrs["apply_block"] = block
        operator.attrs["apply_result"] = result
        operator.attrs["apply_in"] = in_sf
        operator.attrs["apply_out"] = out_sf
        return operator

    def laplacian(self, out: Any, in_: Any) -> Any:
        """Record ``out = Lap(in_)`` (the shared discrete 5-point Laplacian). @p out and @p in_ are
        scalar_field values. Lowered to ``ctx.laplacian(out, in_)``. Used inside an apply sub-block to
        form a Helmholtz operator ``A(in) = in - alpha*Lap(in)`` via the affine algebra."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("laplacian: out must be a scalar_field value")
        if not (isinstance(in_, Value) and in_.vtype == "scalar_field"):
            raise ValueError("laplacian: in must be a scalar_field value")
        return self._new("scalar_field", "laplacian", (out, in_), {}, out.name, None)

    def gradient(self, out: Any, phi: Any) -> Any:
        """Record ``out = grad(phi)`` (centered differences; @p out has >= 2 components). @p out and
        @p phi are scalar_field values. Lowered to ``ctx.gradient(out, phi)``."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("gradient: out must be a scalar_field value")
        if not (isinstance(phi, Value) and phi.vtype == "scalar_field"):
            raise ValueError("gradient: phi must be a scalar_field value")
        return self._new("scalar_field", "gradient", (out, phi), {}, out.name, None)

    def divergence(self, out: Any, fx: Any, fy: Any) -> Any:
        """Record ``out = div(fx, fy)`` (centered FV divergence d fx/dx + d fy/dy, component 0). @p out,
        @p fx and @p fy are scalar_field values. Lowered to ``ctx.divergence(out, fx, fy)``. The exact
        inverse of @ref gradient: chaining ``P.gradient(g, phi); P.divergence(d, gx, gy)`` recovers the
        5-point Laplacian, so a matrix-free apply ``phi - alpha*div(grad phi)`` is the Schur-like flux
        operator ``phi - alpha*Lap(phi)``."""
        for nm, val in (("out", out), ("fx", fx), ("fy", fy)):
            if not (isinstance(val, Value) and val.vtype == "scalar_field"):
                raise ValueError("divergence: %s must be a scalar_field value" % nm)
        return self._new("scalar_field", "divergence", (out, fx, fy), {}, out.name, None)

    # --- finite-difference Jacobian-vector product (ADC-431: implicit-flux BDF Newton-Krylov) --------
    def rhs_jacvec(self, out: Any, in_: Any, *, iterate: Any, r0: Any, c_dt: Any, eps: Any = 1e-7,
                   flux: Any = True, sources: Any = ("default",)) -> Any:
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
        relative FD step (scaled by ``||U^k|| / ||in||`` inside the kernel). @p flux / @p sources select
        the same residual the outer ``rhs`` uses (so the linearized operator is consistent with the
        residual). The op may ONLY appear inside ``set_apply`` (it captures the apply's in/out buffers).

        Unlike the cell-local FD Jacobian of `solve_local_nonlinear` (a per-cell dense inverse), this is a
        GLOBAL operator: ``rhs`` couples the cells through the flux stencil, so the matvec is dense over
        the coupled stencil and the Newton step ``J dU = -F`` is solved by `solve_linear` (GMRES)."""
        if not self._recording:
            raise ValueError("rhs_jacvec may only be recorded inside a matrix_free_operator apply "
                             "(call it from the set_apply body_fn)")
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("rhs_jacvec: out must be the apply sub-block's out scalar_field value")
        if not (isinstance(in_, Value) and in_.vtype == "scalar_field"):
            raise ValueError("rhs_jacvec: in_ must be the apply sub-block's in scalar_field value")
        if not (isinstance(iterate, Value) and iterate.vtype == "state"):
            raise ValueError("rhs_jacvec: iterate must be the frozen Newton-iterate State (iterate=...)")
        if not (isinstance(r0, Value) and r0.is_field()):
            raise ValueError("rhs_jacvec: r0 must be the precomputed rhs(U^k) State/RHS value (r0=...)")
        if not isinstance(c_dt, (int, float, _Coeff)):
            raise ValueError("rhs_jacvec: c_dt must be a number or a dt-polynomial (got %r)" % (c_dt,))
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("rhs_jacvec: eps must be a positive number (got %r)" % (eps,))
        c_d = (c_dt if isinstance(c_dt, _Coeff) else _Coeff({0: float(c_dt)})).as_dict()
        src = list(sources) if sources is not None else None
        return self._new("scalar_field", "rhs_jacvec", (out, in_, iterate, r0),
                         {"c_dt": c_d, "eps": float(eps), "flux": bool(flux), "sources": src},
                         out.name, None)

    # --- anisotropic condensed-Schur coefficient assembly + coefficiented apply (ADC-399 / ADC-421) ---
    def schur_coeffs(self, name: Any = None, state: Any = None, c: Any = None, th_dt: Any = None,
                     c_rho: Any = 0, c_bz: Any = 3) -> Any:
        """Assemble the per-cell tensor coefficient ``A = I + c*rho*B^{-1}`` of the condensed-Schur
        operator from a State (rho at component @p c_rho) and the B_z aux field (component @p c_bz,
        canonical B_z=3). Returns a ``schur_coeffs`` bundle value carrying the four coefficient fields
        (eps_x, eps_y, a_xy, a_yx) -- pass it to ``P.apply_laplacian_coeff`` inside a matrix-free apply.

        @p c = theta^2 * dt^2 * alpha and @p th_dt = theta*dt are scalar coefficients (numbers or
        dt-polynomials via the affine ``P.dt`` algebra; ``B^{-1}`` depends only on ``w = th_dt*B_z``).
        The assembly runs ONCE per step (rho / B_z frozen in the source) and the bundle is reused across
        every Krylov iteration of the phi solve. Lowered to ``pops::coupling::schur::program::assemble_schur_coeffs`` -- the SAME
        native detail::SchurOperatorCoeffKernel + apply_laplacian coefficient path, no reimplementation.
        """
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_coeffs: a State value is required (state=...)")
        for nm, sc in (("c", c), ("th_dt", th_dt)):
            if not isinstance(sc, (int, float, _Coeff)):
                raise ValueError("schur_coeffs: %s must be a number or a dt-polynomial (got %r)"
                                 % (nm, sc))
        for nm, ci in (("c_rho", c_rho), ("c_bz", c_bz)):
            if isinstance(ci, bool) or not isinstance(ci, int) or ci < 0:
                raise ValueError("schur_coeffs: %s must be a Python int >= 0 (got %r)" % (nm, ci))
        c_d = (c if isinstance(c, _Coeff) else _Coeff({0: float(c)})).as_dict()
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        return self._new("schur_coeffs", "schur_coeffs", (state,),
                         {"c": c_d, "th_dt": th_d, "c_rho": int(c_rho), "c_bz": int(c_bz)}, name,
                         state.block)

    def apply_laplacian_coeff(self, out: Any, in_: Any, coeffs: Any) -> Any:
        """Record ``out = div(A grad in_)`` with the tensor ``A`` of a @ref schur_coeffs bundle (the
        coefficiented matrix-free matvec of the condensed-Schur operator, ``pops::apply_laplacian``'s
        coefficient path). @p out and @p in_ are scalar_field values; @p coeffs is a ``schur_coeffs``
        value. Used inside a matrix-free apply: the condensed operator ``L_schur(phi) = -div(A grad
        phi) = -out``, so build it as ``-1 * P.apply_laplacian_coeff(out, in_, A)`` via the affine
        algebra. Lowered to ``pops::coupling::schur::program::apply_laplacian_coeff(ctx, out, in_, eps_x, eps_y, a_xy, a_yx)``."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("apply_laplacian_coeff: out must be a scalar_field value")
        if not (isinstance(in_, Value) and in_.vtype == "scalar_field"):
            raise ValueError("apply_laplacian_coeff: in_ must be a scalar_field value")
        if not (isinstance(coeffs, Value) and coeffs.vtype == "schur_coeffs"):
            raise ValueError("apply_laplacian_coeff: coeffs must be a schur_coeffs bundle "
                             "(P.schur_coeffs(...))")
        return self._new("scalar_field", "apply_laplacian_coeff", (out, in_, coeffs), {}, out.name,
                         None)

    def schur_explicit_flux(self, out: Any, state: Any, th_dt: Any, c_mx: Any = 1, c_my: Any = 2,
                            c_bz: Any = 3) -> Any:
        """Record ``out = B^{-1} (mx, my)`` per cell -- the explicit condensed-Schur flux
        ``F = rho*B^{-1}*v^n`` (Fx in component 0, Fy in component 1). @p out is a scalar_field (>= 2
        components), @p state a State (mx / my at @p c_mx / @p c_my), B_z the aux field at @p c_bz.
        @p th_dt = theta*dt. Chain ``P.divergence(d, out, out)`` for the centered divergence of F.
        Lowered to ``pops::coupling::schur::program::schur_explicit_flux`` (native detail::SchurExplicitFluxKernel)."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("schur_explicit_flux: out must be a scalar_field value (ncomp >= 2)")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_explicit_flux: a State value is required")
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        return self._new("scalar_field", "schur_explicit_flux", (out, state),
                         {"th_dt": th_d, "c_mx": int(c_mx), "c_my": int(c_my), "c_bz": int(c_bz)},
                         out.name, None)

    def schur_rhs(self, out: Any, phi_n: Any, state: Any, th_dt: Any, g: Any, c_mx: Any = 1,
                  c_my: Any = 2, c_bz: Any = 3) -> Any:
        """Record the FUSED condensed-Schur right-hand side ``out = -Lap(phi_n) - g*div(F)`` with
        ``F = B^{-1}(mx, my)`` -- the native ElectrostaticLorentzCondensation::assemble_rhs in one op.
        @p out is a 1-component scalar_field, @p phi_n a scalar_field (phi^n warm start; its ghosts are
        filled for the Laplacian), @p state a State (mx / my at @p c_mx / @p c_my). @p th_dt = theta*dt
        and @p g = theta*dt*alpha are scalar coefficients (numbers or dt-polynomials). Lowered to
        ``pops::coupling::schur::program::assemble_schur_rhs``. A single op because there is no scalar-field affine combine at the
        IR level -- the fused C++ assembler mirrors the native one (bare Lap + explicit flux + the
        SchurRhsAssemble divergence)."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("schur_rhs: out must be a scalar_field value")
        if not (isinstance(phi_n, Value) and phi_n.vtype == "scalar_field"):
            raise ValueError("schur_rhs: phi_n must be a scalar_field value")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_rhs: a State value is required (state=...)")
        for nm, sc in (("th_dt", th_dt), ("g", g)):
            if not isinstance(sc, (int, float, _Coeff)):
                raise ValueError("schur_rhs: %s must be a number or a dt-polynomial (got %r)"
                                 % (nm, sc))
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        g_d = (g if isinstance(g, _Coeff) else _Coeff({0: float(g)})).as_dict()
        return self._new("scalar_field", "schur_rhs", (out, phi_n, state),
                         {"th_dt": th_d, "g": g_d, "c_mx": int(c_mx), "c_my": int(c_my),
                          "c_bz": int(c_bz)}, out.name, None)

    def schur_reconstruct(self, name: Any = None, state: Any = None, phi: Any = None,
                          th_dt: Any = None, c_rho: Any = 0, c_mx: Any = 1, c_my: Any = 2,
                          c_bz: Any = 3) -> Any:
        """Record the condensed-Schur velocity reconstruction ``v^{n+theta} = B^{-1}(v^n - theta*dt*
        grad phi)`` IN PLACE on @p state (rho frozen; mom = rho*v written back). @p phi is the solved
        potential (a scalar_field or 1-component State), @p th_dt = theta*dt; B_z the aux at @p c_bz.
        Returns the updated State. Lowered to ``pops::coupling::schur::program::schur_reconstruct`` (the native centered gradient +
        closed B^{-1}). The final n+1 extrapolation (factor 1/theta) is the caller's affine algebra."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_reconstruct: a State value is required (state=...)")
        if not _is_field_value(phi):
            raise ValueError("schur_reconstruct: phi must be a scalar_field or State value (phi=...)")
        if not isinstance(th_dt, (int, float, _Coeff)):
            raise ValueError("schur_reconstruct: th_dt must be a number or a dt-polynomial (got %r)"
                             % (th_dt,))
        th_d = (th_dt if isinstance(th_dt, _Coeff) else _Coeff({0: float(th_dt)})).as_dict()
        return self._new("state", "schur_reconstruct", (state, phi),
                         {"th_dt": th_d, "c_rho": int(c_rho), "c_mx": int(c_mx), "c_my": int(c_my),
                          "c_bz": int(c_bz)}, name, state.block)

    def schur_energy(self, name: Any = None, state: Any = None, state_old: Any = None, c_rho: Any = 0,
                     c_mx: Any = 1, c_my: Any = 2, c_E: Any = 3) -> Any:
        """Record the condensed-Schur kinetic-energy increment IN PLACE on @p state (ADC-427):
        ``E^{n+1} = E^n + (1/2)*rho*(|v^{n+1}|^2 - |v^n|^2)``, ``v = (mx, my)/rho`` (the native
        SchurEnergyKernel). @p state carries ``rho`` / ``mx`` / ``my`` / ``E`` at @p c_rho / @p c_mx /
        @p c_my / @p c_E AFTER the velocity update (mom = rho*v^{n+1}); @p state_old is U^n (read for
        v^n = mom^n/rho^n and the base energy E^n). rho is frozen, so the same rho is read from both.
        Returns @p state (E overwritten in place). Lowered to ``pops::coupling::schur::program::schur_energy``."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("schur_energy: a State value is required (state=...)")
        if not (isinstance(state_old, Value) and state_old.vtype == "state"):
            raise ValueError("schur_energy: a State value is required (state_old=U^n)")
        if state_old.block != state.block:
            raise ValueError("schur_energy: state and state_old must belong to the same block")
        for nm, ci in (("c_rho", c_rho), ("c_mx", c_mx), ("c_my", c_my), ("c_E", c_E)):
            if isinstance(ci, bool) or not isinstance(ci, int) or ci < 0:
                raise ValueError("schur_energy: %s must be a Python int >= 0 (got %r)" % (nm, ci))
        return self._new("state", "schur_energy", (state, state_old),
                         {"c_rho": int(c_rho), "c_mx": int(c_mx), "c_my": int(c_my), "c_E": int(c_E)},
                         name, state.block)

