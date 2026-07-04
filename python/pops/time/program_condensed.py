"""pops.time Program authoring mixin -- the GENERIC condensed-implicit-solve ops (ADC-637).

The condensed-implicit pattern eliminates a per-cell block-linear source response ``M = I - theta*dt*J``
(the linearization J authored via ``m.local_linear_operator`` on a coupled momentum subset) against a
gradient-linear elliptic coupling, yielding the tensor coefficient ``A = I + c*rho*M^{-1}``, a fused RHS
and a velocity reconstruction. These three IR-builder methods author those stages generically: each
carries the operator NAME + the coupled ``subset`` + the scalar coefficients, and the codegen
(program_emit_condensed) lowers them to inline block_inverse<N> kernels -- parallel to, not replacing,
the P.schur_* ops.

Compile-time refusals (design section 5), fail-loud at build, never a silent partial:
  - N > 8: the dense block-inverse bound (mat_inverse cap);
  - the coefficient assembly requires a 2D momentum subset (the eps_x/eps_y/a_xy/a_yx tensor);
  - subset must be distinct non-negative component indices;
  - the coefficients must be numbers or dt-polynomials, c_rho a non-negative int.
The block-local-linearization contract (J must not depend on U|_K) is enforced UPSTREAM at
``m.local_linear_operator`` / ``m.linear_source`` registration (a cons/prim-dependent coefficient raises
there with an actionable message); the condensed ops reference the already-validated operator by name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_base import _ProgramConstants
from pops.time.values import Value, _Coeff, _is_field_value

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object

# The dense block-inverse bound (pops::detail::mat_inverse<N> stack cap): a coupled subset larger than
# this cannot be eliminated by the closed form or the generic fallback.
_MAX_BLOCK = 8


class _ProgramCondensed(_ProgramConstants, _ProgramBase):
    """The generic condensed-implicit-solve authoring ops (ADC-637): condensed_coeffs / condensed_rhs /
    condensed_reconstruct, carrying an authored linear operator + a coupled momentum subset."""

    def _condensed_operator_name(self, operator: Any) -> str:
        """Resolve @p operator to the authored linear-operator NAME: a typed OperatorHandle (unwrapped
        to ``.name``, so the IR is byte-identical to the string form) or a non-empty name string. The
        operator's block-local-linearization contract was validated at its m.local_linear_operator
        registration; here we only bind its name into the IR."""
        from pops.model import OperatorHandle
        if isinstance(operator, OperatorHandle):
            return operator.name
        if isinstance(operator, str) and operator:
            return operator
        raise ValueError(
            "condensed op: linear_operator must be an authored local linear operator "
            "(m.local_linear_operator(...) or its OperatorHandle / name)")

    def _condensed_subset(self, subset: Any, where: Any, *, require_2d: Any = False) -> tuple:
        """Validate + normalize the coupled component @p subset (the momentum block the solve
        eliminates): a tuple of DISTINCT non-negative ints, size 2..8 (the dense block-inverse bound).
        @p require_2d demands exactly a 2D subset (the tensor elliptic coefficient path)."""
        if not isinstance(subset, (tuple, list)) or not subset:
            raise ValueError("%s: subset must be a non-empty tuple of component indices" % where)
        sub = tuple(subset)
        for c in sub:
            if isinstance(c, bool) or not isinstance(c, int) or c < 0:
                raise ValueError("%s: subset components must be non-negative ints (got %r)"
                                 % (where, c))
        if len(set(sub)) != len(sub):
            raise ValueError("%s: subset components must be distinct (got %r)" % (where, sub))
        if len(sub) > _MAX_BLOCK:
            raise ValueError(
                "%s: subset size N=%d exceeds the dense block-inverse bound (%d); a larger coupled "
                "block cannot be eliminated in closed form" % (where, len(sub), _MAX_BLOCK))
        if require_2d and len(sub) != 2:
            raise NotImplementedError(
                "%s: the tensor elliptic coefficient path supports a 2D momentum subset "
                "(eps_x/eps_y/a_xy/a_yx); got subset size %d. A 3D coupled block needs the "
                "N-component elliptic tensor (a later phase)." % (where, len(sub)))
        return sub

    @staticmethod
    def _coeff_dict(sc: Any, name: Any, where: Any) -> dict:
        """A scalar coefficient (number or dt-polynomial) as the IR power->float dict."""
        if not isinstance(sc, (int, float, _Coeff)):
            raise ValueError("%s: %s must be a number or a dt-polynomial (got %r)" % (where, name, sc))
        return (sc if isinstance(sc, _Coeff) else _Coeff({0: float(sc)})).as_dict()

    @staticmethod
    def _comp_index(ci: Any, name: Any, where: Any) -> int:
        if isinstance(ci, bool) or not isinstance(ci, int) or ci < 0:
            raise ValueError("%s: %s must be a Python int >= 0 (got %r)" % (where, name, ci))
        return int(ci)

    def condensed_coeffs(self, name: Any = None, state: Any = None, linear_operator: Any = None,
                         subset: Any = None, c: Any = None, th_dt: Any = None, c_rho: Any = 0) -> Any:
        """Assemble the per-cell tensor coefficient ``A = I + c*rho*M^{-1}`` of the condensed operator
        from an authored linear operator J (``M = I - th_dt*J``) on the coupled 2D momentum @p subset and
        a State (rho at @p c_rho). Returns a ``condensed_coeffs`` bundle carrying the four coefficient
        fields (eps_x, eps_y, a_xy, a_yx) -- pass it to ``P.apply_laplacian_coeff`` inside a matrix-free
        apply. Generic counterpart of ``P.schur_coeffs``: the codegen inverts M with
        ``pops::detail::block_inverse<2>`` inline (bit-identical to the Schur brick for the Lorentz J).

        @p c = theta^2*dt^2*alpha and @p th_dt = theta*dt are scalars (numbers or dt-polynomials). rho
        (a conservative var) enters only the outer c*rho factor, never M (R2)."""
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("condensed_coeffs: a State value is required (state=...)")
        opname = self._condensed_operator_name(linear_operator)
        sub = self._condensed_subset(subset, "condensed_coeffs", require_2d=True)
        c_d = self._coeff_dict(c, "c", "condensed_coeffs")
        th_d = self._coeff_dict(th_dt, "th_dt", "condensed_coeffs")
        return self._new("condensed_coeffs", "condensed_coeffs", (state,),
                         {"linear_operator": opname, "subset": sub, "c": c_d, "th_dt": th_d,
                          "c_rho": self._comp_index(c_rho, "c_rho", "condensed_coeffs")}, name,
                         state.block)

    def condensed_rhs(self, out: Any = None, phi_n: Any = None, state: Any = None,
                      linear_operator: Any = None, subset: Any = None, th_dt: Any = None,
                      g: Any = None) -> Any:
        """Record the fused RHS ``out = -Lap(phi_n) - g*div(M^{-1}(mx, my))`` (F = M^{-1} applied to the
        momentum @p subset) -- the generic counterpart of ``P.schur_rhs``. @p out is a 1-component
        scalar_field, @p phi_n the warm-start potential (its ghosts are filled for the Laplacian), @p
        state a State. @p th_dt = theta*dt, @p g = theta*dt*alpha (numbers or dt-polynomials). The
        codegen fuses the bare -Lap with the centered divergence of the block-inverse flux inline."""
        if not (isinstance(out, Value) and out.vtype == "scalar_field"):
            raise ValueError("condensed_rhs: out must be a scalar_field value")
        if not (isinstance(phi_n, Value) and phi_n.vtype == "scalar_field"):
            raise ValueError("condensed_rhs: phi_n must be a scalar_field value")
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("condensed_rhs: a State value is required (state=...)")
        opname = self._condensed_operator_name(linear_operator)
        sub = self._condensed_subset(subset, "condensed_rhs")
        th_d = self._coeff_dict(th_dt, "th_dt", "condensed_rhs")
        g_d = self._coeff_dict(g, "g", "condensed_rhs")
        return self._new("scalar_field", "condensed_rhs", (out, phi_n, state),
                         {"linear_operator": opname, "subset": sub, "th_dt": th_d, "g": g_d},
                         out.name, None)

    def condensed_reconstruct(self, name: Any = None, state: Any = None, phi: Any = None,
                              linear_operator: Any = None, subset: Any = None, th_dt: Any = None,
                              c_rho: Any = 0) -> Any:
        """Record the velocity reconstruction ``v^{n+theta} = M^{-1}(v^n - th_dt*grad phi)`` IN PLACE on
        @p state (rho frozen; mom = rho*v written back over the @p subset) -- the generic counterpart of
        ``P.schur_reconstruct``. @p phi is the solved potential (a scalar_field or 1-component State),
        @p th_dt = theta*dt. The final n+1 extrapolation (factor 1/theta) is the caller's affine
        algebra. Returns the updated State."""
        if isinstance(name, Value) and state is None:
            name, state = None, name
        if not (isinstance(state, Value) and state.vtype == "state"):
            raise ValueError("condensed_reconstruct: a State value is required (state=...)")
        if not _is_field_value(phi):
            raise ValueError("condensed_reconstruct: phi must be a scalar_field or State value (phi=...)")
        opname = self._condensed_operator_name(linear_operator)
        sub = self._condensed_subset(subset, "condensed_reconstruct")
        th_d = self._coeff_dict(th_dt, "th_dt", "condensed_reconstruct")
        return self._new("state", "condensed_reconstruct", (state, phi),
                         {"linear_operator": opname, "subset": sub, "th_dt": th_d,
                          "c_rho": self._comp_index(c_rho, "c_rho", "condensed_reconstruct")}, name,
                         state.block)
