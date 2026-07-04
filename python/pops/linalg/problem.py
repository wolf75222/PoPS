"""pops.linalg.problem -- the typed linear system ``A x = b`` and its residual (Spec 5 sec.5.6).

:class:`LinearProblem` NAMES the algebraic system: an operator ``A`` (a
:class:`~pops.linalg.operator.LinearOperator` / :class:`~pops.linalg.operator.MatrixFreeOperator`),
the unknown handle ``x`` and the right-hand side ``b``. :class:`Residual` names ``b - A x`` for
a problem. Both are inert descriptors: they declare the algebra, they do NOT solve it (the C++
runtime Krylov loop does) and they compute nothing.

A :class:`LinearProblem` may also carry a typed Krylov ``method`` (and optional
``preconditioner`` / ``tol`` / ``max_iter`` / ``restart``): :meth:`lower` then emits the
``solve_linear``-shaped lowering RECORD (method / preconditioner / tol / max_iter / restart)
that names the native route, WITHOUT a public Python solver and WITHOUT a per-iteration Python
callback -- the loop stays in C++ (ADC-535). The method / preconditioner lowering delegates to
the SAME predicates the Program ``P.solve_linear`` op uses (``pops.time.program_solve``), so the
typed object and the Program path lower through a single source.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability, Descriptor
from .operator import LinearOperator, MatrixFreeOperator

#: The operator types a :class:`LinearProblem` accepts for ``A``.
_OPERATOR_TYPES = (LinearOperator, MatrixFreeOperator)


def _handle_name(handle: Any) -> Any:
    """A short, stable name for an unknown / rhs handle (its ``name`` attr, else its repr)."""
    if handle is None:
        return None
    return getattr(handle, "name", repr(handle))


def _context_is_amr_layout(context: Any) -> bool:
    """True when the route @p context names an AMR mesh layout (duck-typed, no mesh import).

    A layout descriptor advertises its kind through ``capabilities()["layout"]`` (``"amr"`` /
    ``"uniform"``); the context may carry it under a ``"layout"`` key, a ``.layout`` attribute, or
    simply BE the layout. Recognised WITHOUT importing :mod:`pops.mesh` into the linalg layer (a
    forbidden cross-layer edge). A context with no layout returns False (mirror of the elliptic
    ``_context_is_amr_layout``), so a plain ``available()`` call is never a false AMR refusal.
    """
    if context is None:
        return False
    layout = context.get("layout") if isinstance(context, dict) else getattr(context, "layout", None)
    if layout is None:
        layout = context  # the context may itself be the layout descriptor
    caps = getattr(layout, "capabilities", None)
    if callable(caps):
        try:
            declared: Any = caps()
        except Exception:
            return False  # available() must never raise: an odd context is simply "not AMR"
        # ``declared`` is a typed CapabilitySet (or a plain dict): both expose ``.get`` (ADC-625).
        if hasattr(declared, "get") and declared.get("layout") == "amr":
            return True
    return False


class LinearProblem(Descriptor):
    """The typed linear system ``A x = b`` (Spec 5 sec.5.6).

    ``LinearProblem(operator=A, unknown=x, rhs=b)`` names the algebra: the linear operator
    ``A`` (a :class:`~pops.linalg.operator.LinearOperator` or
    :class:`~pops.linalg.operator.MatrixFreeOperator`), the unknown handle ``x`` and the
    right-hand-side handle ``b``. An optional typed Krylov ``method`` (``pops.solvers.krylov.CG()``
    ...) with ``preconditioner`` / ``tol`` / ``max_iter`` / ``restart`` names the solve; :meth:`lower`
    then emits the ``solve_linear``-shaped record. The unknown / rhs are stored and surfaced, not
    interpreted, here.

    :meth:`available` / :meth:`validate` refuse a bad route with a DISTINGUISHABLE class (the
    ``missing`` tag names it): ``operator`` (not a linear-operator descriptor), ``rhs`` (a linear
    solve needs a right-hand side), ``preconditioner`` (a non-identity preconditioner on a CG /
    Richardson method, which has no preconditioner slot in the matrix-free path) and ``layout`` (a
    method whose declared capabilities do not cover the context mesh layout). It is inert; it does
    NOT solve (the C++ Krylov loop does).
    """

    category = "linear_problem"

    def __init__(self, operator: Any = None, unknown: Any = None, rhs: Any = None,
                 name: Any = None, method: Any = None, preconditioner: Any = None,
                 tol: float = 1e-8, max_iter: Any = None, restart: Any = None) -> None:
        self.operator = operator
        self.unknown = unknown
        self.rhs = rhs
        self._name = None if name is None else str(name)
        self.method = method
        self.preconditioner = preconditioner
        self.tol = tol
        self.max_iter = max_iter
        self.restart = restart

    @property
    def name(self) -> str:
        return self._name if self._name is not None else type(self).__name__

    def options(self) -> dict:
        return {"name": self._name,
                "operator": _handle_name(self.operator),
                "unknown": _handle_name(self.unknown),
                "rhs": _handle_name(self.rhs),
                "method": _handle_name(self.method),
                "preconditioner": _handle_name(self.preconditioner)}

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        return RequirementSet({"operator": True, "unknown": True, "rhs": True})

    def capabilities(self) -> Any:
        from pops.descriptors_report import CapabilitySet
        matrix_free = bool(getattr(self.operator, "capabilities", dict)().get("matrix_free")) \
            if isinstance(self.operator, _OPERATOR_TYPES) else False
        return CapabilitySet({"linear": True, "matrix_free": matrix_free})

    def available(self, context: Any = None) -> Any:
        """An explainable status; the ``missing`` tag NAMES the incompatibility class (ADC-535)."""
        if not isinstance(self.operator, _OPERATOR_TYPES):
            got = type(self.operator).__name__
            return Availability.no(
                "%s needs a LinearOperator/MatrixFreeOperator; got %r" % (self.name, got),
                missing=["operator"])
        if self.rhs is None:
            return Availability.no(
                "%s needs a right-hand side (rhs=); a linear solve A x = b has no b" % self.name,
                missing=["rhs"])
        if self.method is not None:
            scheme = getattr(self.method, "scheme", None)
            precond = self.preconditioner
            precond_scheme = getattr(precond, "scheme", None)
            # A non-identity preconditioner needs the runtime ApplyFn slot only GMRES / BiCGStab
            # expose (generic_krylov.hpp); cg_solve / richardson_solve have no preconditioner
            # parameter. This is an honest capability limit of the matrix-free path.
            if (precond is not None and precond_scheme not in (None, "identity")
                    and scheme in ("cg", "richardson")):
                return Availability.no(
                    "%s: preconditioning is not available for CG/Richardson in the matrix-free "
                    "Krylov path; use GMRES() or BiCGStab()" % self.name,
                    missing=["preconditioner"],
                    alternatives=["pops.solvers.krylov.GMRES()",
                                  "pops.solvers.krylov.BiCGStab()"])
            # Layout: refuse when the method's DECLARED capabilities do not cover the context mesh
            # layout (delegated to the descriptor's own capabilities -- no fabricated rule).
            caps = getattr(self.method, "capabilities", None)
            declared: Any = caps if isinstance(caps, dict) else (caps() if callable(caps) else {})
            if _context_is_amr_layout(context) and declared.get("supports_amr") is False:
                return Availability.no(
                    "%s: method %s does not support an AMR layout (supports_amr=False); use an "
                    "AMR-capable Krylov method" % (self.name, getattr(self.method, "name", scheme)),
                    missing=["layout"])
        return Availability.yes()

    def validate(self, context: Any = None) -> Any:
        status = self.available(context)
        if status.ok:
            return True
        # Preserve the historical TypeError for the operator class (callers catch TypeError); the
        # other classes raise ValueError with the same explainable text.
        if "operator" in status.missing:
            raise TypeError(
                "%s: operator must be a pops.linalg.LinearOperator or MatrixFreeOperator; "
                "got %r" % (self.name, type(self.operator).__name__))
        raise ValueError("%s is not available for this route:\n%s" % (self.name, status))

    def lower(self, context: Any = None) -> Any:
        """The ``solve_linear``-shaped lowering RECORD for this problem (metadata only; ADC-535).

        Emits ``{name, category, native_id, options, operator, unknown, rhs, method, preconditioner,
        tol, max_iter, restart}`` naming the native Krylov route. The ``method`` / ``preconditioner``
        lower through the SAME predicates the Program ``P.solve_linear`` op uses
        (``pops.time.program_solve``), so the typed object and the Program path lower through one
        source. It is INERT: it runs no numeric loop, opens no extension and installs no public
        Python solver or per-iteration Python callback -- the C++ Krylov loop does the solve.
        Requires a ``method``; a problem with no method names the algebra only and cannot lower.
        """
        if self.method is None:
            raise ValueError(
                "%s.lower(): a typed Krylov method is required to lower to solve_linear "
                "(e.g. LinearProblem(..., method=pops.solvers.krylov.CG(max_iter=200)))"
                % self.name)
        self.validate(context)
        # Delegate the method / preconditioner scheme resolution to the SINGLE source the Program
        # op uses (lazy import: keep pops.linalg free of a module-scope pops.time edge).
        from pops.time.program_solve import _lower_krylov_method, _lower_preconditioner
        # ADC-644/645: the shared lowerings now return (scheme, options) tuples; this record keeps
        # naming the scheme tokens only (the options travel on the Program IR node, not here).
        scheme, _method_options = _lower_krylov_method(self.method)
        precond, _precond_options = _lower_preconditioner(self.preconditioner)
        if self.max_iter is None or isinstance(self.max_iter, bool) or \
                not isinstance(self.max_iter, int) or self.max_iter <= 0:
            # The descriptor factory already refused a bad budget, but a LinearProblem(max_iter=)
            # is set directly here: refuse a missing / non-positive budget with the native message.
            raise ValueError(
                "%s.lower(): max_iter is required and must be a positive int (dynamic solver "
                "loops require max_iter); got %r" % (self.name, self.max_iter))
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(
            name=self.name, category=self.category, native_id=self.native_id,
            options=self.options(),
            extra={"operator": _handle_name(self.operator),
                   "unknown": _handle_name(self.unknown),
                   "rhs": _handle_name(self.rhs),
                   "method": scheme,
                   "preconditioner": precond,
                   "tol": float(self.tol),
                   "max_iter": int(self.max_iter),
                   "restart": int(self.restart) if scheme == "gmres" and self.restart is not None
                   else None})

    def inspect(self) -> Any:
        info = super().inspect()
        info["operator"] = _handle_name(self.operator)
        info["unknown"] = _handle_name(self.unknown)
        info["rhs"] = _handle_name(self.rhs)
        info["method"] = _handle_name(self.method)
        info["preconditioner"] = _handle_name(self.preconditioner)
        return info


class Residual(Descriptor):
    """The residual ``b - A x`` of a :class:`LinearProblem` (Spec 5 sec.5.6).

    ``Residual(problem)`` names the residual vector of an ``A x = b`` system; it is the quantity
    a norm / reduction is taken of to measure convergence. It is inert: it references the problem
    and computes nothing (the runtime forms ``b - A x``).
    """

    category = "residual"

    def __init__(self, problem: Any) -> None:
        self.problem = problem

    def options(self) -> dict:
        return {"problem": _handle_name(self.problem)}

    def requirements(self) -> Any:
        from pops.descriptors_report import RequirementSet
        return RequirementSet({"problem": True})

    def available(self, context: Any = None) -> Any:
        if not isinstance(self.problem, LinearProblem):
            got = type(self.problem).__name__
            return Availability.no(
                "%s needs a LinearProblem; got %r" % (self.name, got), missing=["problem"])
        return Availability.yes()

    def validate(self, context: Any = None) -> Any:
        if not isinstance(self.problem, LinearProblem):
            raise TypeError(
                "%s: problem must be a pops.linalg.LinearProblem; got %r"
                % (self.name, type(self.problem).__name__))
        return True


__all__ = ["LinearProblem", "Residual"]
