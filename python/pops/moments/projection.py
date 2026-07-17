"""Typed moment realizability predicates and pointwise projections.

``RealizabilityProjection`` still owns the closure-local floors used while evaluating moment
fluxes, but it also owns the complete HyQMOM15 state contract.  That contract checks the degree-two
Gram matrix built from all 15 moments and projects an invalid, positive-density state onto the
moments of a genuine bivariate Gaussian.  The projection is emitted with the model and therefore
runs in native code through the ordinary ``BlockProjection`` Program primitive.
"""
from __future__ import annotations

import math
from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet
from pops._ir.ops import abs_ as _abs, sign as _sign, sqrt as _sqrt


_HYQMOM15_INDICES = tuple(
    (p, q) for q in range(5) for p in range(5 - q)
)
_HYQMOM15_COMPONENTS = tuple("M%d%d" % index for index in _HYQMOM15_INDICES)
_MOMENT_MATRIX_BASIS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


def _maximum(left: Any, right: Any) -> Any:
    return ((left + right) + _abs(left - right)) / 2.0


def _minimum(left: Any, right: Any) -> Any:
    return ((left + right) - _abs(left - right)) / 2.0


def _positive_indicator(value: Any, threshold: float) -> Any:
    """Return the branch-free exact 0/1 indicator for ``value > threshold``.

    Equality intentionally maps to one half.  It therefore cannot masquerade as an accepted fixed
    point: a boundary state is projected once and rejected if the strict recheck still fails.
    """

    return (_sign(value - threshold) + 1.0) / 2.0


def _moment_matrix(moments: dict[tuple[int, int], Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(moments[(left[0] + right[0], left[1] + right[1])]
              for right in _MOMENT_MATRIX_BASIS)
        for left in _MOMENT_MATRIX_BASIS
    )


def _positive_definite_indicator(
    matrix: tuple[tuple[Any, ...], ...], *, eps_m00: float, eps_cov: float,
) -> Any:
    """Branch-free LDL factorization predicate for the complete 6x6 moment matrix."""

    size = len(matrix)
    lower: list[list[Any]] = [[0.0] * size for _ in range(size)]
    pivots: list[Any] = []
    indicator: Any = 1.0
    for row in range(size):
        pivot = matrix[row][row]
        for column in range(row):
            pivot = pivot - lower[row][column] * lower[row][column] * pivots[column]
        threshold = eps_m00 if row == 0 else eps_cov
        indicator = indicator * _positive_indicator(pivot, threshold)
        safe_pivot = _maximum(pivot, threshold)
        pivots.append(pivot)
        for target in range(row + 1, size):
            numerator = matrix[target][row]
            for column in range(row):
                numerator = (
                    numerator
                    - lower[target][column] * lower[row][column] * pivots[column]
                )
            lower[target][row] = numerator / safe_pivot
    return indicator


def _gaussian_moments(
    moments: dict[tuple[int, int], Any], *, eps_m00: float, eps_cov: float,
) -> dict[tuple[int, int], Any]:
    """Return order-four raw moments of a positive bivariate Gaussian."""

    rho = moments[(0, 0)]
    safe_rho = _maximum(rho, eps_m00)
    ux = moments[(1, 0)] / safe_rho
    uy = moments[(0, 1)] / safe_rho
    scaled_tolerance = eps_cov / safe_rho
    # For a Gaussian, the degree-two Gram-matrix LDL pivots include
    # ``rho * variance`` and ``2 * rho * variance**2``.  This floor keeps both
    # families strictly above ``eps_cov``, including when rho is close to its floor.
    variance_floor = 2.0 * _maximum(scaled_tolerance, _sqrt(scaled_tolerance))
    cxx = _maximum(moments[(2, 0)] / safe_rho - ux * ux, variance_floor)
    cyy = _maximum(moments[(0, 2)] / safe_rho - uy * uy, variance_floor)
    cxy_raw = moments[(1, 1)] / safe_rho - ux * uy
    # A half-width correlation cone leaves the conditional-variance pivots a
    # uniform distance from zero instead of manufacturing a near-rank-one repair.
    correlation_limit = 0.5 * _sqrt(cxx * cyy)
    cxy = _minimum(_maximum(cxy_raw, -correlation_limit), correlation_limit)

    return {
        (0, 0): rho,
        (1, 0): rho * ux,
        (0, 1): rho * uy,
        (2, 0): rho * (ux * ux + cxx),
        (1, 1): rho * (ux * uy + cxy),
        (0, 2): rho * (uy * uy + cyy),
        (3, 0): rho * (ux * ux * ux + 3.0 * ux * cxx),
        (2, 1): rho * (
            ux * ux * uy + uy * cxx + 2.0 * ux * cxy
        ),
        (1, 2): rho * (
            ux * uy * uy + ux * cyy + 2.0 * uy * cxy
        ),
        (0, 3): rho * (uy * uy * uy + 3.0 * uy * cyy),
        (4, 0): rho * (
            ux * ux * ux * ux + 6.0 * ux * ux * cxx + 3.0 * cxx * cxx
        ),
        (3, 1): rho * (
            ux * ux * ux * uy
            + 3.0 * ux * uy * cxx
            + 3.0 * ux * ux * cxy
            + 3.0 * cxx * cxy
        ),
        (2, 2): rho * (
            ux * ux * uy * uy
            + ux * ux * cyy
            + uy * uy * cxx
            + 4.0 * ux * uy * cxy
            + cxx * cyy
            + 2.0 * cxy * cxy
        ),
        (1, 3): rho * (
            ux * uy * uy * uy
            + 3.0 * ux * uy * cyy
            + 3.0 * uy * uy * cxy
            + 3.0 * cyy * cxy
        ),
        (0, 4): rho * (
            uy * uy * uy * uy + 6.0 * uy * uy * cyy + 3.0 * cyy * cyy
        ),
    }


class RealizabilityProjection(Descriptor):
    """The realizability contract a moment hierarchy applies.

    ``(eps_m00, eps_cov, robust)`` map to the engine's smooth ``max(x, eps)`` floors on
    M00 and the covariance C20/C02.  For HyQMOM15, ``robust=True`` additionally installs a
    pointwise state projection over the complete order-four moment matrix.  With ``robust=False``
    the bare guard-free path runs (faithful to the references; may NaN on a degenerate state).
    """

    category = "realizability"

    def __init__(self, eps_m00: Any = 1e-12, eps_cov: Any = 1e-12, robust: bool = True) -> None:
        self.eps_m00 = float(eps_m00)
        self.eps_cov = float(eps_cov)
        if not math.isfinite(self.eps_m00) or self.eps_m00 <= 0.0:
            raise ValueError("eps_m00 must be finite and > 0")
        if not math.isfinite(self.eps_cov) or self.eps_cov <= 0.0:
            raise ValueError("eps_cov must be finite and > 0")
        self.robust = bool(robust)

    @classmethod
    def none(cls) -> Any:
        """The bare, guard-free projection (``robust=False``)."""
        return cls(robust=False)

    def options(self) -> dict:
        return {"eps_m00": self.eps_m00, "eps_cov": self.eps_cov, "robust": self.robust}

    def capabilities(self) -> Any:
        return CapabilitySet({"guard_level": "smooth" if self.robust else "bare"})

    def hyqmom15_projection_expressions(
        self, moments: dict[tuple[int, int], Any],
    ) -> tuple[Any, ...]:
        """Return the native pointwise HyQMOM15 state projector.

        A state whose complete degree-two moment matrix is strictly positive definite is returned
        unchanged.  A positive-density invalid state is replaced by the raw moments of a genuine
        bivariate Gaussian.  Non-positive density is deliberately not manufactured: it remains
        invalid so the enclosing ``ProjectAndRecheck`` guard reaches its explicit terminal action.
        """

        if set(moments) != set(_HYQMOM15_INDICES):
            raise ValueError(
                "HyQMOM15 realizability requires exactly the 15 raw moments through order four"
            )
        ordered = {index: moments[index] for index in _HYQMOM15_INDICES}
        if not self.robust:
            return tuple(ordered[index] for index in _HYQMOM15_INDICES)
        matrix_ok = _positive_definite_indicator(
            _moment_matrix(ordered), eps_m00=self.eps_m00, eps_cov=self.eps_cov,
        )
        density_ok = _positive_indicator(ordered[(0, 0)], self.eps_m00)
        gaussian = _gaussian_moments(
            ordered, eps_m00=self.eps_m00, eps_cov=self.eps_cov,
        )
        repair = density_ok * (1.0 - matrix_ok)
        return tuple(
            ordered[index] + repair * (gaussian[index] - ordered[index])
            for index in _HYQMOM15_INDICES
        )

    @staticmethod
    def _hyqmom15_array(values: Any) -> Any:
        import numpy as np

        result = np.asarray(values, dtype=np.float64)
        if result.ndim < 1 or result.shape[0] != len(_HYQMOM15_INDICES):
            raise ValueError(
                "HyQMOM15 state must have 15 components on axis zero; got shape %r"
                % (result.shape,)
            )
        return result

    def hyqmom15_realizability_mask(self, values: Any) -> Any:
        """Return the pointwise native 6x6 LDL predicate for a NumPy state."""

        import numpy as np

        state = self._hyqmom15_array(values)
        moments = dict(zip(_HYQMOM15_INDICES, state, strict=True))
        shape = state.shape[1:]
        matrix = np.empty(shape + (6, 6), dtype=np.float64)
        for row, left in enumerate(_MOMENT_MATRIX_BASIS):
            for column, right in enumerate(_MOMENT_MATRIX_BASIS):
                matrix[..., row, column] = moments[
                    (left[0] + right[0], left[1] + right[1])
                ]
        finite = np.isfinite(state).all(axis=0)
        lower = np.zeros_like(matrix)
        pivots = []
        valid = finite.copy()
        for row in range(6):
            pivot = matrix[..., row, row].copy()
            for column in range(row):
                pivot -= lower[..., row, column] ** 2 * pivots[column]
            threshold = self.eps_m00 if row == 0 else self.eps_cov
            valid &= pivot > threshold
            safe_pivot = np.maximum(pivot, threshold)
            pivots.append(pivot)
            for target in range(row + 1, 6):
                numerator = matrix[..., target, row].copy()
                for column in range(row):
                    numerator -= (
                        lower[..., target, column]
                        * lower[..., row, column]
                        * pivots[column]
                    )
                lower[..., target, row] = numerator / safe_pivot
        return valid

    def is_hyqmom15_realizable(self, values: Any) -> bool:
        """Return whether every cell satisfies the complete HyQMOM15 predicate."""

        import numpy as np

        return bool(np.all(self.hyqmom15_realizability_mask(values)))

    def project_hyqmom15_array(self, values: Any) -> Any:
        """NumPy oracle for the exact native HyQMOM15 projection."""

        import numpy as np

        state = self._hyqmom15_array(values)
        result = np.array(state, copy=True)
        if not self.robust:
            return result
        valid = self.hyqmom15_realizability_mask(state)
        finite = np.isfinite(state).all(axis=0)
        repair = finite & (state[0] > self.eps_m00) & ~valid
        if not np.any(repair):
            return result

        moment = dict(zip(_HYQMOM15_INDICES, state, strict=True))
        rho = moment[(0, 0)]
        safe_rho = np.maximum(rho, self.eps_m00)
        ux = moment[(1, 0)] / safe_rho
        uy = moment[(0, 1)] / safe_rho
        scaled_tolerance = self.eps_cov / safe_rho
        variance_floor = 2.0 * np.maximum(scaled_tolerance, np.sqrt(scaled_tolerance))
        cxx = np.maximum(moment[(2, 0)] / safe_rho - ux * ux, variance_floor)
        cyy = np.maximum(moment[(0, 2)] / safe_rho - uy * uy, variance_floor)
        cxy_raw = moment[(1, 1)] / safe_rho - ux * uy
        correlation_limit = 0.5 * np.sqrt(cxx * cyy)
        cxy = np.minimum(np.maximum(cxy_raw, -correlation_limit), correlation_limit)
        gaussian = {
            (0, 0): rho,
            (1, 0): rho * ux,
            (0, 1): rho * uy,
            (2, 0): rho * (ux * ux + cxx),
            (1, 1): rho * (ux * uy + cxy),
            (0, 2): rho * (uy * uy + cyy),
            (3, 0): rho * (ux**3 + 3.0 * ux * cxx),
            (2, 1): rho * (ux * ux * uy + uy * cxx + 2.0 * ux * cxy),
            (1, 2): rho * (ux * uy * uy + ux * cyy + 2.0 * uy * cxy),
            (0, 3): rho * (uy**3 + 3.0 * uy * cyy),
            (4, 0): rho * (ux**4 + 6.0 * ux * ux * cxx + 3.0 * cxx * cxx),
            (3, 1): rho * (
                ux**3 * uy + 3.0 * ux * uy * cxx
                + 3.0 * ux * ux * cxy + 3.0 * cxx * cxy
            ),
            (2, 2): rho * (
                ux * ux * uy * uy + ux * ux * cyy + uy * uy * cxx
                + 4.0 * ux * uy * cxy + cxx * cyy + 2.0 * cxy * cxy
            ),
            (1, 3): rho * (
                ux * uy**3 + 3.0 * ux * uy * cyy
                + 3.0 * uy * uy * cxy + 3.0 * cyy * cxy
            ),
            (0, 4): rho * (uy**4 + 6.0 * uy * uy * cyy + 3.0 * cyy * cyy),
        }
        for component, index in enumerate(_HYQMOM15_INDICES):
            result[component] = np.where(repair, gaussian[index], result[component])
        return result

    @staticmethod
    def _projection_fixed_point(program: Any, state: Any, *, name: str) -> Any:
        """Build an all-component native fixed-point predicate on a provisional state."""

        from pops.time import BlockProjection

        probe = program.value("%s_probe" % name, 1 * state, at=state.point)
        projected = program.project(
            name="%s_projected" % name, state=probe, projection=BlockProjection(),
        )
        delta = program.value("%s_delta" % name, projected - state, at=state.point)
        error = program.abs_sum_component(delta, 0)
        for component in range(1, len(_HYQMOM15_COMPONENTS)):
            error = error + program.abs_sum_component(delta, component)
        return error <= 0.0

    def guard_hyqmom15_candidate(
        self,
        program: Any,
        candidate: Any,
        *,
        terminal_action: Any,
        name: str = "hyqmom15_realizability",
    ) -> Any:
        """Register and lower the two HyQMOM15 acceptance guards.

        The density guard rejects a non-repairable candidate after one explicit projection/recheck.
        The moment-matrix guard projects a repairable candidate lazily, then checks all 15
        components are an exact fixed point before allowing commit.
        """

        from pops.time import BlockProjection, FailRun, ProjectAndRecheck, RejectAttempt

        if not self.robust:
            raise ValueError(
                "HyQMOM15 ProjectAndRecheck requires robust=True so a block projection is installed"
            )
        if not isinstance(name, str) or not name:
            raise ValueError("HyQMOM15 guard name must be a non-empty string")
        if not isinstance(terminal_action, (RejectAttempt, FailRun)):
            raise TypeError("terminal_action must be RejectAttempt() or FailRun()")
        if getattr(candidate, "vtype", None) != "state":
            raise TypeError("HyQMOM15 acceptance guard requires a State ProgramValue")
        components = tuple(getattr(getattr(candidate, "space", None), "components", ()))
        if components != _HYQMOM15_COMPONENTS:
            raise ValueError(
                "HyQMOM15 acceptance guard requires canonical components %r; got %r"
                % (_HYQMOM15_COMPONENTS, components)
            )

        density = program.guard(
            "%s_density" % name,
            candidate,
            program.min(candidate) > self.eps_m00,
            action=ProjectAndRecheck(
                BlockProjection(), on_failure=terminal_action,
            ),
            recheck=lambda owner, projected: owner.min(projected) > self.eps_m00,
        )
        fixed_point = self._projection_fixed_point(
            program, density, name="%s_matrix" % name,
        )
        return program.guard(
            "%s_moment_matrix" % name,
            density,
            fixed_point,
            action=ProjectAndRecheck(
                BlockProjection(), on_failure=terminal_action,
            ),
            recheck=lambda owner, projected: self._projection_fixed_point(
                owner, projected, name="%s_recheck" % name,
            ),
        )

    def __repr__(self) -> str:
        return ("RealizabilityProjection(eps_m00=%g, eps_cov=%g, robust=%r)"
                % (self.eps_m00, self.eps_cov, self.robust))


class RealizableSet(Descriptor):
    """The realizable moment cone at ``order`` (an inert capability descriptor).

    Describes WHICH states a moment vector of this order may take (the realizable set the
    :class:`RealizabilityProjection` floors a state back into): a positive density
    (``M00 > 0``), a positive-semidefinite covariance (``C20`` / ``C02 >= 0``) and the Schur
    (Hankel) conditions on the higher moments. It CHOOSES no algorithm and computes nothing --
    it is the typed, inspectable record of the cone's constraints, so tooling can report what
    "realizable" means for an order without touching the runtime.
    """

    category = "realizability_set"

    def __init__(self, order):
        if order < 2:
            raise ValueError("RealizableSet: order >= 2 required (got %r)" % (order,))
        self.order = int(order)

    def options(self):
        return {"order": self.order}

    def capabilities(self):
        return CapabilitySet({"constraints": "m00_positive,cov_psd,schur"})

    def __repr__(self):
        return "RealizableSet(order=%d)" % (self.order,)


__all__ = ["RealizabilityProjection", "RealizableSet"]
