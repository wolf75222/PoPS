"""An implicit temporal stage using one existing prepared physical rate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pops.time.solve_request import DerivativeStrategy, SolveRequest, SolveUnknown


@dataclass(frozen=True, slots=True)
class ImplicitStage:
    """Solve Q(q)-U_n-tau R(Q(q))=0 using the selected physical rate R.

    ``previous`` is the frozen conserved value U_n=Q(q_n), not an initial guess.
    ``accumulation`` is an optional registered Program operator mapping numerical
    coordinates q to conserved state Q(q); omission explicitly means identity.
    The same physical rate is called on Q(q) by both explicit and implicit paths.
    A consumed q must pass through this map before committing conserved state.
    The unknown is one ordered vector state, not a product of unrelated block unknowns.
    Native realization accepts only prepared state, source, diffusive rate, local transform
    and linear-combination operations; transport needs its own exchange accounting.
    """

    rate: Any
    previous: Any
    tau: Any
    accumulation: Any = None
    finite_difference_step: float = 6.055454452393343e-6

    def request(self, *, unknown: SolveUnknown, seed: Any,
                derivative: DerivativeStrategy) -> SolveRequest:
        return SolveRequest(
            self, unknowns=(unknown,),
            equation_inputs={"previous": self.previous, "rate": self.rate,
                             "accumulation": self.accumulation, "tau": self.tau},
            seeds={unknown.name: seed}, derivative=derivative,
            residual_interpretation="Q(q)-U_n-tau*R(Q(q))",
            error_interpretation="spatial_residual_l2",
            problem_metadata={"kind": "evolved_implicit_stage",
                              "previous_representation": "conserved_U_n=Q(q_n)",
                              "accumulation": "identity" if self.accumulation is None
                              else "registered_coordinate_to_conserved_operator"})


__all__ = ["ImplicitStage"]
