"""pops.problem.amr_handle -- the ``problem.amr`` refinement-policy handle (ADC-526).

A thin authoring shim that records the AMR refinement criteria (refine / regrid / nesting /
patches) and returns the :class:`~pops.problem.problem.Case` so calls chain. It owns no AMR
runtime and no layout; the policies it records (``pops.mesh.amr.Refine`` / ``TagUnion`` /
``RegridEvery`` ...) are inert descriptors the deferred AMR route materialises at compile.

The criteria live only on the Case's layout-free constraint registry (ADC-526) and are merged
into a detached ``AMR`` layout by ``pops.compile(problem, layout=...)``. The user-owned layout is
never a second authority and is never mutated by this authoring handle.
"""
from __future__ import annotations

from typing import Any


class CaseAmrHandle:
    """The ``problem.amr`` handle: record refinement criteria, chain back to the Case."""

    def __init__(self, problem: Any) -> None:
        self._problem = problem

    def refine(self, criterion: Any = None, *, regrid: Any = None, nesting: Any = None,
               patches: Any = None) -> Any:
        """Record the refinement criterion / regrid / nesting / patch policies (chains).

        A criterion resolves every Handle leaf through :meth:`Case.resolve` before it enters the
        registry. Therefore a model-local Handle used directly or inside an indicator expression by
        several blocks is rejected as ambiguous with every candidate owner, an explicit
        ``block[handle]`` is accepted, and a foreign / forged Handle is refused. No name matching and
        no single-block special case exists here.
        """
        if criterion is not None:
            from pops.mesh.amr import Refine, TagUnion

            if not isinstance(criterion, (Refine, TagUnion)):
                raise TypeError(
                    "problem.amr.refine criterion must be a pops.mesh.amr.Refine / TagUnion, "
                    "got %r" % type(criterion).__name__)
            criterion.validate()
            criterion = criterion.resolve_references(self._problem.resolve)

        # The constraint registry is the sole authoring authority. Layout resolution later makes
        # a detached AMR value and explicitly rejects a layout-vs-Case policy conflict.
        self._problem._constraints.set_refinement(
            refine=criterion, regrid=regrid, nesting=nesting, patches=patches)
        return self._problem


__all__ = ["CaseAmrHandle"]
