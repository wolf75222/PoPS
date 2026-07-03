"""pops.problem.amr_handle -- the ``problem.amr`` refinement-policy handle (ADC-526).

A thin authoring shim that records the AMR refinement criteria (refine / regrid / nesting /
patches) and returns the :class:`~pops.problem.problem.Problem` so calls chain. It owns no AMR
runtime and no layout; the policies it records (``pops.mesh.amr.Refine`` / ``TagUnion`` /
``RegridEvery`` ...) are inert descriptors the deferred AMR route materialises at compile.

In this commit the Problem still carries a layout (the ADC-553 split preserves the pre-existing
layout-at-construction behaviour); the handle records the criteria on the layout when the layout is
AMR AND on the Problem's constraint registry. Commit 2 (ADC-526) removes the layout from the
constructor, so the criteria then live ONLY on the constraint registry and are applied to the
layout passed to ``pops.compile(problem, layout=...)``.
"""


class ProblemAmrHandle:
    """The ``problem.amr`` handle: record refinement criteria, chain back to the Problem."""

    def __init__(self, problem):
        self._problem = problem

    def _refine_context(self):
        """The single block's physics model the refine subject is checked against, or ``None``.

        Returns ``None`` when there is not exactly one block (the subject would be ambiguous across
        blocks) so the subject check DEFERS rather than guesses -- no false positive.
        """
        blocks = self._problem._blocks
        if len(blocks) != 1:
            return None
        (name,) = blocks.names()
        spec = blocks.spec(name)
        return spec.get("model") if spec else None

    def refine(self, criterion=None, *, regrid=None, nesting=None, patches=None):
        """Record the refinement criterion / regrid / nesting / patch policies (chains).

        When a @p criterion is recorded, its subject (role / state component / named aux) is
        validated against the Problem's block model HERE -- the one place the model is available --
        so a refinement on a bogus role is refused before runtime. The discipline is NO FALSE
        POSITIVE: the subject check only runs when exactly one block model is present.
        """
        if criterion is not None and hasattr(criterion, "validate"):
            criterion.validate(self._refine_context())
        # Record on the layout-free constraint registry (the ADC-526 home for the criteria).
        self._problem._constraints.set_refinement(
            refine=criterion, regrid=regrid, nesting=nesting, patches=patches)
        # Back-compat: while the Problem still carries an AMR layout (pre-ADC-526), mirror the
        # criteria onto it so the existing compile path keeps seeing them.
        layout = getattr(self._problem, "_layout", None)
        if layout is not None:
            if criterion is not None:
                layout.refine = criterion
            if regrid is not None:
                layout.regrid = regrid
            if nesting is not None:
                layout.nesting = nesting
            if patches is not None:
                layout.patches = patches
        return self._problem


__all__ = ["ProblemAmrHandle"]
