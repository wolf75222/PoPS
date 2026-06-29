"""No public nonlinear-solver descriptors are exposed yet.

Spec 5 clean-break rule: public solver descriptors must select a real compiled route. PoPS already
has a real per-cell Newton path through :meth:`pops.time.Program.solve_local_nonlinear`, but there is
no standalone global ``pops::Newton`` / ``pops::FixedPoint`` solver descriptor. Those factories are
therefore intentionally absent instead of being published as ``available=False`` placeholders.
"""

__all__ = []
