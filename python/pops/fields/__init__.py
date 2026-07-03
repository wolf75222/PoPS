"""pops.fields -- typed elliptic field-problem authoring (Spec 5 sec.5.5 / sec.9).

The ``pops.fields`` package describes a self-consistent FIELD solve: an unknown computed
by inverting an elliptic operator each step (the Poisson coupling of a plasma, a pressure
projection, ...). The central object is :class:`FieldProblem` (with the Poisson-family
shortcuts :class:`PoissonProblem` / :class:`ScreenedPoissonProblem` /
:class:`AnisotropicPoissonProblem`); the supporting submodules declare the typed pieces:

* :mod:`pops.fields.bcs` -- field-value boundary conditions + face selectors;
* :mod:`pops.fields.rhs` -- typed right-hand-side sources (``ChargeDensity`` / ``FixedSource`` /
  the composed ``SumRHS``);
* :mod:`pops.fields.outputs` -- typed field outputs (``FieldOutput`` / ``GradientOutput`` /
  ``DerivedField``);
* :mod:`pops.fields.coefficients` -- scalar / reaction operator coefficients;
* :mod:`pops.fields.nullspace` -- ``ConstantNullspace`` for singular operators;
* :mod:`pops.fields.aux` -- static / derived aux fields + the re-exported ``AuxHalo``.

Everything is inert; the runtime materialises and solves after validation. This typed authoring
surface is DISTINCT from the flat elliptic-field brick catalog re-exported as
:data:`pops.fields.catalog` (Spec 5 criterion 7: moved out of ``pops.lib.fields``).
"""
from .problem import FieldProblem, SolveCadence, lower_field_solver
from .poisson import (PoissonProblem, ScreenedPoissonProblem,
                      AnisotropicPoissonProblem)
from .policies import FieldSolvePolicy, HoldPrevious, Recompute
from .nullspace import ConstantNullspace
from .outputs import FieldOutput, GradientOutput, DerivedField
from . import bcs, rhs, coefficients, nullspace, aux, policies, outputs
# The elliptic-field brick catalog (criterion 7: moved out of pops.lib.fields). Bind the SimpleNamespace
# as ``pops.fields.catalog`` (shadowing the submodule) so ``catalog.GeometricMG()`` resolves. It is a
# flat BrickDescriptor catalog, DISTINCT from the typed FieldProblem authoring surface above.
from .catalog import fields as catalog

__all__ = [
    "FieldProblem", "SolveCadence", "lower_field_solver", "PoissonProblem",
    "ScreenedPoissonProblem", "AnisotropicPoissonProblem",
    "FieldSolvePolicy", "HoldPrevious", "Recompute", "ConstantNullspace",
    "FieldOutput", "GradientOutput", "DerivedField",
    "bcs", "rhs", "outputs", "coefficients", "nullspace", "aux", "policies", "catalog",
]
