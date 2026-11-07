"""The user surface of spec 0.3, section 2.6: level 1 (the model, the numerics and the program).

This is lot 1 of N15. Until the cut-over of §21.7 makes the surface ``pops`` itself, a case
imports it as ``import pops.surface as pops`` and ``from pops.surface.math import *``; the words
of the library that it elaborates are ``pops.lib.numerics.schemes`` (``upwind``, ``rusanov``,
``two_point``) and ``pops.lib.time.methods`` (``forward_euler``, ``ssprk2``, ``ssprk3``, ``rk4``).
Building ``pops.Case(...)`` traces the case into a b6 case (2.6.6, ``_elaborate``), and
``pops.inspect(case)`` validates and resolves it without compiling (SU12).

Every other construct of 2.6.2, 2.6.4 and 2.6.8 is importable and refuses its use with
``NotYetElaborated``, which names the lot that elaborates it; a name of an earlier text is refused
with ``SymbolNotInSurface``, whose message is the row of §21.7 (SU13); any other name is refused
with the list of the words of this module.
"""
from __future__ import annotations

from typing import Any

from . import math
from ._elaborate import Choice, Report, Row
from ._geometry import Box, Periodic
from ._method import Candidate, method
from ._model import Components, Model, Positive, Scalar, Tensor, Vector
from ._numerics import CellAverage, Numerics, Realization, SchemeValue, TermView
from ._program import (
    AMR, CUDA, Advance, Case, GaussLegendre, OpenMP, Program, Resources, Uniform, bind, claim,
    compile, inspect,
)
from ._refusal import (
    AmbiguousElaboration, AntiDiffusive, ClosedBodyViolation, ContradictsDerivation,
    CoverageNotExact, IncrementNotDerivable, IncrementTypeMismatch, LevelMismatch,
    MissingBoundaryRealization, NotAPopsOperation, NotDiffusive, NotYetElaborated, OutOfScope,
    RealizationMismatch, ReservedName, SpellingRefusal, SupportMismatch, SurfaceRefusal,
    SymbolNotInSurface, SymbolicTruthValue, TermCountedTwice, UnknownName,
)

_N8_ASSESS = "N8 phase 1 (the assessments of T28)"
_N8_LATER = "a later lot of N8"
_SOLVES = "a later lot of N15 with the solves of N9"
_RUN = "the run of N8 (Simulation.run, RunResult and Output, CC14) and a later lot of N15"
_HIERARCHY = "a later lot of N15 with the hierarchies of N12"
_REPRESENT = "a later lot of N15 with the representations and projections of G03 and G07"
_SUPPORTS = "a later lot of N15 with the supports of G03 and G09"
_LATER_N15 = "a later lot of N15"

# The constructs of 2.6.2, 2.6.4 and 2.6.8 that this lot does not elaborate, with their owner.
# The quadrature rule GaussHermite of 2.6.2 is not listed: CF2 keeps the name of a quadrature
# rule out of the core files (term hermite, N3), so the surface does not have it until the lot
# of the quadratures of G03 settles where the rule lives (a tension that SD86 records).
_LATER = {
    "Accept": _N8_ASSESS, "Reject": _N8_ASSESS, "Stop": _N8_ASSESS, "Transform": _N8_ASSESS,
    "assess": _N8_ASSESS, "fail": _N8_ASSESS, "refuse": _N8_ASSESS, "Finite": _N8_ASSESS,
    "Observed": _N8_ASSESS, "Admissible": _N8_ASSESS, "NotAdmissible": _N8_ASSESS,
    "transform": _N8_ASSESS,
    "Tolerance": _SOLVES, "solve": _SOLVES, "try_solve": _SOLVES, "unknown": _SOLVES,
    "solver": _SOLVES, "Norm": _SOLVES, "Dense": _SOLVES, "Affine": _SOLVES,
    "eliminate": _SOLVES, "SolveFailed": _SOLVES, "Normalization": _SOLVES,
    "Output": _RUN, "record": _RUN, "every": _RUN, "RunFailed": _RUN, "BudgetExhausted": _RUN,
    "Scientific": _RUN, "Bitwise": _RUN, "Reproducible": _RUN, "Cache": _RUN,
    "Regrid": _HIERARCHY, "Level": _HIERARCHY, "invalidate": _HIERARCHY, "remap": _HIERARCHY,
    "restrict_onto": _HIERARCHY, "replace_contribution": _HIERARCHY,
    "project": _REPRESENT, "CellTotal": _REPRESENT,
    "PointValue": _REPRESENT, "FaceNormalIntegral": _REPRESENT, "EdgeCirculation": _REPRESENT,
    "DualLoad": _REPRESENT, "Covector": _REPRESENT, "reconstruct": _REPRESENT,
    "Weak": _REPRESENT, "Pointwise": _REPRESENT, "Approximate": _REPRESENT,
    "Realizable": _REPRESENT,
    "product": _SUPPORTS, "Fibre": _SUPPORTS, "Line": _SUPPORTS, "Ball": _SUPPORTS,
    "Connectivity": _SUPPORTS, "Strided": _SUPPORTS, "Global": _SUPPORTS, "Family": _SUPPORTS,
    "Dimension": _SUPPORTS,
    "MovingConfiguration": "a later lot of N15 with the moving configurations of D18",
    "Swept": "a later lot of N15 with the moving configurations of D18",
    "scheme": "a later lot of N15 with @pops.scheme (A4)",
    "jointly": "a later lot of N15 with the joint realizations of G05",
    "multistep": "a later lot of N8 (T27)", "substep": _N8_LATER, "try_substep": _N8_LATER,
    "exact_flow": _N8_LATER, "choose": _N8_LATER, "action": _N8_LATER, "predictor": _N8_LATER,
    "step_change": _N8_LATER, "step_duration": _N8_LATER, "step_start": _N8_LATER,
    "Controller": "N8 phase 2 (controllers)", "Fixed": "N8 phase 2 (controllers)",
    "Frozen": _N8_LATER, "Revolve": _N8_LATER, "recompute": _N8_LATER, "Off": _N8_LATER,
    "role": _LATER_N15, "parameter": _LATER_N15, "compose": _LATER_N15, "member": _LATER_N15,
    "Auxiliary": _LATER_N15, "explain": _LATER_N15,
    "claims": "a later lot of N15 with the claims of §13",
    "regions": "a later lot of N15 with the region level (SU15)",
    "schedules": _RUN, "external": _LATER_N15, "discrete": _LATER_N15,
}

# Names of earlier texts (§21.7): the construct to write, as the message of SymbolNotInSurface.
_EARLIER = {
    "Rectangle": "pops.Box(x=(0, 1), y=(0, 1)), one keyword per axis",
    "Interval": "pops.Box(x=(0, 1)), a box of one axis",
    "Cartesian1D": "pops.Box(x=...), whose named axes give the Cartesian frame",
    "Cartesian2D": "pops.Box(x=..., y=...), whose named axes give the Cartesian frame",
    "Cartesian3D": "pops.Box(x=..., y=..., z=...), whose named axes give the Cartesian frame",
    "ConstParam": 'model.constant("g", value=...)',
    "RuntimeParam": 'model.parameter("nu", default=...), or pops.parameter(...) for one that no '
                    'relation reads',
    "Space": "pops.Scalar(Ω), pops.Vector(Ω) or pops.Components(Ω, ...), which name the support "
             "first",
    "Realize": "pops.Numerics(schemes={term: scheme}), keyed by what the scheme covers",
    "Represent": "pops.Numerics(represent={u: pops.CellAverage()})",
    "Problem": "pops.solve(relation, X) or pops.solve([relations], [X, Y])",
    "Proposal": "dt= of pops.Advance or pops.substep, and next_dt= of pops.Accept or pops.Reject",
    "Retry": "attempts=n and transforms=k on pops.Advance",
    "Request": "pops.Tolerance(rtol=, atol=, observe=, norm=)",
    "Layout": "layout={Ω: pops.Uniform(cells=...)} in pops.Case",
    "layouts": "pops.Uniform(cells=...) or pops.AMR(...) from pops",
    "mesh": "the layout in pops.Case(layout={Ω: pops.Uniform(cells=...)})",
    "amr": "pops.AMR(...) in layout= and pops.Regrid(...) in pops.Program",
    "domain": "pops.Box, pops.product or pops.Finite, which pops itself exports",
    "spaces": "pops.Scalar, pops.Vector or another space, which pops itself exports",
    "rel": "a relation with == and the words of pops.math; pops.rel has no words",
    "Target": "target=pops.OpenMP() or target=pops.CUDA() on pops.compile",
    "set_threads": "resources=pops.Resources(threads=4) on pops.bind",
    "validate": "pops.inspect(case), which validates and resolves without compiling",
    "resolve": "pops.inspect(case), which validates and resolves without compiling",
    "space_ratio": "level_ratio(Ω, l), or leave ratio= of subcycle out, which takes it",
    "solvers": "the solver, imported from pops.lib.solve by name",
    "UnrealizedOutput": "pops.CoverageNotExact, the refusal of a definition that holds a "
                        "derivative or an integral of a discrete field and is read without a "
                        "scheme",
}

_LIBRARY = "pops.lib.numerics.schemes and pops.lib.time.methods"


def _later(name: str, owner: str) -> Any:
    def later(*args: Any, **kwargs: Any) -> Any:
        raise NotYetElaborated("pops.%s" % name, owner).with_traceback(None)

    later.__name__ = later.__qualname__ = name
    return later


for _name, _owner in _LATER.items():
    globals()[_name] = _later(_name, _owner)


def __getattr__(name: str) -> Any:
    """Refuse a name that the surface does not have (SU3, SU13, §21.7); export nothing."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _EARLIER:
        raise SpellingRefusal(_EARLIER[name]).refuse("pops.%s" % name, "the surface")
    raise SymbolNotInSurface(
        "pops.%s is not a word of the surface; pops has %s, and the library %s"
        % (name, ", ".join(sorted(__all__)), _LIBRARY),
        write="a word of 2.6.2 to 2.6.4 or 2.6.8, or of the library of 2.6.9")


__all__ = [
    "AMR", "Advance", "Box", "CUDA", "Candidate", "Case", "CellAverage", "Components",
    "GaussLegendre", "Model", "Numerics", "OpenMP", "Periodic", "Positive", "Program",
    "Resources", "Scalar", "Tensor", "Uniform", "Vector", "bind", "claim", "compile", "inspect",
    "math", "method",
    # the refusals of §14.1 that this lot raises, and the values that the library reads
    "AmbiguousElaboration", "AntiDiffusive", "ClosedBodyViolation", "ContradictsDerivation",
    "CoverageNotExact", "IncrementNotDerivable", "IncrementTypeMismatch", "LevelMismatch",
    "MissingBoundaryRealization", "NotAPopsOperation", "NotDiffusive", "NotYetElaborated",
    "OutOfScope", "RealizationMismatch", "ReservedName", "SupportMismatch", "SurfaceRefusal",
    "SymbolNotInSurface", "SymbolicTruthValue", "TermCountedTwice", "UnknownName",
    "Choice", "Realization", "Report", "Row", "SchemeValue", "TermView",
]
