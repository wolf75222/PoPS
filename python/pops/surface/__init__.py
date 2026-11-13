"""The user surface of spec 0.3, section 2.6: level 1 (the model, the numerics and the program).

This is lot 1 of N15, with ``@pops.scheme`` and ``pops.discrete`` from lot 2, and the solves of
the lot « solves » (``pops.unknown``, ``pops.solve``, ``pops.Tolerance``, ``solvers=``). Until the cut-over of §21.7 makes the surface ``pops`` itself, a case
imports it as ``import pops.surface as pops`` and ``from pops.surface.math import *``; the words
of the library that it elaborates are ``pops.lib.numerics`` (``upwind``, ``rusanov``, ``hll``,
``two_point``, written with ``@pops.scheme``, N3 lot 1) and ``pops.lib.time.methods`` (``forward_euler``, ``ssprk2``, ``ssprk3``, ``rk4``).
Building ``pops.Case(...)`` traces the case into a b6 case (2.6.6, ``_elaborate``), and
``pops.inspect(case)`` validates and resolves it without compiling (SU12).

Every other construct of 2.6.2, 2.6.4 and 2.6.8 is importable and refuses its use with
``NotYetElaborated``, which names the lot that elaborates it; a name of an earlier text is refused
with ``SymbolNotInSurface``, whose message is the row of §21.7 (SU13); any other name is refused
with the list of the words of this module.
"""
from typing import Any as _Any

from . import discrete, math
from ._elaborate import Choice, Report, Row
from ._geometry import Box, Periodic
from ._method import Candidate, method, solve, unknown
from ._model import Components, Model, Positive, Scalar, Tensor, Vector
from ._numerics import CellAverage, Numerics, SchemeValue, TermView
from ._scheme import check_plain_function, scheme
from ._program import (
    AMR, CUDA, Advance, Case, OpenMP, Program, Resources, Tolerance, Uniform, bind, claim, compile,
    inspect,
)
from ._project import project
from . import _solver
from ._quadrature import Quadrature
from ._run import BudgetExhausted, Output, RunFailed, SolveFailed, every
from ._step import StepRuleValue
from ._refusal import (
    AmbiguousElaboration, AntiDiffusive, ClosedBodyViolation, ContradictsDerivation,
    CoverageNotExact, IncrementNotDerivable, IncrementTypeMismatch, LevelMismatch,
    MissingBoundaryRealization, NotAPopsOperation, NotDiffusive, NotYetElaborated, OutOfScope,
    RealizationMismatch, ReservedName, SupportMismatch, SurfaceRefusal,
    SymbolNotInSurface, SymbolicTruthValue, TermCountedTwice, UnknownName,
)
from ._refusal import AmbiguousSolvePath, UnknownSolvePath
from ._refusal import SpellingRefusal as _SpellingRefusal
from ._solver import PropertyNotDerived, choose, fail, record, solver

_N8_ASSESS = "N8 phase 1 (the assessments of T28)"
_N8_LATER = "a later lot of N8"
_SOLVES = "a later lot of N15 with the solves of N9"
_RUN = "the run of N8 (Simulation.run, RunResult and Output, CC14) and a later lot of N15"
_HIERARCHY = "a later lot of N15 with the hierarchies of N12"
_REPRESENT = "a later lot of N15 with the representations and projections of G03 and G07"
_SUPPORTS = "a later lot of N15 with the supports of G03 and G09"
_LATER_N15 = "a later lot of N15"

# The constructs of 2.6.2, 2.6.4 and 2.6.8 that this lot does not elaborate, with their owner.
# No named quadrature rule is listed: the core knows pops.Quadrature only (D0, review of 11/11,
# P3), and GaussLegendre and GaussHermite are words of pops.lib.quadrature, outside the core.
_LATER = {
    "Accept": _N8_ASSESS, "Reject": _N8_ASSESS, "Stop": _N8_ASSESS, "Transform": _N8_ASSESS,
    "assess": _N8_ASSESS, "refuse": _N8_ASSESS, "Finite": _N8_ASSESS,
    "Observed": _N8_ASSESS, "Admissible": _N8_ASSESS, "NotAdmissible": _N8_ASSESS,
    "transform": _N8_ASSESS,
    "try_solve": _SOLVES, "Norm": _SOLVES, "Dense": _SOLVES, "Affine": _SOLVES,
    "eliminate": _SOLVES, "Normalization": _SOLVES,
    "Scientific": _RUN, "Bitwise": _RUN, "Reproducible": _RUN, "Cache": _RUN,
    "Regrid": _HIERARCHY, "Level": _HIERARCHY, "invalidate": _HIERARCHY, "remap": _HIERARCHY,
    "restrict_onto": _HIERARCHY, "replace_contribution": _HIERARCHY,
    "CellTotal": _REPRESENT,
    "PointValue": _REPRESENT, "FaceNormalIntegral": _REPRESENT, "EdgeCirculation": _REPRESENT,
    "DualLoad": _REPRESENT, "Covector": _REPRESENT, "reconstruct": _REPRESENT,
    "Weak": _REPRESENT, "Pointwise": _REPRESENT, "Approximate": _REPRESENT,
    "Realizable": _REPRESENT,
    "product": _SUPPORTS, "Fibre": _SUPPORTS, "Line": _SUPPORTS, "Ball": _SUPPORTS,
    "Connectivity": _SUPPORTS, "Strided": _SUPPORTS, "Global": _SUPPORTS, "Family": _SUPPORTS,
    "Dimension": _SUPPORTS,
    "MovingConfiguration": "a later lot of N15 with the moving configurations of D18",
    "Swept": "a later lot of N15 with the moving configurations of D18",
    "jointly": "a later lot of N15 with the joint realizations of G05",
    "multistep": "a later lot of N8 (T27)", "substep": _N8_LATER, "try_substep": _N8_LATER,
    "exact_flow": _N8_LATER, "action": _N8_LATER, "predictor": _N8_LATER,
    "step_change": _N8_LATER, "step_duration": _N8_LATER, "step_start": _N8_LATER,
    "Controller": "N8 phase 2 (controllers)", "Fixed": "N8 phase 2 (controllers)",
    "Frozen": _N8_LATER, "Revolve": _N8_LATER, "recompute": _N8_LATER, "Off": _N8_LATER,
    "role": _LATER_N15, "parameter": _LATER_N15, "compose": _LATER_N15, "member": _LATER_N15,
    "Auxiliary": _LATER_N15, "explain": _LATER_N15,
    "claims": "a later lot of N15 with the claims of §13",
    "regions": "a later lot of N15 with the region level (SU15)",
    "schedules": _RUN, "external": _LATER_N15,
}

# Names of earlier texts (§21.7): the construct to write, as the message of SymbolNotInSurface.
_EARLIER = {
    "Rectangle": _SpellingRefusal("pops.Box(x=(0, 1), y=(0, 1)), one keyword per axis"),
    "Interval": _SpellingRefusal("pops.Box(x=(0, 1)), a box of one axis"),
    "Cartesian1D": _SpellingRefusal("pops.Box(x=...), whose named axes give the Cartesian frame"),
    "Cartesian2D": _SpellingRefusal("pops.Box(x=..., y=...), whose named axes give the Cartesian "
                                    "frame"),
    "Cartesian3D": _SpellingRefusal("pops.Box(x=..., y=..., z=...), whose named axes give the "
                                    "Cartesian frame"),
    "ConstParam": _SpellingRefusal('model.constant("g", value=...)'),
    "RuntimeParam": _SpellingRefusal('model.parameter("nu", default=...), or pops.parameter(...) '
                                     "for one that no relation reads"),
    "Space": _SpellingRefusal("pops.Scalar(Ω), pops.Vector(Ω) or pops.Components(Ω, ...), which "
                              "name the support first"),
    "Realize": _SpellingRefusal("pops.Numerics(schemes={term: scheme}), keyed by what the scheme "
                                "covers"),
    "Represent": _SpellingRefusal("pops.Numerics(represent={u: pops.CellAverage()})"),
    "Problem": _SpellingRefusal("pops.solve(relation, X) or pops.solve([relations], [X, Y])"),
    "Proposal": _SpellingRefusal("dt= of pops.Advance or pops.substep, and next_dt= of pops.Accept"
                                 " or pops.Reject"),
    "Retry": _SpellingRefusal("attempts=n and transforms=k on pops.Advance"),
    "Request": _SpellingRefusal("pops.Tolerance(rtol=, atol=, observe=, norm=)"),
    "Layout": _SpellingRefusal("layout={Ω: pops.Uniform(cells=...)} in pops.Case"),
    "layouts": _SpellingRefusal("pops.Uniform(cells=...) or pops.AMR(...) from pops"),
    "mesh": _SpellingRefusal("the layout in pops.Case(layout={Ω: pops.Uniform(cells=...)})"),
    "amr": _SpellingRefusal("pops.AMR(...) in layout= and pops.Regrid(...) in pops.Program"),
    "domain": _SpellingRefusal("pops.Box, pops.product or pops.Finite, which pops itself exports"),
    "spaces": _SpellingRefusal("pops.Scalar, pops.Vector or another space, which pops itself "
                               "exports"),
    "rel": _SpellingRefusal("a relation with == and the words of pops.math; pops.rel has no words"),
    "Target": _SpellingRefusal("target=pops.OpenMP() or target=pops.CUDA() on pops.compile"),
    "set_threads": _SpellingRefusal("resources=pops.Resources(threads=4) on pops.bind"),
    "validate": _SpellingRefusal("pops.inspect(case), which validates and resolves without "
                                 "compiling"),
    "resolve": _SpellingRefusal("pops.inspect(case), which validates and resolves without "
                                "compiling"),
    "space_ratio": _SpellingRefusal("level_ratio(Ω, l), or leave ratio= of subcycle out, which "
                                    "takes it"),
    "solvers": _SpellingRefusal("the solver, imported from pops.lib.solve by name"),
    "UnrealizedOutput": _SpellingRefusal("pops.CoverageNotExact, the refusal of a definition that "
                                         "holds a derivative or an integral of a discrete field "
                                         "and is read without a scheme"),
    "Realization": _SpellingRefusal("solvers={key: solver} in pops.Program, keyed by handles"),
    "from_data": _SpellingRefusal("initial={u: pops.project(e, quadrature=...)}, or initial={u: "
                                  "u0} for a datum"),
    "integrate_rates": _SpellingRefusal("the state arithmetic, step.state + step.dt * "
                                        "rate(step.state, step.time)"),
    "apply_increment": _SpellingRefusal("S + J, a state plus an increment"),
    "accept_all": _SpellingRefusal("accept=[check1, check2]; a list of checks accepts when all of "
                                   "them do"),
    "cfl_guard": _SpellingRefusal("dt=cfl_limit(advection, cfl=..., max_dt=...) in "
                                  "pops.Advance, master's AdaptiveCFL and its formula"),
    "step_bound": _SpellingRefusal("the bound as dt=cfl_limit(flux, cfl=...) or as dt= a function "
                                   "of cell, since accept= holds checks only"),
}

_LIBRARY = "pops.lib.numerics, pops.lib.time.methods and pops.lib.quadrature"


def _later(name: str, owner: str) -> _Any:
    def later(*args: _Any, **kwargs: _Any) -> _Any:
        raise NotYetElaborated("pops.%s" % name, owner).with_traceback(None)

    later.__name__ = later.__qualname__ = name
    return later


for _name, _owner in _LATER.items():
    globals()[_name] = _later(_name, _owner)

# @pops.solver and the words of its body (N9 lot 4, SV9). pops.choose and pops.fail are
# elaborated inside a solver only; elsewhere they refuse with the lot that owns them. pops.record is
# the record of a solver inside one and the record of the run (N15 lot G) elsewhere.
_solver.OUTSIDE_OWNERS.update({"pops.choose": _N8_LATER, "pops.fail": _N8_ASSESS})


def __getattr__(name: str) -> _Any:
    """Refuse a name that the surface does not have (SU3, SU13, §21.7); export nothing."""
    if name.startswith("_"):
        raise AttributeError(name)
    if name in _EARLIER:
        raise _EARLIER[name].refuse("pops.%s" % name, "the surface")
    raise SymbolNotInSurface(
        "pops.%s is not a word of the surface; pops has %s, and the library %s"
        % (name, ", ".join(sorted(__all__)), _LIBRARY),
        write="a word of 2.6.2 to 2.6.4 or 2.6.8, or of the library of 2.6.9")


__all__ = [
    "AMR", "Advance", "Box", "CUDA", "Candidate", "Case", "CellAverage", "Components",
    "Model", "Numerics", "OpenMP", "Output", "Periodic", "Positive", "Program", "Quadrature",
    "Resources", "Scalar", "Tensor", "Tolerance", "Uniform", "Vector", "bind", "claim", "compile",
    "every", "inspect", "project", "record", "discrete", "math", "method", "scheme", "solve",
    "unknown",
    # @pops.solver and the words of its body (N9 lot 4); pops.fail takes the variants of T39
    # listed with the failures of a run below
    "PropertyNotDerived", "choose", "fail", "solver",
    # the refusals of §14.1 that this lot raises, and the values that the library reads
    "AmbiguousElaboration", "AntiDiffusive", "ClosedBodyViolation", "ContradictsDerivation",
    "CoverageNotExact", "IncrementNotDerivable", "IncrementTypeMismatch", "LevelMismatch",
    "MissingBoundaryRealization", "NotAPopsOperation", "NotDiffusive", "NotYetElaborated",
    "OutOfScope", "RealizationMismatch", "ReservedName", "SupportMismatch", "SurfaceRefusal",
    "SymbolNotInSurface", "SymbolicTruthValue", "TermCountedTwice", "UnknownName",
    "AmbiguousSolvePath", "UnknownSolvePath",
    # the failures of a run (T39), lot G, which a solver also fails with (N9 lot 4)
    "BudgetExhausted", "RunFailed", "SolveFailed",
    "Choice", "Report", "Row", "SchemeValue", "StepRuleValue", "TermView",
    "check_plain_function",
]
