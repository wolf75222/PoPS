"""Algebraic authoring on the PoPS compiler and native Program runtime.

This namespace deliberately reuses PoPS values, ownership, regions, code generation and
execution. It has no interpreter, execution backend or registry of physical models.
The implementation profile is experimental: exporting a constructor does not assert
conformance to every representation and effect in the API 0.2 specification.

``Program.product/fold/iterate/match`` construct typed native regions. Scientific equations
use the same scalar nodes as fluxes and finite operators. ``Realize`` and ``Increment``
retain the actual SSA values used by an update, so their equations are executable authority.
"""
from pops.time import Program
from pops.time._program.finite_arrays import IndexFamily
from pops.math import (
    Expr, Const, Var, ValueExpr, where, sqrt, exp, expm1, log, sin, cos,
    minimum, maximum, grad, div, laplacian, ddt, integral,
)
from pops.realization import Use, Realize, Realized, require_coverage, require_continuation_coverage
from pops.contributions import Contribution, Delivery, Increment, combine, bdf2_increment, FiniteInterval, LinearMap
from pops.finite_contributions import InventoryErrors
from pops.bundles import BundleSchema, Remap, remap
from pops.intervals import Interval, HistoryInterval
from pops.derivatives import jvp, vjp, Pairings
from pops.graph_derivatives import GraphPairings, tangent_type
from pops.fibres import FiberAxis, FiberRepresentation
from pops.numerics.stencil_flux import StencilFlux
from pops.graph_functions import GraphFunction, graph, call
from pops.time.variant_types import (
    ValueType, Bool, Int64, Float64, Product, Variant, Result, RuntimeFailure, Failure, type_of,
)
from pops.representations import Representation, ScientificType, ScientificValue, ScientificMap
from pops.claims import Claim, claims_manifest
from pops.relations import Problem, Algorithm, Accuracy, Evidence, Approximation, Check, SolveFailure, RelationErrors, solve

from pops.actions import Action, ActionFailure, ActionErrors

from pops.implicit_derivatives import ImplicitRelation, Linearization, linearize, implicit_jvp, implicit_vjp

from pops.continuations import ContinuationSchema, Candidate, Assess, DecisionAssess, assess, accept, Advance, advance_once, advance
from pops.time._program.distributed_arrays import Distributed

IMPLEMENTATION_PROFILE = "0.2-experimental"

__all__ = [
    "Program", "IndexFamily", "Expr", "Const", "Var", "ValueExpr", "where", "sqrt", "exp",
    "expm1", "log", "sin", "cos", "minimum", "maximum", "grad", "div",
    "laplacian", "ddt", "integral", "Use", "Realize", "Realized", "require_coverage",
    "Contribution", "Delivery", "Increment", "combine", "bdf2_increment", "Interval", "HistoryInterval",
    "FiniteInterval", "LinearMap", "InventoryErrors", "BundleSchema", "Remap", "remap", "require_continuation_coverage",
    "jvp", "vjp", "Pairings", "GraphPairings", "tangent_type",
    "FiberAxis", "FiberRepresentation", "StencilFlux",
    "GraphFunction", "graph", "call", "ValueType", "Bool", "Int64", "Float64",
    "Product", "Variant", "Result", "RuntimeFailure", "Failure", "type_of",
    "Representation", "ScientificType", "ScientificValue", "ScientificMap", "Claim", "claims_manifest",
    "Problem", "Algorithm", "Accuracy", "Evidence", "Approximation", "Check", "SolveFailure", "RelationErrors", "solve",
    "Action", "ActionFailure", "ActionErrors",
    "ImplicitRelation", "Linearization", "linearize", "implicit_jvp", "implicit_vjp",
    "ContinuationSchema", "Candidate", "Assess", "DecisionAssess", "assess", "accept", "Advance", "advance_once", "advance", "Distributed",
    "IMPLEMENTATION_PROFILE",
]
