"""Operator-first type system (Spec 2, phase S2-1).

This package defines the abstract spaces and typed operators that a model-free
``pops.time.Program`` composes:

* ``StateSpace`` -- a conservative/primitive state space (the components of ``U``);
* ``FieldSpace`` -- an auxiliary or solved-field space (e.g. ``phi, grad_x, grad_y``);
* ``RateSpace`` / ``Rate(U)`` -- the tangent of a ``StateSpace`` (``dU/dt``);
* ``LocalLinearOperator(U, U)`` / ``MatrixFreeOperator`` -- operator-valued types;
* ``Signature`` -- a typed ``(inputs) -> output`` contract;
* ``Operator`` and ``OperatorRegistry`` -- a named, typed, integer-id'd registry.

These types are a TYPED VIEW: they carry no numerics and no array data.
``pops.physics.Model`` and direct ``pops.model.Module`` authoring populate the
same typed operator registry; ``pops.time.Program`` calls those operators by
handle and the compiled problem lowers the graph to C++.

The package imports only the standard library so it can be exercised without the
compiled ``_pops`` extension.
"""
from .bundles import RateBundle
from .handles import OperatorHandle
from .module import Module
from .operators import (
    OPERATOR_KINDS,
    LocalLinearOperator,
    MatrixFreeOperator,
    Operator,
)
from .registry import OperatorRegistry
from .signatures import Signature
from .spaces import (
    AuxSpace,
    FieldSpace,
    ParameterSpace,
    Rate,
    RateSpace,
    Space,
    StateSpace,
)

__all__ = [
    "Space",
    "StateSpace",
    "FieldSpace",
    "RateSpace",
    "Rate",
    "LocalLinearOperator",
    "MatrixFreeOperator",
    "Signature",
    "Operator",
    "OperatorRegistry",
    "ParameterSpace",
    "AuxSpace",
    "Module",
    "RateBundle",
    "OperatorHandle",
    "OPERATOR_KINDS",
]
