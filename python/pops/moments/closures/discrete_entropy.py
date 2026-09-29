"""Finite positive-quadrature minimum-entropy equations.

This is authoring algebra.  ``populations`` and ``residual`` return ordinary
``pops.math`` expressions, which a public ``LocalResidual`` lowers into its
native cell solve.  No Python callback is retained in a cell kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pops.math import exp


@dataclass(frozen=True)
class DiscreteEntropyQuadrature:
    """Positive weights and a full-rank moment basis sampled at distinct nodes."""

    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    basis: tuple[tuple[float, ...], ...]

    def __init__(self, nodes: Any, weights: Any, basis: Any) -> None:
        nodes = tuple(nodes)
        weights = tuple(weights)
        basis = tuple(tuple(row) for row in basis)
        if not nodes or len(nodes) != len(weights) or len(set(nodes)) != len(nodes):
            raise ValueError("entropy quadrature requires distinct nodes with matching weights")
        if not basis or any(len(row) != len(nodes) for row in basis):
            raise ValueError("entropy basis must supply every moment at every node")
        if len(basis) > len(nodes):
            raise ValueError("entropy basis cannot have more moments than nodes")
        for item in (*nodes, *weights, *(item for row in basis for item in row)):
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError("entropy quadrature requires finite real coefficients")
        if any(weight <= 0 for weight in weights):
            raise ValueError("entropy quadrature weights must be strictly positive")
        # A rank-deficient basis yields a singular dual Hessian for every population.
        # Detect it at authoring, before a native solve or its publication exists.
        import numpy as np
        if np.linalg.matrix_rank(np.asarray(basis, dtype=float)) != len(basis):
            raise ValueError("entropy moment basis must have full row rank")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "basis", basis)

    @property
    def moment_count(self) -> int:
        return len(self.basis)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def populations(self, multipliers: Any) -> tuple[Any, ...]:
        coefficients = tuple(multipliers)
        if len(coefficients) != self.moment_count:
            raise ValueError("entropy multipliers must match the moment basis")
        return tuple(
            weight * exp(sum(row[index] * coefficients[component]
                             for component, row in enumerate(self.basis)))
            for index, weight in enumerate(self.weights)
        )

    def moments(self, populations: Any) -> tuple[Any, ...]:
        values = tuple(populations)
        if len(values) != self.node_count:
            raise ValueError("entropy populations must match the quadrature nodes")
        return tuple(sum(coefficient * value for coefficient, value in zip(row, values, strict=True))
                     for row in self.basis)

    def residual(self, multipliers: Any, target: Any) -> tuple[Any, ...]:
        prescribed = tuple(target)
        if len(prescribed) != self.moment_count:
            raise ValueError("entropy target must match the moment basis")
        return tuple(reconstructed - original for reconstructed, original in
                     zip(self.moments(self.populations(multipliers)), prescribed, strict=True))


__all__ = ["DiscreteEntropyQuadrature"]
