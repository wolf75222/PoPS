"""Finite tensor fibres authored as equations on existing native cell components.

This module allocates no storage and executes no time step. Its maps return the
ordinary PoPS scalar expressions; the existing Program/flux compiler emits them.
The physical cell layout is separate from the explicitly recorded fibre axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isfinite, prod
from typing import Any

from pops._ir.quantity import PhysicalDimension
from pops.representations import Representation


@dataclass(frozen=True)
class FiberAxis:
    name: str
    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    units: PhysicalDimension
    boundary: str
    edges: tuple[float, ...] | None = None

    def __post_init__(self):
        nodes, weights = tuple(map(float, self.nodes)), tuple(map(float, self.weights))
        if type(self.name) is not str or not self.name:
            raise TypeError("fibre axis requires a nonempty name")
        if not isinstance(self.units, PhysicalDimension):
            raise TypeError("fibre coordinate units must be explicitly declared")
        if not nodes or len(nodes) != len(weights) or any(not isfinite(x) for x in (*nodes, *weights)):
            raise ValueError("fibre axis requires equally sized finite nodes and weights")
        if any(w <= 0 for w in weights) or any(a >= b for a, b in zip(nodes, nodes[1:])):
            raise ValueError("fibre nodes must increase and quadrature weights must be positive")
        if self.boundary not in ("periodic", "zero_flux"):
            raise ValueError("finite fibre boundary must explicitly be periodic or zero_flux")
        edges = None if self.edges is None else tuple(map(float, self.edges))
        if edges is not None:
            if len(edges) != len(nodes)+1 or any(not isfinite(x) for x in edges):
                raise ValueError("fibre control volumes require N+1 finite edges")
            if any(not edges[i] < nodes[i] < edges[i+1] for i in range(len(nodes))):
                raise ValueError("each fibre node must lie inside its control volume")
            if any(abs((b-a)-w) > 1e-12*max(1., abs(w)) for a,b,w in zip(edges, edges[1:], weights)):
                raise ValueError("finite-volume fibre weights must equal their cell measures")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "edges", edges)

    def to_data(self):
        return dict(name=self.name, nodes=list(self.nodes), weights=list(self.weights),
                    units=self.units.to_data(), boundary=self.boundary,
                    edges=None if self.edges is None else list(self.edges))


@dataclass(frozen=True)
class FiberRepresentation:
    axes: tuple[FiberAxis, ...]
    prefix: str = "f"

    def __post_init__(self):
        axes = tuple(self.axes)
        if not axes or any(type(axis) is not FiberAxis for axis in axes):
            raise TypeError("finite fibre requires explicitly typed axes")
        if len({axis.name for axis in axes}) != len(axes):
            raise ValueError("fibre axes must have unique names")
        if type(self.prefix) is not str or not self.prefix.isidentifier():
            raise ValueError("fibre component prefix must be an identifier")
        object.__setattr__(self, "axes", axes)

    @property
    def shape(self):
        return tuple(len(axis.nodes) for axis in self.axes)

    @property
    def size(self):
        return prod(self.shape)

    @property
    def indices(self):
        return tuple(product(*(range(n) for n in self.shape)))

    @property
    def components(self):
        return tuple(self.prefix+"_"+"_".join(map(str, index)) for index in self.indices)

    @property
    def representation(self):
        return Representation("conservative", self.to_data())

    def to_data(self):
        return dict(schema_version=1, kind="finite_tensor_fibre", axes=[axis.to_data() for axis in self.axes],
                    order="last_axis_fastest", indices=[list(index) for index in self.indices],
                    components=list(self.components), physical_cell_layout="external")

    def flat_index(self, index):
        index = tuple(index)
        if len(index) != len(self.axes) or any(type(k) is not int or not 0 <= k < n
                                             for k,n in zip(index, self.shape)):
            raise IndexError("fibre index lies outside its declared product")
        result = 0
        for k, n in zip(index, self.shape):
            result = result*n+k
        return result

    def multi_index(self, component):
        if type(component) is not int or not 0 <= component < self.size:
            raise IndexError("component lies outside its declared fibre")
        return self.indices[component]

    def neighbour(self, index, *, axis, offset):
        self.flat_index(index)
        if type(axis) is not int or not 0 <= axis < len(self.axes) or type(offset) is not int:
            raise ValueError("neighbour requires an axis ordinal and integer offset")
        out = list(index)
        out[axis] += offset
        n = self.shape[axis]
        if not 0 <= out[axis] < n:
            if self.axes[axis].boundary == "zero_flux":
                return None
            out[axis] %= n
        return tuple(out)

    def validate(self, state):
        space = state.space
        if (tuple(space.components) != self.components
                or Representation(space.representation, space.representation_data) != self.representation):
            raise ValueError("state has a different finite fibre representation")
        return state

    def values(self, state):
        self.validate(state)
        return tuple(state[k] for k in range(self.size))

    def moment(self, values, powers):
        values, powers = tuple(values), tuple(powers)
        if len(values) != self.size or len(powers) != len(self.axes) or any(
                type(k) is not int or k < 0 for k in powers):
            raise ValueError("moment requires exact fibre width and nonnegative multi-degree")
        return sum(value*prod(axis.weights[j]*axis.nodes[j]**power
                              for axis,j,power in zip(self.axes,index,powers))
                   for value,index in zip(values,self.indices))

    def pullback(self, scalar):
        """Explicit scalar field constant along every fibre coordinate."""
        return tuple(scalar for _ in self.indices)

    def flux_divergence(self, values, *, axis, flux):
        """FV divergence in one fibre axis, expanded into ordinary scalar algebra.

        ``flux(left,right,face_coordinate)`` runs during authoring only. Boundary
        flux is exactly zero or periodic as declared. No callback enters C++.
        """
        values = tuple(values)
        if len(values) != self.size or type(axis) is not int or not 0 <= axis < len(self.axes):
            raise ValueError("fibre divergence requires the exact width and one axis")
        selected = self.axes[axis]
        edges = selected.edges
        if edges is None:
            raise ValueError("finite-volume divergence requires explicit control-volume edges")
        faces = {}
        def face(index, direction):
            other = self.neighbour(index, axis=axis, offset=direction)
            if other is None:
                return 0.
            left,right = (index,other) if direction == 1 else (other,index)
            key = (left,right)
            if key not in faces:
                faces[key] = flux(values[self.flat_index(left)], values[self.flat_index(right)],
                                  edges[left[axis]+1])
            return faces[key]
        return tuple((face(index,1)-face(index,-1))/selected.weights[index[axis]]
                     for index in self.indices)


__all__ = ["FiberAxis", "FiberRepresentation"]
