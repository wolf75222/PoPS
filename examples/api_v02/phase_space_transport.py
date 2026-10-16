"""Tensor-product finite-volume algebra on genuine distributed index sets.

Every run-time operation below builds a normal PoPS Program node.  Configuration
and velocity coordinates are indices, never Model components.  Flattening is
last-axis-fastest and preserves neighbours, measures, fibres and MPI imports.
The Python loops elaborate dimensions; the C++ graph traverses the cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod

from pops.algebra import Distributed


@dataclass(frozen=True)
class TensorGrid:
    shape: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]
    boundaries: tuple[str, ...]

    def __post_init__(self):
        if not self.shape or any(type(n) is not int or n < 2 for n in self.shape):
            raise ValueError("a tensor grid requires at least two cells on each axis")
        if len(self.bounds) != len(self.shape) or len(self.boundaries) != len(self.shape):
            raise ValueError("every tensor axis needs bounds and a boundary rule")
        if any(not isfinite(a) or not isfinite(b) or not a < b for a, b in self.bounds):
            raise ValueError("finite increasing bounds are required")
        if any(rule not in ("periodic", "zero_flux") for rule in self.boundaries):
            raise ValueError("declare periodic or zero_flux on every axis")

    @property
    def size(self):
        return prod(self.shape)

    @property
    def strides(self):
        return tuple(prod(self.shape[k + 1 :]) for k in range(len(self.shape)))

    @property
    def widths(self):
        return tuple((b - a) / n for (a, b), n in zip(self.bounds, self.shape, strict=True))

    @property
    def volume(self):
        return prod(self.widths)

    def ordinal(self, gid, axis):
        return (gid // self.strides[axis]) % self.shape[axis]

    def coordinate(self, gid, axis):
        ordinal = self.ordinal(gid, axis)
        if hasattr(ordinal, "prog"):
            ordinal = ordinal.prog.float64(ordinal)
        return self.bounds[axis][0] + (ordinal + 0.5) * self.widths[axis]

    def periodic_neighbour(self, gid, axis, direction):
        coordinate = self.ordinal(gid, axis)
        target = (coordinate + self.shape[axis] + direction) % self.shape[axis]
        return gid + (target - coordinate) * self.strides[axis]

    def product(self, other):
        return TensorGrid(
            self.shape + other.shape, self.bounds + other.bounds, self.boundaries + other.boundaries
        )


def partition(count, ranks):
    if type(ranks) is not int or ranks < 1:
        raise ValueError("a positive MPI rank count is required")
    return tuple(count * rank // ranks for rank in range(ranks + 1))


class DistributedTensor:
    """Authoring helper for one explicitly partitioned tensor index domain."""

    def __init__(self, program, grid, domain, *, byte_capacity=64 << 20):
        self.program, self.grid, self.domain = program, grid, domain
        self.byte_capacity = byte_capacity

    @classmethod
    def create(cls, program, grid, ranks, *, byte_capacity=64 << 20):
        domain = Distributed(program).domain(partition(grid.size, ranks), capacity=grid.size)
        return cls(program, grid, domain, byte_capacity=byte_capacity)

    def rebind(self, program, domain=None):
        return DistributedTensor(
            program,
            self.grid,
            self.domain if domain is None else domain,
            byte_capacity=self.byte_capacity,
        )

    def map(self, body):
        p = self.program
        return p.require_success(
            Distributed(p).local_map(
                self.domain,
                lambda b, i: b.capture(lambda c: body(c, Distributed(c).gid(i), i)),
                byte_capacity=self.byte_capacity,
            )
        )

    def zip(self, arrays, body):
        return self.map(
            lambda b, gid, i: body(
                b, gid, tuple(Distributed(b).at_local(array, i) for array in arrays)
            )
        )

    def constant(self, value):
        return self.map(lambda b, gid, i: b.float64(value))

    def sum(self, values):
        p = self.program
        return Distributed(p).reduce(
            values,
            p.float64(0),
            lambda b, s, x: s + x,
            order="global_gid_ascending",
            byte_capacity=self.byte_capacity,
        )

    def dot(self, left, right):
        return self.sum(self.zip((left, right), lambda b, gid, v: v[0] * v[1]))

    def axpy(self, left, scale, right):
        return self.zip((left, right), lambda b, gid, v: v[0] + scale * v[1])

    def shift(self, values, axis, direction):
        """Import the neighbour including neighbours owned by a different rank.

        A zero-flux boundary is closed by the flux formula, not by hiding a
        physical extrapolation in this indexing operation. The wrapped value
        is imported but an inactive boundary branch never consumes its flux.
        """
        p = self.program
        requests = Distributed(p).requests(
            self.domain,
            self.domain,
            lambda b, i: Distributed(b).index(
                self.domain, self.grid.periodic_neighbour(Distributed(b).gid(i), axis, direction)
            ),
        )
        return Distributed(p).import_values(values, requests, byte_capacity=self.byte_capacity)

    def gradient(self, values, axis):
        left, right = self.shift(values, axis, -1), self.shift(values, axis, 1)
        return self.zip(
            (left, right), lambda b, gid, v: (v[1] - v[0]) / (2 * self.grid.widths[axis])
        )

    def negative_laplacian(self, values):
        result = self.constant(0)
        for axis, width in enumerate(self.grid.widths):
            left, right = self.shift(values, axis, -1), self.shift(values, axis, 1)
            part = self.zip(
                (left, values, right),
                lambda b, gid, v, width=width: (2 * v[1] - v[0] - v[2]) / width**2,
            )
            result = self.axpy(result, 1, part)
        return result

    def divergence(self, values, axis, flux):
        """One shared oriented face formula with telescoping FV divergence.

        flux(builder, left_gid, left, right) is an algebra builder.  Its gid
        identifies the left cell. It never runs during numerical execution.
        """
        right = self.shift(values, axis, 1)

        def face(b, gid, v):
            if self.grid.boundaries[axis] == "periodic":
                return flux(b, gid, v[0], v[1])
            return b.branch(
                self.grid.ordinal(gid, axis) == self.grid.shape[axis] - 1,
                lambda c: c.float64(0),
                lambda c: flux(c, gid, v[0], v[1]),
            )

        faces = self.zip((values, right), face)
        left_faces = self.shift(faces, axis, -1)
        return self.zip(
            (faces, left_faces), lambda b, gid, v: (v[0] - v[1]) / self.grid.widths[axis]
        )


def fibre_sum(phase, configuration, values):
    """Integrate the velocity fibre using physical velocity-cell measures."""
    p = phase.program
    velocity_size = phase.grid.size // configuration.grid.size
    weight = phase.grid.volume / configuration.grid.volume
    weighted = phase.zip((values,), lambda b, gid, v: weight * v[0])
    requests = Distributed(p).requests(
        phase.domain,
        configuration.domain,
        lambda b, i: Distributed(b).index(
            configuration.domain, Distributed(b).gid(i) // velocity_size
        ),
    )
    return Distributed(p).scatter_reduce(
        weighted,
        requests,
        configuration.domain,
        p.float64(0),
        lambda b, total, value: total + value,
        order="global_gid_ascending",
        byte_capacity=phase.byte_capacity,
    )


def lift(configuration, phase, values):
    p = phase.program
    velocity_size = phase.grid.size // configuration.grid.size
    requests = Distributed(p).requests(
        phase.domain,
        configuration.domain,
        lambda b, i: Distributed(b).index(
            configuration.domain, Distributed(b).gid(i) // velocity_size
        ),
    )
    return Distributed(p).import_values(values, requests, byte_capacity=phase.byte_capacity)


def poisson_cg(space, rhs, *, tolerance=1e-11, max_steps=None):
    """Mean-zero periodic -Delta solve, with original residual re-evaluated.

    The removed mean is returned separately: the caller must check compatibility
    with its prescribed background, rather than silently changing its physics.
    No precomputed inverse or model-specific native operator is involved.
    """
    p = space.program
    mean = space.sum(rhs) / space.grid.size
    compatible = space.zip((rhs,), lambda b, gid, v: v[0] - mean)
    initial = p.product(x=space.constant(0), r=compatible, direction=compatible)

    def step(b, state):
        local = space.rebind(b)
        r, direction = state["r"], state["direction"]
        applied = local.negative_laplacian(direction)
        rr = local.dot(r, r)
        alpha = rr / local.dot(direction, applied)
        x = local.axpy(state["x"], alpha, direction)
        updated = local.axpy(r, -alpha, applied)
        beta = local.dot(updated, updated) / rr
        return b.product(x=x, r=updated, direction=local.axpy(updated, beta, direction))

    def stop(b, state):
        local = space.rebind(b)
        return local.dot(state["r"], state["r"]) <= tolerance**2

    solved = p.require_success(
        p.iterate(
            initial, step, stop, max_steps=space.grid.size + 2 if max_steps is None else max_steps
        )
    )["state"]["x"]
    residual = space.axpy(rhs, -1, space.negative_laplacian(solved))
    return solved, space.dot(residual, residual), mean
