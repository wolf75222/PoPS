"""Version 1: immutable explicit finite DOF maps, evaluated in native kernels."""
from dataclasses import dataclass
import math
from pops._ir.expr import _wrap


@dataclass(frozen=True)
class FiniteSupport:
    name: str
    dofs: tuple[str, ...]

    def __post_init__(self):
        if type(self.name) is not str or not self.name:
            raise ValueError("finite support requires a nonempty name")
        object.__setattr__(self, "dofs", tuple(self.dofs))
        if not self.dofs or any(type(x) is not str or not x for x in self.dofs):
            raise ValueError("finite support requires explicit nonempty DOF labels")
        if len(set(self.dofs)) != len(self.dofs):
            raise ValueError("finite support DOF labels must be unique")

    @property
    def contract(self):
        return (self.name, self.dofs)

    def bind(self, values):
        from pops.time.values import ProgramValue
        from pops.time.expressions import as_expression, component_names
        if isinstance(values, ProgramValue):
            if component_names(values) != self.dofs:
                raise ValueError("Program components differ from the ordered finite support labels")
            values = as_expression(values).components
        return FiniteVector(self, tuple(values))


@dataclass(frozen=True)
class FiniteVector:
    support: FiniteSupport
    components: tuple

    def __post_init__(self):
        if not isinstance(self.support, FiniteSupport):
            raise TypeError("finite vector requires a FiniteSupport")
        object.__setattr__(self, "components", tuple(_wrap(x) for x in self.components))
        if len(self.components) != len(self.support.dofs):
            raise ValueError("finite vector width differs from its support")

    def materialize(self, program, name, *, template, at=None):
        from pops.time.expressions import ProgramExpression, component_names
        if component_names(template) != self.support.dofs:
            raise ValueError("finite output template differs from its ordered support")
        expression = ProgramExpression(self.components, template)
        return program._pointwise_expression(name, expression, at=at, finite_support=self.support)

    def __len__(self): return len(self.components)
    def __iter__(self): return iter(self.components)
    def __getitem__(self, index): return self.components[index]

    def _binary(self, other, op):
        if not isinstance(other, FiniteVector) or self.support != other.support:
            raise ValueError("finite vector operation requires the exact same ordered support")
        return self.support.bind(op(a, b) for a, b in zip(self, other, strict=True))

    def __add__(self, other): return self._binary(other, lambda a, b: a+b)
    def __sub__(self, other): return self._binary(other, lambda a, b: a-b)
    def __neg__(self): return self.support.bind(-x for x in self)
    def __mul__(self, scalar): return self.support.bind(x*_wrap(scalar) for x in self)
    def __rmul__(self, scalar): return self*scalar


@dataclass(frozen=True)
class FiniteLinearMap:
    source: FiniteSupport
    target: FiniteSupport
    coefficients: tuple

    def __post_init__(self):
        if not isinstance(self.source, FiniteSupport) or not isinstance(self.target, FiniteSupport):
            raise TypeError("finite map requires exact source and target supports")
        rows = tuple(tuple(row) for row in self.coefficients)
        if len(rows) != len(self.target.dofs) or any(len(r) != len(self.source.dofs) for r in rows):
            raise ValueError("finite map coefficients differ from source/target shape")
        if any(type(x) not in (int, float) or not math.isfinite(x) for r in rows for x in r):
            raise ValueError("finite map coefficients must be finite numeric constants")
        object.__setattr__(self, "coefficients", rows)

    def _evaluate(self, value, operation):
        from pops._ir.finite_linear import FiniteApplication, FiniteProjection
        expected = self.source if operation == "apply" else self.target
        output = self.target if operation == "apply" else self.source
        if not isinstance(value, FiniteVector) or value.support != expected:
            raise ValueError("finite map argument differs from its exact ordered support")
        call = FiniteApplication(operation, self.source.contract, self.target.contract,
                                 self.coefficients, value.components)
        return output.bind(FiniteProjection(call, i) for i in range(len(output.dofs)))

    def apply(self, value): return self._evaluate(value, "apply")

    def solve(self, rhs):
        if len(self.source.dofs) != len(self.target.dofs):
            raise ValueError("finite solve requires a square map; invertibility is checked natively")
        return self._evaluate(rhs, "solve")


@dataclass(frozen=True)
class FiniteMeasure:
    """Positive finite quadrature measure on one ordered finite support."""

    support: FiniteSupport
    weights: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.support, FiniteSupport):
            raise TypeError("finite measure requires an exact FiniteSupport")
        weights = tuple(self.weights)
        if len(weights) != len(self.support.dofs) or any(
                type(weight) not in (int, float) or not math.isfinite(weight)
                or not weight > 0 for weight in weights):
            raise ValueError("finite measure requires one strictly positive finite weight per DOF")
        object.__setattr__(self, "weights", weights)

    def pair(self, left: FiniteVector, right: FiniteVector):
        """Return the exact authored weighted pairing, with no spatial-grid reduction."""
        if not isinstance(left, FiniteVector) or not isinstance(right, FiniteVector) \
                or left.support != self.support or right.support != self.support:
            raise ValueError("finite pairing requires the measure's exact ordered support")
        total = _wrap(0)
        for weight, x, y in zip(self.weights, left, right, strict=True):
            total = total + _wrap(weight) * x * y
        return total


@dataclass(frozen=True)
class FiniteSymmetricInteraction:
    """Measured self-adjoint finite kernel and its quadratic energy.

    The native action is the generic FiniteLinearMap with entries W_ij*m_j.
    Exact symmetry is checked on represented coefficients; no tolerance or
    silent symmetrization changes the authored interaction.
    """

    measure: FiniteMeasure
    kernel: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if not isinstance(self.measure, FiniteMeasure):
            raise TypeError("finite symmetric interaction requires a FiniteMeasure")
        rows = tuple(tuple(row) for row in self.kernel)
        width = len(self.measure.support.dofs)
        if len(rows) != width or any(len(row) != width for row in rows):
            raise ValueError("finite interaction kernel differs from its measured support")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for row in rows for value in row):
            raise ValueError("finite interaction kernel requires finite numeric constants")
        if any(rows[i][j] != rows[j][i] for i in range(width) for j in range(i)):
            raise ValueError("finite interaction kernel must be exactly symmetric")
        object.__setattr__(self, "kernel", rows)

    @property
    def map(self) -> FiniteLinearMap:
        support = self.measure.support
        return FiniteLinearMap(support, support, tuple(
            tuple(value * weight for value, weight in zip(row, self.measure.weights,
                                                           strict=True))
            for row in self.kernel))

    def apply(self, density: FiniteVector) -> FiniteVector:
        return self.map.apply(density)

    def adjoint(self, value: FiniteVector) -> FiniteVector:
        """Weighted adjoint, equal to the action by the certified symmetry."""
        return self.map.apply(value)

    def energy(self, density: FiniteVector):
        return _wrap(0.5) * self.measure.pair(density, self.apply(density))

    def directional_derivative(self, density: FiniteVector, direction: FiniteVector):
        return self.measure.pair(direction, self.apply(density))
