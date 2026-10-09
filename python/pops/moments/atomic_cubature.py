"""Explicit finite-atomic raw-moment closure, composed at Python authoring time."""
from fractions import Fraction
from math import isfinite


class AtomicCubature:
    """Unisolvent nodes and monomials; no positivity repair or implicit closure.

    Rational inversion is performed once when constructing the declaration.
    Evaluation expands ordinary scalar arithmetic in the supplied storage order,
    including PoPS expressions. It never calls Python for individual cells.
    Coefficients are converted to finite binary64 explicitly; summation follows
    node order and does not claim compensated or bit-exact alternative algebra.
    """
    def __init__(self, *, indices, nodes):
        indices = tuple(tuple(i) for i in indices)
        nodes = tuple(tuple(v) for v in nodes)
        if not indices or not indices[0]:
            raise ValueError("nonempty monomial basis required")
        dim = len(indices[0])
        if any(len(i) != dim or any(type(p) is not int or p < 0 for p in i) for i in indices):
            raise ValueError("indices require nonnegative integer exponents of one rank")
        if len(set(indices)) != len(indices) or len(nodes) != len(indices):
            raise ValueError("distinct indices and square cubature required")
        if any(len(v) != dim or any(type(x) not in (int, float, Fraction) for x in v) for v in nodes):
            raise ValueError("nodes require finite real coordinates of the basis rank")
        try:
            nodes = tuple(tuple(Fraction(x) for x in v) for v in nodes)
        except (ValueError, OverflowError) as error:
            raise ValueError("nodes require finite coordinates") from error
        n = len(indices)
        rows = [[self._monomial(v, i) for v in nodes] + [Fraction(r == c) for c in range(n)]
                for r, i in enumerate(indices)]
        for c in range(n):
            pivot = next((r for r in range(c, n) if rows[r][c]), None)
            if pivot is None:
                raise ValueError("cubature monomials are not unisolvent")
            rows[c], rows[pivot] = rows[pivot], rows[c]
            divisor = rows[c][c]
            rows[c] = [x / divisor for x in rows[c]]
            for r in range(n):
                if r != c:
                    factor = rows[r][c]
                    rows[r] = [x - factor*y for x, y in zip(rows[r], rows[c])]
        self.indices, self.nodes = indices, nodes
        self.inverse = tuple(tuple(row[n:]) for row in rows)

    @staticmethod
    def _monomial(node, index):
        result = Fraction(1)
        for x, p in zip(node, index):
            result *= x**p
        return result

    @staticmethod
    def _coefficient(value):
        try:
            result = float(value)
        except OverflowError as error:
            raise ValueError("cubature coefficient exceeds binary64") from error
        if not isfinite(result):
            raise ValueError("cubature coefficient exceeds binary64")
        return result

    def weights(self, moments):
        """Recover signed weights; positivity is a separate physical admission."""
        if len(moments) != len(self.indices):
            raise ValueError("moment width differs from declared basis")
        return tuple(sum((self._coefficient(c)*m for c, m in zip(row, moments)), 0)
                     for row in self.inverse)

    def moment(self, moments, index):
        """Evaluate an explicitly requested monomial under this atomic measure."""
        index = tuple(index)
        if len(index) != len(self.indices[0]) or any(type(p) is not int or p < 0 for p in index):
            raise ValueError("requested monomial has invalid rank or exponent")
        return sum((self._coefficient(self._monomial(v, index))*w
                    for v, w in zip(self.nodes, self.weights(moments))), 0)
