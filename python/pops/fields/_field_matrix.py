"""Exact matrix proofs for a selected elliptic numerical realization."""
from fractions import Fraction


def inverse(matrix, *, where):
    n = len(matrix)
    rows = [[Fraction(x) for x in row] + [Fraction(i == j) for j in range(n)]
            for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if rows[i][column]), None)
        if pivot is None:
            raise ValueError(where)
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [x / scale for x in rows[column]]
        for i in range(n):
            if i != column:
                scale = rows[i][column]
                rows[i] = [a - scale * b for a, b in zip(rows[i], rows[column])]
    return tuple(tuple(row[n:]) for row in rows)


def positive_definite(matrix):
    """Exact LDL proof: no tolerance turns a negative or zero pivot positive."""
    n = len(matrix)
    if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n)):
        return False
    work = [[Fraction(x) for x in row] for row in matrix]
    for k in range(n):
        pivot = work[k][k]
        if pivot <= 0:
            return False
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                work[i][j] -= work[i][k] * work[k][j] / pivot
    return True


def gauge_modes(gauge, size):
    from .problem import ConstantModeGauge
    if gauge is None:
        return (), ()
    if isinstance(gauge, ConstantModeGauge):
        return tuple(tuple(Fraction(x) for x in row) for row in gauge.modes), tuple(Fraction(x) for x in gauge.values)
    return ((Fraction(1),) * size,), (Fraction(gauge.value),)
