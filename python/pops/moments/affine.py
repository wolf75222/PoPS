"""Polynomial push-forward of raw moments in explicit velocity coordinates."""
def affine_push_forward(moments, *, indices, matrix, offset):
    """Return raw moments of ``v' = matrix @ v + offset`` in ``indices`` order.

    ``moments[k]`` represents the integral of ``v**indices[k]`` against the
    original measure. Indices are distinct tuples of nonnegative Python integers
    of one fixed, positive dimension. Every multi-index of total degree at most
    the largest requested degree must be supplied: no missing moment is closed,
    interpolated, or inferred from a coefficient that happens to be zero.

    The matrix is square and the offset has the velocity dimension. Entries and
    moments may be numbers or ordinary PoPS scalar expressions, including physical
    declarations, selected Program components, and LocalResidual unknowns. This
    function builds their common expression DAG at authoring time; it has no cell
    callback, numerical opcode, density normalization, or realizability repair.

    The zero moment is returned unchanged, by identity. Other moments use a fixed
    polynomial expansion and a lexicographically ordered sum. This is not a
    compensated sum and does not promise bitwise equivalence with another
    expansion. Common expression finite checks, lazy branches, and explicit
    ``pops.math.rounded`` barriers apply without library-specific simplification.
    There is no imposed velocity dimension or moment-order ceiling.
    """
    indices = tuple(tuple(index) for index in indices)
    if not indices or not indices[0]:
        raise ValueError("affine_push_forward requires nonempty, positive-rank indices")
    dimension = len(indices[0])
    if any(len(index) != dimension or any(type(n) is not int or n < 0 for n in index)
           for index in indices):
        raise ValueError("affine_push_forward indices must have one rank and nonnegative integer entries")
    if len(set(indices)) != len(indices):
        raise ValueError("affine_push_forward indices must be distinct")
    matrix = tuple(tuple(row) for row in matrix)
    offset = tuple(offset)
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError("affine_push_forward matrix must be square with the index dimension")
    if len(offset) != dimension:
        raise ValueError("affine_push_forward offset must have the index dimension")
    try:
        width = len(moments)
    except TypeError:
        # Temporal values deliberately have no Python length. Their public
        # component selection checks each requested index below.
        width = None
    if width is not None and width != len(indices):
        raise ValueError("affine_push_forward moments and indices must have equal length")

    zero = (0,) * dimension
    positions = {index: k for k, index in enumerate(indices)}
    # Do not prune zero coefficients: closure must not depend on runtime values,
    # and symbolic comparisons must never become Python control flow.
    polynomials = {zero: {zero: 1}}

    def polynomial(index):
        # Build the same parent chain in post-order without using Python's call
        # stack. A descending complete basis must have the same realization as
        # an ascending one, including orders above the interpreter recursion cap.
        pending = [index]
        while pending:
            current = pending[-1]
            if current in polynomials:
                pending.pop()
                continue
            axis = max(i for i, power in enumerate(current) if power)
            parent = list(current)
            parent[axis] -= 1
            parent = tuple(parent)
            if parent not in polynomials:
                pending.append(parent)
                continue
            terms = {}

            def add(exponent, term, terms=terms):
                terms[exponent] = terms[exponent] + term if exponent in terms else term

            for exponent, coefficient in polynomials[parent].items():
                add(exponent, offset[axis] * coefficient)
                for coordinate in range(dimension):
                    raised = list(exponent)
                    raised[coordinate] += 1
                    add(tuple(raised), matrix[axis][coordinate] * coefficient)
            polynomials[current] = terms
            pending.pop()
        return polynomials[index]

    for index in indices:
        missing = polynomial(index).keys() - positions.keys()
        if missing:
            raise ValueError("affine_push_forward requires missing raw moments: " +
                             repr(tuple(sorted(missing))))
    raw = {index: moments[k] for index, k in positions.items()}
    result = []
    for index in indices:
        if index == zero:
            result.append(raw[zero])
            continue
        terms = [coefficient * raw[exponent]
                 for exponent, coefficient in sorted(polynomials[index].items())]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        result.append(total)
    return tuple(result)
