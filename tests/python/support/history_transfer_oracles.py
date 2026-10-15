"""Independent exact moment oracle for the quadratic AMR history reception."""
from fractions import Fraction


def quadratic_moment_weights(times, target):
    """Solve the three moment constraints, independently of authored Lagrange products."""
    times = tuple(map(Fraction, times))
    target = Fraction(target)
    assert len(times) == 3 and len(set(times)) == 3
    matrix = [[time**power for time in times] + [target**power] for power in range(3)]
    for column in range(3):
        pivot = next(row for row in range(column, 3) if matrix[row][column])
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value/divisor for value in matrix[column]]
        for row in range(3):
            if row != column:
                scale = matrix[row][column]
                matrix[row] = [a-scale*b for a, b in zip(matrix[row], matrix[column], strict=True)]
    return tuple(matrix[row][-1] for row in range(3))


def assert_projected_quadratic_flux(parent_slots, child_slots, source_times, targets):
    """Authenticate every projected source and compare its full exact polynomial."""
    assert len(parent_slots) == len(source_times) == len(child_slots) == len(targets) == 3
    sources = {}
    for column, slot in enumerate(parent_slots):
        assert slot
        for basis in slot:
            snapshot = basis["snapshot"]
            assert snapshot["kind"] == 1 and snapshot["level"] == 0
            key = snapshot["identity"]
            assert key not in sources, "source snapshots must distinguish recycled local basis IDs"
            sources[key] = (column, basis)
    assert len(sources) == 3, "this declared RHS has one physical face evaluation per source slot"
    for slot, target in zip(child_slots, targets, strict=True):
        weights = quadratic_moment_weights(source_times, target)
        assert all(sum(weight*Fraction(time)**power for weight, time in
                       zip(weights, source_times, strict=True)) == Fraction(target)**power
                   for power in range(3))
        expected = {}
        for key, (column, parent) in sources.items():
            polynomial = [(power, coefficient*weights[column]) for power, coefficient in parent["weights"]
                          if coefficient*weights[column]]
            if polynomial:
                expected[key] = polynomial
        actual = {}
        assert len({basis["identity"] for basis in slot}) == len(slot)
        for basis in slot:
            snapshot = basis["snapshot"]
            assert basis["level"] == snapshot["level"] == 1 and snapshot["kind"] == 2
            key = snapshot["parent"]["identity"]
            assert key in sources and key not in actual
            _, parent = sources[key]
            assert snapshot["parent"] == parent["snapshot"]
            assert basis["runtime_block"] == parent["runtime_block"]
            actual[key] = basis["weights"]
        assert actual == expected, "projected flux differs from the exact quadratic moment matrix"
