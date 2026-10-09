"""Independent checks of actual Original diagnostics and persisted state bytes."""
import math
import re
import numpy as np


def accepted_triplets(diagnostics, backend, convergence):
    """No legacy max(1,ref): admission is exactly the declared typed policy."""
    prefix, count = ("field_residual", 2) if backend == "uniform" else ("amr_original_field", 1)
    members = {"residual_norm", "reference_residual_norm", "rel_residual"}
    groups = {}
    for key, value in diagnostics.items():
        match = re.fullmatch(prefix + r"_([0-9]+)\.(residual_norm|reference_residual_norm|rel_residual)", key)
        if match:
            assert type(value) is float and math.isfinite(value) and value >= 0, (key, value)
            groups.setdefault(int(match[1]), {})[match[2]] = value
    assert len(groups) == count, groups
    relative = float.fromhex(convergence["relative"]["value"])
    absolute = float.fromhex(convergence["absolute"]["value"])
    rows = []
    for identifier, row in sorted(groups.items()):
        assert set(row) == members
        norm, reference, raw = (row[k] for k in ("residual_norm", "reference_residual_norm", "rel_residual"))
        assert raw == norm / (reference if reference > 0 else 1.), row
        cutoff = absolute if convergence["kind"] == "absolute" else max(absolute, relative * reference)
        assert math.isfinite(cutoff) and norm <= cutoff, (row, cutoff)
        rows.append(dict(node_id=identifier, **row, cutoff=cutoff))
    return rows


def same_bits(lhs, rhs):
    lhs, rhs = np.asarray(lhs), np.asarray(rhs)
    assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape
    assert lhs.tobytes(order="C") == rhs.tobytes(order="C")


def uniform_original_norms(solutions, loads, coefficient, residual):
    """Recompute full Original F from saved physical arrays, not diagnostic strings."""
    references, norms = [], []
    for index, solution in enumerate(solutions):
        seed = np.zeros_like(solution) if index == 0 else .8 * solutions[0]
        references.append(float(np.linalg.norm(residual(seed, coefficient) - loads)))
        norms.append(float(np.linalg.norm(residual(solution, coefficient) - loads)))
    return references, norms
