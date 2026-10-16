"""Independent integration of native cut-cell observer pieces, including replicas."""
import numpy as np
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.spatial_interaction_receipts import _allgather_array_values


def _dense(copies, key, shape):
    result = np.full(shape, np.nan)
    for copy in copies:
        for piece in copy[key]:
            lo, hi = piece["lower"], piece["upper"]
            region = tuple(slice(lo[a], hi[a]) for a in range(2))
            # Native observer bounds and arrays both use (y, x), with an
            # additional leading component axis on values.
            value = np.asarray(piece["values"])[0]
            previous = result[region]
            occupied = np.isfinite(previous)
            np.testing.assert_array_equal(previous[occupied], value[occupied])
            result[region] = value
    return result


def snapshot(runtime, world, *, cells=(16, 12)):
    states, weights, unweighted, fractions = [], [], [], []
    for level in range(runtime.n_levels()):
        scale = 2**level
        shape = (cells[0]*scale, cells[1]*scale)
        spacing = tuple(1./n for n in shape)
        native = runtime._executor
        geometry = collective_call(world, lambda native=native, level=level, spacing=spacing, shape=shape: native._output_geometry_snapshot(
            level, (0., 0.), spacing, shape, (2, 2) if level+1 < runtime.n_levels() else (0, 0),
            "pops://cell-measures/cartesian-area@1"))
        local = {key: collective_call(world, lambda name=name, native=native, level=level:
            native.output_embedded_boundary_local_pieces(name, level))
            for key, name in (("active", "pops_active"), ("fraction", "pops_kappa"))}
        copies = _allgather_array_values(world, local)
        with collective_check(world):
            active, fraction = (_dense(copies, key, shape[::-1]) for key in ("active", "fraction"))
            valid = np.asarray(geometry["valid_cells"], dtype=bool)
            owned = valid & ~np.asarray(geometry["coverage"], dtype=bool) & (active > .5)
            assert np.all(np.isfinite(fraction[valid]))
            assert np.all((fraction[valid] >= 0) & (fraction[valid] <= 1))
            measure = np.where(owned, np.asarray(geometry["cell_volumes"]), 0.)
            weights.append(np.where(owned, measure*fraction, 0.))
            unweighted.append(measure)
            fractions.append(fraction)
        raw = collective_call(world, lambda level=level: runtime.block_level_state_global("method", level))
        states.append(None if raw is None else np.asarray(raw).reshape(3, shape[1], shape[0]))
    return {"states": states, "weights": weights, "unweighted": unweighted,
            "fractions": fractions, "time": runtime.time(), "macro": runtime.macro_step()}


def pair(left, right, weights):
    return sum(float(np.sum(np.where(w > 0, (a[0]*b[0]+a[1]*b[1])*w, 0.)))
               for a, b, w in zip(left, right, weights, strict=True))


def factor(old, candidate, weights):
    first = pair(old, old, weights)
    cross = pair(old, candidate, weights)
    final = pair(candidate, candidate, weights)
    return -2*(cross-first)/(final-2*cross+first)
