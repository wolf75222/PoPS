"""Full refinement witness for the two actual accepted transport/diffusion fluxes."""
import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_public_diffusion_matrix import (
    REFINEMENTS, _bind, _build, _centers, _record,
)

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _rates(value, speeds, nu):
    n = value.shape[0]
    transport = np.zeros_like(value)
    diffusion = np.zeros_like(value)
    for axis, speed in enumerate(speeds):
        storage_axis = 1-axis
        previous = np.roll(value, 1, storage_axis)
        following = np.roll(value, -1, storage_axis)
        transport -= n*speed*((value-previous) if speed >= 0 else (following-value))
        diffusion += nu*n*n*(previous-2*value+following)
    return transport, diffusion


def _advance(value, speeds, nu, dt, method):
    first = _rates(value, speeds, nu)
    predictor = value+dt*sum(first)
    if method == "euler":
        return predictor, (value,), (first,)
    last = _rates(predictor, speeds, nu)
    return .5*value+.5*(predictor+dt*sum(last)), (value, predictor), (first, last)


@pytest.mark.parametrize("method", ("euler", "ssprk2"))
def test_full_combined_accepted_face_quadrature(isolated_native_cache, native_cxx, kokkos_root, method):
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    speeds, nu = (.7, -.4), .1
    for n in REFINEMENTS:
        x, y = _centers(n)
        initial = 2+.3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+.1*np.cos(4*np.pi*x)
        dt = .15/(4*nu*n*n+sum(map(abs, speeds))*n)
        previous, _, _ = _advance(initial, speeds, nu, dt, method)
        expected, stage_states, stage_rates = _advance(previous, speeds, nu, dt, method)
        case, layout, _ = _build("constant", n, dt, method=method, transport=speeds)
        runtime, artifact = _bind(case, layout, initial, {})
        report = pops.run(runtime, t_end=2*dt, max_steps=2, console=False)
        assert report.accepted_steps == 2 and report.rejected_steps == 0
        actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        records = runtime._executor._program_exchange_records()
        stages = len(stage_states)
        assert len(records) == 8*n*n*stages
        contexts = tuple(dict.fromkeys(row["evaluation_context"] for row in records))
        assert len(contexts) == stages
        context_stage = dict(zip(contexts, range(stages), strict=True))
        occurrences = {}
        increments = {"transport": np.zeros_like(initial), "diffusion": np.zeros_like(initial)}
        flux_defect = 0.
        for row in records:
            cell_token, axis_token, side_token = row["quadrature_identity"].split("/")
            i, j = map(int, cell_token.split(":")[1:])
            axis, side = int(axis_token.split(":")[1]), int(side_token.split(":")[1])
            kind = "transport" if row["orientation"] == (1 if side == 0 else -1) else "diffusion"
            occurrences.setdefault(kind, set()).add(row["occurrence_identity"])
            assert row["face_measure"] == 1/n and row["multiplicity"] == 1
            assert abs(row["temporal_weight"]-dt/stages) <= 2e-16
            state = stage_states[context_stage[row["evaluation_context"]]]
            cell = (j, i)
            other = list(cell)
            other[1-axis] = (other[1-axis]+(1 if side else -1)) % n
            neighbor = state[tuple(other)]
            left, right = (state[cell], neighbor) if side else (neighbor, state[cell])
            if kind == "transport":
                speed = speeds[axis]
                oracle = .5*speed*(left+right)-.5*abs(speed)*(right-left)
            else:
                oracle = nu*n*(right-left)
            flux_defect = max(flux_defect, abs(row["numerical_flux"]-oracle))
            increments[kind][cell] += row["integrated_amount"]
        assert set(occurrences) == {"transport", "diffusion"}
        assert all(len(values) == 1 for values in occurrences.values())
        assert occurrences["transport"].isdisjoint(occurrences["diffusion"])
        assert flux_defect < 2e-12
        for index, kind in enumerate(("transport", "diffusion")):
            oracle = dt/stages*sum(rate[index] for rate in stage_rates)/n**2
            np.testing.assert_allclose(increments[kind], oracle, rtol=0, atol=2e-13)
        delta = increments["transport"]+increments["diffusion"]
        np.testing.assert_allclose((actual-previous)/n**2, delta, rtol=0, atol=2e-13)
        rows.append({"n": n, "method": method, "accepted_steps": report.accepted_steps,
                     "records": len(records), "contexts": len(contexts),
                     "face_flux_defect": flux_defect,
                     "cell_change_defect": float(np.max(np.abs((actual-previous)/n**2-delta))),
                     "artifact": artifact.artifact_identity.token})
    _record("combined-accepted-exchanges-"+method, rows)
