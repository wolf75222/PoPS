"""Selected/unselected Python faces must differ at the true AMR interface."""

import numpy as np
import pops
import pytest

from examples.api_v02.method_amr_reception import composite_base_projection
from examples.api_v02.method_flux_correction_amr import build_t12_amr
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import profile_gather_value
from tests.python.support.relaxed_flux_receipts import fragments
from tests.python.support.stencil_selection_oracle import inspect_selection

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _levels(runtime, world):
    return [
        collective_call(
            world, lambda level=level: runtime.block_level_state_global("method", level)
        )
        for level in range(runtime.n_levels())
    ]


def _inspect(initial, actual, boxes, payload, *, cells, duration, selected):
    if initial[0] is None:
        return None
    metrics = inspect_selection(
        initial[0],
        actual[0],
        boxes,
        fragments(payload),
        cells=cells,
        duration=duration,
        selected=selected,
    )
    old = composite_base_projection(initial, boxes, cells=cells)["means"]
    final = composite_base_projection(actual, boxes, cells=cells)["means"]
    metrics["mass_defect"] = float(abs(final.mean() - old.mean()))
    metrics["coarse"] = np.asarray(actual[0]).reshape(cells[1], cells[0]).tolist()
    metrics["uncovered"] = metrics["uncovered"].tolist()
    metrics["fine_integrals"] = {
        f"{x}:{y}": value for (x, y), value in metrics["fine_integrals"].items()
    }
    return metrics


def test_selected_interface_payload_and_reflux_differ_from_unselected(
    isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, results, boxes, before = (32, 16), [], None, None
    for selected in (False, True):
        spec, artifact, world = _compile(
            "T12_interface_" + str(selected),
            builder=build_t12_amr,
            cells=cells,
            selected=selected,
            steps=1,
        )
        runtime = _bind(spec, artifact, world)
        initial = _levels(runtime, world)
        current_boxes = runtime.patch_boxes()
        with collective_check(world):
            assert runtime.n_levels() == 2
            if boxes is not None:
                assert current_boxes == boxes
                for left, right in zip(initial, before, strict=True):
                    if left is not None:
                        np.testing.assert_array_equal(left, right)
        boxes, before = current_boxes, initial
        report = collective_call(
            world,
            lambda runtime=runtime, duration=spec.dt: pops.run(
                runtime, t_end=duration, max_steps=1, console=False
            ),
        )
        actual = _levels(runtime, world)
        payload = collective_call(
            world, lambda runtime=runtime: bytes(runtime._executor.program_accepted_state())
        )
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert runtime.time() == spec.dt and runtime.macro_step() == 1

        gathered = profile_gather_value(
            world,
            collective_call(
                world,
                lambda initial=initial, actual=actual, boxes=boxes, payload=payload, duration=spec.dt, selected=selected: (
                    _inspect(
                        initial,
                        actual,
                        boxes,
                        payload,
                        cells=cells,
                        duration=duration,
                        selected=selected,
                    )
                ),
            ),
        )
        with collective_check(world):
            owners = [row for row in gathered if row is not None]
            assert owners and all(row["mass_defect"] < 2e-12 for row in owners)
        results.append(owners[0])
        record_property("artifact_" + str(selected), artifact.artifact_identity.hexdigest)
    with collective_check(world):
        baseline, corrected = results
        assert set(baseline["fine_integrals"]) == set(corrected["fine_integrals"])
        integral_gap = max(
            abs(value - baseline["fine_integrals"][key])
            for key, value in corrected["fine_integrals"].items()
        )
        state_gap = float(
            np.max(
                abs(np.asarray(corrected["coarse"]) - np.asarray(baseline["coarse"]))[
                    np.asarray(corrected["uncovered"])
                ]
            )
        )
        assert integral_gap > 1e-10, "the accepted fine interface did not observe the correction"
        assert state_gap > 1e-6, "selected fine faces did not change the uncovered coarse recipient"
    record_property("selected_interface_faces", corrected["selected_interface_faces"])
    record_property("uncovered_state_error", corrected["uncovered_state_error"])
    record_property("fine_integral_selection_gap", integral_gap)
    record_property("uncovered_selection_gap", state_gap)
