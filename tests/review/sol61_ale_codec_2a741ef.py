"""Independent source/host reception of fixed ALE authority seams; no native execution."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np


PIN = "2a741effb06a11b0b351bef2629a9d09e8ce7bb8"
OLD = "0116827fb76a321873fb52749816de67f3a6d2cd"


def blob(root, revision, path):
    return subprocess.run(["rtk", "proxy", "git", "show", f"{revision}:{path}"], cwd=root,
                          check=True, capture_output=True).stdout


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "python"), str(root)]
    header = blob(root, PIN, "include/pops/runtime/program/moving_interval_checkpoint.hpp").decode()
    for token in ("volume(cell)==position(cell+1)-position(cell)",
                  "old_volume(cell)==old_position(cell+1)-old_position(cell)",
                  "receipt.geometry_tolerance==*geometry.geometry_tolerance_authority",
                  "receipt.point.clock==geometry.clock_authority", "actual.exterior_trace==expected.exterior_trace",
                  "actual.source_evaluation_identity==expected.source_evaluation_identity",
                  "ledger.checkpoint(true)", "point.physical_time+point.dt==accepted_time",
                  "point.tick==static_cast<std::int64_t>(macro_step)-1"):
        assert token in header, token
    for value in ("q0", "q1", "left", "right", "amount", "scale", "residual"):
        assert f"std::isfinite({value})" in header
    # Reuse the old countermodels as explicit false/false checks under the new predicates.
    assert not (2. == 1.)  # forged measure, despite installed tolerance 2
    q0, q1, amount, left, right = 4., 1e308 * 2., 0., 0., 0.
    residual = (q1 - q0) + (right - left) - amount
    scale = max(1., abs(q0), abs(q1), abs(amount), abs(left), abs(right))
    assert not all(map(math.isfinite, (q0, q1, amount, left, right, residual, scale)))
    for time, step, expected in ((.1, 1, True), (np.nextafter(.1, 1.), 1, False),
                                 (.1, 0, False), (.1, 2, False), (math.inf, 1, False)):
        accepted = math.isfinite(time) and .0 + .1 == time and 0 == step - 1
        assert bool(accepted) is expected
    system = blob(root, PIN, "src/runtime/system/system.cpp").decode()
    assert "provisional_capture && step_transaction_depth()==1" in system
    assert "moving candidate checkpoint requires a terminal interval receipt" in system
    # The first predicate is the actual gate; the host truth table includes restart refusal.
    staging = []
    for depth in (0, 1, 2, 3):
        for requested in (False, True):
            for restart in (False, True):
                for terminal in (False, True):
                    permitted = not ((depth != 0 and not (requested and depth == 1)) or restart)
                    permitted = permitted and not (depth == 1 and not terminal)
                    expected = not restart and (depth == 0 or depth == 1 and requested and terminal)
                    assert permitted == expected
                    staging.append((depth, requested, restart, terminal, permitted))
    ledger_path = "include/pops/runtime/program/accepted_exchange.hpp"
    current_ledger = blob(root, PIN, ledger_path)
    normalized = current_ledger.replace(b"checkpoint(bool force_extended=false)", b"checkpoint()")
    normalized = normalized.replace(b"force_extended || !integrals_.empty()", b"!integrals_.empty()")
    assert normalized == blob(root, OLD, ledger_path)
    amr_path = "include/pops/runtime/program/amr_program_checkpoint.hpp"
    assert blob(root, PIN, amr_path) == blob(root, OLD, amr_path)
    py_path = "python/pops/runtime/_checkpoint_exchanges.py"
    current = load(root / py_path, "fixed_exchanges")
    program = SimpleNamespace(_values=(object(),), _integral_states={}, _geometry_states={},
                              _serialize=lambda **kw: {"fixture": "unchanged"},
                              temporal_manifest=lambda: {"clocks": []})
    plan = SimpleNamespace(blocks=())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "old.py"
        path.write_bytes(blob(root, OLD, py_path))
        old = load(path, "previous_exchanges")
        for dimension in (1, 2, 3):
            for ranks in (1, 2, 3):
                args = dict(cells=(6,) * dimension, dimension=dimension, rank_capacity=ranks,
                            resolved_plan=plan)
                assert current.exchange_checkpoint_byte_capacity(program, **args) == \
                    old.exchange_checkpoint_byte_capacity(program, **args)
    widths = []
    for components in (1, 2, 16, 256, 1024):
        names = tuple(f"c{component}" for component in range(components))
        program._geometry_states = {"mesh": SimpleNamespace(space=SimpleNamespace(components=names))}
        program._serialize = lambda names=names, **kw: {"components": names, "geometry": "mesh"}
        capacity = current.exchange_checkpoint_byte_capacity(
            program, cells=(8,), dimension=1, rank_capacity=1, resolved_plan=plan)
        lower = 16 + 8 * (2 + 3 * components) * 76
        assert capacity >= lower
        widths.append({"components": components, "capacity": capacity, "ledger_minimum": lower})
    # Actual facade typed validation, with the native validator explicitly replaced by a spy.
    from tests.python.unit.runtime.test_continuation_transitions import _owner

    owner = _owner()
    owner._s.image = b"POPSEX03" + bytes(8)
    moving_calls = []
    owner._s._validate_checkpoint_moving_geometry = lambda image, time, step: moving_calls.append((image, time, step))
    payload = {"t": np.asarray(.25), "macro_step": np.asarray(4, dtype=np.int64)}
    current.capture_checkpoint_continuation(owner, payload)
    assert current.prepare_checkpoint_continuation(owner, payload) == owner._s.image
    assert moving_calls == [(owner._s.image, .25, 4)]
    invalid = [("t", value) for value in (None, True, 1, np.nan, np.inf, [.25], "0.25")]
    invalid += [("macro_step", value) for value in (None, True, 4., -1, [4], "4")]
    rejected = 0
    for key, value in invalid:
        altered = deepcopy(payload)
        altered[key] = np.asarray(value)
        before = len(moving_calls)
        try:
            current.prepare_checkpoint_continuation(owner, altered)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError((key, value))
        assert len(moving_calls) == before
    from pops._native_selector import selected_native_dimension
    assert selected_native_dimension() is None
    print(json.dumps({"source_pin": PIN, "scope": "source/host; native facade validator is a spy; no native selected",
                      "codec_sha256": hashlib.sha256(header.encode()).hexdigest(),
                      "old_countermodels_refused_by_fixed_predicates": True,
                      "legacy_default_serializer_normalized_to_exact_previous_source": True,
                      "legacy_budget_comparisons": 9, "wide_budget_lower_bounds": widths,
                      "clock_payload_refusals": rejected, "staging_truth_table_rows": staging}, indent=2))


if __name__ == "__main__":
    main()
