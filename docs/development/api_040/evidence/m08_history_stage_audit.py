#!/usr/bin/env python3
"""Read an existing M08 state archive and identify its saved stage-0 time."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] /
                           "examples" / "migration" / "scientific"))
    from api040_m08_oracle import ssprk2_reference

    with np.load(archive) as saved:
        dt, final_time = float(saved["dt"]), float(saved["time"])
        steps = round(final_time / dt)
        if steps != 2 or not np.isclose(final_time, steps * dt, rtol=0, atol=1e-14):
            raise ValueError("the discriminant requires an authentic two-step M08 archive")
        initial_q = saved["initial_q"]
        initial_c = saved["initial_c"]
        observed_stage0_q = saved["stage0_q"]
        reference_start_of_final_step, _ = ssprk2_reference(
            initial_q, initial_c, dt=dt, steps=steps - 1)
        result = {
            "archive": str(archive),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "final_time": final_time,
            "step_dt": dt,
            "steps": steps,
            "expected_stage0_time": (steps - 1) * dt,
            "saved_stage0_vs_initial_max": float(np.max(np.abs(
                observed_stage0_q - initial_q))),
            "saved_stage0_vs_final_step_oracle_max": float(np.max(np.abs(
                observed_stage0_q - reference_start_of_final_step))),
            "oracle_final_step_vs_initial_max": float(np.max(np.abs(
                reference_start_of_final_step - initial_q))),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
