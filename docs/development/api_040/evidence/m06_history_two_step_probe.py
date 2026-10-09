#!/usr/bin/env python3
"""Inspect authentic M06 ring slots after one and two accepted Uniform steps."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


EXPECTED_NATIVE = "6b5f452e432976a94b69635b96aad1a04c42c8d5e933cad5cf785301089488fb"


def main() -> None:
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes

    example_dir = Path(__file__).resolve().parents[4] / "examples" / "migration" / "scientific"
    sys.path.insert(0, str(example_dir))
    import api040_m06_enthalpy as m06

    pops.set_threads(1)
    identity = m06._installed_identity()
    if identity["native_sha256"] != EXPECTED_NATIVE:
        raise RuntimeError("M06 probe loaded a different native artifact: %r" % identity)
    frame = Rectangle("M06_history_probe", (0., 0.), (1., 1.)).frame(Cartesian2D())
    case, declaration = m06.build_case(frame, name="M06_history_probe")
    validated = pops.validate(case)
    param = validated.resolve(declaration)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4),
                                   periodic=PeriodicAxes(frame.axes)))
    artifact = pops.compile(pops.resolve(validated, layout=layout))
    initial = np.full((1, 4, 4), 1.5)
    expected = {
        "sensible": {"input_rate": .25, "H": (1.53125, 1.5625),
                     "T": (.765625, .78125), "f": (0., 0.)},
        "plateau": {"input_rate": 4., "H": (2., 2.5),
                    "T": (1., 1.), "f": (0., 1 / 6)},
    }
    trajectories = {}
    for label, target_values in expected.items():
        runtime = pops.bind(
            artifact, initial_state={"material": initial},
            params={param: target_values["input_rate"]},
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)})
        if runtime.history_depth("M06_temperature") != 2:
            raise AssertionError("depth=1 should allocate current and previous slots")
        observations = []
        for step, target in enumerate((.125, .25)):
            report = pops.run(runtime, t_end=target, max_steps=1, console=False)
            observed = {
                "H": np.asarray(runtime.state_global("material")).reshape(1, 4, 4),
                "T0": np.asarray(runtime.history_global("M06_temperature", 0)).reshape(1, 4, 4),
                "T1": np.asarray(runtime.history_global("M06_temperature", 1)).reshape(1, 4, 4),
                "f0": np.asarray(runtime.history_global("M06_liquid_fraction", 0)).reshape(1, 4, 4),
                "f1": np.asarray(runtime.history_global("M06_liquid_fraction", 1)).reshape(1, 4, 4),
            }
            for field, expected_value in (
                ("H", target_values["H"][step]), ("T1", target_values["T"][step]),
                ("f1", target_values["f"][step]),
                ("T0", target_values["T"][0 if step == 0 else step - 1]),
                ("f0", target_values["f"][0 if step == 0 else step - 1]),
            ):
                np.testing.assert_allclose(observed[field], expected_value, rtol=0, atol=1e-14)
            observations.append({"time": runtime.time(), "accepted_steps": report.accepted_steps,
                                 **{name: float(values[0, 0, 0])
                                    for name, values in observed.items()}})
        trajectories[label] = observations
    print(json.dumps({"native": identity, "trajectories": trajectories,
                      "latest_accepted_slot": 1}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
