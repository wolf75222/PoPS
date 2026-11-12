#!/usr/bin/env python3
"""ROM-107: the native face and the Python face of M01 and M03 under AdaptiveCFL.

The api_v02 builders author their SSPRK2 under FixedDt. This script builds the same cases, their
two stages written as the builders' b6 Program (``_ssprk2``), with the step strategy replaced by
``AdaptiveCFL(cfl=--cfl)`` (rebound in the builders' module while they build), compiles each
route, binds the same initial data and runs to ``--steps`` times the builder's step (the default
lets AdaptiveCFL take about a dozen steps on 16 x 8 cells). For each route it reports the
accepted steps, the attempts that a retry lost (ROM-108), the final clock, the reason of a refused
attempt, the final state against the native route (bitwise equality, L1 and L-infinity
differences), and the strategy that each compiled Program carries.

``--rule incident_sum`` builds the Python faces with the guard of the base, before ROM-107: the
face frequency sum_a s_a / h_a against cfl (``StepStrategy.face_frequency_rule``), which is the
code the base generated for these cases. The script measures; it asserts nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", choices=("M01", "M03"))
    parser.add_argument("--cells", default="16x8", help="NXxNY")
    parser.add_argument("--cfl", type=float, default=.45)
    parser.add_argument("--steps", type=int, default=200,
                        help="final time in builder steps (AdaptiveCFL chooses its own)")
    parser.add_argument("--rule", choices=("strategy", "incident_sum"), default="strategy")
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


@contextlib.contextmanager
def _adaptive(cases, cfl):
    from pops.time import AdaptiveCFL

    # M01 and M03 author their SSPRK2 through the traced Advance (_traced_ssprk2), whose
    # elaboration runs FixedDt only; their b6 reference, _ssprk2, writes the same two stages as a
    # Program and reads FixedDt from the builders' module, which is rebound here.
    fixed, traced = cases.FixedDt, cases._traced_ssprk2
    cases.FixedDt = lambda dt: AdaptiveCFL(cfl=cfl, max_dt=None)
    cases._traced_ssprk2 = lambda case, block, state, flow, dt, *, name: cases._ssprk2(
        case, block, state, lambda program, value, point, label: flow(value), dt, name=name)
    try:
        yield
    finally:
        cases.FixedDt, cases._traced_ssprk2 = fixed, traced


@contextlib.contextmanager
def _rule(rule):
    from pops.time import AdaptiveCFL

    if rule == "strategy":
        yield
        return
    previous = AdaptiveCFL.face_frequency_rule
    AdaptiveCFL.face_frequency_rule = rule
    try:
        yield
    finally:
        AdaptiveCFL.face_frequency_rule = previous


def _revision():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    args = _arguments()
    import numpy as np
    import pops
    from examples.api_v02 import models_M01_M14 as cases

    builder = {"M01": cases.build_m01, "M03": cases.build_m03}[args.case]
    cells = tuple(int(count) for count in args.cells.split("x"))
    routes = {"native_rusanov": cases.native_rusanov, "python_rusanov": cases.rusanov}
    results, states = {}, {}
    for route, scheme in routes.items():
        with _adaptive(cases, args.cfl), _rule(args.rule if route.startswith("python") else
                                               "strategy"):
            spec = builder(cells=cells, scheme=scheme)
            plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
            artifact = pops.compile(plan)
        runtime = pops.bind(artifact, initial_values=dict(spec.initial_values))
        refusal = None
        try:
            report = pops.run(runtime, t_end=args.steps*spec.dt, max_steps=args.steps,
                              console=False)
            steps, rejected = report.accepted_steps, report.rejected_steps
        except Exception as error:  # the refusal of the base is the measurement
            refusal, steps = "%s: %s" % (type(error).__name__, str(error).splitlines()[0]), None
            rejected = None
        results[route] = {"accepted_steps": steps, "rejected_attempts": rejected,
                          "macro_step": int(runtime.macro_step()),
                          "time": float(runtime.time()), "refusal": refusal,
                          "strategy": type(spec.case._time._step_strategy).__name__}
        states[route] = np.asarray(runtime.state_global("fluid"), dtype=float)
    reference = states["native_rusanov"]
    difference = states["python_rusanov"] - reference
    results["parity"] = {"bitwise_equal": bool(np.array_equal(states["python_rusanov"],
                                                              reference)),
                         "l1": float(np.mean(np.abs(difference))),
                         "linf": float(np.max(np.abs(difference)))}
    results["setup"] = {"case": args.case, "cells": args.cells, "cfl": args.cfl,
                        "t_end_builder_steps": args.steps, "python_rule": args.rule,
                        "revision": _revision()}
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
