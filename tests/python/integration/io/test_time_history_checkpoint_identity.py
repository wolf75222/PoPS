#!/usr/bin/env python3
"""Section (C) of test_time_history_checkpoint.py, in its own process: the Program-identity guard.

A Uniform AB2 checkpoint is refused, before any state mutation, by a Forward Euler artifact (its
exact live resource budget) and by an equal-footprint AB2 artifact of another Program identity (the
canonical identity guard). The section moved here unchanged so that its three cold compiles and the
two of the parent file each fit the 300-second process limit (B03 F5).
"""
from tests.python.integration.io.test_time_history_checkpoint import _pops_time, _run_section_c


def test_uniform_restart_refuses_a_different_compiled_program():
    """Collect the authenticated Uniform Program-identity guard as an ordinary test."""
    _run_section_c(_pops_time())


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print("PASS test_time_history_checkpoint_identity (%d checks)" % len(fns))


if __name__ == "__main__":
    _run()
