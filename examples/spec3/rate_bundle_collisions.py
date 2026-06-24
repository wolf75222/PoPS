"""Spec 3 multi-output operators: a RateBundle of arbitrary arity.

A coupled operator (a collision term) returns one typed rate per participating
species; the arity is not limited to two. RateBundle.require enforces that each
block's rate lives over its own StateSpace -- a wrong rate on a wrong state is
rejected.

Run: python3 examples/spec3/rate_bundle_collisions.py
"""
from adc import model


def build():
    e = model.StateSpace("electron_state", ["ne", "mex", "mey"])
    i = model.StateSpace("ion_state", ["ni", "mix", "miy"])
    n = model.StateSpace("neutral_state", ["nn", "mnx", "mny"])

    # an "electron-ion-neutral collision" operator returning three typed rates
    collisions = model.RateBundle({
        "electrons": model.Rate(e),
        "ions": model.Rate(i),
        "neutrals": model.Rate(n),
    })
    return collisions, (e, i, n)


if __name__ == "__main__":
    collisions, (e, i, n) = build()
    print("RateBundle arity:", len(collisions), "->", collisions.keys())
    for block, state in (("electrons", e), ("ions", i), ("neutrals", n)):
        print("  %-10s -> %r (ok)" % (block, collisions.require(block, state)))
    try:
        collisions.require("electrons", i)  # wrong StateSpace
    except TypeError as exc:
        print("rejected wrong rate on wrong state:", exc)
