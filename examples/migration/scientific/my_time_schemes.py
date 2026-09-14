"""User-authored time methods: importing this module does not register a C++ preset."""

from fractions import Fraction
import pops
from pops.time import FixedDt


def ssprk2(state, rate, *, dt):
    """Build the explicit two-stage Shu--Osher method with ordinary Program values."""
    program = pops.Program("user_ssprk2")
    q = program.state(state)
    stage_0 = program.stage("stage_0", c=0)
    k0 = program.value("k0", rate(q.n), at=stage_0)
    stage_1 = program.stage("stage_1", c=1)
    q1 = program.value("q1", q.n + program.dt * k0, at=stage_1)
    k1 = program.value("k1", rate(q1), at=stage_1)
    half = Fraction(1, 2)
    advanced = program.value(
        "advanced",
        q.n + program.dt * half * k0 + program.dt * half * k1,
        at=q.next.point,
    )
    program.commit(q.next, advanced)
    program.step_strategy(FixedDt(dt))
    return program
