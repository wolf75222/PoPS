"""pops.lib.time.ssprk -- operator-first SSPRK2 / SSPRK3 schemes."""

from ._helpers import _stage_rate


def ssprk2(P, block, *, rhs_operator, fields_operator=None):
    """SSPRK2 (Heun / Shu-Osher): U1 = U0 + dt k0; U^{n+1} = 1/2 U0 + 1/2 (U1 + dt k1)."""
    U0 = P._state_value(block)
    k0 = _stage_rate(P, U0, rhs_operator=rhs_operator, fields_operator=fields_operator, tag="ssprk2_0_")
    U1 = P.linear_combine("ssprk2_U1", U0 + P.dt * k0)
    k1 = _stage_rate(P, U1, rhs_operator=rhs_operator, fields_operator=fields_operator, tag="ssprk2_1_")
    P.commit(block, P.linear_combine("ssprk2_step", 0.5 * U0 + 0.5 * (U1 + P.dt * k1)))


def ssprk3(P, block, *, rhs_operator, fields_operator=None):
    """SSPRK3 (Shu-Osher): U1 = U0 + dt k0; U2 = 3/4 U0 + 1/4 (U1 + dt k1);
    U^{n+1} = 1/3 U0 + 2/3 (U2 + dt k2)."""
    U0 = P._state_value(block)
    k0 = _stage_rate(P, U0, rhs_operator=rhs_operator, fields_operator=fields_operator, tag="ssprk3_0_")
    U1 = P.linear_combine("ssprk3_U1", U0 + P.dt * k0)
    k1 = _stage_rate(P, U1, rhs_operator=rhs_operator, fields_operator=fields_operator, tag="ssprk3_1_")
    U2 = P.linear_combine("ssprk3_U2", 0.75 * U0 + 0.25 * (U1 + P.dt * k1))
    k2 = _stage_rate(P, U2, rhs_operator=rhs_operator, fields_operator=fields_operator, tag="ssprk3_2_")
    P.commit(block, P.linear_combine("ssprk3_step", (1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + P.dt * k2)))
