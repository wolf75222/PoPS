"""Spec 3 board-like time program, and proof it lowers to the primitive IR.

Builds a Heun/Crank-Nicolson-style predictor-corrector with the blackboard sugar
(T.fields / T.define / T.solve / T.commit) and with the equivalent primitive calls
(solve_fields / linear_combine / solve_local_linear / commit), then checks the two
Program IRs are identical -- the board notation is sugar, not a new IR.

Run: python3 examples/spec3/board_time_predictor_corrector.py
"""
from pops.time import Program
from pops.math import unknown
from pops import model


def _ir(P):
    idx = {id(v): k for k, v in enumerate(P._values)}
    return [(v.vtype, v.op, tuple(idx[id(i)] for i in v.inputs),
             repr(sorted(v.attrs.items())), v.block) for v in P._values]


def _module():
    mod = model.Module("pc_ops")
    U = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    fields_op = mod.operator(name="fields", signature=(U,) >> fields,
                             kind="field_operator", expr="rho")
    rate_op = mod.rate_operator("explicit_rate", state_space="U", flux=True, sources=["default"])
    implicit_op = mod.operator(
        name="lorentz", signature=(fields,) >> model.LocalLinearOperator(U, U),
        kind="local_linear_operator", expr="B_z")
    return mod, U, fields_op, rate_op, implicit_op


def board():
    mod, U_space, fields_op, _rate_op, _implicit_op = _module()
    T = Program("pc_board").bind_operators(mod)
    dt = T.dt
    u_n = T.state("plasma", space=U_space)
    f_n = T.call(fields_op, u_n, name="fields_n")
    r_n = T.op("explicit_rate")(u_n, value_name="R_n")
    L_n = T.op("lorentz")(f_n, value_name="L_n")
    u_star = T.solve(
        "U_star",
        (T.I - dt * L_n) @ unknown("U_star") == u_n + dt * r_n,
    )
    T.commit("plasma", u_star)
    return T


def primitive():
    mod, U_space, fields_op, rate_op, implicit_op = _module()
    P = Program("pc_primitive").bind_operators(mod)
    dt = P.dt
    u_n = P.state("U", block="plasma", space=U_space).n
    f_n = P.call(fields_op, u_n, name="fields_n")
    r_n = P.call(rate_op, u_n, name="R_n")
    L_n = P.call(implicit_op, f_n, name="L_n")
    op = P.I - dt * L_n
    rhs = P.linear_combine("U_star_rhs", u_n + dt * r_n)
    u_star = P.solve_local_linear(name="U_star", operator=op, rhs=rhs)
    P.commit("plasma", u_star)
    return P


if __name__ == "__main__":
    same = _ir(board()) == _ir(primitive())
    print("board IR == primitive IR:", same)
    assert same, "board sugar must lower to the same IR as the primitive calls"
    print("OK: blackboard notation is sugar over the operator-first IR")
