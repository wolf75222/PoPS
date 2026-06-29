"""Spec 3: the board facade and the operator-first kernel share ONE IR.

Builds the same forward-Euler step two ways -- the board sugar (T.fields / T.define /
T.commit) and the explicit operator-first builder (P.solve_fields / P.call /
P.linear_combine / P.commit) -- and asserts the two Program IRs are identical.
This is the anti-duplication guarantee: the facade only generates the Spec 2 IR.

Run: python3 examples/spec3/board_vs_operator_ir_equivalence.py
"""
from pops.time import Program
from pops import model


def _ir(P):
    idx = {id(v): k for k, v in enumerate(P._values)}
    return [(v.vtype, v.op, tuple(idx[id(i)] for i in v.inputs),
             repr(sorted(v.attrs.items())), v.block) for v in P._values]


def _module():
    mod = model.Module("fe_ops")
    U = mod.state_space("U", ("rho", "mx", "my"))
    fields = mod.field_space("fields", ("phi",))
    fields_op = mod.operator(name="fields", signature=(U,) >> fields,
                             kind="field_operator", expr="rho")
    rate_op = mod.rate_operator("explicit_rate", state_space="U", flux=True, sources=["default"])
    return mod, U, fields_op, rate_op


def board():
    mod, U_space, _fields_op, _rate_op = _module()
    T = Program("fe_board").bind_operators(mod)
    dt = T.dt
    u = T.state("plasma", space=U_space)
    T.fields("f", from_state=u, operator="fields")
    r = T.op("explicit_rate")(u, value_name="R")
    u1 = T.define("U1", u + dt * r)
    T.commit("plasma", u1)
    return T


def operator_first():
    mod, U_space, fields_op, rate_op = _module()
    P = Program("fe_operator_first").bind_operators(mod)
    dt = P.dt
    u = P.state("plasma", space=U_space)
    P.call(fields_op, u, name="f")
    r = P.call(rate_op, u, name="R")
    u1 = P.linear_combine("U1", u + dt * r)
    P.commit("plasma", u1)
    return P


if __name__ == "__main__":
    T = board()
    same = _ir(T) == _ir(operator_first())
    print("board IR == operator-first IR:", same)
    print()
    print(T.dump_operator_ir())
    print()
    print(T.dump_cpp_plan())
    assert same, "the board facade must generate the same IR as the operator-first kernel"
