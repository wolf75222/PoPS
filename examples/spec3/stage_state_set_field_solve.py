"""Spec 3: resolve a field solve from a coherent set of stage states (StageStateSet).

When several blocks are at DIFFERENT stages (e.g. electrons and ions already at U*,
neutrals still at U^n), a field solve must read an unambiguous version of each block.
``T.state_set`` packages that choice; ``T.fields(from_state_set=...)`` solves the
coupled fields from exactly those stage states (lowering to the multi-block
``solve_fields_from_blocks`` operator-first op). The alternative, ``from_states=[...]``,
is equivalent and lighter for a few blocks.

Run: python3 examples/spec3/stage_state_set_field_solve.py
"""
from pops.time import Program
from pops import model


def _module():
    mod = model.Module("multi_species_stage_ops")
    e = mod.state_space("electrons", ("ne", "mex", "mey"))
    i = mod.state_space("ions", ("ni", "mix", "miy"))
    n = mod.state_space("neutrals", ("nn", "mnx", "mny"))
    fields = mod.field_space("fields", ("phi",))
    e_rate = mod.rate_operator("electron_rate", state_space="electrons", flux=True, sources=[])
    i_rate = mod.rate_operator("ion_rate", state_space="ions", flux=True, sources=[])
    fields_op = mod.operator(name="fields_from_species", signature=(e, i, n) >> fields,
                             kind="field_operator", expr="rho_e_plus_rho_i_plus_rho_n")
    return mod, (e, i, n), (e_rate, i_rate), fields_op


def main():
    mod, (e_space, i_space, n_space), (e_rate, i_rate), fields_op = _module()
    P = Program("multi_species_stage").bind_operators(mod)
    dt = P.dt

    e_n = P.state("electrons", space=e_space)
    i_n = P.state("ions", space=i_space)
    n_n = P.state("neutrals", space=n_space)

    # electrons and ions advanced to a predictor stage; neutrals held at n.
    e_star = P.linear_combine("e_star", e_n + dt * P.call(e_rate, e_n, name="Re"))
    i_star = P.linear_combine("i_star", i_n + dt * P.call(i_rate, i_n, name="Ri"))

    star = P.state_set("star", {"electrons": e_star, "ions": i_star, "neutrals": n_n})
    assert len(star) == 3
    assert [s.block for s in star.states()] == ["electrons", "ions", "neutrals"]

    fields_star = P.call(fields_op, *star.states(), name="fields_star")
    # the coherent solve lowers to the multi-block operator-first op:
    assert fields_star.vtype == "fields"
    assert fields_star.op == "call"
    assert fields_star.attrs["kind"] == "field_operator"
    assert len(fields_star.inputs) == 3  # exactly the three chosen stage states

    print("StageStateSet 'star' ->", [b for b, _ in star.items()])
    print("fields_star op       :", fields_star.op, "(inputs:", len(fields_star.inputs), ")")
    print("\nOK: the field solve reads an unambiguous stage of each block.")


if __name__ == "__main__":
    main()
