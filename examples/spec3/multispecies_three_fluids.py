"""Spec 3 generic multi-species: three fluids over the operator-first multi-block kernel.

No species is hardcoded: each is a named block of a StateSpace. This builds a 3-species
time step with the operator-first kernel (pops.model multi-state spaces + RateBundle for a
typed multi-output coupling + pops.time multi-block Program + commit_many). It builds the IR
and checks structure; it does not run a simulation. The blackboard sugar for this
(m.species for N>1, m.coupled_rate) and the multi-block field-solve / coupled-rate RUNTIME
are tracked by ADC-457.

Run: python3 examples/spec3/multispecies_three_fluids.py
"""
import pops.model as model
import pops.time as adctime


def species_spaces():
    """Three species, each a StateSpace -- the core knows BlockInstances, not 'electrons'."""
    e = model.StateSpace("electron_state", ["ne", "mex", "mey"],
                         roles={"ne": "Density", "mex": "MomentumX", "mey": "MomentumY"})
    i = model.StateSpace("ion_state", ["ni", "mix", "miy"])
    n = model.StateSpace("neutral_state", ["nn", "mnx", "mny"])
    return e, i, n


def collision_bundle(e, i, n):
    """A coupled collision operator's typed multi-output: one Rate per species (arity 3)."""
    coll = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i),
                             "neutrals": model.Rate(n)})
    # the bundle is typed: a wrong rate on a wrong state is rejected
    coll.require("electrons", e)
    coll.require("ions", i)
    coll.require("neutrals", n)
    return coll


def species_module():
    """Operator-first module for a three-fluid step."""
    mod = model.Module("three_fluids")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"),
                        roles={"ne": "Density", "mex": "MomentumX", "mey": "MomentumY"})
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    n = mod.state_space("neutral_state", ("nn", "mnx", "mny"))
    fields = mod.field_space("fields", ("phi",))
    fields_op = mod.operator(name="fields_from_species", signature=(e, i, n) >> fields,
                             kind="field_operator", expr="rho_e_plus_rho_i_plus_rho_n")
    e_rate = mod.rate_operator("electron_rate", state_space="electron_state", flux=True, sources=[])
    i_rate = mod.rate_operator("ion_rate", state_space="ion_state", flux=True, sources=[])
    n_rate = mod.rate_operator("neutral_rate", state_space="neutral_state", flux=True, sources=[])
    return mod, (e, i, n), fields_op, (e_rate, i_rate, n_rate)


def multi_species_step():
    """A forward step coupling three blocks through a shared field solve + per-species rates,
    committed atomically. IR-level (the coupled field-solve runtime is ADC-457)."""
    mod, (e_space, i_space, n_space), fields_op, rates = species_module()
    e_rate, i_rate, n_rate = rates
    P = adctime.Program("three_fluids_step").bind_operators(mod)
    dt = P.dt
    e_n = P.state("electrons", space=e_space)
    i_n = P.state("ions", space=i_space)
    n_n = P.state("neutrals", space=n_space)

    P.call(fields_op, e_n, i_n, n_n, name="fields")  # coupled, arity 3
    e1 = P.linear_combine("e1", e_n + dt * P.call(e_rate, e_n, name="Re"))
    i1 = P.linear_combine("i1", i_n + dt * P.call(i_rate, i_n, name="Ri"))
    n1 = P.linear_combine("n1", n_n + dt * P.call(n_rate, n_n, name="Rn"))

    P.commit_many({"electrons": e1, "ions": i1, "neutrals": n1})  # atomic multi-block commit
    return P


def main():
    e, i, n = species_spaces()
    coll = collision_bundle(e, i, n)
    print("species:", [s.name for s in (e, i, n)])
    print("RateBundle arity:", len(coll), "->", coll.keys())
    try:
        coll.require("electrons", i)   # wrong StateSpace
    except TypeError as exc:
        print("typed multi-output rejects a wrong rate:", str(exc)[:70], "...")

    P = multi_species_step()
    assert set(P.commits()) == {"electrons", "ions", "neutrals"}
    fields = next(v for v in P._values
                  if v.op == "call" and v.attrs.get("operator") == "fields_from_species")
    assert len(fields.inputs) == 3
    print("committed blocks:", sorted(P.commits()))
    print("coupled field solve inputs:", len(fields.inputs))
    print("\nOK: 3 species, no hardcoding, typed RateBundle, atomic commit_many.")


if __name__ == "__main__":
    main()
