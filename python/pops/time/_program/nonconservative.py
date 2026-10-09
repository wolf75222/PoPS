"""Retain a physical path balance in its stage-qualified RHS invocation."""


def lower_path_rate(program, operator, args, name):
    from pops.numerics.nonconservative import path_balance_supported
    view = operator.lowering.get("physical_balance")
    if not path_balance_supported(view):
        return None
    state = args[0]
    fields = args[1] if len(args) > 1 else None
    value = program._rhs_primitive(name=name, state=state, fields=fields,
                                   flux=True, sources=[], fluxes=None)
    return program._replace_value(value, attrs={**value.attrs,
        "path_conservative": True, "physical_balance": view})
