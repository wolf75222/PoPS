"""Authenticate the selected derivative of captured imported residual laws."""
from __future__ import annotations


def coupled_derivative_contract(expressions, strategy=None):
    from pops._ir.native_call import native_functions
    from pops.time.solve_request import DerivativeStrategy
    functions = native_functions(expressions)
    if strategy is None:
        if functions:
            raise TypeError("an imported residual requires an explicit DerivativeStrategy")
        strategy = DerivativeStrategy("finite_difference")
    if type(strategy) is not DerivativeStrategy:
        raise TypeError("coupled implicit derivative must be an exact DerivativeStrategy")
    if strategy.route == "unavailable":
        raise ValueError("coupled implicit derivative route is unavailable")
    providers = []
    for function in functions:
        if "lagged" in function.effects:
            raise ValueError("a lagged native residual has no full-iterate derivative realization")
        providers.append({"function": function.identity,
                          **function.require_derivative(strategy.route)})
    return {"route": strategy.route, "providers": tuple(providers)}, functions
