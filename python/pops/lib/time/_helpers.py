"""pops.lib.time._helpers -- shared helpers for operator-first scheme macros.

These helpers are used by predictor_corrector_local_linear, explicit_rk, and
imex_local_linear to introspect the operator registry bound to a Program and
dispatch calls with the correct arity.

# SPEC4-TODO: repoint to pops.time once it's a package.
"""

# _op_space_arity and _opcall reference P._registry, which is an attribute of
# pops.time.Program.  No import of Program is needed here: the functions receive
# the live Program instance as their first argument.


def _op_space_arity(P, name):
    """Number of space-typed inputs (State / FieldSpace) of operator @p name in the bound registry."""
    if P._registry is None:
        raise ValueError("operator-first macro: bind a module first (P.bind_operators(module))")
    op = P._registry.get(name)
    return sum(1 for t in op.signature.inputs if getattr(t, "kind", None) in ("state", "field"))


def _opcall(P, name, *candidate_args, value_name=None):
    """Call operator @p name passing exactly as many leading args as its signature's space inputs
    (so an operator that ignores the fields is called with the state alone, and a fields-free linear
    operator with no args)."""
    arity = _op_space_arity(P, name)
    return P.call(name, *candidate_args[:arity], name=value_name)
