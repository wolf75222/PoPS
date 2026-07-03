"""pops.lib.time._helpers -- shared scheme-builder helpers.

The single-stage RHS assembler ``_stage_rhs`` is the canonical home for the
explicit / split schemes (Spec 4 s6 / s14: the ready schemes live in
``pops.lib.time``). The operator-registry helpers ``_op_space_arity`` / ``_opcall``
introspect the Program-bound registry for the operator-first macros
(predictor_corrector_local_linear, explicit_rk, imex_local_linear) and dispatch
calls with the correct arity.

The ``@program_macro`` decorator (ADC-554) makes a scheme builder ONE IR route with
the manual Program: called with a live ``Program`` first argument it is the historical
in-place builder (unchanged, returns the final state / value); called WITHOUT one (block
name first) it builds a fresh ``Program``, lowers into it and RETURNS that Program, so
``isinstance(pops.lib.time.forward_euler("plasma"), Program)`` holds.

The stage helpers take the live ``pops.time.Program`` instance as their first argument,
so they need no ``pops.time`` import; the decorator imports ``pops.time.Program`` LAZILY
(function-local), keeping this module free of a lib -> time module-scope edge beyond the
layering allowance.
"""
import functools


def program_macro(build):
    """Make a scheme builder both an in-place mutator AND a Program factory (ADC-554).

    ``build(P, block, ...)`` is the historical in-place builder. The wrapper dispatches on the FIRST
    positional argument:

      - a live ``pops.time.Program`` -> the legacy path, called unchanged, returning ``build``'s own
        result (the final state / value / ``None``) so every existing ``macro(P, block, ...)`` caller
        is byte-identical;
      - anything else (the block name) -> a fresh ``Program`` (named after the scheme) is created, the
        builder lowers into it, and the PROGRAM is returned -- so a macro invoked as ``macro(block,
        ...)`` yields an inspectable Program, the same type the manual route produces.

    The two forms lower through the SAME builder into the SAME IR (only the Program's ``name`` differs
    between an explicit ``Program("x")`` and the fresh default), so a macro and the equivalent manual
    Program produce the same logical IR.
    """
    @functools.wraps(build)
    def macro(*args, **kwargs):
        from pops.time import Program  # lazy: keep pops.lib.time free of a module-scope time edge
        if args and isinstance(args[0], Program):
            return build(*args, **kwargs)  # legacy in-place path, result unchanged
        prog = Program(build.__name__)
        build(prog, *args, **kwargs)
        return prog

    return macro


def _stage_rhs(P, U, sources, flux):
    """Solve the elliptic fields from U and assemble its RHS for one stage. The FieldContext is
    distinct per stage (no stale global aux). flux=False builds a source-only sub-flow (e.g. Strang S).

    Uses the PRIVATE ``P._rhs_legacy`` builder: the macros author the RHS from the (flux, sources)
    pair, which is the internal lowering of the public typed ``P.rhs(terms=[...])`` -- not a second
    public path."""
    fields = P.solve_fields(U) if flux else None
    return P._rhs_legacy(state=U, fields=fields, flux=flux, sources=list(sources))


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
    # The PRIVATE _call: the macro resolves the operator by its internal registry name (the user
    # already named it at module-build time), not the public handle-only P.call.
    return P._call(name, *candidate_args[:arity], name=value_name)
