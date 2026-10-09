"""Typed C++ locals, separate from exact public component identities.

cpp-local-symbols@2 preserves noncolliding legacy spellings. ContextVar isolates
concurrent and reentrant emitters and restores the outer table on failure.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import re
from types import MappingProxyType

CONTRACT = "cpp-local-symbols@2"
_ACTIVE = ContextVar("pops_cpp_local_symbols", default={})
_RESERVED = frozenset(("index", "providers", "params", "outA", "statusA",
                       "U", "F", "dir", "dt", "th_dt_", "State", "Prim", "Primitive",
                       "Schema", "Real", "Axis", "ProviderParameters", "state", "interior",
                       "left", "right", "direction", "UL", "UR", "Up", "result",
                       "bound", "component", "n_vars", "dimension"))


def symbol_table(symbols):
    from .cpp_writer import _cpp_identifier

    symbols = set(symbols)
    bases = {symbol: _cpp_identifier(symbol[1]) for symbol in symbols}
    occupied = set(bases.values()) | _RESERVED
    used, result = set(_RESERVED), {}
    order = {"cons": 0, "prim": 1, "aux": 2}
    # Already valid public spellings win over aliases produced by sanitization.
    for kind, name in sorted(symbols, key=lambda s: (
            bases[s] != s[1], order.get(s[0], 3), s[1].encode("utf8"), s[0])):
        base = bases[kind, name]
        internal = re.fullmatch(r"cse\d+_", base) or (kind != "cons" and
            re.fullmatch(r"pops_input_\d+_component_\d+_*", base))
        if base not in used and not internal:
            chosen = base
        else:
            chosen = "pops_cell_symbol_" + kind.encode("utf8").hex() + "_" + name.encode("utf8").hex()
            # A public name may itself look like our escape. Do not reserve user namespaces.
            while chosen in occupied or chosen in used:
                chosen += "_q"
        result[kind, name] = chosen
        used.add(chosen)
    return MappingProxyType(result)


def variable_identifier(name, kind):
    from .cpp_writer import _cpp_identifier

    identifier = _ACTIVE.get().get((kind, name))
    return _cpp_identifier(name) if identifier is None else identifier


def variable_bindings(expressions):
    """Observe every typed scalar leaf, including an Aux NaN hidden by IEEE min."""
    from pops._ir.expr import Var
    from pops._ir.visitors import _children

    pending, seen, result = list(expressions), set(), {}
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Var):
            result[node.kind, node.name] = variable_identifier(node.name, node.kind)
        pending.extend(_children(node))
    return result


@contextmanager
def variable_scope(symbols):
    token = _ACTIVE.set(symbol_table(symbols))
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)


def _model_symbols(authority):
    models = getattr(authority, "models_by_owner", None)
    models = tuple(models.values()) if models is not None else (authority,)
    result = set()
    for model in models:
        model = getattr(model, "_dsl", model)
        model = getattr(model, "_m", model)
        result.update(("cons", name) for name in getattr(model, "cons_names", ()))
        result.update(("hoist", "inv_" + name) for name in getattr(model, "cons_names", ()))
        result.update(("prim", name) for name in getattr(model, "prim_defs", {}))
        result.update(("aux", name) for name in getattr(model, "_provider_components", ()))
    return result


def printer_scope(function):
    @wraps(function)
    def emit(*args, **kwargs):
        authority = kwargs.get("model_graph")
        if authority is None:
            authority = kwargs.get("model", args[0] if args else None)
        with variable_scope(_model_symbols(authority)):
            return function(*args, **kwargs)
    return emit


def block_scratch_identifiers(prefix, names):
    """Collision-free internal token families (block-scratch-identifiers@1).

    Public labels remain exact registry keys. Preserve legacy valid tokens; the
    scratch and its appended FieldView `A` must both avoid other token families
    and actual scalar locals in the common printer scope.
    """
    from .cpp_writer import _cpp_identifier

    names = set(names)
    bases = {name: _cpp_identifier(prefix + name) for name in names}
    family = lambda token: {token, token + "A"}
    occupied = set().union(*(family(base) for base in bases.values()))
    used = set(_ACTIVE.get().values()) | set(_RESERVED)
    result = {}
    for name in sorted(names, key=lambda name: (
            bases[name] != prefix + name, name.encode("utf8"))):
        token = bases[name]
        if family(token) & used:
            token = prefix + "pops_block_" + name.encode("utf8").hex()
            while family(token) & (occupied | used):
                token += "_q"
        result[name] = token
        used.update(family(token))
    return MappingProxyType(result)
