"""Capture and verify the closed scalar language of a sampled face expression.

This module owns expression leaves and traversal. It has no numerical solver,
mesh execution, or model selection authority. The numerical descriptor retains
its rank, width, bounds, and halo declarations independently.
"""
from pops._ir import expr as ir
from pops._ir.control_expr import Where, Rounded
from pops._ir.expr import Var, _wrap
from pops._ir.visitors import _children


def capture_stencil_body(*, dimension, components, radius, field_components, body, parameters=()):
    spacing = tuple(Var("stencil_h_%d" % d, "stencil_spacing") for d in range(dimension))
    scalars = tuple(Var("stencil_p_%d" % i, "stencil_parameter") for i in range(len(parameters)))
    reads, field_reads = {}, {}

    def sample(offset, component, *, field=None):
        if (type(offset) is not tuple or len(offset) != dimension
                or any(type(x) is not int or abs(x) > radius for x in offset)):
            raise ValueError("stencil sample requires a bounded integer offset tuple")
        if field is not None:
            if type(field) is not int or not 0 <= field < len(field_components):
                raise ValueError("stencil field index is outside its declared input tuple")
            if type(component) is not int or not 0 <= component < field_components[field]:
                raise ValueError("stencil field component is outside its declared input")
            key = field, offset, component
            if key not in field_reads:
                field_reads[key] = Var("stencil_f_%d" % len(field_reads), "stencil_field_sample")
            return field_reads[key]
        if type(component) is not int or not 0 <= component < components:
            raise ValueError("stencil sample component is outside its declared state")
        key = offset, component
        if key not in reads:
            reads[key] = Var("stencil_q_%d" % len(reads), "stencil_sample")
        return reads[key]

    rows = []
    for axis in range(dimension):
        result = body(sample, spacing, axis, *scalars)
        if type(result) is not tuple or len(result) != 2:
            raise TypeError("stencil body returns (flux_tuple, incident_frequency)")
        flux, frequency = result
        if type(flux) is not tuple or len(flux) != components:
            raise ValueError("stencil flux width differs from its declared state")
        rows.append(tuple(map(_wrap, (*flux, frequency))))
    samples = tuple((offset, c, v) for (offset, c), v in reads.items())
    field_samples = tuple((*key, v) for key, v in field_reads.items())
    return tuple(rows), samples, spacing, field_samples, scalars



def row_leaf_names(row):
    """The names of the variables that the expressions of one stencil row hold."""
    names, pending, seen = set(), list(row), set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value) is ir.Var:
            names.add(value.name)
        pending.extend(_children(value))
    return frozenset(names)


def validate_closed_stencil(rows, samples, field_samples, spacing, *, exact=False, scalars=()):
    symbols = tuple(v for *_,v in samples)+tuple(v for *_,v in field_samples)+spacing+scalars
    if any(type(v) is not ir.Var for v in symbols) or len({v.name for v in symbols}) != len(symbols):
        raise ValueError("stencil leaf authority is not unique")
    permitted = {ir.Const, ir.Var, ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Pow,
                 ir.Neg, ir.Abs, ir.Sqrt, ir.Exp, ir.Sin, ir.Cos, ir.Log,
                 ir.Expm1, ir.Sign, ir.Minimum, ir.Maximum, ir.Compare,
                 ir.BooleanAnd, ir.BooleanOr, ir.BooleanNot, Where, Rounded}
    leaves = {v.name: v for *_, v in samples}
    leaves.update((v.name, v) for *_, v in field_samples)
    leaves.update((v.name, v) for v in spacing)
    leaves.update((v.name, v) for v in scalars)
    pending = [x for row in rows for x in row]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value) not in permitted:
            raise TypeError("stencil body contains an unsupported scalar operation")
        if type(value) is ir.Var:
            expected = leaves.get(value.name)
            if (expected is None or value.kind != expected.kind
                    or exact and value is not expected):
                raise ValueError("stencil body captured a foreign variable")
        pending.extend(_children(value))
