"""Inert closed-stencil serialization for composable GraphFunctions.

Only the existing scalar IR is reconstructed. No Python numerical evaluator or callable is stored.
"""
from pops._ir import expr as ir
from pops._ir.control_expr import Rounded, Where
from pops._ir.visitors import _children
from pops.identity.scalar import ScalarLiteral

_TYPES = {kind.__name__: kind for kind in (
    ir.Const, ir.Var, ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Pow, ir.Neg, ir.Abs, ir.Sqrt,
    ir.Exp, ir.Sin, ir.Cos, ir.Log, ir.Expm1, ir.Sign, ir.Minimum, ir.Maximum,
    ir.Compare, ir.BooleanAnd, ir.BooleanOr, ir.BooleanNot, Where, Rounded)}


def encode_stencil(flux):
    flux.validate(exact=True)
    nodes, indexes = [], {}
    symbols = tuple(v for *_, v in flux.samples)+tuple(v for *_, v in flux.field_samples)+flux.spacing+flux.parameter_symbols
    roots = (*symbols, *(v for row in flux.rows for v in row))
    for root in roots:
        stack = [(root, False)]
        while stack:
            value, leave = stack.pop()
            if id(value) in indexes:
                continue
            children = tuple(_children(value))
            if not leave:
                stack.append((value, True))
                stack.extend((child, False) for child in reversed(children))
                continue
            if type(value).__name__ not in _TYPES or _TYPES[type(value).__name__] is not type(value):
                raise TypeError("closed stencil has an unsupported scalar IR node")
            metadata = ()
            if type(value) is ir.Var:
                metadata = (value.name, value.kind)
            elif type(value) is ir.Const:
                metadata = (value.literal,)
            elif type(value) is ir.Compare:
                metadata = (value.comparison,)
            indexes[id(value)] = len(nodes)
            nodes.append((type(value).__name__, tuple(indexes[id(child)] for child in children), metadata))
    return dict(schema=1, dimension=flux.dimension, components=flux.components, radius=flux.radius,
                field_components=flux.field_components, parameters=flux.parameters,
                samples=tuple((offset, c, indexes[id(v)]) for offset, c, v in flux.samples),
                field_samples=tuple((f, offset, c, indexes[id(v)]) for f, offset, c, v in flux.field_samples),
                spacing=tuple(indexes[id(v)] for v in flux.spacing),
                parameter_symbols=tuple(indexes[id(v)] for v in flux.parameter_symbols),
                rows=tuple(tuple(indexes[id(v)] for v in row) for row in flux.rows), nodes=tuple(nodes))


def decode_stencil(data):
    from pops.numerics.stencil_flux import StencilFlux
    expected = {"schema", "dimension", "components", "radius", "field_components", "parameters",
                "samples", "field_samples", "spacing", "parameter_symbols", "rows", "nodes"}
    if set(data) != expected or data["schema"] != 1:
        raise ValueError("invalid closed stencil schema")
    nodes = []
    def reference(index):
        if type(index) is not int or not 0 <= index < len(nodes):
            raise ValueError("closed stencil has an absent or forward reference")
        return nodes[index]
    for name, children, metadata in data["nodes"]:
        kind = _TYPES.get(name)
        if kind is None:
            raise ValueError("closed stencil has an unsupported scalar IR node")
        arguments = tuple(reference(index) for index in children)
        if kind is ir.Var:
            if arguments or len(metadata) != 2 or any(type(x) is not str for x in metadata):
                raise ValueError("invalid closed stencil symbol")
            value = ir.Var(*metadata)
        elif kind is ir.Const:
            if arguments or len(metadata) != 1 or not isinstance(metadata[0], ScalarLiteral) or metadata[0].cpp is not None:
                raise ValueError("invalid closed stencil literal")
            value = ir.Const(metadata[0])
        elif kind is ir.Compare:
            if len(metadata) != 1 or len(arguments) != 2:
                raise ValueError("invalid closed stencil comparison")
            value = ir.Compare(metadata[0], *arguments)
        else:
            if metadata:
                raise ValueError("unexpected closed stencil scalar metadata")
            value = kind(*arguments)
        nodes.append(value)
    flux = object.__new__(StencilFlux)
    for name in ("dimension", "components", "radius", "field_components", "parameters"):
        object.__setattr__(flux, name, data[name])
    for name in ("spacing", "parameter_symbols"):
        object.__setattr__(flux, name, tuple(reference(i) for i in data[name]))
    object.__setattr__(flux, "rows", tuple(tuple(reference(i) for i in row) for row in data["rows"]))
    object.__setattr__(flux, "samples", tuple((offset, c, reference(i)) for offset, c, i in data["samples"]))
    object.__setattr__(flux, "field_samples", tuple((f, offset, c, reference(i)) for f, offset, c, i in data["field_samples"]))
    flux.validate(exact=True)
    if encode_stencil(flux) != data:
        raise ValueError("noncanonical or orphan closed stencil data")
    return flux
