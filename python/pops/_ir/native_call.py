"""Immutable native-call expressions; Python captures but never executes them."""
from __future__ import annotations

from .application import ApplicationContext
from .expr import Expr, _wrap
from .visitors import _dependencies


class NativeCall(Expr):
    def __init__(self, function, inputs, *, context=None, occurrence=None):
        from pops.native_calls import NativeFunction
        from .quantity import QuantityRef
        if type(function) is not NativeFunction:
            raise TypeError("native call requires an authenticated NativeFunction")
        if len(inputs) != len(function.signature.inputs):
            raise ValueError("native call input arity differs from its signature")
        values = []
        for space, argument in zip(function.signature.inputs, inputs, strict=True):
            entries = (argument,) if isinstance(argument, Expr) else tuple(argument)
            if len(entries) != len(space.components):
                raise ValueError("native call input shape differs from its signature")
            for entry in entries:
                if isinstance(entry, QuantityRef) and entry.space != space:
                    raise ValueError("native call input support/representation differs from signature")
            values.append(tuple(_wrap(entry) for entry in entries))
        self.function = function
        self.inputs = tuple(values)
        self.context = context or ApplicationContext()
        if type(self.context) is not ApplicationContext:
            raise TypeError("native call context must be ApplicationContext")
        self.effects = function.effects
        if occurrence is not None and (type(occurrence) is not str or not occurrence):
            raise TypeError("native call occurrence must be nonempty immutable text")
        if "diagnostic_counter" in self.effects and occurrence is None:
            raise ValueError("diagnostic native calls require an explicit logical occurrence")
        self.occurrence = occurrence

    def __getitem__(self, output):
        entries = dict(self.function.output_entries)
        return tuple(NativeProjection(self, output, index)
                     for index in range(len(entries[output].components)))

    def __getattr__(self, output):
        if output.startswith("_"):
            raise AttributeError(output)
        if output not in dict(self.function.output_entries):
            raise AttributeError(output)
        return self[output]

    def __pops_ir_children__(self):
        # Domain predicates are reads too. Missing footprint is conservative.
        reads = self.function.reads
        if reads is None:
            return tuple(entry for values in self.inputs for entry in values)
        selected = set(reads) | {(d.input, d.component) for d in self.function.domains}
        return tuple(value for i, values in enumerate(self.inputs)
                     for j, value in enumerate(values) if (i, j) in selected)

    def __pops_ir_key__(self, recurse):
        from pops.model.spaces import _metadata_key
        return ("native_call", self.function.identity, self.occurrence,
                tuple(tuple(recurse(v) for v in values) for values in self.inputs),
                _metadata_key((self.context.stage, self.context.iterate,
                               self.context.location, self.context.sampling)))

    def deps(self):
        return _dependencies(self.__pops_ir_children__())

    def derivative_contract(self, route):
        return self.function.require_derivative(route)

    def jacobian(self, *, route):
        """Explicit row-major native Jacobian; no opaque differentiation is inferred."""
        from pops.model import FieldSpace, Signature
        from pops.native_calls import NativeFunction
        contract = self.derivative_contract(route)
        if contract["target"] is None:
            raise TypeError("finite_difference belongs to the explicitly selected solver strategy")
        if "lagged" in self.effects:
            raise ValueError("a lagged native residual cannot supply a full residual derivative")
        width = self.function.output_width * sum(len(values) for values in self.inputs)
        output = FieldSpace("native_jacobian", components=tuple("entry_%d" % i for i in range(width)))
        function = NativeFunction(self.function.component, contract["target"],
            Signature(self.function.signature.inputs, output),
            execution_domains=self.function.execution_domains, reads=self.function.reads,
            domains=self.function.domains, effects=self.effects)
        return NativeCall(function, self.inputs, context=self.context, occurrence=self.occurrence)

    def eval(self, env):
        raise TypeError("native calls execute only in compiled native kernels; no Python callback")

    def to_cpp(self):
        raise TypeError("native call requires its authenticated joint emission context")

    def to_data(self):
        from .visitors import _dag_key_data
        return _dag_key_data((self,))


class NativeProjection(Expr):
    def __init__(self, call, output, index):
        if type(call) is not NativeCall:
            raise TypeError("native projection requires its NativeCall")
        entries = dict(call.function.output_entries)
        if output not in entries or type(index) is not int or not 0 <= index < len(entries[output].components):
            raise ValueError("native projection is outside its declared output shape")
        self.call, self.output, self.index = call, output, index

    @property
    def flat_index(self):
        offset = 0
        for output, space in self.call.function.output_entries:
            if output == self.output:
                return offset + self.index
            offset += len(space.components)
        raise AssertionError("validated native output disappeared")

    def __pops_ir_children__(self):
        return (self.call,)

    def __pops_ir_key__(self, recurse):
        return ("native_projection", recurse(self.call), self.output, self.index)

    def deps(self):
        return self.call.deps()

    def __pops_ir_diff__(self, *, recurse, target, definitions):
        from .expr import Const
        from .lowering import _s_add, _s_mul
        jacobian = self.call.jacobian(route="exact").value
        inputs = tuple(value for values in self.call.inputs for value in values)
        result = Const(0)
        for column, value in enumerate(inputs):
            result = _s_add(result, _s_mul(
                jacobian[self.flat_index * len(inputs) + column], recurse(value)))
        return result

    def eval(self, env):
        return self.call.eval(env)

    def to_cpp(self):
        raise TypeError("native projection must preserve its one joint native call")


def native_functions(expressions):
    """Discover exact immutable providers from captured expression roots, without execution."""
    from collections.abc import Mapping
    from .visitors import _children
    found, seen = {}, set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, NativeCall):
            found.setdefault(value.function.identity, value.function)
        if isinstance(value, Expr):
            for child in _children(value):
                visit(child)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
    visit(expressions)
    return tuple(found.values())
