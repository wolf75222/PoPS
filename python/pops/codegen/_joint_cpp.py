"""Local joint-result materialization inside the caller's existing kernel/guard."""
from __future__ import annotations

from pops._ir.application import ApplicationProjection, OperatorApplication
from pops._ir.native_call import NativeCall, NativeProjection


def joint_kind(expression):
    return isinstance(expression, (OperatorApplication, NativeCall))


def joint_expand(expression, render):
    if isinstance(expression, ApplicationProjection):
        offset = 0
        for name, values in expression.application.outputs.items():
            if name == expression.output:
                return "%s[%d]" % (render(expression.application), offset + expression.index)
            offset += len(values)
    if isinstance(expression, NativeProjection):
        return "%s.read(%d)" % (render(expression.call), expression.flat_index)
    if isinstance(expression, OperatorApplication):
        values = [render(e) for output in expression.outputs.values() for e in output]
        return "Kokkos::Array<pops::Real, %d>{%s}" % (len(values), ", ".join(values))
    return None


def native_declaration(expression, name, render, indent):
    """Return direct native call statements, guarded before entering the external target."""
    function = expression.function
    selected = (None if function.reads is None else
                set(function.reads) | {(d.input, d.component) for d in function.domains})
    arguments = {(i, j): render(value) if selected is None or (i, j) in selected else "pops::Real(0)"
                 for i, values in enumerate(expression.inputs) for j, value in enumerate(values)}
    predicates = ["Kokkos::isfinite(%s)" % text for pair, text in arguments.items()
                  if selected is None or pair in selected]
    for domain in function.domains:
        value = arguments[domain.input, domain.component]
        if domain.lower is not None:
            predicates.append("(%s %s %r)" % (value, ">" if domain.lower_open else ">=", domain.lower))
        if domain.upper is not None:
            predicates.append("(%s %s %r)" % (value, "<" if domain.upper_open else "<=", domain.upper))
    call = "%s(%s)" % (function.target, ", ".join(arguments.values()))
    result = "pops::NativeCallResult<%d>" % function.output_width
    support = (str("host" in function.execution_domains).lower(),
               str("device" in function.execution_domains).lower())
    return [indent + "static_assert(pops::native_call_execution_supported<%s, %s>, "
            '"native function has no declared route for the selected Kokkos execution target");' % support,
            indent + result + " " + name + " = " + result + "::rejected();",
            indent + "if (%s) { %s = %s; }" % (" && ".join(predicates) or "true", name, call)]


def instantiated_body(operator, application):
    """Authenticate an instantiated body against the current captured declaration."""
    from pops._ir.application import OperatorApplication, substitute_quantities
    from pops._ir.quantity import QuantityRef
    from pops._ir.visitors import _children
    from pops.model.hash_data import canonical_hash_data
    if not isinstance(application, OperatorApplication) or (
            application.operator.registered_operator_name != operator.name
            or application.operator.signature != operator.signature):
        raise ValueError("joint application differs from its declared operator signature")
    bindings = {}
    for expressions in operator.body.values():
        stack = list(expressions)
        seen = set()
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, QuantityRef):
                indices = [index for index, space in enumerate(operator.signature.inputs)
                           if node.space == space]
                if len(indices) != 1:
                    raise ValueError("joint application declaration input is ambiguous")
                bindings[node.handle, node.index] = application.inputs[indices[0]][node.index]
            stack.extend(_children(node))
    expected = substitute_quantities(operator.body, bindings)
    if canonical_hash_data(expected) != canonical_hash_data(application.outputs):
        raise ValueError("joint application differs from its authenticated captured body")
    return application.outputs
