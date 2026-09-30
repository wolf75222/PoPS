"""Materialize mathematical expressions through the native Program IR."""
from pops._ir.expr import Expr
from pops.time._authoring import atomic_authoring
from pops.time.expressions import ProgramExpression, component_names, encode_expressions
from pops.time._program.value_validation import require_compatible_spaces, require_owned


def is_pointwise_expression(value):
    return isinstance(value, (Expr, ProgramExpression, tuple, list))


class _ProgramExpressions:
    @atomic_authoring
    def _pointwise_expression(self, name, expression, *, at=None, finite_support=None):
        if isinstance(expression, ProgramExpression):
            expressions = expression.components
            template = expression.template
        else:
            expressions = tuple(expression) if isinstance(expression, (tuple, list)) else (expression,)
            template = None
        encoded, nodes, inputs = encode_expressions(expressions, self)
        if not inputs:
            raise ValueError("a pointwise expression needs a temporal value to define its support")
        if template is None:
            template = next((item for item in inputs if item.vtype == "state"), inputs[0])
        require_owned(self, template, "pointwise expression template")
        if template.vtype != "state":
            raise TypeError("pointwise materialization currently requires a typed State template")
        if len(expressions) != len(component_names(template)):
            raise ValueError("pointwise output component count must match its StateSpace")
        attrs = {"expressions": encoded, "expression_nodes": nodes}
        if finite_support is not None:
            from pops.linalg.finite import FiniteSupport
            if not isinstance(finite_support, FiniteSupport) or finite_support.dofs != component_names(template):
                raise ValueError("finite materialization requires its exact output support")
            if all(value is not template for value in inputs):
                inputs = (*inputs, template)
            attrs["finite_support_v1"] = finite_support.contract
            attrs["finite_template_index"] = next(i for i, value in enumerate(inputs) if value is template)
            for value in inputs:
                for attribute in ("layout", "centering", "support", "sampling", "frame", "clock"):
                    if getattr(value.space, attribute) != getattr(template.space, attribute):
                        raise ValueError("finite materialization co-location obligation: different " + attribute)
        for value in (() if finite_support is not None else inputs):
            if value.vtype == "scalar":
                continue
            if value.block != template.block:
                raise ValueError("pointwise inputs require the same block support; use an explicit map")
            require_compatible_spaces(template.space, value.space, "pointwise expression", typed_pair=True)
        from pops.time.field_context import merge_field_provenance
        context = None
        for value in inputs:
            context = merge_field_provenance(context, value.field_context)
        return self._new("state", "pointwise_expression", inputs,
                         attrs, name, template.block,
                         space=template.space, point=template.point if at is None else at,
                         field_context=context, state_ref=template.state_ref)
