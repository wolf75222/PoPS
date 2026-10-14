"""Materialize mathematical expressions through the native Program IR."""
from typing import TYPE_CHECKING
from pops._ir.expr import is_scalar_expression
from pops.time._authoring import atomic_authoring
from pops.time.expressions import ProgramExpression, component_names, encode_expressions
from pops.time._program.value_validation import require_compatible_spaces, require_owned


def is_pointwise_expression(value):
    return is_scalar_expression(value) or isinstance(value, (ProgramExpression, tuple, list))


if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramExpressions(_ProgramBase):
    @atomic_authoring
    def scalar_expression(self, name, expression):
        """Materialize the common mathematical DAG over explicit native Scalars."""
        encoded, nodes, inputs = encode_expressions((expression,), self)
        if any(value.vtype != "scalar" for value in inputs):
            raise TypeError("scalar expressions require explicit Scalar inputs, not field components")
        if any(node[0] == "coefficient" for node in nodes):
            raise TypeError("scalar equations require an explicit requested_dt input, not an ambient coefficient")
        blocks = {value.block for value in inputs if value.block is not None}
        if len(blocks) > 1:
            raise ValueError("scalar expression inputs belong to different blocks")
        return self._new("scalar", "scalar_expression", inputs,
                         {"expressions": encoded, "expression_nodes": nodes}, name,
                         next(iter(blocks), None), inherit_state_ref=False)

    @atomic_authoring
    def isfinite(self, value):
        """A native Bool testing a Scalar, without evaluating it in Python."""
        require_owned(self, value, "isfinite", vtype="scalar")
        return self._new("bool", "scalar_isfinite", (value,), {}, None, value.block,
                         point=value.point, inherit_state_ref=False)

    @atomic_authoring
    def rate(self, name, expression, *, of, at=None):
        """Evaluate an authored pointwise numerical RHS in Rate(of.space).

        The expression is a rate, not an integrated increment. For a time
        average, the author writes the division by the actual duration.
        This is not a stability, consistency or admissibility certificate.
        Raw spatial rates must retain their own affine update and flux receipt;
        this pointwise constructor cannot hide them inside a new local rate.
        """
        if type(name) is not str or not name:
            raise TypeError("pointwise rate requires a nonempty name")
        require_owned(self, of, "pointwise rate state")
        if of.vtype != "state" or of.state_ref is None:
            raise TypeError("pointwise rate requires an exact qualified State")
        if isinstance(expression, ProgramExpression):
            if expression.template is not of:
                raise ValueError("pointwise rate template differs from its declared State")
            components = expression.components
        else:
            components = tuple(expression) if isinstance(expression, (tuple, list)) else (expression,)
        return self._pointwise_expression(name, ProgramExpression(components, of),
                                         at=at, numerical_rate=of)

    def _materialize_finite_vector(self, name, components, *, support, template, at=None):
        from pops._ir.finite_linear import lower_finite_scalars
        if component_names(template) != support[1]:
            raise ValueError("finite output template differs from its ordered support")
        expression = ProgramExpression(lower_finite_scalars(components), template)
        return self._pointwise_expression(name, expression, at=at, finite_support=support)

    @atomic_authoring
    def _pointwise_expression(self, name, expression, *, at=None, finite_support=None,
                              field_product_space=None, field_product_authority=None,
                              numerical_rate=None):
        if isinstance(expression, ProgramExpression):
            expressions = expression.components
            template = expression.template
        else:
            expressions = tuple(expression) if isinstance(expression, (tuple, list)) else (expression,)
            template = None
        encoded, nodes, inputs = encode_expressions(expressions, self)
        expression_input_count = len(inputs)
        if not any(value.vtype in ("state", "scalar_field") for value in inputs):
            if template is None:
                raise ValueError("a pointwise expression needs a temporal value to define its support")
            # Literal fields still need a layout/space authority. An explicitly
            # supplied template supplies geometry and storage, not a fictitious
            # numerical dependence such as 0*state (which is NaN for NaN state).
            require_owned(self, template, "constant pointwise support template")
            inputs = (*inputs, template)
        if template is None:
            template = next((item for item in inputs if item.vtype == "state"), None)
            if template is None:
                template = next((item for item in inputs
                                 if item.op != "integral_candidate"), None)
        if template is None:
            raise ValueError("a global capture cannot define cell support")
        require_owned(self, template, "pointwise expression template")
        if template.vtype != "state":
            raise TypeError("pointwise materialization currently requires a typed State template")
        if len(expressions) != len(component_names(template)) and field_product_space is None:
            raise ValueError("pointwise output component count must match its StateSpace")
        attrs = {"expressions": encoded, "expression_nodes": nodes}
        if numerical_rate is not None:
            if template is not numerical_rate or field_product_space is not None or finite_support is not None:
                raise ValueError("pointwise rate requires its exact State template")
            if at is not None and at != template.point:
                raise ValueError("pointwise rate evaluation point differs from its State")
            if all(value is not template for value in inputs):
                inputs = (*inputs, template)
            attrs.update(numerical_pointwise_rate="pops.pointwise-rate@1",
                         target_input=next(i for i, value in enumerate(inputs) if value is template))
        if field_product_space is not None:
            from pops.model.spaces import FieldSpace
            if type(field_product_space) is not FieldSpace or field_product_authority is None or finite_support is not None:
                raise TypeError("field seed product requires an explicit FieldSpace and original authority")
            if len(expressions) != len(field_product_space.components):
                raise ValueError("field seed product width differs from its physical FieldSpace")
            attrs.update(field_product_seed=field_product_authority,
                         ncomp=len(expressions), field_product_template_index=next(i for i, row in enumerate(inputs) if row is template))
        globals_ = tuple(value for value in inputs if value.op == "integral_candidate")
        if globals_:
            if all(value is not template for value in inputs):
                inputs = (*inputs, template)
            attrs["global_template_index"] = next(i for i, value in enumerate(inputs) if value is template)
            from pops.time._evaluation_point import evaluation_stage_fraction
            if any(evaluation_stage_fraction(value) != evaluation_stage_fraction(template)
                   for value in globals_):
                raise ValueError("global capture and cell evaluation require the same exact point")
        if finite_support is not None:
            if type(finite_support) is not tuple or len(finite_support) != 2 or \
                    type(finite_support[0]) is not str or not finite_support[0] or \
                    finite_support[1] != component_names(template):
                raise ValueError("finite materialization requires its exact output support")
            if all(value is not template for value in inputs):
                inputs = (*inputs, template)
            attrs["finite_support_v1"] = finite_support
            attrs["finite_template_index"] = next(i for i, value in enumerate(inputs) if value is template)
            for value in inputs:
                if value.op == "integral_candidate" or value.vtype == "scalar":
                    continue
                for attribute in ("layout", "centering", "support", "sampling", "frame", "clock"):
                    if getattr(value.space, attribute) != getattr(template.space, attribute):
                        raise ValueError("finite materialization co-location obligation: different " + attribute)
        if len(inputs) > expression_input_count:
            # Every appended template supplies authority/layout only. This also
            # covers a rate whose expressions read another explicit field.
            attrs["support_only_inputs"] = tuple(range(expression_input_count, len(inputs)))
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
        from pops.time._program.value_validation import rate_space_for
        result = self._new("rhs" if numerical_rate is not None else "scalar_field" if field_product_space is not None else "state", "pointwise_expression", inputs,
                         attrs, name, template.block,
                         space=rate_space_for(template.space) if numerical_rate is not None else template.space if field_product_space is None else field_product_space,
                         point=template.point if at is None else at,
                         field_context=context, state_ref=template.state_ref if field_product_space is None else None,
                         inherit_state_ref=field_product_space is None)
        if numerical_rate is not None:
            validate_pointwise_rate(result)
        return result


def _validate_pointwise_rate_dependencies(value, source):
    """Retain spatial proof obligations across scalar algebra and SSA aliases.

    The exact declared input State is a stage boundary: its earlier explicit
    update is checked by the usual consumer traversal. Other captured values
    cannot silently turn an unevaluated spatial rate into a local source.
    """
    from collections.abc import Mapping
    from pops.time.values import ProgramValue

    seen = set()

    def leaves(item):
        if isinstance(item, ProgramValue):
            yield item
        elif isinstance(item, Mapping):
            for child in item.values():
                yield from leaves(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                yield from leaves(child)

    def visit(node, *, boundary=False):
        if (node is source and not boundary) or node.id in seen:
            return
        seen.add(node.id)
        if node.vtype == "rhs":
            local = (node.op in {"source", "linear_combine"}
                     or (node.op == "rhs" and not node.attrs.get("flux", True))
                     or (node.op == "pointwise_expression"
                         and node.attrs.get("numerical_pointwise_rate") == "pops.pointwise-rate@1"))
            if not local:
                raise ValueError("pointwise rate cannot absorb a spatial or unclassified rate; "
                                 "keep its affine update, stability bound and accepted flux receipt explicit")
        support_only = node.attrs.get("support_only_inputs", ())
        for index, dependency in enumerate(node.inputs):
            if index not in support_only:
                visit(dependency)
        for name, attribute in node.attrs.items():
            if not name.endswith("_block"):
                for dependency in leaves(attribute):
                    visit(dependency)

    # A retagged RHS is not a State-stage authority. Only the affine update
    # shape traversed by explicit_update_consumers may retain spatial leaves
    # here; its exact weights/budget are subsequently checked at lowering.
    has_state_leaf = False

    def stage(node):
        nonlocal has_state_leaf
        if node.op == "linear_combine":
            for child in node.inputs:
                stage(child)
        elif node.op == "acceptance_guard":
            stage(node.inputs[0])
        elif node.vtype != "rhs":
            has_state_leaf |= node.vtype == "state"
            visit(node, boundary=True)

    if source.op in {"linear_combine", "acceptance_guard"}:
        stage(source)
        if not has_state_leaf:
            raise ValueError("pointwise rate State is only a retagged rate; "
                             "a spatial stage requires an explicit affine state update")
    else:
        visit(source, boundary=True)
    visit(value)


def validate_pointwise_rate(value):
    from pops.time._program.value_validation import rate_space_for
    index=value.attrs.get("target_input")
    if (value.op != "pointwise_expression" or value.vtype != "rhs"
            or value.attrs.get("numerical_pointwise_rate") != "pops.pointwise-rate@1"
            or type(index) is not int or not 0<=index<len(value.inputs)):
        raise ValueError("pointwise rate lost its output authority")
    source=value.inputs[index]
    if (source.vtype != "state" or source.state_ref is None
            or source.block != value.block or source.state_ref != value.state_ref
            or source.point != value.point
            or value.space.to_data() != rate_space_for(source.space).to_data()):
        raise ValueError("pointwise rate State, space or evaluation point changed")
    _validate_pointwise_rate_dependencies(value, source)
    return source
