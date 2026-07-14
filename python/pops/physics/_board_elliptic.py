"""Generic field-operator authoring for the blackboard physics facade."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pops.model import Handle

from ._board_contract import atomic_attrs, require_name

if TYPE_CHECKING:
    from pops.fields import FieldOperator
    from ._model_contract import _BoardModel
else:
    _BoardModel = object


class _EllipticAuthoringMixin(_BoardModel):
    """Construct physics-only field operators; numerics stay on ``Problem``."""

    def field_operator(
        self,
        name: Any,
        *,
        unknown: Any,
        equation: Any,
        outputs: Any = (),
    ) -> FieldOperator:
        """Return a generic physical operator over a model-owned unknown.

        The returned descriptor is paired later with exactly one
        :class:`pops.fields.FieldDiscretization` by ``Problem.field`` or
        ``Problem.add_field``. This method never selects a solver, boundary law,
        hierarchy policy or discretization.
        """
        name = require_name(name, "field_operator name")
        if name in self._field_operators:
            raise ValueError("field operator %r is already declared" % name)
        if not isinstance(unknown, Handle):
            raise TypeError("field_operator unknown must be a declared field Handle")
        if unknown.owner_path != self.owner_path or self._fields.get(unknown.local_id) != unknown:
            raise ValueError(
                "field_operator unknown %r is not declared by this physics model"
                % unknown.local_id)
        # The physics descriptor remains the public authority. Every field operator is emitted as
        # its own named native provider, including the first one: there is no privileged
        # ``fields_from_state`` slot whose global solver configuration later fields could overwrite.
        from pops import math as _math
        from pops._ir.elliptic import constant_reaction_scalar
        from pops._ir.values import RuntimeParamRef

        terms = _math.elliptic_terms(equation.lhs)
        laplacians = [term for term in terms if isinstance(term, _math.Laplacian)]
        reactions = [term for term in terms if isinstance(term, _math.Reaction)]
        unsupported = [
            term for term in terms
            if not isinstance(term, (_math.Laplacian, _math.Reaction))
        ]
        if len(laplacians) == 1 and len(reactions) <= 1 and not unsupported:
            laplacian_term = laplacians[0]

            def same_unknown(field: Any) -> bool:
                if isinstance(field, Handle):
                    return field == unknown
                if isinstance(field, _math.Unknown):
                    reference = getattr(field, "reference", None)
                    return reference == unknown if reference is not None else field.name == unknown.local_id
                return False

            if not same_unknown(laplacian_term.field):
                raise ValueError(
                    "field_operator principal term must act on its declared unknown %r"
                    % unknown.local_id)
            normalization = -float(laplacian_term.scale)
            if not math.isfinite(normalization) or normalization == 0.0:
                raise ValueError("field_operator Laplacian scale must be finite and non-zero")
            reaction = reactions[0] if reactions else None
            if reaction is not None:
                if not same_unknown(reaction.field):
                    raise ValueError(
                        "field_operator reaction term must act on its declared unknown %r"
                        % unknown.local_id)
                coefficient = reaction.coeff
                handle = getattr(coefficient, "handle", None)
                constant = constant_reaction_scalar(coefficient)
                bind_parameter = (
                    isinstance(coefficient, RuntimeParamRef)
                    and handle is not None
                    and getattr(handle, "kind", None) == "parameter"
                    and getattr(handle, "param_kind", None) in ("runtime", "derived")
                    and coefficient.dtype == "Real"
                )
                if constant is NotImplemented and not bind_parameter:
                    raise TypeError(
                        "screened field_operator reaction coefficient must be an exact finite "
                        "real/ConstParam or a typed Real RuntimeParam/DerivedParam read produced "
                        "by model.value(parameter)")
                multiplier = float(reaction.scale) / normalization
                if not math.isfinite(multiplier):
                    raise ValueError(
                        "screened field_operator must normalize to -laplacian(phi) + "
                        "kappa*phi with a strictly positive reaction coefficient")
                if constant is NotImplemented:
                    positive = multiplier > 0.0
                else:
                    try:
                        effective = float(constant) * multiplier
                    except (TypeError, ValueError, OverflowError):
                        effective = float("nan")
                    positive = math.isfinite(effective) and effective > 0.0
                if not positive:
                    raise ValueError(
                        "screened field_operator must normalize to -laplacian(phi) + "
                        "kappa*phi with a strictly positive reaction coefficient")
            model = self._dsl._m
            # Keep the provider in the public ``-laplacian+kappa = rhs`` convention.  Builtin
            # native backends adapt that physical RHS to their internal ``laplacian-kappa``
            # residual; external providers receive the public convention unchanged.
            rhs = self._to_expr(equation.rhs)
            if normalization != 1.0:
                rhs = rhs / normalization
            from pops.fields import FieldOutput, GradientOutput
            output_tuple = tuple(outputs)
            if not output_tuple or not isinstance(output_tuple[0], FieldOutput):
                raise ValueError(
                    "field_operator outputs must start with FieldOutput for the solved unknown")
            aux_names = [output_tuple[0].name]
            gradient_sign = 1
            if len(output_tuple) == 2 and isinstance(output_tuple[1], GradientOutput):
                aux_names.extend((output_tuple[1].name + "_x", output_tuple[1].name + "_y"))
                gradient_sign = output_tuple[1].sign
            elif len(output_tuple) != 1:
                raise ValueError(
                    "native field_operator outputs must be FieldOutput or "
                    "FieldOutput + GradientOutput")
            with atomic_attrs(
                    (model, "_elliptic_fields"), (model, "aux_extra_names"),
                    (self, "_module_cache"), (self, "_field_operators")):
                from .aux import AUX_CANONICAL
                for aux_name in aux_names:
                    if aux_name in model.aux_names or aux_name in model.aux_extra_names:
                        continue
                    if aux_name in AUX_CANONICAL:
                        self._dsl.aux(aux_name)
                    else:
                        self._dsl.aux_field(aux_name)
                self._dsl.elliptic_field(
                    name, rhs, operator="poisson", aux=aux_names,
                    gradient_sign=gradient_sign)
                self._invalidate_authoring_views()
                provider = self.module.operator_handle(name)
                from pops.fields import FieldOperator, ScreenedPoissonOperator

                operator_type = ScreenedPoissonOperator if reaction is not None else FieldOperator
                operator = operator_type(
                    name, unknown=unknown, equation=equation, providers=provider,
                    outputs=output_tuple)
                operator.validate()
                self._field_operators[name] = operator
                return operator
        raise ValueError(
            "field_operator %r has no formula-backend lowering for principal operator %s"
            % (name, type(equation.lhs).__name__))


__all__ = ["_EllipticAuthoringMixin"]
