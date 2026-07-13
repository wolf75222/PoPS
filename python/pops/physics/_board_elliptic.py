"""Generic field-operator authoring for the blackboard physics facade."""
from __future__ import annotations

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
        if isinstance(equation.lhs, _math.Laplacian):
            model = self._dsl._m
            rhs = self._to_expr(equation.rhs)
            if equation.lhs.scale > 0:
                rhs = -rhs
            from pops.fields import FieldOutput, GradientOutput
            output_tuple = tuple(outputs)
            if not output_tuple or not isinstance(output_tuple[0], FieldOutput):
                raise ValueError(
                    "field_operator outputs must start with FieldOutput for the solved unknown")
            aux_names = [output_tuple[0].name]
            if len(output_tuple) == 2 and isinstance(output_tuple[1], GradientOutput):
                aux_names.extend((output_tuple[1].name + "_x", output_tuple[1].name + "_y"))
            elif len(output_tuple) != 1:
                raise ValueError(
                    "native field_operator outputs must be FieldOutput or "
                    "FieldOutput + GradientOutput")
            with atomic_attrs(
                    (model, "_elliptic_fields"), (model, "aux_extra_names"),
                    (self, "_module_cache")):
                for aux_name in aux_names:
                    if aux_name not in model.aux_extra_names:
                        self._dsl.aux_field(aux_name)
                self._dsl.elliptic_field(
                    name, rhs, operator="poisson", aux=aux_names)
                self._invalidate_authoring_views()
                provider = self.module.operator_handle(name)
                from pops.fields import FieldOperator

                operator = FieldOperator(
                    name, unknown=unknown, equation=equation, providers=provider,
                    outputs=output_tuple)
                operator.validate()
                return operator
        raise ValueError(
            "field_operator %r has no formula-backend lowering for principal operator %s"
            % (name, type(equation.lhs).__name__))


__all__ = ["_EllipticAuthoringMixin"]
