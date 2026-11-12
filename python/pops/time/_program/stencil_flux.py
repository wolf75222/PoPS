"""Program construction of a closed native conservative face stencil."""
from typing import TYPE_CHECKING
from pops._ir.data_model import field_role

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object

from pops.time._authoring import atomic_authoring

# The operation that this module builds, named once: the method of the program below, the op of its
# values, and the elaborator of the surface, which enters a diffusive face through it (N3 lot 1).
OPERATION = "stencil_flux_divergence"


class _ProgramStencilFlux(_ProgramBase):
    @atomic_authoring
    def stencil_flux_divergence(self, name, state, *, flux, fields=(), parameters=()):
        """Return ``-div_h(flux)`` using one stored oriented face transfer.

        This is a numerical primitive, not an implicit physical-model declaration.
        A Realize can bind it to retained balance occurrences explicitly.
        """
        from pops.numerics.stencil_flux import StencilFlux
        from pops.time._program.value_validation import require_owned, rate_space_for
        from pops.time.expressions import component_names
        require_owned(self, state, "stencil flux state")
        if state.vtype != "state" or state.state_ref is None:
            raise TypeError("stencil flux requires a qualified State")
        if type(flux) is not StencilFlux:
            raise TypeError("stencil flux requires a closed StencilFlux declaration")
        flux.validate()
        if len(component_names(state)) != flux.components:
            raise ValueError("stencil component space differs from the input State")
        if type(fields) is not tuple:
            raise TypeError("stencil fields require an explicit ordered tuple")
        for field in fields:
            require_owned(self, field, "stencil field input")
        _validate_fields(state, fields, flux)
        if type(parameters) is not tuple:
            raise TypeError("stencil parameters require an explicit ordered tuple")
        for parameter in parameters:
            require_owned(self, parameter, "stencil scalar parameter")
        _validate_parameters(parameters, flux)
        _validate_runtime_parameters(state, flux)
        return self._new("rhs", OPERATION, (state, *fields, *parameters), {"stencil_flux": flux},
                         name, state.block, space=rate_space_for(state.space),
                         field_context=state.field_context)


def validate_stencil_flux(value):
    from pops.numerics.stencil_flux import StencilFlux
    from pops.time.expressions import component_names
    if (value.op != OPERATION or len(value.inputs) < 1
            or value.inputs[0].vtype != "state"
            or value.block != value.inputs[0].block
            or value.state_ref != value.inputs[0].state_ref):
        raise ValueError("stencil flux input/support authority changed")
    flux = value.attrs.get("stencil_flux")
    if type(flux) is not StencilFlux or len(component_names(value.inputs[0])) != flux.components:
        raise ValueError("stencil flux declaration/width changed")
    flux.validate()
    end = 1 + len(flux.field_components)
    _validate_fields(value.inputs[0], value.inputs[1:end], flux)
    _validate_parameters(value.inputs[end:], flux)
    _validate_runtime_parameters(value.inputs[0], flux)
    return flux


def _validate_runtime_parameters(state, flux):
    """A runtime parameter that the body reads is a RuntimeParam declared by the model of the
    block of ``state``, as for the body of a ``riemann.User`` face: the generated face reads its
    current value from ``ctx.program_params(block)``, so the value is not in the identity. Any
    other parameter is refused by its name (N3 lot 1)."""
    from pops.model.stencil_expression import runtime_parameters
    block = getattr(state.state_ref, "block_ref", None)
    for node in runtime_parameters(flux.rows):
        handle = node.handle
        if handle is None:
            raise ValueError("stencil flux reads the runtime parameter %r without its declaring "
                             "handle" % node.name)
        if handle.param_kind != "runtime":
            raise ValueError("stencil flux reads %r, a %s parameter; a stencil body reads a "
                             "RuntimeParam" % (node.name, handle.param_kind))
        declaration = handle.declaration_ref if handle.is_instance else handle
        if block is None or not _declares(block, declaration):
            raise ValueError("stencil flux reads the runtime parameter %r, which the model of the "
                             "block of its state does not declare" % node.name)


def _validate_parameters(parameters, flux):
    if len(parameters) != len(flux.parameters):
        raise ValueError("stencil scalar parameter arity differs from its declaration")
    if any(value.vtype != "scalar" for value in parameters):
        raise TypeError("stencil parameters require native Float64 scalar values")


def _validate_fields(state, fields, flux):
    """Authenticate numerical SSA samples and physical solved observations separately.

    A midpoint sample keeps its own producer and exact point: it is an explicit
    numerical argument, so a provisional stage may inform a flux on an earlier state.
    Physical observations retain their same-stage/exact-solve-source contract.
    """
    from pops.fields._observation_contract import validate_field_observation
    from pops.model import FieldSpace
    from pops.time._evaluation_point import evaluation_stage_fraction
    if len(fields) != len(flux.field_components):
        raise ValueError("stencil field input arity differs from its closed declaration")
    for field, width in zip(fields, flux.field_components, strict=True):
        if (field_role(field) != "field" or type(field.space) is not FieldSpace
                or len(field.space.components) != width
                or field.space.centering != "cell" or field.space.representation != "field"
                or any(unit is None for unit in field.space.units)):
            raise ValueError("stencil field requires a declared cell-sampled physical FieldSpace")
        for attribute in ("layout", "centering", "frame", "clock", "support"):
            if getattr(field.space, attribute) != getattr(state.space, attribute):
                raise ValueError("stencil field co-location differs on " + attribute)
        if field.attrs.get("cell_field_projection") == "pops.cell-midpoint-sampling@1":
            from pops.time._program.spatial_interaction import validate_cell_field
            from pops.time._program.value_validation import require_owned
            producer = validate_cell_field(field)
            require_owned(state.prog, producer, "stencil computed field producer")
            if (field.prog is not state.prog or field.block != state.block
                    or producer.state_ref is None or producer.clock != state.clock):
                raise ValueError("stencil computed field changed its exact Program/block/clock authority")
            # validate_cell_field checks its sampling map, components, representation metadata
            # and point against the actual producer. Do not relabel it with state.point.
            continue
        if field.space.sampling != "cell":
            raise ValueError("stencil field requires an authenticated observation or midpoint SSA sample")
        if evaluation_stage_fraction(field) != evaluation_stage_fraction(state):
            raise ValueError("stencil field belongs to a different evaluation stage")
        _packed, _selected, solve = validate_field_observation(field)
        if solve.op == "solve_spatial_field":
            raise NotImplementedError("stencil input needs an authenticated Uniform field buffer")
        load = solve.inputs[1]
        if load.op != "field_problem_load" or not any(
                source.id == state.id and source.block == state.block
                and source.state_ref == state.state_ref for source in load.inputs):
            raise ValueError("stencil field was not solved from this exact State")


def _declares(block, declaration):
    """Whether the model of ``block`` declares ``declaration``, in its authoring form or in the
    resolved form that a resolved Program carries."""
    if block.accepts(declaration):
        return True
    try:
        resolved = declaration._resolved()
    except Exception:  # an authoring handle of no resolved model is not the block's
        return False
    return resolved is not None and block.accepts(resolved)
