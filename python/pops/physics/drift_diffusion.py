"""Potential-driven physical drift, independent of a fitted face construction."""
from dataclasses import dataclass
from pops._ir.expr import Expr, Partial, _wrap
from pops._ir.quantity import QuantityRef, _hash_owner
from pops.model import Handle, Signature, FieldSpace, DeclarationIndex
from pops.model.operators import Operator
from .diffusion import _authenticate_expression, _physical_boundaries, _law_inputs


@dataclass(frozen=True, slots=True, eq=False)
class DriftFluxLaw:
    state: Handle
    density: Expr
    mobility: Expr
    potential: Expr
    axes: tuple
    inputs: tuple
    boundaries: tuple
    __pops_ir_immutable__ = True

    @property
    def dimension(self):
        return len(self.axes)

    @property
    def expressions(self):
        return self.density,self.mobility,self.potential

    def declaration_references(self):
        from pops._ir.expr_references import collect_reference_value
        result=[self.state]
        collect_reference_value(self.expressions,result,set())
        return tuple(result)

    def resolve_references(self,resolver):
        from pops._ir.expr_references import resolve_reference_value
        density, mobility, potential = (resolve_reference_value(
            value,resolver,{},allow_formula_vars=True) for value in self.expressions)
        return DriftFluxLaw(resolver(self.state), density, mobility, potential,
                            self.axes,self.inputs,self.boundaries)

    def to_data(self):
        from pops._ir.balance import _handle_data
        from pops.model.hash_data import canonical_hash_data
        if not self.state.is_resolved and self.state.owner_path!=_hash_owner.get():
            return self.resolve_references(lambda handle:handle._resolved(handle.owner_path.canonical())).to_data()
        return {"kind":"potential_driven_drift_flux","state":_handle_data(self.state),
                "expressions":canonical_hash_data(self.expressions),"axes":self.axes,
                "inputs":[space.to_data() for space in self.inputs],
                "potential_boundaries":[row.to_data() for row in self.boundaries]}

    def flux_expressions(self):
        return tuple(-self.mobility*self.density*Partial(self.potential,axis)
                     for axis in range(len(self.axes)))


class DriftFluxHandle(Handle):
    __slots__=("reg_name","state","law")

    def __init__(self,name,law,*,owner):
        super().__init__(name,kind="drift_flux",owner=owner)
        object.__setattr__(self,"reg_name",name)
        object.__setattr__(self,"state",law.state)
        object.__setattr__(self,"law",law)


def declare_drift_flux(model,name,*,state,mobility,potential,boundaries=None):
    from ._board_contract import require_name
    from .board_handles import StateHandle, FieldHandle
    model._guard_mutable("declare potential-driven drift")
    name=require_name(name,"drift flux name")
    if not isinstance(state,StateHandle) or model._states.get(state.name)!=state or len(state.components)!=1:
        raise ValueError("drift flux requires this Model's exact scalar state")
    if model.frame is None:
        raise ValueError("drift requires an explicit physical Cartesian frame")
    if isinstance(potential,FieldHandle):
        if model._fields.get(potential.name)!=potential:
            raise ValueError("drift potential belongs to a foreign field declaration")
        space=model.field_spaces()[potential.name]
        if len(space.components)!=1:
            raise ValueError("drift potential must select one scalar field output")
        potential=QuantityRef(potential,space.components[0],space=space)
    potential,mobility=_wrap(potential),_wrap(mobility)
    for expression in (potential,mobility):
        _authenticate_expression(model,expression,state)
    if any(ref.kind=="state" for ref in potential.declaration_references()):
        raise ValueError("potential-driven drift requires an independent field observation")
    inputs=_law_inputs(model,state,(potential,mobility))
    existing=getattr(model,"_drift_fluxes",{})
    if name in existing or name in model._fluxes or name in getattr(model,"_diffusive_fluxes",{}):
        raise ValueError("physical flux name is already declared")
    law=DriftFluxLaw(state,state[0],mobility,potential,tuple(axis.name for axis in model.frame.axes),
                     tuple(inputs),_physical_boundaries(boundaries,len(model.frame.axes)))
    handle=DriftFluxHandle(name,law,owner=model.owner_path)
    model._drift_fluxes={**existing,name:handle}
    model._invalidate_authoring_views()
    return handle


def install_drift_fluxes(model,module):
    registry=module.operator_registry()
    for handle in getattr(model,"_drift_fluxes",{}).values():
        if handle.reg_name in registry.names():
            if registry.get(handle.reg_name).lowering.get("drift_law") is handle.law:
                continue
            raise ValueError("drift flux collides with a registered operator")
        output=FieldSpace(handle.name+"_flux",components=handle.law.axes,
                          layout="face",representation="physical_drift_flux")
        registry.register(Operator(handle.reg_name,"expression",Signature(handle.law.inputs,output),
            body=handle.law.flux_expressions(),lowering={"drift_law":handle.law},
            capabilities={"local":False,"produces_rate":False}))
        declarations=module._register_operator_binding_authority(DeclarationIndex(owner=model.owner_path,handles=(handle,)))
        module._bind_operator(handle,module.operator_handle(handle.reg_name),declarations=declarations)
