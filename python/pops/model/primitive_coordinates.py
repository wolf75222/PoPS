"""Immutable joint coordinates and their explicitly authored conservative inverse."""
from dataclasses import dataclass, replace
from pops._ir import Var, _wrap
from pops._ir.quantity import QuantityRef
from pops._ir.visitors import _children
from pops._ir.primitive_expansion import expand_primitive_recipes


@dataclass(frozen=True, slots=True)
class PrimitiveCoordinates:
    states: tuple
    coordinates: tuple
    forward: tuple
    inverse: tuple
    roles: tuple = ()
    constraints: tuple = ()
    __pops_ir_immutable__ = True

    @property
    def names(self):
        return tuple(q.component if isinstance(q, QuantityRef) else q.name for q in self.coordinates)

    def to_data(self):
        from pops.model.hash_data import body_identity, canonical_hash_data
        return {'schema_version': 1, 'kind': 'joint_primitive_coordinates',
                'states': [state.space.to_data() for state in self.states],
                'names': self.names, 'roles': canonical_hash_data(self.roles), 'forward': body_identity(self.forward),
                'inverse': body_identity(self.inverse),
                'constraints': [(index,body_identity(body)) for index,body in self.constraints]}


def _walk(roots):
    stack,seen=list(roots),set()
    while stack:
        node=stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(_children(node))


def _validate_parameters(module, roots):
    from pops._ir.values import RuntimeParamRef
    for node in _walk(roots):
        if isinstance(node,RuntimeParamRef):
            if node.handle.is_instance:
                raise ValueError('primitive coordinates require declaration-owned parameters')
            try:
                module._param_registry.handle(node.handle)
            except (KeyError,ValueError) as exc:
                raise ValueError('primitive coordinates capture another physical model') from exc


def declare_joint_coordinates(model, components, conservative, states, roles):
    selected=tuple(model._states.values()) if states is None else tuple(states)
    if not selected or len({id(state) for state in selected}) != len(selected):
        raise ValueError('joint primitive coordinates require distinct exact state declarations')
    if any(not any(state.owner_path == owned.owner_path and state.space is owned.space
                   for owned in model._states.values()) for state in selected):
        raise ValueError('joint primitive state belongs to another physical model')
    values=tuple(components)
    conservative=tuple(conservative)
    count=sum(len(state.components) for state in selected)
    if len(values) != count or len(conservative) != count:
        raise ValueError('joint primitive coordinates and explicit inverse must match the complete group')
    owned=tuple(q for state in selected for q in state)+tuple(model._primitive_vars.values())
    def is_owned(q):
        return any(q is value or (isinstance(q,QuantityRef) and isinstance(value,QuantityRef)
                   and q.handle == value.handle and q.space is value.space and q.index == value.index)
                   for value in owned)
    if any(not isinstance(q,(Var,QuantityRef)) or not is_owned(q) for q in values):
        raise ValueError('joint primitive coordinates require exact variables owned by this model')
    names=tuple(q.component if isinstance(q,QuantityRef) else q.name for q in values)
    if len(set(names)) != len(names):
        raise ValueError('joint primitive coordinate names must be distinct')
    from pops.physics._board_contract import normalize_roles
    role_map=normalize_roles(roles,names,'joint primitive coordinates')
    inverse=tuple(_wrap(q) for q in conservative)
    for node in _walk(inverse):
        if isinstance(node,(Var,QuantityRef)) and not any(node is q for q in values):
            raise ValueError('joint primitive inverse reads an unselected or foreign coordinate')
    forward=expand_primitive_recipes(values,model._dsl._m.prim_defs)
    module=model._multi_module
    handles=tuple(module.state_handle(state.space) for state in selected)
    if len(set(handles)) != len(handles):
        raise ValueError('joint primitive coordinates repeat an exact state declaration')
    for node in _walk(forward):
        if isinstance(node,Var):
            raise ValueError('joint primitive recovery contains an unbound scalar variable')
        if isinstance(node,QuantityRef) and node.handle not in handles:
            raise ValueError('joint primitive recovery reads an unsampled state')
    _validate_parameters(module,(*forward,*inverse))
    record=PrimitiveCoordinates(handles,values,tuple(forward),inverse,
                                roles=tuple(role_map.get(name) for name in names))
    previous=module.primitive_coordinates()
    if any(set(row.states)&set(handles) for row in previous):
        raise ValueError('a state already belongs to declared primitive coordinates')
    module._set_primitive_coordinates((*previous,record))
    model._invalidate_authoring_views()


def declare_joint_admissibility(model, constraints, states):
    module=model._multi_module
    records=module.primitive_coordinates()
    candidates=[row for row in records if (states is None or
        tuple(module.state_handle(state.space) for state in states) == row.states)
        and set(constraints) <= set(row.names)]
    if len(candidates) != 1 or not constraints:
        raise ValueError('joint admissibility requires one exact primitive coordinate group')
    row=candidates[0]
    if row.constraints:
        raise ValueError('joint primitive admissibility is already declared')
    result=[]
    for name,predicate in constraints.items():
        predicate=_wrap(predicate)
        if not callable(getattr(predicate,'resolve_for_amr_predicate',None)):
            raise TypeError('joint admissibility requires a typed Boolean expression')
        for node in _walk((predicate,)):
            if isinstance(node,(Var,QuantityRef)) and not any(node is q for q in row.coordinates):
                raise ValueError('joint admissibility reads an unselected primitive coordinate')
        result.append((row.names.index(name),predicate))
    _validate_parameters(module,tuple(body for _,body in result))
    replacement=replace(row,constraints=tuple(sorted(result)))
    module._set_primitive_coordinates(tuple(replacement if item is row else item for item in records))
    model._invalidate_authoring_views()
