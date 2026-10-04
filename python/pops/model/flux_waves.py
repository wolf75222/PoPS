"""Exact output-State authority for a flux's authored eigenvalue law (@1 spectra; additive @2 signed bounds)."""
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from pops._cartesian_axes import canonical_axis_mapping
from .hash_data import canonical_hash_data


@dataclass(frozen=True)
class FluxWaveLaw:
    output_state: object
    values: object
    signed_bounds: object = None

    def __post_init__(self):
        from .spaces import StateSpace
        from pops._ir.expr import Expr, _wrap
        if type(self.output_state) is not StateSpace:
            raise TypeError('flux wave authority requires an exact output StateSpace')
        if not isinstance(self.values, Mapping):
            raise TypeError('flux wave values must be an axis mapping')
        values = canonical_axis_mapping(self.values, where='flux wave law')
        normalized = {}
        for axis, expressions in values.items():
            if type(expressions) is not tuple or len(expressions) != len(self.output_state.components):
                raise ValueError('flux wave law must cover every output-State component')
            expressions = tuple(_wrap(value) for value in expressions)
            if any(not isinstance(value, Expr) for value in expressions):
                raise TypeError('flux wave law requires symbolic expressions')
            normalized[axis] = expressions
        object.__setattr__(self, 'values', MappingProxyType(normalized))
        if self.signed_bounds is not None:
            if not isinstance(self.signed_bounds, Mapping):
                raise TypeError('signed flux bounds must be an axis mapping')
            bounds = canonical_axis_mapping(self.signed_bounds, where='signed flux bounds')
            if tuple(bounds) != tuple(normalized):
                raise ValueError('signed flux bounds must cover the exact spectrum axes')
            pairs = {}
            for axis, pair in bounds.items():
                if type(pair) is not tuple or len(pair) != 2:
                    raise ValueError('signed flux bounds require one (lower, upper) pair per axis')
                pairs[axis] = tuple(_wrap(value) for value in pair)
                if any(not isinstance(value, Expr) for value in pairs[axis]):
                    raise TypeError('signed flux bounds require symbolic expressions')
            object.__setattr__(self, 'signed_bounds', MappingProxyType(pairs))


    def to_data(self):
        data = dict(contract='pops.flux-wave-law@1', output_state=self.output_state.to_data(),
                    values=canonical_hash_data(self.values, where='flux wave expressions'))
        if self.signed_bounds is not None:
            data['contract'] = 'pops.flux-wave-law@2'
            data['signed_bounds'] = canonical_hash_data(self.signed_bounds, where='signed flux bounds')
        return data

    def declaration_references(self):
        from pops._ir.expr_references import collect_reference_value
        references = []
        collect_reference_value(self.values, references, set())
        collect_reference_value(self.signed_bounds, references, set())
        return tuple(references)


def flux_waves(module, operator):
    """Authenticate the exact operator/output pair, then select its law or legacy default."""
    registered = module.operator_registry().get(operator.name)
    if registered is not operator or operator.kind != 'grid_operator':
        raise ValueError('flux waves require a registered grid operator')
    law = operator.lowering.get('flux_wave_law')
    if law is None:
        return module._eigenvalues
    if type(law) is not FluxWaveLaw or law.output_state != operator.signature.output.base_space:
        raise ValueError('flux wave law output State differs from its operator signature')
    if isinstance(operator.body, Mapping) and tuple(law.values) != tuple(canonical_axis_mapping(operator.body, where='flux body')):
        raise ValueError('flux wave law axes differ from physical flux')
    return law.values


def common_flux_waves(module, names, *, global_authority=False):
    """A shared native view must have one unambiguous law, never a first-State guess."""
    if global_authority and module._eigenvalues is not None:
        return module._eigenvalues
    laws = [flux_waves(module, module.operator_registry().get(name)) for name in sorted(names)]
    present = [law for law in laws if law is not None]
    if not present:
        return None
    if len(present) != len(laws) or any(canonical_hash_data(law) != canonical_hash_data(present[0]) for law in present[1:]):
        raise ValueError('selected fluxes have ambiguous wave laws; declare an explicit joint wave authority')
    return present[0]


def common_flux_signed_bounds(module, names):
    """Select signed pairs from the same exact flux/output-State law, never another State."""
    laws = []
    for name in sorted(names):
        operator = module.operator_registry().get(name)
        flux_waves(module, operator)  # authenticate the registered operator and its output State
        law = operator.lowering.get('flux_wave_law')
        laws.append(None if law is None else law.signed_bounds)
    present = [law for law in laws if law is not None]
    if not present:
        return None
    if len(present) != len(laws) or any(canonical_hash_data(law) != canonical_hash_data(present[0]) for law in present[1:]):
        raise ValueError('selected fluxes have ambiguous signed bounds; declare one exact shared pair law')
    return present[0]


def validate_flux_wave_data(value, signature):
    """Validate the detached manifest codec; it never reconstructs executable expressions."""
    if not isinstance(value, Mapping):
        raise ValueError('flux wave codec requires a mapping')
    signed = value.get('contract') == 'pops.flux-wave-law@2'
    keys = {'contract','output_state','values'} | ({'signed_bounds'} if signed else set())
    if set(value) != keys:
        raise ValueError('flux wave codec requires exact versioned fields')
    if value['contract'] not in ('pops.flux-wave-law@1', 'pops.flux-wave-law@2'):
        raise ValueError('unsupported flux wave authority contract')
    output = signature.get('output', {})
    if output.get('kind') != 'rate' or value['output_state'] != output.get('base_space'):
        raise ValueError('flux wave codec output State differs from operator signature')
    axes = canonical_axis_mapping(value['values'], where='flux wave codec')
    count = len(value['output_state']['components'])
    if not axes:
        raise ValueError('flux wave codec has no ranked axes')
    for graph in axes.values():
        if not isinstance(graph, Mapping) or set(graph) != {'protocol','nodes','roots'} or graph['protocol'] != 'pops.expr.dag.v1':
            raise ValueError('flux waves require canonical expression DAG @1')
        nodes, roots = graph['nodes'], graph['roots']
        if not isinstance(nodes, (list,tuple)) or not nodes or not isinstance(roots,(list,tuple)) or len(roots) != count:
            raise ValueError('flux wave codec component shape is invalid')
        if any(type(root) is not int or not 0 <= root < len(nodes) for root in roots):
            raise ValueError('flux wave codec root is invalid')

    if signed:
        pairs = canonical_axis_mapping(value['signed_bounds'], where='signed flux codec')
        if tuple(pairs) != tuple(axes):
            raise ValueError('signed flux codec axes differ from the spectrum')
        for graph in pairs.values():
            if not isinstance(graph, Mapping) or set(graph) != {'protocol','nodes','roots'} or graph['protocol'] != 'pops.expr.dag.v1':
                raise ValueError('signed flux bounds require canonical expression DAG @1')
            nodes, roots = graph['nodes'], graph['roots']
            if not isinstance(nodes, (list,tuple)) or not nodes or not isinstance(roots,(list,tuple)) or len(roots) != 2:
                raise ValueError('signed flux codec requires exactly two roots per axis')
            if any(type(root) is not int or not 0 <= root < len(nodes) for root in roots):
                raise ValueError('signed flux codec root is invalid')
