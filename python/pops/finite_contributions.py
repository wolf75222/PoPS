"""Finite inventory construction, using only the public native graph algebra.

No routine here evaluates a numerical value in Python. A finite interval is
scientific data: the parent Assess must compare it with its requested window.
It is not an independently issued permission to publish a runtime frontier.
"""
from dataclasses import dataclass, field

from pops.time.values import ProgramValue
from pops.time.variant_types import Failure, Float64, Product, Result, type_of
from pops.time._program.value_validation import require_owned
from pops.time._program.structured_control import value_signature, same_signature

InventoryErrors = Failure(inventory_domain=Product(begin=Float64, end=Float64))


def inventory(value, where):
    if not isinstance(value, ProgramValue):
        raise TypeError(where + " requires native finite inventory values")
    require_owned(value.prog, value, where)
    def check(signature):
        kind = signature['vtype']
        if kind == 'array':
            check(signature['element'])
        elif kind == 'product':
            for _, child in signature['members']:
                check(child)
        elif kind != 'scalar':
            raise TypeError(where + " requires a real vector space, not discrete geometry")
    check(value_signature(value))
    return value


def checked(p, value, condition, *, begin=None, end=None):
    errors = InventoryErrors
    result = Result(type_of(value), errors)
    begin = p.float64(0) if begin is None else begin
    end = p.float64(0) if end is None else end
    return p.require_success(p.branch(condition,
        lambda q: q.variant(result, 'ok', value),
        lambda q: q.variant(result, 'err', q.variant(errors, 'inventory_domain',
                                                    q.product(begin=begin, end=end)))))


@dataclass(frozen=True, eq=False)
class FiniteInterval:
    """An explicit native scientific window, retained with its historical value."""
    begin: ProgramValue
    end: ProgramValue
    clock: str

    def __post_init__(self):
        require_owned(self.begin.prog, self.begin, 'finite interval begin', vtype='scalar')
        require_owned(self.begin.prog, self.end, 'finite interval end', vtype='scalar')
        if type(self.clock) is not str or not self.clock:
            raise TypeError('finite interval clock requires an explicit identifier')

    @property
    def program(self): return self.begin.prog

    @property
    def role(self): return 'scientific'

    @property
    def duration(self): return self.end - self.begin

    def to_data(self):
        return {'schema_version': 1, 'role': self.role, 'clock': self.clock,
                'begin': self.begin.id, 'end': self.end.id,
                'program': self.program.owner_path.canonical().to_data()}

    def bind(self, program, value, *, name):
        require_owned(program, value, 'finite interval delivery')
        if program is not self.program:
            raise ValueError('finite interval belongs to another Program')
        return checked(program, value, self.end >= self.begin, begin=self.begin, end=self.end)

    def pack(self, program):
        if program is not self.program:
            raise ValueError('finite interval belongs to another Program')
        return program.product(begin=self.begin, end=self.end)

    @classmethod
    def unpack(cls, value, *, clock):
        return cls(value['begin'], value['end'], clock)


@dataclass(frozen=True, eq=False)
class LinearMap:
    """The differential D(function)(at), linear in the selected direction.

    The original function need not be linear. This constructor defines the
    action as its actual graph differential, rather than trusting a claim of
    linearity. Primal parameters and topology are fixed by immutable SSA.
    """
    function: object
    at: ProgramValue
    wrt: str
    transport_byte_capacity: int | None = None
    differentiate_iterations: bool = False
    action: object = field(init=False)

    @classmethod
    def differential(cls, function, *, at, wrt, differentiate_iterations=False,
                     transport_byte_capacity=None):
        return cls(function, at, wrt, transport_byte_capacity, differentiate_iterations)

    def __post_init__(self):
        from pops.graph_functions import GraphFunction
        from pops.graph_derivatives import jvp
        function, at, wrt = self.function, self.at, self.wrt
        if type(function) is not GraphFunction or not isinstance(at, ProgramValue):
            raise TypeError('LinearMap.differential requires a closed graph and exact primal Product')
        if not function.inputs.matches(at) or at.vtype != 'product':
            raise TypeError('linear map fixed point differs from the graph input type')
        if wrt not in dict(value_signature(at)['members']):
            raise ValueError('linear map direction is absent from its graph inputs')
        inventory(at[wrt], 'linear map selected primal')
        object.__setattr__(self, 'action', jvp(function,
            differentiate_iterations=self.differentiate_iterations,
            transport_byte_capacity=self.transport_byte_capacity))

    def __call__(self, program, value):
        from pops._finite_differential import zero_like, with_transport
        require_owned(program, self.at, 'linear map immutable point')
        require_owned(program, value, 'linear map direction')
        @with_transport(self.transport_byte_capacity)
        def build(p):
            tangent = p.product(**{key: value if key == self.wrt else zero_like(p, self.at[key])
                                   for key, _ in value_signature(self.at)['members']})
            return self.action(p, primal=self.at, tangent=tangent)['tangent']
        return build(program)

    def to_data(self):
        return {'kind': 'graph_differential', 'function': self.function.identity,
                'action': self.action.identity, 'at': self.at.id, 'wrt': self.wrt,
                'transport_byte_capacity': self.transport_byte_capacity}


def recipient_key(value):
    return value.state_ref if value.is_field() else ('finite', value.id)


def add(program, left, right, *, byte_capacity=None):
    """Coordinate addition with exact live domain checks, including empty arrays."""
    inventory(left, 'inventory addition'); inventory(right, 'inventory addition')
    if not same_signature(value_signature(left), value_signature(right)):
        raise TypeError('inventory addition requires one finite representation')
    if left.vtype == 'scalar':
        return left + right
    if left.vtype == 'product':
        return program.product(**{key: add(program, left[key], right[key], byte_capacity=byte_capacity)
                                  for key, _ in value_signature(left)['members']})
    domain = program.domain(left)
    right = checked(program, right, program.same_domain(domain, program.domain(right)))
    options = {'byte_capacity': byte_capacity} if value_signature(left)['ownership'] != 'replicated' else {}
    return program.map(domain, lambda p, i: add(p, p.at(left, i), p.at(right, i),
                                               byte_capacity=byte_capacity), **options)


def scale(program, value, factor, *, byte_capacity=None):
    inventory(value, 'inventory weighting')
    if value.vtype == 'scalar':
        return value * factor
    if value.vtype == 'product':
        return program.product(**{key: scale(program, value[key], factor, byte_capacity=byte_capacity)
                                  for key, _ in value_signature(value)['members']})
    options = {'byte_capacity': byte_capacity} if value_signature(value)['ownership'] != 'replicated' else {}
    return program.map(program.domain(value), lambda p, i:
        scale(p, p.at(value, i), factor, byte_capacity=byte_capacity), **options)


def apply_increment(increment, program, base, *, name, byte_capacity=None):
    inventory(base, 'finite increment base')
    selected = tuple(d for d in increment.deliveries if recipient_key(d.recipient) == recipient_key(base))
    if increment.contributions and not selected:
        raise ValueError('increment has no delivery for this exact finite recipient')
    from pops.ledger_provenance import increment_contract, realize_application, bind_application
    value = realize_application(program, increment_contract(increment), base, byte_capacity=byte_capacity)
    value = bind_application(program, value, increment, base, byte_capacity=byte_capacity)
    return program._replace_value(value, attrs=dict(value.attrs,
        increment_v02=increment.to_data(), increment_identity_v02=increment.identity.token))


__all__ = ['FiniteInterval', 'LinearMap', 'InventoryErrors']
