"""Declared finite-quadrature separating covectors; authoring algebra only."""
from dataclasses import dataclass
from fractions import Fraction
import math
from .discrete_entropy import DiscreteEntropyQuadrature

@dataclass(frozen=True, init=False)
class DiscreteEntropyCertificate:
    """One nonnegative support covector and its strict finite-dual obligation.

    Contract pops.discrete-entropy-certificate@2. A negative target margin is
    certified outside the cone. A zero margin is compatible with a boundary
    measure but incompatible with strictly positive exponential populations.
    No claim that these declared certificates enumerate every cone facet.
    """
    quadrature: DiscreteEntropyQuadrature
    covector: tuple[float, ...]
    label: str

    def __init__(self, quadrature, covector, *, label):
        if type(quadrature) is not DiscreteEntropyQuadrature:
            raise TypeError('certificate requires exact declared quadrature')
        vector = tuple(covector)
        if len(vector) != quadrature.moment_count or any(type(v) not in (int,float) or not math.isfinite(v) for v in vector):
            raise ValueError('certificate covector requires finite basis coefficients')
        if type(label) is not str or not label.isidentifier():
            raise ValueError('certificate label requires an identifier')
        # A binary rescaling must preserve EVERY declared coefficient exactly.
        largest = max(abs(float(c)) for c in vector)
        if largest == 0:
            raise ValueError('certificate direction is zero')
        shift = 1 - math.frexp(largest)[1]
        scaled = tuple(math.ldexp(float(c), shift) for c in vector)
        factor = Fraction(2) ** shift
        if any(Fraction(v) != Fraction(c)*factor for c,v in zip(vector,scaled,strict=True)):
            raise ValueError('certificate direction cannot be normalized exactly in binary64')
        vector = scaled
        # Exact signs of the declared binary numbers: no hidden feasibility tolerance.
        margins = tuple(sum(Fraction(c)*Fraction(row[j]) for c,row in zip(vector,quadrature.basis,strict=True)) for j in range(quadrature.node_count))
        if any(v < 0 for v in margins) or not any(v > 0 for v in margins):
            raise ValueError('certificate must be nonnegative at ALL nodes and positive at some node')
        object.__setattr__(self,'quadrature',quadrature)
        object.__setattr__(self,'covector',vector)
        object.__setattr__(self,'label',label)

    def margin(self, target):
        values = tuple(target)
        if len(values) != self.quadrature.moment_count:
            raise ValueError('certificate target basis width differs')
        return sum(c*v for c,v in zip(self.covector,values,strict=True))

    def guard_finite_dual(self, program, seed, target):
        """Native collective guards before solving; never change target or seed."""
        from pops.time import FailRun
        values = tuple(target)
        margin = self.margin(values)
        # Conservative forward-error envelope for binary64 products and summation.
        # Unresolved cancellation is NOT a proof of an exact boundary.
        n = len(values)
        u = 2.0**-53
        k = 2*n + 4
        if k*u >= 1:
            raise ValueError('certificate arithmetic error bound is not representable')
        gamma = k*u/(1-k*u)
        error = gamma*sum(abs(c*v) for c,v in zip(self.covector,values,strict=True)) + n*math.ulp(0.0)
        from pops._ir.expr import Expr
        from pops._ir.control_expr import Where
        limit = float.fromhex('0x1.fffffffffffffp+1023')
        def indicator(value):
            predicate = abs(value) <= limit
            # Native Where converts NaN comparisons to zero before reduction.
            return Where(predicate,1.,0.) if isinstance(predicate,Expr) else predicate*1.
        finite = indicator(margin)*indicator(error)
        for c,v in zip(self.covector,values,strict=True):
            finite = finite*indicator(v)*indicator(c*v)
        field = program.value(self.label+'_finite',(finite,)*n)
        checked = program.guard(self.label+'_arithmetic_indeterminate',seed,program.min(field) >= 1.,action=FailRun())
        # Keep each error paired with the margin that produced it.
        lower = program.value(self.label+'_lower_margin',(margin+error,)*n)
        upper = program.value(self.label+'_upper_margin',(margin-error,)*n)
        guarded = program.guard(self.label+'_target_infeasible',checked,program.min(lower) >= 0.,action=FailRun())
        return program.guard(self.label+'_finite_dual_not_certified',guarded,program.min(upper) > 0.,action=FailRun())

    def to_data(self):
        return {'contract':'pops.discrete-entropy-certificate@2','nodes':list(self.quadrature.nodes),
                'weights':list(self.quadrature.weights),'basis':[list(r) for r in self.quadrature.basis],
                'covector':list(self.covector),'label':self.label}
