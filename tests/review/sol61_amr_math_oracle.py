"""SOURCE_ONLY arithmetic receipt; no PoPS/native imports or saved-data fabrication."""
import json
from math import sqrt

def q(a, b):
    return a+a*a+0.1*b*b, b+b*b+0.2*a*b

def conserved_b(a, total):
    linear = 1+0.2*a
    rhs = total-a-a*a
    return 2*rhs/(linear+sqrt(linear*linear+4*1.1*rhs))

def coupled_positive(target, total):
    lo, hi = 0.15, 0.20
    assert q(lo, conserved_b(lo,total))[0] < target
    assert q(hi, conserved_b(hi,total))[0] > target
    for _ in range(90):
        mid=(lo+hi)/2
        if q(mid,conserved_b(mid,total))[0] < target:
            lo=mid
        else:
            hi=mid
    a=(lo+hi)/2
    return a, conserved_b(a,total)

dt=0.01
initial=q(0.15,0.25)
total=sum(initial)
b1=conserved_b(0.16,total)
first=q(0.16,b1)
load=(first[0]-initial[0])/dt
second=(first[0]+dt*load,first[1]-dt*load)
a2,b2=coupled_positive(second[0],total)
assert max(abs(x-y) for x,y in zip(q(a2,b2),second)) < 1e-15
assert abs(sum(second)-total) < 1e-15
scalar_q0=0.15+0.15**2
scalar_q1=0.16+0.16**2
scalar_q2=2*scalar_q1-scalar_q0
scalar_t2=2*scalar_q2/(1+sqrt(1+4*scalar_q2))
assert abs(scalar_t2+scalar_t2**2-scalar_q2) < 1e-15
jacobian=((1+2*a2,0.2*b2),(0.2*b2,1+2*b2+0.2*a2))
det=jacobian[0][0]*jacobian[1][1]-jacobian[0][1]**2
assert jacobian[0][0]>0 and det>0
print(json.dumps(dict(identity="SOURCE_ONLY",scalar_second_T=scalar_t2,coupled_first_T=[0.16,b1],coupled_second_Q=second,coupled_second_T=[a2,b2],coupled_second_z=0.25*a2+0.5*b2,jacobian=jacobian,determinant=det),indent=2))
