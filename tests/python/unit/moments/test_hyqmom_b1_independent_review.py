"""Independent exact Gaussian linearization of the published Eq. (B.1)."""
from fractions import Fraction as Q
from functools import lru_cache
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def example(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"examples/migration/scientific"/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(None)
def gaussian(p,q):
    if p<0 or q<0:return Q(0)
    if p+q==0:return Q(1)
    # Wick recurrence for covariance [[1,1/2],[1/2,1]], without production transforms.
    if p:return (p-1)*gaussian(p-2,q)+Q(q,2)*gaussian(p-1,q-1)
    return (q-1)*gaussian(0,q-2)


def exact_jacobians():
    indices=tuple((p,q) for q in range(5) for p in range(5-q))
    def basis(index):return [Q(int(item==index)) for item in indices]
    def add(*terms):return [sum(values) for values in zip(*terms)]
    def scale(a,v):return [a*x for x in v]
    def ds(p,q):
        moment=gaussian(p,q)
        return add(basis((p,q)),scale(-moment,basis((0,0))),
            scale(-p*gaussian(p-1,q),basis((1,0))),
            scale(-q*gaussian(p,q-1),basis((0,1))),
            scale(-Q(p,2)*moment,add(basis((2,0)),scale(-1,basis((0,0))))),
            scale(-Q(q,2)*moment,add(basis((0,2)),scale(-1,basis((0,0))))))
    # Differentiated directly from the three formulas on published p36, B.1;
    # here S30=S03=S21=S12=0, S40=S04=3, S11=1/2, S31=S22=S13=3/2.
    top={(5,0):scale(7,ds(3,0)),
         (4,1):add(scale(Q(1,2),ds(3,0)),scale(6,ds(2,1))),
         (3,2):add(scale(3,ds(1,2)),scale(Q(7,4),ds(3,0))),
         (2,3):add(scale(3,ds(2,1)),scale(Q(7,4),ds(0,3))),
         (1,4):add(scale(Q(1,2),ds(0,3)),scale(6,ds(1,2))),
         (0,5):scale(7,ds(0,3))}
    for (p,q),derivative in tuple(top.items()):
        top[p,q]=add(derivative,scale(p*gaussian(p-1,q),basis((1,0))),
                    scale(q*gaussian(p,q-1),basis((0,1))))
    def derivative(index):return basis(index) if sum(index)<=4 else top[index]
    return indices,[[derivative((p+1,q)) for p,q in indices],
                    [derivative((p,q+1)) for p,q in indices]]


def multiply(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def characteristic(matrix):
    # Exact Faddeev-LeVerrier, independent of NumPy eigenvalue/np.poly paths.
    n=len(matrix);identity=[[Q(int(i==j)) for j in range(n)] for i in range(n)]
    b=identity;coefficients=[Q(1)]
    for k in range(1,n+1):
        product=multiply(matrix,b)
        coefficient=-sum(product[i][i] for i in range(n))/k
        coefficients.append(coefficient)
        b=[[product[i][j]+coefficient*identity[i][j] for j in range(n)] for i in range(n)]
    assert all(value==0 for row in b for value in row)
    return coefficients


def test_all_flux_indices_and_oblique_polynomial_match_exact_independent_derivative():
    indices,(jx,jy)=exact_jacobians()
    witness=example("api040_m16_hyqmom_b1_oblique").oblique_witness()
    assert witness["classification"]=="MATH"
    np.testing.assert_array_equal([float(gaussian(p,q)) for p,q in indices],
        example("api040_m16_hyqmom_b1_oblique").CORRELATED_GAUSSIAN)
    np.testing.assert_allclose(witness["jx"],np.array(jx,dtype=float),rtol=0,atol=3e-14)
    np.testing.assert_allclose(witness["jy"],np.array(jy,dtype=float),rtol=0,atol=3e-14)
    oblique=[[a-b for a,b in zip(x,y)] for x,y in zip(jx,jy)]
    assert characteristic(oblique)==[Q(1),Q(0),Q(-67,2),Q(0),Q(5641,16),Q(0),
        Q(-1422),Q(0),Q(9747,4),Q(0),Q(-1593),Q(0),Q(4131,16),Q(0),Q(0),Q(0)]
    # Exact realizability witness: positive covariance and all Gram LDL pivots.
    basis=((0,0),(1,0),(0,1),(2,0),(1,1),(0,2))
    gram=[[gaussian(p+i,q+j) for i,j in basis] for p,q in basis]
    for i in range(6):
        pivot=gram[i][i]
        assert pivot>0
        for r in range(i+1,6):
            for c in range(i+1,6):gram[r][c]-=gram[r][i]*gram[i][c]/pivot


def test_axial_source_model_enforces_its_declared_realizable_domain():
    case,_,_=example("api040_m15_hyqmom_axial_b1").build_case()
    model=case._block_registry.spec("moments")["model"]
    constraints=getattr(model._dsl._m,"_recovery_admissibility",{})
    assert constraints, "M15 must not accept negative-density Gaussian moments by finiteness alone"

    cases=[((1,0,1,0,3),True),((-1,0,-1,0,-3),False),
           ((1,0,1,0,.5),False),((1,0,0,0,0),False),((1,0,1,0,1),True)]
    for raw,expected in cases:
        environment={q.qualified_id:value for q,value in zip(model.states["M"],raw)}
        assert all(bool(predicate.eval(environment)) for predicate in constraints.values()) is expected
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.module_codegen import _emit_bricks
    import pops
    case,layout,_=example("api040_m15_hyqmom_axial_b1").build_case()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    _,source,_=_emit_bricks(ProgramModelGraph.from_resolved_blocks(resolved.blocks).model_for_block("moments")._m)
    assert "recovery_admissible(result.value" in source


def test_non_gaussian_cross_moment_distinguishes_b1_from_equation_40():
    from math import comb, factorial
    from pops._ir.expr import Var
    from pops.moments import moment_flux_expressions, HyQMOM15Closure
    def normal(k,mean,variance):
        return sum(Q(comb(k,2*j)*factorial(2*j),2**j*factorial(j))*variance**j*mean**(k-2*j)
                   for j in range(k//2+1))
    indices=tuple((p,q) for q in range(5) for p in range(5-q))
    # A positive mixture of full-rank independent Gaussians; its mean is exactly zero.
    components=((Q(3,10),Q(-7,10),Q(1,5),Q(2,5)),
                (Q(7,10),Q(3,10),Q(3,5),Q(1,10)))
    moments={(p,q):sum(w*normal(p,mean,vx)*normal(q,mean,vy)
                        for w,mean,vx,vy in components) for p,q in indices}
    assert moments[1,0]==moments[0,1]==0
    class Author:
        def primitive(self,name,body):return body
    names=tuple("M%d%d"%index for index in indices)
    flux=moment_flux_expressions(Author(),tuple(Var(name,"cons") for name in names),
                                4,HyQMOM15Closure(),robust=False)
    environment={name:float(moments[index]) for name,index in zip(names,indices)}
    for swap in (False,True):
        m={(p,q):moments[q,p] if swap else moments[p,q] for p,q in indices}
        # B.1 S32 converted algebraically to raw moments at zero mean, without square roots.
        b1=(m[4,0]*m[1,2]/m[2,0]-Q(3,2)*m[3,0]**2*m[1,2]/m[2,0]**2
            +Q(3,2)*m[2,2]*m[3,0]/m[2,0]-Q(1,2)*m[3,0]*m[0,2])
        correction=m[0,3]/m[0,2]*(m[3,1]-m[4,0]*m[1,1]/m[2,0]
            +Q(3,2)*m[3,0]**2*m[1,1]/m[2,0]**2-Q(3,2)*m[3,0]*m[2,1]/m[2,0])
        assert abs(float(correction))>1e-4
        selected=flux.y if swap else flux.x
        actual=selected[indices.index((2,2))].eval(environment)
        assert abs(actual-float(b1))<2e-14
        assert abs(actual-float(b1+correction))>1e-4
