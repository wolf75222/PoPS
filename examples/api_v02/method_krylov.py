"""Krylov equations authored entirely with the ordinary native Program algebra."""
from pops.fields import CellMidpoint
from pops.time.expressions import ProgramExpression, component_names


def laplacian(program, state, *, name):
    """Explicit midpoint sampling and the existing ghost-filled Cartesian stencil."""
    quadrature=CellMidpoint()
    field=program.cell_field(state,quadrature=quadrature,name=name+"_point_values")
    applied=program.laplacian(program.scalar_field(name+"_work",like=field),field)
    return program.cell_mean(applied,template=state,quadrature=quadrature,name=name)


def conjugate_gradient(program,rhs,apply,*,name,tolerance=1.e-11,max_steps=100):
    """CG for an explicitly supplied SPD operator and native collective product.

    The caller owns positive definiteness and compatible domain/boundary data.
    This construction performs no algorithm registration and emits no C++ text.
    Residual and direction values retain their state support and version.
    """
    zero=program.value(name+"_zero",ProgramExpression((0.,)*len(component_names(rhs)),rhs),at=rhs.point)
    initial=program.product(x=zero,r=rhs,d=rhs)
    def step(p,state):
        x,r,d=state["x"],state["r"],state["d"]
        Ad=apply(p,d)
        rr=p.dot_all(r,r)
        alpha=rr/p.dot_all(d,Ad)
        new_x=p.value(name+"_x",x+alpha*d,at=x.point)
        new_r=p.value(name+"_r",r-alpha*Ad,at=r.point)
        beta=p.dot_all(new_r,new_r)/rr
        new_d=p.value(name+"_d",new_r+beta*d,at=d.point)
        return p.product(x=new_x,r=new_r,d=new_d)
    result=program.iterate(initial,step,
        lambda p,state:p.dot_all(state["r"],state["r"])<tolerance*tolerance,max_steps=max_steps)
    candidate=program.require_success(result)["state"]["x"]
    true_residual=program.value(name+"_true_residual",rhs-apply(program,candidate),at=rhs.point)
    program.record_scalar(name+"_true_residual_squared",program.dot_all(true_residual,true_residual))
    return candidate
