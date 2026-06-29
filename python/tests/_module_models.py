"""Pure Module fixtures shared by clean-route runtime tests."""

from pops import model
from pops.ir.expr import Const, Var
from pops.ir.ops import sqrt
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.numerics.spatial import spatial as spatial_catalog
from pops.runtime.bricks import Explicit


def isothermal_transport_module(name, *, cs2=0.5):
    mod = model.Module(name)
    u = mod.state_space(
        "U",
        ("rho", "mx", "my"),
        roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"},
    )
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    rho, mx, my = Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")
    cs = sqrt(cs2)
    zero = Const(0.0)
    mod.operator(
        name="fields_from_state",
        signature=(u,) >> fields,
        kind="field_operator",
        capabilities={"default": True},
        expr=rho,
    )
    mod.operator(
        name="flux",
        signature=(u,) >> model.Rate(u),
        kind="grid_operator",
        expr={
            "x": [mx, mx * mx / rho + cs2 * rho, mx * my / rho],
            "y": [my, mx * my / rho, my * my / rho + cs2 * rho],
        },
    )
    mod.eigenvalues(
        x=[mx / rho - cs, mx / rho, mx / rho + cs],
        y=[my / rho - cs, my / rho, my / rho + cs],
    )
    mod.operator(
        name="source",
        signature=(u, fields) >> model.Rate(u),
        kind="local_source",
        capabilities={"default": True},
        expr=[zero, zero, zero],
    )
    return mod


def passive_scalar_module(name):
    mod = model.Module(name)
    u = mod.state_space("U", ("rho",), roles={"rho": "density"})
    fields = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    zero = Const(0.0)
    mod.operator(
        name="fields_from_state",
        signature=(u,) >> fields,
        kind="field_operator",
        capabilities={"default": True},
        expr=zero,
    )
    mod.operator(
        name="flux",
        signature=(u,) >> model.Rate(u),
        kind="grid_operator",
        expr={"x": [zero], "y": [zero]},
    )
    mod.eigenvalues(x=[zero], y=[zero])
    mod.operator(
        name="source",
        signature=(u, fields) >> model.Rate(u),
        kind="local_source",
        capabilities={"default": True},
        expr=[zero],
    )
    return mod


def first_order_rusanov():
    return spatial_catalog.FiniteVolume(
        reconstruction=FirstOrder(),
        riemann=Rusanov(),
    )


def explicit_euler():
    return Explicit.euler()


def explicit_ssprk2():
    return Explicit.ssprk2()
